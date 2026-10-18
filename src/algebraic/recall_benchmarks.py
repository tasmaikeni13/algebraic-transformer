"""Deterministic retrieval prompts and metrics; no model or JAX dependency.

Raw token IDs define context lengths. Only distractor material may be
cut to fit: instructions, needles, and the question are always retained.
"""

from collections import defaultdict
import hashlib
import math
import random
import re

TASKS = ("needle_passkey", "associative_recall", "multi_needle", "latest_value")
DEMO_CODES = ("222333", "333444", "444555")
CODE_PATTERN = re.compile(r"(?<!\d)\d{6}(?!\d)")


def trial_rng(seed, task, trial):
    digest = hashlib.sha256(f"{seed}/{task}/{trial}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def fresh_code(rng, used):
    while True:
        code = str(rng.randrange(100000, 1000000))
        if code not in used:
            used.add(code)
            return code


def fresh_key(rng, used):
    while True:
        key = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
        if key not in used:
            used.add(key)
            return key


def make_case(task, seed, trial, enc=None):
    count_multi = 2 if (enc and getattr(enc, "n_vocab", 0) == 32768) else 3
    rng, used_codes, used_keys = trial_rng(seed, task, trial), set(DEMO_CODES), {"EXAMPLE"}
    count = count_multi if task == "multi_needle" else (3 if task == "latest_value" else 1)
    codes = [fresh_code(rng, used_codes) for _ in range(count)]
    keys = [fresh_key(rng, used_keys) for _ in range(count_multi if task == "multi_needle" else 1)]
    if task == "multi_needle":
        order = list(range(count_multi))
        rng.shuffle(order)
        expected = [codes[i] for i in order]
        question = ", ".join(keys[i] for i in order)
        if count_multi == 2:
            header = ("Read the archive and copy the requested six-digit codes, in order.\n"
                      "Example: AA=222333; BB=333444.\n"
                      "Codes for BB, AA: 333444, 222333\nArchive:\n")
        else:
            header = ("Read the archive and copy the requested six-digit codes, in order.\n"
                      "Example: AA=222333; BB=333444; CC=444555.\n"
                      "Codes for BB, AA, CC: 333444, 222333, 444555\nArchive:\n")
        facts = [f"\nLocker {key} has code {code}.\n" for key, code in zip(keys, codes)]
        query = f"\nEnd of archive.\nCodes for {question}:"
    elif task == "latest_value":
        expected = [codes[-1]]
        header = ("Read the archive in order. Copy the most recent six-digit code.\n"
                  "Example: EXAMPLE has code 222333; later it has code 333444.\n"
                  "Latest code for EXAMPLE: 333444\nArchive:\n")
        facts = [f"\nUpdate: locker {keys[0]} now has code {code}.\n" for code in codes]
        query = f"\nEnd of archive.\nLatest code for {keys[0]}:"
    else:
        expected = codes
        header = ("Read the archive and copy the requested six-digit code.\n"
                  "Example: locker EXAMPLE has code 222333.\n"
                  "Code for EXAMPLE: 222333\nArchive:\n")
        facts = [f"\nLocker {keys[0]} has code {codes[0]}.\n"]
        query = f"\nEnd of archive.\nCode for {keys[0]}:"
    target = ", ".join(expected)
    choices = [target]
    if task == "latest_value":
        choices.append(codes[-2])  # Explicit stale-value distractor.
    while len(choices) < 4:
        choices.append(", ".join(fresh_code(rng, used_codes) for _ in expected))
    foils = choices[1:]
    rng.shuffle(foils)
    answer = trial % 4  # Equal gold positions in the frozen 32-trial cells.
    choices = foils[:answer] + [target] + foils[answer:]
    return {"header": header, "facts": facts, "query": query, "codes": codes,
            "keys": keys, "expected_codes": expected, "target": " " + target,
            "choices": [" " + c for c in choices], "answer": answer,
            "used_codes": used_codes, "used_keys": used_keys}


def filler_tokens(enc, task, seed, trial, minimum, case):
    rng = trial_rng(seed, task + "/filler", trial)
    used_codes, used_keys = set(case["used_codes"]), set(case["used_keys"])
    words = ("garden", "harbor", "workshop", "library", "village", "station", "museum", "orchard")
    tokens = []
    while len(tokens) < minimum:
        place, other = rng.sample(words, 2)
        if task in ("associative_recall", "latest_value"):
            key, code = fresh_key(rng, used_keys), fresh_code(rng, used_codes)
            text = f"\nLocker {key} has code {code}. The {place} register was stored near the {other}.\n"
        else:
            text = (f"\nThe {place} kept a register of ordinary supplies. A clerk visited the {other} "
                    "and returned the folders before sunset. The records described routine work.\n")
        tokens.extend(enc.encode_ordinary(text))
    return tokens


def placements(task, depth, enc=None):
    if task == "multi_needle":
        count_multi = 2 if (enc and getattr(enc, "n_vocab", 0) == 32768) else 3
        return [(depth + i / count_multi) % 1.0 for i in range(count_multi)]
    if task == "latest_value":
        return [depth / 3, 2 * depth / 3, depth]
    return [depth]


def build_example(enc, task, seed, trial, budget, depth, max_new_tokens,
                  *, condition="present", short=False):
    case = make_case(task, seed, trial, enc=enc)
    head = enc.encode_ordinary(case["header"])
    query = enc.encode_ordinary(case["query"])
    facts = [enc.encode_ordinary(text) for text in case["facts"]]
    fixed = len(head) + len(query) + sum(map(len, facts))
    available = budget - max_new_tokens - fixed
    if available < 0:
        raise ValueError(f"budget {budget} cannot hold the complete {task} prompt")
    filler = filler_tokens(enc, task, seed, trial, max(available, max(map(len, facts))), case)
    count = 0 if short else available
    inserts = sorted((round(p * count), i) for i, p in enumerate(placements(task, depth, enc=enc)))
    prompt, spans, previous = list(head), [], 0
    for offset, index in inserts:
        prompt.extend(filler[previous:offset])
        start = len(prompt)
        tokens = facts[index] if condition == "present" else filler[:len(facts[index])]
        prompt.extend(tokens)
        spans.append({"fact_index": index, "start": start, "end": len(prompt),
                      "token_ids": facts[index], "present": condition == "present"})
        previous = offset
    prompt.extend(filler[previous:count])
    prompt.extend(query)
    group = "short" if short else f"b{budget}/d{depth:g}"
    row = {"id": f"{task}/{group}/t{trial}/{condition}", "task": task,
           "trial": trial, "condition": "short" if short else condition,
           "context_budget": len(prompt) + max_new_tokens if short else budget,
           "requested_depth": None if short else depth, "prompt_tokens": len(prompt),
           "prompt_token_ids": prompt, "prompt": enc.decode(prompt),
           "max_new_tokens": max_new_tokens, "expected_codes": case["expected_codes"],
           "target": case["target"], "target_token_ids": enc.encode_ordinary(case["target"]),
           "choices": case["choices"], "choice_token_ids": [enc.encode_ordinary(c) for c in case["choices"]],
           "answer": case["answer"], "needle_spans": spans,
           "actual_depths": [s["start"] / len(prompt) for s in spans]}
    validate_example(row, enc, trained_context=2048)
    return row


def validate_example(row, enc, trained_context):
    ids, steps = row["prompt_token_ids"], row["max_new_tokens"]
    if len(ids) != row["prompt_tokens"] or len(ids) + steps != row["context_budget"]:
        raise ValueError(f"inconsistent context budget: {row['id']}")
    if not ids or len(ids) + steps > trained_context:
        raise ValueError(f"prompt plus answer exceeds trained context: {row['id']}")
    if any(not isinstance(t, int) or not 0 <= t < enc.n_vocab for t in ids):
        raise ValueError(f"invalid prompt token: {row['id']}")
    if enc.decode(ids) != row["prompt"]:
        raise ValueError(f"prompt text and token IDs disagree: {row['id']}")
    answer_ids = [row["target_token_ids"], *row["choice_token_ids"]]
    if any(not tokens or len(tokens) > steps for tokens in answer_ids):
        raise ValueError(f"answer exceeds reserved generation budget: {row['id']}")
    if row["target_token_ids"] != enc.encode_ordinary(row["target"]):
        raise ValueError(f"target tokenization mismatch: {row['id']}")
    if row["choice_token_ids"] != [enc.encode_ordinary(c) for c in row["choices"]]:
        raise ValueError(f"choice tokenization mismatch: {row['id']}")
    if len(row["choices"]) != 4 or len(set(row["choices"])) != 4:
        raise ValueError(f"need four distinct recognition choices: {row['id']}")
    if row["choices"][row["answer"]] != row["target"]:
        raise ValueError(f"recognition target mismatch: {row['id']}")
    for span in row["needle_spans"]:
        if span["present"] and ids[span["start"]:span["end"]] != span["token_ids"]:
            raise ValueError(f"needle was truncated or changed: {row['id']}")
    codes = set(CODE_PATTERN.findall(row["prompt"]))
    expected = set(row["expected_codes"])
    if row["condition"] == "absent" and codes & expected:
        raise ValueError(f"missing-evidence control leaks its target: {row['id']}")
    if row["condition"] != "absent" and not expected <= codes:
        raise ValueError(f"missing target evidence: {row['id']}")


def generated_metrics(text, expected):
    # Count the requested leading codes; trailing prose is allowed, extra codes
    # are retained in predictions. The decoder never sees the gold answer.
    codes = CODE_PATTERN.findall(text)
    prefix = codes[:len(expected)]
    return {"generated_codes": codes, "exact_match": prefix == expected,
            "format_valid": len(codes) >= len(expected),
            "needle_hits": sum(code in codes for code in expected),
            "needles": len(expected)}


def wilson(correct, total):
    if total == 0:
        return [None, None]
    z, p = 1.959963984540054, correct / total
    denominator = 1 + z*z / total
    center = (p + z*z / (2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0.0, center-half), min(1.0, center+half)]


def summarize(predictions):
    groups = defaultdict(list)
    for row in predictions:
        budget = None if row["condition"] == "short" else row["context_budget"]
        groups[(row["task"], row["condition"], budget, row["requested_depth"])].append(row)
    cells = []
    for (task, condition, budget, depth), rows in groups.items():
        total, correct = len(rows), sum(r["exact_match"] for r in rows)
        recognition = sum(r["recognition_correct"] for r in rows)
        cells.append({"task": task, "condition": condition, "context_budget": budget,
                      "primary_metric": "unsupported_target_hit" if condition == "absent" else "exact_retrieval",
                      "requested_depth": depth, "total": total, "correct": correct,
                      "exact_match": correct / total, "wilson_95": wilson(correct, total),
                      "recognition_accuracy": recognition / total, "recognition_chance": 0.25,
                      "target_prefix_match": sum(r["target_prefix_match"] for r in rows) / total,
                      "format_valid": sum(r["format_valid"] for r in rows) / total,
                      "needle_recall": sum(r["needle_hits"] for r in rows) / sum(r["needles"] for r in rows)})
    # Cell intervals have 32 independent trials; reused trials across lengths
    # and depths are paired, so no pooled independent-item confidence interval.
    return cells
