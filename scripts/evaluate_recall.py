#!/usr/bin/env python3
"""Score a completed LM on frozen needle and recall prompts (no training).

Primary: free greedy generation, temperature zero, fixed answer budget.
Diagnostics: partial needle recall, target-prefix match, and four-choice
recognition. Absent-evidence scores measure unsupported target hits.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from algebraic.benchmarking import (RECALL_PROTOCOL, Scorer, load_model, provenance,
                                   read_dataset, sha256_file, write_json)
from algebraic.recall_benchmarks import TASKS, generated_metrics, summarize, validate_example


def evaluate(scorer, benchmarks, tasks, limit=None):
    import numpy as np
    meta = json.loads((benchmarks / "metadata.json").read_text())
    if meta["protocol"] != RECALL_PROTOCOL or meta["tokenizer"] != "llama32k":
        raise ValueError("recall dataset uses a different evaluation protocol")
    results, predictions, datasets = {}, [], {}
    start = time.monotonic()
    for task in tasks:
        rows, datasets[task] = read_dataset(benchmarks, task, limit)
        for row in rows:
            validate_example(row, scorer.enc, scorer.T)
        budgets = {r["max_new_tokens"] for r in rows}
        if budgets != {16}:
            raise ValueError("this protocol requires a 16-token generation budget")
        print(f"{task}: decoding {len(rows)} prompts, no evidence truncation", flush=True)
        generated = scorer.greedy([r["prompt_token_ids"] for r in rows], max_new_tokens=budgets.pop())
        pairs = [(r["prompt_token_ids"], tokens) for r in rows for tokens in r["choice_token_ids"]]
        lps, hits = scorer.score_token_pairs(pairs, allow_truncation=False)
        details = []
        for i, (row, output) in enumerate(zip(rows, generated)):
            scores = lps[4*i:4*i+4]
            recognition = int(np.argmax(scores))
            details.append({key: row[key] for key in ("id", "task", "trial", "condition",
                            "context_budget", "requested_depth", "prompt_tokens", "actual_depths")})
            details[-1].update({**generated_metrics(output["text"], row["expected_codes"]),
                               "generated_text": output["text"], "generated_token_ids": output["token_ids"],
                               "expected_codes": row["expected_codes"],
                               "target_prefix_match": hits[4*i+row["answer"]],
                               "target_mean_logprob": scores[row["answer"]],
                               "recognition_prediction": recognition,
                               "recognition_correct": recognition == row["answer"],
                               "choice_mean_logprobs": scores, "evidence_truncated": False})
        results[task] = summarize(details)
        predictions.extend(details)
        present = [d for d in details if d["condition"] == "present"]
        count = sum(d["exact_match"] for d in present)
        print(f"{task}: supported exact retrieval {count}/{len(present)} "
              f"({time.monotonic()-start:.0f}s)", flush=True)
    return {"results": results, "dataset_metadata": {**meta, "datasets": datasets},
            "limited": limit is not None or any(d.get("limited", False) for d in datasets.values()),
            "limit": limit, "seconds": time.monotonic()-start,
            "decoding": {"temperature": 0, "beam_size": 1, "kv_cache": False,
                         "context_extension": False, "evidence_truncation": False,
                         "answer_parser": "leading requested six-digit codes; trailing prose allowed"}}, predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--benchmarks", type=Path, default=Path.home() / "data/recall")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--limit", type=int, help="development only; first N examples, not a full matrix")
    a = p.parse_args()
    tasks = a.tasks.split(",")
    if len(set(tasks)) != len(tasks) or any(t not in TASKS for t in tasks):
        p.error(f"--tasks must select from {','.join(TASKS)}")
    if a.limit is not None and a.limit < 1:
        p.error("--limit must be positive")
    cfg, params = load_model(a.run)
    scorer = Scorer(cfg, params, a.batch)
    record, predictions = evaluate(scorer, a.benchmarks, tasks, a.limit)
    record.update(provenance(a.run, scorer, RECALL_PROTOCOL))
    record["config"] = record["checkpoint"]["config"]
    path = a.out.with_suffix(".predictions.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in predictions))
    temporary.replace(path)
    record["predictions"] = str(path.resolve())
    record["predictions_sha256"] = sha256_file(path)
    write_json(a.out, record)


if __name__ == "__main__":
    main()
