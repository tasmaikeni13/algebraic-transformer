#!/usr/bin/env python3
"""Freeze the public post-training datasets in the project's scoring format."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import urlopen
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from algebraic.benchmarking import PUBLIC_TASKS, ZERO_SHOT_PROTOCOL, write_json
API = "https://datasets-server.huggingface.co/rows"


def fetch_rows(dataset: str, config: str, split: str, limit: int | None = None):
    rows = []
    offset = 0
    while limit is None or len(rows) < limit:
        length = min(100, limit - len(rows)) if limit is not None else 100
        url = API + "?" + urlencode({
            "dataset": dataset, "config": config, "split": split,
            "offset": offset, "length": length,
        })
        payload = None
        for attempt in range(10):
            try:
                with urlopen(url, timeout=60) as response:
                    payload = json.load(response)
                break
            except Exception as exc:
                if attempt == 9:
                    raise
                time.sleep(2.0 * (attempt + 1))
        batch = [item["row"] for item in payload.get("rows", [])]
        rows.extend(batch)
        offset += len(batch)
        if not batch or offset >= payload.get("num_rows_total", offset):
            break
        time.sleep(0.05)
    return rows


def normalize_arc(row):
    labels = [str(item) for item in row["choices"]["label"]]
    return {
        "context": f"Question: {row['question']}\nAnswer:",
        "choices": [" " + str(item) for item in row["choices"]["text"]],
        "answer": labels.index(str(row["answerKey"])),
    }


def normalize_hellaswag(row):
    return {
        "context": str(row["ctx"]),
        "choices": [" " + str(item).lstrip() for item in row["endings"]],
        "answer": int(row["label"]),
    }


def normalize_piqa(row):
    return {
        "context": str(row["goal"]),
        "choices": [" " + str(row["sol1"]), " " + str(row["sol2"])],
        "answer": int(row["label"]),
    }


def normalize_lambada(row):
    context, target = str(row["text"]).rsplit(maxsplit=1)
    return {"context": context, "target": " " + target}


def normalize_openbookqa(row):
    # "main" excludes the supporting fact supplied by "additional".
    return normalize_arc({**row, "question": row["question_stem"]})


def normalize_winogrande(row):
    prefix, suffix = str(row["sentence"]).split("_", 1)
    return {"context": prefix.rstrip(),
            "choices": [" " + str(row[key]) + suffix for key in ("option1", "option2")],
            "answer": int(row["answer"]) - 1}


def normalize_boolq(row):
    question = str(row["question"]).rstrip("? ") + "?"
    return {"context": f"{row['passage']}\nQuestion: {question}\nAnswer:",
            "choices": [" yes", " no"], "answer": 0 if row["answer"] else 1}


DATASETS = {
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test", normalize_arc),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test", normalize_arc),
    "hellaswag": ("Rowan/hellaswag", "default", "validation", normalize_hellaswag),
    "piqa": ("lighteval/piqa", "plain_text", "validation", normalize_piqa),
    "lambada": ("cimec/lambada", "plain_text", "test", normalize_lambada),
    "openbookqa": ("allenai/openbookqa", "main", "test", normalize_openbookqa),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation", normalize_winogrande),
    "boolq": ("google/boolq", "default", "validation", normalize_boolq),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/evals")
    parser.add_argument("--limit", type=int, help="Development-only row limit")
    parser.add_argument("--tasks", default=",".join(PUBLIC_TASKS), help="Comma-separated task names")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = args.tasks.split(",")
    if not tasks or len(set(tasks)) != len(tasks) or any(name not in DATASETS for name in tasks):
        raise ValueError(f"--tasks must select from {','.join(PUBLIC_TASKS)} without duplicates")
    meta_path = args.output_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {"datasets": {}}
    metadata.update({"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "protocol": ZERO_SHOT_PROTOCOL})

    for name in tasks:
        dataset, config, split, normalizer = DATASETS[name]
        raw = fetch_rows(dataset, config, split, args.limit)
        rows = [{"id": f"{name}/{row.get('id', i)}", **normalizer(row)} for i, row in enumerate(raw)]
        for row in rows:
            if "choices" in row and not 0 <= row["answer"] < len(row["choices"]):
                raise ValueError(f"unlabeled or invalid example: {row['id']}")
        if not rows:
            raise ValueError(f"empty dataset: {name}")
        path = args.output_dir / f"{name}.jsonl"
        encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text(encoded)
        temporary.replace(path)
        metadata["datasets"][name] = {
            "source": dataset, "config": config, "split": split,
            "source_url": f"https://huggingface.co/datasets/{dataset}",
            "rows": len(rows), "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "limited": args.limit is not None,
        }
        print(f"{name}: {len(rows):,} rows -> {path}")
        write_json(meta_path, metadata)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
