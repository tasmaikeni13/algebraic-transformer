#!/usr/bin/env python3
"""Download and normalize the four Pretraining study zero-shot evaluation datasets."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
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


DATASETS = {
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test", normalize_arc),
    "hellaswag": ("Rowan/hellaswag", "default", "validation", normalize_hellaswag),
    "piqa": ("lighteval/piqa", "plain_text", "validation", normalize_piqa),
    "lambada": ("cimec/lambada", "plain_text", "test", normalize_lambada),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/evals")
    parser.add_argument("--limit", type=int, help="Development-only row limit")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "datasets": {}}

    for name, (dataset, config, split, normalizer) in DATASETS.items():
        rows = [normalizer(row) for row in fetch_rows(dataset, config, split, args.limit)]
        path = args.output_dir / f"{name}.jsonl"
        encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        path.write_text(encoded)
        metadata["datasets"][name] = {
            "source": dataset, "config": config, "split": split,
            "rows": len(rows), "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "limited": args.limit is not None,
        }
        print(f"{name}: {len(rows):,} rows -> {path}")

    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
