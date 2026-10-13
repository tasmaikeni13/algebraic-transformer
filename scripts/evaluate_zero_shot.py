#!/usr/bin/env python3
"""Public task evaluation for one saved model (no training).

MC choices use mean continuation-token log-probability; LAMBADA uses exact
greedy final-word prefix match. This is the project's protocol, not a claim
of numerical equivalence to every external evaluation harness.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from algebraic.benchmarking import (PUBLIC_TASKS, ZERO_SHOT_PROTOCOL, Scorer,
                                   load_jsonl, load_model, provenance, read_dataset, sha256_file, write_json)


def evaluate(scorer, benchmarks, tasks, limit=None):
    import numpy as np
    results, predictions, datasets = {}, [], {}
    t0 = time.monotonic()
    for name in tasks:
        rows, datasets[name] = read_dataset(benchmarks, name, limit)
        truncated_before = scorer.truncated_pairs
        if name == "lambada":
            lps, hits = scorer.score([(r["context"], r["target"]) for r in rows])
            details = [{"id": r.get("id", f"{name}/{i}"), "task": name,
                        "correct": hit, "target_mean_logprob": lp}
                       for i, (r, lp, hit) in enumerate(zip(rows, lps, hits))]
        else:
            pairs = [(r["context"], choice) for r in rows for choice in r["choices"]]
            lps, _ = scorer.score(pairs)
            details, offset = [], 0
            for i, row in enumerate(rows):
                scores = lps[offset:offset + len(row["choices"])]
                predicted = int(np.argmax(scores))
                details.append({"id": row.get("id", f"{name}/{i}"), "task": name,
                                "answer": row["answer"], "prediction": predicted,
                                "correct": predicted == row["answer"],
                                "choice_mean_logprobs": scores})
                offset += len(scores)
        correct = sum(int(d["correct"]) for d in details)
        results[name] = {"correct": correct, "total": len(rows),
                         "accuracy": correct / len(rows),
                         "truncated_pairs": scorer.truncated_pairs - truncated_before}
        predictions.extend(details)
        print(f"{name}: {correct}/{len(rows)} = {correct/len(rows):.4f} "
              f"({time.monotonic()-t0:.0f}s)", flush=True)
    return {"results": results, "dataset_metadata": {"datasets": datasets},
            "limited": limit is not None or any(d.get("limited", False) for d in datasets.values()),
            "limit": limit, "seconds": time.monotonic() - t0}, predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path, help="completed run with params.npz and meta.json")
    p.add_argument("--benchmarks", type=Path, default=Path.home() / "data/evals")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", default=",".join(PUBLIC_TASKS))
    p.add_argument("--limit", type=int, help="development only: first N rows per task")
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    tasks = a.tasks.split(",")
    if len(set(tasks)) != len(tasks) or any(t not in PUBLIC_TASKS for t in tasks):
        p.error(f"--tasks must select from {','.join(PUBLIC_TASKS)}")
    if a.limit is not None and a.limit < 1:
        p.error("--limit must be positive")
    cfg, params = load_model(a.run)
    scorer = Scorer(cfg, params, a.batch)
    record, predictions = evaluate(scorer, a.benchmarks, tasks, a.limit)
    record.update(provenance(a.run, scorer, ZERO_SHOT_PROTOCOL))
    record["config"] = record["checkpoint"]["config"]
    pred_path = a.out.with_suffix(".predictions.jsonl")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = pred_path.with_name(pred_path.name + ".tmp")
    temporary.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in predictions))
    temporary.replace(pred_path)
    record["predictions"] = str(pred_path.resolve())
    record["predictions_sha256"] = sha256_file(pred_path)
    write_json(a.out, record)


if __name__ == "__main__":
    main()
