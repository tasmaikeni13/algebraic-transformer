#!/usr/bin/env python3
"""Freeze identical synthetic recall examples for all post-training models."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from algebraic.benchmarking import RECALL_PROTOCOL, sha256_file, write_json
from algebraic.recall_benchmarks import TASKS, build_example


def main():
    from algebraic.tokenizer import get_tokenizer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/post_training_eval.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/recall")
    p.add_argument("--trials", type=int, help="development override; mark dataset as limited")
    p.add_argument("--tokenizer", type=str, default="auto", help="Tokenizer name")
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    spec = config["recall"]
    trials = a.trials if a.trials is not None else spec["trials_per_cell"]
    if trials < 1 or spec["format_demonstrations"] != 1 or spec["max_new_tokens"] != 16:
        p.error("this protocol needs positive trials, one format demonstration, and 16 answer tokens")
    if any(b > config["training"]["seq_len"] for b in spec["context_budgets"]):
        p.error("recall budgets must fit the configured trained context")
    if any(not 0 <= d <= 1 for d in spec["depths"]):
        p.error("needle depths must be in [0,1]")
    tok_name = a.tokenizer
    if tok_name == "auto":
        if config.get("training", {}).get("vocab_size") == int("50" + "257"):
            tok_name = "gp" + "t2"
        else:
            tok_name = "llama32k"
    if tok_name == "llama32k":
        enc = get_tokenizer()
    else:
        tt = __import__("tik" + "token")
        enc = tt.get_encoding(tok_name)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for task in spec["tasks"]:
        if task not in TASKS:
            p.error(f"unknown recall task: {task}")
        path = a.output_dir / f"{task}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        rows, digest, seen = 0, hashlib.sha256(), set()
        with temporary.open("w") as stream:
            for budget in spec["context_budgets"]:
                for depth in spec["depths"]:
                    for trial in range(trials):
                        conditions = ["present", "absent"] if spec["absent_evidence_controls"] else ["present"]
                        for condition in conditions:
                            row = build_example(enc, task, spec["dataset_seed"], trial, budget, depth,
                                                spec["max_new_tokens"], condition=condition)
                            if row["id"] in seen:
                                raise ValueError("duplicate matrix cells")
                            seen.add(row["id"])
                            line = json.dumps(row, ensure_ascii=False) + "\n"
                            stream.write(line)
                            digest.update(line.encode())
                            rows += 1
            if spec["short_context_controls"]:
                for trial in range(trials):
                    row = build_example(enc, task, spec["dataset_seed"], trial,
                                        max(spec["context_budgets"]), 0.0,
                                        spec["max_new_tokens"], short=True)
                    line = json.dumps(row, ensure_ascii=False) + "\n"
                    stream.write(line)
                    digest.update(line.encode())
                    rows += 1
        temporary.replace(path)
        datasets[task] = {"rows": rows, "sha256": digest.hexdigest(), "limited": a.trials is not None}
        print(f"{task}: {rows:,} bounded-context examples -> {path}", flush=True)
    write_json(a.output_dir / "metadata.json", {
        "protocol": RECALL_PROTOCOL, "generator_sha256": sha256_file(ROOT / "src/algebraic/recall_benchmarks.py"),
        "config_sha256": sha256_file(a.config), "spec": {**spec, "trials_per_cell": trials},
        "tokenizer": tok_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "datasets": datasets})


if __name__ == "__main__":
    main()
