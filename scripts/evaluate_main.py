#!/usr/bin/env python3
"""Evaluate all completed main-run seeds and write the three-arm comparison.

This driver only starts evaluation subprocesses. --dry-run writes the job
plan without loading weights or initializing JAX. Training is separate.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sota.benchmarking import (RECALL_PROTOCOL, ZERO_SHOT_PROTOCOL, checkpoint_fingerprint,
                                   load_jsonl, read_dataset, sha256_file, source_fingerprint, write_json)


def validate_run(run, arch, seed, plan):
    for name in ("meta.json", "result.json", "params.npz", "log.jsonl"):
        if not (run / name).is_file():
            raise ValueError(f"main run is not complete: missing {run / name}")
    meta = json.loads((run / "meta.json").read_text())
    result = json.loads((run / "result.json").read_text())
    args, cfg, training = meta["args"], meta["config"], plan["training"]
    expected_args = {"preset": arch, "seed": seed, "size": training["size"],
                     "tokens": training["tokens"], "seq_len": training["seq_len"]}
    if any(args.get(k) != v for k, v in expected_args.items()):
        raise ValueError(f"{run} is not the configured 125M / 2.5B-token main run")
    if Path(args["train"]).name != "fineweb_train_2_5B.npy":
        raise ValueError(f"unexpected main training corpus: {run}")
    expected_dimensions = {"d_model": 768, "n_layers": 12, "n_heads": 12, "d_ff": 2048,
                           "vocab_size": 50257, "seq_len": training["seq_len"]}
    expected_components = {
        "standard": {"ffn": "swiglu", "pos": "rope", "attn": "softmax", "head": "ce",
                     "qk_norm": False, "sink": "none"},
        "algebraic": {"ffn": "alu_glu", "pos": "cayley_geo", "attn": "asinh", "head": "alg_ce",
                      "qk_norm": True, "sink": "learned", "attn_n": 16, "attn_c": 8.0,
                      "head_n": 8, "head_c": 2.0, "alu_beta": 0.5},
        "hybrid": {"ffn": "swiglu", "pos": "rope", "attn": "softmax", "head": "alg_ce",
                   "qk_norm": True, "sink": "none", "head_n": 8, "head_c": 2.0},
    }
    expected = {**expected_dimensions, "norm": "rms", **expected_components[arch]}
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError(f"unexpected architecture or model size: {run}")
    steps = math.ceil(training["tokens"] / meta["tokens_per_step"])
    losses = [r for r in load_jsonl(run / "log.jsonl") if "loss" in r]
    if (not losses or losses[-1]["step"] != steps or not math.isfinite(losses[-1]["loss"])
            or meta["total_steps"] != steps or result["total_steps"] != steps
            or not math.isfinite(result["final_eval_nll"]) or result["config"] != cfg):
        raise ValueError(f"incomplete or non-finite training history: {run}")


def reusable(output, run, data, tasks, protocol, batch):
    if not output.is_file():
        return False
    record = json.loads(output.read_text())
    if (record.get("protocol") != protocol or record.get("limited", True)
            or record.get("batch") != batch or record.get("limit") is not None
            or record.get("source", {}).get("sha256") != source_fingerprint()["sha256"]
            or record.get("checkpoint") != checkpoint_fingerprint(run)
            or set(record.get("results", {})) != set(tasks)):
        return False
    metadata = json.loads((data / "metadata.json").read_text())["datasets"]
    recorded = record["dataset_metadata"]["datasets"]
    if any(recorded[t]["sha256"] != metadata[t]["sha256"] for t in tasks):
        return False
    prediction_path = output.with_suffix(".predictions.jsonl")
    return prediction_path.is_file() and sha256_file(prediction_path) == record.get("predictions_sha256")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/post_training_eval.json")
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs/main")
    p.add_argument("--benchmarks", type=Path, default=Path.home() / "data/evals")
    p.add_argument("--recall-benchmarks", type=Path, default=Path.home() / "data/recall")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/main")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--archs", help="optional comma-separated subset; default all three")
    p.add_argument("--seeds", type=int, nargs="+", help="optional subset; default 42 43 44")
    p.add_argument("--dry-run", action="store_true", help="write plan only; checkpoints not required")
    p.add_argument("--resume", action="store_true", help="reuse only matching complete evaluation outputs")
    a = p.parse_args()
    plan = json.loads(a.config.read_text())
    archs = a.archs.split(",") if a.archs else plan["architectures"]
    seeds = a.seeds if a.seeds is not None else plan["training_seeds"]
    if (a.batch < 1 or len(set(archs)) != len(archs) or len(set(seeds)) != len(seeds)
            or not set(archs) <= set(plan["architectures"])
            or not set(seeds) <= set(plan["training_seeds"])):
        p.error("select configured architectures/seeds without duplicates and a positive batch")
    jobs = []
    for seed in seeds:
        for arch in archs:
            name = f"{arch}_s{seed}"
            run = a.runs_root / name
            for suite, script, data, tasks, protocol in (
                ("zero_shot", "evaluate_zero_shot.py", a.benchmarks, plan["public_tasks"], ZERO_SHOT_PROTOCOL),
                ("recall", "evaluate_recall.py", a.recall_benchmarks, plan["recall"]["tasks"], RECALL_PROTOCOL),
            ):
                out = a.out_dir / f"{name}_{suite}.json"
                command = [sys.executable, str(ROOT / "scripts" / script), str(run),
                           "--benchmarks", str(data), "--tasks", ",".join(tasks),
                           "--batch", str(a.batch), "--out", str(out)]
                jobs.append({"architecture": arch, "seed": seed, "suite": suite, "run": str(run),
                             "output": str(out), "data": str(data), "tasks": tasks,
                             "protocol": protocol, "command": command})
    write_json(a.out_dir / "evaluation_plan.json", {
        "status": "planned", "training_launched": False, "config": plan,
        "config_sha256": sha256_file(a.config), "jobs": jobs})
    if a.dry_run:
        print(f"Prepared {len(jobs)} evaluation jobs for {len(archs)*len(seeds)} models. "
              f"Plan: {a.out_dir / 'evaluation_plan.json'}. No weights loaded.")
        return

    # Check every selected model and dataset before launching the first scorer.
    for seed in seeds:
        for arch in archs:
            validate_run(a.runs_root / f"{arch}_s{seed}", arch, seed, plan)
    for directory, tasks in ((a.benchmarks, plan["public_tasks"]),
                             (a.recall_benchmarks, plan["recall"]["tasks"])):
        for task in tasks:
            _, info = read_dataset(directory, task)
            if info.get("limited", False):
                raise ValueError(f"main evaluation requires the full dataset: {directory}/{task}")
    recall_meta = json.loads((a.recall_benchmarks / "metadata.json").read_text())
    if recall_meta["spec"] != plan["recall"] or recall_meta["config_sha256"] != sha256_file(a.config):
        raise ValueError("recall matrix does not match the frozen post-training plan; prepare it again")
    for job in jobs:
        output, run, data = Path(job["output"]), Path(job["run"]), Path(job["data"])
        if a.resume and reusable(output, run, data, job["tasks"], job["protocol"], a.batch):
            print(f"Reuse matching {output}", flush=True)
            continue
        print(f"Evaluate {job['architecture']} seed {job['seed']} / {job['suite']}", flush=True)
        # A fresh process releases JAX executable/device memory between suites.
        subprocess.run(job["command"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/summarize_main_eval.py"),
                    "--out-dir", str(a.out_dir), "--archs", ",".join(archs),
                    "--seeds", *map(str, seeds)], cwd=ROOT, check=True)
    write_json(a.out_dir / "evaluation_plan.json", {
        "status": "complete", "training_launched": False, "config": plan,
        "config_sha256": sha256_file(a.config), "jobs": jobs})


if __name__ == "__main__":
    main()
