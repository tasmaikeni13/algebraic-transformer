#!/usr/bin/env python3
"""Evaluate the architectures and seeds in a frozen comparison configuration.

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
sys.path.insert(0, str(ROOT / "src"))
from algebraic.benchmarking import (RECALL_PROTOCOL, ZERO_SHOT_PROTOCOL, checkpoint_fingerprint,
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
        raise ValueError(f"{run} does not match the configured training experiment")
    if args.get("steps", 0) or args.get("profile_steps", 0):
        raise ValueError(f"explicit step overrides are not a complete token-budget run: {run}")
    if Path(args["train"]).name != training.get("train_file", "fineweb_train_2_5B.npy"):
        raise ValueError(f"unexpected main training corpus: {run}")
    expected_dimensions = {**training.get("dimensions", {"d_model": 768, "n_layers": 12,
                                                        "n_heads": 12, "d_ff": 2048}),
                           "vocab_size": training.get("vocab_size", 32768), "seq_len": training["seq_len"]}
    expected_execution = {k: training[k] for k in ("attn_impl", "head_impl", "flash_block",
                                                 "head_block_n", "head_block_v") if k in training}
    recipe = {**training, **plan.get("selected_hyperparameters", {}).get(arch, {})}
    recipe_keys = ("batch", "micro", "data_seed", "lr", "wd", "warmup", "schedule", "b1", "b2",
                   "clip", "min_lr_ratio", "eval_tokens", "eval_batch")
    if any(args.get(k) != recipe[k] for k in recipe_keys if k in recipe):
        raise ValueError(f"training recipe differs from the frozen comparison: {run}")
    if meta["tokens_per_step"] != args["batch"] * args["seq_len"]:
        raise ValueError(f"inconsistent token accounting: {run}")
    expected_components = {
        "standard": {"ffn": "swiglu", "pos": "rope", "attn": "softmax", "head": "ce",
                     "qk_norm": False, "sink": "none"},
        "algebraic": {"ffn": "alu_glu", "pos": "cayley_geo", "attn": "asinh", "head": "alg_ce",
                      "qk_norm": True, "sink": "learned", "attn_n": 16, "attn_c": 8.0,
                      "head_n": 8, "head_c": 2.0, "alu_beta": 0.5},
    }
    expected = {**expected_dimensions, **expected_execution, "norm": "rms", **expected_components[arch]}
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError(f"unexpected architecture or model size: {run}")
    steps = math.ceil(training["tokens"] / meta["tokens_per_step"])
    losses = [r for r in load_jsonl(run / "log.jsonl") if "loss" in r]
    if (not losses or losses[-1]["step"] != steps or not math.isfinite(losses[-1]["loss"])
            or meta["total_steps"] != steps or result["total_steps"] != steps
            or result.get("completed_steps", steps) != steps
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
    p.add_argument("--config", type=Path, default=ROOT / "configs/comparison_350m.json",
                   help="comparison configuration file; default configs/comparison_350m.json")
    p.add_argument("--hparams", type=Path, help="completed budget-sweep selection for this configuration")
    p.add_argument("--runs-root", type=Path, help="root directory for model runs; default runs/<config_stem>")
    p.add_argument("--benchmarks", type=Path, default=Path.home() / "data/evals")
    p.add_argument("--recall-benchmarks", type=Path, default=Path.home() / "data/recall")
    p.add_argument("--out-dir", type=Path, help="output directory for evaluations; default results/<config_stem>")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--archs", help="optional comma-separated subset; default standard and algebraic")
    p.add_argument("--seeds", type=int, nargs="+", help="optional subset of training seeds; default configured seeds")
    p.add_argument("--dry-run", action="store_true", help="write plan only; checkpoints not required")
    p.add_argument("--resume", action="store_true", help="reuse only matching complete evaluation outputs")
    a = p.parse_args()
    if not a.config.is_file():
        p.error(f"configuration file not found: {a.config}")
    plan = json.loads(a.config.read_text())
    if a.hparams:
        from scripts.run_comparison import read_selection
        plan["selected_hyperparameters"] = read_selection(a.hparams, a.config)
    archs = a.archs.split(",") if a.archs else plan["architectures"]
    seeds = a.seeds if a.seeds is not None else plan["training_seeds"]
    if (a.batch < 1 or len(set(archs)) != len(archs) or len(set(seeds)) != len(seeds)
            or not set(archs) <= set(plan["architectures"])
            or not set(seeds) <= set(plan["training_seeds"])):
        p.error("select configured architectures/seeds without duplicates and a positive batch")
    runs_root = a.runs_root or (ROOT / "runs" / a.config.stem)
    out_dir = a.out_dir or (ROOT / "results" / a.config.stem)
    jobs = []
    for seed in seeds:
        for arch in archs:
            name = f"{arch}_s{seed}"
            run = runs_root / name
            for suite, script, data, tasks, protocol in (
                ("zero_shot", "evaluate_zero_shot.py", a.benchmarks, plan["public_tasks"], ZERO_SHOT_PROTOCOL),
                ("recall", "evaluate_recall.py", a.recall_benchmarks, plan["recall"]["tasks"], RECALL_PROTOCOL),
            ):
                out = out_dir / f"{name}_{suite}.json"
                command = [sys.executable, str(ROOT / "scripts" / script), str(run),
                           "--benchmarks", str(data), "--tasks", ",".join(tasks),
                           "--batch", str(a.batch), "--out", str(out)]
                jobs.append({"architecture": arch, "seed": seed, "suite": suite, "run": str(run),
                             "output": str(out), "data": str(data), "tasks": tasks,
                             "protocol": protocol, "command": command})
    if a.dry_run:
        if a.out_dir is not None:
            write_json(out_dir / "evaluation_plan.json", {
                "status": "planned", "training_launched": False, "config": plan,
                "config_sha256": sha256_file(a.config), "jobs": jobs})
            print(f"Prepared {len(jobs)} evaluation jobs for {len(archs)*len(seeds)} models. "
                  f"Plan: {out_dir / 'evaluation_plan.json'}. No weights loaded.")
        else:
            print(f"Prepared {len(jobs)} evaluation jobs for {len(archs)*len(seeds)} models. "
                  f"No weights loaded.")
        return

    # Check every selected model and dataset before launching the first scorer.
    for seed in seeds:
        for arch in archs:
            validate_run(runs_root / f"{arch}_s{seed}", arch, seed, plan)
    for directory, tasks in ((a.benchmarks, plan["public_tasks"]),
                             (a.recall_benchmarks, plan["recall"]["tasks"])):
        for task in tasks:
            _, info = read_dataset(directory, task)
            if info.get("limited", False):
                raise ValueError(f"main evaluation requires the full dataset: {directory}/{task}")
    recall_meta = json.loads((a.recall_benchmarks / "metadata.json").read_text())
    if recall_meta["spec"] != plan["recall"] or recall_meta["config_sha256"] != sha256_file(a.config):
        raise ValueError("recall matrix does not match the frozen post-training plan; prepare it again")
    write_json(out_dir / "evaluation_plan.json", {
        "status": "planned", "training_launched": False, "config": plan,
        "config_sha256": sha256_file(a.config), "jobs": jobs})
    for job in jobs:
        output, run, data = Path(job["output"]), Path(job["run"]), Path(job["data"])
        if a.resume and reusable(output, run, data, job["tasks"], job["protocol"], a.batch):
            print(f"Reuse matching {output}", flush=True)
            continue
        print(f"Evaluate {job['architecture']} seed {job['seed']} / {job['suite']}", flush=True)
        # A fresh process releases JAX executable/device memory between suites.
        subprocess.run(job["command"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/summarize_main_eval.py"),
                    "--out-dir", str(out_dir), "--archs", ",".join(archs),
                    "--seeds", *map(str, seeds)], cwd=ROOT, check=True)
    write_json(out_dir / "evaluation_plan.json", {
        "status": "complete", "training_launched": False, "config": plan,
        "config_sha256": sha256_file(a.config), "jobs": jobs})


if __name__ == "__main__":
    main()
