#!/usr/bin/env python3
"""Plan or run matched 350M/7B and 1B/20B experiments on a TPU slice.

The default prints a plan without training. --run trains both models
sequentially using comparison config parameters (or optional --hparams),
and evaluates the same eight public tasks and full recall matrix. Repeating --run resumes.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCALES = {"125m": 2_500_000_000, "350m": 7_000_000_000, "1b": 20_000_000_000}


def training_jobs(config, data_dir, selected=None, *, save_params=True):
    jobs = []
    options = ("size", "seq_len", "batch", "micro", "tokens", "data_seed", "lr",
               "min_lr_ratio", "warmup", "schedule", "wd", "b1", "b2", "clip",
               "eval_tokens", "eval_batch", "eval_every", "log_every", "ckpt_every", "keep_ckpts")
    for seed in config["training_seeds"]:
        for arch in config["architectures"]:
            t = {**config["training"], **(selected or {}).get(arch, {})}
            args = ["--preset", arch, "--seed", str(seed)]
            for key in options:
                args += ["--" + key.replace("_", "-"), str(t[key])]
            args += ["--train", str(data_dir / t["train_file"]), "--valid", str(data_dir / t["valid_file"])]
            for key in ("attn_impl", "head_impl", "flash_block", "head_block_n", "head_block_v"):
                args += ["--set", f"{key}={t[key]}"]
            args += ["--resume"]
            if save_params:
                args += ["--save-params"]
            jobs.append({"name": f"{arch}_s{seed}", "args": args})
    return jobs


def parameter_count(t, arch):
    d, f, layers, heads = (t["dimensions"][k] for k in ("d_model", "d_ff", "n_layers", "n_heads"))
    count = t["vocab_size"] * d + layers * (4*d*d + 3*d*f + 2*d) + 2*d
    return count + (layers * (2*(d//heads) + heads) if arch == "algebraic" else 0)


def read_selection(path, config_path):
    selection = json.loads(path.read_text())
    if (selection.get("comparison_config_sha256") != hashlib.sha256(config_path.read_bytes()).hexdigest()
            or not selection.get("complete") or set(selection.get("selected", {})) != {"standard", "algebraic"}):
        raise ValueError("hyperparameter selection does not match this complete comparison configuration")
    for hp in selection["selected"].values():
        if (set(hp) != {"lr", "wd"} or not all(math.isfinite(v) for v in hp.values())
                or hp["lr"] <= 0 or hp["wd"] < 0):
            raise ValueError("invalid selected learning rate or weight decay")
    return selection["selected"]


def build_plan(config_path, data_dir, benchmarks, recall, group, eval_batch, hparams_path=None):
    config = json.loads(config_path.read_text())
    t = config["training"]
    if (config["architectures"] != ["standard", "algebraic"] or len(config["training_seeds"]) != 1
            or t["size"] not in SCALES or t["tokens"] != SCALES[t["size"]]
            or t["attn_impl"] != "flash" or t["head_impl"] != "fused"):
        raise ValueError("comparison requires matched arms, one seed, the configured token budget, and fused kernels")
    if not group or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in group):
        raise ValueError("group must be a simple directory name")
    if eval_batch < 1:
        raise ValueError("evaluation batch must be positive")
    selected = read_selection(hparams_path, config_path) if hparams_path and hparams_path.is_file() else None
    jobs = training_jobs(config, data_dir, selected)
    tokens_per_step = t["batch"] * t["seq_len"]
    steps = math.ceil(t["tokens"] / tokens_per_step)
    evaluation = [sys.executable, str(ROOT / "scripts/evaluate_main.py"), "--config", str(config_path),
                  "--runs-root", str(ROOT / "runs" / group), "--out-dir", str(ROOT / "results" / group),
                  "--benchmarks", str(benchmarks), "--recall-benchmarks", str(recall),
                  "--batch", str(eval_batch), "--resume"]
    if selected:
        evaluation += ["--hparams", str(hparams_path)]
    return {"training_launched": False, "config": config,
            "hyperparameters_status": "selected from hparams" if selected else "comparison config parameters",
            "selected_hyperparameters": selected,
            "parameter_counts": {a: parameter_count(t, a) for a in config["architectures"]},
            "optimizer_steps": steps, "tokens_per_step": tokens_per_step,
            "effective_training_tokens": steps * tokens_per_step,
            "jobs": jobs, "evaluation_command": evaluation}


def preflight(config, data_dir, benchmarks=None, recall=None, config_path=None):
    import numpy as np
    t = config["training"]
    for name, minimum in ((t["train_file"], SCALES[t["size"]]), (t["valid_file"], 10_000_000)):
        tokens = np.load(data_dir / name, mmap_mode="r")
        if tokens.ndim != 1 or tokens.dtype != np.uint16 or len(tokens) < minimum:
            raise ValueError(f"incomplete token corpus: {data_dir / name}")
    meta = json.loads((data_dir / "fineweb_metadata.json").read_text())
    if (not meta.get("complete") or meta["dataset"] != "HuggingFaceFW/fineweb-edu"
            or meta["tokenizer"] != "llama32k" or meta["train_tokens"] != SCALES[t["size"]]
            or meta["train_file"] != t["train_file"] or not meta.get("validation_documents_disjoint")):
        raise ValueError("corpus metadata does not match the complete FineWeb-Edu experiment")
    if benchmarks is None:
        return
    sys.path.insert(0, str(ROOT / "src"))
    from algebraic.benchmarking import read_dataset, sha256_file
    for directory, tasks in ((benchmarks, config["public_tasks"]), (recall, config["recall"]["tasks"])):
        for task in tasks:
            _, info = read_dataset(directory, task)
            if info.get("limited", False):
                raise ValueError("full comparison requires complete benchmark datasets")
    recall_meta = json.loads((recall / "metadata.json").read_text())
    if recall_meta["spec"] != config["recall"] or recall_meta["config_sha256"] != sha256_file(config_path):
        raise ValueError("prepare recall examples with the selected comparison's --config")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", choices=SCALES, default="350m")
    p.add_argument("--config", type=Path)
    p.add_argument("--hparams", type=Path)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--benchmarks", type=Path, default=Path.home() / "data/evals")
    p.add_argument("--recall-benchmarks", type=Path)
    p.add_argument("--group")
    p.add_argument("--hosts", default="w0,w1,w2,w3")
    p.add_argument("--eval-chip", type=int, choices=range(4), default=0)
    p.add_argument("--eval-batch", type=int, default=1)
    p.add_argument("--jobs-out", type=Path)
    p.add_argument("--run", action="store_true", help="train comparison models, then evaluate")
    a = p.parse_args()
    config_path = (a.config or ROOT / f"configs/comparison_{a.scale}.json").resolve()
    hparams = (a.hparams or ROOT / f"results/tuning_{a.scale}/selected.json").resolve()
    data_dir = (a.data_dir or Path.home() / ("data/fw7b" if a.scale == "350m" else "data/fw20b")).resolve()
    recall = (a.recall_benchmarks or Path.home() / f"data/recall_{a.scale}").resolve()
    group = a.group or f"comparison_{a.scale}"
    plan = build_plan(config_path, data_dir, a.benchmarks.resolve(), recall, group, a.eval_batch, hparams)
    if plan["config"]["training"]["size"] != a.scale:
        p.error("--scale must match --config")
    if a.jobs_out:
        a.jobs_out.parent.mkdir(parents=True, exist_ok=True)
        a.jobs_out.write_text(json.dumps(plan["jobs"], indent=2) + "\n")
    if not a.run:
        print(json.dumps(plan, indent=2))
        return 0
    sys.path.insert(0, str(ROOT))
    preflight(plan["config"], data_dir, a.benchmarks, recall, config_path)
    job_path = ROOT / "runs" / group / "jobs.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps(plan["jobs"], indent=2) + "\n")
    print(f"Training {len(plan['jobs'])} models; {plan['optimizer_steps']:,} steps each.", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/run_slice.py"), str(job_path),
                    "--group", group, "--hosts", a.hosts, "--skip-done"], cwd=ROOT, check=True)
    evaluation = ["bash", str(ROOT / "scripts/tpu_python.sh"), str(a.eval_chip), *plan["evaluation_command"][1:]]
    print(shlex.join(evaluation), flush=True)
    subprocess.run(evaluation, cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "analysis/plot_figures.py"),
                    "--results-root", str(ROOT / "results" / group),
                    "--out-dir", str(ROOT / "results" / group / "figures")], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
