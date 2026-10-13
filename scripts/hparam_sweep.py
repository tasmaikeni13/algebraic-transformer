#!/usr/bin/env python3
"""Budgeted learning-rate and weight-decay sweeps for both model scales.

Four candidates per architecture, one seed, matched short token budgets and
held-out windows. The default prints jobs only. --stage run launches the
pilots; --stage select requires all eight finite, complete pilot results and
writes per-architecture hyperparameters for scripts/run_comparison.py.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_comparison import SCALES, preflight, training_jobs


def sweep_jobs(config, sweep, data_dir):
    if (sweep["scale"] != config["training"]["size"]
            or config["training_seeds"] != [sweep["seed"]]
            or not 0 < sweep["tokens_per_candidate"] < config["training"]["tokens"]):
        raise ValueError("sweep must use the comparison's scale and seed with a smaller budget")
    names = [candidate["name"] for candidate in sweep["candidates"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("sweep candidates require unique names")
    pilot = copy.deepcopy(config)
    pilot["training"].update(tokens=sweep["tokens_per_candidate"], eval_tokens=sweep["eval_tokens"],
                             eval_every=0, ckpt_every=50, keep_ckpts=1, log_every=10)
    jobs = []
    for candidate in sweep["candidates"]:
        hp = {k: candidate[k] for k in ("lr", "wd")}
        if hp["lr"] <= 0 or hp["wd"] < 0:
            raise ValueError("candidate learning rate must be positive; weight decay nonnegative")
        selected = {arch: hp for arch in config["architectures"]}
        for job in training_jobs(pilot, data_dir, selected, save_params=False):
            job["name"] = f"{job['name']}_{candidate['name']}"
            jobs.append(job)
    return jobs


def select(config_path, sweep_path, data_dir, runs_root, out):
    from scripts.run_slice import validate_completed_job
    # Every candidate is required, including the same seed and validation window.
    config = json.loads(config_path.read_text())
    sweep = json.loads(sweep_path.read_text())
    jobs = sweep_jobs(config, sweep, data_dir)
    measurements = {}
    selected, best_names = {}, {}
    for job in jobs:
        result = validate_completed_job(runs_root / job["name"], job)
        measurements[job["name"]] = {"final_eval_nll": result["final_eval_nll"],
                                      "effective_tokens": result["total_steps"] * result["tokens_per_step"],
                                      "result_sha256": hashlib.sha256((runs_root / job["name"] / "result.json").read_bytes()).hexdigest()}
    for arch in config["architectures"]:
        best = min(sweep["candidates"], key=lambda candidate:
                   measurements[f"{arch}_s{sweep['seed']}_{candidate['name']}"]["final_eval_nll"])
        selected[arch] = {k: best[k] for k in ("lr", "wd")}
        best_names[arch] = best["name"]
    record = {"complete": True, "scale": sweep["scale"], "seed": sweep["seed"],
              "comparison_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "sweep_config_sha256": hashlib.sha256(sweep_path.read_bytes()).hexdigest(),
              "selected": selected, "best_candidates": best_names, "measurements": measurements,
              "selection_metric": "final held-out NLL on an identical validation window",
              "interpretation": "short-budget pilot ranking; full-budget optimality and seed variance are not established"}
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(out)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", choices=SCALES, default="350m")
    p.add_argument("--stage", choices=("jobs", "run", "select"), default="jobs")
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--hosts", default="w0,w1,w2,w3")
    p.add_argument("--group")
    p.add_argument("--runs-root", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--jobs-out", type=Path)
    a = p.parse_args()
    config_path = ROOT / f"configs/comparison_{a.scale}.json"
    sweep_path = ROOT / f"configs/sweep_{a.scale}.json"
    config, sweep = (json.loads(path.read_text()) for path in (config_path, sweep_path))
    data_dir = (a.data_dir or Path.home() / ("data/fw7b" if a.scale == "350m" else "data/fw20b")).resolve()
    group = a.group or f"tuning_{a.scale}"
    runs_root = (a.runs_root or ROOT / "runs" / group).resolve()
    out = (a.out or ROOT / f"results/tuning_{a.scale}/selected.json").resolve()
    jobs = sweep_jobs(config, sweep, data_dir)
    if a.jobs_out:
        a.jobs_out.parent.mkdir(parents=True, exist_ok=True)
        a.jobs_out.write_text(json.dumps(jobs, indent=2) + "\n")
    if a.stage == "select":
        print(json.dumps(select(config_path, sweep_path, data_dir, runs_root, out), indent=2))
        return 0
    if a.stage == "jobs":
        step_tokens = config["training"]["batch"] * config["training"]["seq_len"]
        effective = len(jobs) * math.ceil(sweep["tokens_per_candidate"] / step_tokens) * step_tokens
        print(json.dumps({"training_launched": False, "scale": a.scale, "seed": sweep["seed"],
                          "pilot_steps": math.ceil(sweep["tokens_per_candidate"] / step_tokens),
                          "total_pilot_tokens": effective,
                          "full_comparison_fraction": effective / (2 * SCALES[a.scale]),
                          "selection_output": str(out), "jobs": jobs}, indent=2))
        return 0
    if runs_root != ROOT / "runs" / group:
        p.error("--stage run uses runs/<group>; --runs-root is for offline selection")
    preflight(config, data_dir)
    job_path = runs_root / "jobs.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps(jobs, indent=2) + "\n")
    subprocess.run([sys.executable, str(ROOT / "scripts/run_slice.py"), str(job_path),
                    "--group", group, "--hosts", a.hosts, "--skip-done"], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
