#!/usr/bin/env python3
"""Main Pipeline Runner for 125M / 2.5B-Token Comparison (9 runs).

Executes the 9 runs sequentially across all 16 TPU chips (v4-32 slice):
3 seeds (42, 43, 44) x 3 architectures (standard, algebraic, hybrid).

For each model:
  1. Distributed training across w0..w3 (16 chips, batch 512 x 2048, 2385 steps).
  2. Immediate post-training benchmarks on chip 0:
     - Zero-shot public tasks (ARC-Easy, ARC-Challenge, HellaSwag, PIQA, OpenBookQA, WinoGrande, BoolQ, LAMBADA)
     - Recall matrix (Needle Passkey, Associative Recall, Multi-Needle, Latest-Value)
  3. Space cleanup: deletes large training checkpoints (step-*.pkl) across all hosts w0..w3,
     retaining params.npz, result.json, log.jsonl, and evaluation files.
  4. Updates live pipeline_status.json with progress, metrics, and IST ETA.

After all 9 runs finish:
  - Generates full publication tables (comparison.md, comparison.csv, recall_cells.csv)
  - Plots needle recall heatmaps (needle_recall.png, needle_recall.pdf).
"""

import argparse
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sota.benchmarking import (
    PUBLIC_TASKS,
    RECALL_PROTOCOL,
    ZERO_SHOT_PROTOCOL,
    load_jsonl,
    write_json,
)
from scripts.evaluate_main import validate_run

IST_OFFSET = timedelta(hours=5, minutes=30)
HOSTS = ["w0", "w1", "w2", "w3"]


def get_ist_str(dt: datetime = None) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    ist_time = dt.astimezone(timezone(IST_OFFSET))
    return ist_time.strftime("%Y-%m-%d %H:%M:%S IST")


def get_disk_space() -> dict:
    usage = {}
    for h in HOSTS:
        try:
            out = subprocess.check_output(
                ["ssh", h, "df -B1G / | tail -n 1"],
                text=True,
                timeout=10,
            ).split()
            usage[h] = {
                "size_gb": int(out[1]),
                "used_gb": int(out[2]),
                "avail_gb": int(out[3]),
                "use_pct": out[4],
            }
        except Exception as e:
            usage[h] = {"error": str(e)}
    return usage


def update_status(status_file: Path, status_data: dict):
    status_data["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    status_data["last_updated_ist"] = get_ist_str()
    write_json(status_file, status_data)


def run_training_job(job: dict, runs_root: Path, status_file: Path, pipeline_state: dict, log_file) -> None:
    name = job["name"]
    out_dir = runs_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_remote = f"~/sota-algebra/runs/main/{name}"

    args_str = " ".join(job["args"])
    cmd = (
        f"mkdir -p ~/sota-algebra/runs/main && cd ~/sota-algebra && "
        f"JAX_PLATFORMS=tpu ~/venv/bin/python -m src.sota.train --distributed "
        f"{args_str} --out {out_remote} > {out_remote}.$(hostname).log 2>&1"
    )

    log_file.write(f"[{get_ist_str()}] Launching distributed training for {name} on {HOSTS}\n")
    log_file.flush()

    procs = [subprocess.Popen(["ssh", h, cmd]) for h in HOSTS]
    log_jsonl = out_dir / "log.jsonl"
    t_train_start = time.time()

    # Monitor training progress while processes run
    while any(p.poll() is None for p in procs):
        time.sleep(5)
        if log_jsonl.exists():
            try:
                records = load_jsonl(log_jsonl)
                losses = [r for r in records if "loss" in r]
                if losses:
                    last = losses[-1]
                    cur_step = last.get("step", 0)
                    pipeline_state["current_job"]["step"] = cur_step
                    pipeline_state["current_job"]["loss"] = last.get("loss")
                    pipeline_state["current_job"]["gnorm"] = last.get("gnorm")
                    pipeline_state["current_job"]["lr"] = last.get("lr")
                    pipeline_state["current_job"]["tok_per_s"] = last.get("tok_per_s")

                    # Calculate job-level ETA
                    elapsed = time.time() - t_train_start
                    if cur_step > 0:
                        sec_per_step = elapsed / cur_step
                        job_rem_sec = (2385 - cur_step) * sec_per_step
                        job_eta_dt = datetime.now(timezone.utc) + timedelta(seconds=job_rem_sec)
                        pipeline_state["current_job"]["eta_ist"] = get_ist_str(job_eta_dt)

                        # Pipeline-level ETA estimation
                        # completed jobs + current job progress + future jobs
                        # Estimate avg job time from completed or current speed
                        est_job_time = sec_per_step * 2385 + 240 # ~4 min eval
                        future_jobs = pipeline_state["total_jobs"] - pipeline_state["current_job"]["index"]
                        pipe_rem_sec = job_rem_sec + 240 + (future_jobs * est_job_time)
                        pipe_eta_dt = datetime.now(timezone.utc) + timedelta(seconds=pipe_rem_sec)
                        pipeline_state["estimated_total_remaining_seconds"] = pipe_rem_sec
                        pipeline_state["eta_ist"] = get_ist_str(pipe_eta_dt)

                    update_status(status_file, pipeline_state)
            except Exception:
                pass

    codes = [p.wait() for p in procs]
    log_file.write(f"[{get_ist_str()}] Training processes finished with codes {codes}\n")
    log_file.flush()

    if any(c != 0 for c in codes):
        w0_log_path = runs_root / f"{name}.t1v-n-79750f9f-w-0.log"
        w0_log = w0_log_path.read_text() if w0_log_path.exists() else "No w0 log found"
        raise RuntimeError(
            f"Training {name} failed with exit codes {codes} across hosts. Main log tail:\n{w0_log[-1000:]}"
        )


def run_evaluations(name: str, arch: str, seed: int, runs_root: Path, results_root: Path, log_file) -> dict:
    run_dir = runs_root / name
    evals_data = Path.home() / "data/evals"
    recall_data = Path.home() / "data/recall"

    zero_shot_out = results_root / f"{name}_zero_shot.json"
    recall_out = results_root / f"{name}_recall.json"

    # Zero-shot public tasks evaluation on chip 0
    log_file.write(f"[{get_ist_str()}] Running zero-shot evaluation for {name} on chip 0\n")
    log_file.flush()

    chiprun = Path.home() / "bin/chiprun"
    zs_cmd = [
        str(chiprun), "0", "scripts/evaluate_zero_shot.py", str(run_dir),
        "--benchmarks", str(evals_data),
        "--tasks", ",".join(PUBLIC_TASKS),
        "--batch", "8",
        "--out", str(zero_shot_out),
    ]
    subprocess.check_call(zs_cmd, cwd=str(ROOT))

    # Recall evaluation on chip 0
    log_file.write(f"[{get_ist_str()}] Running recall evaluation for {name} on chip 0\n")
    log_file.flush()

    rec_tasks = ["needle_passkey", "associative_recall", "multi_needle", "latest_value"]
    rec_cmd = [
        str(chiprun), "0", "scripts/evaluate_recall.py", str(run_dir),
        "--benchmarks", str(recall_data),
        "--tasks", ",".join(rec_tasks),
        "--batch", "8",
        "--out", str(recall_out),
    ]
    subprocess.check_call(rec_cmd, cwd=str(ROOT))

    zs_res = json.loads(zero_shot_out.read_text())
    rec_res = json.loads(recall_out.read_text())

    return {
        "zero_shot": {t: zs_res["results"][t]["accuracy"] for t in PUBLIC_TASKS},
        "recall_seconds": rec_res.get("seconds"),
        "zero_shot_seconds": zs_res.get("seconds"),
    }


def cleanup_checkpoints(name: str, runs_root: Path, log_file) -> int:
    """Deletes large intermediate checkpoint files (ckpts/) across all hosts w0..w3."""
    log_file.write(f"[{get_ist_str()}] Cleaning intermediate checkpoints for {name} across all hosts\n")
    log_file.flush()
    freed_estimate_mb = 0

    local_ckpt_dir = runs_root / name / "ckpts"
    if local_ckpt_dir.exists():
        for f in local_ckpt_dir.glob("*.pkl"):
            freed_estimate_mb += f.stat().st_size / (1024 * 1024)
        shutil.rmtree(local_ckpt_dir, ignore_errors=True)

    for h in ["w1", "w2", "w3"]:
        try:
            subprocess.run(
                ["ssh", h, f"rm -rf ~/sota-algebra/runs/main/{name}/ckpts"],
                check=True,
                timeout=30,
            )
        except Exception as e:
            log_file.write(f"Warning: Failed to clean ckpts on {h}: {e}\n")

    log_file.write(f"[{get_ist_str()}] Cleaned checkpoints (~{freed_estimate_mb:.1f} MB freed locally on w0)\n")
    log_file.flush()
    return freed_estimate_mb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=Path, default=ROOT / "main.json")
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs/main")
    p.add_argument("--results-root", type=Path, default=ROOT / "results/main")
    p.add_argument("--status-file", type=Path, default=ROOT / "runs/main/pipeline_status.json")
    p.add_argument("--log-file", type=Path, default=ROOT / "runs/main/pipeline.log")
    args = p.parse_args()

    args.runs_root.mkdir(parents=True, exist_ok=True)
    args.results_root.mkdir(parents=True, exist_ok=True)

    log_fh = open(args.log_file, "a", buffering=1)
    log_fh.write(f"\n{'='*70}\n[{get_ist_str()}] Main 125M comparison pipeline started.\n{'='*70}\n")

    eval_config = json.loads((ROOT / "configs/post_training_eval.json").read_text())
    jobs = json.loads(args.jobs.read_text())
    total_jobs = len(jobs)

    start_time = time.time()
    pipeline_state = {
        "status": "RUNNING",
        "pipeline_start_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_start_ist": get_ist_str(),
        "total_jobs": total_jobs,
        "completed_count": 0,
        "completed_jobs": [],
        "current_job": None,
        "disk_space": get_disk_space(),
        "estimated_total_remaining_seconds": None,
        "eta_ist": None,
    }
    update_status(args.status_file, pipeline_state)

    # Sync workers first
    log_fh.write(f"[{get_ist_str()}] Syncing code to worker nodes w1..w3...\n")
    subprocess.check_call(["bash", str(ROOT / "scripts/setup_workers.sh")])
    log_fh.write(f"[{get_ist_str()}] Workers synced.\n")

    durations = []

    for idx, job in enumerate(jobs):
        name = job["name"]
        arch, _, seed_str = name.partition("_s")
        seed = int(seed_str)
        run_dir = args.runs_root / name

        log_fh.write(f"\n{'-'*50}\n[{get_ist_str()}] Processing job {idx+1}/{total_jobs}: {name}\n{'-'*50}\n")

        job_start_t = time.time()
        pipeline_state["current_job"] = {
            "name": name,
            "architecture": arch,
            "seed": seed,
            "index": idx + 1,
            "stage": "CHECKING",
            "step": 0,
            "total_steps": 2385,
            "loss": None,
            "tok_per_s": None,
            "lr": None,
            "job_start_ist": get_ist_str(),
        }
        update_status(args.status_file, pipeline_state)

        # Check if already complete and validated
        zs_file = args.results_root / f"{name}_zero_shot.json"
        rec_file = args.results_root / f"{name}_recall.json"
        res_file = run_dir / "result.json"

        is_already_done = False
        if res_file.exists() and zs_file.exists() and rec_file.exists():
            try:
                validate_run(run_dir, arch, seed, eval_config)
                is_already_done = True
                log_fh.write(f"[{get_ist_str()}] {name} is already complete and validated. Skipping.\n")
                cleanup_checkpoints(name, args.runs_root, log_fh)
            except Exception as e:
                log_fh.write(f"[{get_ist_str()}] Found existing files for {name} but validation failed: {e}. Re-running.\n")

        if not is_already_done:
            # Stage 1: Training
            pipeline_state["current_job"]["stage"] = "TRAINING"
            update_status(args.status_file, pipeline_state)

            t_train_start = time.time()
            run_training_job(job, args.runs_root, args.status_file, pipeline_state, log_fh)
            t_train_end = time.time()

            # Validate training output
            validate_run(run_dir, arch, seed, eval_config)
            log_fh.write(f"[{get_ist_str()}] Training validated successfully for {name} ({(t_train_end - t_train_start)/60:.1f} min)\n")

            # Stage 2: Evaluation
            pipeline_state["current_job"]["stage"] = "EVALUATION"
            update_status(args.status_file, pipeline_state)

            t_eval_start = time.time()
            eval_metrics = run_evaluations(name, arch, seed, args.runs_root, args.results_root, log_fh)
            t_eval_end = time.time()
            log_fh.write(f"[{get_ist_str()}] Evaluations finished for {name} ({(t_eval_end - t_eval_start)/60:.1f} min)\n")

            # Stage 3: Cleanup
            pipeline_state["current_job"]["stage"] = "CLEANUP"
            update_status(args.status_file, pipeline_state)
            cleanup_checkpoints(name, args.runs_root, log_fh)

        # Job complete
        job_total_duration = time.time() - job_start_t
        durations.append(job_total_duration)

        res_info = json.loads((run_dir / "result.json").read_text())
        zs_info = json.loads((args.results_root / f"{name}_zero_shot.json").read_text())

        job_summary = {
            "name": name,
            "architecture": arch,
            "seed": seed,
            "final_eval_nll": res_info["final_eval_nll"],
            "final_eval_ppl": res_info["final_eval_ppl"],
            "mean_tok_per_s": res_info["mean_tok_per_s"],
            "train_seconds": res_info["train_seconds"],
            "zero_shot_accuracies": {t: zs_info["results"][t]["accuracy"] for t in PUBLIC_TASKS},
            "job_duration_minutes": job_total_duration / 60,
        }
        pipeline_state["completed_jobs"].append(job_summary)
        pipeline_state["completed_count"] = len(pipeline_state["completed_jobs"])

        # Update remaining ETA
        remaining_jobs = total_jobs - pipeline_state["completed_count"]
        if remaining_jobs > 0:
            avg_duration = sum(durations) / len(durations) if durations else 2100.0
            rem_sec = remaining_jobs * avg_duration
            eta_dt = datetime.now(timezone.utc) + timedelta(seconds=rem_sec)
            pipeline_state["estimated_total_remaining_seconds"] = rem_sec
            pipeline_state["eta_ist"] = get_ist_str(eta_dt)
        else:
            pipeline_state["estimated_total_remaining_seconds"] = 0
            pipeline_state["eta_ist"] = "NOW"

        pipeline_state["disk_space"] = get_disk_space()
        update_status(args.status_file, pipeline_state)

    # All jobs finished! Run full summary and needle plots
    log_fh.write(f"\n{'='*70}\n[{get_ist_str()}] All {total_jobs} runs complete. Generating summary and plots...\n{'='*70}\n")

    subprocess.check_call([
        str(ROOT / "venv/bin/python" if (ROOT / "venv").exists() else Path.home() / "venv/bin/python"),
        str(ROOT / "scripts/summarize_main_eval.py"),
        "--out-dir", str(args.results_root),
        "--archs", "standard,algebraic,hybrid",
        "--seeds", "42", "43", "44",
    ], cwd=str(ROOT))

    write_json(args.results_root / "evaluation_plan.json", {
        "status": "complete",
        "training_launched": True,
        "config": eval_config,
        "completed_at_ist": get_ist_str(),
    })

    pipeline_state["status"] = "COMPLETE"
    pipeline_state["current_job"] = None
    pipeline_state["pipeline_end_ist"] = get_ist_str()
    pipeline_state["total_pipeline_minutes"] = (time.time() - start_time) / 60
    update_status(args.status_file, pipeline_state)

    log_fh.write(f"[{get_ist_str()}] Pipeline finished successfully in {pipeline_state['total_pipeline_minutes']:.1f} minutes.\n")
    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
