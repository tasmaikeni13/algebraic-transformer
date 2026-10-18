#!/usr/bin/env python3
"""Unified pipeline for 350M hyperparameter sweep and main comparison runs.

TPU Pod Slice: Google Cloud TPU v4-32 (16 chips, 4 workers: w0, w1, w2, w3).

Workflow:
1. Phase 1: Hyperparameter testing (sweep) on 2B tokens per arm:
   - Standard Transformer (350M) vs Pure Algebraic Transformer (350M)
   - 3 candidates per architecture: lr in {1.0e-3, 2.0e-3, 4.0e-3}
   - Evaluated on held-out validation window (8M tokens) every 250 steps
   - Zero-checkpoint retention during sweep to preserve disk space
2. Automatic selection of best learning rate per architecture by validation NLL.
3. Storage maintenance across all 4 workers (w0, w1, w2, w3).
4. Automatic commit and push of sweep results and summary to GitHub.
5. Phase 2: Main 350M comparison runs on 7B tokens:
   - Standard Transformer vs Pure Algebraic Transformer
   - 3 seeds each (42, 43, 44) -> 6 runs total
   - Full model weight export (params.npz) and result.json
   - Post-run checkpoint pruning to keep disk storage free (<50% disk usage at all times).
6. Final summary and push of main comparison results to GitHub.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = Path.home() / "data" / "fw7b"
HOSTS = ("w0", "w1", "w2", "w3")
IST = timezone(timedelta(hours=5, minutes=30))
ALGEBRAIC_PYTHON = Path.home() / "venv" / "bin" / "python"
if not ALGEBRAIC_PYTHON.exists():
    ALGEBRAIC_PYTHON = Path(sys.executable)


def ist_now_str() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def get_disk_info(hosts=HOSTS) -> dict[str, str]:
    info = {}
    for h in hosts:
        try:
            line = subprocess.check_output(
                ["ssh", h, "df -h / | tail -n 1"], text=True, timeout=15
            ).strip()
            parts = line.split()
            # filesystem, size, used, avail, use%, mount
            info[h] = f"Size: {parts[1]}, Used: {parts[2]}, Avail: {parts[3]} ({parts[4]})"
        except Exception as e:
            info[h] = f"Error: {e}"
    return info


def clean_storage(hosts=HOSTS, prune_run_dir=None):
    for h in hosts:
        cmds = [
            "rm -rf /tmp/pytest* /tmp/tpu_logs/* /tmp/*.log* /tmp/data-gym-cache 2>/dev/null || true",
            "sudo rm -f /tmp/main.*.log.* 2>/dev/null || true",
        ]
        if prune_run_dir:
            cmds.append(f"rm -rf {shlex.quote(str(prune_run_dir))}/ckpt || true")
        cmd_str = " ; ".join(cmds)
        subprocess.run(["ssh", h, cmd_str], timeout=30, check=False)


def sync_data_to_workers(data_dir=DATA_DIR, hosts=HOSTS):
    """Sync prepared token datasets to worker slices w1, w2, w3."""
    print(f"\n[{ist_now_str()}] Syncing token dataset from w0 to all workers...", flush=True)
    required = [
        data_dir / "fineweb_train_7B.npy",
        data_dir / "fineweb_sweep_2B.npy",
        data_dir / "fineweb_valid.npy",
        data_dir / "fineweb_metadata.json",
    ]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required dataset file: {p}")

    for h in hosts:
        if h == "w0":
            continue
        print(f"[{ist_now_str()}] Syncing {data_dir} to {h}...", flush=True)
        subprocess.run(["ssh", h, f"mkdir -p {shlex.quote(str(data_dir))}"], check=True)
        subprocess.run(
            ["rsync", "-a", "--info=progress2", f"{data_dir}/", f"{h}:{data_dir}/"],
            check=True,
        )
    print(f"[{ist_now_str()}] Token datasets synced across all workers successfully!\n", flush=True)


def run_command_stream(cmd, log_file=None):
    """Run a subprocess and stream output to both console and optional log file."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    log_fh = open(log_file, "a") if log_file else None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_fh:
                log_fh.write(line)
                log_fh.flush()
    finally:
        proc.wait()
        if log_fh:
            log_fh.close()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {cmd}")


def get_vocab_size() -> int:
    meta_file = DATA_DIR / "fineweb_metadata.json"
    if meta_file.is_file():
        return json.loads(meta_file.read_text()).get("vocab_size", 32768)
    return 32768


def make_sweep_jobs():
    vocab_size = get_vocab_size()
    candidates = [
        {"name": "lr1e-3", "lr": 0.0010},
        {"name": "lr2e-3", "lr": 0.0020},
        {"name": "lr4e-3", "lr": 0.0040},
    ]
    archs = ["standard", "algebraic"]
    seed = 42
    jobs = []
    for cand in candidates:
        for arch in archs:
            name = f"{arch}_{cand['name']}_s{seed}"
            args = [
                "--preset", arch,
                "--size", "350m",
                "--seq-len", "2048",
                "--batch", "512",
                "--micro", "2",
                "--tokens", "2000000000",
                "--lr", str(cand["lr"]),
                "--wd", "0.1",
                "--warmup", "0.02",
                "--min-lr-ratio", "0.1",
                "--b2", "0.95",
                "--schedule", "cosine",
                "--seed", str(seed),
                "--train", str(DATA_DIR / "fineweb_sweep_2B.npy"),
                "--valid", str(DATA_DIR / "fineweb_valid.npy"),
                "--eval-tokens", "8000000",
                "--eval-batch", "2",
                "--eval-every", "250",
                "--log-every", "25",
                "--set", f"vocab_size={vocab_size}",
                "--set", "attn_impl=flash",
                "--set", "head_impl=fused",
                "--set", "flash_block=512",
                "--set", "head_block_n=128",
                "--set", "head_block_v=512",
            ]
            jobs.append({
                "name": name,
                "arch": arch,
                "cand_name": cand["name"],
                "lr": cand["lr"],
                "seed": seed,
                "args": args,
            })
    return jobs


def make_main_jobs(selected_lrs, seeds=(42, 43, 44)):
    vocab_size = get_vocab_size()
    archs = ["standard", "algebraic"]
    jobs = []
    for seed in seeds:
        for arch in archs:
            lr = selected_lrs[arch]
            name = f"{arch}_s{seed}"
            args = [
                "--preset", arch,
                "--size", "350m",
                "--seq-len", "2048",
                "--batch", "512",
                "--micro", "2",
                "--tokens", "7000000000",
                "--lr", str(lr),
                "--wd", "0.1",
                "--warmup", "0.02",
                "--min-lr-ratio", "0.1",
                "--b2", "0.95",
                "--schedule", "cosine",
                "--seed", str(seed),
                "--train", str(DATA_DIR / "fineweb_train_7B.npy"),
                "--valid", str(DATA_DIR / "fineweb_valid.npy"),
                "--eval-tokens", "8000000",
                "--eval-batch", "2",
                "--eval-every", "250",
                "--ckpt-every", "200",
                "--keep-ckpts", "1",
                "--log-every", "25",
                "--resume",
                "--save-params",
                "--set", f"vocab_size={vocab_size}",
                "--set", "attn_impl=flash",
                "--set", "head_impl=fused",
                "--set", "flash_block=512",
                "--set", "head_block_n=128",
                "--set", "head_block_v=512",
            ]
            jobs.append({
                "name": name,
                "arch": arch,
                "seed": seed,
                "lr": lr,
                "args": args,
            })
    return jobs


def run_phase1_sweep():
    print(f"\n=======================================================", flush=True)
    print(f"[{ist_now_str()}] STARTING PHASE 1: 350M HYPERPARAMETER SWEEP", flush=True)
    print(f"=======================================================\n", flush=True)

    group = "hparam_sweep_350m"
    jobs = make_sweep_jobs()
    job_path = ROOT / "runs" / group / "jobs.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps([{"name": j["name"], "args": j["args"]} for j in jobs], indent=2))

    print(f"Total sweep jobs: {len(jobs)} (2B tokens each) across Standard and Algebraic 350M architectures", flush=True)
    clean_storage(HOSTS)
    print("Initial disk info across workers:")
    for h, s in get_disk_info(HOSTS).items():
        print(f"  {h}: {s}")

    for i, j in enumerate(jobs, 1):
        name = j["name"]
        print(f"\n[{ist_now_str()}] [{i}/{len(jobs)}] Launching sweep job: {name} ...", flush=True)
        single_job_path = ROOT / "runs" / group / f"job_{name}.json"
        single_job_path.write_text(json.dumps([{"name": j["name"], "args": j["args"]}], indent=2))

        cmd = [
            str(ALGEBRAIC_PYTHON), str(ROOT / "scripts/run_slice.py"),
            str(single_job_path),
            "--group", group,
            "--hosts", ",".join(HOSTS),
            "--skip-done",
        ]
        t0 = time.time()
        run_command_stream(cmd)
        dt = (time.time() - t0) / 60
        print(f"[{ist_now_str()}] Completed {name} in {dt:.1f} minutes.", flush=True)

        clean_storage(HOSTS)
        print("Current disk info across workers:")
        for h, s in get_disk_info(HOSTS).items():
            print(f"  {h}: {s}")

    # Collect and analyze sweep results
    results_dir = ROOT / "results" / "hparam_sweep_350m"
    results_dir.mkdir(parents=True, exist_ok=True)

    measurements = {}
    table = {}
    for j in jobs:
        res_file = ROOT / "runs" / group / j["name"] / "result.json"
        if not res_file.is_file():
            raise FileNotFoundError(f"Missing result file: {res_file}")
        res = json.loads(res_file.read_text())
        nll = res["final_eval_nll"]
        ppl = res["final_eval_ppl"]
        tok_s = res.get("mean_tok_per_s", 0)
        measurements[j["name"]] = {
            "arch": j["arch"],
            "cand_name": j["cand_name"],
            "lr": j["lr"],
            "seed": j["seed"],
            "final_eval_nll": nll,
            "final_eval_ppl": ppl,
            "mean_tok_per_s": tok_s,
            "train_seconds": res.get("train_seconds", 0),
        }
        table[f"{j['arch']}/{j['cand_name']}"] = {
            "nll": nll,
            "ppl": ppl,
            "tok_per_s": tok_s,
            "lr": j["lr"],
        }

    # Select best per arch
    selected = {}
    print("\n--- 350M Hyperparameter Sweep Results ---")
    for arch in ["standard", "algebraic"]:
        cands_for_arch = [m for m in measurements.values() if m["arch"] == arch]
        best = min(cands_for_arch, key=lambda x: x["final_eval_nll"])
        selected[arch] = {
            "cand_name": best["cand_name"],
            "lr": best["lr"],
            "final_eval_nll": best["final_eval_nll"],
            "final_eval_ppl": best["final_eval_ppl"],
        }
        print(f"{arch.upper()} Best: {best['cand_name']} (LR={best['lr']}) with NLL={best['final_eval_nll']:.4f}, PPL={best['final_eval_ppl']:.2f}")

    sweep_record = {
        "timestamp_ist": ist_now_str(),
        "purpose": "350M hyperparameter sweep across standard and algebraic architectures on 2B tokens per arm",
        "tokens_per_arm": 2_000_000_000,
        "selected": selected,
        "table": table,
        "measurements": measurements,
    }
    (results_dir / "summary.json").write_text(json.dumps(sweep_record, indent=2))
    (results_dir / "selected.json").write_text(json.dumps({"selected": selected}, indent=2))

    # Clean intermediate logs/tmp
    clean_storage(HOSTS)

    # Push to GitHub
    print("\nPushing Phase 1 350M hyperparameter sweep results to GitHub...", flush=True)
    subprocess.run(["git", "add", "results/hparam_sweep_350m/"], cwd=ROOT, check=True)
    status = subprocess.check_output(["git", "status", "--porcelain", "results/hparam_sweep_350m/"], cwd=ROOT, text=True)
    if status.strip():
        subprocess.run(
            ["git", "commit", "-m", f"Record 350M hyperparameter sweep results (Std: {selected['standard']['lr']}, Alg: {selected['algebraic']['lr']})"],
            cwd=ROOT, check=True
        )
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
        print("Phase 1 sweep successfully pushed to GitHub!", flush=True)
    else:
        print("Results already committed.", flush=True)

    return {arch: selected[arch]["lr"] for arch in selected}


def run_phase2_main(selected_lrs):
    print(f"\n=======================================================", flush=True)
    print(f"[{ist_now_str()}] STARTING PHASE 2: MAIN 350M COMPARISON RUNS", flush=True)
    print(f"=======================================================\n", flush=True)
    print(f"Selected LRs: Standard={selected_lrs['standard']}, Algebraic={selected_lrs['algebraic']}", flush=True)

    group = "main_350m"
    seeds = (42, 43, 44)
    jobs = make_main_jobs(selected_lrs, seeds=seeds)
    job_path = ROOT / "runs" / group / "jobs.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps([{"name": j["name"], "args": j["args"]} for j in jobs], indent=2))

    print(f"Total main runs: {len(jobs)} (7B tokens each across 3 seeds)", flush=True)

    for i, j in enumerate(jobs, 1):
        name = j["name"]
        print(f"\n[{ist_now_str()}] [{i}/{len(jobs)}] Launching main run: {name} ...", flush=True)
        single_job_path = ROOT / "runs" / group / f"job_{name}.json"
        single_job_path.write_text(json.dumps([{"name": j["name"], "args": j["args"]}], indent=2))

        cmd = [
            str(ALGEBRAIC_PYTHON), str(ROOT / "scripts/run_slice.py"),
            str(single_job_path),
            "--group", group,
            "--hosts", ",".join(HOSTS),
            "--skip-done",
        ]
        t0 = time.time()
        run_command_stream(cmd)
        dt = (time.time() - t0) / 60
        print(f"[{ist_now_str()}] Completed {name} in {dt:.1f} minutes.", flush=True)

        # Storage maintenance: Verify params.npz and result.json exist, then prune ckpt/
        out_dir = ROOT / "runs" / group / name
        if (out_dir / "result.json").is_file() and (out_dir / "params.npz").is_file():
            print(f"Pruning intermediate checkpoints for completed run {name} across all workers...", flush=True)
            clean_storage(HOSTS, prune_run_dir=out_dir)
        else:
            print(f"Warning: {name} completed without params.npz or result.json; retaining checkpoints.", flush=True)

        print("Current disk info across workers:")
        for h, s in get_disk_info(HOSTS).items():
            print(f"  {h}: {s}")

    # Summarize Phase 2 results
    results_dir = ROOT / "results" / "main_350m"
    results_dir.mkdir(parents=True, exist_ok=True)

    main_records = {}
    for arch in ["standard", "algebraic"]:
        nlls = []
        ppls = []
        for s in seeds:
            res_file = ROOT / "runs" / group / f"{arch}_s{s}" / "result.json"
            if res_file.is_file():
                d = json.loads(res_file.read_text())
                nlls.append(d["final_eval_nll"])
                ppls.append(d["final_eval_ppl"])
        mean_nll = sum(nlls) / len(nlls) if nlls else float("nan")
        std_nll = (sum((x - mean_nll) ** 2 for x in nlls) / max(1, len(nlls) - 1)) ** 0.5 if len(nlls) > 1 else 0.0
        mean_ppl = sum(ppls) / len(ppls) if ppls else float("nan")
        main_records[arch] = {
            "seeds": list(seeds),
            "nll_per_seed": nlls,
            "mean_nll": mean_nll,
            "std_nll": std_nll,
            "mean_ppl": mean_ppl,
            "selected_lr": selected_lrs[arch],
        }
        print(f"\n{arch.upper()} FINAL (3 seeds): NLL = {mean_nll:.4f} ± {std_nll:.4f}, Mean PPL = {mean_ppl:.2f}")

    final_report = {
        "completed_at_ist": ist_now_str(),
        "scale": "350m",
        "tokens_per_run": 7_000_000_000,
        "seeds": list(seeds),
        "results": main_records,
    }
    (results_dir / "comparison_summary.json").write_text(json.dumps(final_report, indent=2))

    # Push final results
    print("\nPushing final main comparison results to GitHub...", flush=True)
    subprocess.run(["git", "add", "results/main_350m/"], cwd=ROOT, check=True)
    status = subprocess.check_output(["git", "status", "--porcelain", "results/main_350m/"], cwd=ROOT, text=True)
    if status.strip():
        subprocess.run(
            ["git", "commit", "-m", "Record completed 350M main comparison runs across 3 seeds on 7B tokens"],
            cwd=ROOT, check=True
        )
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
        print("Main run comparison results pushed to GitHub!", flush=True)

    print(f"\n[{ist_now_str()}] ALL 350M RUNS COMPLETED SUCCESSFULLY!", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("all", "sweep", "main", "sync-data"), default="all")
    p.add_argument("--standard-lr", type=float, default=None)
    p.add_argument("--algebraic-lr", type=float, default=None)
    args = p.parse_args()

    if args.phase == "sync-data":
        sync_data_to_workers()
        return

    selected_lrs = {}
    if args.phase in ("all", "sweep"):
        selected_lrs = run_phase1_sweep()
    if args.phase in ("all", "main"):
        if not selected_lrs:
            sel_file = ROOT / "results" / "hparam_sweep_350m" / "selected.json"
            if sel_file.is_file():
                sel_data = json.loads(sel_file.read_text()).get("selected", {})
                selected_lrs = {
                    "standard": args.standard_lr or sel_data.get("standard", {}).get("lr", 0.002),
                    "algebraic": args.algebraic_lr or sel_data.get("algebraic", {}).get("lr", 0.002),
                }
            else:
                selected_lrs = {
                    "standard": args.standard_lr or 0.002,
                    "algebraic": args.algebraic_lr or 0.002,
                }
        run_phase2_main(selected_lrs)


if __name__ == "__main__":
    raise SystemExit(main())
