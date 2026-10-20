#!/usr/bin/env python3
"""Unified pipeline for 1B hyperparameter sweep and main comparison runs.

TPU Pod Slice: Google Cloud TPU v4-32 (16 chips, 4 workers: w0, w1, w2, w3).

Workflow:
1. Phase 0: Dataset preparation — tokenize 20B training tokens + 5B sweep tokens
   from HuggingFaceFW/fineweb-edu (sample/100BT) using the llama-style 32k BPE
   tokenizer, then sync all arrays to workers w1, w2, w3.
2. Phase 1: Hyperparameter sweep on 5B tokens per arm:
   - Standard Transformer (1B) vs Pure Algebraic Transformer (1B)
   - 3 candidates per architecture: lr in {6e-4, 1.2e-3, 2.4e-3}  (lower, baseline, upper)
   - Evaluated on held-out validation window (8M tokens) every 250 steps
   - Zero-checkpoint retention during sweep to preserve disk space
3. Automatic selection of best learning rate per architecture by validation NLL.
4. Storage maintenance across all 4 workers (w0, w1, w2, w3).
5. Automatic commit and push of sweep results and summary to GitHub.
6. Phase 2: Main 1B comparison runs on 20B tokens:
   - Standard Transformer vs Pure Algebraic Transformer
   - 1 seed each (42) -> 2 runs total
   - Full model weight export (params.npz) and result.json
   - Post-run checkpoint pruning to keep disk storage free.
7. Final summary and push of main comparison results to GitHub.
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

DATA_DIR = Path.home() / "data" / "fw20b"
HOSTS = ("w0", "w1", "w2", "w3")
IST = timezone(timedelta(hours=5, minutes=30))

# Use the venv python if available (same pattern as 350m pipeline)
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


def print_disk_info(hosts=HOSTS):
    for h, s in get_disk_info(hosts).items():
        print(f"  {h}: {s}", flush=True)


def get_disk_bytes(host: str) -> tuple[int, int]:
    """Return (used_bytes, free_bytes) for / on host. Returns (-1, -1) on error."""
    try:
        out = subprocess.check_output(
            ["ssh", host, "df --block-size=1 / | tail -n 1"], text=True, timeout=15
        ).strip()
        parts = out.split()
        return int(parts[2]), int(parts[3])
    except Exception:
        return -1, -1


def clean_storage(hosts=HOSTS, prune_run_dir=None):
    """Remove tmp junk and optionally prune a finished run's ckpt/ directory."""
    for h in hosts:
        cmds = [
            "rm -rf /tmp/pytest* /tmp/tpu_logs/* /tmp/*.log* /tmp/data-gym-cache 2>/dev/null || true",
            "sudo rm -f /tmp/main.*.log.* 2>/dev/null || true",
        ]
        if prune_run_dir:
            cmds.append(f"rm -rf {shlex.quote(str(prune_run_dir))}/ckpt || true")
        cmd_str = " ; ".join(cmds)
        subprocess.run(["ssh", h, cmd_str], timeout=30, check=False)


def rebalance_storage(hosts=HOSTS, warn_pct: int = 70, critical_pct: int = 85):
    """
    Inspect disk usage on every worker. If any host exceeds warn_pct:
      1. Remove stale /tmp junk first (lightest touch).
      2. If still over warn_pct, remove ckpt/ subdirs in ~/algebraic-transformer/runs/
         that belong to completed runs (result.json present).
      3. If still over critical_pct, remove old log *.log files from run dirs on that host.
      4. Print a summary table of free space so the caller can monitor.
    Returns the host with the most free space (useful for choosing where to prep data).
    """
    print(f"\n[{ist_now_str()}] Storage rebalance check across all workers:", flush=True)
    best_host = hosts[0]
    best_free = -1
    for h in hosts:
        used, free = get_disk_bytes(h)
        if used < 0:
            print(f"  {h}: ERROR reading disk info", flush=True)
            continue
        total = used + free
        pct = int(used * 100 / total) if total else 0
        free_gb = free / 1e9
        print(f"  {h}: {pct}% used, {free_gb:.1f} GB free", flush=True)
        if free > best_free:
            best_free = free
            best_host = h
        if pct >= warn_pct:
            print(f"  [{h}] ⚠ Above {warn_pct}% — running targeted cleanup...", flush=True)
            # Step 1: tmp junk
            subprocess.run(
                ["ssh", h,
                 "rm -rf /tmp/pytest* /tmp/tpu_logs/* /tmp/*.log* /tmp/data-gym-cache "
                 "/tmp/auditor_* /tmp/smoke_* /tmp/challenger_* 2>/dev/null || true"],
                timeout=30, check=False,
            )
            used2, free2 = get_disk_bytes(h)
            pct2 = int(used2 * 100 / (used2 + free2)) if (used2 + free2) else pct
            if pct2 >= warn_pct:
                # Step 2: prune ckpt/ for completed runs on this host
                runs_dir = str(ROOT / "runs")
                prune_cmd = (
                    f"find {shlex.quote(runs_dir)} -maxdepth 3 -name result.json "
                    f"-exec dirname {{}} \\; | xargs -I{{}} rm -rf {{}}/ckpt 2>/dev/null || true"
                )
                subprocess.run(["ssh", h, prune_cmd], timeout=60, check=False)
                print(f"  [{h}] Pruned completed-run ckpt/ dirs.", flush=True)
            used3, free3 = get_disk_bytes(h)
            pct3 = int(used3 * 100 / (used3 + free3)) if (used3 + free3) else pct2
            if pct3 >= critical_pct:
                # Step 3: remove *.log and *.log.gz inside run dirs (keep result.json / params.npz)
                log_prune = (
                    f"find {shlex.quote(runs_dir)} -maxdepth 4 -name '*.log' -o -name '*.log.gz' "
                    f"| xargs rm -f 2>/dev/null || true"
                )
                subprocess.run(["ssh", h, log_prune], timeout=60, check=False)
                print(f"  [{h}] ⚠ Critical: pruned run log files too.", flush=True)
            used4, free4 = get_disk_bytes(h)
            free4_gb = free4 / 1e9
            total4 = used4 + free4
            pct4 = int(used4 * 100 / total4) if total4 else 0
            print(f"  [{h}] After cleanup: {pct4}% used, {free4_gb:.1f} GB free", flush=True)
    print(f"  Most free space: {best_host} ({best_free/1e9:.1f} GB)", flush=True)
    return best_host


def get_freest_host(hosts=HOSTS) -> str:
    """Return the worker hostname with the most free disk space."""
    best, best_free = hosts[0], -1
    for h in hosts:
        _, free = get_disk_bytes(h)
        if free > best_free:
            best_free, best = free, h
    return best


def sync_data_to_workers(data_dir=DATA_DIR, hosts=HOSTS):
    """Sync prepared token datasets from w0 to workers w1, w2, w3."""
    print(f"\n[{ist_now_str()}] Syncing token dataset from w0 to all workers...", flush=True)
    required = [
        data_dir / "fineweb_train_20B.npy",
        data_dir / "fineweb_sweep_5B.npy",
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


# ---------------------------------------------------------------------------
# Phase 0 – Dataset preparation
# ---------------------------------------------------------------------------

def ensure_data_ready_and_synced():
    """Prepare 20B training + 5B sweep token arrays on the freest worker, then sync to all."""
    meta_path = DATA_DIR / "fineweb_metadata.json"
    train_path = DATA_DIR / "fineweb_train_20B.npy"
    sweep_path = DATA_DIR / "fineweb_sweep_5B.npy"
    valid_path = DATA_DIR / "fineweb_valid.npy"

    if not (meta_path.is_file() and train_path.is_file() and sweep_path.is_file() and valid_path.is_file()):
        # Pick the worker with the most free space to run dataset preparation.
        # Data prep downloads raw parquet to /tmp on that host, tokenises, then syncs.
        # Required: ~50 GB for arrays + ~5 GB working space for parquet shards.
        REQUIRED_GB = 58.0
        prep_host = get_freest_host(HOSTS)
        _, prep_free = get_disk_bytes(prep_host)
        prep_free_gb = prep_free / 1e9
        print(
            f"[{ist_now_str()}] Freest worker for data prep: {prep_host} "
            f"({prep_free_gb:.1f} GB free, need ~{REQUIRED_GB:.0f} GB)",
            flush=True,
        )
        if prep_free_gb < REQUIRED_GB:
            # Try rebalancing first, then re-check
            rebalance_storage(HOSTS)
            prep_host = get_freest_host(HOSTS)
            _, prep_free = get_disk_bytes(prep_host)
            prep_free_gb = prep_free / 1e9
            if prep_free_gb < REQUIRED_GB:
                raise OSError(
                    f"Insufficient disk space on all workers after cleanup. "
                    f"Freest is {prep_host} with {prep_free_gb:.1f} GB; need {REQUIRED_GB} GB."
                )

        print(
            f"[{ist_now_str()}] Preparing FineWeb-Edu 20B dataset on {prep_host} "
            f"(train=20B, sweep=5B, valid=10M)...",
            flush=True,
        )

        # If the freest worker is not w0, run the prep script there via SSH
        if prep_host == "w0":
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(ALGEBRAIC_PYTHON),
                str(ROOT / "scripts/prepare_fineweb_edu.py"),
                "--output-dir", str(DATA_DIR),
                "--train-tokens", "20000000000",
                "--sweep-tokens", "5000000000",
                "--valid-tokens", "10000000",
                "--dataset-path", "sample/100BT",
                "--threads", "64",
            ]
            print(f"[{ist_now_str()}] Running dataset preparation locally on w0...", flush=True)
            run_command_stream(cmd)
        else:
            # Run on a remote worker that has more space, then rsync back to w0
            remote_data = str(DATA_DIR)
            remote_root = str(ROOT)
            remote_py = str(ALGEBRAIC_PYTHON)
            prep_cmd = (
                f"mkdir -p {shlex.quote(remote_data)} && "
                f"cd {shlex.quote(remote_root)} && "
                f"PYTHONPATH={shlex.quote(str(ROOT / 'src'))} "
                f"{shlex.quote(remote_py)} {shlex.quote(str(ROOT / 'scripts/prepare_fineweb_edu.py'))} "
                f"--output-dir {shlex.quote(remote_data)} "
                f"--train-tokens 20000000000 --sweep-tokens 5000000000 "
                f"--valid-tokens 10000000 --dataset-path sample/100BT --threads 64"
            )
            print(
                f"[{ist_now_str()}] Running dataset preparation on {prep_host} (more free space)...",
                flush=True,
            )
            subprocess.run(["ssh", prep_host, prep_cmd], check=True)
            # Rsync the completed arrays back to w0
            print(f"[{ist_now_str()}] Syncing prepared dataset back to w0...", flush=True)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["rsync", "-a", "--info=progress2",
                 f"{prep_host}:{remote_data}/", f"{DATA_DIR}/"],
                check=True,
            )

        # Wait for completion marker
        for _ in range(60):
            if meta_path.is_file():
                try:
                    data = json.loads(meta_path.read_text())
                    if data.get("complete") and train_path.is_file() and sweep_path.is_file() and valid_path.is_file():
                        print(f"[{ist_now_str()}] Dataset preparation complete!", flush=True)
                        break
                except Exception:
                    pass
            time.sleep(10)
        else:
            raise RuntimeError("Dataset preparation did not complete within the expected time.")
    else:
        print(f"[{ist_now_str()}] FineWeb-Edu 20B dataset already present on w0.", flush=True)

    # Verify sweep file name: prepare_fineweb_edu uses sweep_filename(5B) -> "fineweb_sweep_5B.npy"
    if not sweep_path.is_file():
        alt_sweep = DATA_DIR / "fineweb_sweep_5000000000.npy"
        if alt_sweep.is_file():
            alt_sweep.rename(sweep_path)
            print(f"[{ist_now_str()}] Renamed sweep file to fineweb_sweep_5B.npy", flush=True)
        else:
            raise FileNotFoundError(f"Sweep file not found at {sweep_path} or {alt_sweep}")

    # Rebalance before the large sync (ensures all workers have headroom)
    rebalance_storage(HOSTS)

    # Sync dataset to all workers that don't already have it
    needs_sync_hosts = []
    for h in ("w1", "w2", "w3"):
        try:
            res = subprocess.check_output(
                ["ssh", h, f"test -f {shlex.quote(str(meta_path))} && "
                           f"test -f {shlex.quote(str(train_path))} && echo ok || echo missing"],
                text=True, timeout=10,
            ).strip()
            if res != "ok":
                needs_sync_hosts.append(h)
        except Exception:
            needs_sync_hosts.append(h)

    if needs_sync_hosts:
        print(f"[{ist_now_str()}] Syncing dataset to workers: {needs_sync_hosts}", flush=True)
        for h in needs_sync_hosts:
            _, free = get_disk_bytes(h)
            free_gb = free / 1e9
            if free_gb < 55.0:
                print(f"  [{h}] Only {free_gb:.1f} GB free — running rebalance before sync...", flush=True)
                rebalance_storage([h])
            print(f"[{ist_now_str()}] Syncing {DATA_DIR} to {h}...", flush=True)
            subprocess.run(["ssh", h, f"mkdir -p {shlex.quote(str(DATA_DIR))}"], check=True)
            subprocess.run(
                ["rsync", "-a", "--info=progress2", f"{DATA_DIR}/", f"{h}:{DATA_DIR}/"],
                check=True,
            )
        print(f"[{ist_now_str()}] Token datasets synced across all workers!\n", flush=True)
    else:
        print(f"[{ist_now_str()}] Dataset already synced across all workers.", flush=True)


# ---------------------------------------------------------------------------
# Phase 1 – Hyperparameter sweep
# ---------------------------------------------------------------------------

def make_sweep_jobs():
    """3 LR candidates × 2 architectures = 6 sweep arms, 5B tokens each."""
    vocab_size = get_vocab_size()
    candidates = [
        {"name": "lr6e-4",   "lr": 6e-4},   # lower bound
        {"name": "lr1.2e-3", "lr": 1.2e-3}, # baseline
        {"name": "lr2.4e-3", "lr": 2.4e-3}, # upper bound
    ]
    archs = ["standard", "algebraic"]
    seed = 42
    jobs = []
    for cand in candidates:
        for arch in archs:
            name = f"{arch}_{cand['name']}_s{seed}"
            args = [
                "--preset", arch,
                "--size", "1b",
                "--seq-len", "2048",
                "--batch", "512",
                "--micro", "1",
                "--tokens", "5000000000",
                "--lr", str(cand["lr"]),
                "--wd", "0.1",
                "--warmup", "0.02",
                "--min-lr-ratio", "0.1",
                "--b2", "0.95",
                "--schedule", "cosine",
                "--seed", str(seed),
                "--train", str(DATA_DIR / "fineweb_sweep_5B.npy"),
                "--valid", str(DATA_DIR / "fineweb_valid.npy"),
                "--eval-tokens", "8000000",
                "--eval-batch", "1",
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


def run_phase1_sweep():
    print(f"\n=======================================================", flush=True)
    print(f"[{ist_now_str()}] STARTING PHASE 1: 1B HYPERPARAMETER SWEEP", flush=True)
    print(f"=======================================================\n", flush=True)

    group = "hparam_sweep_1b"
    jobs = make_sweep_jobs()
    job_path = ROOT / "runs" / group / "jobs.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps([{"name": j["name"], "args": j["args"]} for j in jobs], indent=2))

    print(
        f"Total sweep jobs: {len(jobs)} ({len(jobs)//2} candidates × 2 archs, 5B tokens each)",
        flush=True,
    )
    rebalance_storage(HOSTS)

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

        # Immediately prune sweep checkpoints on ALL workers — we only need result.json
        run_dir = ROOT / "runs" / group / name
        clean_storage(HOSTS, prune_run_dir=run_dir)
        # Also delete params.npz if it somehow got saved (sweep runs don't use --save-params)
        for h in HOSTS:
            subprocess.run(
                ["ssh", h, f"rm -f {shlex.quote(str(run_dir / 'params.npz'))} 2>/dev/null || true"],
                timeout=10, check=False,
            )
        rebalance_storage(HOSTS)

    # Collect and analyze sweep results
    results_dir = ROOT / "results" / "hparam_sweep_1b"
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

    # Select best per arch (lowest validation NLL)
    selected = {}
    print("\n--- 1B Hyperparameter Sweep Results ---")
    for arch in ["standard", "algebraic"]:
        cands_for_arch = [m for m in measurements.values() if m["arch"] == arch]
        best = min(cands_for_arch, key=lambda x: x["final_eval_nll"])
        selected[arch] = {
            "cand_name": best["cand_name"],
            "lr": best["lr"],
            "final_eval_nll": best["final_eval_nll"],
            "final_eval_ppl": best["final_eval_ppl"],
        }
        print(
            f"{arch.upper()} Best: {best['cand_name']} (LR={best['lr']}) "
            f"with NLL={best['final_eval_nll']:.4f}, PPL={best['final_eval_ppl']:.2f}"
        )

    sweep_record = {
        "timestamp_ist": ist_now_str(),
        "purpose": "1B hyperparameter sweep across standard and algebraic architectures on 5B tokens per arm",
        "arms": ["lr6e-4 (lower)", "lr1.2e-3 (baseline)", "lr2.4e-3 (upper)"],
        "tokens_per_arm": 5_000_000_000,
        "architectures": ["standard", "algebraic"],
        "selected": selected,
        "table": table,
        "measurements": measurements,
    }
    (results_dir / "summary.json").write_text(json.dumps(sweep_record, indent=2))
    (results_dir / "selected.json").write_text(json.dumps({"selected": selected}, indent=2))

    # Clean intermediate logs/tmp
    rebalance_storage(HOSTS)

    # Push sweep results to GitHub
    print("\nPushing Phase 1 1B hyperparameter sweep results to GitHub...", flush=True)
    subprocess.run(["git", "add", "results/hparam_sweep_1b/"], cwd=ROOT, check=True)
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "results/hparam_sweep_1b/"], cwd=ROOT, text=True
    )
    if status.strip():
        subprocess.run(
            [
                "git", "commit", "-m",
                f"Record 1B hyperparameter sweep results (Std: {selected['standard']['lr']}, "
                f"Alg: {selected['algebraic']['lr']})",
            ],
            cwd=ROOT, check=True,
        )
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
        print("Phase 1 1B sweep successfully pushed to GitHub!", flush=True)
    else:
        print("Results already committed.", flush=True)

    return {arch: selected[arch]["lr"] for arch in selected}


# ---------------------------------------------------------------------------
# Phase 2 – Main 1B runs
# ---------------------------------------------------------------------------

def make_main_jobs(selected_lrs, seeds=(42,)):
    """1 seed × 2 architectures = 2 main runs, 20B tokens each."""
    vocab_size = get_vocab_size()
    archs = ["standard", "algebraic"]
    jobs = []
    for seed in seeds:
        for arch in archs:
            lr = selected_lrs[arch]
            name = f"{arch}_s{seed}"
            args = [
                "--preset", arch,
                "--size", "1b",
                "--seq-len", "2048",
                "--batch", "512",
                "--micro", "1",
                "--tokens", "20000000000",
                "--lr", str(lr),
                "--wd", "0.1",
                "--warmup", "0.02",
                "--min-lr-ratio", "0.1",
                "--b2", "0.95",
                "--schedule", "cosine",
                "--seed", str(seed),
                "--train", str(DATA_DIR / "fineweb_train_20B.npy"),
                "--valid", str(DATA_DIR / "fineweb_valid.npy"),
                "--eval-tokens", "8000000",
                "--eval-batch", "1",
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


def evaluate_model(name: str, run_dir: Path, results_dir: Path):
    """Run zero-shot and recall evaluations on a single model checkpoint."""
    results_dir.mkdir(parents=True, exist_ok=True)
    zs_out = results_dir / f"{name}_zero_shot.json"
    if not zs_out.is_file():
        print(f"\n[{ist_now_str()}] Evaluating zero-shot public tasks for {name}...", flush=True)
        cmd_zs = [
            "bash", str(ROOT / "scripts/tpu_python.sh"), "0",
            str(ROOT / "scripts/evaluate_zero_shot.py"),
            str(run_dir),
            "--benchmarks", str(Path.home() / "data/evals"),
            "--out", str(zs_out),
            "--batch", "8",
        ]
        run_command_stream(cmd_zs)
    else:
        print(f"[{ist_now_str()}] Zero-shot evaluation already exists for {name}.", flush=True)

    rc_out = results_dir / f"{name}_recall.json"
    if not rc_out.is_file():
        print(f"[{ist_now_str()}] Evaluating recall tasks for {name}...", flush=True)
        cmd_rc = [
            "bash", str(ROOT / "scripts/tpu_python.sh"), "0",
            str(ROOT / "scripts/evaluate_recall.py"),
            str(run_dir),
            "--benchmarks", str(Path.home() / "data/recall_llama32k"),
            "--out", str(rc_out),
            "--batch", "8",
        ]
        run_command_stream(cmd_rc)
    else:
        print(f"[{ist_now_str()}] Recall evaluation already exists for {name}.", flush=True)


def upload_model_to_hf(name: str, run_dir: Path, results_dir: Path):
    """Upload single model run (weights + evals) to Hugging Face under 1b/<name>/"""
    from huggingface_hub import HfApi
    api = HfApi()
    repo_id = "tasmaikeni13/algebraic-transformer"

    print(f"\n[{ist_now_str()}] Uploading {name} to Hugging Face ({repo_id}: 1b/{name})...", flush=True)

    run_files = ["params.npz", "meta.json", "result.json", "log.jsonl"]
    for fname in run_files:
        fpath = run_dir / fname
        if fpath.is_file():
            sz_mb = fpath.stat().st_size / 1e6
            print(f"  Uploading {fname} ({sz_mb:.1f} MB)...", flush=True)
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=f"1b/{name}/{fname}",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload 1B {name} {fname}",
            )

    eval_files = [
        f"{name}_zero_shot.json",
        f"{name}_zero_shot.predictions.jsonl",
        f"{name}_recall.json",
        f"{name}_recall.predictions.jsonl",
    ]
    for fname in eval_files:
        fpath = results_dir / fname
        if fpath.is_file():
            print(f"  Uploading eval {fname}...", flush=True)
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=f"1b/{name}/{fname}",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload 1B {name} eval {fname}",
            )
    print(f"[{ist_now_str()}] Successfully uploaded {name} to Hugging Face!", flush=True)


def delete_model_weights_locally(name: str, run_dir: Path):
    """Delete params.npz and checkpoints locally across workers after HF upload."""
    print(f"[{ist_now_str()}] Deleting local params.npz and pruning checkpoints for {name}...", flush=True)
    params_p = run_dir / "params.npz"
    if params_p.is_file():
        params_p.unlink()
    clean_storage(HOSTS, prune_run_dir=run_dir)
    # Also remove params.npz on all remote workers
    for h in HOSTS:
        if h == "w0":
            continue
        subprocess.run(
            ["ssh", h, f"rm -f {shlex.quote(str(run_dir / 'params.npz'))} 2>/dev/null || true"],
            timeout=15, check=False,
        )
    rebalance_storage(HOSTS)


def _load_result(group: str, arch: str, seed: int) -> dict:
    """Load result.json for a completed run, returning {} on failure."""
    res_file = ROOT / "runs" / group / f"{arch}_s{seed}" / "result.json"
    if res_file.is_file():
        return json.loads(res_file.read_text())
    return {}


def update_readme_github(final_report: dict, sweep_record: dict):
    """Append a 1B results section to the GitHub README.md and push."""
    print(f"\n[{ist_now_str()}] Updating GitHub README.md with 1B results...", flush=True)
    readme_path = ROOT / "README.md"
    content = readme_path.read_text()

    std = final_report["results"].get("standard", {})
    alg = final_report["results"].get("algebraic", {})
    std_nll = std.get("mean_nll", float("nan"))
    alg_nll = alg.get("mean_nll", float("nan"))
    std_ppl = std.get("mean_ppl", float("nan"))
    alg_ppl = alg.get("mean_ppl", float("nan"))
    delta_nll = alg_nll - std_nll
    delta_ppl_rel = (alg_ppl - std_ppl) / std_ppl * 100 if std_ppl else float("nan")

    std_lr = std.get("selected_lr", "?")
    alg_lr = alg.get("selected_lr", "?")
    sweep_std_best = sweep_record.get("selected", {}).get("standard", {}).get("cand_name", "?")
    sweep_alg_best = sweep_record.get("selected", {}).get("algebraic", {}).get("cand_name", "?")

    section = f"""
---

## 1B Results (20B Tokens, Seed: 42)

The 1B runs used a 32k LLaMA-style BPE tokenizer on FineWeb-Edu (100BT subset), trained across 4 worker \
hosts on a TPU v4-32 slice (batch size 512, sequence length 2,048). Prior to the main run, a hyperparameter \
sweep was conducted across `{{6e-4, 1.2e-3, 2.4e-3}}` (lower / baseline / upper) on 5B tokens per arm; \
`{std_lr}` was optimal for Standard and `{alg_lr}` for Algebraic (sweep winners: std={sweep_std_best}, \
alg={sweep_alg_best}).

### Language Modeling (1B, Seed 42)

| Metric | Standard 1B | Algebraic 1B | Delta (Δ) |
|---|---|---|---|
| **Held-Out NLL (nats/token)** | {std_nll:.4f} | **{alg_nll:.4f}** | **{delta_nll:+.4f}** |
| **Held-Out Perplexity** | {std_ppl:.2f} | **{alg_ppl:.2f}** | **{delta_ppl_rel:+.2f}%** |
| **Selected LR (Standard)** | {std_lr} | — | — |
| **Selected LR (Algebraic)** | — | {alg_lr} | — |

Full zero-shot benchmark and recall results are available in \
[`results/main_1b/`](results/main_1b/) and on \
[Hugging Face](https://huggingface.co/tasmaikeni13/algebraic-transformer).
"""

    # Inject before the "Why Eliminate Transcendentals?" section so the 1B block
    # appears alongside 125M and 350M results.
    marker = "## Why Eliminate Transcendentals?"
    if "## 1B Results" not in content:
        if marker in content:
            content = content.replace(f"\n## {marker.lstrip('## ')}", section + f"\n## {marker.lstrip('## ')}")
        else:
            content = content + section
        readme_path.write_text(content)

    # Also update the HF Hub tree listing
    hub_tree_marker = "└── 350m/"
    new_tree_line = (
        "└── 350m/                          # 350M models (7.0B tokens, 3 seeds)\n"
        "    ├── standard_s42/ ... s44/     # Model weights (params.npz), meta.json, result.json, log.jsonl\n"
        "    ├── algebraic_s42/ ... s44/    # Model weights (params.npz), meta.json, result.json, log.jsonl\n"
        "    └── evaluations/               # comparison.md, comparison.json, recall_cells.csv, needle_recall.png, etc.\n"
        "└── 1b/                            # 1B models (20B tokens, seed 42)\n"
        "    ├── standard_s42/              # Model weights (params.npz), meta.json, result.json, log.jsonl\n"
        "    ├── algebraic_s42/             # Model weights (params.npz), meta.json, result.json, log.jsonl\n"
        "    └── evaluations/               # comparison_summary.json, zero-shot, recall results"
    )
    readme_content = readme_path.read_text()
    if "└── 1b/" not in readme_content and hub_tree_marker in readme_content:
        readme_content = readme_content.replace(hub_tree_marker, new_tree_line)
        readme_path.write_text(readme_content)

    # Also update the HF badge to mention 1B
    readme_content = readme_path.read_text()
    old_badge = "[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-125M%20%26%20350M%20Models-blue)](https://huggingface.co/tasmaikeni13/algebraic-transformer)"
    new_badge = "[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-125M%20%7C%20350M%20%7C%201B%20Models-blue)](https://huggingface.co/tasmaikeni13/algebraic-transformer)"
    if old_badge in readme_content:
        readme_content = readme_content.replace(old_badge, new_badge)
        readme_path.write_text(readme_content)

    subprocess.run(["git", "add", "README.md"], cwd=ROOT, check=True)
    status = subprocess.check_output(["git", "status", "--porcelain", "README.md"], cwd=ROOT, text=True)
    if status.strip():
        subprocess.run(
            ["git", "commit", "-m", "docs: add 1B results to README (20B tokens, seed 42)"],
            cwd=ROOT, check=True,
        )
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
        print("[{ist_now_str()}] README.md updated and pushed to GitHub!", flush=True)
    else:
        print(f"[{ist_now_str()}] README.md already up to date.", flush=True)


def update_readme_huggingface(final_report: dict, sweep_record: dict):
    """Update the Hugging Face model card (README.md on the HF repo) with 1B results."""
    print(f"\n[{ist_now_str()}] Updating Hugging Face model card with 1B results...", flush=True)
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_id = "tasmaikeni13/algebraic-transformer"

        std = final_report["results"].get("standard", {})
        alg = final_report["results"].get("algebraic", {})
        std_nll = std.get("mean_nll", float("nan"))
        alg_nll = alg.get("mean_nll", float("nan"))
        std_ppl = std.get("mean_ppl", float("nan"))
        alg_ppl = alg.get("mean_ppl", float("nan"))
        delta_nll = alg_nll - std_nll
        delta_ppl_rel = (alg_ppl - std_ppl) / std_ppl * 100 if std_ppl else float("nan")
        std_lr = std.get("selected_lr", "?")
        alg_lr = alg.get("selected_lr", "?")
        sweep_std_best = sweep_record.get("selected", {}).get("standard", {}).get("cand_name", "?")
        sweep_alg_best = sweep_record.get("selected", {}).get("algebraic", {}).get("cand_name", "?")

        # Upload the local README.md (which we already updated) as the HF model card
        api.upload_file(
            path_or_fileobj=str(ROOT / "README.md"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message=(
                f"docs: add 1B results (Std NLL={std_nll:.4f}, Alg NLL={alg_nll:.4f}, "
                f"ΔNLL={delta_nll:+.4f}, ΔPPLrel={delta_ppl_rel:+.2f}%)"
            ),
        )
        print(f"[{ist_now_str()}] HuggingFace model card updated successfully!", flush=True)
    except Exception as e:
        print(f"[{ist_now_str()}] Warning: HF model card update failed: {e}", flush=True)


def run_phase2_main(selected_lrs):
    print(f"\n=======================================================", flush=True)
    print(f"[{ist_now_str()}] STARTING PHASE 2: MAIN 1B COMPARISON RUNS", flush=True)
    print(f"=======================================================\n", flush=True)
    print(f"Selected LRs: Standard={selected_lrs['standard']}, Algebraic={selected_lrs['algebraic']}", flush=True)

    group = "main_1b"
    seeds = (42,)
    jobs = make_main_jobs(selected_lrs, seeds=seeds)
    job_path = ROOT / "runs" / group / "jobs.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps([{"name": j["name"], "args": j["args"]} for j in jobs], indent=2))

    results_dir = ROOT / "results" / "main_1b"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Total main runs: {len(jobs)} (20B tokens, seed=42 only)", flush=True)

    for i, j in enumerate(jobs, 1):
        name = j["name"]
        out_dir = ROOT / "runs" / group / name

        # Rebalance storage before launching each run (20B run checkpoints are large)
        rebalance_storage(HOSTS)

        # 1. Train
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
        print(f"[{ist_now_str()}] Completed {name} training in {dt:.1f} minutes.")

        # 2. Prune training intermediate checkpoints (keep params.npz for evaluation)
        clean_storage(HOSTS, prune_run_dir=out_dir)

        # 3. Evaluate on zero-shot + recall benchmarks
        evaluate_model(name, out_dir, results_dir)

        # 4. Upload weights + evals to Hugging Face under 1b/<name>/
        upload_model_to_hf(name, out_dir, results_dir)

        # 5. Delete local weights to free disk before next run
        delete_model_weights_locally(name, out_dir)

    # 6. Summarize results
    print(f"\n[{ist_now_str()}] Summarizing main 1B evaluation results ...", flush=True)
    try:
        cmd_sum = [
            str(ALGEBRAIC_PYTHON), str(ROOT / "scripts/summarize_main_eval.py"),
            "--out-dir", str(results_dir),
            "--archs", "standard,algebraic",
            "--seeds", "42",
        ]
        run_command_stream(cmd_sum)
    except Exception as e:
        print(f"Warning: summarize_main_eval failed: {e}", flush=True)

    main_records = {}
    for arch in ["standard", "algebraic"]:
        nlls, ppls = [], []
        for s in seeds:
            res_file = ROOT / "runs" / group / f"{arch}_s{s}" / "result.json"
            if res_file.is_file():
                d = json.loads(res_file.read_text())
                nlls.append(d["final_eval_nll"])
                ppls.append(d["final_eval_ppl"])
        mean_nll = nlls[0] if nlls else float("nan")
        mean_ppl = ppls[0] if ppls else float("nan")
        main_records[arch] = {
            "seeds": list(seeds),
            "nll_per_seed": nlls,
            "mean_nll": mean_nll,
            "mean_ppl": mean_ppl,
            "selected_lr": selected_lrs[arch],
        }
        print(f"\n{arch.upper()} FINAL (seed 42): NLL = {mean_nll:.4f}, PPL = {mean_ppl:.2f}")

    final_report = {
        "completed_at_ist": ist_now_str(),
        "scale": "1b",
        "tokens_per_run": 20_000_000_000,
        "seeds": list(seeds),
        "results": main_records,
    }
    (results_dir / "comparison_summary.json").write_text(json.dumps(final_report, indent=2))

    # 7. Push evaluation results to GitHub
    print("\nPushing final 1B main comparison results to GitHub...", flush=True)
    subprocess.run(["git", "add", "results/main_1b/"], cwd=ROOT, check=True)
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "results/main_1b/"], cwd=ROOT, text=True
    )
    if status.strip():
        subprocess.run(
            [
                "git", "commit", "-m",
                "Record completed 1B main comparison runs (standard vs algebraic) on 20B tokens",
            ],
            cwd=ROOT, check=True,
        )
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
        print("Main 1B comparison results pushed to GitHub!", flush=True)

    # 8. Load sweep record for README context
    sweep_record = {}
    sweep_summary = ROOT / "results" / "hparam_sweep_1b" / "summary.json"
    if sweep_summary.is_file():
        sweep_record = json.loads(sweep_summary.read_text())

    # 9. Update GitHub README.md with 1B results section
    update_readme_github(final_report, sweep_record)

    # 10. Update Hugging Face model card (uploads the updated README.md)
    update_readme_huggingface(final_report, sweep_record)

    print(f"\n[{ist_now_str()}] ALL 1B RUNS, EVALUATIONS, AND README UPDATES COMPLETED SUCCESSFULLY!", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=("all", "data", "sweep", "main"),
        default="all",
        help="'data'=dataset prep only; 'sweep'=sweep only; 'main'=main runs only; 'all'=full pipeline",
    )
    p.add_argument("--standard-lr", type=float, default=None, help="Override standard LR for main run")
    p.add_argument("--algebraic-lr", type=float, default=None, help="Override algebraic LR for main run")
    args = p.parse_args()

    if args.phase in ("all", "data"):
        ensure_data_ready_and_synced()

    selected_lrs = {}
    if args.phase in ("all", "sweep"):
        if args.phase != "all":
            # data not run above; just check it's ready
            ensure_data_ready_and_synced()
        selected_lrs = run_phase1_sweep()

    if args.phase in ("all", "main"):
        if not selected_lrs:
            sel_file = ROOT / "results" / "hparam_sweep_1b" / "selected.json"
            if sel_file.is_file():
                sel_data = json.loads(sel_file.read_text()).get("selected", {})
                selected_lrs = {
                    "standard": args.standard_lr or sel_data.get("standard", {}).get("lr", 1.2e-3),
                    "algebraic": args.algebraic_lr or sel_data.get("algebraic", {}).get("lr", 1.2e-3),
                }
            else:
                selected_lrs = {
                    "standard": args.standard_lr or 1.2e-3,
                    "algebraic": args.algebraic_lr or 1.2e-3,
                }
        run_phase2_main(selected_lrs)


if __name__ == "__main__":
    raise SystemExit(main())
