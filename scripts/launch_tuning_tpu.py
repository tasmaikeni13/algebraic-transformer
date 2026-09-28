#!/usr/bin/env python3
"""Copy a committed snapshot and launch Tuning study hyperparameter sweep on the TPU v4-32 Pod slice.

Executes the equal-budget sweep study across 16 TPU v4 chips across 4 hosts in us-central2-b.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="my-tpu-v4")
    parser.add_argument("--zone", default="us-central2-b")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tuning/tpu")
    parser.add_argument("--tokens-per-run", type=int, default=600_000_000, help="Tokens per run")
    parser.add_argument("--batch-size", type=int, default=512, help="Global batch size in sequences")
    parser.add_argument("--seq-len", type=int, default=2048, help="Context length")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("Refusing to launch Tuning study from a dirty worktree; commit the exact source snapshot first")

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    def run(command, log):
        with (out / log).open("w") as file:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                file.write(line)
                file.flush()
                print(line, end="", flush=True)
            code = process.wait()
        if code:
            raise RuntimeError(f"Command failed with exit {code}; see {out / log}")

    def ssh(command, log, worker="all"):
        run(["gcloud", "compute", "tpus", "tpu-vm", "ssh", args.name, "--zone", args.zone,
             "--worker", worker, "--quiet", "--command", command], log)

    print("Checking TPU accelerator availability...", flush=True)
    ssh("if sudo -n fuser -s /dev/accel*; then echo 'TPU devices in use; waiting/exiting'; exit 73; fi", "availability.log")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if not branch:
        raise RuntimeError("Refusing to launch Tuning study from a detached HEAD")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = f"algebraic-tuning-{commit[:12]}-{stamp}"
    remote = f"/tmp/{label}"
    quote = shlex.quote

    print(f"Creating Git bundle for snapshot: {label}...", flush=True)
    with tempfile.TemporaryDirectory(prefix="algebraic-tuning-") as temporary:
        bundle = Path(temporary) / "source.bundle"
        subprocess.run(["git", "bundle", "create", str(bundle), branch], cwd=ROOT, check=True)
        run(["gcloud", "compute", "tpus", "tpu-vm", "scp", str(bundle), f"{args.name}:{remote}.bundle",
             "--zone", args.zone, "--worker", "all", "--quiet"], "copy.log")

    print("Cloning snapshot and installing dependencies across all TPU workers...", flush=True)
    setup = (
        f"git clone --quiet --branch {quote(branch)} {quote(remote+'.bundle')} {quote(remote)} && "
        f"python3 -m venv {quote(remote+'/venv')} && "
        f"{quote(remote+'/venv/bin/pip')} install -r {quote(remote+'/requirements-tpu.txt')}"
    )
    ssh(setup, "setup.log")

    # Distribute FineWeb-Edu tokenized cache to all workers
    sweep_file = ROOT / "data/fineweb_sweep_600M.npy"
    if not sweep_file.exists():
        sweep_file = ROOT / "data/fineweb_train_2_5B.npy"
    valid_file = ROOT / "data/fineweb_valid.npy"

    if not (sweep_file.exists() and valid_file.exists()):
        raise FileNotFoundError(
            "Tuning study requires data/fineweb_sweep_600M.npy (or fineweb_train_2_5B.npy) "
            "and data/fineweb_valid.npy"
        )
    print("Distributing FineWeb-Edu dataset cache to all TPU workers...", flush=True)
    ssh(f"mkdir -p {quote(remote+'/data')}", "mkdir-data.log")
    run([
        "gcloud", "compute", "tpus", "tpu-vm", "scp",
        str(sweep_file), str(valid_file),
        f"{args.name}:{remote}/data/",
        "--zone", args.zone, "--worker", "all", "--quiet"
    ], "copy-data.log")

    print("Running host unit tests across workers...", flush=True)
    ssh(f"cd {quote(remote)} && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 venv/bin/python -m pytest -q tests/test_hparam_contracts.py", "host-tests.log")
    ssh(f"mkdir -p {quote(remote+'/measurements')}", "mkdir-measurements.log")
    (out / "snapshot.txt").write_text(f"commit={commit}\nremote_directory={remote}\n")

    seeds_str = " ".join(str(s) for s in args.seeds)
    cmd = (
        f"cd {quote(remote)} && PYTHONUNBUFFERED=1 venv/bin/python scripts/run_hparam_sweep.py "
        f"--output-dir {quote(remote+'/measurements')} "
        f"--tokens-per-run {args.tokens_per_run} "
        f"--batch-size {args.batch_size} "
        f"--seq-len {args.seq_len} "
        f"--seeds {seeds_str} "
        f"--log-every {args.log_every}"
    )

    run_error = None
    try:
        print("Launching distributed hyperparameter sweep across all 16 TPU v4 chips...", flush=True)
        ssh(cmd, "run.log")
        verify = (
            f"cd {quote(remote)} && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 "
            f"venv/bin/python scripts/run_verify_tuning.py "
            f"--metrics {quote(remote+'/measurements/metrics.json')} "
            f"--output-dir {quote(remote+'/measurements')}"
        )
        ssh(verify, "verify.log", worker="0")
    except Exception as exc:
        run_error = exc

    print("Downloading sweep results from workers...", flush=True)
    download = out / "download" / label
    download.mkdir(parents=True, exist_ok=True)
    download_errors = []
    for worker in range(4):
        try:
            run([
                "gcloud", "compute", "tpus", "tpu-vm", "scp", "--recurse",
                f"{args.name}:{remote}/measurements", str(download / f"worker-{worker}"),
                "--zone", args.zone, "--worker", str(worker), "--quiet"
            ], f"download-{worker}.log")
        except Exception as exc:
            download_errors.append((worker, exc))
    if run_error is not None:
        raise run_error
    if download_errors:
        raise RuntimeError(f"Failed to download Tuning study results: {download_errors}")
    candidates = list(download.rglob("metrics.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one coordinator metrics.json, found {len(candidates)}")
    coord_dir = candidates[0].parent
    for file in coord_dir.iterdir():
        if file.is_file():
            shutil.copy2(file, out / file.name)
            # Also copy to results/tuning/
            shutil.copy2(file, (ROOT / "results/tuning") / file.name)
    print(f"Successfully retrieved Tuning study artifacts to {out} and results/tuning/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
