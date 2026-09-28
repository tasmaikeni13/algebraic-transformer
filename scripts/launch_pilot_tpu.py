#!/usr/bin/env python3
"""Copy a committed snapshot and run Pilot study pilot pretraining on the idle TPU slice.

Does not stop other jobs or create cloud resources. Source snapshots are isolated
in new per-run directories on each host. Requires gcloud and existing SSH access.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="my-tpu-v4")
    parser.add_argument("--zone", default="us-central2-b")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pilot/tpu")
    parser.add_argument("--steps", type=int, default=100_000, help="Total pretraining steps")
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(
            "Refusing to launch Pilot study from a dirty worktree; commit the exact snapshot first"
        )

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

    ssh("if sudo -n fuser -s /dev/accel*; then echo 'TPU devices are in use; leave the existing job running.'; exit 73; fi", "availability.log")

    # Ensure the bundle contains reviewed source files.
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts", "tests", "formal", "requirements.txt", "requirements-tpu.txt"], cwd=ROOT, check=True)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "src", "scripts", "tests", "formal", "requirements.txt", "requirements-tpu.txt"], cwd=ROOT, text=True)
    if untracked:
        raise RuntimeError(f"Commit Pilot study source files before launching a distributed snapshot:\n{untracked}")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if not branch:
        raise RuntimeError("Launch from a named branch.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = f"algebraic-pilot-{commit[:12]}-{stamp}"
    remote = f"/tmp/{label}"
    quote = shlex.quote

    with tempfile.TemporaryDirectory(prefix="algebraic-pilot-") as temporary:
        bundle = Path(temporary) / "source.bundle"
        subprocess.run(["git", "bundle", "create", str(bundle), branch], cwd=ROOT, check=True)
        run(["gcloud", "compute", "tpus", "tpu-vm", "scp", str(bundle), f"{args.name}:{remote}.bundle", "--zone", args.zone, "--worker", "all", "--quiet"], "copy.log")

    setup = (f"git clone --quiet --branch {quote(branch)} {quote(remote+'.bundle')} {quote(remote)} && "
             f"python3 -m venv {quote(remote+'/venv')} && "
             f"{quote(remote+'/venv/bin/pip')} install -r {quote(remote+'/requirements-tpu.txt')}")
    ssh(setup, "setup.log")

    # Copy the same tokenized cache to every remote worker, including worker 0.
    train_data = ROOT / "data/train.npy"
    valid_data = ROOT / "data/valid.npy"
    if not (train_data.exists() and valid_data.exists()):
        raise FileNotFoundError("Pilot study requires data/train.npy and data/valid.npy")
    print("Distributing pre-tokenized dataset cache to all TPU workers...", flush=True)
    ssh(f"mkdir -p {quote(remote+'/data')}", "mkdir-data.log")
    run([
        "gcloud", "compute", "tpus", "tpu-vm", "scp",
        str(train_data), str(valid_data), f"{args.name}:{remote}/data/",
        "--zone", args.zone, "--worker", "all", "--quiet",
    ], "copy-data.log")

    # Separate CPU environments on every host do not initialize the TPU runtime.
    ssh(f"cd {quote(remote)} && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 venv/bin/python -m pytest -q", "host-tests.log")
    ssh(f"mkdir -p {quote(remote+'/measurements')}", "mkdir-measurements.log")
    (out / "snapshot.txt").write_text(f"commit={commit}\nremote_directory={remote}\n")

    cmd = (
        f"cd {quote(remote)} && PYTHONUNBUFFERED=1 venv/bin/python scripts/run_pilot_20m.py "
        f"--output {quote(remote+'/measurements')} --steps {args.steps} "
        f"--log-every {args.log_every}"
    )

    run_error = None
    try:
        ssh(cmd, "run.log")
    except Exception as exc:
        run_error = exc

    download = out / "download" / label
    download.mkdir(parents=True)
    download_errors = []
    for worker in range(4):
        try:
            run(["gcloud", "compute", "tpus", "tpu-vm", "scp", "--recurse",
                 f"{args.name}:{remote}/measurements", str(download / f"worker-{worker}"),
                 "--zone", args.zone, "--worker", str(worker), "--quiet"], f"download-{worker}.log")
        except Exception as exc:
            download_errors.append((worker, exc))
    if run_error is not None:
        raise run_error
    if download_errors:
        raise RuntimeError(f"Failed to download Pilot study results: {download_errors}")
    candidates = list(download.rglob("metrics.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one coordinator record; found {len(candidates)} in {download}")
    import shutil
    for file in candidates[0].parent.iterdir():
        if file.is_file():
            shutil.copy2(file, out / file.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
