#!/usr/bin/env python3
"""Launch all six Pretraining study runs and evaluations on an existing TPU v4-32."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="my-tpu-v4")
    parser.add_argument("--zone", default="us-central2-b")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pretraining/tpu")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("Refusing to launch Pretraining study from a dirty worktree")

    # Fail locally before any cloud work when Tuning study evidence is absent or stale.
    subprocess.run([
        str(ROOT / ".venv/bin/python"), "-c",
        "from scripts.pretraining_experiments import load_verified_tuning_configs; "
        "from pathlib import Path; load_verified_tuning_configs(Path('results/tuning'))",
    ], cwd=ROOT, check=True)

    train_data = ROOT / "data/fineweb_train_2_5B.npy"
    valid_data = ROOT / "data/fineweb_valid.npy"
    eval_dir = ROOT / "data/evals"
    required_evals = [eval_dir / f"{name}.jsonl" for name in (
        "arc_easy", "hellaswag", "piqa", "lambada"
    )] + [eval_dir / "metadata.json"]
    if not train_data.exists() or not valid_data.exists() or not all(p.exists() for p in required_evals):
        raise FileNotFoundError(
            "Prepare FineWeb-Edu and run scripts/prepare_eval_benchmarks.py before Pretraining study"
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    def run(command, log):
        with (output / log).open("w") as stream:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            code = process.wait()
        if code:
            raise RuntimeError(f"Command failed with exit {code}; see {output / log}")

    def ssh(command, log, worker="all"):
        run([
            "gcloud", "compute", "tpus", "tpu-vm", "ssh", args.name,
            "--zone", args.zone, "--worker", worker, "--quiet", "--command", command,
        ], log)

    ssh(
        "if sudo -n fuser -s /dev/accel*; then echo 'TPU devices are in use'; exit 73; fi",
        "availability.log",
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if not branch:
        raise RuntimeError("Refusing to launch Pretraining study from a detached HEAD")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = f"algebraic-pretraining-{commit[:12]}-{stamp}"
    remote = f"/tmp/{label}"
    quote = shlex.quote

    with tempfile.TemporaryDirectory(prefix="algebraic-pretraining-") as directory:
        bundle = Path(directory) / "source.bundle"
        subprocess.run(["git", "bundle", "create", str(bundle), branch], cwd=ROOT, check=True)
        run([
            "gcloud", "compute", "tpus", "tpu-vm", "scp", str(bundle),
            f"{args.name}:{remote}.bundle", "--zone", args.zone,
            "--worker", "all", "--quiet",
        ], "copy-source.log")

    setup = (
        f"git clone --quiet --branch {quote(branch)} {quote(remote + '.bundle')} {quote(remote)} && "
        f"python3 -m venv {quote(remote + '/venv')} && "
        f"{quote(remote + '/venv/bin/pip')} install -r {quote(remote + '/requirements-tpu.txt')}"
    )
    ssh(setup, "setup.log")
    ssh(f"mkdir -p {quote(remote + '/data/evals')}", "mkdir-data.log")
    run([
        "gcloud", "compute", "tpus", "tpu-vm", "scp",
        str(train_data), str(valid_data), f"{args.name}:{remote}/data/",
        "--zone", args.zone, "--worker", "all", "--quiet",
    ], "copy-training-data.log")
    run([
        "gcloud", "compute", "tpus", "tpu-vm", "scp",
        *[str(path) for path in required_evals], f"{args.name}:{remote}/data/evals/",
        "--zone", args.zone, "--worker", "all", "--quiet",
    ], "copy-eval-data.log")

    ssh(
        f"cd {quote(remote)} && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 "
        f"venv/bin/python scripts/smoke_pretraining.py",
        "smoke.log",
    )
    (output / "snapshot.txt").write_text(
        f"commit={commit}\nremote_directory={remote}\n"
    )
    ssh(f"mkdir -p {quote(remote + '/measurements')}", "mkdir-measurements.log")

    runs = [(architecture, seed) for architecture in ("algebraic", "standard") for seed in (42, 43, 44)]
    if args.resume_from is not None:
        for architecture, seed in runs:
            run_name = f"{architecture}-{seed}"
            matches = sorted(args.resume_from.rglob(f"{run_name}/checkpoints/step-*.pkl"))
            if matches:
                destination = f"{remote}/measurements/{run_name}/checkpoints"
                ssh(f"mkdir -p {quote(destination)}", f"resume-mkdir-{run_name}.log")
                run([
                    "gcloud", "compute", "tpus", "tpu-vm", "scp", str(matches[-1]),
                    f"{args.name}:{destination}/", "--zone", args.zone,
                    "--worker", "all", "--quiet",
                ], f"resume-copy-{run_name}.log")

    run_error = None
    try:
        for architecture, seed in runs:
            run_name = f"{architecture}-{seed}"
            run_output = f"{remote}/measurements/{run_name}"

            train_command = (
                f"cd {quote(remote)} && PYTHONUNBUFFERED=1 venv/bin/python "
                f"scripts/run_pretrain_125m.py --architecture {architecture} --seed {seed} "
                f"--output-dir {quote(run_output)} --log-every {args.log_every}"
            )
            ssh(train_command, f"train-{run_name}.log")

            eval_command = (
                f"cd {quote(remote)} && TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 PYTHONUNBUFFERED=1 venv/bin/python "
                f"scripts/evaluate_benchmarks.py --architecture {architecture} --seed {seed} "
                f"--checkpoint-dir {quote(run_output + '/checkpoints')} "
                f"--benchmark-dir {quote(remote + '/data/evals')} "
                f"--output {quote(run_output + '/benchmarks.json')}"
            )
            ssh(eval_command, f"evaluate-{run_name}.log", worker="0")

            # Prune heavy checkpoints across all workers to guarantee bounded disk usage
            clean_checkpoints = f"rm -rf {quote(run_output + '/checkpoints')}"
            ssh(clean_checkpoints, f"clean-checkpoints-{run_name}.log")

        aggregate = (
            f"cd {quote(remote)} && venv/bin/python scripts/aggregate_pretraining.py "
            f"--input-dir {quote(remote + '/measurements')} "
            f"--output-dir {quote(remote + '/measurements')}"
        )
        ssh(aggregate, "aggregate.log", worker="0")
        verify = (
            f"cd {quote(remote)} && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 "
            f"venv/bin/python scripts/run_verify_pretraining.py "
            f"--ledger {quote(remote + '/measurements/pretraining_ledger.json')} "
            f"--output-dir {quote(remote + '/measurements')}"
        )
        ssh(verify, "verify.log", worker="0")
    except Exception as exc:
        run_error = exc

    download = output / "download" / label
    download.mkdir(parents=True, exist_ok=True)
    download_error = None
    try:
        run([
            "gcloud", "compute", "tpus", "tpu-vm", "scp", "--recurse",
            f"{args.name}:{remote}/measurements", str(download),
            "--zone", args.zone, "--worker", "0", "--quiet",
        ], "download.log")
    except Exception as exc:
        download_error = exc
    if run_error is not None:
        raise run_error
    if download_error is not None:
        raise download_error
    measurement_dirs = list(download.rglob("pretraining_ledger.json"))
    if len(measurement_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one Pretraining study pretraining ledger, found {len(measurement_dirs)}"
        )
    source_dir = measurement_dirs[0].parent
    for path in source_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, ROOT / "results/pretraining" / path.name)
        elif path.is_dir():
            shutil.copytree(path, ROOT / "results/pretraining" / path.name, dirs_exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
