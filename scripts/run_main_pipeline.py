#!/usr/bin/env python3
"""Run the 125M comparison and then its complete public and recall benchmarks."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.main_run import main_jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hosts", default="w0,w1,w2,w3")
    p.add_argument("--data-dir", type=Path, default=Path.home() / "data/fw25")
    p.add_argument("--benchmarks", type=Path, default=Path.home() / "data/evals")
    p.add_argument("--recall-benchmarks", type=Path, default=Path.home() / "data/recall")
    p.add_argument("--eval-chip", type=int, choices=range(4), default=0)
    p.add_argument("--run", action="store_true", help="train and evaluate; otherwise print jobs only")
    a = p.parse_args()
    import scripts.main_run as generator
    generator.DATA = a.data_dir.resolve()
    jobs = main_jobs(["standard", "algebraic", "hybrid"])
    if not a.run:
        print(json.dumps(jobs, indent=2))
        return 0
    job_file = ROOT / "runs/main/jobs.json"
    job_file.parent.mkdir(parents=True, exist_ok=True)
    job_file.write_text(json.dumps(jobs, indent=2) + "\n")
    subprocess.run([sys.executable, str(ROOT / "scripts/run_slice.py"), str(job_file),
                    "--group", "main", "--hosts", a.hosts, "--skip-done"], cwd=ROOT, check=True)
    subprocess.run(["bash", str(ROOT / "scripts/tpu_python.sh"), str(a.eval_chip),
                    "scripts/evaluate_main.py", "--benchmarks", str(a.benchmarks),
                    "--recall-benchmarks", str(a.recall_benchmarks), "--resume"], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
