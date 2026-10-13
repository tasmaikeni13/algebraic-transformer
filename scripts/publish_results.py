#!/usr/bin/env python3
"""Publish 125M comparison outputs to GitHub and Hugging Face Hub.

Actions:
  1. Git: Stages and commits results/main/ (comparison tables, CSVs, eval jsons, plots)
     and pushes to origin/main.
  2. Hugging Face: Uploads model checkpoints (params.npz), evaluation suite,
     and model card to tasmaikeni13/sota-algebra-125m.
"""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="tasmaikeni13/algebraic-transformer")
    p.add_argument("--no-git", action="store_true")
    p.add_argument("--no-hf", action="store_true")
    args = p.parse_args()

    # 1. GitHub push
    if not args.no_git:
        print("=== Staging and pushing results to GitHub ===")
        results_dir = ROOT / "results/main"
        if not results_dir.exists():
            print(f"Error: {results_dir} does not exist yet.")
            return 1

        subprocess.check_call(["git", "add", "results/main", "scripts/run_main_pipeline.py", "scripts/upload_to_hf.py", "scripts/publish_results.py", "main.json"], cwd=str(ROOT))
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT), text=True)
        if status.strip():
            subprocess.check_call([
                "git", "commit", "-m",
                "Publish 125M 3-arm comparison benchmark results, comparison tables, and needle recall heatmaps",
            ], cwd=str(ROOT))
            subprocess.check_call(["git", "push", "origin", "main"], cwd=str(ROOT))
            print("Successfully pushed results to GitHub main branch!")
        else:
            print("Nothing to commit on GitHub.")

    # 2. Hugging Face upload
    if not args.no_hf:
        print("=== Uploading models and evaluations to Hugging Face Hub ===")
        subprocess.check_call([
            sys.executable,
            str(ROOT / "scripts/upload_to_hf.py"),
            "--repo-id", args.repo_id,
        ], cwd=str(ROOT))
        print(f"Successfully uploaded to Hugging Face: https://huggingface.co/{args.repo_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
