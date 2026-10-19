#!/usr/bin/env python3
"""Stage and push Algebraic Transformer checkpoints to Hugging Face Hub."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def get_model_card() -> str:
    frontmatter = """---
license: apache-2.0
pipeline_tag: text-generation
tags:
- language-modeling
- algebraic-transformer
- jax
- pallas
- tpu
---

"""
    readme_path = ROOT / "README.md"
    if readme_path.is_file():
        return frontmatter + readme_path.read_text()
    return frontmatter + "# Algebraic Transformer\n"


MODELS_125M = [
    "standard_s42", "standard_s43", "standard_s44", "standard_s45", "standard_s46",
    "algebraic_s42", "algebraic_s43", "algebraic_s44", "algebraic_s45", "algebraic_s46",
]

MODELS_350M = [
    "standard_s42", "standard_s43", "standard_s44",
    "algebraic_s42", "algebraic_s43", "algebraic_s44",
]


def stage_repository(stage_root):
    stage_root = Path(stage_root)
    (stage_root / "125M").mkdir(parents=True, exist_ok=True)
    (stage_root / "350m").mkdir(parents=True, exist_ok=True)
    readme_path = stage_root / "README.md"
    readme_path.write_text(get_model_card())

    # Stage 125M models
    for model_name in MODELS_125M:
        src_dir = ROOT / "runs/main_125m" / model_name
        dest_dir = stage_root / "125M" / model_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("params.npz", "meta.json", "result.json", "log.jsonl"):
            src_file = src_dir / fname
            dest_file = dest_dir / fname
            if src_file.exists():
                try:
                    os.link(src_file, dest_file)
                except OSError:
                    shutil.copy2(src_file, dest_file)

    # Stage 125M evaluations
    eval_src_125m = ROOT / "results/main_125m"
    eval_dest_125m = stage_root / "125M/evaluations"
    eval_dest_125m.mkdir(parents=True, exist_ok=True)
    for fname in ("comparison.md", "comparison.json", "comparison.csv",
                  "recall_cells.csv", "needle_recall.png", "needle_recall.pdf", "comparison_summary.json"):
        src = eval_src_125m / fname
        if src.exists():
            shutil.copy2(src, eval_dest_125m / fname)

    # Stage 350M models
    for model_name in MODELS_350M:
        src_dir = ROOT / "runs/main_350m" / model_name
        dest_dir = stage_root / "350m" / model_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("params.npz", "meta.json", "result.json", "log.jsonl"):
            src_file = src_dir / fname
            dest_file = dest_dir / fname
            if src_file.exists():
                try:
                    os.link(src_file, dest_file)
                except OSError:
                    shutil.copy2(src_file, dest_file)

    # Stage 350M evaluations
    eval_src_350m = ROOT / "results/main_350m"
    eval_dest_350m = stage_root / "350m/evaluations"
    eval_dest_350m.mkdir(parents=True, exist_ok=True)
    for fname in ("comparison.md", "comparison.json", "comparison.csv",
                  "recall_cells.csv", "needle_recall.png", "needle_recall.pdf", "comparison_summary.json"):
        src = eval_src_350m / fname
        if src.exists():
            shutil.copy2(src, eval_dest_350m / fname)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", help="Hugging Face repo id, e.g. tasmaikeni13/algebraic-transformer")
    p.add_argument("--token", help="Hugging Face access token")
    p.add_argument("--private", action="store_true", help="Make repository private")
    args = p.parse_args()

    import huggingface_hub
    from huggingface_hub import HfApi

    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"Authenticated as Hugging Face user: {username}", flush=True)
    except Exception as e:
        print(f"Authentication error: {e}", file=sys.stderr)
        print("Please log in with 'hf auth login' or pass --token / set HF_TOKEN environment variable.", file=sys.stderr)
        sys.exit(1)

    repo_id = args.repo_id or f"{username}/algebraic-transformer"
    print(f"Target repository: {repo_id}", flush=True)

    print("Creating/verifying repository on Hugging Face Hub...", flush=True)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=args.private)

    staging_dir = ROOT / "hf_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Staging repository files (using zero-cost hard links)...", flush=True)
        stage_repository(staging_dir)

        print("Uploading README.md to Hub...", flush=True)
        api.upload_file(
            path_or_fileobj=str(staging_dir / "README.md"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add Hugging Face model card README",
        )

        print("Uploading 125M models folder to Hub...", flush=True)
        api.upload_folder(
            folder_path=str(staging_dir / "125M"),
            path_in_repo="125M",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Push 125M models",
        )

        print(f"Successfully uploaded: https://huggingface.co/{repo_id}", flush=True)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


if __name__ == "__main__":
    main()
