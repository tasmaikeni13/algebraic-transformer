#!/usr/bin/env python3
"""Upload completed 125M comparison models and evaluation suite to Hugging Face Hub.

Pushes:
  1. The 9 trained parameter checkpoints (params.npz, meta.json, result.json).
  2. The full post-training evaluation suite (zero-shot benchmarks, recall matrix, predictions).
  3. The final comparison tables, CSVs, and needle recall heatmap figures.
  4. An auto-generated Model Card (README.md) with benchmark summaries.
"""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def generate_model_card(out_path: Path, comparison_md: Path = None):
    header = """---
license: apache-2.0
tags:
- language-modeling
- transformer
- algebraic-transformer
- jax
- pallas
- tpu
---

# SOTA Algebra: 125M / 2.5B-Token Comparison Checkpoints

This repository hosts the official weights, evaluation traces, and benchmark comparisons for the **second-generation Algebraic Transformer (v2d)** compared against the **Standard Transformer** and the **Minimal-Algebra Hybrid**.

### Evaluated Arms (3 seeds each: 42, 43, 44):
1. **Standard Transformer**: RMSNorm, SwiGLU, RoPE, Softmax FlashAttention, Cross-Entropy.
2. **Hybrid Transformer**: Standard backbone + QK-normalization + Algebraic Likelihood Head $\\rho(z/2)^8$.
3. **Pure Algebraic (v2d)**: Exact log-free and exp-free training path using asinh attention $\\rho(s/8)^{16}$ with learned sink, ALU-GLU ($\\beta=1/2$), Cayley rotations, learned RMS gains, and algebraic maximum-likelihood head.

### Benchmark Coverage:
* **Zero-Shot Public Benchmarks**: ARC-Easy, ARC-Challenge, HellaSwag, PIQA, OpenBookQA, WinoGrande, BoolQ, LAMBADA.
* **Long-Context Retrieval**: Needle-in-a-Haystack Passkey, Associative Recall, Multi-Needle, Latest-Value Recall (256 to 2,048 tokens).

---

"""
    content = header
    if comparison_md and comparison_md.is_file():
        content += comparison_md.read_text()
    out_path.write_text(content)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="tasmaikeni13/algebraic-transformer")
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs/main")
    p.add_argument("--results-root", type=Path, default=ROOT / "results/main")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    hf_cli = Path.home() / ".local/bin/huggingface-cli"
    if not hf_cli.exists():
        hf_cli = Path("huggingface-cli")

    # Generate model card from results
    comparison_md = args.results_root / "comparison.md"
    card_path = ROOT / "HF_MODEL_CARD.md"
    generate_model_card(card_path, comparison_md)

    staging_dir = ROOT / "hf_upload_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Copy model card as README.md
    (staging_dir / "README.md").write_text(card_path.read_text())

    # Stage weights and metadata for each completed run
    for run_dir in sorted(args.runs_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        params_file = run_dir / "params.npz"
        meta_file = run_dir / "meta.json"
        res_file = run_dir / "result.json"
        if params_file.exists():
            target_sub = staging_dir / "checkpoints" / run_dir.name
            target_sub.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(["cp", str(params_file), str(target_sub / "params.npz")])
            if meta_file.exists():
                subprocess.check_call(["cp", str(meta_file), str(target_sub / "meta.json")])
            if res_file.exists():
                subprocess.check_call(["cp", str(res_file), str(target_sub / "result.json")])

    # Stage evaluation results
    eval_target = staging_dir / "evaluations"
    eval_target.mkdir(parents=True, exist_ok=True)
    if args.results_root.exists():
        subprocess.check_call(["cp", "-r", f"{args.results_root}/.", str(eval_target)])

    try:
        print(f"Staged files in {staging_dir}. Uploading to {args.repo_id}...")
        upload_cmd = [
            str(hf_cli), "upload", args.repo_id, str(staging_dir), ".",
            "--commit-message", "Upload 125M comparison checkpoints and benchmark results",
        ]
        if args.private:
            upload_cmd.append("--private")

        subprocess.check_call(upload_cmd)
        print(f"Upload complete: https://huggingface.co/{args.repo_id}")
    finally:
        import shutil
        shutil.rmtree(staging_dir, ignore_errors=True)
        print(f"Cleaned staging directory {staging_dir}.")


if __name__ == "__main__":
    main()
