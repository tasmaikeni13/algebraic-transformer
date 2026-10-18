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

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
pipeline_tag: text-generation
tags:
- language-modeling
- algebraic-transformer
- jax
- pallas
- tpu
---

# Algebraic Transformer: Pure Algebraic vs Standard Transformer (125M)

This repository hosts the official checkpoints, evaluation logs, and reproducibility suite for the **Algebraic Transformer** (125M parameter scale).

Matched 125M Standard and Pure Algebraic Transformer models were trained from scratch on 2.5 Billion tokens of FineWeb-Edu across 5 independent seeds each (10 complete runs total) on Google Cloud TPU v4-32 pods.

> **Status Note**: 125M scale models are published in this release under `/125M`. Larger scales (350M and 1B parameters) are actively scheduled and will be added to this repository upon completion.

---

## Architecture Overview

The **Algebraic Transformer** investigates whether foundational sequence models can be formulated entirely with algebraic operations—avoiding transcendental functions (such as exp, softmax, GELU, SiLU, and transcendentals in rotary embeddings) in both forward and backward passes:

| Component | Standard Transformer | Pure Algebraic Transformer |
|---|---|---|
| **Attention Activation** | Softmax | Algebraic asinh with power contraction |
| **Attention Sink** | None | Learned sink token parameter |
| **Position Representation**| RoPE | Rational Cayley geometric rotations |
| **Feed-Forward Network** | SwiGLU | ALU-GLU (rational algebraic non-linearity) |
| **Normalization** | RMSNorm | Rational scale normalization |
| **Output Head** | Softmax Cross-Entropy | Normalized algebraic cross-entropy |

---

## 125M Comparison Results (5 Seeds: 42, 43, 44, 45, 46)

### 1. Language Modeling & Training Telemetry (2.5B tokens)

| Metric | Standard 125M | Algebraic 125M | Paired Delta | Significance |
|---|---|---|---|---|
| **Held-Out NLL (nats/token)** | 3.2001 ± 0.0037 | **3.1582 ± 0.0044** | **-0.0419 ± 0.0051** | p < 0.001 (Algebraic wins) |
| **Held-Out Perplexity (exp NLL)**| 24.53 ± 0.09 | **23.53 ± 0.10** | **-1.00 ± 0.12** | Statistically significant |
| **Training Time per Run** | 1605.80 ± 0.55 s | 1735.65 ± 0.49 s | +8.0% runtime | Modest overhead on TPU |
| **Throughput (tok/s)** | 1,570,799 ± 423 | 1,455,789 ± 482 | 92.7% of standard | Hardware-efficient Pallas kernels |

### 2. Public Zero-Shot Evaluation (Accuracy %)

Evaluated on official test/validation splits:

| Benchmark | Standard 125M | Algebraic 125M | Paired Delta (pp) | Winner |
|---|---|---|---|:---:|
| **ARC-Challenge** | 24.10% ± 0.92 | **24.83% ± 0.74** | **+0.73 ± 0.64** | **Algebraic** |
| **ARC-Easy** | 43.16% ± 0.59 | **43.48% ± 0.97** | **+0.32 ± 0.79** | **Algebraic** |
| **HellaSwag** | 28.43% ± 0.09 | **28.87% ± 0.45** | **+0.44 ± 0.43** | **Algebraic** |
| **WinoGrande** | 51.32% ± 0.68 | **52.19% ± 1.29** | **+0.87 ± 1.68** | **Algebraic** |
| **BoolQ** | 60.33% ± 1.60 | **60.60% ± 1.83** | **+0.27 ± 1.16** | **Algebraic** |
| **LAMBADA** | 17.15% ± 0.76 | **17.45% ± 0.65** | **+0.30 ± 1.17** | **Algebraic** |
| **PIQA** | 61.75% ± 0.56 | **61.94% ± 0.38** | **+0.18 ± 0.81** | **Algebraic** |
| **OpenBookQA** | **27.12% ± 1.71** | 27.08% ± 1.68 | -0.04 ± 2.15 | Neutral / Tie |

### 3. Associative & Long-Range Needle Recall

| Task & Metric | Standard 125M | Algebraic 125M | Paired Delta (pp) |
|---|---|---|---|
| **Associative Recall** (Exact Match) | 1.88% ± 1.60 | **2.66% ± 1.62** | **+0.78 ± 2.31** |
| **Associative Recall** (4-Choice Recognition) | 77.97% ± 4.16 | **85.72% ± 2.24** | **+7.75 ± 3.10** |
| **Latest Value Update** (Exact Match) | 3.00% ± 2.67 | **3.31% ± 0.98** | **+0.31 ± 2.97** |
| **Latest Value Update** (4-Choice Recognition) | 46.88% ± 4.74 | **51.12% ± 7.07** | **+4.24 ± 3.65** |
| **Multi-Needle** (4-Choice Recognition) | 96.69% ± 2.40 | **97.62% ± 3.18** | **+0.93 ± 1.12** |
| **Passkey Needle** (Exact Match) | **3.03% ± 1.74** | 2.12% ± 2.07 | -0.91 ± 2.82 |
| **Passkey Needle** (4-Choice Recognition) | **95.06% ± 0.99** | 94.09% ± 2.32 | -0.97 ± 1.84 |
| **Absent Evidence Controls** (Unsupported Hits)| 0.00% ± 0.00 | 0.00% ± 0.00 | 0.00 (No hallucination) |

---

## Repository Structure

```
algebraic-transformer/
├── README.md                      # Model card and evaluation report
└── 125M/
    ├── standard_s42/              # Standard Transformer checkpoint, seed 42
    │   ├── params.npz             # Model weights (bfloat16)
    │   ├── meta.json              # Model configuration & training arguments
    │   ├── result.json            # Final training loss & evaluation metrics
    │   └── log.jsonl              # Step-by-step training telemetry
    ├── standard_s43/ ... s46/     # Seeds 43, 44, 45, 46
    ├── algebraic_s42/             # Pure Algebraic Transformer checkpoint, seed 42
    │   ├── params.npz
    │   ├── meta.json
    │   ├── result.json
    │   └── log.jsonl
    ├── algebraic_s43/ ... s46/    # Seeds 43, 44, 45, 46
    └── evaluations/
        ├── comparison.md          # Comprehensive evaluation markdown table
        ├── comparison.json        # Raw statistical aggregates
        ├── comparison.csv         # Metric table in CSV format
        ├── recall_cells.csv       # Depth/context cell metrics for needle recall
        ├── needle_recall.png      # Retrieval heatmap visualization
        └── needle_recall.pdf      # Publication-ready vector plot
```

## Citation & Code
Repository and training implementation: [github.com/tasmaikeni13/algebraic-transformer](https://github.com/tasmaikeni13/algebraic-transformer)
"""

MODELS = [
    "standard_s42", "standard_s43", "standard_s44", "standard_s45", "standard_s46",
    "algebraic_s42", "algebraic_s43", "algebraic_s44", "algebraic_s45", "algebraic_s46",
]


def stage_repository(stage_root):
    stage_root = Path(stage_root)
    (stage_root / "125M").mkdir(parents=True, exist_ok=True)
    readme_path = stage_root / "README.md"
    readme_path.write_text(MODEL_CARD_TEMPLATE)

    # Stage 125M models with hard links
    for model_name in MODELS:
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

    # Stage evaluations
    eval_src = ROOT / "results/main_125m"
    eval_dest = stage_root / "125M/evaluations"
    eval_dest.mkdir(parents=True, exist_ok=True)
    for fname in ("comparison.md", "comparison.json", "comparison.csv",
                  "recall_cells.csv", "needle_recall.png", "needle_recall.pdf", "comparison_summary.json"):
        src = eval_src / fname
        if src.exists():
            shutil.copy2(src, eval_dest / fname)


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
