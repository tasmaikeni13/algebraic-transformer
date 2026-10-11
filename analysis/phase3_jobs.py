#!/usr/bin/env python3
"""Phase 3 job lists for scripts/run_slice.py: 60M models, 1B tokens, 16 chips.

Protocol (identical for both arms): FineWeb-Edu-1B, GPT-2 tokens, context
2048, global batch 256 sequences (524,288 tokens), ceil(1e9 / 524,288) =
1,908 AdamW steps (1,000,341,504 tokens), cosine schedule to 10% with 3%
warmup, b1 0.9, b2 0.95, weight decay 0.1, clip 1, one seed. Validation on
the first 8M tokens of the held-out split, every 200 steps and at the end.
Each arm gets the same three-point learning-rate sweep; the best run per arm
is the Phase 3 comparison.

  python analysis/phase3_jobs.py sweep > jobs.json
"""

import json
import sys

COMMON = ["--size", "60m", "--seq-len", "2048", "--batch", "256", "--micro", "16",
          "--tokens", "1e9", "--warmup", "0.03", "--b2", "0.95", "--wd", "0.1",
          "--schedule", "cosine", "--min-lr-ratio", "0.1", "--eval-every", "200",
          "--eval-tokens", "8e6", "--eval-batch", "8", "--log-every", "50",
          "--set", "attn_impl=flash"]


def job(name, preset, lr, seed=0, sets=()):
    args = ["--preset", preset, *COMMON, "--lr", str(lr), "--seed", str(seed)]
    for s in sets:
        args += ["--set", s]
    return {"name": name, "args": args}


def sweep(lrs=(1.5e-3, 3e-3, 6e-3)):
    jobs = []
    for lr in lrs:
        jobs.append(job(f"std_lr{lr:g}", "standard", lr))
        jobs.append(job(f"alg_lr{lr:g}", "algebraic", lr))
    return jobs


def hybrid():
    """The hybrid arm with the same learning-rate grid (3e-3 was run as the
    control ``ctrl_qk_head_lr0.003``, identical configuration)."""
    return [job(f"hyb_lr{lr:g}", "hybrid", lr) for lr in (1.5e-3, 6e-3)]


if __name__ == "__main__":
    print(json.dumps(globals()[sys.argv[1]](), indent=1))
