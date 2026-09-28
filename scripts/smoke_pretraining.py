#!/usr/bin/env python3
"""Fast CPU smoke test for Pretraining study contracts; never produces study evidence."""

import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.tuning_experiments import training_steps_for_budget
from scripts.pretraining_experiments import PRETRAINING_TOKEN_BUDGET, summarize_pretraining
from scripts.prepare_eval_benchmarks import (
    normalize_arc,
    normalize_hellaswag,
    normalize_lambada,
    normalize_piqa,
)
from scripts.run_hparam_sweep import load_training_checkpoint, save_training_checkpoint
from scripts.smoke_tuning import _smoke_architecture


def _fixture_records():
    rows = []
    for architecture, base in (("algebraic", 60.0), ("baseline", 61.0)):
        for seed, delta in zip((42, 43, 44), (-0.1, 0.0, 0.1)):
            rows.append({
                "architecture": architecture,
                "seed": seed,
                "total_tokens": 2_500_853_760,
                "validation_loss": 4.1 + delta / 100,
                "validation_perplexity": base + delta,
                "nan_or_inf_count": 0,
                "loss_spike_count": 0,
                "peak_gradient_norm": 1.0,
                "normalization_second_moment_min": 0.99,
                "normalization_second_moment_max": 1.0,
            })
    return rows


def main() -> int:
    steps, actual = training_steps_for_budget(PRETRAINING_TOKEN_BUDGET, 512 * 2048)
    summary = summarize_pretraining(_fixture_records())
    if not summary["pretraining_safety_passed"]:
        raise RuntimeError("Pretraining study aggregation fixture failed")

    with tempfile.TemporaryDirectory(prefix="pretraining-smoke-") as directory:
        path = Path(directory) / "step-000001.pkl"
        payload = {"completed_steps": 1, "params": {"w": np.ones((2, 2))}}
        save_training_checkpoint(path, payload)
        loaded = load_training_checkpoint(path)
        if loaded["completed_steps"] != 1:
            raise RuntimeError("checkpoint round trip failed")

    normalized = [
        normalize_arc({
            "question": "Q?", "choices": {"label": ["A", "B"], "text": ["x", "y"]},
            "answerKey": "B",
        }),
        normalize_hellaswag({"ctx": "c", "endings": [" a", " b"], "label": "0"}),
        normalize_piqa({"goal": "g", "sol1": "a", "sol2": "b", "label": 1}),
        normalize_lambada({"text": "the final word"}),
    ]
    result = {
        "status": "PASS",
        "scope": "smoke_only",
        "budget_steps": steps,
        "actual_tokens": actual,
        "summary": summary,
        "normalizers_checked": len(normalized),
        "algebraic_loss": _smoke_architecture("algebraic"),
        "baseline_loss": _smoke_architecture("baseline"),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
