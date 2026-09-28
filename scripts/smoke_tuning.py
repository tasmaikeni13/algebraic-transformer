#!/usr/bin/env python3
"""Fast CPU smoke test for Tuning study orchestration and both training stacks.

This intentionally uses tiny synthetic tensors. It validates candidate parsing,
token-budget arithmetic, one optimizer update for each architecture, and the
multi-seed selection logic. It does not produce Tuning study evidence.
"""

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

from scripts.tuning_experiments import (
    HparamConfig,
    create_cosine_schedule,
    load_hparam_candidates,
    select_best_candidate,
    train_step_algebraic_fn,
    train_step_baseline_fn,
    training_steps_for_budget,
)
from src.attention import build_cayley_rotary_matrix
from src.baseline import BaselineConfig, StandardTransformerLM, _build_standard_rope
from src.model import AlgebraicTransformerLM, ModelConfig
from src.optimizer import algebraic_adamw, ards_schedule


def _smoke_architecture(architecture: str) -> float:
    key = jax.random.PRNGKey(810 if architecture == "algebraic" else 811)
    tokens = jax.random.randint(key, (2, 1, 16), 0, 128)
    targets = jax.random.randint(jax.random.fold_in(key, 1), (2, 1, 16), 0, 128)

    if architecture == "algebraic":
        model = AlgebraicTransformerLM(ModelConfig(
            vocab_size=128, d_model=32, num_layers=1, num_heads=2,
            d_ff=64, max_seq_len=16, dtype=jnp.float32,
            attention_block_size=16, vocab_chunk_size=64,
        ))
        schedule = ards_schedule(learning_rate=3e-4, warmup_steps=1, decay_steps=2)
        optimizer = algebraic_adamw(schedule, weight_decay=0.01)
        step = train_step_algebraic_fn(
            model, optimizer, build_cayley_rotary_matrix(16, 16), accum_steps=2
        )
    else:
        model = StandardTransformerLM(BaselineConfig(
            vocab_size=128, d_model=32, num_layers=1, num_heads=2,
            d_ff=64, max_seq_len=16, dtype=jnp.float32,
            attention_block_size=16, vocab_chunk_size=64,
        ))
        schedule = create_cosine_schedule(3e-4, warmup_steps=1, total_steps=2)
        optimizer = algebraic_adamw(schedule, weight_decay=0.01)
        cos_angles, sin_angles = _build_standard_rope(16, 16)
        step = train_step_baseline_fn(
            model, optimizer, cos_angles, sin_angles, accum_steps=2
        )

    params = model.init_params(jax.random.fold_in(key, 2))
    opt_state = optimizer.init(params)
    _, _, metrics = jax.jit(step)(params, opt_state, tokens, targets)
    loss = float(jax.device_get(metrics["loss"]))
    if not bool(jax.device_get(metrics["is_finite"])):
        raise RuntimeError(f"{architecture} smoke step produced non-finite metrics")
    return loss


def main() -> int:
    candidates = load_hparam_candidates(ROOT / "configs/tuning_candidates.json")
    steps, actual_tokens = training_steps_for_budget(600_000_000, 512 * 2048)

    template = HparamConfig(learning_rate=6e-4, warmup_steps=28, weight_decay=0.01,
                            beta1=0.9, beta2=0.99)
    records = []
    for candidate, offset in (("stable", 0.0), ("worse", 0.2)):
        for seed, jitter in zip((42, 43, 44), (0.0, 0.005, -0.005)):
            records.append({
                "architecture": "algebraic", "candidate": candidate, "seed": seed,
                "hparams": vars(template), "validation_loss": 4.0 + offset + jitter,
                "nan_or_inf_count": 0, "loss_spike_count": 0,
                "peak_gradient_norm": 1.0, "token_budget_satisfied": True,
                "normalization_second_moment_min": 0.99,
                "normalization_second_moment_max": 1.0,
            })
    _, winner, _ = select_best_candidate(records, "algebraic", [42, 43, 44])

    result = {
        "status": "PASS",
        "scope": "smoke_only",
        "candidate_counts": {name: len(rows) for name, rows in candidates.items()},
        "budget_steps": steps,
        "actual_tokens": actual_tokens,
        "selected_fixture_candidate": winner[0]["candidate"],
        "algebraic_loss": _smoke_architecture("algebraic"),
        "baseline_loss": _smoke_architecture("baseline"),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
