"""Tuning study hyperparameter sweep experimental logic and training configurations.

Implements the equal-budget search space and step functions for 125M scale models
on FineWeb-Edu across Seeds 42, 43, 44.
"""

from dataclasses import dataclass, asdict
from functools import partial
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from src.model import AlgebraicTransformerLM, ModelConfig, count_parameters
from src.baseline import StandardTransformerLM, BaselineConfig
from src.optimizer import algebraic_adamw, ards_schedule
from src.attention import build_cayley_rotary_matrix
from src.baseline import _build_standard_rope
from src.dataset import evaluate_perplexity_and_nll_tokens
from scripts.audit_primitives import source_audit

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class HparamConfig:
    """Hyperparameter bundle for a candidate run."""
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    sink_omega: float = 0.5
    gamma: float = 2.0
    min_lr: float = 1e-5
    schedule: str = "ards"  # "ards" or "cosine"
    max_grad_norm: float = 1.0


def load_hparam_candidates(path: Path) -> Dict[str, List[Tuple[str, HparamConfig]]]:
    """Load and validate the preregistered Tuning study candidate matrix."""
    payload = json.loads(Path(path).read_text())
    expected_architectures = {"algebraic", "baseline"}
    if set(payload) != expected_architectures:
        raise ValueError(f"candidate file must contain exactly {sorted(expected_architectures)}")

    result: Dict[str, List[Tuple[str, HparamConfig]]] = {}
    for architecture in sorted(expected_architectures):
        rows = payload[architecture]
        if not isinstance(rows, list) or len(rows) < 2:
            raise ValueError(f"{architecture} requires at least two candidates for a sweep")
        parsed = []
        seen = set()
        for row in rows:
            row = dict(row)
            name = row.pop("name", None)
            if not isinstance(name, str) or not name or name in seen:
                raise ValueError(f"{architecture} candidate names must be nonempty and unique")
            seen.add(name)
            config = HparamConfig(**row)
            expected_schedule = "ards" if architecture == "algebraic" else "cosine"
            if config.schedule != expected_schedule:
                raise ValueError(
                    f"{architecture}/{name} must use {expected_schedule}, got {config.schedule}"
                )
            if not 1e-4 <= config.learning_rate <= 2e-3:
                raise ValueError(f"{architecture}/{name} learning_rate is outside [1e-4, 2e-3]")
            if config.warmup_steps <= 0:
                raise ValueError(f"{architecture}/{name} warmup_steps must be positive")
            if not 0.0 < config.weight_decay <= 0.15:
                raise ValueError(f"{architecture}/{name} weight_decay must be in (0, 0.15]")
            if not 0.0 < config.beta1 < 1.0 or not 0.0 < config.beta2 < 1.0:
                raise ValueError(f"{architecture}/{name} beta values must be in (0, 1)")
            if config.sink_omega < 0.0 or config.gamma <= 0.0:
                raise ValueError(f"{architecture}/{name} sink_omega/gamma are invalid")
            if not 0.0 < config.min_lr <= config.learning_rate:
                raise ValueError(f"{architecture}/{name} min_lr must be in (0, learning_rate]")
            if config.max_grad_norm <= 0.0:
                raise ValueError(f"{architecture}/{name} max_grad_norm must be positive")
            parsed.append((name, config))
        result[architecture] = parsed
    return result


def training_steps_for_budget(token_budget: int, tokens_per_step: int) -> Tuple[int, int]:
    """Return full-batch steps and actual tokens, never undershooting the budget."""
    if token_budget <= 0 or tokens_per_step <= 0:
        raise ValueError("token_budget and tokens_per_step must be positive")
    steps = math.ceil(token_budget / tokens_per_step)
    return steps, steps * tokens_per_step


def select_best_candidate(
    records: List[Dict[str, Any]],
    architecture: str,
    seeds: List[int],
    max_seed_std_pct: float = 2.0,
) -> Tuple[HparamConfig, List[Dict[str, Any]], float]:
    """Select the lowest mean validation-loss candidate satisfying all gates."""
    expected_seeds = set(seeds)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        if record["architecture"] == architecture:
            grouped.setdefault(record["candidate"], []).append(record)

    eligible = []
    for name, candidate_records in grouped.items():
        if (
            len(candidate_records) != len(expected_seeds)
            or {r["seed"] for r in candidate_records} != expected_seeds
        ):
            continue
        losses = np.asarray([r["validation_loss"] for r in candidate_records], dtype=np.float64)
        mean_loss = float(np.mean(losses))
        std_pct = float(np.std(losses) / mean_loss * 100.0) if mean_loss > 0 else float("inf")
        stable = (
            np.all(np.isfinite(losses))
            and std_pct < max_seed_std_pct
            and all(r["nan_or_inf_count"] == 0 for r in candidate_records)
            and all(r["loss_spike_count"] == 0 for r in candidate_records)
            and all(r["peak_gradient_norm"] <= 5.0 for r in candidate_records)
            and all(r["token_budget_satisfied"] for r in candidate_records)
            and (
                architecture != "algebraic"
                or all(
                    r["normalization_second_moment_min"] >= 0.8
                    and r["normalization_second_moment_max"] <= 1.3
                    for r in candidate_records
                )
            )
        )
        if stable:
            eligible.append((mean_loss, name, candidate_records, std_pct))

    if not eligible:
        raise RuntimeError(f"no eligible {architecture} candidate satisfied the Tuning study gates")
    _, _, winning_records, std_pct = min(eligible, key=lambda row: (row[0], row[1]))
    return HparamConfig(**winning_records[0]["hparams"]), winning_records, std_pct


def get_125m_algebraic_config(sink_omega: float = 0.5, gamma: float = 2.0) -> ModelConfig:
    """Returns 125M parameter ModelConfig for AlgebraicTransformerLM."""
    return ModelConfig(
        vocab_size=50257,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=2048,
        max_seq_len=2048,
        eps=1e-5,
        eps_vocab=100.0,
        sink_omega=sink_omega,
        gamma=gamma,
        tie_embeddings=True,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        remat=True,
    )


def get_125m_baseline_config() -> BaselineConfig:
    """Returns 125M parameter BaselineConfig for StandardTransformerLM."""
    return BaselineConfig(
        vocab_size=50257,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=2048,
        max_seq_len=2048,
        eps=1e-5,
        tie_embeddings=True,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        remat=True,
    )


def _clip_grad_norm_algebraic(grads: Any, max_norm: float = 1.0) -> Tuple[Any, jax.Array]:
    """Clips gradients using pure hardware rsqrt without transcendental functions."""
    leaves = jax.tree_util.tree_leaves(grads)
    sum_sq = sum(jnp.sum(jnp.square(g.astype(jnp.float32))) for g in leaves)
    safe_sq = jnp.maximum(sum_sq, 1e-12)
    inv_norm = lax.rsqrt(safe_sq)
    norm = sum_sq * inv_norm

    clip_factor = jnp.minimum(1.0, max_norm * inv_norm)
    clipped_grads = jax.tree_util.tree_map(lambda g: (g * clip_factor).astype(g.dtype), grads)
    clipped_norm = norm * clip_factor
    return clipped_grads, clipped_norm


def create_cosine_schedule(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-5,
) -> Callable[[int], jax.Array]:
    """Standard Cosine Annealing learning rate schedule for baseline model."""
    def schedule(step):
        step_f = jnp.asarray(step, dtype=jnp.float32)
        warmup_f = float(warmup_steps)
        total_f = float(total_steps)

        warmup_factor = jnp.minimum(1.0, step_f / jnp.maximum(1.0, warmup_f))
        progress = jnp.maximum(0.0, (step_f - warmup_f) / jnp.maximum(1.0, total_f - warmup_f))
        cosine_decay = 0.5 * (1.0 + jnp.cos(math.pi * jnp.minimum(1.0, progress)))
        lr = min_lr + (learning_rate - min_lr) * cosine_decay
        return jnp.where(step_f < warmup_f, learning_rate * warmup_factor, lr)

    return schedule


def train_step_algebraic_fn(
    model: AlgebraicTransformerLM,
    optimizer_tx,
    rotary_params,
    max_grad_norm: float = 1.0,
    accum_steps: int = 16,
    data_axis_names: Optional[Tuple[str, ...]] = None,
):
    """Factory returning a JIT-compilable single training step for AlgebraicTransformerLM with gradient accumulation."""
    def step_fn(params, opt_state, tokens, targets):
        init_grads = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)

        def micro_step(carry, mb):
            accum_loss, accum_grads = carry
            tok, tgt = mb
            def loss_fn(p):
                loss_val, aux = model.loss(p, tok, tgt, rotary_params=rotary_params)
                return loss_val, aux
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            new_loss = accum_loss + loss / accum_steps
            new_grads = jax.tree_util.tree_map(
                lambda acc, g: acc + (g / accum_steps).astype(acc.dtype),
                accum_grads,
                grads,
            )
            return (new_loss, new_grads), None

        (total_loss, accum_grads), _ = lax.scan(
            micro_step, (jnp.array(0.0, dtype=jnp.float32), init_grads), (tokens, targets)
        )

        if data_axis_names:
            total_loss = lax.pmean(total_loss, axis_name=data_axis_names)
            accum_grads = jax.tree_util.tree_map(
                lambda grad: lax.pmean(grad, axis_name=data_axis_names),
                accum_grads,
            )
        clipped_grads, grad_norm = _clip_grad_norm_algebraic(accum_grads, max_grad_norm)
        updates, new_opt_state = optimizer_tx.update(clipped_grads, opt_state, params)
        new_params = jax.tree_util.tree_map(lambda p, u: (p + u).astype(p.dtype), params, updates)

        metrics = {
            "loss": total_loss,
            "grad_norm": grad_norm,
            "is_finite": jnp.isfinite(total_loss) & jnp.isfinite(grad_norm),
        }
        return new_params, new_opt_state, metrics

    return step_fn


def train_step_baseline_fn(
    model: StandardTransformerLM,
    optimizer_tx,
    cos_angles,
    sin_angles,
    max_grad_norm: float = 1.0,
    accum_steps: int = 16,
    data_axis_names: Optional[Tuple[str, ...]] = None,
):
    """Factory returning a JIT-compilable single training step for StandardTransformerLM with gradient accumulation."""
    def step_fn(params, opt_state, tokens, targets):
        init_grads = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)

        def micro_step(carry, mb):
            accum_loss, accum_grads = carry
            tok, tgt = mb
            def loss_fn(p):
                loss_val, aux = model.loss(p, tok, tgt, cos_angles=cos_angles, sin_angles=sin_angles)
                return loss_val, aux
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            new_loss = accum_loss + loss / accum_steps
            new_grads = jax.tree_util.tree_map(
                lambda acc, g: acc + (g / accum_steps).astype(acc.dtype),
                accum_grads,
                grads,
            )
            return (new_loss, new_grads), None

        (total_loss, accum_grads), _ = lax.scan(
            micro_step, (jnp.array(0.0, dtype=jnp.float32), init_grads), (tokens, targets)
        )

        if data_axis_names:
            total_loss = lax.pmean(total_loss, axis_name=data_axis_names)
            accum_grads = jax.tree_util.tree_map(
                lambda grad: lax.pmean(grad, axis_name=data_axis_names),
                accum_grads,
            )
        clipped_grads, grad_norm = _clip_grad_norm_algebraic(accum_grads, max_grad_norm)
        updates, new_opt_state = optimizer_tx.update(clipped_grads, opt_state, params)
        new_params = jax.tree_util.tree_map(lambda p, u: (p + u).astype(p.dtype), params, updates)

        metrics = {
            "loss": total_loss,
            "grad_norm": grad_norm,
            "is_finite": jnp.isfinite(total_loss) & jnp.isfinite(grad_norm),
        }
        return new_params, new_opt_state, metrics

    return step_fn


def evaluate_perplexity_fast(
    model,
    params,
    valid_tokens: np.ndarray,
    seq_len: int = 2048,
    batch_size: int = 4,
    num_eval_batches: int = 10,
    is_algebraic: bool = True,
    rotary_or_angles: Optional[Any] = None,
) -> Tuple[float, float]:
    """Fast bounded-memory validation perplexity on FineWeb-Edu."""
    if rotary_or_angles is None:
        rotary_or_angles = (
            build_cayley_rotary_matrix(model.head_dim, seq_len, dtype=model.config.dtype)
            if is_algebraic
            else _build_standard_rope(model.head_dim, seq_len)
        )
    ppl, avg_nll = evaluate_perplexity_and_nll_tokens(
        model,
        params,
        valid_tokens,
        seq_len=seq_len,
        batch_size=batch_size,
        max_eval_batches=num_eval_batches,
        is_algebraic=is_algebraic,
        rotary_or_angles=rotary_or_angles,
    )
    return ppl, avg_nll
