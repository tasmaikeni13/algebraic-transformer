"""Pilot study pilot pretraining experiments and training execution logic."""

import ast
from functools import partial
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional, Tuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from src.model import AlgebraicTransformerLM, ModelConfig, count_parameters
from src.baseline import StandardTransformerLM, BaselineConfig
from src.optimizer import algebraic_adamw, ards_schedule
from src.dataset import ShardedTokenLoader, evaluate_perplexity
from scripts.audit_primitives import source_audit, FORBIDDEN

ROOT = Path(__file__).resolve().parents[1]


def audit_pilot_ast() -> Dict[str, Any]:
    """Audit all production files in the algebraic stack for zero transcendentals."""
    files_to_audit = [
        "src/model.py",
        "src/primitives.py",
        "src/attention.py",
        "src/loss.py",
        "src/optimizer.py",
        "src/mesh.py",
        "src/kernels/pallas_afa.py",
        "src/kernels/pallas_oace.py",
    ]
    all_violations = {}
    for rel_path in files_to_audit:
        full_path = ROOT / rel_path
        if full_path.exists():
            violations = source_audit(full_path.read_text())
            if violations:
                all_violations[rel_path] = violations

    return {
        "audited_files": files_to_audit,
        "violations": all_violations,
        "passed": len(all_violations) == 0,
    }


def _clip_grad_norm_algebraic(grads: Any, max_norm: float = 1.0) -> Tuple[Any, jax.Array]:
    """Clips gradients using pure hardware rsqrt without transcendental functions."""
    leaves = jax.tree_util.tree_leaves(grads)
    sum_sq = sum(jnp.sum(jnp.square(g.astype(jnp.float32))) for g in leaves)
    # Hardware rsqrt norm computation: norm = sum_sq * rsqrt(sum_sq)
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
    min_lr: float = 0.0,
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
    data_axis_names: Optional[Tuple[str, ...]] = None,
):
    """Factory returning a JIT-compilable single training step for AlgebraicTransformerLM."""
    def step_fn(params, opt_state, tokens, targets):
        def loss_fn(p):
            loss_val, aux = model.loss(p, tokens, targets, rotary_params=rotary_params)
            return loss_val, aux

        (loss, aux), raw_grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        if data_axis_names:
            loss = lax.pmean(loss, axis_name=data_axis_names)
            raw_grads = jax.tree_util.tree_map(
                lambda grad: lax.pmean(grad, axis_name=data_axis_names),
                raw_grads,
            )
        clipped_grads, grad_norm = _clip_grad_norm_algebraic(raw_grads, max_grad_norm)
        updates, new_opt_state = optimizer_tx.update(clipped_grads, opt_state, params)
        new_params = jax.tree_util.tree_map(lambda p, u: (p + u).astype(p.dtype), params, updates)

        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
            "is_finite": jnp.isfinite(loss) & jnp.isfinite(grad_norm),
        }
        return new_params, new_opt_state, metrics

    return step_fn


def train_step_baseline_fn(
    model: StandardTransformerLM,
    optimizer_tx,
    cos_angles,
    sin_angles,
    max_grad_norm: float = 1.0,
    data_axis_names: Optional[Tuple[str, ...]] = None,
):
    """Factory returning a JIT-compilable single training step for StandardTransformerLM."""
    def step_fn(params, opt_state, tokens, targets):
        def loss_fn(p):
            loss_val, aux = model.loss(p, tokens, targets, cos_angles=cos_angles, sin_angles=sin_angles)
            return loss_val, aux

        (loss, aux), raw_grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        if data_axis_names:
            loss = lax.pmean(loss, axis_name=data_axis_names)
            raw_grads = jax.tree_util.tree_map(
                lambda grad: lax.pmean(grad, axis_name=data_axis_names),
                raw_grads,
            )
        clipped_grads, grad_norm = _clip_grad_norm_algebraic(raw_grads, max_grad_norm)
        updates, new_opt_state = optimizer_tx.update(clipped_grads, opt_state, params)
        new_params = jax.tree_util.tree_map(lambda p, u: (p + u).astype(p.dtype), params, updates)

        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
            "is_finite": jnp.isfinite(loss) & jnp.isfinite(grad_norm),
        }
        return new_params, new_opt_state, metrics

    return step_fn
