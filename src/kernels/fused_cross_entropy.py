"""Vocabulary-tiled linear projection and cross-entropy.

Chunked logits contribute to a stable online log-sum-exp reduction. The
backward pass accumulates weight and hidden-state gradients without retaining
the full token-by-vocabulary logit matrix."""

from functools import partial
import math
from typing import Any, Optional, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp


def standard_fused_linear_ce_forward(
    h: jax.Array,
    w_vocab: jax.Array,
    targets: jax.Array,
    chunk_size: int = 4096,
) -> Tuple[jax.Array, Tuple[Any, ...]]:
    """Tiled forward pass for Standard Linear + Cross-Entropy.

    Args:
        h: Hidden activations of shape (..., d_model).
        w_vocab: Un-embedding matrix of shape (d_model, vocab_size).
        targets: Integer target class indices of shape matching h[..., 0].
        chunk_size: Vocabulary tile size.

    Returns:
        (loss, cache) with zero materialization of the full logit tensor.
    """
    orig_shape = h.shape
    d_model = orig_shape[-1]
    vocab_size = w_vocab.shape[-1]

    h_flat = h.reshape(-1, d_model)
    targets_flat = targets.reshape(-1)
    N = h_flat.shape[0]

    calc_dtype = jnp.float32
    num_chunks = math.ceil(vocab_size / chunk_size)

    # Online log-sum-exp accumulators
    m_global = jnp.full((N, 1), -1e9, dtype=calc_dtype)
    s_global = jnp.zeros((N, 1), dtype=calc_dtype)
    target_logit = jnp.zeros((N, 1), dtype=calc_dtype)

    for c in range(num_chunks):
        c_start = c * chunk_size
        c_end = min(c_start + chunk_size, vocab_size)
        w_c = w_vocab[:, c_start:c_end].astype(calc_dtype)
        z_c = jnp.matmul(h_flat.astype(calc_dtype), w_c)

        m_c = jnp.max(z_c, axis=-1, keepdims=True)
        m_new = jnp.maximum(m_global, m_c)

        alpha = jnp.exp(m_global - m_new)
        p_c = jnp.exp(z_c - m_new)

        s_global = s_global * alpha + jnp.sum(p_c, axis=-1, keepdims=True)
        m_global = m_new

        in_chunk = (targets_flat >= c_start) & (targets_flat < c_end)
        local_idx = jnp.clip(targets_flat - c_start, 0, (c_end - c_start) - 1)
        z_target_local = jnp.take_along_axis(z_c, local_idx[:, None], axis=-1)
        target_logit = jnp.where(in_chunk[:, None], z_target_local, target_logit)

    log_sum_exp = m_global + jnp.log(jnp.maximum(s_global, 1e-12))
    loss_per_token = log_sum_exp - target_logit
    # Keep the scalar reduction in FP32 even when activations are BF16.  Casting
    # the loss to the activation dtype quantizes training telemetry and the
    # upstream scalar cotangent before the analytical backward pass.
    mean_loss = jnp.mean(loss_per_token, dtype=calc_dtype)

    cache = (
        h_flat,
        w_vocab,
        targets_flat,
        m_global,
        s_global,
        chunk_size,
        orig_shape,
    )
    return mean_loss, cache


def standard_fused_linear_ce_backward(
    cache: Tuple[Any, ...],
    g: jax.Array,
) -> Tuple[jax.Array, jax.Array, None]:
    """Tiled analytical backward pass for Standard Linear + Cross-Entropy."""
    h_flat, w_vocab, targets_flat, m_global, s_global, chunk_size, orig_shape = cache

    calc_dtype = jnp.float32
    d_model = h_flat.shape[-1]
    vocab_size = w_vocab.shape[-1]
    N = h_flat.shape[0]
    num_chunks = math.ceil(vocab_size / chunk_size)

    scale_g = g.astype(calc_dtype) / float(N)
    dh_acc = jnp.zeros((N, d_model), dtype=calc_dtype)
    dw_chunks = []

    for c in range(num_chunks):
        c_start = c * chunk_size
        c_end = min(c_start + chunk_size, vocab_size)
        w_c = w_vocab[:, c_start:c_end].astype(calc_dtype)
        z_c = jnp.matmul(h_flat.astype(calc_dtype), w_c)

        # Softmax probabilities: p_c = exp(z_c - m_global) / s_global
        p_c = jnp.exp(z_c - m_global) / jnp.maximum(s_global, 1e-12)

        indices = jnp.arange(c_start, c_end)
        one_hot = (targets_flat[:, None] == indices[None, :]).astype(calc_dtype)

        # dz_c = (p_c - y_c) * g / N
        dz_c = (p_c - one_hot) * scale_g

        dw_c = jnp.matmul(h_flat.astype(calc_dtype).T, dz_c)
        dw_chunks.append(dw_c.astype(w_vocab.dtype))
        dh_acc = dh_acc + jnp.matmul(dz_c, w_c.T)

    dh = dh_acc.reshape(orig_shape).astype(h_flat.dtype)
    dw_vocab = jnp.concatenate(dw_chunks, axis=1)

    return dh, dw_vocab, None


def _std_fused_ce_fwd_vjp(h, w_vocab, targets, chunk_size):
    loss, cache = standard_fused_linear_ce_forward(h, w_vocab, targets, chunk_size=chunk_size)
    return loss, cache


def _std_fused_ce_bwd_vjp(chunk_size, cache, g):
    del chunk_size
    dh, dw, _ = standard_fused_linear_ce_backward(cache, g)
    return dh, dw, None


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def standard_fused_cross_entropy(
    h: jax.Array,
    w_vocab: jax.Array,
    targets: jax.Array,
    chunk_size: int = 4096,
) -> jax.Array:
    """Standard Fused Linear + Cross-Entropy Projection Head with Zero (B, T, V) Allocation."""
    return standard_fused_linear_ce_forward(h, w_vocab, targets, chunk_size=chunk_size)[0]


standard_fused_cross_entropy.defvjp(_std_fused_ce_fwd_vjp, _std_fused_ce_bwd_vjp)
