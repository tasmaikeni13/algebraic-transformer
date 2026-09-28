"""Tiled linear projection and octic algebraic scoring loss.

The vocabulary is processed in chunks so the full token-by-vocabulary logit
matrix need not be stored. Forward and backward passes share the same
normalization and target-token accounting."""

from functools import partial
import math
from typing import Any, Dict, Optional, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp


def _vmu_octic_kernel(s: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluates octic algebraic kernel and powers on VMU registers.

    Returns:
        (k8, rho7, rho, r)
    """
    s_sq = s * s
    one = jnp.array(1.0, dtype=s.dtype)
    rad = one + s_sq
    r = lax.rsqrt(rad)
    u = s * r
    denominator = jnp.where(s < 0, one - u, one)
    rho = jnp.where(s < 0, r / denominator, s + rad * r)
    k2 = rho * rho
    k4 = k2 * k2
    k8 = k4 * k4
    rho7 = rho * k2 * k4
    return k8, rho7, rho, r


def fused_linear_oace_forward(
    h: jax.Array,
    w_vocab: jax.Array,
    targets: jax.Array,
    eps_vocab: float = 100.0,
    gamma: float = 2.0,
    chunk_size: int = 4096,
) -> Tuple[jax.Array, Tuple[Any, ...]]:
    """Tiled chunked forward evaluation of Linear + OACE loss.

    Args:
        h: Hidden activations of shape (..., d_model).
        w_vocab: Vocabulary projection matrix of shape (d_model, vocab_size).
        targets: Integer target class indices of shape matching h[..., 0].
        eps_vocab: AVN epsilon for vocabulary normalizer.
        gamma: OACE proper score multiplier.
        chunk_size: Tile width along vocabulary dimension.

    Returns:
        (loss, cache) where loss is scalar mean OACE loss and cache holds
        compact scalar reductions needed for analytical backward streaming.
    """
    orig_shape = h.shape
    d_model = orig_shape[-1]
    vocab_size = w_vocab.shape[-1]

    # Flatten batch and sequence into N tokens
    h_flat = h.reshape(-1, d_model)
    targets_flat = targets.reshape(-1)
    N = h_flat.shape[0]

    calc_dtype = jnp.float64 if h.dtype == jnp.float64 else jnp.float32
    inv_V = float(1.0 / vocab_size)
    num_chunks = math.ceil(vocab_size / chunk_size)

    # Exact Gram factorization of ||h W||² = h (W Wᵀ) hᵀ.  Since d_model is
    # far smaller than the number of tokens in a training batch, this replaces
    # one N-by-V projection sweep with a compact d_model-by-d_model product.
    # The custom VJP below still streams the exact logit-space derivative.
    h_calc = h_flat.astype(calc_dtype)
    reduced_weights = w_vocab.dtype in (jnp.bfloat16, jnp.float16)
    gram_input = w_vocab if reduced_weights else w_vocab.astype(calc_dtype)
    vocab_gram = jnp.matmul(
        gram_input,
        gram_input.T,
        precision=lax.Precision.DEFAULT if reduced_weights else lax.Precision.HIGHEST,
        preferred_element_type=calc_dtype,
    )
    ss_total = jnp.sum(
        jnp.matmul(h_calc, vocab_gram) * h_calc,
        axis=-1,
        keepdims=True,
    )

    tau = lax.rsqrt(ss_total * inv_V + eps_vocab)

    # Vocabulary pass: octic partition sums, target rho, and radial reductions
    sum_k8 = jnp.zeros((N, 1), dtype=calc_dtype)
    sum_rho7 = jnp.zeros((N, 1), dtype=calc_dtype)
    sum_r_y_k8 = jnp.zeros((N, 1), dtype=calc_dtype)
    sum_r_y_rho7 = jnp.zeros((N, 1), dtype=calc_dtype)
    rho_target = jnp.zeros((N, 1), dtype=calc_dtype)
    target_r_y = jnp.zeros((N, 1), dtype=calc_dtype)

    for c in range(num_chunks):
        c_start = c * chunk_size
        c_end = min(c_start + chunk_size, vocab_size)
        w_c = w_vocab[:, c_start:c_end].astype(calc_dtype)
        z_c = jnp.matmul(h_flat.astype(calc_dtype), w_c)
        normed_c = z_c * tau

        k8_c, rho7_c, rho_c, r_c = _vmu_octic_kernel(normed_c)
        r_y_c = r_c * normed_c
        sum_k8 = sum_k8 + jnp.sum(k8_c, axis=-1, keepdims=True)
        sum_rho7 = sum_rho7 + jnp.sum(rho7_c, axis=-1, keepdims=True)
        sum_r_y_k8 = sum_r_y_k8 + jnp.sum(r_y_c * k8_c, axis=-1, keepdims=True)
        sum_r_y_rho7 = sum_r_y_rho7 + jnp.sum(
            r_y_c * rho7_c, axis=-1, keepdims=True
        )

        # Target token extraction within active chunk
        in_chunk = (targets_flat >= c_start) & (targets_flat < c_end)
        local_idx = jnp.clip(targets_flat - c_start, 0, (c_end - c_start) - 1)
        local_rho = jnp.take_along_axis(rho_c, local_idx[:, None], axis=-1)
        local_r_y = jnp.take_along_axis(r_y_c, local_idx[:, None], axis=-1)
        rho_target = jnp.where(in_chunk[:, None], local_rho, rho_target)
        target_r_y = jnp.where(in_chunk[:, None], local_r_y, target_r_y)

    # Scalar 3-rsqrt cascade on partition sum
    inv_S = lax.reciprocal(sum_k8)
    S_eighth = lax.rsqrt(lax.rsqrt(lax.rsqrt(inv_S)))
    partition_scale = S_eighth * inv_S
    sum_p78 = partition_scale * sum_rho7
    p_c_inv8 = S_eighth * lax.reciprocal(rho_target)

    # The AVN Jacobian subtracts y * mean(g_y * y).  Expanding the OACE
    # bracket lets us compute that scalar from three forward reductions:
    #   sum(r*y*p^(7/8)) - r_t*y_t*p_t^(-1/8)
    #     - (sum(p^(7/8)) - p_t^(-1/8)) * sum(r*y*p).
    # Caching it here removes a full vocabulary-projection sweep in backward.
    scalar_diff = sum_p78 - p_c_inv8
    radial = (
        partition_scale * sum_r_y_rho7
        - target_r_y * p_c_inv8
        - scalar_diff * inv_S * sum_r_y_k8
    ) * inv_V

    # OACE Loss functional
    loss_per_token = gamma * (
        8.0 * p_c_inv8.squeeze(-1)
        + (8.0 / 7.0) * sum_p78.squeeze(-1)
        - (64.0 / 7.0)
    )
    # Scalar loss reductions stay in the accumulator dtype.  In particular,
    # BF16 activations must not quantize the OACE objective before backprop.
    mean_loss = jnp.mean(loss_per_token, dtype=calc_dtype)

    cache = (
        h_flat,
        w_vocab,
        targets_flat,
        tau,
        inv_S,
        S_eighth,
        sum_p78,
        p_c_inv8,
        radial,
        gamma,
        chunk_size,
        orig_shape,
    )
    return mean_loss, cache


def fused_linear_oace_backward(
    cache: Tuple[Any, ...],
    g: jax.Array,
) -> Tuple[jax.Array, jax.Array, None]:
    """Tiled chunked analytical backward evaluation of Linear + OACE loss.

    Streams gradient accumulations directly into dW and dh without ever
    materializing the full (N, V) logits tensor.
    """
    (
        h_flat,
        w_vocab,
        targets_flat,
        tau,
        inv_S,
        S_eighth,
        sum_p78,
        p_c_inv8,
        radial,
        gamma,
        chunk_size,
        orig_shape,
    ) = cache

    calc_dtype = jnp.float64 if h_flat.dtype == jnp.float64 else jnp.float32
    d_model = h_flat.shape[-1]
    vocab_size = w_vocab.shape[-1]
    N = h_flat.shape[0]
    num_chunks = math.ceil(vocab_size / chunk_size)

    scalar_diff = sum_p78 - p_c_inv8
    upstream_scale = (8.0 * gamma / float(N)) * g.astype(calc_dtype)

    # Single backward pass: stream gradients into dW_c and dh.  The radial
    # reduction was accumulated without materializing logits during forward.
    dh_acc = jnp.zeros((N, d_model), dtype=calc_dtype)
    dw_chunks = []

    for c in range(num_chunks):
        c_start = c * chunk_size
        c_end = min(c_start + chunk_size, vocab_size)
        w_c = w_vocab[:, c_start:c_end].astype(calc_dtype)
        z_c = jnp.matmul(h_flat.astype(calc_dtype), w_c)
        normed_c = z_c * tau

        k8_c, rho7_c, _, r_c = _vmu_octic_kernel(normed_c)
        p_c = k8_c * inv_S
        p_78_c = (S_eighth * inv_S) * rho7_c

        indices = jnp.arange(c_start, c_end)
        one_hot = (targets_flat[:, None] == indices[None, :]).astype(calc_dtype)
        bracket = p_78_c - one_hot * p_c_inv8 - p_c * scalar_diff
        r_bracket = r_c * bracket

        dz_normed = upstream_scale * (r_bracket - radial * normed_c)
        dz_c = tau * dz_normed

        # Stream directly into weight chunk and hidden activations
        dw_c = jnp.matmul(h_flat.astype(calc_dtype).T, dz_c)
        dw_chunks.append(dw_c.astype(w_vocab.dtype))
        dh_acc = dh_acc + jnp.matmul(dz_c, w_c.T)

    dh = dh_acc.reshape(orig_shape).astype(h_flat.dtype)
    dw_vocab = jnp.concatenate(dw_chunks, axis=1)

    return dh, dw_vocab, None


def _fused_linear_oace_fwd_vjp(h, w_vocab, targets, eps_vocab, gamma, chunk_size):
    loss, cache = fused_linear_oace_forward(
        h, w_vocab, targets, eps_vocab=eps_vocab, gamma=gamma, chunk_size=chunk_size
    )
    return loss, cache


def _fused_linear_oace_bwd_vjp(eps_vocab, gamma, chunk_size, cache, g):
    del eps_vocab, gamma, chunk_size
    dh, dw, _ = fused_linear_oace_backward(cache, g)
    return dh, dw, None


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def fused_linear_oace(
    h: jax.Array,
    w_vocab: jax.Array,
    targets: jax.Array,
    eps_vocab: float = 100.0,
    gamma: float = 2.0,
    chunk_size: int = 4096,
) -> jax.Array:
    """Fused Linear + OACE Projection Head with Zero (B, T, V) Allocation.

    Evaluates proper Bregman scoring loss L_{1/8} and analytical gradients
    tiled across vocabulary chunks with strictly zero transcendental functions.
    """
    return fused_linear_oace_forward(
        h, w_vocab, targets, eps_vocab=eps_vocab, gamma=gamma, chunk_size=chunk_size
    )[0]


fused_linear_oace.defvjp(_fused_linear_oace_fwd_vjp, _fused_linear_oace_bwd_vjp)
