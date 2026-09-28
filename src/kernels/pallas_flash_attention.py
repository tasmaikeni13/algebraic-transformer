"""Tiled softmax attention with a maintained TPU kernel and an XLA fallback.

The fallback tracks a row maximum and denominator across key tiles, then
recomputes attention weights in its analytical backward pass."""

from functools import partial
import math
from typing import Optional, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp


def tiled_flash_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    causal: bool = True,
    block_q: int = 128,
    block_k: int = 128,
    valid_seq_len: Optional[int] = None,
    return_stats: bool = False,
) -> Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:
    """Standard FlashAttention-2 tiled forward pass with online softmax rescaling.

    Args:
        q: Query tensor of shape (B, H, T, D).
        k: Key tensor of shape (B, H, T, D).
        v: Value tensor of shape (B, H, T, D).
        causal: Autoregressive causal masking flag.
        block_q: Query tile sequence length.
        block_k: Key/value tile sequence length.
        return_stats: If True, returns (output, running_max, running_sum) for backward.

    Returns:
        Attention output tensor of shape (B, H, T, D).
    """
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same four-dimensional shape")
    if block_q <= 0 or block_k <= 0:
        raise ValueError("block_q and block_k must be positive")

    batch_size, num_heads, seq_len, head_dim = q.shape
    if seq_len % block_q != 0 or seq_len % block_k != 0:
        raise ValueError("sequence length must be divisible by both tile dimensions")
    if valid_seq_len is not None and not 0 < valid_seq_len <= seq_len:
        raise ValueError("valid_seq_len must be in [1, seq_len]")
    scale = float(1.0 / math.sqrt(head_dim))
    num_q_blocks = seq_len // block_q
    num_k_blocks = seq_len // block_k

    accum_dtype = jnp.float64 if q.dtype == jnp.float64 else jnp.float32

    def _query_block_step(qi_idx):
        q_block = lax.dynamic_slice_in_dim(q, qi_idx * block_q, block_q, axis=2)

        def _key_block_step(kj_idx, acc):
            o_acc, m_prev, l_prev = acc
            k_block = lax.dynamic_slice_in_dim(k, kj_idx * block_k, block_k, axis=2)
            v_block = lax.dynamic_slice_in_dim(v, kj_idx * block_k, block_k, axis=2)

            # Raw score tile: S_ij = (Q_i @ K_j^T) * scale
            s_ij = jnp.matmul(q_block.astype(accum_dtype), jnp.swapaxes(k_block.astype(accum_dtype), -1, -2)) * scale

            row_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 0) + qi_idx * block_q
            col_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 1) + kj_idx * block_k

            if valid_seq_len is not None:
                s_ij = jnp.where(col_ids < valid_seq_len, s_ij, -1e9)

            # Position-based causal masking also covers unequal tile sizes.
            if causal:
                mask = col_ids <= row_ids
                s_ij = jnp.where(mask[None, None, :, :], s_ij, -1e9)

            # Online softmax tracking (FlashAttention-2)
            m_curr = jnp.max(s_ij, axis=-1, keepdims=True)
            m_new = jnp.maximum(m_prev, m_curr)

            # Exponential rescaling factor: alpha = exp(m_prev - m_new)
            alpha = jnp.exp(m_prev - m_new)
            p_ij = jnp.exp(s_ij - m_new)

            # Rescale previous accumulator and add current block
            o_new = o_acc * alpha + jnp.matmul(p_ij.astype(v_block.dtype), v_block.astype(accum_dtype))
            l_new = l_prev * alpha + jnp.sum(p_ij, axis=-1, keepdims=True)

            return (o_new, m_new, l_new)

        init_o = jnp.zeros_like(q_block, dtype=accum_dtype)
        init_m = jnp.full((batch_size, num_heads, block_q, 1), -1e9, dtype=accum_dtype)
        init_l = jnp.zeros((batch_size, num_heads, block_q, 1), dtype=accum_dtype)

        upper_k = (
            min(num_k_blocks, ((qi_idx + 1) * block_q + block_k - 1) // block_k)
            if causal
            else num_k_blocks
        )
        final_o, final_m, final_l = lax.fori_loop(0, upper_k, _key_block_step, (init_o, init_m, init_l))
        out_block = final_o / jnp.maximum(final_l, 1e-12)

        return out_block.astype(q.dtype), final_m, final_l

    results = [_query_block_step(i) for i in range(num_q_blocks)]
    out_blocks = [r[0] for r in results]
    m_blocks = [r[1] for r in results]
    l_blocks = [r[2] for r in results]

    out = jnp.concatenate(out_blocks, axis=2)
    m = jnp.concatenate(m_blocks, axis=2)
    l = jnp.concatenate(l_blocks, axis=2)

    if return_stats:
        return out, m, l
    return out


def tiled_flash_attention_backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    m: jax.Array,
    l: jax.Array,
    g_out: jax.Array,
    causal: bool = True,
    block_q: int = 128,
    block_k: int = 128,
    valid_seq_len: Optional[int] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Standard FlashAttention-2 tiled analytical backward pass."""
    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = float(1.0 / math.sqrt(head_dim))
    num_q_blocks = seq_len // block_q
    num_k_blocks = seq_len // block_k
    accum_dtype = jnp.float64 if q.dtype == jnp.float64 else jnp.float32

    # Row contraction D_i = sum_d g_out,id * out_id
    D = jnp.sum(g_out.astype(accum_dtype) * out.astype(accum_dtype), axis=-1, keepdims=True)

    dq = jnp.zeros_like(q, dtype=accum_dtype)
    dk = jnp.zeros_like(k, dtype=accum_dtype)
    dv = jnp.zeros_like(v, dtype=accum_dtype)

    for i in range(num_q_blocks):
        q_i = lax.dynamic_slice_in_dim(q, i * block_q, block_q, axis=2).astype(accum_dtype)
        g_out_i = lax.dynamic_slice_in_dim(g_out, i * block_q, block_q, axis=2).astype(accum_dtype)
        m_i = lax.dynamic_slice_in_dim(m, i * block_q, block_q, axis=2)
        l_i = lax.dynamic_slice_in_dim(l, i * block_q, block_q, axis=2)
        D_i = lax.dynamic_slice_in_dim(D, i * block_q, block_q, axis=2)

        dq_i = jnp.zeros_like(q_i)
        max_k = (
            min(num_k_blocks, ((i + 1) * block_q + block_k - 1) // block_k)
            if causal
            else num_k_blocks
        )

        for j in range(max_k):
            k_j = lax.dynamic_slice_in_dim(k, j * block_k, block_k, axis=2).astype(accum_dtype)
            v_j = lax.dynamic_slice_in_dim(v, j * block_k, block_k, axis=2).astype(accum_dtype)

            # Recompute softmax weights: P_ij = exp(S_ij - m_i) / l_i
            s_ij = jnp.matmul(q_i, jnp.swapaxes(k_j, -1, -2)) * scale
            row_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 0) + i * block_q
            col_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 1) + j * block_k
            if valid_seq_len is not None:
                s_ij = jnp.where(col_ids < valid_seq_len, s_ij, -1e9)
            if causal:
                cmask = col_ids <= row_ids
                s_ij = jnp.where(cmask[None, None, :, :], s_ij, -1e9)

            p_ij = jnp.exp(s_ij - m_i) / jnp.maximum(l_i, 1e-12)
            if valid_seq_len is not None:
                p_ij = jnp.where(col_ids < valid_seq_len, p_ij, 0.0)
            if causal:
                p_ij = jnp.where(cmask[None, None, :, :], p_ij, 0.0)

            # Gradient with respect to scores: dS_ij = P_ij * (g_out_i @ V_j^T - D_i)
            g_w = jnp.matmul(g_out_i, jnp.swapaxes(v_j, -1, -2))
            ds_ij = p_ij * (g_w - D_i)

            dq_i = dq_i + jnp.matmul(ds_ij, k_j) * scale
            dk_j = jnp.matmul(jnp.swapaxes(ds_ij, -1, -2), q_i) * scale
            dv_j = jnp.matmul(jnp.swapaxes(p_ij, -1, -2), g_out_i)

            curr_dk_j = lax.dynamic_slice_in_dim(dk, j * block_k, block_k, axis=2)
            curr_dv_j = lax.dynamic_slice_in_dim(dv, j * block_k, block_k, axis=2)
            dk = lax.dynamic_update_slice_in_dim(dk, curr_dk_j + dk_j, j * block_k, axis=2)
            dv = lax.dynamic_update_slice_in_dim(dv, curr_dv_j + dv_j, j * block_k, axis=2)

        curr_dq_i = lax.dynamic_slice_in_dim(dq, i * block_q, block_q, axis=2)
        dq = lax.dynamic_update_slice_in_dim(dq, curr_dq_i + dq_i, i * block_q, axis=2)

    return dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)


def _flash_attn_fwd_vjp(q, k, v, causal, block_q, block_k, valid_seq_len):
    out, m, l = tiled_flash_attention_forward(
        q,
        k,
        v,
        causal=causal,
        block_q=block_q,
        block_k=block_k,
        valid_seq_len=valid_seq_len,
        return_stats=True,
    )
    return out, (q, k, v, out, m, l)


def _flash_attn_bwd_vjp(causal, block_q, block_k, valid_seq_len, res, g_out):
    q, k, v, out, m, l = res
    dq, dk, dv = tiled_flash_attention_backward(
        q,
        k,
        v,
        out,
        m,
        l,
        g_out,
        causal=causal,
        block_q=block_q,
        block_k=block_k,
        valid_seq_len=valid_seq_len,
    )
    return dq, dk, dv


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _standard_flash_attention_tiled(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    causal: bool = True,
    block_q: int = 128,
    block_k: int = 128,
    valid_seq_len: Optional[int] = None,
) -> jax.Array:
    return tiled_flash_attention_forward(
        q,
        k,
        v,
        causal=causal,
        block_q=block_q,
        block_k=block_k,
        valid_seq_len=valid_seq_len,
    )


_standard_flash_attention_tiled.defvjp(_flash_attn_fwd_vjp, _flash_attn_bwd_vjp)


def standard_flash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    causal: bool = True,
    block_q: int = 128,
    block_k: int = 128,
) -> jax.Array:
    """Hardware-routed FlashAttention with padding and an analytical CPU VJP.

    TPU execution uses JAX's maintained Pallas FlashAttention kernel. Other
    platforms use the tiled implementation in this module, which is also the
    deterministic reference fallback used by the unit tests.
    """
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same four-dimensional shape")
    if block_q <= 0 or block_k <= 0:
        raise ValueError("block_q and block_k must be positive")

    seq_len = q.shape[-2]
    tile_multiple = math.lcm(block_q, block_k)
    remainder = seq_len % tile_multiple

    # Use the maintained TPU kernel and its custom derivative when available.
    try:
        is_tpu = jax.devices()[0].platform == "tpu"
    except Exception:
        is_tpu = False
    if is_tpu and remainder == 0:
        from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention

        return flash_attention(
            q,
            k,
            v,
            causal=causal,
            sm_scale=float(1.0 / math.sqrt(q.shape[-1])),
        )

    if remainder:
        pad_len = tile_multiple - remainder
        padding = [(0, 0), (0, 0), (0, pad_len), (0, 0)]
        q = jnp.pad(q, padding)
        k = jnp.pad(k, padding)
        v = jnp.pad(v, padding)
        valid_seq_len = seq_len
    else:
        valid_seq_len = None

    out = _standard_flash_attention_tiled(
        q,
        k,
        v,
        causal=causal,
        block_q=block_q,
        block_k=block_k,
        valid_seq_len=valid_seq_len,
    )
    return out[:, :, :seq_len, :]
