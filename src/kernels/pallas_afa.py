"""Tiled octic attention for TPU and a portable XLA fallback.

The forward pass accumulates numerator and denominator across valid key tiles.
The custom VJP computes query, key, and value gradients while preserving the
causal mask and FP32 accumulation for reduced-precision inputs."""

import functools
import math
from typing import Optional, Tuple, Union

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _vmu_octic_kernel(s: jax.Array) -> jax.Array:
    """Evaluates octic algebraic kernel rho(s)^8 on VMU vector registers.

    Base map: rho(s) = s + sqrt(1 + s^2) = s + (1 + s^2) * rsqrt(1 + s^2).
    Exact 3-stage squaring hierarchy: rho -> rho^2 -> rho^4 -> rho^8.
    Strictly zero transcendentals (0 exp, 0 log, 0 trig).
    """
    s_sq = s * s
    one = jnp.array(1.0, dtype=s.dtype)
    rad = one + s_sq
    r = lax.rsqrt(rad)
    u = s * r
    # Avoid catastrophic cancellation in s + sqrt(1+s^2) for negative scores.
    denominator = jnp.where(s < 0, one - u, one)
    rho = jnp.where(s < 0, r / denominator, s + rad * r)
    k2 = rho * rho
    k4 = k2 * k2
    return k4 * k4


def afa_kernel(
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    d_ref,
    o_acc_ref,
    d_acc_ref,
    *,
    scale: float,
    sink_omega: float,
    causal: bool = False,
    num_k_blocks: int = 1,
    block_q: int = 128,
    block_k: int = 128,
    head_dim: int = 128,
    matmul_precision: lax.Precision = lax.Precision.DEFAULT,
    valid_seq_len: Optional[int] = None,
):
    """Pure Additive Algebraic FlashAttention Tile Kernel for TPU v4.

    Executed inside TPU v4 TensorCore VMEM across 128x128 MXU systolic arrays.
    Accumulates unnormalized numerators and denominators additively without
    any running-max subtraction or transcendental rescaling.
    """
    b_idx = pl.program_id(0)
    h_idx = pl.program_id(1)
    q_blk_idx = pl.program_id(2)
    k_blk_idx = pl.program_id(3)

    # 1. Initialize scratch accumulators on the first key tile (k_blk_idx == 0)
    @pl.when(k_blk_idx == 0)
    def _init():
        o_acc_ref[...] = jnp.zeros_like(o_acc_ref)
        d_acc_ref[...] = jnp.zeros_like(d_acc_ref)

    # 2. In causal attention, skip key tiles that start strictly after the last
    # query position in this query tile.  Comparing tile indices directly only
    # works when block_q == block_k.
    should_compute = (
        k_blk_idx * block_k < (q_blk_idx + 1) * block_q
        if causal
        else True
    )

    @pl.when(should_compute)
    def _compute_step():
        q_tile = q_ref[0, 0]  # (block_q, head_dim)
        k_tile = k_ref[0, 0]  # (block_k, head_dim)
        v_tile = v_ref[0, 0]  # (block_k, head_dim)

        # 2a. Systolic Matrix Multiplication on MXU: S = (Q @ K^T) * scale
        s_bc = lax.dot_general(
            q_tile,
            k_tile,
            (((1,), (1,)), ((), ())),
            precision=matmul_precision,
            preferred_element_type=jnp.float32,
        ) * scale

        # 2b. Three-Stage Squaring Kernel on VMU: P = rho(S)^8
        p_bc = _vmu_octic_kernel(s_bc)

        # 2c. Padding mask: zero out weights for dummy keys outside valid_seq_len
        col_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 1) + k_blk_idx * block_k
        if valid_seq_len is not None:
            p_bc = jnp.where(col_ids < valid_seq_len, p_bc, 0.0)

        # 2d. Position-based causal masking also covers unequal tile sizes.
        if causal:
            row_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 0) + q_blk_idx * block_q
            causal_mask = col_ids <= row_ids
            p_bc = jnp.where(causal_mask, p_bc, 0.0)

        # 2e. Pure Additive Tile Accumulation:
        v_matmul_prec = (
            lax.Precision.DEFAULT
            if v_tile.dtype in (jnp.bfloat16, jnp.float16)
            else matmul_precision
        )
        o_acc_ref[...] = o_acc_ref[...] + lax.dot(
            p_bc.astype(v_tile.dtype),
            v_tile,
            precision=v_matmul_prec,
            preferred_element_type=jnp.float32,
        )

        d_acc_ref[...] = d_acc_ref[...] + jnp.sum(p_bc, axis=1)[:, None]

    # 3. Write out accumulated sums on the final key tile (k_blk_idx == num_k_blocks - 1)
    @pl.when(k_blk_idx == (num_k_blocks - 1))
    def _finalize():
        if head_dim <= 128:
            denominator = d_acc_ref[:, :head_dim]
        else:
            denominator = pltpu.repeat(d_acc_ref[...], head_dim // 128, axis=1)
        d_total = denominator + sink_omega
        normalized = o_acc_ref[...] / d_total
        o_ref[0, 0] = normalized.astype(o_ref.dtype)
        d_ref[0, 0] = d_total[:, :1].astype(d_ref.dtype)


def pallas_afa_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sink_omega: float = 0.5,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
    valid_seq_len: Optional[int] = None,
    interpret: Optional[bool] = None,
    return_denominator: bool = False,
) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
    """Full forward call orchestrating Pallas TPU execution."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must all have shape (batch, heads, sequence, dimension)")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, and v shapes must match, got {q.shape}, {k.shape}, and {v.shape}")
    if not (0.0 <= sink_omega < float("inf")):
        raise ValueError("sink_omega must be finite and nonnegative")
    if block_q <= 0 or block_k <= 0:
        raise ValueError("block_q and block_k must be positive")

    batch_size, num_heads, seq_len, head_dim = q.shape
    if head_dim <= 0:
        raise ValueError("head dimension must be positive")
    if head_dim > 128 and head_dim % 128 != 0:
        raise ValueError("head dimensions above 128 must be divisible by 128 for TPU layout")
    if seq_len % block_q != 0 or seq_len % block_k != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by block_q ({block_q}) and block_k ({block_k})"
        )
    if valid_seq_len is not None and not 0 < valid_seq_len <= seq_len:
        raise ValueError("valid_seq_len must be in [1, seq_len]")

    num_q_blocks = seq_len // block_q
    num_k_blocks = seq_len // block_k
    scale = float(1.0 / math.sqrt(head_dim))

    grid = (batch_size, num_heads, num_q_blocks, num_k_blocks)

    in_specs = [
        pl.BlockSpec((1, 1, block_q, head_dim), lambda b, h, i, j: (b, h, i, 0)),
        pl.BlockSpec((1, 1, block_k, head_dim), lambda b, h, i, j: (b, h, j, 0)),
        pl.BlockSpec((1, 1, block_k, head_dim), lambda b, h, i, j: (b, h, j, 0)),
    ]
    out_specs = (
        pl.BlockSpec(
            (1, 1, block_q, head_dim), lambda b, h, i, j: (b, h, i, 0)
        ),
        pl.BlockSpec(
            (1, 1, block_q, 1), lambda b, h, i, j: (b, h, i, 0)
        ),
    )

    accum_dtype = jnp.float64 if q.dtype == jnp.float64 else jnp.float32

    scratch_shapes = [
        pltpu.VMEM((block_q, head_dim), accum_dtype),
        pltpu.VMEM((block_q, 128), accum_dtype),
    ]

    kernel_fn = functools.partial(
        afa_kernel,
        scale=scale,
        sink_omega=float(sink_omega),
        causal=causal,
        num_k_blocks=num_k_blocks,
        block_q=block_q,
        block_k=block_k,
        head_dim=head_dim,
        matmul_precision=(
            lax.Precision.DEFAULT
            if q.dtype in (jnp.bfloat16, jnp.float16)
            else lax.Precision.HIGHEST
        ),
        valid_seq_len=valid_seq_len,
    )

    if interpret is None:
        try:
            platform = jax.devices()[0].platform
            interpret = (platform != "tpu")
        except Exception:
            interpret = True

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
    )

    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
    )

    out_o, out_d = pl.pallas_call(
        kernel_fn,
        out_shape=(
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(q.shape[:-1] + (1,), accum_dtype),
        ),
        grid_spec=grid_spec,
        compiler_params=compiler_params,
        interpret=interpret,
    )(q, k, v)

    if return_denominator:
        return out_o, out_d
    return out_o


def exact_afa_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sink_omega: float = 0.5,
    causal: bool = False,
) -> jax.Array:
    """Exact un-tiled mathematical reference for Algebraic FlashAttention."""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same four-dimensional shape")
    if not (0.0 <= sink_omega < float("inf")):
        raise ValueError("sink_omega must be finite and nonnegative")
    head_dim = q.shape[-1]
    seq_len = q.shape[-2]
    scale = float(1.0 / math.sqrt(head_dim))

    q_scaled = (q * scale).astype(q.dtype)
    s = jnp.matmul(q_scaled, jnp.swapaxes(k, -1, -2))

    p = _vmu_octic_kernel(s)

    if causal:
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        p = jnp.where(mask[None, None, :, :], p, 0.0)

    p_v = p.astype(v.dtype)
    o = jnp.matmul(p_v, v)
    d = jnp.sum(p_v, axis=-1, keepdims=True)

    inv_denom = lax.reciprocal(d + jnp.array(sink_omega, dtype=d.dtype))
    return (o * inv_denom.astype(o.dtype)).astype(q.dtype)


def exact_afa_with_denominator(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sink_omega: float = 0.5,
    causal: bool = False,
) -> Tuple[jax.Array, jax.Array]:
    """Computes exact output and denominator D = S + Omega for analytical backward."""
    head_dim = q.shape[-1]
    seq_len = q.shape[-2]
    scale = float(1.0 / math.sqrt(head_dim))

    q_scaled = (q * scale).astype(q.dtype)
    s = jnp.matmul(q_scaled, jnp.swapaxes(k, -1, -2))

    p = _vmu_octic_kernel(s)

    if causal:
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        p = jnp.where(mask[None, None, :, :], p, 0.0)

    p_v = p.astype(v.dtype)
    o = jnp.matmul(p_v, v)
    d = jnp.sum(p_v, axis=-1, keepdims=True)
    d_total = d + jnp.array(sink_omega, dtype=d.dtype)
    inv_denom = lax.reciprocal(d_total)
    out = (o * inv_denom.astype(o.dtype)).astype(q.dtype)
    return out, d_total


def tiled_afa_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sink_omega: float = 0.5,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
    valid_seq_len: Optional[int] = None,
    return_denominator: bool = False,
) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
    """XLA-tiled additive accumulation implementation of Algebraic FlashAttention."""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same four-dimensional shape")
    if not (0.0 <= sink_omega < float("inf")):
        raise ValueError("sink_omega must be finite and nonnegative")
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
            o_acc, d_acc = acc
            k_block = lax.dynamic_slice_in_dim(k, kj_idx * block_k, block_k, axis=2)
            v_block = lax.dynamic_slice_in_dim(v, kj_idx * block_k, block_k, axis=2)

            s_bc = jnp.matmul(q_block, jnp.swapaxes(k_block, -1, -2)) * scale
            p_bc = _vmu_octic_kernel(s_bc)

            col_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 1) + kj_idx * block_k
            if valid_seq_len is not None:
                p_bc = jnp.where(col_ids < valid_seq_len, p_bc, 0.0)

            if causal:
                row_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 0) + qi_idx * block_q
                mask = col_ids <= row_ids
                p_bc = jnp.where(mask[None, None, :, :], p_bc, 0.0)

            o_step = jnp.matmul(p_bc.astype(v_block.dtype), v_block).astype(accum_dtype)
            d_step = jnp.sum(p_bc, axis=-1, keepdims=True).astype(accum_dtype)

            return (o_acc + o_step, d_acc + d_step)

        init_o = (q_block * 0.0).astype(accum_dtype)
        init_d = (q_block[..., :1] * 0.0).astype(accum_dtype)

        upper_k = (
            min(num_k_blocks, ((qi_idx + 1) * block_q + block_k - 1) // block_k)
            if causal
            else num_k_blocks
        )
        final_o, final_d = lax.fori_loop(0, upper_k, _key_block_step, (init_o, init_d))
        d_total = final_d + sink_omega
        out_block = (final_o / d_total.astype(final_o.dtype)).astype(q.dtype)
        return out_block, d_total.astype(accum_dtype)

    results = [_query_block_step(i) for i in range(num_q_blocks)]
    out_tensors = [r[0] for r in results]
    d_tensors = [r[1] for r in results]
    out = jnp.concatenate(out_tensors, axis=2)
    d_total = jnp.concatenate(d_tensors, axis=2)

    if return_denominator:
        return out, d_total
    return out


def tiled_afa_backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    d_total: jax.Array,
    g_out: jax.Array,
    sink_omega: float = 0.5,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
    valid_seq_len: Optional[int] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute the attention gradient over key and query tiles.

    For each visible score, computes:
        dL / ds_ij = (8 * scale) * r_ij * w_ij * (g_w,ij - sum_d g_out,id * out_id)
    where:
        E_i = sum_d g_out,id * out_id is precomputed once per query row,
        w_ij = rho(s_ij)^8 / D_i is recomputed inside the tile,
        g_w,ij = g_out,i @ v_j^T,
    and accumulates dQ, dK, and dV without storing the attention matrix.
    """
    del sink_omega
    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = float(1.0 / math.sqrt(head_dim))
    num_q_blocks = seq_len // block_q
    num_k_blocks = seq_len // block_k
    accum_dtype = jnp.float64 if q.dtype == jnp.float64 else jnp.float32
    reduced_input = q.dtype in (jnp.bfloat16, jnp.float16)
    dot_dtype = q.dtype if reduced_input else accum_dtype
    dot_precision = lax.Precision.DEFAULT if reduced_input else lax.Precision.HIGHEST

    def accum_matmul(lhs, rhs):
        """Use MXU input precision while retaining FP32 dot accumulation."""
        return jnp.matmul(
            lhs.astype(dot_dtype),
            rhs.astype(dot_dtype),
            precision=dot_precision,
            preferred_element_type=accum_dtype,
        )

    # Precompute scalar row contraction E_i = sum_d g_out,id * out_id once
    E = jnp.sum(g_out.astype(accum_dtype) * out.astype(accum_dtype), axis=-1, keepdims=True)

    dq = jnp.zeros_like(q, dtype=accum_dtype)
    dk = jnp.zeros_like(k, dtype=accum_dtype)
    dv = jnp.zeros_like(v, dtype=accum_dtype)

    for i in range(num_q_blocks):
        q_i = lax.dynamic_slice_in_dim(q, i * block_q, block_q, axis=2)
        g_out_i = lax.dynamic_slice_in_dim(g_out, i * block_q, block_q, axis=2)
        d_i = lax.dynamic_slice_in_dim(d_total, i * block_q, block_q, axis=2)
        E_i = lax.dynamic_slice_in_dim(E, i * block_q, block_q, axis=2)

        dq_i = jnp.zeros_like(q_i, dtype=accum_dtype)
        max_k = (
            min(num_k_blocks, ((i + 1) * block_q + block_k - 1) // block_k)
            if causal
            else num_k_blocks
        )

        # Static loop bounds exclude key tiles wholly beyond the causal mask.
        for j in range(max_k):
            k_j = lax.dynamic_slice_in_dim(k, j * block_k, block_k, axis=2)
            v_j = lax.dynamic_slice_in_dim(v, j * block_k, block_k, axis=2)

            # Recompute raw scores and rational kernel in registers
            s_ij = accum_matmul(q_i, jnp.swapaxes(k_j, -1, -2)) * scale
            s_sq = s_ij * s_ij
            rad = 1.0 + s_sq
            r = lax.rsqrt(rad)
            u = s_ij * r
            denom = jnp.where(s_ij < 0, 1.0 - u, 1.0)
            rho = jnp.where(s_ij < 0, r / denom, s_ij + rad * r)
            k2 = rho * rho
            k4 = k2 * k2
            w_unnorm = k4 * k4

            col_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 1) + j * block_k
            if valid_seq_len is not None:
                w_unnorm = jnp.where(col_ids < valid_seq_len, w_unnorm, 0.0)

            if causal:
                row_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 0) + i * block_q
                cmask = col_ids <= row_ids
                w_unnorm = jnp.where(cmask[None, None, :, :], w_unnorm, 0.0)

            w_ij = w_unnorm / d_i
            g_w_ij = accum_matmul(g_out_i, jnp.swapaxes(v_j, -1, -2))

            ds_ij = (8.0 * scale) * r * w_ij * (g_w_ij - E_i)
            if causal:
                ds_ij = jnp.where(cmask[None, None, :, :], ds_ij, 0.0)

            # Stream updates into dQ, dK, dV
            dq_i = dq_i + accum_matmul(ds_ij, k_j)

            dk_j = accum_matmul(jnp.swapaxes(ds_ij, -1, -2), q_i)
            dv_j = accum_matmul(jnp.swapaxes(w_ij, -1, -2), g_out_i)

            # Accumulate into dk and dv slices
            curr_dk_j = lax.dynamic_slice_in_dim(dk, j * block_k, block_k, axis=2)
            curr_dv_j = lax.dynamic_slice_in_dim(dv, j * block_k, block_k, axis=2)
            dk = lax.dynamic_update_slice_in_dim(dk, curr_dk_j + dk_j, j * block_k, axis=2)
            dv = lax.dynamic_update_slice_in_dim(dv, curr_dv_j + dv_j, j * block_k, axis=2)

        curr_dq_i = lax.dynamic_slice_in_dim(dq, i * block_q, block_q, axis=2)
        dq = lax.dynamic_update_slice_in_dim(dq, curr_dq_i + dq_i, i * block_q, axis=2)

    return dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)


def _running_on_tpu() -> bool:
    try:
        return jax.devices()[0].platform == "tpu"
    except Exception:
        return False


def _afa_fwd_vjp(q, k, v, sink_omega, causal, block_q, block_k, valid_seq_len):
    if _running_on_tpu():
        out, d_total = pallas_afa_forward(
            q,
            k,
            v,
            sink_omega=sink_omega,
            causal=causal,
            block_q=block_q,
            block_k=block_k,
            valid_seq_len=valid_seq_len,
            interpret=False,
            return_denominator=True,
        )
    else:
        out, d_total = tiled_afa_forward(
            q,
            k,
            v,
            sink_omega=sink_omega,
            causal=causal,
            block_q=block_q,
            block_k=block_k,
            valid_seq_len=valid_seq_len,
            return_denominator=True,
        )
    return out, (q, k, v, out, d_total)


def _afa_bwd_vjp(sink_omega, causal, block_q, block_k, valid_seq_len, res, g_out):
    q, k, v, out, d_total = res
    dq, dk, dv = tiled_afa_backward(
        q,
        k,
        v,
        out,
        d_total,
        g_out,
        sink_omega=sink_omega,
        causal=causal,
        block_q=block_q,
        block_k=block_k,
        valid_seq_len=valid_seq_len,
    )
    return dq, dk, dv


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def pallas_afa(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sink_omega: float = 0.5,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
    valid_seq_len: Optional[int] = None,
) -> jax.Array:
    """Hardware-fused Octic Algebraic FlashAttention with analytical single-pass VJP."""
    if _running_on_tpu():
        return pallas_afa_forward(
            q,
            k,
            v,
            sink_omega=sink_omega,
            causal=causal,
            block_q=block_q,
            block_k=block_k,
            valid_seq_len=valid_seq_len,
            interpret=False,
        )
    return tiled_afa_forward(
        q,
        k,
        v,
        sink_omega=sink_omega,
        causal=causal,
        block_q=block_q,
        block_k=block_k,
        valid_seq_len=valid_seq_len,
    )


pallas_afa.defvjp(_afa_fwd_vjp, _afa_bwd_vjp)


def distributed_ring_afa(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sink_omega: float = 0.5,
    causal: bool = False,
    axis_name: str = "devices",
    num_devices: int = 16,
) -> jax.Array:
    """Lock-Free Distributed Ring Attention across TPU chips over ICI."""
    device_idx = lax.axis_index(axis_name)
    head_dim = q.shape[-1]
    scale = float(1.0 / math.sqrt(head_dim))
    shard_len = q.shape[-2]
    ring_perm = [(i, (i + 1) % num_devices) for i in range(num_devices)]

    def _ring_step(hop, state):
        o_acc, d_acc, curr_k, curr_v = state

        s = jnp.matmul(q, jnp.swapaxes(curr_k, -1, -2)) * scale
        p = _vmu_octic_kernel(s)

        if causal:
            kv_device = (device_idx - hop) % num_devices
            row_ids = lax.broadcasted_iota(jnp.int32, (shard_len, shard_len), 0) + device_idx * shard_len
            col_ids = lax.broadcasted_iota(jnp.int32, (shard_len, shard_len), 1) + kv_device * shard_len
            mask = col_ids <= row_ids
            tile_active = kv_device <= device_idx
            p = jnp.where(tile_active & mask[None, None, :, :], p, 0.0)

        o_acc = o_acc + jnp.matmul(p.astype(v.dtype), curr_v)
        d_acc = d_acc + jnp.sum(p, axis=-1, keepdims=True)

        next_k = lax.ppermute(curr_k, axis_name=axis_name, perm=ring_perm)
        next_v = lax.ppermute(curr_v, axis_name=axis_name, perm=ring_perm)

        return (o_acc, d_acc, next_k, next_v)

    init_o = (q * 0.0).astype(jnp.float32)
    init_d = (q[..., :1] * 0.0).astype(jnp.float32)

    final_o, final_d, _, _ = lax.fori_loop(0, num_devices, _ring_step, (init_o, init_d, k, v))

    y = final_o / (final_d + sink_omega).astype(final_o.dtype)
    return y.astype(q.dtype)


def sharded_pallas_afa(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mesh: Mesh,
    sink_omega: float = 0.5,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
) -> jax.Array:
    """SPMD Distributed Megacore Sharded AFA on Mesh(data, fsdp, model).

    Shards:
      - Batch dimension across 'data'
      - Head dimension across 'fsdp'
      - Sequence / model parallel across 'model'
    """
    in_sharding = NamedSharding(mesh, P("data", "fsdp", None, None))
    q_s = jax.device_put(q, in_sharding)
    k_s = jax.device_put(k, in_sharding)
    v_s = jax.device_put(v, in_sharding)

    def _call(q_arr, k_arr, v_arr):
        return pallas_afa(
            q_arr, k_arr, v_arr, sink_omega=sink_omega, causal=causal, block_q=block_q, block_k=block_k
        )

    return jax.jit(_call)(q_s, k_s, v_s)


def algebraic_flash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sink_omega: float = 0.5,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
) -> jax.Array:
    """Unified Algebraic FlashAttention interface with hardware autotuning."""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same four-dimensional shape")
    if block_q <= 0 or block_k <= 0:
        raise ValueError("block_q and block_k must be positive")

    orig_seq_len = q.shape[-2]
    tile_multiple = math.lcm(block_q, block_k)

    rem = orig_seq_len % tile_multiple
    if rem != 0:
        pad_len = tile_multiple - rem
        pad_config = [(0, 0), (0, 0), (0, pad_len), (0, 0)]
        q_pad = jnp.pad(q, pad_config)
        k_pad = jnp.pad(k, pad_config)
        v_pad = jnp.pad(v, pad_config)
        valid_len = orig_seq_len
    else:
        q_pad, k_pad, v_pad = q, k, v
        valid_len = None

    out = pallas_afa(
        q_pad,
        k_pad,
        v_pad,
        sink_omega=sink_omega,
        causal=causal,
        block_q=block_q,
        block_k=block_k,
        valid_seq_len=valid_len,
    )

    if rem != 0:
        out = out[:, :, :orig_seq_len, :]

    return out
