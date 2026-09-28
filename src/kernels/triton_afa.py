"""Triton implementation of tiled octic attention.

The kernel evaluates the rational octic weight in registers and accumulates
attention values across key tiles. Its backward pass recomputes weights for
query, key, and value gradients."""

import math
from typing import Optional, Tuple

import triton
import triton.language as tl

try:
    import torch
except ImportError:
    torch = None


@triton.jit
def _triton_vmu_octic_kernel(s):
    """Evaluates octic algebraic kernel rho(s)^8 on Triton register blocks."""
    s_sq = s * s
    rad = 1.0 + s_sq
    r = tl.rsqrt(rad)
    u = s * r
    denom = tl.where(s < 0.0, 1.0 - u, 1.0)
    rho = tl.where(s < 0.0, r / denom, s + rad * r)
    k2 = rho * rho
    k4 = k2 * k2
    w = k4 * k4
    return w, r


@triton.jit
def _triton_afa_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, D_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_db, stride_dh, stride_dm,
    scale: tl.constexpr,
    sink_omega: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Triton Forward Kernel for Octic Algebraic FlashAttention."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // num_heads
    off_h = off_hz % num_heads

    # Tile offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, head_dim)

    # Base pointers for this batch and head
    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    out_offset = off_b * stride_ob + off_h * stride_oh
    d_offset = off_b * stride_db + off_h * stride_dh

    # Load Q block into SRAM registers: (BLOCK_M, head_dim)
    q_ptrs = Q_ptr + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < seq_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize accumulators in SRAM
    o_acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    d_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Determine key/value loop bounds
    end_n = (start_m + 1) * BLOCK_M if IS_CAUSAL else seq_len

    for start_n in range(0, end_n, BLOCK_N):
        curr_offs_n = start_n + offs_n
        k_mask = curr_offs_n[None, :] < seq_len
        v_mask = curr_offs_n[:, None] < seq_len

        k_ptrs = K_ptr + k_offset + curr_offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        v_ptrs = V_ptr + v_offset + curr_offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Raw scores in SRAM registers: S = (Q @ K^T) * scale
        scores = tl.dot(q, k) * scale

        # Evaluate octic rational kernel: W = rho(S)^8
        w, _ = _triton_vmu_octic_kernel(scores)
        w = tl.where(curr_offs_n[None, :] < seq_len, w, 0.0)

        # Causal mask on the diagonal tiles
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= curr_offs_n[None, :]
            w = tl.where(causal_mask, w, 0.0)

        # Additive tile accumulation: NO running-max subtraction!
        o_acc += tl.dot(w.to(v.dtype), v)
        d_acc += tl.sum(w, axis=1)

    # Final rational normalization in registers: Out = O / (D + Omega)
    d_total = d_acc + sink_omega
    out = o_acc / d_total[:, None]

    # Write normalized output and scalar denominator back to HBM
    out_ptrs = Out_ptr + out_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=q_mask)

    d_ptrs = D_ptr + d_offset + offs_m * stride_dm
    tl.store(d_ptrs, d_total.to(D_ptr.dtype.element_ty), mask=(offs_m < seq_len))


@triton.jit
def _triton_afa_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, D_ptr, GOut_ptr,
    DQ_ptr, DK_ptr, DV_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_db, stride_dh, stride_dm,
    stride_gob, stride_goh, stride_gom, stride_gok,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_dvb, stride_dvh, stride_dvn, stride_dvk,
    scale: tl.constexpr,
    sink_omega: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Single-pass analytical Triton backward kernel for Octic AFA."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // num_heads
    off_h = off_hz % num_heads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, head_dim)

    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    out_offset = off_b * stride_ob + off_h * stride_oh
    d_offset = off_b * stride_db + off_h * stride_dh
    gout_offset = off_b * stride_gob + off_h * stride_goh

    dq_offset = off_b * stride_dqb + off_h * stride_dqh
    dk_offset = off_b * stride_dkb + off_h * stride_dkh
    dv_offset = off_b * stride_dvb + off_h * stride_dvh

    q_mask = offs_m[:, None] < seq_len
    q_ptrs = Q_ptr + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    gout_ptrs = GOut_ptr + gout_offset + offs_m[:, None] * stride_gom + offs_d[None, :] * stride_gok
    gout = tl.load(gout_ptrs, mask=q_mask, other=0.0)

    out_ptrs = Out_ptr + out_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    out = tl.load(out_ptrs, mask=q_mask, other=0.0)

    d_ptrs = D_ptr + d_offset + offs_m * stride_dm
    d_total = tl.load(d_ptrs, mask=(offs_m < seq_len), other=1.0)

    # Precompute scalar row contraction E_i = sum_d g_out,id * out_id once in registers
    E_i = tl.sum(gout * out, axis=1)

    dq_acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    end_n = (start_m + 1) * BLOCK_M if IS_CAUSAL else seq_len

    for start_n in range(0, end_n, BLOCK_N):
        curr_offs_n = start_n + offs_n
        k_mask = curr_offs_n[:, None] < seq_len

        k_ptrs = K_ptr + k_offset + curr_offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        v_ptrs = V_ptr + v_offset + curr_offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # Recompute scores in SRAM: S = Q @ K^T * scale
        scores = tl.dot(q, tl.trans(k)) * scale
        w_unnorm, r = _triton_vmu_octic_kernel(scores)
        w_unnorm = tl.where(curr_offs_n[None, :] < seq_len, w_unnorm, 0.0)

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= curr_offs_n[None, :]
            w_unnorm = tl.where(causal_mask, w_unnorm, 0.0)

        w_ij = w_unnorm / d_total[:, None]
        g_w = tl.dot(gout, tl.trans(v))

        # Analytical FlashAttention cotangent: ds_ij = (8 * scale) * r * w * (g_w - E)
        ds_ij = (8.0 * scale) * r * w_ij * (g_w - E_i[:, None])
        if IS_CAUSAL:
            ds_ij = tl.where(causal_mask, ds_ij, 0.0)

        # Accumulate dQ directly in registers
        dq_acc += tl.dot(ds_ij, k)

        # Accumulate dK and dV atomically / streamed
        dk = tl.dot(tl.trans(ds_ij), q)
        dv = tl.dot(tl.trans(w_ij), gout)

        dk_ptrs = DK_ptr + dk_offset + curr_offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk
        dv_ptrs = DV_ptr + dv_offset + curr_offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvk
        tl.atomic_add(dk_ptrs, dk.to(DK_ptr.dtype.element_ty), mask=k_mask)
        tl.atomic_add(dv_ptrs, dv.to(DV_ptr.dtype.element_ty), mask=k_mask)

    dq_ptrs = DQ_ptr + dq_offset + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqk
    tl.store(dq_ptrs, dq_acc.to(DQ_ptr.dtype.element_ty), mask=q_mask)


def triton_algebraic_flash_attention(
    q,
    k,
    v,
    sink_omega: float = 0.5,
    causal: bool = False,
    block_m: int = 128,
    block_n: int = 128,
):
    """High-level entrypoint launching Triton Octic FlashAttention on GPU."""
    if torch is None:
        raise RuntimeError("PyTorch is required to execute Triton GPU kernels.")
    if not q.is_cuda:
        raise ValueError("Input tensors must reside on CUDA GPU.")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical (B, H, T, D) shapes")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have identical dtypes")
    if block_m <= 0 or block_n <= 0:
        raise ValueError("block_m and block_n must be positive")

    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = float(1.0 / math.sqrt(head_dim))

    out = torch.empty_like(q)
    d_total = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32)

    grid = (math.ceil(seq_len / block_m), batch_size * num_heads)

    _triton_afa_fwd_kernel[grid](
        q, k, v, out, d_total,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        d_total.stride(0), d_total.stride(1), d_total.stride(2),
        scale=scale,
        sink_omega=float(sink_omega),
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_CAUSAL=causal,
    )
    return out
