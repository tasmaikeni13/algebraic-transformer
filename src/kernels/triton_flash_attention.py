"""Triton softmax attention reference for GPU comparisons.

The kernel maintains a running row maximum and denominator across key tiles."""

import math
from typing import Optional

import triton
import triton.language as tl

try:
    import torch
except ImportError:
    torch = None


@triton.jit
def _triton_std_flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, M_ptr, L_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_mb, stride_mh, stride_mm,
    stride_lb, stride_lh, stride_lm,
    scale: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Triton Forward Kernel for Standard FlashAttention-2."""
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

    q_ptrs = Q_ptr + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < seq_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Accumulators
    o_acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    end_n = (start_m + 1) * BLOCK_M if IS_CAUSAL else seq_len

    for start_n in range(0, end_n, BLOCK_N):
        curr_offs_n = start_n + offs_n
        k_mask = curr_offs_n[None, :] < seq_len
        v_mask = curr_offs_n[:, None] < seq_len

        k_ptrs = K_ptr + k_offset + curr_offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        v_ptrs = V_ptr + v_offset + curr_offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Raw scores
        scores = tl.dot(q, k) * scale
        scores = tl.where(curr_offs_n[None, :] < seq_len, scores, -1.0e9)

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= curr_offs_n[None, :]
            scores = tl.where(causal_mask, scores, -1e9)

        # Online Softmax Update
        m_curr = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_curr)

        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        o_acc = o_acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Final normalization
    out = o_acc / tl.maximum(l_i[:, None], 1e-12)

    out_ptrs = Out_ptr + out_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=q_mask)

    m_ptrs = M_ptr + off_b * stride_mb + off_h * stride_mh + offs_m * stride_mm
    l_ptrs = L_ptr + off_b * stride_lb + off_h * stride_lh + offs_m * stride_lm
    tl.store(m_ptrs, m_i, mask=(offs_m < seq_len))
    tl.store(l_ptrs, l_i, mask=(offs_m < seq_len))


def triton_standard_flash_attention(q, k, v, causal: bool = True, block_m: int = 128, block_n: int = 128):
    """High-level Python launcher for standard FlashAttention-2 on GPU."""
    if torch is None:
        raise RuntimeError("PyTorch required for Triton GPU kernels.")
    if not q.is_cuda:
        raise ValueError("Inputs must reside on CUDA GPU.")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical (B, H, T, D) shapes")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have identical dtypes")
    if block_m <= 0 or block_n <= 0:
        raise ValueError("block_m and block_n must be positive")

    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = float(1.0 / math.sqrt(head_dim))

    out = torch.empty_like(q)
    m = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32)
    l = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32)

    grid = (math.ceil(seq_len / block_m), batch_size * num_heads)

    _triton_std_flash_attn_fwd_kernel[grid](
        q, k, v, out, m, l,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        m.stride(0), m.stride(1), m.stride(2),
        l.stride(0), l.stride(1), l.stride(2),
        scale=scale,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_CAUSAL=causal,
    )
    return out
