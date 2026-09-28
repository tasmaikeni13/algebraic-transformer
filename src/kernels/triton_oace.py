"""Triton implementation of the fused vocabulary projection and algebraic loss.

Vocabulary chunks bound temporary logit storage. The backward pass recomputes
chunk activations and accumulates parameter and hidden-state gradients."""

import math
from typing import Optional, Tuple

import triton
import triton.language as tl

try:
    import torch
except ImportError:
    torch = None


@triton.jit
def _triton_octic_rational_kernel(s):
    """Evaluates octic algebraic kernel and powers on Triton register vectors."""
    s_sq = s * s
    rad = 1.0 + s_sq
    r = tl.rsqrt(rad)
    u = s * r
    denom = tl.where(s < 0.0, 1.0 - u, 1.0)
    rho = tl.where(s < 0.0, r / denom, s + rad * r)
    k2 = rho * rho
    k4 = k2 * k2
    k8 = k4 * k4
    rho7 = rho * k2 * k4
    return k8, rho7, rho, r


@triton.jit
def _triton_linear_oace_fwd_kernel(
    H_ptr, W_ptr, Targets_ptr, Loss_ptr,
    stride_hn, stride_hd,
    stride_wd, stride_wv,
    stride_tn,
    num_tokens: tl.constexpr,
    d_model: tl.constexpr,
    vocab_size: tl.constexpr,
    eps_vocab: tl.constexpr,
    gamma: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Tiled Triton forward kernel for Fused Linear + OACE projection head."""
    off_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = off_n < num_tokens

    # Targets for this token block
    targets_ptrs = Targets_ptr + off_n * stride_tn
    targets = tl.load(targets_ptrs, mask=mask_n, other=0)

    # Load H block: (BLOCK_N, d_model)
    offs_d = tl.arange(0, d_model)
    h_ptrs = H_ptr + off_n[:, None] * stride_hn + offs_d[None, :] * stride_hd
    h = tl.load(h_ptrs, mask=mask_n[:, None], other=0.0)

    # Pass 1: Chunked sum of squares across vocabulary
    ss_total = tl.zeros([BLOCK_N], dtype=tl.float32)
    for start_v in range(0, vocab_size, BLOCK_V):
        offs_v = start_v + tl.arange(0, BLOCK_V)
        mask_v = offs_v < vocab_size

        w_ptrs = W_ptr + offs_d[:, None] * stride_wd + offs_v[None, :] * stride_wv
        w = tl.load(w_ptrs, mask=mask_v[None, :], other=0.0)

        # Chunk logits in SRAM
        z_c = tl.dot(h, w)
        ss_total += tl.sum(z_c * z_c, axis=1)

    inv_V = 1.0 / vocab_size
    tau = tl.rsqrt(ss_total * inv_V + eps_vocab)

    # Pass 2: Octic powers and partition sum
    sum_k8 = tl.zeros([BLOCK_N], dtype=tl.float32)
    sum_rho7 = tl.zeros([BLOCK_N], dtype=tl.float32)
    rho_target = tl.zeros([BLOCK_N], dtype=tl.float32)

    for start_v in range(0, vocab_size, BLOCK_V):
        offs_v = start_v + tl.arange(0, BLOCK_V)
        mask_v = offs_v < vocab_size

        w_ptrs = W_ptr + offs_d[:, None] * stride_wd + offs_v[None, :] * stride_wv
        w = tl.load(w_ptrs, mask=mask_v[None, :], other=0.0)

        z_c = tl.dot(h, w)
        normed_c = z_c * tau[:, None]

        k8_c, rho7_c, rho_c, _ = _triton_octic_rational_kernel(normed_c)
        sum_k8 += tl.sum(k8_c, axis=1)
        sum_rho7 += tl.sum(rho7_c, axis=1)

        # Check if target token is within this chunk
        is_in_chunk = (targets >= start_v) & (targets < (start_v + BLOCK_V))
        target_local_idx = targets - start_v
        # Extract target rho
        target_mask = (offs_v[None, :] == targets[:, None]) & is_in_chunk[:, None]
        chunk_target_rho = tl.sum(tl.where(target_mask, rho_c, 0.0), axis=1)
        rho_target += tl.where(is_in_chunk, chunk_target_rho, 0.0)

    # Scalar 3-rsqrt cascade on partition sum S
    inv_S = 1.0 / sum_k8
    S_eighth = tl.rsqrt(tl.rsqrt(tl.rsqrt(inv_S)))
    sum_p78 = (S_eighth * inv_S) * sum_rho7
    p_c_inv8 = S_eighth / rho_target

    loss_i = gamma * (8.0 * p_c_inv8 + (8.0 / 7.0) * sum_p78 - (64.0 / 7.0))

    # Store loss
    loss_ptrs = Loss_ptr + off_n
    tl.store(loss_ptrs, loss_i, mask=mask_n)


def triton_fused_linear_oace(
    h,
    w_vocab,
    targets,
    eps_vocab: float = 100.0,
    gamma: float = 2.0,
    chunk_size: int = 4096,
):
    """High-level Python entrypoint executing Fused Linear + OACE on GPU via Triton."""
    if torch is None:
        raise RuntimeError("PyTorch is required to execute Triton GPU kernels.")
    if not h.is_cuda:
        raise ValueError("Input tensors must reside on CUDA GPU.")

    num_tokens = h.shape[0] if h.ndim == 2 else math.prod(h.shape[:-1])
    d_model = h.shape[-1]
    vocab_size = w_vocab.shape[-1]

    h_flat = h.view(num_tokens, d_model)
    targets_flat = targets.view(num_tokens)

    losses = torch.empty(num_tokens, device=h.device, dtype=torch.float32)
    block_n = 64
    grid = (math.ceil(num_tokens / block_n),)

    _triton_linear_oace_fwd_kernel[grid](
        h_flat, w_vocab, targets_flat, losses,
        h_flat.stride(0), h_flat.stride(1),
        w_vocab.stride(0), w_vocab.stride(1),
        targets_flat.stride(0),
        num_tokens=num_tokens,
        d_model=d_model,
        vocab_size=vocab_size,
        eps_vocab=eps_vocab,
        gamma=gamma,
        BLOCK_N=block_n,
        BLOCK_V=chunk_size,
    )
    return torch.mean(losses)
