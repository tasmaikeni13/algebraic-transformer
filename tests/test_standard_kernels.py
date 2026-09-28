"""Tests for optimized standard transformer kernels (FlashAttention-2 & Fused Cross-Entropy).

Verifies:
1. FlashAttention-2 numerical equivalence to reference softmax attention.
2. FlashAttention-2 analytical gradient accuracy vs automatic differentiation.
3. Fused Linear + Cross-Entropy equivalence to standard cross-entropy loss.
4. Fused Linear + Cross-Entropy gradient accuracy vs automatic differentiation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.kernels.pallas_flash_attention import (
    tiled_flash_attention_forward,
    tiled_flash_attention_backward,
    standard_flash_attention,
)
from src.kernels.fused_cross_entropy import (
    standard_fused_linear_ce_forward,
    standard_fused_linear_ce_backward,
    standard_fused_cross_entropy,
)


def test_flash_attention_forward_parity():
    """Verify FlashAttention-2 tiled forward pass matches standard softmax attention."""
    key = jax.random.PRNGKey(601)
    B, H, L, D = 1, 2, 256, 64
    scale = 1.0 / (D ** 0.5)
    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.float32)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, L, D), dtype=jnp.float32)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, L, D), dtype=jnp.float32)

    # Reference un-tiled attention
    mask = jnp.tril(jnp.ones((L, L), dtype=bool))[None, None, :, :]
    s = jnp.matmul(q * scale, jnp.swapaxes(k, -1, -2))
    s_masked = jnp.where(mask, s, -1e9)
    w = jax.nn.softmax(s_masked, axis=-1)
    ref_out = jnp.matmul(w, v)

    flash_out = tiled_flash_attention_forward(q, k, v, causal=True, block_q=128, block_k=128)

    diff = np.max(np.abs(np.array(flash_out - ref_out)))
    rel_err = diff / np.max(np.abs(np.array(ref_out)))
    assert rel_err <= 1e-4, f"FlashAttention mismatch: rel_err={rel_err}"


def test_flash_attention_unequal_tiles_and_arbitrary_length():
    key = jax.random.PRNGKey(605)
    B, H, L, D = 1, 1, 90, 16
    q = jax.random.normal(key, (B, H, L, D))
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, L, D))
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, L, D))

    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / (D ** 0.5)
    mask = jnp.tril(jnp.ones((L, L), dtype=bool))[None, None]
    expected = jnp.matmul(jax.nn.softmax(jnp.where(mask, scores, -1e9), axis=-1), v)
    actual = standard_flash_attention(q, k, v, causal=True, block_q=32, block_k=64)

    assert actual.shape == (B, H, L, D)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)


def test_flash_attention_gradient_accuracy():
    """Verify FlashAttention-2 analytical VJP gradients match AD reference."""
    key = jax.random.PRNGKey(602)
    B, H, L, D = 1, 1, 128, 32
    scale = 1.0 / (D ** 0.5)
    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.float32)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, L, D), dtype=jnp.float32)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, L, D), dtype=jnp.float32)

    def ref_loss(q_a, k_a, v_a):
        mask = jnp.tril(jnp.ones((L, L), dtype=bool))[None, None, :, :]
        s = jnp.matmul(q_a * scale, jnp.swapaxes(k_a, -1, -2))
        s_masked = jnp.where(mask, s, -1e9)
        w = jax.nn.softmax(s_masked, axis=-1)
        return jnp.sum(jnp.matmul(w, v_a))

    dq_ad, dk_ad, dv_ad = jax.grad(ref_loss, argnums=(0, 1, 2))(q, k, v)

    def flash_loss(q_a, k_a, v_a):
        return jnp.sum(standard_flash_attention(q_a, k_a, v_a, causal=True, block_q=64, block_k=64))

    _, (dq_fl, dk_fl, dv_fl) = jax.value_and_grad(flash_loss, argnums=(0, 1, 2))(q, k, v)

    err_dq = np.max(np.abs(np.array(dq_fl - dq_ad))) / np.max(np.abs(np.array(dq_ad)))
    err_dk = np.max(np.abs(np.array(dk_fl - dk_ad))) / np.max(np.abs(np.array(dk_ad)))
    err_dv = np.max(np.abs(np.array(dv_fl - dv_ad))) / np.max(np.abs(np.array(dv_ad)))

    assert err_dq <= 1e-4, f"dq error: {err_dq}"
    assert err_dk <= 1e-4, f"dk error: {err_dk}"
    assert err_dv <= 1e-4, f"dv error: {err_dv}"


def test_standard_fused_cross_entropy_parity():
    """Verify Standard Fused Linear + Cross-Entropy matches materialized cross-entropy."""
    key = jax.random.PRNGKey(603)
    N, D, V = 16, 32, 256
    h = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w_vocab = jax.random.normal(jax.random.fold_in(key, 1), (D, V), dtype=jnp.float32)
    targets = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    # Reference materialized CE
    logits = jnp.matmul(h, w_vocab)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ref_loss = -jnp.mean(jnp.take_along_axis(log_probs, targets[:, None], axis=-1))

    fused_loss, _ = standard_fused_linear_ce_forward(h, w_vocab, targets, chunk_size=64)

    diff = np.abs(float(fused_loss) - float(ref_loss))
    rel_err = diff / float(ref_loss)
    assert rel_err <= 1e-5, f"Fused CE loss mismatch: rel_err={rel_err}"


def test_standard_fused_cross_entropy_accumulates_bfloat16_in_float32():
    key = jax.random.PRNGKey(606)
    h = jax.random.normal(key, (8, 16), dtype=jnp.bfloat16)
    w = jax.random.normal(jax.random.fold_in(key, 1), (16, 64), dtype=jnp.bfloat16)
    targets = jax.random.randint(jax.random.fold_in(key, 2), (8,), 0, 64)
    loss = standard_fused_cross_entropy(h, w, targets, chunk_size=32)
    assert loss.dtype == jnp.float32


def test_standard_fused_cross_entropy_gradients():
    """Verify Standard Fused Cross-Entropy analytical gradients match AD."""
    key = jax.random.PRNGKey(604)
    N, D, V = 8, 16, 128
    h = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w_vocab = jax.random.normal(jax.random.fold_in(key, 1), (D, V), dtype=jnp.float32)
    targets = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    def ref_fn(h_a, w_a):
        logits = jnp.matmul(h_a, w_a)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(jnp.take_along_axis(log_probs, targets[:, None], axis=-1))

    ref_loss, (dh_ad, dw_ad) = jax.value_and_grad(ref_fn, argnums=(0, 1))(h, w_vocab)

    def fused_fn(h_a, w_a):
        return standard_fused_cross_entropy(h_a, w_a, targets, chunk_size=32)

    fused_loss, (dh_fused, dw_fused) = jax.value_and_grad(fused_fn, argnums=(0, 1))(h, w_vocab)

    err_dh = np.max(np.abs(np.array(dh_fused - dh_ad))) / np.max(np.abs(np.array(dh_ad)))
    err_dw = np.max(np.abs(np.array(dw_fused - dw_ad))) / np.max(np.abs(np.array(dw_ad)))

    assert err_dh <= 1e-4, f"dh error: {err_dh}"
    assert err_dw <= 1e-4, f"dw error: {err_dw}"
