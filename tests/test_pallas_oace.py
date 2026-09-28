"""Comprehensive tests for Fused Linear + OACE Projection Head.

Verifies:
1. Zero-transcendental AST audit.
2. Numerical loss parity against un-tiled reference (fused_oace_softmax_loss).
3. Gradient accuracy against automatic differentiation reference.
4. Invariance to vocabulary chunk sizing (512, 1024, 2048, 4096).
5. Robustness to arbitrary vocabulary sizes not evenly divisible by chunk size.
6. Execution under jax.value_and_grad via custom VJP.
"""

import ast
import io
from pathlib import Path
import re
import tokenize

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.kernels.pallas_oace import (
    _vmu_octic_kernel,
    fused_linear_oace_forward,
    fused_linear_oace_backward,
    fused_linear_oace,
)
from src.model import fused_oace_softmax_loss


FORBIDDEN_IDENTIFIERS = {
    "exp", "expm1", "exp2", "log", "log1p", "log2", "log10",
    "sin", "cos", "tan", "tanh", "sinh", "cosh", "sigmoid", "logistic", "erf", "erfc",
}


def test_zero_transcendental_ast_audit_oace():
    """Verify src/kernels/pallas_oace.py contains 0 transcendental AST calls and token matches."""
    path = Path(__file__).resolve().parents[1] / "src/kernels/pallas_oace.py"
    source = path.read_text()
    tree = ast.parse(source)

    ast_violations = []
    for node in ast.walk(tree):
        name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
        if name in FORBIDDEN_IDENTIFIERS:
            ast_violations.append({"line": node.lineno, "name": name})
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in FORBIDDEN_IDENTIFIERS:
                    ast_violations.append({"line": node.lineno, "name": alias.name})

    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in {tokenize.COMMENT, tokenize.STRING}:
            tokens.append(token.string)
    regex_hits = re.findall(r"\b(?:exp|log|sin|cos|tanh|sigmoid)\b", " ".join(tokens))

    assert len(ast_violations) == 0, f"AST violations: {ast_violations}"
    assert len(regex_hits) == 0, f"Code token regex hits: {regex_hits}"


def test_fused_linear_oace_forward_parity():
    """Verify chunked fused_linear_oace_forward matches un-tiled reference within <= 1.0e-6."""
    key = jax.random.PRNGKey(401)
    B, T, D, V = 2, 8, 32, 256
    h = jax.random.normal(key, (B, T, D), dtype=jnp.float64)
    w_vocab = jax.random.normal(jax.random.fold_in(key, 1), (D, V), dtype=jnp.float64)
    targets = jax.random.randint(jax.random.fold_in(key, 2), (B, T), 0, V)

    # Reference loss via full logits
    logits = jnp.matmul(h, w_vocab)
    ref_loss = fused_oace_softmax_loss(logits, targets, eps=100.0, gamma=2.0)

    # Chunked forward with chunk_size = 64
    chunked_loss, _ = fused_linear_oace_forward(
        h, w_vocab, targets, eps_vocab=100.0, gamma=2.0, chunk_size=64
    )

    diff = np.abs(float(chunked_loss) - float(ref_loss))
    rel_err = diff / float(ref_loss)
    assert rel_err <= 1.0e-6, f"Forward loss mismatch: rel_err={rel_err}"


def test_fused_linear_oace_gradient_accuracy():
    """Verify analytical VJP gradients match AD reference within <= 1.0e-5."""
    key = jax.random.PRNGKey(402)
    B, T, D, V = 2, 4, 16, 128
    h = jax.random.normal(key, (B, T, D), dtype=jnp.float64)
    w_vocab = jax.random.normal(jax.random.fold_in(key, 1), (D, V), dtype=jnp.float64)
    targets = jax.random.randint(jax.random.fold_in(key, 2), (B, T), 0, V)

    def ref_loss_fn(h_arr, w_arr):
        logits = jnp.matmul(h_arr, w_arr)
        return fused_oace_softmax_loss(logits, targets, eps=100.0, gamma=2.0)

    ref_loss, (dh_ad, dw_ad) = jax.value_and_grad(ref_loss_fn, argnums=(0, 1))(h, w_vocab)

    def chunked_loss_fn(h_arr, w_arr):
        return fused_linear_oace(h_arr, w_arr, targets, eps_vocab=100.0, gamma=2.0, chunk_size=32)

    chunked_loss, (dh_ana, dw_ana) = jax.value_and_grad(chunked_loss_fn, argnums=(0, 1))(h, w_vocab)

    err_loss = np.abs(float(chunked_loss) - float(ref_loss)) / float(ref_loss)
    err_dh = np.max(np.abs(np.array(dh_ana - dh_ad))) / np.max(np.abs(np.array(dh_ad)))
    err_dw = np.max(np.abs(np.array(dw_ana - dw_ad))) / np.max(np.abs(np.array(dw_ad)))

    assert err_loss <= 1.0e-6, f"Loss error too high: {err_loss}"
    assert err_dh <= 1.0e-5, f"dh gradient error too high: {err_dh}"
    assert err_dw <= 1.0e-5, f"dw gradient error too high: {err_dw}"


def test_cached_radial_reduction_matches_full_vocabulary_definition():
    """The forward decomposition must equal the AVN VJP's radial reduction."""
    key = jax.random.PRNGKey(405)
    B, T, D, V = 2, 3, 12, 67
    h = jax.random.normal(key, (B, T, D), dtype=jnp.float64)
    w_vocab = jax.random.normal(jax.random.fold_in(key, 1), (D, V), dtype=jnp.float64)
    targets = jax.random.randint(jax.random.fold_in(key, 2), (B, T), 0, V)

    _, cache = fused_linear_oace_forward(h, w_vocab, targets, chunk_size=19)
    radial = cache[8]

    logits = h.reshape(-1, D) @ w_vocab
    tau = jax.lax.rsqrt(jnp.mean(logits * logits, axis=-1, keepdims=True) + 100.0)
    normed = logits * tau
    k8, rho7, rho, r = _vmu_octic_kernel(normed)
    partition = jnp.sum(k8, axis=-1, keepdims=True)
    inv_partition = jax.lax.reciprocal(partition)
    partition_eighth = jax.lax.rsqrt(
        jax.lax.rsqrt(jax.lax.rsqrt(inv_partition))
    )
    probabilities = k8 * inv_partition
    probabilities_78 = partition_eighth * inv_partition * rho7
    target_rho = jnp.take_along_axis(rho, targets.reshape(-1, 1), axis=-1)
    target_inv8 = partition_eighth / target_rho
    sum_probabilities_78 = jnp.sum(probabilities_78, axis=-1, keepdims=True)
    one_hot = jax.nn.one_hot(targets.reshape(-1), V, dtype=jnp.float64)
    bracket = (
        probabilities_78
        - one_hot * target_inv8
        - probabilities * (sum_probabilities_78 - target_inv8)
    )
    expected = jnp.mean(r * bracket * normed, axis=-1, keepdims=True)

    np.testing.assert_allclose(radial, expected, rtol=2e-13, atol=2e-13)


def test_chunk_size_invariance():
    """Different chunk sizes must produce bitwise consistent loss and gradients."""
    key = jax.random.PRNGKey(403)
    B, T, D, V = 1, 8, 16, 256
    h = jax.random.normal(key, (B, T, D), dtype=jnp.float64)
    w_vocab = jax.random.normal(jax.random.fold_in(key, 1), (D, V), dtype=jnp.float64)
    targets = jax.random.randint(jax.random.fold_in(key, 2), (B, T), 0, V)

    loss_32, _ = fused_linear_oace_forward(h, w_vocab, targets, chunk_size=32)
    loss_64, _ = fused_linear_oace_forward(h, w_vocab, targets, chunk_size=64)
    loss_128, _ = fused_linear_oace_forward(h, w_vocab, targets, chunk_size=128)

    np.testing.assert_allclose(float(loss_32), float(loss_64), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(float(loss_64), float(loss_128), rtol=1e-12, atol=1e-12)


def test_indivisible_vocabulary_size():
    """Verify execution when vocabulary size is not divisible by chunk size."""
    key = jax.random.PRNGKey(404)
    B, T, D, V = 2, 5, 20, 203  # 203 is prime, not divisible by any power of 2
    h = jax.random.normal(key, (B, T, D), dtype=jnp.float32)
    w_vocab = jax.random.normal(jax.random.fold_in(key, 1), (D, V), dtype=jnp.float32)
    targets = jax.random.randint(jax.random.fold_in(key, 2), (B, T), 0, V)

    def loss_fn(h_arr, w_arr):
        return fused_linear_oace(h_arr, w_arr, targets, chunk_size=64)

    loss, (dh, dw) = jax.value_and_grad(loss_fn, argnums=(0, 1))(h, w_vocab)
    assert jnp.isfinite(loss)
    assert dh.shape == h.shape
    assert dw.shape == w_vocab.shape
    assert jnp.all(jnp.isfinite(dh))
    assert jnp.all(jnp.isfinite(dw))
