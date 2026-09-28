"""Comprehensive verification test suite for Algebraic Geometric Oscillators (AGO)."""
import ast
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.audit_primitives import FORBIDDEN, primitives_in, source_audit
from src.attention import (
    CayleyRotary,
    apply_ago_rotations,
    build_cayley_rotary_matrix,
)
from tests.reference_ago import (
    apply_ago_rotations_fp64,
    apply_rope_rotations_fp64,
    build_cayley_rotary_matrix_fp64,
    cayley_rotation_matrix_fp64,
)


@pytest.mark.parametrize("dtype,tol", [(jnp.float64, 1.0e-12), (jnp.float32, 2.0e-5), (jnp.bfloat16, 0.04)])
def test_oracle_parity_and_vjp(dtype, tol):
    """Verifies AGO JAX implementation against independent float64 NumPy oracle."""
    rng = np.random.default_rng(42)
    dim = 32
    seq_len = 64
    batch_size = 2
    num_heads = 4
    head_dim = dim

    q_np = rng.standard_normal((batch_size, seq_len, num_heads, head_dim))
    k_np = rng.standard_normal((batch_size, seq_len, num_heads, head_dim))

    q = jnp.asarray(q_np, dtype=dtype)
    k = jnp.asarray(k_np, dtype=dtype)
    g = jnp.asarray(rng.standard_normal(q_np.shape), dtype=dtype)

    rot = build_cayley_rotary_matrix(head_dim, seq_len, dtype=dtype)
    qr, kr = apply_ago_rotations(q, k, rotary_params=rot)

    # Reference oracle
    qr_ref, kr_ref = apply_ago_rotations_fp64(q_np, k_np)

    np.testing.assert_allclose(np.asarray(qr, dtype=float), qr_ref, rtol=tol, atol=tol)
    np.testing.assert_allclose(np.asarray(kr, dtype=float), kr_ref, rtol=tol, atol=tol)
    assert qr.dtype == q.dtype and kr.dtype == k.dtype

    # Differentiation via VJP
    def loss(a, b):
        x, y = apply_ago_rotations(a, b, rotary_params=rot)
        return jnp.sum(x * g)

    grad_q = jax.grad(loss, argnums=0)(q, k)
    assert grad_q.dtype == q.dtype
    assert jnp.all(jnp.isfinite(grad_q))


def test_unimodularity_and_orthogonality():
    """Verifies det(R(w))=1.0 and c1 . c2 = 0 with error <= 1e-15 across 10^5 samples."""
    from scripts.positions_experiments import determinant_and_orthogonality_study
    res = determinant_and_orthogonality_study(trials=100_000, seed=42)
    assert res["passed"] is True
    assert res["max_determinant_error"] <= 1.0e-15
    assert res["max_orthogonality_error"] <= 1.0e-15
    assert res["max_col1_norm_error"] <= 1.0e-15
    assert res["max_col2_norm_error"] <= 1.0e-15


def test_cumulative_norm_conservation():
    """Verifies cumulative norm conservation drift <= 1.0e-6 across L=8192."""
    from scripts.positions_experiments import norm_conservation_study
    res = norm_conservation_study(max_seq_len=8192, dim=64, seed=42)
    assert res["passed"] is True
    assert res["max_norm_conservation_drift"] <= 1.0e-6


def test_long_context_shift_equivariance():
    """Verifies relative shift equivariance ||R_m^T R_n - R_{n-m}||_inf <= 1.0e-6 across L=4096."""
    from scripts.positions_experiments import shift_equivariance_study
    res = shift_equivariance_study(max_seq_len=4096, dim=64, stride=32)
    assert res["passed"] is True
    assert res["max_shift_equivariance_error"] <= 1.0e-6


def test_relative_attention_dot_product_error():
    """Verifies relative attention dot product error <= 1.0e-6 across 10^5 pairs."""
    from scripts.positions_experiments import relative_dot_product_study
    res = relative_dot_product_study(trials=100_000, dim=64, max_seq_len=4096, seed=42)
    assert res["passed"] is True
    assert res["max_relative_dot_product_error"] <= 1.0e-6


def test_associative_recall_out_of_distribution():
    """Verifies model trained on L=256 reaches >= 95% retrieval accuracy on L=1024 and L=2048."""
    from scripts.positions_experiments import associative_recall_study
    res = associative_recall_study(seed=42)
    assert res["passed"] is True
    assert res["results"]["acc_L1024"] >= 0.95
    assert res["results"]["acc_L2048"] >= 0.95


def test_ago_vs_rope_benchmark():
    """Verifies AGO execution and latency measurement vs standard trigonometric RoPE."""
    from scripts.positions_experiments import benchmark_ago_vs_rope
    res = benchmark_ago_vs_rope(seq_len=512, batch_size=4, repetitions=10)
    assert res["ago_latency_ms"] > 0
    assert res["rope_latency_ms"] > 0
    assert res["throughput_ratio"] > 0


def test_zero_trigonometric_and_purity_audit():
    """AST audit confirming zero occurrences of sin, cos, or complex numbers."""
    from scripts.positions_experiments import audit
    res = audit()
    assert res["passed"] is True, f"Audit failed: {res}"
    assert len(res["source_violations"]) == 0
    assert len(res["regex_hits"]) == 0


def test_tensor_shapes_and_broadcasting():
    """Verifies apply_ago_rotations across 2D, 3D, and 4D tensor layouts."""
    rng = np.random.default_rng(123)
    dim = 16
    rot = build_cayley_rotary_matrix(dim, 64)

    # 2D: (seq_len, dim)
    q2 = jnp.asarray(rng.standard_normal((32, dim)), dtype=jnp.float32)
    k2 = jnp.asarray(rng.standard_normal((32, dim)), dtype=jnp.float32)
    qr2, kr2 = apply_ago_rotations(q2, k2, rotary_params=rot)
    assert qr2.shape == (32, dim) and kr2.shape == (32, dim)

    # 3D: (batch, seq_len, dim)
    q3 = jnp.asarray(rng.standard_normal((4, 32, dim)), dtype=jnp.float32)
    k3 = jnp.asarray(rng.standard_normal((4, 32, dim)), dtype=jnp.float32)
    qr3, kr3 = apply_ago_rotations(q3, k3, rotary_params=rot)
    assert qr3.shape == (4, 32, dim) and kr3.shape == (4, 32, dim)

    # 4D: (batch, seq_len, num_heads, head_dim)
    q4 = jnp.asarray(rng.standard_normal((2, 32, 4, dim)), dtype=jnp.float32)
    k4 = jnp.asarray(rng.standard_normal((2, 32, 4, dim)), dtype=jnp.float32)
    qr4, kr4 = apply_ago_rotations(q4, k4, rotary_params=rot)
    assert qr4.shape == (2, 32, 4, dim) and kr4.shape == (2, 32, 4, dim)

    # 4D transposed: (batch, num_heads, seq_len, head_dim)
    q4_t = jnp.asarray(rng.standard_normal((2, 4, 32, dim)), dtype=jnp.float32)
    k4_t = jnp.asarray(rng.standard_normal((2, 4, 32, dim)), dtype=jnp.float32)
    qr4_t, kr4_t = apply_ago_rotations(q4_t, k4_t, rotary_params=rot, seq_axis=2)
    assert qr4_t.shape == (2, 4, 32, dim) and kr4_t.shape == (2, 4, 32, dim)


def test_invalid_arguments():
    """Verifies appropriate ValueError on invalid configurations."""
    with pytest.raises(ValueError):
        build_cayley_rotary_matrix(15, 64) # Odd dimension

    with pytest.raises(ValueError):
        build_cayley_rotary_matrix(16, 0) # Nonpositive length

    q = jnp.ones((2, 16, 16))
    k = jnp.ones((2, 16, 8))
    with pytest.raises(ValueError):
        apply_ago_rotations(q, k) # Mismatched shapes
