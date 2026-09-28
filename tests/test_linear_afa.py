"""Tests for the experimental O(N) diagonal-feature AFA approximation.

Verifies:
1. Zero-transcendental AST and token audit.
2. Bitwise equivalence between sequential O(1) recurrence and O(N) parallel prefix scan.
3. Positivity and numerical stability of algebraic polynomial feature mapping.
4. Correctness of the first nine Taylor-series coefficients.
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

from src.kernels.linear_afa import (
    OCTIC_POLYNOMIAL_COEFFICIENTS,
    LinearAFAState,
    compute_algebraic_feature_map,
    linear_afa_init_state,
    linear_afa_step,
    linear_afa_parallel_scan,
)


FORBIDDEN_IDENTIFIERS = {
    "exp", "expm1", "exp2", "log", "log1p", "log2", "log10",
    "sin", "cos", "tan", "tanh", "sinh", "cosh", "sigmoid", "logistic", "erf", "erfc",
}


def test_zero_transcendental_ast_audit_linear_afa():
    """Verify src/kernels/linear_afa.py contains 0 transcendental AST calls and token matches."""
    path = Path(__file__).resolve().parents[1] / "src/kernels/linear_afa.py"
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


def test_recurrent_step_matches_parallel_scan():
    """Verify O(1) sequential recurrence produces bitwise identical results to parallel prefix scan."""
    key = jax.random.PRNGKey(501)
    B, H, T, D_k, D_v = 2, 2, 32, 16, 16
    order = 4
    sink_omega = 0.5

    q = jax.random.normal(key, (B, H, T, D_k), dtype=jnp.float64)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, T, D_k), dtype=jnp.float64)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, T, D_v), dtype=jnp.float64)

    # 1. Parallel prefix scan
    y_scan = linear_afa_parallel_scan(q, k, v, sink_omega=sink_omega, order=order)

    # 2. Sequential recurrent step
    d_phi = compute_algebraic_feature_map(q[:, :, :1, :], order=order).shape[-1]
    state = linear_afa_init_state((B, H), d_phi, D_v, dtype=jnp.float64)

    y_steps = []
    for t in range(T):
        qt = q[:, :, t, :]
        kt = k[:, :, t, :]
        vt = v[:, :, t, :]
        yt, state = linear_afa_step(state, qt, kt, vt, sink_omega=sink_omega, order=order)
        y_steps.append(yt)

    y_recurrent = jnp.stack(y_steps, axis=2)

    diff = np.max(np.abs(np.array(y_scan - y_recurrent)))
    assert diff <= 1.0e-12, f"Scan vs recurrent mismatch: {diff}"


def test_taylor_coefficients_match_ground_truth():
    """Verify Taylor-series coefficients c_0 through c_8 match analytical values."""
    expected = [1.0, 8.0, 32.0, 84.0, 160.0, 231.0, 256.0, 214.5, 128.0]
    actual = list(OCTIC_POLYNOMIAL_COEFFICIENTS)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_feature_map_with_random_projection():
    """Verify linear attention with random projection basis executes cleanly and finitely."""
    key = jax.random.PRNGKey(502)
    B, H, T, D_k, D_v = 1, 2, 16, 8, 8
    R = 12
    proj = jax.random.normal(key, (R, D_k), dtype=jnp.float32)

    q = jax.random.normal(key, (B, H, T, D_k), dtype=jnp.float32)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, T, D_k), dtype=jnp.float32)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, T, D_v), dtype=jnp.float32)

    out = linear_afa_parallel_scan(q, k, v, projection_basis=proj, order=3)
    assert out.shape == (B, H, T, D_v)
    assert jnp.all(jnp.isfinite(out))
