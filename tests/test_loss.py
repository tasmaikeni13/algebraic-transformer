"""Tests for Loss study algebraic loss functionals: OACE and Pearson divergence."""
import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.audit_primitives import FORBIDDEN, primitives_in, source_audit
from src.attention import algebraic_softmax
from src.loss import oace_loss, pearson_divergence
from tests.reference_loss import (
    cross_entropy_fp64,
    kl_divergence_fp64,
    oace_loss_fp64,
    oace_vjp_fp64,
    pearson_divergence_fp64,
    pearson_vjp_fp64,
)


@pytest.mark.parametrize("dtype,tol", [(jnp.float64, 1e-13), (jnp.float32, 1e-5), (jnp.bfloat16, 0.02)])
@pytest.mark.parametrize("gamma", [1.0, 2.0])
def test_oace_oracle_and_vjp(dtype, tol, gamma):
    rng = np.random.default_rng(42)
    K = 16
    B = 4
    raw = rng.uniform(0.1, 1.0, size=(B, K))
    probs = (raw / np.sum(raw, axis=-1, keepdims=True)).astype(np.float64)
    targets = rng.integers(0, K, size=B)
    g = rng.normal(size=B).astype(np.float64)

    p_jax = jnp.asarray(probs, dtype=dtype)
    targets_jax = jnp.asarray(targets)
    g_jax = jnp.asarray(g, dtype=dtype)

    # 1. Forward oracle parity
    loss_jax, vjp_fn = jax.vjp(lambda p: oace_loss(p, targets_jax, gamma=gamma, reduction="none"), p_jax)
    loss_ref = oace_loss_fp64(probs, targets, gamma=gamma, reduction="none")
    np.testing.assert_allclose(np.asarray(loss_jax, dtype=float), loss_ref, rtol=tol, atol=tol)

    # 2. Backward VJP oracle parity
    grad_jax = vjp_fn(g_jax)[0]
    grad_ref = oace_vjp_fp64(probs, targets, g, gamma=gamma)
    np.testing.assert_allclose(np.asarray(grad_jax, dtype=float), grad_ref, rtol=tol, atol=tol)
    assert loss_jax.dtype == dtype
    assert grad_jax.dtype == dtype


def test_oace_distribution_targets_parity():
    rng = np.random.default_rng(43)
    K = 8
    B = 5
    raw = rng.uniform(0.1, 1.0, size=(B, K))
    probs = raw / np.sum(raw, axis=-1, keepdims=True)
    targets = rng.integers(0, K, size=B)
    targets_one_hot = np.zeros((B, K), dtype=np.float64)
    targets_one_hot[np.arange(B), targets] = 1.0

    p_jax = jnp.asarray(probs, dtype=jnp.float64)
    loss_int = oace_loss(p_jax, jnp.asarray(targets), gamma=2.0, reduction="none")
    loss_dist = oace_loss(p_jax, jnp.asarray(targets_one_hot), gamma=2.0, reduction="none")
    np.testing.assert_allclose(loss_int, loss_dist, atol=1e-15)

    g_int = jax.grad(lambda p: oace_loss(p, jnp.asarray(targets), gamma=2.0, reduction="sum"))(p_jax)
    g_dist = jax.grad(lambda p: oace_loss(p, jnp.asarray(targets_one_hot), gamma=2.0, reduction="sum"))(p_jax)
    np.testing.assert_allclose(g_int, g_dist, atol=1e-15)


def test_simplex_boundary_stability():
    """Distinguish the probability gradient from the bounded composed score gradient."""
    K = 10
    pk_vals = np.logspace(-9, np.log10(1.0 - 1e-9), num=1000)
    for pk in pk_vals:
        # Build probability vector with target probability pk
        p = np.full(K, (1.0 - pk) / (K - 1), dtype=np.float64)
        p[0] = pk
        target = 0

        p_jax = jnp.asarray(p, dtype=jnp.float64)
        loss = float(oace_loss(p_jax, target, gamma=2.0))
        assert np.isfinite(loss), f"Loss not finite at pk={pk}: {loss}"
        assert loss >= 0.0

        grad = np.asarray(jax.grad(lambda x: oace_loss(x, target, gamma=1.0))(p_jax))
        assert np.all(np.isfinite(grad)), f"Grad not finite at pk={pk}: {grad}"

        expected_target_grad = pk ** (-1.0 / 8.0) - pk ** (-9.0 / 8.0)
        np.testing.assert_allclose(grad[0], expected_target_grad, rtol=2e-12, atol=1e-12)

    # AVN-bounded A-Softmax supplies the finite score-space guarantee used by
    # training.  This is deliberately a separate assertion from dL/dp above.
    def composed(scores):
        probabilities = algebraic_softmax(scores, sink_omega=0.5)
        return oace_loss(probabilities, 0, gamma=1.0)

    for scale in np.logspace(-3, 15, 64):
        scores = jnp.linspace(-scale, scale, K, dtype=jnp.float64)
        score_grad = np.asarray(jax.grad(composed)(scores))
        assert np.all(np.isfinite(score_grad))


def test_strict_propriety_and_monotonicity():
    """The corrected power score is minimized at p=y, including soft targets."""
    rng = np.random.default_rng(47)
    for _ in range(100):
        y_raw = rng.uniform(0.05, 1.0, size=8)
        q_raw = rng.uniform(0.05, 1.0, size=8)
        y = y_raw / y_raw.sum()
        q = q_raw / q_raw.sum()
        y_jax = jnp.asarray(y, dtype=jnp.float64)
        q_jax = jnp.asarray(q, dtype=jnp.float64)

        at_truth = float(oace_loss(y_jax, y_jax, gamma=1.0))
        away = float(oace_loss(q_jax, y_jax, gamma=1.0))
        assert abs(at_truth) <= 2e-14
        assert away > at_truth

        grad_at_truth = jax.grad(lambda p: oace_loss(p, y_jax, gamma=1.0))(y_jax)
        np.testing.assert_allclose(grad_at_truth, 0.0, atol=2e-13)

    # Along the symmetric hard-label path, increasing the target probability
    # strictly decreases the proper score toward its unique boundary minimum.
    pk_vals = np.linspace(1e-6, 1.0 - 1e-9, 10_000)
    probs = np.full((len(pk_vals), 8), 0.0)
    probs[:, 0] = pk_vals
    probs[:, 1:] = ((1.0 - pk_vals) / 7.0)[:, None]
    losses = np.asarray(oace_loss(jnp.asarray(probs), jnp.zeros(len(pk_vals), dtype=jnp.int32), gamma=1.0, reduction="none"))
    assert np.all(losses >= -1e-12)
    assert np.all(np.diff(losses) < 0.0)


@pytest.mark.parametrize("dtype,tol", [(jnp.float64, 1e-13), (jnp.float32, 1e-5)])
def test_pearson_divergence_oracle_and_vjp(dtype, tol):
    rng = np.random.default_rng(44)
    K = 10
    B = 3
    raw_p = rng.uniform(0.1, 1.0, size=(B, K))
    raw_q = rng.uniform(0.1, 1.0, size=(B, K))
    p = raw_p / np.sum(raw_p, axis=-1, keepdims=True)
    q = raw_q / np.sum(raw_q, axis=-1, keepdims=True)
    g = rng.normal(size=B)

    p_jax = jnp.asarray(p, dtype=dtype)
    q_jax = jnp.asarray(q, dtype=dtype)
    g_jax = jnp.asarray(g, dtype=dtype)

    div_jax, pb = jax.vjp(lambda a, b: pearson_divergence(a, b, reduction="none"), p_jax, q_jax)
    div_ref = pearson_divergence_fp64(p, q, reduction="none")
    np.testing.assert_allclose(np.asarray(div_jax, dtype=float), div_ref, rtol=tol, atol=tol)

    dp_jax, dq_jax = pb(g_jax)
    dp_ref, dq_ref = pearson_vjp_fp64(p, q, g)
    np.testing.assert_allclose(np.asarray(dp_jax, dtype=float), dp_ref, rtol=tol, atol=tol)
    np.testing.assert_allclose(np.asarray(dq_jax, dtype=float), dq_ref, rtol=tol, atol=tol)


def test_pearson_divergence_properties():
    """Verify D_A(p || q) >= 0 with equality iff p == q."""
    rng = np.random.default_rng(45)
    K = 8
    for _ in range(100):
        raw_p = rng.uniform(0.05, 1.0, size=K)
        raw_q = rng.uniform(0.05, 1.0, size=K)
        p = jnp.asarray(raw_p / np.sum(raw_p), dtype=jnp.float64)
        q = jnp.asarray(raw_q / np.sum(raw_q), dtype=jnp.float64)

        # Non-negativity
        div = float(pearson_divergence(p, q))
        assert div >= 0.0

        # Zero iff equal
        div_self = float(pearson_divergence(p, p))
        assert abs(div_self) <= 1e-15


def test_fisher_information_ratio():
    """Verify Hessian ratio H(D_A) / H(D_KL) at p = y is identically [2.0, 2.0, ...]."""
    rng = np.random.default_rng(46)
    K = 6
    for _ in range(20):
        raw = rng.uniform(0.1, 1.0, size=K)
        y = jnp.asarray(raw / np.sum(raw), dtype=jnp.float64)

        # Riemannian Hessian of Pearson divergence with respect to prediction q at q = y
        h_pearson = jax.hessian(lambda q: pearson_divergence(y, q))(y)
        # Riemannian Hessian of KL divergence with respect to prediction q at q = y
        h_kl = jax.hessian(lambda q: jnp.sum(y * jnp.log(y / q)))(y)

        diag_p = jnp.diag(h_pearson)
        diag_kl = jnp.diag(h_kl)
        ratio = np.asarray(diag_p / diag_kl)

        expected = np.full(K, 2.0)
        np.testing.assert_allclose(ratio, expected, rtol=1e-12, atol=1e-12)


def test_purity_ast_audit():
    """Verify zero transcendental function calls in src/loss.py."""
    source_path = Path(__file__).resolve().parents[1] / "src/loss.py"
    source = source_path.read_text()

    # 1. Source AST walk
    violations = source_audit(source)
    # Also check for sqrt, softmax, log_softmax, cross_entropy
    for node in ast.walk(ast.parse(source)):
        name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
        if name in {"log_softmax", "softmax_cross_entropy", "cross_entropy", "sqrt"}:
            violations.append({"line": node.lineno, "name": name})
    assert not violations, f"AST violations found: {violations}"

    # 2. Token / regex check
    import io
    import re
    import tokenize

    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in {tokenize.COMMENT, tokenize.STRING}:
            tokens.append(token.string)
    token_str = " ".join(tokens)
    regex_hits = re.findall(r"\b(?:exp|log|sin|cos|tanh|sigmoid)\b", token_str)
    assert not regex_hits, f"Regex violations found in code tokens: {regex_hits}"

    # 3. Traced graph primitive inspection
    x = jnp.array([0.2, 0.5, 0.3], dtype=jnp.float32)
    t = jnp.array(1, dtype=jnp.int32)
    traces = {
        "oace_forward": jax.make_jaxpr(lambda p: oace_loss(p, t))(x),
        "oace_backward": jax.make_jaxpr(jax.grad(lambda p: oace_loss(p, t)))(x),
        "pearson_forward": jax.make_jaxpr(lambda p: pearson_divergence(p, p))(x),
        "pearson_backward": jax.make_jaxpr(jax.grad(lambda p: pearson_divergence(p, p)))(x),
    }
    for name, trace in traces.items():
        counts = primitives_in(trace)
        invalid = sorted(set(counts.keys()) & (FORBIDDEN | {"sqrt"}))
        assert not invalid, f"Forbidden primitives in trace {name}: {invalid}"


def test_input_validation():
    """Verify input validation and error raising on invalid arguments."""
    with pytest.raises(TypeError):
        oace_loss(jnp.array([1, 2, 3]), 0)  # integer array
    with pytest.raises(ValueError):
        oace_loss(jnp.array([]), 0)  # empty
    with pytest.raises(ValueError):
        oace_loss(jnp.ones(4) / 4, 0, gamma=-1.0)  # negative gamma
    with pytest.raises(ValueError):
        oace_loss(jnp.ones(4) / 4, 0, gamma=float("nan"))
    with pytest.raises(ValueError):
        oace_loss(jnp.ones((2, 4)) / 4, jnp.zeros(5, dtype=jnp.int32))  # shape mismatch
    with pytest.raises(ValueError):
        pearson_divergence(jnp.ones(4), jnp.ones(5))  # shape mismatch
