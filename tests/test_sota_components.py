"""Numerical checks of the second-generation components (CPU, float64)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.sota import algebraic as A
from src.sota import standard as S
from src.sota.config import algebraic, algebraic_v1, hybrid, standard
from src.sota.evaluation import logprob_from_logits
from src.sota.model import eval_logprobs, forward_hidden, head_loss, init_params, position_tables

jax.config.update("jax_enable_x64", True)


def test_rho_identities():
    x = jnp.linspace(-50, 50, 1001, dtype=jnp.float64)
    y = jnp.linspace(-3, 7, 1001, dtype=jnp.float64)
    np.testing.assert_allclose(A.rho(x), np.exp(np.arcsinh(np.asarray(x))), rtol=1e-12)
    np.testing.assert_allclose(A.rho(x) * A.rho(-x), 1.0, rtol=1e-12)
    oplus = x * jnp.sqrt(1 + y * y) + y * jnp.sqrt(1 + x * x)
    np.testing.assert_allclose(A.rho(x) * A.rho(y), A.rho(oplus), rtol=1e-10)


@pytest.mark.parametrize("n,c", [(8, 1.0), (16, 16.0), (32, 4.0)])
def test_asinh_weights_equal_softmax_of_compressed_logits(n, c):
    key = jax.random.PRNGKey(0)
    s = jax.random.normal(key, (2, 3, 7, 7), jnp.float64) * 5
    mask = jnp.tril(jnp.ones((7, 7), bool))[None, None]
    b = jnp.full((1, 3, 1, 1), 0.7, jnp.float64)
    w = A.asinh_power_weights(s, mask, n, c, b)
    logits = jnp.where(mask, n * jnp.arcsinh(s / c), -jnp.inf)
    full = jnp.concatenate([logits, jnp.broadcast_to(n * jnp.arcsinh(b / c), s.shape[:-1] + (1,))], -1)
    ref = jax.nn.softmax(full, -1)[..., :-1]
    np.testing.assert_allclose(w, ref, rtol=1e-10, atol=1e-14)
    # the max shift keeps huge scores finite
    big = A.asinh_power_weights(s * 1e6, mask, n, c, None)
    assert np.all(np.isfinite(big)) and np.allclose(big.sum(-1), 1.0)


@pytest.mark.parametrize("n,c", [(16, 16.0), (8, 2.0)])
def test_algebraic_likelihood_gradient_is_exact_log_loss_gradient(n, c):
    key = jax.random.PRNGKey(1)
    z = jax.random.normal(key, (6, 11), jnp.float64) * 4
    t = jnp.array([0, 3, 10, 5, 5, 1])

    def true_nll(z):
        return -jnp.mean(logprob_from_logits(z, t, "alg_ce", n, c))

    g_ref = jax.grad(true_nll)(z)
    g = jax.grad(lambda z: A.algebraic_likelihood(z, t, n, c)[0])(z)
    np.testing.assert_allclose(g, g_ref, rtol=1e-9, atol=1e-12)
    pt = A.algebraic_likelihood(z, t, n, c)[1]
    np.testing.assert_allclose(np.log(pt), logprob_from_logits(z, t, "alg_ce", n, c), rtol=1e-9)


def test_oace_port_matches_power_score_definition():
    key = jax.random.PRNGKey(2)
    z = jax.random.normal(key, (5, 13), jnp.float64) * 30
    t = jnp.array([1, 2, 12, 0, 7])
    eps, gamma = 100.0, 2.0

    def reference(z):
        zn = z / jnp.sqrt(jnp.mean(z * z, -1, keepdims=True) + eps)
        k = A.rho(zn) ** 8
        p = k / k.sum(-1, keepdims=True)
        pt = jnp.take_along_axis(p, t[:, None], -1)[:, 0]
        return jnp.mean(gamma * (8 * pt ** (-1 / 8) + (8 / 7) * jnp.sum(p ** (7 / 8), -1) - 64 / 7))

    loss, _ = A.oace(z, t, eps, gamma)
    np.testing.assert_allclose(loss, reference(z), rtol=1e-10)
    g = jax.grad(lambda z: A.oace(z, t, eps, gamma)[0])(z)
    np.testing.assert_allclose(g, jax.grad(reference)(z), rtol=1e-8, atol=1e-12)


def test_cayley_tables_are_exact_rotations():
    for table, angles in [
        (A.cayley_table_geometric(64, 300),
         2 * np.arctan(0.5 * 1e4 ** (-np.arange(32) / 32))),
        (A.cayley_table_original(64, 300),
         2 * np.arctan(1 / (1 + (99 / 31) * np.arange(32)) ** 2)),
    ]:
        c, s = map(np.asarray, table)
        m = np.arange(300)[:, None]
        np.testing.assert_allclose(c, np.cos(m * angles), atol=2e-5)
        np.testing.assert_allclose(s, np.sin(m * angles), atol=2e-5)
        np.testing.assert_allclose(c * c + s * s, 1.0, atol=1e-6)


def test_alu_and_its_vjp():
    x = jnp.linspace(-40, 40, 2001, dtype=jnp.float64)
    for beta in (1.0, 0.5):
        ref = 0.5 * x * (1 + beta * x / jnp.sqrt(1 + (beta * x) ** 2))
        np.testing.assert_allclose(A.alu(x, beta), ref, rtol=1e-9, atol=1e-15)
        g = jax.vmap(jax.grad(lambda v: A.alu(v, beta)))(x)
        g_ref = jax.vmap(jax.grad(lambda v: 0.5 * v * (1 + beta * v / jnp.sqrt(1 + (beta * v) ** 2))))(x)
        np.testing.assert_allclose(g, g_ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("make", [standard, algebraic_v1, algebraic, hybrid])
def test_tiny_models_train_step_is_finite(make):
    cfg = make("proxy", d_model=64, n_layers=2, n_heads=2, d_ff=96, vocab_size=97,
               seq_len=16, dtype="float32", remat=False)
    params = init_params(cfg, jax.random.PRNGKey(0))
    tok = jax.random.randint(jax.random.PRNGKey(1), (2, 16), 0, 97)
    tables = position_tables(cfg, 16)

    def loss_fn(p):
        return head_loss(p, forward_hidden(p, tok, cfg, tables), tok, cfg)[0]

    loss, grads = jax.value_and_grad(loss_fn)(params)
    assert np.isfinite(loss)
    assert all(np.all(np.isfinite(g)) for g in jax.tree_util.tree_leaves(grads))
    lp = eval_logprobs(params, tok, tok, cfg, tables)
    assert lp.shape == tok.shape and np.all(lp <= 0)


@pytest.mark.parametrize("n,c", [(16, 16.0), (8, 2.0)])
def test_lean_likelihood_equals_exact(n, c):
    key = jax.random.PRNGKey(3)
    z = jax.random.normal(key, (7, 300), jnp.float64) * 6
    t = jnp.array([0, 5, 299, 17, 17, 100, 3])
    o1, p1 = A.algebraic_likelihood_lean(z, t, n, c)
    o2, p2 = A.algebraic_likelihood(z, t, n, c)
    np.testing.assert_allclose(p1, p2, rtol=1e-9)
    g1 = jax.grad(lambda z: A.algebraic_likelihood_lean(z, t, n, c)[0])(z)
    g2 = jax.grad(lambda z: A.algebraic_likelihood(z, t, n, c)[0])(z)
    np.testing.assert_allclose(g1, g2, rtol=1e-8, atol=1e-13)
