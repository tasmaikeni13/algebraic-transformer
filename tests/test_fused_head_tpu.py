"""TPU check: fused vocabulary projection + loss against dense references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algebraic import algebraic as A
from algebraic import standard as S

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="needs a TPU")


@pytest.mark.parametrize("kind,n,c", [("ce", 16, 16.0), ("alg_ce", 16, 16.0), ("alg_ce", 8, 2.0)])
@pytest.mark.parametrize("V", [32768, 4096])
def test_fused_head_matches_dense(kind, n, c, V):
    from algebraic.kernels.fused_head import fused_head
    N, d = 1024, 256
    ks = jax.random.split(jax.random.PRNGKey(0), 3)
    h = jax.random.normal(ks[0], (N, d), jnp.float32).astype(jnp.bfloat16)
    E = (jax.random.normal(ks[1], (V, d), jnp.float32) * 0.15)
    t = jax.random.randint(ks[2], (N,), 0, V)

    def dense(h, E):
        z = jnp.matmul(h, E.astype(h.dtype).T, preferred_element_type=jnp.float32)
        if kind == "ce":
            return S.cross_entropy(z, t)
        return A.algebraic_likelihood(z, t, n, c)

    def fused(h, E):
        return fused_head(h, E, t, kind=kind, n=n, c=c, block_n=256, block_v=1024)

    (o1, t1), (o2, t2) = fused(h, E), dense(h, E)
    np.testing.assert_allclose(o1, o2, rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(t1, t2, rtol=2e-2, atol=2e-4)
    g1 = jax.grad(lambda h, E: fused(h, E)[0], argnums=(0, 1))(h, E)
    g2 = jax.grad(lambda h, E: dense(h, E)[0], argnums=(0, 1))(h, E)
    for a, b in zip(g1, g2):
        a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
        scale = np.abs(b).max() + 1e-12
        np.testing.assert_allclose(a / scale, b / scale, atol=2e-2)
        cos = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
        assert cos > 0.999


@pytest.mark.parametrize("width", [1024, 1536])
@pytest.mark.parametrize("kind", ["ce", "alg_ce"])
def test_production_head_tiles_at_model_widths(width, kind):
    """New low-memory defaults against an independent dense projection/VJP."""
    from algebraic.kernels.fused_head import fused_head
    tokens, vocab = 256, 32768
    keys = jax.random.split(jax.random.PRNGKey(42), 3)
    hidden = jax.random.normal(keys[0], (tokens, width), jnp.float32).astype(jnp.bfloat16)
    embedding = jax.random.normal(keys[1], (vocab, width), jnp.float32) * .02
    targets = jax.random.randint(keys[2], (tokens,), 0, vocab)

    def dense(h, e):
        logits = jnp.matmul(h, e.astype(h.dtype).T, preferred_element_type=jnp.float32)
        return S.cross_entropy(logits, targets) if kind == "ce" else A.algebraic_likelihood(logits, targets, 8, 2.)

    def fused(h, e):
        return fused_head(h, e, targets, kind=kind, n=8, c=2.)

    (got, telemetry), grads = jax.jit(jax.value_and_grad(fused, argnums=(0, 1), has_aux=True))(hidden, embedding)
    (expected, expected_telemetry), expected_grads = jax.jit(jax.value_and_grad(dense, argnums=(0, 1), has_aux=True))(hidden, embedding)
    np.testing.assert_allclose(got, expected, rtol=.002, atol=.0002)
    np.testing.assert_allclose(telemetry, expected_telemetry, rtol=.02, atol=.0000002)
    for actual, reference in zip(grads, expected_grads):
        actual, reference = np.asarray(actual, np.float32), np.asarray(reference, np.float32)
        assert np.linalg.norm(actual-reference) / np.linalg.norm(reference) < .01
