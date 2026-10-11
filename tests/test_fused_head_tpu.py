"""TPU check: fused vocabulary projection + loss against dense references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.sota import algebraic as A
from src.sota import standard as S

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="needs a TPU")


@pytest.mark.parametrize("kind,n,c", [("ce", 16, 16.0), ("alg_ce", 16, 16.0), ("alg_ce", 8, 2.0)])
@pytest.mark.parametrize("V", [50257, 4096])
def test_fused_head_matches_dense(kind, n, c, V):
    from src.sota.kernels.fused_head import fused_head
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
