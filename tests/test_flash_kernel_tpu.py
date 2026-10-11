"""TPU check: Pallas flash attention against the dense references.

Run on a TPU host:  python -m pytest -q tests/test_flash_kernel_tpu.py
(skipped on other platforms).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.sota import algebraic as A
from src.sota import standard as S

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="needs a TPU")


def reference(q, k, v, sink, kind, n, c):
    if kind == "softmax":
        return S.softmax_attention(q, k, v, sink_logit=sink)
    return A.asinh_attention(q, k, v, n=n, c=c, sink_logit=sink)


CASES = [("softmax", 16, 1.0, False, None, None), ("softmax", 16, 1.0, True, None, None),
         ("asinh", 16, 16.0, True, True, False), ("asinh", 16, 16.0, True, True, True),
         ("asinh", 16, 16.0, True, False, True), ("asinh", 16, 16.0, False, False, True),
         ("asinh", 8, 1.0, True, True, False), ("asinh", 32, 8.0, False, True, True)]


@pytest.mark.parametrize("kind,n,c,with_sink,shift,fast", CASES)
@pytest.mark.parametrize("block", [128, 256])
def test_flash_matches_dense(kind, n, c, with_sink, shift, fast, block):
    from src.sota.kernels.flash import flash_attention
    B, H, T, D = 2, 3, 512, 64
    ks = jax.random.split(jax.random.PRNGKey(0), 5)
    q = jax.random.normal(ks[0], (B, H, T, D), jnp.float32) * 1.5
    k = jax.random.normal(ks[1], (B, H, T, D), jnp.float32) * 1.5
    v = jax.random.normal(ks[2], (B, H, T, D), jnp.float32)
    g = jax.random.normal(ks[3], (B, H, T, D), jnp.float32)
    sink = (jax.random.normal(ks[4], (H,)) * 0.5)[None, :, None, None] if with_sink else None

    def f_flash(q, k, v, s):
        return jnp.sum(flash_attention(q, k, v, s, kind=kind, n=n, c=c, block=block, shift=shift, fast_rho=fast) * g)

    def f_ref(q, k, v, s):
        return jnp.sum(reference(q, k, v, s, kind, n, c).astype(jnp.float32) * g)

    with jax.default_matmul_precision("highest"):
        o1 = flash_attention(q, k, v, sink, kind=kind, n=n, c=c, block=block, shift=shift, fast_rho=fast)
        o2 = reference(q, k, v, sink, kind, n, c)
        np.testing.assert_allclose(o1, o2, atol=2e-3, rtol=2e-3)
        args = (q, k, v, sink) if with_sink else (q, k, v, None)
        argnums = (0, 1, 2, 3) if with_sink else (0, 1, 2)
        g1 = jax.grad(f_flash, argnums=argnums)(*args)
        g2 = jax.grad(f_ref, argnums=argnums)(*args)
        for a, b in zip(g1, g2):
            scale = float(jnp.max(jnp.abs(b))) + 1e-6
            np.testing.assert_allclose(np.asarray(a) / scale, np.asarray(b) / scale, atol=3e-3)
