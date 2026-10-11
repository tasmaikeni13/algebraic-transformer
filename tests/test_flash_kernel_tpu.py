"""TPU check: Pallas flash attention against the dense references.

Run on a TPU host:  python -m pytest -q tests/test_flash_kernel_tpu.py
(skipped on other platforms).
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.sota import algebraic as A
from src.sota import standard as S

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="needs a TPU")


@pytest.fixture(autouse=True)
def float32_pallas_context():
    # The component suite enables x64 for mathematical oracles. TPU Mosaic
    # cannot lower the resulting float64 literal casts, even for float32 data.
    with jax.experimental.enable_x64(False):
        yield


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


@pytest.mark.parametrize("mode", ["root", "inverse", "compact", "guarded", "polynomial"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("blocks", [(128, 128), (128, 256), (256, 128)])
def test_model_attention_statistics(mode, dtype, blocks):
    """Actual n=16,c=8, including the learned sink's VJP and rectangular tiles."""
    from src.sota.kernels.flash import flash_attention
    shape = (1, 2, 512, 64)
    keys = jax.random.split(jax.random.PRNGKey(13), 5)
    q, k, v, g = (jax.random.normal(key, shape, dtype) for key in keys[:4])
    sink = jax.random.normal(keys[4], (1, 2, 1, 1), jnp.float32)
    bq, bk = blocks
    f = lambda q, k, v, s: flash_attention(q, k, v, s, kind="asinh", n=16, c=8.,
                                          block_q=bq, block_k=bk, shift=False,
                                          fast_rho=True, inv_stat=mode != "root",
                                          compact_stat=mode in ("compact", "guarded"),
                                          range_guard=mode == "guarded", poly_rho=mode == "polynomial")
    ref = lambda q, k, v, s: A.asinh_attention(q, k, v, n=16, c=8., sink_logit=s)
    with jax.default_matmul_precision("highest" if dtype == jnp.float32 else "default"):
        output = jax.jit(f)(q, k, v, sink)
        expected = jax.jit(ref)(q, k, v, sink)
        np.testing.assert_allclose(np.asarray(output, dtype=np.float32),
                                   np.asarray(expected, dtype=np.float32), atol=.012, rtol=.012)
        loss = lambda fn, q, k, v, s: jnp.sum(fn(q, k, v, s).astype(jnp.float32) * g.astype(jnp.float32))
        got = jax.jit(jax.grad(partial(loss, f), argnums=(0, 1, 2, 3)))(q, k, v, sink)
        want = jax.jit(jax.grad(partial(loss, ref), argnums=(0, 1, 2, 3)))(q, k, v, sink)
        for a, b in zip(got, want):
            a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
            rel_l2 = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-6)
            assert rel_l2 < (.02 if dtype == jnp.bfloat16 else .005)


@pytest.mark.parametrize("score_sign", [-1., 1.])
@pytest.mark.parametrize("with_sink", [False, True])
@pytest.mark.parametrize("magnitude", [1000., 1e7])
def test_guarded_attention_tail(score_sign, with_sink, magnitude):
    """Guarded fallback handles both weight overflow and whole-row underflow."""
    from src.sota.kernels.flash import flash_attention
    shape = (1, 1, 256, 64)
    q = jnp.full(shape, math.sqrt(magnitude), jnp.float32)
    k = q * score_sign
    v = jax.random.normal(jax.random.PRNGKey(29), shape, jnp.float32)
    sink = jnp.zeros((1,), jnp.float32) if with_sink else None
    fn = lambda q, k, v: flash_attention(q, k, v, sink, kind="asinh", n=16, c=8.,
                                         block=128, compact_stat=True, range_guard=True)
    with jax.default_matmul_precision("highest"):
        out = jax.jit(fn)(q, k, v)
        ref = A.asinh_attention(q, k, v, n=16, c=8., sink_logit=sink)
        assert np.isfinite(np.asarray(out)).all()
        np.testing.assert_allclose(out, ref, atol=2e-3, rtol=2e-3)
        grad = jax.jit(jax.grad(lambda q, k, v: fn(q, k, v).sum(), argnums=(0, 1, 2)))(q, k, v)
        assert all(np.isfinite(np.asarray(x)).all() for x in grad)


def test_explicit_clipping_vjp_is_zero_above_cap():
    """A saturated score or sink is constant in the clipped primal."""
    from src.sota.kernels.flash import flash_attention
    shape = (1, 1, 256, 64)
    q = jnp.full(shape, math.sqrt(1000), jnp.float32)
    k = q
    v = jax.random.normal(jax.random.PRNGKey(39), shape, jnp.float32)
    g = jax.random.normal(jax.random.PRNGKey(40), shape, jnp.float32)
    sink = jnp.full((1,), 1000., jnp.float32)
    def loss(q, k, s):
        o = flash_attention(q, k, v, s, kind="asinh", n=16, c=8.,
                            block=128, shift=False, range_guard=False)
        return jnp.sum(o * g)
    with jax.default_matmul_precision("highest"):
        grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, sink)
        for grad in grads:
            np.testing.assert_array_equal(grad, jnp.zeros_like(grad))
