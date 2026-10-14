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

from algebraic import algebraic as A
from algebraic import standard as S

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="needs a TPU")


@pytest.fixture(autouse=True)
def float32_pallas_context():
    # The component suite enables x64 for mathematical oracles. TPU Mosaic
    # cannot lower the resulting float64 literal casts, even for float32 data.
    enable_x64 = jax.enable_x64 if hasattr(jax, "enable_x64") else jax.experimental.enable_x64
    with enable_x64(False):
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
    from algebraic.kernels.flash import flash_attention
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
    from algebraic.kernels.flash import flash_attention
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
    from algebraic.kernels.flash import flash_attention
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
    from algebraic.kernels.flash import flash_attention
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


@pytest.mark.parametrize("kind,mode", [("softmax", "sink"), ("softmax", "no_sink"),
                                      ("asinh", "inverse"), ("asinh", "root"),
                                      ("asinh", "shifted"), ("asinh", "guarded")])
@pytest.mark.parametrize("seq", [512, 2048])
@pytest.mark.parametrize("diagonal", [False, True])
def test_resident_backward_matches_dense(kind, mode, seq, diagonal):
    """Independent dense VJPs for forced resident and automatic model dispatch."""
    from algebraic.kernels.flash import flash_attention
    shape = (1, 2, seq, 64)
    keys = jax.random.split(jax.random.PRNGKey(67), 5)
    q, k, v, g = [jax.random.normal(key, shape, jnp.bfloat16) for key in keys[:4]]
    sink = jax.random.normal(keys[4], (1, 2, 1, 1), jnp.float32) if mode != "no_sink" else None
    opts = dict(kind=kind, n=16, c=8., block=512, compact_stat=mode != "root",
                resident_bwd=True, diagonal_bwd=diagonal, shift=mode == "shifted", inv_stat=mode != "root" and mode != "shifted",
                range_guard=mode == "guarded")
    f = lambda q, k, v, s: flash_attention(q, k, v, s, **opts)
    ref = lambda q, k, v, s: reference(q, k, v, s, kind, 16, 8.)
    with jax.default_matmul_precision("default"):
        got = jax.jit(f)(q, k, v, sink)
        want = jax.jit(ref)(q, k, v, sink)
        np.testing.assert_allclose(np.asarray(got, dtype=np.float32),
                                   np.asarray(want, dtype=np.float32), atol=.012, rtol=.012)
        loss = lambda fn, q, k, v, s: jnp.sum(fn(q, k, v, s).astype(jnp.float32) * g.astype(jnp.float32))
        argnums = (0, 1, 2, 3) if sink is not None else (0, 1, 2)
        ga = jax.jit(jax.grad(partial(loss, f), argnums=argnums))(q, k, v, sink)
        gb = jax.jit(jax.grad(partial(loss, ref), argnums=argnums))(q, k, v, sink)
        for a, b in zip(ga, gb):
            a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
            assert np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-6) < .02
        if seq == 2048 and mode == "guarded":
            auto = lambda q, k, v, s: flash_attention(q, k, v, s, **{**opts, "resident_bwd": None})
            gc = jax.jit(jax.grad(partial(loss, auto), argnums=argnums))(q, k, v, sink)
            for a, b in zip(ga, gc):
                np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("seq", [512, 2048])
@pytest.mark.parametrize("mode", ["ordinary", "negative_no_sink"])
def test_diagonal_forward_matches_dense(dtype, seq, mode):
    """FP32 denominator sums remain independent of BF16 matrix-operand rounding."""
    from algebraic.kernels.flash import flash_attention
    shape = (1, 2, seq, 64)
    keys = jax.random.split(jax.random.PRNGKey(439), 4)
    q, k, v, g = [jax.random.normal(key, shape, dtype) for key in keys]
    sink = jnp.array([-.4, .6], jnp.float32)[None, :, None, None]
    if mode == "negative_no_sink":
        q = (1 + .05 * q.astype(jnp.float32)).astype(dtype)
        k = (-1 + .05 * k.astype(jnp.float32)).astype(dtype)
        sink = None
    fn = lambda q, k, v, s: flash_attention(q, k, v, s, kind="asinh", n=16, c=8.,
        block=512, range_guard=True, diagonal_fwd=True,
        resident_bwd=dtype == jnp.bfloat16, diagonal_bwd=dtype == jnp.bfloat16)
    ref = lambda q, k, v, s: A.asinh_attention(q, k, v, n=16, c=8., sink_logit=s)
    argnums = (0, 1, 2, 3) if sink is not None else (0, 1, 2)
    with jax.default_matmul_precision("highest" if dtype == jnp.float32 else "default"):
        got = jax.jit(fn)(q, k, v, sink)
        expected = jax.jit(ref)(q, k, v, sink)
        np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(expected, np.float32),
                                   atol=.015 if dtype == jnp.bfloat16 else .002,
                                   rtol=.015 if dtype == jnp.bfloat16 else .002)
        def loss(fn, q, k, v, s):
            return jnp.sum(fn(q, k, v, s).astype(jnp.float32) * g.astype(jnp.float32))
        gradients = jax.jit(jax.grad(partial(loss, fn), argnums=argnums))(q, k, v, sink)
        reference_gradients = jax.jit(jax.grad(partial(loss, ref), argnums=argnums))(q, k, v, sink)
        # Nearly common negative scores make dQ sensitive to the BF16 weight
        # rounding already present in the original kernel. Check the complete
        # VJP against the independent oracle and preserve each original VJP.
        negative_bf16 = mode == "negative_no_sink" and dtype == jnp.bfloat16
        baseline_gradients = None
        if negative_bf16:
            baseline = lambda q, k, v, s: flash_attention(q, k, v, s, kind="asinh", n=16, c=8.,
                block=512, range_guard=True, resident_bwd=True, diagonal_bwd=False, diagonal_fwd=False)
            baseline_gradients = jax.jit(jax.grad(partial(loss, baseline), argnums=argnums))(q, k, v, sink)
        squared_error, squared_reference = 0., 0.
        for a, b in zip(gradients, reference_gradients):
            a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
            assert np.isfinite(a).all()
            squared_error += float(np.sum((a-b)**2))
            squared_reference += float(np.sum(b**2))
            error = np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-6)
            if not negative_bf16:
                assert error < (.025 if dtype == jnp.bfloat16 else .005)
        assert math.sqrt(squared_error/max(squared_reference, 1e-12)) < (
            .025 if dtype == jnp.bfloat16 else .005)
        if baseline_gradients is not None:
            for a, b in zip(gradients, baseline_gradients):
                a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
                assert np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-6) < .003


@pytest.mark.parametrize("score_sign", [-1., 1.])
@pytest.mark.parametrize("with_sink", [False, True])
def test_resident_guarded_tail(score_sign, with_sink):
    from algebraic.kernels.flash import flash_attention
    shape = (1, 1, 512, 64)
    q = jnp.full(shape, math.sqrt(1000), jnp.bfloat16)
    k = q * score_sign
    v = jax.random.normal(jax.random.PRNGKey(69), shape, jnp.bfloat16)
    sink = jnp.zeros((1,), jnp.float32) if with_sink else None
    fn = lambda q, k, v: flash_attention(q, k, v, sink, kind="asinh", n=16, c=8.,
                                         block=128, compact_stat=True, range_guard=True,
                                         resident_bwd=True)
    with jax.default_matmul_precision("default"):
        out = jax.jit(fn)(q, k, v)
        want = A.asinh_attention(q, k, v, n=16, c=8., sink_logit=sink)
        assert np.isfinite(np.asarray(out, dtype=np.float32)).all()
        np.testing.assert_allclose(np.asarray(out, dtype=np.float32),
                                   np.asarray(want, dtype=np.float32), atol=.012, rtol=.012)
        grads = jax.jit(jax.grad(lambda q, k, v: fn(q, k, v).sum(), argnums=(0, 1, 2)))(q, k, v)
        assert all(np.isfinite(np.asarray(x, dtype=np.float32)).all() for x in grads)


@pytest.mark.parametrize("kind,mode", [("asinh", "sink"), ("asinh", "no_sink"),
                                      ("asinh", "negative"), ("asinh", "gain_two"),
                                      ("asinh", "positive_tail"), ("asinh", "negative_tail"),
                                      ("softmax", "sink"), ("softmax", "no_sink")])
def test_whole_head_residency_matches_dense(kind, mode):
    """Independent outputs and all VJPs for whole-head dispatch and tail fallback."""
    from algebraic.kernels.flash import flash_attention
    shape = (1, 2, 2048, 64)
    keys = jax.random.split(jax.random.PRNGKey(563), 4)
    q, k, v, g = [jax.random.normal(key, shape, jnp.bfloat16) for key in keys]
    sink = jnp.array([-.35, .125], jnp.float32)[None, :, None, None]
    if mode in ("no_sink", "negative"):
        sink = None
    if mode == "negative":
        q = (1 + .05*q.astype(jnp.float32)).astype(q.dtype)
        k = (-1 + .05*k.astype(jnp.float32)).astype(k.dtype)
    if mode == "gain_two":
        q, k = 2*A.rms_norm(q, None), 2*A.rms_norm(k, None)
    if mode.endswith("tail"):
        q = jnp.full(shape, math.sqrt(1000), jnp.bfloat16)
        k = q if mode == "positive_tail" else -q
    fn = lambda q,k,v,s: flash_attention(q,k,v,s,kind=kind,n=16,c=8.,
        range_guard=kind == "asinh", head_resident=True)
    ref = lambda q,k,v,s: reference(q,k,v,s,kind,16,8.)
    if mode.endswith("tail"):
        # Autodiff through BF16 probability operands gives a rounding residue
        # in the exactly centered, constant-score dQ. Use FP32 contractions
        # as the mathematical oracle instead of treating that residue as truth.
        ref = lambda q,k,v,s: reference(q.astype(jnp.float32),k.astype(jnp.float32),
                                        v.astype(jnp.float32),s,kind,16,8.)
    argnums = (0,1,2,3) if sink is not None else (0,1,2)
    got, want = [jax.jit(f)(q,k,v,sink) for f in (fn,ref)]
    np.testing.assert_allclose(np.asarray(got,np.float32),np.asarray(want,np.float32),atol=.015,rtol=.015)
    loss = lambda f,q,k,v,s: jnp.sum(f(q,k,v,s).astype(jnp.float32)*g.astype(jnp.float32))
    gradients = [jax.jit(jax.grad(partial(loss,fn),argnums=argnums))(q,k,v,sink)]
    with jax.default_matmul_precision("highest" if mode.endswith("tail") else "default"):
        gradients.append(jax.jit(jax.grad(partial(loss,ref),argnums=argnums))(q,k,v,sink))
    error2, norm2 = 0., 0.
    for index,(a,b) in enumerate(zip(*gradients)):
        a,b = np.asarray(a,np.float32),np.asarray(b,np.float32)
        assert np.isfinite(a).all()
        error2 += float(np.sum((a-b)**2))
        norm2 += float(np.sum(b**2))
        if mode == "positive_tail" and index == 0:
            # Equal keys give zero dQ up to the vanishing sink mass. Its
            # remaining error comes from rounding the centered score VJP
            # to BF16 before the MXU contraction. Bound that rounding by
            # u*sum(abs(ds)), using independent float64 value/cotangent dots.
            va,ga = np.asarray(v,np.float64),np.asarray(g,np.float64)
            count = np.arange(1,shape[-2]+1,dtype=np.float64)
            ideal_output = np.cumsum(va,axis=-2)/count[None,None,:,None]
            centered = np.sum(ga*ideal_output,axis=-1)
            dp = ga@np.swapaxes(va,-1,-2)
            visible = np.tril(np.ones((shape[-2],shape[-2]),bool))
            mean_abs = np.sum(np.where(visible,np.abs(dp-centered[...,None]),0.),axis=-1)/count
            kval = float(k[0,0,0,0])
            x = shape[-1]*kval*kval/(8*math.sqrt(shape[-1]))
            factor = (16/(8*math.sqrt(shape[-1])))*abs(kval)/math.sqrt(1+x*x)
            # Include FP32 summation and final BF16 result rounding. This is
            # a case-specific error bound, not a universal machine proof.
            roundoff = (2**-8+shape[-2]*2**-24)*(1+2**-8)
            bound = factor*roundoff*mean_abs+2e-6
            assert np.all(np.abs(a-b) <= bound[...,None])
        elif mode != "negative":
            assert np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-6) < .025
    assert math.sqrt(error2/max(norm2,1e-12)) < .025
    if kind == "asinh":
        auto = lambda q,k,v,s: flash_attention(q,k,v,s,kind=kind,n=16,c=8.,range_guard=True)
        automatic = jax.jit(jax.grad(partial(loss,auto),argnums=argnums))(q,k,v,sink)
        for a,b in zip(gradients[0],automatic):
            np.testing.assert_array_equal(a,b)
        if mode.endswith("tail"):
            previous = lambda q,k,v,s: flash_attention(q,k,v,s,kind=kind,n=16,c=8.,
                                                       range_guard=True,head_resident=False)
            old_gradients = jax.jit(jax.grad(partial(loss,previous),argnums=argnums))(q,k,v,sink)
            for a,b in zip(gradients[0],old_gradients):
                np.testing.assert_array_equal(a,b)


def test_whole_head_residency_rejects_unsupported_shape():
    from algebraic.kernels.flash import flash_attention
    x = jnp.ones((1,1,512,64),jnp.bfloat16)
    with pytest.raises(ValueError,match="whole-head residency"):
        flash_attention(x,x,x,kind="asinh",c=8.,head_resident=True)
