"""Numerical and dispatch checks for tiled QK gain-gradient reduction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algebraic import algebraic as A, config, model
from algebraic.kernels.qk_norm import qk_rms_norm

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="needs a TPU")


@pytest.fixture(autouse=True)
def float32_context():
    enable_x64 = jax.enable_x64 if hasattr(jax, "enable_x64") else jax.experimental.enable_x64
    with enable_x64(False):
        yield


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("block", [256, 1024])
@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_qk_norm_output_and_vjp(dtype, block, eps):
    keys = jax.random.split(jax.random.PRNGKey(181), 3)
    shape = (2, 3, 512, 64)
    x = (3*jax.random.normal(keys[0], shape, dtype)).at[0, 0, 0].set(0)
    gain = .25+4*jax.random.uniform(keys[1], (64,), jnp.float32)
    do = jax.random.normal(keys[2], shape, dtype)
    fn = lambda x, g: qk_rms_norm(x, g, eps, block=block)
    ref = lambda x, g: A.rms_norm(x, g, eps)
    oa, ob = jax.jit(fn)(x, gain), jax.jit(ref)(x, gain)
    np.testing.assert_allclose(np.asarray(oa, np.float32), np.asarray(ob, np.float32),
                               atol=2e-6 if dtype == jnp.float32 else .01,
                               rtol=2e-6 if dtype == jnp.float32 else .01)
    loss = lambda f, x, g: jnp.sum(f(x, g).astype(jnp.float32)*do.astype(jnp.float32))
    ga = jax.jit(jax.grad(lambda x, g: loss(fn, x, g), argnums=(0, 1)))(x, gain)
    gb = jax.jit(jax.grad(lambda x, g: loss(ref, x, g), argnums=(0, 1)))(x, gain)
    for a, b in zip(ga, gb):
        av, bv = np.asarray(a, np.float32), np.asarray(b, np.float32)
        assert np.isfinite(av).all()
        assert np.linalg.norm(av-bv)/max(np.linalg.norm(bv), 1e-6) < (
            2e-6 if dtype == jnp.float32 else .001)


@pytest.mark.parametrize("arch,impl,seq,dtype,expected", [
    ("algebraic", "flash", 2048, jnp.bfloat16, False),
    ("standard", "flash", 2048, jnp.bfloat16, False),
])
def test_model_dispatch(monkeypatch, arch, impl, seq, dtype, expected):
    import algebraic.kernels.qk_norm as module
    calls = []
    def tracked(x, gain, eps):
        calls.append(x.shape)
        return A.rms_norm(x, gain, eps)
    monkeypatch.setattr(module, "qk_rms_norm", tracked)
    cfg = config.build(arch, "125m", seq_len=seq, attn_impl=impl)
    x = jnp.ones((1, 12, seq, 64), dtype)
    model._qk_norm(x, jnp.ones((64,), jnp.float32), cfg)
    assert bool(calls) == expected
