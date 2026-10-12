"""QK RMS normalization with tiled learned-gain gradient reduction on TPU.

The forward and input VJP use the original JAX RMS formula. Only the gain
gradient is reduced in Pallas tiles, then summed in XLA. This retains the
original input-gradient operation order and avoids a large global gain
reduction. The real operator is unchanged; summation rounding can differ.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from src.sota.algebraic import rms_norm as _native_norm


def _gain_kernel(x_ref, do_ref, dg_ref, *, eps):
    x = x_ref[...].astype(jnp.float32)
    do = do_ref[...].astype(jnp.float32)
    tau = lax.rsqrt(jnp.mean(x*x, axis=-1, keepdims=True)+eps)
    partial_gain = jnp.sum(do*(x*tau), axis=0, keepdims=True)
    # Eight rows satisfy the native register/block layout; one is consumed.
    dg_ref[0] = jnp.broadcast_to(partial_gain, dg_ref.shape[1:])


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _norm(x, gain, eps, block):
    return _native_norm(x, gain, eps)


def _norm_fwd(x, gain, eps, block):
    return _native_norm(x, gain, eps), (x, gain)


def _norm_bwd(eps, block, res, do):
    x, gain = res
    # Preserve the native AD input gradient rather than reassociate its
    # radial subtraction. The unused native gain reduction is eliminated.
    _, pullback = jax.vjp(lambda x, g: _native_norm(x, g, eps), x, gain)
    dx, _ = pullback(do)
    D = x.shape[-1]
    xf, dof = x.reshape(-1, D), do.reshape(-1, D)
    N = xf.shape[0]
    dg = pl.pallas_call(
        partial(_gain_kernel, eps=eps), grid=(N//block,),
        in_specs=[pl.BlockSpec((block, D), lambda i: (i, 0)),
                  pl.BlockSpec((block, D), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((1, 8, D), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((N//block, 8, D), jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(xf, dof)
    dgain = jnp.sum(dg[:, 0, :], axis=0).astype(gain.dtype)
    return dx, dgain


_norm.defvjp(_norm_fwd, _norm_bwd)


def qk_rms_norm(x, gain, eps=1e-5, *, block=1024):
    """RMS-normalize D=64 rows with a learned gain; requires a TPU.

The measured model dispatch uses BF16, context 2048, and 1024-row blocks.
The explicit function also accepts float32 for independent numerical checks.
"""
    if x.ndim < 2 or x.shape[-1] != 64 or gain.shape != (64,):
        raise ValueError("QK RMS kernel needs D=64 and a one-dimensional gain")
    if x.dtype not in (jnp.bfloat16, jnp.float32) or gain.dtype != jnp.float32:
        raise ValueError("QK RMS kernel needs float32/BF16 input and float32 gain")
    if block < 8 or block % 8 or x.size//64 % block:
        raise ValueError("QK RMS row count must be divisible by a block aligned to eight rows")
    if eps <= 0:
        raise ValueError("QK RMS epsilon must be positive")
    return _norm(x, gain, float(eps), int(block))
