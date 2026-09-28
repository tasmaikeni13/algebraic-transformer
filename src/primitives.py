"""Algebraic activation and second-moment normalization.

Half precision inputs accumulate in float32. Outputs and cotangents retain the
input dtype. Inputs must be finite, and squared reductions must fit in the
accumulation dtype. A zero normalization epsilon requires a nonzero vector."""

from functools import partial

import jax
import jax.numpy as jnp


def _accumulate(x):
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError("expected a floating point array")
    return x.astype(jnp.promote_types(x.dtype, jnp.float32))


def _alu_forward(x):
    z = _accumulate(x)
    r = jax.lax.rsqrt(1.0 + z * z)
    u = z * r
    # Rationalize the negative tail: 1+u = r²/(1-u). Computing 1+u
    # directly loses the entire tail (or magnifies rounding error) for x << 0.
    # Keep the unselected denominator finite at positive, saturated u=1.
    denominator = jnp.where(z < 0, 1.0 - u, 1.0)
    negative = (0.5 * u * r) / denominator
    positive = 0.5 * z * (1.0 + u)
    return jnp.where(z < 0, negative, positive).astype(x.dtype), u


@jax.custom_vjp
def _alu(x):
    return _alu_forward(x)[0]


def _alu_backward(u, g):
    derivative = 0.5 + u * (1.0 - 0.5 * u * u)
    return ((g.astype(u.dtype) * derivative).astype(g.dtype),)


_alu.defvjp(_alu_forward, _alu_backward)


@jax.jit
def alu(x):
    """ALU with a single cached u and a multiplication-only cubic VJP."""
    return _alu(jnp.asarray(x))


def _avn_forward(x, eps):
    z = _accumulate(x)
    if z.ndim == 0 or z.shape[-1] == 0:
        raise ValueError("AVN needs a nonempty feature axis")
    if eps < 0:
        raise ValueError("eps must be nonnegative")
    inverse_width = 1.0 / z.shape[-1]
    tau = jax.lax.rsqrt(jnp.sum(z * z, axis=-1, keepdims=True) * inverse_width + eps)
    y = z * tau
    return y.astype(x.dtype), (y, tau)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _avn(x, eps):
    return _avn_forward(x, eps)[0]


def _avn_backward(eps, cache, g):
    del eps
    y, tau = cache
    upstream = g.astype(y.dtype)
    inverse_width = 1.0 / y.shape[-1]
    radial = jnp.sum(upstream * y, axis=-1, keepdims=True) * inverse_width
    return ((tau * (upstream - radial * y)).astype(g.dtype),)


_avn.defvjp(_avn_forward, _avn_backward)


@partial(jax.jit, static_argnames=("eps",))
def avn(x, eps=1e-5):
    """Parameter-free last-axis normalization with an analytical cached VJP."""
    return _avn(jnp.asarray(x), eps)
