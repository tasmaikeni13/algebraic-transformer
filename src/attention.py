"""Octic attention using rational operations and reciprocal square roots.

Last-axis normalization, FP32 accumulation for half inputs, input/output dtype
preservation. Finite squared reductions and representable octic powers are
required; AVN bounds coordinates by sqrt(length). eps and sink_omega are static.
The entrywise Jacobian bound of 2 applies to normalized scores, not raw scores
or the spectral norm. No uniform quantization advantage is implied.
"""
from functools import partial
from typing import NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from src.primitives import _accumulate, _avn_forward, _avn_backward


def _kernel(z):
    r = jax.lax.rsqrt(1.0 + z * z)
    u = z * r
    # Conjugate identity avoids cancellation when z is negative.
    denominator = jnp.where(z < 0, 1.0 - u, 1.0)
    rho = jnp.where(z < 0, r / denominator, z + (1.0 + z * z) * r)
    k2 = rho * rho
    k4 = k2 * k2
    return k4 * k4, r


def _octic_forward(x):
    k, r = _kernel(_accumulate(x))
    return k.astype(x.dtype), (k, r)


@jax.custom_vjp
def _octic(x):
    return _octic_forward(x)[0]


def _octic_backward(cache, g):
    k, r = cache
    return ((8.0 * r * k * g.astype(k.dtype)).astype(g.dtype),)


_octic.defvjp(_octic_forward, _octic_backward)


@jax.jit
def octic_kernel(x):
    """rho(x)^8 via exactly three squarings after evaluating rho."""
    return _octic(jnp.asarray(x))


def _normalized_forward(z, sink_omega):
    k, r = _kernel(z)
    p = k / (jnp.sum(k, axis=-1, keepdims=True) + sink_omega)
    return p, r


def _attention_forward(x, sink_omega, eps):
    if not 0 <= sink_omega < float("inf"):
        raise ValueError("sink_omega must be finite and nonnegative")
    if not 0 < eps < float("inf"):
        raise ValueError("eps must be finite and positive")
    z = _accumulate(x)
    normalized, (_, tau) = _avn_forward(z, eps)
    p, r = _normalized_forward(normalized, sink_omega)
    return p.astype(x.dtype), (normalized, tau, p, r)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _attention(x, sink_omega, eps):
    return _attention_forward(x, sink_omega, eps)[0]


def _attention_backward(sink_omega, eps, cache, g):
    del sink_omega
    normalized, tau, p, r = cache
    upstream = g.astype(p.dtype)
    dot = jnp.sum(p * upstream, axis=-1, keepdims=True)
    normalized_g = 8.0 * r * p * (upstream - dot)
    dx = _avn_backward(eps, (normalized, tau), normalized_g)[0]
    return (dx.astype(g.dtype),)


_attention.defvjp(_attention_forward, _attention_backward)


@partial(jax.jit, static_argnames=("sink_omega", "eps"))
def algebraic_softmax(scores, sink_omega=0.5, eps=1e-5):
    """AVN-bounded octic attention; omitted mass belongs to a rational sink."""
    return _attention(jnp.asarray(scores), sink_omega, eps)


class CayleyRotary(NamedTuple):
    """Precomputed Cayley rotary rotation parameters.

    Can be unpacked as (c, s), or accessed via .c, .s, or .matrix.
    """
    c: jax.Array
    s: jax.Array

    @property
    def matrix(self):
        """Full 2x2 orthogonal rotation matrix blocks in SO(2)."""
        row0 = jnp.stack([self.c, -self.s], axis=-1)
        row1 = jnp.stack([self.s, self.c], axis=-1)
        return jnp.stack([row0, row1], axis=-2)


def _extract_cs(rotary_params):
    if isinstance(rotary_params, (tuple, list)) and len(rotary_params) >= 2:
        return rotary_params[0], rotary_params[1]
    elif hasattr(rotary_params, "c") and hasattr(rotary_params, "s"):
        return rotary_params.c, rotary_params.s
    elif hasattr(rotary_params, "shape"):
        if rotary_params.ndim >= 4 and rotary_params.shape[-2:] == (2, 2):
            return rotary_params[..., 0, 0], rotary_params[..., 1, 0]
        elif rotary_params.ndim >= 3 and rotary_params.shape[-1] == 2:
            return rotary_params[..., 0], rotary_params[..., 1]
    raise TypeError(f"Unsupported rotary parameters type: {type(rotary_params)}")


def _align_param(param, q_shape, seq_axis=None):
    ndim = len(q_shape)
    num_pairs = q_shape[-1] // 2
    if seq_axis is None:
        if ndim == 2:
            seq_axis = 0
        elif ndim == 3:
            seq_axis = 1
        elif ndim == 4:
            seq_axis = 1 if q_shape[1] <= param.shape[0] and q_shape[1] > q_shape[2] else 2 if q_shape[2] <= param.shape[0] else 1
        else:
            seq_axis = ndim - 2
    seq_len = q_shape[seq_axis]
    sliced = param[:seq_len]
    bcast_shape = [1] * (ndim - 1) + [num_pairs]
    bcast_shape[seq_axis] = seq_len
    return sliced.reshape(bcast_shape)


def _rotate_tensor(x, c, s):
    dim = x.shape[-1]
    num_pairs = dim // 2
    x_pairs = x.reshape(x.shape[:-1] + (num_pairs, 2))
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]
    c_cast = c.astype(x.dtype)
    s_cast = s.astype(x.dtype)
    x0_rot = c_cast * x0 - s_cast * x1
    x1_rot = s_cast * x0 + c_cast * x1
    return jnp.stack([x0_rot, x1_rot], axis=-1).reshape(x.shape)


@partial(jax.jit, static_argnames=("dim", "max_seq_len", "dtype"))
def build_cayley_rotary_matrix(dim: int, max_seq_len: int, freqs=None, base: float = 10000.0, dtype=jnp.float32):
    """Precomputes rational Cayley rotation parameters without transcendentals.

    Args:
        dim: Feature dimension per head (must be positive and even).
        max_seq_len: Maximum sequence context length (positive integer).
        freqs: Optional explicit rational frequency array w_k in (0, 1].
        base: Frequency scaling base parameter for rational decay.
        dtype: Numerical precision dtype (float32 or float64).

    Returns:
        CayleyRotary namedtuple containing (c, s) tables of shape (max_seq_len, dim // 2).
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError("dim must be a positive even integer")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer")

    calc_dtype = jnp.float32 if dtype == jnp.bfloat16 else dtype
    num_pairs = dim // 2
    if freqs is not None:
        w = jnp.asarray(freqs, dtype=calc_dtype)
        if w.shape != (num_pairs,):
            raise ValueError(f"freqs must have shape ({num_pairs},), got {w.shape}")
    else:
        k = jnp.arange(num_pairs, dtype=calc_dtype)
        base_arr = jnp.asarray(base, dtype=calc_dtype)
        sqrt_base = base_arr * jax.lax.rsqrt(base_arr)
        denom_k = jnp.maximum(jnp.asarray(num_pairs - 1, dtype=calc_dtype), 1.0)
        alpha = (sqrt_base - 1.0) / denom_k
        lin = 1.0 + alpha * k
        w = 1.0 / (lin * lin)

    w_sq = w * w
    c_base = (1.0 - w_sq) / (1.0 + w_sq)
    s_base = (2.0 * w) / (1.0 + w_sq)

    def scan_step(carry, _):
        c_prev, s_prev = carry
        c_next = c_base * c_prev - s_base * s_prev
        s_next = s_base * c_prev + c_base * s_prev
        r = jax.lax.rsqrt(c_next * c_next + s_next * s_next)
        c_norm = c_next * r
        s_norm = s_next * r
        return (c_norm, s_norm), (c_prev, s_prev)

    init = (jnp.ones(num_pairs, dtype=calc_dtype), jnp.zeros(num_pairs, dtype=calc_dtype))
    _, (c_table, s_table) = jax.lax.scan(scan_step, init, None, length=max_seq_len)
    return CayleyRotary(c_table.astype(dtype), s_table.astype(dtype))


def apply_ago_rotations(q, k, rotary_params=None, seq_axis=None):
    """Applies AGO rational Cayley rotations to query and key tensors.

    Executes natively across TPU v4 VMU vector registers via 4 FMAs per channel pair.

    Args:
        q: Query tensor with shape (..., dim) where dim is even.
        k: Key tensor with shape matching q.
        rotary_params: Optional precomputed CayleyRotary or (c, s) tuple.
        seq_axis: Optional explicit sequence length axis index.

    Returns:
        Rotated (q_rot, k_rot) with preserved shapes and input dtypes.
    """
    if q.shape != k.shape:
        raise ValueError(f"q and k shapes must match, got {q.shape} and {k.shape}")
    dim = q.shape[-1]
    if dim % 2 != 0:
        raise ValueError(f"Feature dimension must be even, got {dim}")

    if rotary_params is None:
        ndim = len(q.shape)
        if seq_axis is None:
            seq_axis = 0 if ndim == 2 else 1 if ndim == 3 else (1 if q.shape[1] > q.shape[2] else 2)
        seq_len = q.shape[seq_axis]
        rotary_params = build_cayley_rotary_matrix(dim, seq_len, dtype=q.dtype)

    c_raw, s_raw = _extract_cs(rotary_params)
    c = _align_param(c_raw, q.shape, seq_axis=seq_axis)
    s = _align_param(s_raw, q.shape, seq_axis=seq_axis)

    q_rot = _rotate_tensor(q, c, s)
    k_rot = _rotate_tensor(k, c, s)
    return q_rot.astype(q.dtype), k_rot.astype(k.dtype)

