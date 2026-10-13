"""Algebraic building blocks: arithmetic, comparisons, and ``lax.rsqrt`` only.

This module is covered by the source audit (``scripts/audit_algebraic.py``):
it must not call exponential, logarithmic, trigonometric, hyperbolic, or
sigmoid functions. See ``paper/theory.md`` for the identities used here.

``rho(x) = x + sqrt(1 + x^2)`` (= exp(asinh x)) satisfies
``rho(x) rho(y) = rho(x (+) y)`` with ``x (+) y = x sqrt(1+y^2) + y sqrt(1+x^2)``
and ``rho(-x) = 1 / rho(x)``. The asinh-power kernel ``rho(s / c)^n`` therefore
admits an exact max shift: ``(rho(x) rho(-m))^n <= 1`` for ``x <= m``, and
the shared factor ``rho(-m)^n`` cancels in every normalized weight.
"""

from functools import partial
import math
from typing import Optional, Tuple

import jax
from jax import lax
import jax.numpy as jnp

F32 = jnp.float32


def _acc(x: jax.Array) -> jax.Array:
    """Promote to at least float32 (bfloat16 -> float32, float64 stays)."""
    x = jnp.asarray(x)
    return x.astype(jnp.promote_types(x.dtype, F32))


def rho(x: jax.Array) -> jax.Array:
    """``x + sqrt(1 + x^2)`` without cancellation for negative ``x``."""
    rad = 1.0 + x * x
    t = jnp.abs(x) + rad * lax.rsqrt(rad)  # |x| + sqrt(1 + x^2) >= 1
    return jnp.where(x >= 0, t, 1.0 / t)


def int_pow(x: jax.Array, n: int) -> jax.Array:
    """``x**n`` for a static positive integer by binary exponentiation."""
    n = int(n)
    if n < 1:
        raise ValueError("n must be a positive integer")
    result = None
    base = x
    while n:
        if n & 1:
            result = base if result is None else result * base
        n >>= 1
        if n:
            base = base * base
    return result


def kernel_sink_logit(omega: float, n: int, c: float) -> float:
    """Sink logit ``b`` with ``rho(b / c)^n = omega`` (a configuration constant).

    Solves ``rho(y) = omega^(1/n)`` for ``y = (t - 1/t) / 2`` with
    ``t = omega^(1/n)``; ``n`` must be a power of two so the root is a chain
    of square roots.
    """
    if omega <= 0:
        raise ValueError("omega must be positive")
    t = float(omega)
    k = int(round(math.log2(n)))
    if 1 << k != n:
        raise ValueError("n must be a power of two")
    for _ in range(k):
        t = math.sqrt(t)
    return c * 0.5 * (t - 1.0 / t)


# ---------------------------------------------------------------------------
# Normalization and activation
# ---------------------------------------------------------------------------

def rms_norm(x: jax.Array, gain: Optional[jax.Array], eps: float = 1e-5) -> jax.Array:
    """RMS normalization; ``gain=None`` gives the parameter-free AVN."""
    xf = _acc(x)
    tau = lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    y = xf * tau
    if gain is not None:
        y = y * gain.astype(y.dtype)
    return y.astype(x.dtype)


def _alu_parts(x, beta):
    z = _acc(x)
    bz = beta * z
    r = lax.rsqrt(1.0 + bz * bz)
    u = bz * r
    # For bz < 0 use 1 + u = r^2 / (1 - u) so the negative tail survives.
    one_plus_u = jnp.where(bz < 0, (r * r) / (1.0 - u), 1.0 + u)
    return (0.5 * z * one_plus_u).astype(x.dtype), u, r


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def alu(x: jax.Array, beta: float = 1.0) -> jax.Array:
    """Algebraic linear unit ``x/2 (1 + bx / sqrt(1 + b^2 x^2))``.

    ``beta = 1/2`` matches the SiLU gate slope at the origin;
    ``beta ~ 0.8`` matches GELU's.
    """
    return _alu_parts(x, beta)[0]


def _alu_fwd(x, beta):
    y, u, r = _alu_parts(x, beta)
    return y, (u, r)


def _alu_bwd(beta, cache, g):
    u, r = cache
    derivative = 0.5 * (1.0 + u) + 0.5 * u * (r * r)
    return ((g.astype(derivative.dtype) * derivative).astype(g.dtype),)


alu.defvjp(_alu_fwd, _alu_bwd)


# ---------------------------------------------------------------------------
# Cayley rotation tables, cos/sin of shape (T, head_dim // 2)
# ---------------------------------------------------------------------------

def cayley_powers(w: jax.Array, seq_len: int) -> Tuple[jax.Array, jax.Array]:
    """Rows ``((1 + i w)^2 / (1 + w^2))^m`` for m < seq_len, per frequency.

    Binary exponentiation over the bits of m: every entry is a product of at
    most log2(seq_len) renormalized squarings.
    """
    w = jnp.asarray(w, F32)
    w2 = w * w
    c = (1.0 - w2) / (1.0 + w2)
    s = (2.0 * w) / (1.0 + w2)
    positions = jnp.arange(seq_len, dtype=jnp.int32)
    out_c = jnp.ones((seq_len, w.shape[0]), F32)
    out_s = jnp.zeros((seq_len, w.shape[0]), F32)
    bit = 0
    while (1 << bit) < seq_len:
        take = ((positions >> bit) & 1).astype(bool)[:, None]
        new_c = out_c * c - out_s * s
        new_s = out_c * s + out_s * c
        out_c = jnp.where(take, new_c, out_c)
        out_s = jnp.where(take, new_s, out_s)
        c, s = c * c - s * s, 2.0 * c * s
        inv = lax.rsqrt(c * c + s * s)
        c, s = c * inv, s * inv
        bit += 1
    inv = lax.rsqrt(out_c * out_c + out_s * out_s)
    return out_c * inv, out_s * inv


def cayley_table_geometric(head_dim: int, seq_len: int, base: float = 10000.0,
                           w0: float = 0.5, fraction: float = 1.0):
    """Geometric grid ``w_k = w0 base^(-k/half)``; ``base^(-1/half)`` by roots.

    ``fraction < 1`` keeps only the first ``fraction * half`` frequencies
    rotating (partial rotary); the rest get ``w = 0`` (identity rotation).
    """
    half = head_dim // 2
    levels = int(round(math.log2(half)))
    if 1 << levels != half:
        raise ValueError("geometric Cayley grid needs head_dim/2 to be a power of two")
    root = float(base)
    for _ in range(levels):
        root = math.sqrt(root)
    ratio = 1.0 / root
    active = int(round(fraction * half))
    w = [w0 * ratio ** i if i < active else 0.0 for i in range(half)]
    return cayley_powers(jnp.asarray(w, F32), seq_len)


def apply_rotary(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Rotate-half rotation of (B, H, T, D) by tables of shape (T, D/2)."""
    half = x.shape[-1] // 2
    t = x.shape[-2]
    c = cos[:t].astype(x.dtype)[None, None]
    s = sin[:t].astype(x.dtype)[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


# ---------------------------------------------------------------------------
# Attention weights
# ---------------------------------------------------------------------------

def asinh_power_weights(scores: jax.Array, mask: jax.Array, n: int, c: float,
                        sink_logit: Optional[jax.Array]) -> jax.Array:
    """Normalized ``rho(s/c)^n`` weights with max shift and optional sink.

    ``scores`` (..., Tq, Tk) float32; ``mask`` broadcastable bool;
    ``sink_logit`` broadcastable to (..., Tq, 1) or None.
    """
    x = scores / c
    m = jnp.max(jnp.where(mask, x, -1e30), axis=-1, keepdims=True)
    if sink_logit is not None:
        xb = sink_logit / c
        m = jnp.maximum(m, xb)
    m = lax.stop_gradient(m)
    shift = rho(-m)
    w = jnp.where(mask, int_pow(rho(x) * shift, n), 0.0)
    denom = jnp.sum(w, axis=-1, keepdims=True)
    if sink_logit is not None:
        denom = denom + int_pow(rho(xb) * shift, n)
    return w / denom


def asinh_attention(q, k, v, *, n: int, c: float, sink_logit=None):
    """Dense causal asinh-power attention on (B, H, T, D)."""
    d = q.shape[-1]
    t = q.shape[-2]
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=F32)
    scores = scores * (1.0 / math.sqrt(d))
    mask = jnp.tril(jnp.ones((t, t), bool))[None, None]
    p = asinh_power_weights(scores, mask, n, c, sink_logit)
    out = jnp.einsum("bhqk,bhkd->bhqd", p.astype(v.dtype), v, preferred_element_type=F32)
    return out.astype(q.dtype)


# ---------------------------------------------------------------------------
# Training heads on materialized logits (N, V)
# ---------------------------------------------------------------------------

def algebraic_probs(z: jax.Array, n: int, c: float) -> jax.Array:
    """``p_j ~ rho(z_j / c)^n`` over the last axis, max shifted."""
    x = z / c
    m = lax.stop_gradient(jnp.max(x, axis=-1, keepdims=True))
    w = int_pow(rho(x) * rho(-m), n)
    return w / jnp.sum(w, axis=-1, keepdims=True)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def algebraic_likelihood(logits: jax.Array, targets: jax.Array, n: int, c: float):
    """Maximum-likelihood head for ``p_j ~ rho(z_j / c)^n``.

    Returns ``(objective, p_target)``. The negative log-likelihood
    ``-n asinh(z_t / c) + log sum_j rho(z_j / c)^n`` has the algebraic logit
    gradient ``(n / c)(1 + (z_j / c)^2)^(-1/2)(p_j - y_j)``, which the custom
    VJP applies (divided by the number of tokens). The returned objective is
    the mean target-miss probability, an algebraic monotone stand-in used only
    as the primal value; the per-token target probability is telemetry and
    receives no gradient.
    """
    p = algebraic_probs(_acc(logits), n, c)
    pt = jnp.take_along_axis(p, targets[:, None], axis=-1)[:, 0]
    return jnp.mean(1.0 - pt), pt


def _alg_lik_fwd(logits, targets, n, c):
    z = _acc(logits)
    p = algebraic_probs(z, n, c)
    pt = jnp.take_along_axis(p, targets[:, None], axis=-1)[:, 0]
    return (jnp.mean(1.0 - pt), pt), (p, z, targets)


def _alg_lik_bwd(n, c, cache, g):
    p, z, targets = cache
    g_obj = g[0]
    x = z / c
    r = lax.rsqrt(1.0 + x * x)
    y = (jnp.arange(p.shape[-1])[None, :] == targets[:, None]).astype(p.dtype)
    grad = (g_obj * (n / c) / p.shape[0]) * r * (p - y)
    return grad, None


algebraic_likelihood.defvjp(_alg_lik_fwd, _alg_lik_bwd)


# ---------------------------------------------------------------------------
# Lean dense variant of the algebraic likelihood (no running maximum).
# Logits enter as x = min(z / c, X) with X chosen so that V (2X)^n fits in
# float32 (X ~ 30 for n = 16 and the GPT-2 vocabulary): the clamp never binds
# for realistic logits and its gradient is straight-through. rho is evaluated
# as x + sqrt(1 + x^2); cancellation for x << 0 only perturbs weights that are
# negligible after normalization.
# ---------------------------------------------------------------------------

def likelihood_clamp(vocab: int, n: int) -> float:
    out = 1e38 / vocab
    k = n
    while k > 1:
        out = math.sqrt(out)
        k //= 2
    return 0.25 * out


def _lean_parts(z, n, c, clamp):
    x = jnp.minimum(z / c, clamp)
    rad = 1.0 + x * x
    r = lax.rsqrt(rad)
    w = int_pow(x + rad * r, n)
    return w, r


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def algebraic_likelihood_lean(logits, targets, n: int, c: float):
    """Same objective and gradient as ``algebraic_likelihood``, fewer passes."""
    z = _acc(logits)
    w, _ = _lean_parts(z, n, c, likelihood_clamp(z.shape[-1], n))
    wt = jnp.take_along_axis(w, targets[:, None], axis=-1)[:, 0]
    pt = wt / jnp.sum(w, axis=-1)
    return jnp.mean(1.0 - pt), pt


def _lean_fwd(logits, targets, n, c):
    out = algebraic_likelihood_lean(logits, targets, n, c)
    return out, (logits, targets)


def _lean_bwd(n, c, res, g):
    logits, targets = res
    z = _acc(logits)
    w, r = _lean_parts(z, n, c, likelihood_clamp(z.shape[-1], n))
    p = w / jnp.sum(w, axis=-1, keepdims=True)
    y = (jnp.arange(z.shape[-1])[None, :] == targets[:, None]).astype(p.dtype)
    grad = (g[0] * (n / c) / z.shape[0]) * r * (p - y)
    return grad.astype(logits.dtype), None


algebraic_likelihood_lean.defvjp(_lean_fwd, _lean_bwd)
