"""Pallas TPU fused vocabulary projection and likelihood loss.

Both training heads share one kernel structure:

* ``ce``: softmax cross-entropy. Per-token statistic ``lse = m + log l``.
* ``alg_ce``: algebraic likelihood ``p_j ~ rho(z_j / c)^n``. The logit
  gradient of ``-log p_t`` is ``(n/c) (1 + (z_j/c)^2)^(-1/2) (p_j - y_j)``,
  which is algebraic, so the head trains on exact maximum likelihood without
  evaluating a logarithm. Per-token statistic ``sigma = Z^(-1/n)``.
  Accumulation is additive with the kernel argument clamped above at
  ``X`` such that ``V (2X)^n`` fits in float32 (X ~ 31 for n = 16 and the
  32,768 vocabulary, i.e. logits up to 31 c); the clamp never binds for
  realistic logits and its gradient is straight-through.

Forward: grid (token blocks, vocabulary blocks), vocabulary innermost; the
logit tile ``h E^T`` lives only in VMEM. Backward: the same grid recomputes
each tile, writes the scaled logit gradient in bfloat16 (used by one XLA
matmul for the embedding gradient) and accumulates the hidden-state
gradient in VMEM. The vocabulary is padded to a tile multiple; padded
columns are masked in the last tile only.
"""

from functools import partial
import math

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32 = jnp.float32
LANES = 128
TRANS_B = (((1,), (1,)), ((), ()))
TRANS_A = (((0,), (0,)), ((), ()))
NEG = -1e30


def _root(value, n):
    out = float(value)
    while n > 1:
        out = math.sqrt(out)
        n //= 2
    return out


def _ipow(x, n):
    out = None
    base = x
    while n:
        if n & 1:
            out = base if out is None else out * base
        n >>= 1
        if n:
            base = base * base
    return out


def _inv_root(l, n):
    y = lax.rsqrt(l)
    k = n // 2
    while k > 1:
        y = y * lax.rsqrt(y)
        k //= 2
    return y


def _params(sem=("parallel", "arbitrary")):
    return pltpu.CompilerParams(dimension_semantics=sem, vmem_limit_bytes=100 * 2 ** 20)


def _fwd_kernel(h_ref, e_ref, t_ref, stat_ref, zt_ref, m_sc, l_sc, zt_sc, *,
                kind, n, inv_c, clamp, vocab, bv, nv):
    iv = pl.program_id(1)

    @pl.when(iv == 0)
    def _init():
        m_sc[...] = jnp.full(m_sc.shape, NEG, F32)
        l_sc[...] = jnp.zeros(l_sc.shape, F32)
        zt_sc[...] = jnp.zeros(zt_sc.shape, F32)

    def step(last):
        z = lax.dot_general(h_ref[...], e_ref[...], TRANS_B, preferred_element_type=F32)
        bn = z.shape[0]
        col = iv * bv + lax.broadcasted_iota(jnp.int32, z.shape, 1)
        tgt = jnp.expand_dims(t_ref[0, 0], -1)
        zt_sc[...] += jnp.sum(jnp.where(col == tgt, z, 0.0), axis=1, keepdims=True)
        valid = col < vocab if last else None
        if kind == "ce":
            if last:
                z = jnp.where(valid, z, NEG)
            m_prev = m_sc[...]
            m_new = jnp.maximum(m_prev, jnp.max(z, axis=1, keepdims=True))
            p = jnp.exp(z - m_new[:, :1])
            l_sc[...] = jnp.exp(m_prev - m_new) * l_sc[...] + jnp.sum(p, axis=1, keepdims=True)
            m_sc[...] = m_new
        else:
            x = jnp.minimum(z * inv_c, clamp)
            rad = 1.0 + x * x
            w = _ipow(x + rad * lax.rsqrt(rad), n)
            if last:
                w = jnp.where(valid, w, 0.0)
            l_sc[...] += jnp.sum(w, axis=1, keepdims=True)
        del bn

    @pl.when(iv < nv - 1)
    def _mid():
        step(False)

    @pl.when(iv == nv - 1)
    def _last():
        step(True)
        if kind == "ce":
            stat_ref[...] = m_sc[...] + jnp.log(l_sc[...])
        else:
            stat_ref[...] = _inv_root(l_sc[...], n)
        zt_ref[...] = zt_sc[...]


def _bwd_kernel(scale_ref, h_ref, e_ref, t_ref, stat_ref, dz_ref, dh_ref, dh_sc, *,
                kind, n, inv_c, clamp, vocab, bv, nv):
    iv = pl.program_id(1)

    @pl.when(iv == 0)
    def _init():
        dh_sc[...] = jnp.zeros(dh_sc.shape, F32)

    def step(last):
        e = e_ref[...]
        z = lax.dot_general(h_ref[...], e, TRANS_B, preferred_element_type=F32)
        col = iv * bv + lax.broadcasted_iota(jnp.int32, z.shape, 1)
        tgt = jnp.expand_dims(t_ref[0, 0], -1)
        stat = jnp.expand_dims(stat_ref[0, 0], -1)
        y = (col == tgt).astype(F32)
        if kind == "ce":
            dz = jnp.exp(z - stat) - y
        else:
            x = jnp.minimum(z * inv_c, clamp)
            rad = 1.0 + x * x
            r = lax.rsqrt(rad)
            p = _ipow((x + rad * r) * stat, n)
            dz = (p - y) * r
        if last:
            dz = jnp.where(col < vocab, dz, 0.0)
        dzb = (dz * scale_ref[0]).astype(dz_ref.dtype)
        dz_ref[...] = dzb
        dh_sc[...] += lax.dot(dzb, e, preferred_element_type=F32)

    @pl.when(iv < nv - 1)
    def _mid():
        step(False)

    @pl.when(iv == nv - 1)
    def _last():
        step(True)
        dh_ref[...] = dh_sc[...].astype(dh_ref.dtype)


def _setup(h, embed, block_n, block_v):
    N, d = h.shape
    V = embed.shape[0]
    bn = min(block_n, N)
    if N % bn:
        raise ValueError(f"token count {N} must be divisible by block {bn}")
    bv = block_v
    nv = -(-V // bv)
    pad = nv * bv - V
    e = embed.astype(h.dtype)
    if pad:
        e = jnp.pad(e, ((0, pad), (0, 0)))
    return N, d, V, bn, bv, nv, e


def _forward(h, embed, targets, *, kind, n, c, clamp, block_n, block_v):
    N, d, V, bn, bv, nv, e = _setup(h, embed, block_n, block_v)
    t2 = targets.astype(jnp.int32).reshape(N // bn, 1, bn)
    kern = partial(_fwd_kernel, kind=kind, n=n, inv_c=1.0 / c, clamp=clamp, vocab=V, bv=bv, nv=nv)
    stat, zt = pl.pallas_call(
        kern,
        grid=(N // bn, nv),
        in_specs=[pl.BlockSpec((bn, d), lambda i, v: (i, 0)),
                  pl.BlockSpec((bv, d), lambda i, v: (v, 0)),
                  pl.BlockSpec((1, 1, bn), lambda i, v: (i, 0, 0))],
        out_specs=[pl.BlockSpec((bn, LANES), lambda i, v: (i, 0)),
                   pl.BlockSpec((bn, LANES), lambda i, v: (i, 0))],
        out_shape=[jax.ShapeDtypeStruct((N, LANES), F32), jax.ShapeDtypeStruct((N, LANES), F32)],
        scratch_shapes=[pltpu.VMEM((bn, LANES), F32), pltpu.VMEM((bn, LANES), F32),
                        pltpu.VMEM((bn, LANES), F32)],
        compiler_params=_params(),
    )(h, e, t2)
    return stat[:, 0], zt[:, 0]


def _backward(h, embed, targets, stat, scale, *, kind, n, c, clamp, block_n, block_v):
    N, d, V, bn, bv, nv, e = _setup(h, embed, block_n, block_v)
    t2 = targets.astype(jnp.int32).reshape(N // bn, 1, bn)
    s2 = stat.reshape(N // bn, 1, bn)
    kern = partial(_bwd_kernel, kind=kind, n=n, inv_c=1.0 / c, clamp=clamp, vocab=V, bv=bv, nv=nv)
    dz, dh = pl.pallas_call(
        kern,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(N // bn, nv),
            in_specs=[pl.BlockSpec((bn, d), lambda i, v, *_: (i, 0)),
                      pl.BlockSpec((bv, d), lambda i, v, *_: (v, 0)),
                      pl.BlockSpec((1, 1, bn), lambda i, v, *_: (i, 0, 0)),
                      pl.BlockSpec((1, 1, bn), lambda i, v, *_: (i, 0, 0))],
            out_specs=[pl.BlockSpec((bn, bv), lambda i, v, *_: (i, v)),
                       pl.BlockSpec((bn, d), lambda i, v, *_: (i, 0))],
            scratch_shapes=[pltpu.VMEM((bn, d), F32)],
        ),
        out_shape=[jax.ShapeDtypeStruct((N, nv * bv), h.dtype), jax.ShapeDtypeStruct((N, d), F32)],
        compiler_params=_params(),
    )(scale, h, e, t2, s2)
    de = lax.dot_general(dz, h, TRANS_A, preferred_element_type=F32)[:V]
    return dh, de


def _telemetry(stat, zt, kind, n, c, clamp):
    """(per-token NLL or target probability, objective value)."""
    if kind == "ce":
        nll = stat - zt
        return nll, jnp.mean(nll)
    x = jnp.minimum(zt / c, clamp)
    rad = 1.0 + x * x
    t = jnp.abs(x) + rad * lax.rsqrt(rad)
    rho = jnp.where(x >= 0, t, 1.0 / t)
    pt = _ipow(rho * stat, n)
    return pt, jnp.mean(1.0 - pt)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def _head(h, embed, targets, kind, n, c, clamp, block_n, block_v):
    stat, zt = _forward(h, embed, targets, kind=kind, n=n, c=c, clamp=clamp,
                        block_n=block_n, block_v=block_v)
    tel, obj = _telemetry(stat, zt, kind, n, c, clamp)
    return obj, tel


def _head_fwd(h, embed, targets, kind, n, c, clamp, block_n, block_v):
    stat, zt = _forward(h, embed, targets, kind=kind, n=n, c=c, clamp=clamp,
                        block_n=block_n, block_v=block_v)
    tel, obj = _telemetry(stat, zt, kind, n, c, clamp)
    return (obj, tel), (h, embed, targets, stat)


def _head_bwd(kind, n, c, clamp, block_n, block_v, res, g):
    h, embed, targets, stat = res
    g_obj = g[0]
    slope = 1.0 if kind == "ce" else n / c
    scale = jnp.reshape(g_obj * (slope / h.shape[0]), (1,)).astype(F32)
    dh, de = _backward(h, embed, targets, stat, scale, kind=kind, n=n, c=c, clamp=clamp,
                       block_n=block_n, block_v=block_v)
    return dh.astype(h.dtype), de.astype(embed.dtype), None


_head.defvjp(_head_fwd, _head_bwd)


def fused_head(h, embed, targets, *, kind="ce", n=16, c=16.0, block_n=128, block_v=512):
    """Fused tied-embedding projection and likelihood objective.

    ``h`` (N, d) hidden states, ``embed`` (V, d), ``targets`` (N,). Returns
    ``(objective, telemetry)`` as in ``algebraic.model.head_loss``: for "ce"
    the mean cross-entropy and per-token NLL; for "alg_ce" an algebraic
    stand-in objective whose gradient is the exact log-likelihood gradient,
    and the per-token target probability.
    """
    if kind not in ("ce", "alg_ce"):
        raise ValueError(kind)
    if kind == "alg_ce" and (n & (n - 1)):
        raise ValueError("n must be a power of two")
    clamp = 0.25 * _root(1e38 / embed.shape[0], n) if kind == "alg_ce" else 0.0
    obj, tel = _head(h, embed, targets, kind, int(n), float(c), float(clamp), block_n, block_v)
    return obj, lax.stop_gradient(tel)
