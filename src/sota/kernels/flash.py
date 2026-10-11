"""Pallas TPU flash attention for softmax and asinh-power kernels.

One tiled implementation serves both arms; they differ only in the weight
function applied to a score tile and in the per-row statistic saved for the
backward pass.

* softmax: weight ``exp(s - m)``; saved statistic ``L = m + log l``;
  backward weight ``exp(s - L)``.
* asinh:   kernel-unit score ``x = q.k / (sqrt(d) c)`` and weight
  ``rho(x)^n``. Two numerically exact strategies:
  - ``shift=True``: running maximum ``m`` with weights ``(rho(x) rho(-m))^n``
    (``rho(a) rho(-b) = rho(a (-) b)``) and rescale ``(rho(m_old) rho(-m_new))^n``;
  - ``shift=False``: no running maximum. ``x`` is clamped to ``[-X, X]`` with
    ``X`` chosen so that ``T (2X)^n`` stays below the float32 range, which
    makes plain additive accumulation safe. The clamp only binds for scores
    beyond ``X c`` (``X ~ 79`` for n=16, T=2048).
  The saved statistic is ``sigma = Z^(-1/n)`` (log2 n square roots per row),
  and the backward weight is ``(rho(x) sigma)^n``. No logarithm or
  exponential is evaluated on this path.
  ``fast_rho`` evaluates ``rho(x) = x + sqrt(1 + x^2)`` without the
  reciprocal used for negative ``x``; its cancellation error is relative
  ``~2 eps x^2`` and only affects weights that are negligible after
  normalization unless a whole row has ``x << -100``.

A sink is a key with zero value and logit ``b`` (per head). It enters through
the initial running state. Sink-logit gradients are formed outside the kernels
from the saved statistic.

Forward grid (B, H, q blocks, k blocks); the backward uses a dK/dV kernel
(grid over k blocks, q inner) and a dQ kernel (q blocks, k inner). Causal
tiles strictly above the diagonal are neither computed nor fetched, and the
causal mask is only evaluated on tiles that cross the diagonal.
"""

from functools import partial
import math
from typing import NamedTuple, Optional

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32 = jnp.float32
LANES = 128
NEG = -1e4                 # finite "minus infinity" in kernel units
VMEM_LIMIT = 100 * 2 ** 20  # generous scoped-VMEM budget (TPU v4 cores)


class Variant(NamedTuple):
    kind: str          # "softmax" | "asinh"
    n: int = 16
    shift: bool = True
    fast_rho: bool = False
    clamp: float = 0.0  # used when shift is False


def _rho_exact(x):
    rad = 1.0 + x * x
    r = lax.rsqrt(rad)
    t = jnp.abs(x) + rad * r
    return jnp.where(x >= 0, t, 1.0 / t), r


def _rho(x, fast):
    if fast:
        rad = 1.0 + x * x
        r = lax.rsqrt(rad)
        return x + rad * r, r
    return _rho_exact(x)


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
    """l^(-1/n) for n a power of two (rsqrt followed by square roots)."""
    y = lax.rsqrt(l)
    k = n // 2
    while k > 1:
        y = y * lax.rsqrt(y)
        k //= 2
    return y


def _root(value: float, n: int) -> float:
    """value^(1/n) for n a power of two, by repeated square roots."""
    out = float(value)
    while n > 1:
        out = math.sqrt(out)
        n //= 2
    return out


def _bcast(x, width):
    """Broadcast a (rows, 128) lane-replicated array to (rows, width)."""
    if width == LANES:
        return x
    if width < LANES:
        return x[:, :width]
    return pltpu.repeat(x, width // LANES, axis=1)


TRANS_B = (((1,), (1,)), ((), ()))   # a @ b.T
TRANS_A = (((0,), (0,)), ((), ()))   # a.T @ b


def _causal_mask(i, j, bq, bk):
    row = i * bq + lax.broadcasted_iota(jnp.int32, (bq, bk), 0)
    col = j * bk + lax.broadcasted_iota(jnp.int32, (bq, bk), 1)
    return col <= row


def _last_kblock(i, bq, bk, nk):
    return jnp.minimum(nk - 1, lax.div((i + 1) * bq - 1, jnp.int32(bk)))


def _crosses_diagonal(i, j, bq, bk):
    """True if some key in tile (i, j) lies after some query in it."""
    return (j + 1) * bk - 1 > i * bq


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------

def _fwd_kernel(sink_ref, q_ref, k_ref, v_ref, o_ref, stat_ref, m_sc, l_sc, acc_sc, *,
                var: Variant, bq, bk, nk, has_sink, head_dim):
    h = pl.program_id(1)
    i = pl.program_id(2)
    j = pl.program_id(3)
    j_last = _last_kblock(i, bq, bk, nk)
    asinh_noshift = var.kind == "asinh" and not var.shift

    @pl.when(j == 0)
    def _init():
        acc_sc[...] = jnp.zeros(acc_sc.shape, F32)
        if asinh_noshift:
            if has_sink:
                b = jnp.minimum(jnp.full(l_sc.shape, sink_ref[h], F32), var.clamp)
                l_sc[...] = _ipow(_rho(b, var.fast_rho)[0], var.n)
            else:
                l_sc[...] = jnp.zeros(l_sc.shape, F32)
            return
        if has_sink:
            m_sc[...] = jnp.full(m_sc.shape, sink_ref[h], F32)
            l_sc[...] = jnp.ones(l_sc.shape, F32)
        else:
            m_sc[...] = jnp.full(m_sc.shape, NEG, F32)
            l_sc[...] = jnp.zeros(l_sc.shape, F32)

    def step(masked):
        q = q_ref[0, 0]
        k = k_ref[0, 0]
        v = v_ref[0, 0]
        s = lax.dot_general(q, k, TRANS_B, preferred_element_type=F32)
        mask = _causal_mask(i, j, bq, bk) if masked else None
        if asinh_noshift:
            x = jnp.minimum(s, var.clamp)
            p = _ipow(_rho(x, var.fast_rho)[0], var.n)
            if masked:
                p = jnp.where(mask, p, 0.0)
            l_sc[...] = l_sc[...] + jnp.sum(p, axis=1, keepdims=True)
            acc_sc[...] = acc_sc[...] + lax.dot(p.astype(v.dtype), v, preferred_element_type=F32)
            return
        if masked:
            s = jnp.where(mask, s, NEG)
        m_prev = m_sc[...]
        m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
        if var.kind == "softmax":
            p = jnp.exp(s - _bcast(m_new, bk))
            alpha = jnp.exp(m_prev - m_new)
        else:
            shift = _rho_exact(-m_new)[0]
            p = _ipow(_rho(s, var.fast_rho)[0] * _bcast(shift, bk), var.n)
            alpha = _ipow(_rho_exact(m_prev)[0] * shift, var.n)
        if masked:
            p = jnp.where(mask, p, 0.0)
        l_sc[...] = alpha * l_sc[...] + jnp.sum(p, axis=1, keepdims=True)
        acc_sc[...] = acc_sc[...] * _bcast(alpha, head_dim) + lax.dot(
            p.astype(v.dtype), v, preferred_element_type=F32)
        m_sc[...] = m_new

    diag = _crosses_diagonal(i, j, bq, bk)

    @pl.when(jnp.logical_and(j <= j_last, jnp.logical_not(diag)))
    def _full():
        step(False)

    @pl.when(jnp.logical_and(j <= j_last, diag))
    def _diag():
        step(True)

    @pl.when(j == j_last)
    def _fin():
        l = l_sc[...]
        o_ref[0, 0] = (acc_sc[...] * _bcast(1.0 / l, head_dim)).astype(o_ref.dtype)
        if var.kind == "softmax":
            stat_ref[0, 0] = m_sc[...] + jnp.log(l)
        elif asinh_noshift:
            stat_ref[0, 0] = _inv_root(l, var.n)
        else:
            stat_ref[0, 0] = _rho_exact(-m_sc[...])[0] * _inv_root(l, var.n)


def _params():
    return pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
                                vmem_limit_bytes=VMEM_LIMIT)


def _forward(q, k, v, sink, *, var, bq, bk, has_sink):
    B, H, T, D = q.shape
    nq, nk = T // bq, T // bk

    def kv_map(b, h, i, j, *_):
        return (b, h, jnp.minimum(j, _last_kblock(i, bq, bk, nk)), 0)

    kernel = partial(_fwd_kernel, var=var, bq=bq, bk=bk, nk=nk,
                     has_sink=has_sink, head_dim=D)
    o, stat = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(B, H, nq, nk),
            in_specs=[
                pl.BlockSpec((1, 1, bq, D), lambda b, h, i, j, *_: (b, h, i, 0)),
                pl.BlockSpec((1, 1, bk, D), kv_map),
                pl.BlockSpec((1, 1, bk, D), kv_map),
            ],
            out_specs=[
                pl.BlockSpec((1, 1, bq, D), lambda b, h, i, j, *_: (b, h, i, 0)),
                pl.BlockSpec((1, 1, bq, LANES), lambda b, h, i, j, *_: (b, h, i, 0)),
            ],
            scratch_shapes=[pltpu.VMEM((bq, LANES), F32), pltpu.VMEM((bq, LANES), F32),
                            pltpu.VMEM((bq, D), F32)],
        ),
        out_shape=[jax.ShapeDtypeStruct(q.shape, q.dtype),
                   jax.ShapeDtypeStruct((B, H, T, LANES), F32)],
        compiler_params=_params(),
    )(sink, q, k, v)
    return o, stat[..., 0]


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------

def _tile_grads(q, k, v, do, stat, di, i, j, *, var, bq, bk, masked):
    """Recompute weights for one tile; return (p, ds) with ds = dL/ds / n for asinh.

    Scores arrive pre-scaled (q carries 1/(sqrt(d) c)); the constant factor n of
    the asinh logit slope n r(x) is applied once to the dQ/dK accumulators.
    """
    s = lax.dot_general(q, k, TRANS_B, preferred_element_type=F32)
    mask = _causal_mask(i, j, bq, bk) if masked else None
    if var.kind == "softmax":
        p = jnp.exp(s - stat)
        r = None
    else:
        if not var.shift:
            x = jnp.minimum(s, var.clamp)   # forward clamp; straight-through slope
        else:
            x = jnp.where(mask, s, 0.0) if masked else s
        rho, r = _rho(x, var.fast_rho)
        p = _ipow(rho * stat, var.n)
    if masked:
        p = jnp.where(mask, p, 0.0)
    dp = lax.dot_general(do, v, TRANS_B, preferred_element_type=F32)
    ds = p * (dp - di)
    if r is not None:
        ds = ds * r
    return p, ds


def _dkv_kernel(sink_ref, q_ref, k_ref, v_ref, do_ref, stat_ref, di_ref, dk_ref, dv_ref,
                dk_sc, dv_sc, *, var, bq, bk, nq):
    j = pl.program_id(2)
    i = pl.program_id(3)
    i_first = lax.div(j * bk, jnp.int32(bq))

    @pl.when(i == 0)
    def _init():
        dk_sc[...] = jnp.zeros(dk_sc.shape, F32)
        dv_sc[...] = jnp.zeros(dv_sc.shape, F32)

    def step(masked):
        q = q_ref[0, 0]
        do = do_ref[0, 0]
        stat = jnp.expand_dims(stat_ref[0, 0, 0], -1)
        di = jnp.expand_dims(di_ref[0, 0, 0], -1)
        p, ds = _tile_grads(q, k_ref[0, 0], v_ref[0, 0], do, stat, di, i, j,
                            var=var, bq=bq, bk=bk, masked=masked)
        dv_sc[...] += lax.dot_general(p.astype(do.dtype), do, TRANS_A, preferred_element_type=F32)
        dk_sc[...] += lax.dot_general(ds.astype(q.dtype), q, TRANS_A, preferred_element_type=F32)

    diag = _crosses_diagonal(i, j, bq, bk)

    @pl.when(jnp.logical_and(i >= i_first, jnp.logical_not(diag)))
    def _full():
        step(False)

    @pl.when(jnp.logical_and(i >= i_first, diag))
    def _diag():
        step(True)

    @pl.when(i == nq - 1)
    def _fin():
        slope = 1.0 if var.kind == "softmax" else float(var.n)
        dk_ref[0, 0] = (dk_sc[...] * slope).astype(dk_ref.dtype)
        dv_ref[0, 0] = dv_sc[...].astype(dv_ref.dtype)


def _fused_bwd_kernel(sink_ref, q_ref, k_ref, v_ref, do_ref, stat_ref, di_ref,
                      dk_ref, dv_ref, dqp_ref, dk_sc, dv_sc, *, var, bq, bk, nq):
    """dK/dV accumulation plus a per-key-block dQ partial (summed in XLA)."""
    j = pl.program_id(2)
    i = pl.program_id(3)
    i_first = lax.div(j * bk, jnp.int32(bq))
    slope = 1.0 if var.kind == "softmax" else float(var.n)

    @pl.when(i == 0)
    def _init():
        dk_sc[...] = jnp.zeros(dk_sc.shape, F32)
        dv_sc[...] = jnp.zeros(dv_sc.shape, F32)

    @pl.when(i < i_first)
    def _skip():
        dqp_ref[...] = jnp.zeros(dqp_ref.shape, dqp_ref.dtype)

    def step(masked):
        q = q_ref[0, 0]
        do = do_ref[0, 0]
        k = k_ref[0, 0]
        stat = jnp.expand_dims(stat_ref[0, 0, 0], -1)
        di = jnp.expand_dims(di_ref[0, 0, 0], -1)
        p, ds = _tile_grads(q, k, v_ref[0, 0], do, stat, di, i, j,
                            var=var, bq=bq, bk=bk, masked=masked)
        dv_sc[...] += lax.dot_general(p.astype(do.dtype), do, TRANS_A, preferred_element_type=F32)
        dsb = ds.astype(q.dtype)
        dk_sc[...] += lax.dot_general(dsb, q, TRANS_A, preferred_element_type=F32)
        dqp_ref[0, 0, 0] = lax.dot(dsb, k, preferred_element_type=F32) * slope

    diag = _crosses_diagonal(i, j, bq, bk)

    @pl.when(jnp.logical_and(i >= i_first, jnp.logical_not(diag)))
    def _full():
        step(False)

    @pl.when(jnp.logical_and(i >= i_first, diag))
    def _diag():
        step(True)

    @pl.when(i == nq - 1)
    def _fin():
        dk_ref[0, 0] = (dk_sc[...] * slope).astype(dk_ref.dtype)
        dv_ref[0, 0] = dv_sc[...].astype(dv_ref.dtype)


def _dq_kernel(sink_ref, q_ref, k_ref, v_ref, do_ref, stat_ref, di_ref, dq_ref, dq_sc, *,
               var, bq, bk, nk):
    i = pl.program_id(2)
    j = pl.program_id(3)
    j_last = _last_kblock(i, bq, bk, nk)

    @pl.when(j == 0)
    def _init():
        dq_sc[...] = jnp.zeros(dq_sc.shape, F32)

    def step(masked):
        k = k_ref[0, 0]
        stat = jnp.expand_dims(stat_ref[0, 0, 0], -1)
        di = jnp.expand_dims(di_ref[0, 0, 0], -1)
        _, ds = _tile_grads(q_ref[0, 0], k, v_ref[0, 0], do_ref[0, 0], stat, di, i, j,
                            var=var, bq=bq, bk=bk, masked=masked)
        dq_sc[...] += lax.dot(ds.astype(k.dtype), k, preferred_element_type=F32)

    diag = _crosses_diagonal(i, j, bq, bk)

    @pl.when(jnp.logical_and(j <= j_last, jnp.logical_not(diag)))
    def _full():
        step(False)

    @pl.when(jnp.logical_and(j <= j_last, diag))
    def _diag():
        step(True)

    @pl.when(j == j_last)
    def _fin():
        slope = 1.0 if var.kind == "softmax" else float(var.n)
        dq_ref[0, 0] = (dq_sc[...] * slope).astype(dq_ref.dtype)


def _backward_fused(q, k, v, sink, stat4, di4, do, di, *, var, bq, bk):
    B, H, T, D = q.shape
    nq, nk = T // bq, T // bk

    def q_map(b, h, j, i, *_):
        return (b, h, jnp.maximum(i, lax.div(j * bk, jnp.int32(bq))), 0)

    def row_map(b, h, j, i, *_):
        return (b, h, 0, jnp.maximum(i, lax.div(j * bk, jnp.int32(bq))))

    def k_map(b, h, j, i, *_):
        return (b, h, j, 0)

    dk, dv, dqp = pl.pallas_call(
        partial(_fused_bwd_kernel, var=var, bq=bq, bk=bk, nq=nq),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(B, H, nk, nq),
            in_specs=[pl.BlockSpec((1, 1, bq, D), q_map), pl.BlockSpec((1, 1, bk, D), k_map),
                      pl.BlockSpec((1, 1, bk, D), k_map), pl.BlockSpec((1, 1, bq, D), q_map),
                      pl.BlockSpec((1, 1, 1, bq), row_map), pl.BlockSpec((1, 1, 1, bq), row_map)],
            out_specs=[pl.BlockSpec((1, 1, bk, D), k_map), pl.BlockSpec((1, 1, bk, D), k_map),
                       pl.BlockSpec((1, 1, 1, bq, D), lambda b, h, j, i, *_: (j, b, h, i, 0))],
            scratch_shapes=[pltpu.VMEM((bk, D), F32), pltpu.VMEM((bk, D), F32)],
        ),
        out_shape=[jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype),
                   jax.ShapeDtypeStruct((nk, B, H, T, D), F32)],
        compiler_params=_params(),
    )(sink, q, k, v, do, stat4, di4)
    dq = jnp.sum(dqp, axis=0).astype(q.dtype)
    return dq, dk, dv, di


def _backward(q, k, v, sink, o, stat, do, *, var, bq, bk, fused=True):
    B, H, T, D = q.shape
    nq, nk = T // bq, T // bk
    di = jnp.sum(do.astype(F32) * o.astype(F32), axis=-1)            # (B, H, T)
    stat4 = stat[:, :, None, :]
    di4 = di[:, :, None, :]
    if fused:
        return _backward_fused(q, k, v, sink, stat4, di4, do, di, var=var, bq=bq, bk=bk)

    def q_map(b, h, j, i, *_):
        return (b, h, jnp.maximum(i, lax.div(j * bk, jnp.int32(bq))), 0)

    def row_map(b, h, j, i, *_):
        return (b, h, 0, jnp.maximum(i, lax.div(j * bk, jnp.int32(bq))))

    def k_map(b, h, j, i, *_):
        return (b, h, j, 0)

    dk, dv = pl.pallas_call(
        partial(_dkv_kernel, var=var, bq=bq, bk=bk, nq=nq),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(B, H, nk, nq),
            in_specs=[pl.BlockSpec((1, 1, bq, D), q_map), pl.BlockSpec((1, 1, bk, D), k_map),
                      pl.BlockSpec((1, 1, bk, D), k_map), pl.BlockSpec((1, 1, bq, D), q_map),
                      pl.BlockSpec((1, 1, 1, bq), row_map), pl.BlockSpec((1, 1, 1, bq), row_map)],
            out_specs=[pl.BlockSpec((1, 1, bk, D), k_map), pl.BlockSpec((1, 1, bk, D), k_map)],
            scratch_shapes=[pltpu.VMEM((bk, D), F32), pltpu.VMEM((bk, D), F32)],
        ),
        out_shape=[jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype)],
        compiler_params=_params(),
    )(sink, q, k, v, do, stat4, di4)

    def kv_map(b, h, i, j, *_):
        return (b, h, jnp.minimum(j, _last_kblock(i, bq, bk, nk)), 0)

    dq = pl.pallas_call(
        partial(_dq_kernel, var=var, bq=bq, bk=bk, nk=nk),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(B, H, nq, nk),
            in_specs=[pl.BlockSpec((1, 1, bq, D), lambda b, h, i, j, *_: (b, h, i, 0)),
                      pl.BlockSpec((1, 1, bk, D), kv_map), pl.BlockSpec((1, 1, bk, D), kv_map),
                      pl.BlockSpec((1, 1, bq, D), lambda b, h, i, j, *_: (b, h, i, 0)),
                      pl.BlockSpec((1, 1, 1, bq), lambda b, h, i, j, *_: (b, h, 0, i)),
                      pl.BlockSpec((1, 1, 1, bq), lambda b, h, i, j, *_: (b, h, 0, i))],
            out_specs=pl.BlockSpec((1, 1, bq, D), lambda b, h, i, j, *_: (b, h, i, 0)),
            scratch_shapes=[pltpu.VMEM((bq, D), F32)],
        ),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        compiler_params=_params(),
    )(sink, q, k, v, do, stat4, di4)
    return dq, dk, dv, di


# ---------------------------------------------------------------------------
# custom VJP and public entry point
# ---------------------------------------------------------------------------

def _sink_mass_and_slope(sink, stat, var):
    """Per-row sink probability (B, H, T) and d(kernel logit)/d(sink)."""
    if var.kind == "softmax":
        return jnp.exp(sink[None, :, None] - stat), 1.0
    b = jnp.minimum(sink, var.clamp) if not var.shift else sink   # (H,)
    rho, r = _rho_exact(b)
    return _ipow(rho[None, :, None] * stat, var.n), var.n * r


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def _flash(q, k, v, sink, var, bq, bk, has_sink):
    return _forward(q, k, v, sink, var=var, bq=bq, bk=bk, has_sink=has_sink)[0]


def _flash_fwd(q, k, v, sink, var, bq, bk, has_sink):
    o, stat = _forward(q, k, v, sink, var=var, bq=bq, bk=bk, has_sink=has_sink)
    return o, (q, k, v, sink, o, stat)


def _flash_bwd(var, bq, bk, has_sink, res, do):
    q, k, v, sink, o, stat = res
    dq, dk, dv, di = _backward(q, k, v, sink, o, stat, do, var=var, bq=bq, bk=bk)
    if has_sink:
        mass, slope = _sink_mass_and_slope(sink, stat, var)
        dsink = -jnp.sum(mass * di, axis=(0, 2)) * slope
    else:
        dsink = jnp.zeros_like(sink)
    return dq, dk, dv, dsink.reshape(sink.shape)


_flash.defvjp(_flash_fwd, _flash_bwd)


def make_variant(kind: str, n: int = 16, c: float = 16.0, seq_len: int = 2048,
                 shift: Optional[bool] = None, fast_rho: Optional[bool] = None) -> Variant:
    if kind == "softmax":
        return Variant("softmax")
    if n & (n - 1):
        raise ValueError("asinh kernel needs n to be a power of two")
    # Defaults: the reciprocal-free rho, and no running maximum whenever the
    # clamp it requires sits at least 20 kernel units out (scores up to 20 c
    # untouched; e.g. 40 c for n = 16 and 5000 c for n = 8 at T = 2048).
    clamp = 0.25 * _root(1e38 / seq_len, n)
    if shift is None:
        shift = clamp < 20.0
    if fast_rho is None:
        fast_rho = True
    return Variant("asinh", int(n), bool(shift), bool(fast_rho), float(clamp))


def flash_attention(q, k, v, sink_logit: Optional[jax.Array] = None, *, kind: str = "softmax",
                    n: int = 16, c: float = 16.0, block: int = 512,
                    block_q: Optional[int] = None, block_k: Optional[int] = None,
                    shift: Optional[bool] = None, fast_rho: Optional[bool] = None):
    """Causal attention on (B, H, T, D). ``sink_logit`` broadcastable to (H,).

    For ``kind="asinh"`` the weights are ``rho(s / c)^n`` with ``s`` the
    scaled dot product; ``n`` must be a power of two.
    """
    B, H, T, D = q.shape
    bq = min(block_q or block, T)
    bk = min(block_k or block, T)
    if T % bq or T % bk:
        raise ValueError(f"sequence length {T} must be divisible by blocks ({bq}, {bk})")
    var = make_variant(kind, n, c, T, shift, fast_rho)
    unit = 1.0 if kind == "softmax" else float(c)
    scale = 1.0 / (math.sqrt(D) * unit)
    has_sink = sink_logit is not None
    if has_sink:
        sink_in = jnp.asarray(sink_logit, F32)
        sink = jnp.broadcast_to(jnp.reshape(sink_in, (-1,)), (H,)) / unit
    else:
        sink = jnp.zeros((H,), F32)
    # Fold the score scale into q (exact for power-of-two scales); autodiff
    # applies the matching factor to dq outside the kernels.
    q_scaled = (q.astype(F32) * scale).astype(q.dtype)
    return _flash(q_scaled, k, v, sink, var, bq, bk, has_sink)
