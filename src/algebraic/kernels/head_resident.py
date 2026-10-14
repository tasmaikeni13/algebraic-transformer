"""Exact causal attention with a full head resident in TPU VMEM.

Static score tiles bound the nonlinear working set and end at the last query.
Only supported BF16 model shapes select this backend. The public dispatcher
retains the shifted fallback for out-of-range algebraic scores. Softmax is
available as an explicit benchmark control, using its separately tuned tiles.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import flash as fk


def _causal_mask(i, start, bq, bk):
    rows = i * bq + lax.broadcasted_iota(jnp.int32, (bq, bk), 0)
    cols = start + lax.broadcasted_iota(jnp.int32, (bq, bk), 1)
    return cols <= rows

def _fwd_kernel(sink_ref, q_ref, k_ref, v_ref, o_ref, stat_ref, m_ref, l_ref, acc_ref, *, var, has_sink, T, D, bq, bk, trim):
    h = pl.program_id(1)
    for i in range(T // bq):
        qs = slice(i * bq, (i + 1) * bq)
        q = q_ref[0, 0, qs, :]
        unshifted = var.kind == 'asinh' and (not var.shift)
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
        if unshifted:
            if has_sink:
                s = jnp.full(l_ref.shape, sink_ref[h], jnp.float32)
                if not var.bounded_scores:
                    s = jnp.minimum(s, var.clamp)
                l_ref[...] = fk._ipow(fk._variant_rho(s, var)[0], var.n)
            else:
                l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
        else:
            m_ref[...] = jnp.full(m_ref.shape, sink_ref[h] if has_sink else fk.NEG, jnp.float32)
            l_ref[...] = jnp.ones(l_ref.shape, jnp.float32) if has_sink else jnp.zeros(l_ref.shape, jnp.float32)
        for start in range(0, (i + 1) * bq, bk):
            end = min(start + bk, (i + 1) * bq) if trim else start + bk
            width = end - start
            (k, v) = (k_ref[0, 0, start:end, :], v_ref[0, 0, start:end, :])
            masked = end > i * bq
            if masked and bq == width == 512 and (start == i * bq) and var.diagonal_fwd and unshifted:
                (p, row_sum) = fk._diagonal_forward_weights(q, k, i, i, var=var)
                l_ref[...] += row_sum
            else:
                s = lax.dot_general(q, k, fk.TRANS_B, preferred_element_type=jnp.float32)
                mask = _causal_mask(i, start, bq, width) if masked else None
                if unshifted:
                    x = s if var.bounded_scores else jnp.minimum(s, var.clamp)
                    p = fk._ipow(fk._variant_rho(x, var)[0], var.n)
                    if masked:
                        p = jnp.where(mask, p, 0.0)
                    l_ref[...] += jnp.sum(p, axis=1, keepdims=True)
                else:
                    tile_max = jnp.max(jnp.where(mask, s, -jnp.inf) if masked else s, axis=1, keepdims=True)
                    previous = m_ref[...]
                    maximum = jnp.maximum(previous, tile_max)
                    if not has_sink and start == 0:
                        maximum = jnp.broadcast_to(tile_max, m_ref.shape)
                        previous = maximum
                    if var.kind == 'softmax':
                        p = jnp.exp(s - fk._bcast(maximum, width))
                        alpha = jnp.exp(previous - maximum)
                    else:
                        shift = fk._rho_exact(-maximum)[0]
                        x = jnp.where(mask, s, 0.0) if masked else s
                        p = fk._ipow(fk._rho(x, var.fast_rho)[0] * fk._bcast(shift, width), var.n)
                        alpha = fk._ipow(fk._rho_exact(previous)[0] * shift, var.n)
                    if masked:
                        p = jnp.where(mask, p, 0.0)
                    l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=1, keepdims=True)
                    acc_ref[...] *= fk._bcast(alpha, D)
                    m_ref[...] = maximum
            acc_ref[...] += lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32)
        inv = 1.0 / l_ref[...]
        o_ref[0, 0, qs, :] = (acc_ref[...] * fk._bcast(inv, D)).astype(o_ref.dtype)
        if var.kind == 'softmax':
            stat = m_ref[...] + jnp.log(l_ref[...])
        elif unshifted:
            stat = inv if var.inv_stat else fk._inv_root(l_ref[...], var.n)
        else:
            stat = fk._rho_exact(-m_ref[...])[0] * fk._inv_root(l_ref[...], var.n)
        stat_ref[0, 0, 0, qs] = jnp.transpose(stat)[:1, :][0]

def _bwd_kernel(sink_ref, q_ref, k_ref, v_ref, do_ref, stat_ref, di_ref, dk_ref, dv_ref, dq_ref, dk_sc, dv_sc, dq_sc, *, var, T, bq, bk, trim):
    dk_sc[...] = jnp.zeros(dk_sc.shape, jnp.float32)
    dv_sc[...] = jnp.zeros(dv_sc.shape, jnp.float32)
    slope = 1.0 if var.kind == 'softmax' else float(var.n)
    for i in range(T // bq):
        qs = slice(i * bq, (i + 1) * bq)
        (q, do) = (q_ref[0, 0, qs, :], do_ref[0, 0, qs, :])
        (stat, di) = (stat_ref[0, 0, 0, qs][:, None], di_ref[0, 0, 0, qs][:, None])
        dq_sc[...] = jnp.zeros(dq_sc.shape, jnp.float32)
        for start in range(0, (i + 1) * bq, bk):
            end = min(start + bk, (i + 1) * bq) if trim else start + bk
            width = end - start
            ks = slice(start, end)
            (k, v) = (k_ref[0, 0, ks, :], v_ref[0, 0, ks, :])
            masked = end > i * bq
            if masked and bq == width == 512 and (start == i * bq) and var.diagonal_bwd:
                (p, ds) = fk._diagonal_tile_grads(q, k, v, do, stat, di, i, i, var=var, bq=bq, bk=width, masked=True)
            else:
                (p, ds) = fk._tile_grads(q, k, v, do, stat, di, i, 0, var=var, bq=bq, bk=width, masked=False)
                if masked:
                    mask = _causal_mask(i, start, bq, width)
                    (p, ds) = (jnp.where(mask, p, 0.0), jnp.where(mask, ds, 0.0))
            ds = ds.astype(q.dtype)
            dv_sc[ks, :] += lax.dot_general(p.astype(do.dtype), do, fk.TRANS_A, preferred_element_type=jnp.float32)
            dk_sc[ks, :] += lax.dot_general(ds, q, fk.TRANS_A, preferred_element_type=jnp.float32)
            dq_sc[...] += lax.dot(ds, k, preferred_element_type=jnp.float32)
        dq_ref[0, 0, qs, :] = (dq_sc[...] * slope).astype(dq_ref.dtype)
    dk_ref[0, 0] = (dk_sc[...] * slope).astype(dk_ref.dtype)
    dv_ref[0, 0] = dv_sc[...].astype(dv_ref.dtype)

def _tiles(var):
    return (128, 512) if var.kind == "asinh" else (512, 512)


def forward(q, k, v, sink, *, var, has_sink):
    B, H, T, D = q.shape
    bq, bk = _tiles(var)
    spec = pl.BlockSpec((1, 1, T, D), lambda b, h, x, y, *_: (b, h, 0, 0))
    row_spec = pl.BlockSpec((1, 1, 1, T), lambda b, h, x, y, *_: (b, h, 0, 0))
    o, stat = pl.pallas_call(
        partial(_fwd_kernel, var=var, has_sink=has_sink, T=T, D=D, bq=bq, bk=bk, trim=True),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1, grid=(B, H, 1, 1), in_specs=[spec, spec, spec],
            out_specs=[spec, row_spec],
            scratch_shapes=[pltpu.VMEM((bq, fk.LANES), jnp.float32),
                            pltpu.VMEM((bq, fk.LANES), jnp.float32),
                            pltpu.VMEM((bq, D), jnp.float32)]),
        out_shape=[jax.ShapeDtypeStruct(q.shape, q.dtype),
                   jax.ShapeDtypeStruct((B, H, 1, T), jnp.float32)],
        compiler_params=fk._params(),
        name="head_resident_attention_forward")(sink, q, k, v)
    return o, stat[:, :, 0, :]


def backward(q, k, v, sink, stat4, di4, do, di, *, var):
    B, H, T, D = q.shape
    bq, bk = _tiles(var)
    spec = pl.BlockSpec((1, 1, T, D), lambda b, h, x, y, *_: (b, h, 0, 0))
    row_spec = pl.BlockSpec((1, 1, 1, T), lambda b, h, x, y, *_: (b, h, 0, 0))
    dk, dv, dq = pl.pallas_call(
        partial(_bwd_kernel, var=var, T=T, bq=bq, bk=bk, trim=True),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1, grid=(B, H, 1, 1),
            in_specs=[spec, spec, spec, spec, row_spec, row_spec],
            out_specs=[spec, spec, spec],
            scratch_shapes=[pltpu.VMEM((T, D), jnp.float32),
                            pltpu.VMEM((T, D), jnp.float32),
                            pltpu.VMEM((bq, D), jnp.float32)]),
        out_shape=[jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype),
                   jax.ShapeDtypeStruct(q.shape, q.dtype)],
        compiler_params=fk._params(),
        name="head_resident_attention_backward")(sink, q, k, v, do, stat4, di4)
    return dq, dk, dv, di
