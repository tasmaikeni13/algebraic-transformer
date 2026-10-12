#!/usr/bin/env python3
"""Bounded Hermite weight/VJP interpolation, with exact asinh outside range."""

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np

from analysis.attention_performance import ratio_ci
from src.sota import algebraic as A
from src.sota.kernels import flash as fk


LOW, HIGH, BINS = -4., 4., 128
WIDTH = (HIGH - LOW) / BINS
COEFFICIENT_REF = None


def coefficients(degree):
    x = np.linspace(LOW, HIGH, BINS + 1, dtype=np.float64)
    root = np.sqrt(1. + x * x)
    t = np.abs(x) + root
    rho = np.where(x >= 0., t, 1. / t)
    weight = rho ** 16
    d1 = weight * 16. / root
    d2 = weight * (256. / (root * root) - 16. * x / (root ** 3))
    c0, c1 = weight[:-1], WIDTH * d1[:-1]
    if degree == 3:
        f, d = weight[1:] - c0 - c1, WIDTH * d1[1:] - c1
        arrays = [c0, c1, 3. * f - d, d - 2. * f]
    else:
        c2 = .5 * WIDTH * WIDTH * d2[:-1]
        f = weight[1:] - c0 - c1 - c2
        d = WIDTH * d1[1:] - c1 - 2. * c2
        dd = WIDTH * WIDTH * d2[1:] - 2. * c2
        arrays = [c0, c1, c2, 10. * f - 4. * d + .5 * dd,
                  -15. * f + 7. * d - dd, 6. * f - 3. * d + .5 * dd]
    return np.asarray(arrays, np.float32)


def make_lookup(coeff):
    def lookup_one(x):
        coord = (jnp.clip(x, LOW, HIGH) - LOW) / WIDTH
        index = jnp.minimum(coord.astype(jnp.int32), BINS - 1)
        u = coord - index.astype(jnp.float32)
        cs = []
        for power, row in enumerate(coeff):
            table = (COEFFICIENT_REF[pl.ds(power, 1), :] if COEFFICIENT_REF is not None else
                     jnp.asarray(row)[None, :])
            if x.ndim == 2:
                # Mosaic supports only same-shape 2D take_along_axis. This
                # 128-wide lookup stays within the native lane dimension.
                table = jnp.broadcast_to(table, x.shape)
                cs.append(jnp.take_along_axis(table, index, axis=1))
            else:
                cs.append(table.reshape(-1)[index])
        w = jnp.zeros_like(x)
        dw = jnp.zeros_like(x)
        for power in range(len(cs) - 1, -1, -1):
            w = w * u + cs[power]
            if power:
                dw = dw * u + power * cs[power]
        return w, dw / WIDTH

    def lookup(x):
        if x.ndim == 2 and x.shape[0] > 8:
            pieces = [lookup(x[start:start + 8]) for start in range(0, x.shape[0], 8)]
            return (jnp.concatenate([p[0] for p in pieces], axis=0),
                    jnp.concatenate([p[1] for p in pieces], axis=0))
        if x.ndim == 2 and x.shape[-1] > 128:
            pieces = [lookup_one(x[:, start:start + 128]) for start in range(0, x.shape[-1], 128)]
            return (jnp.concatenate([p[0] for p in pieces], axis=1),
                    jnp.concatenate([p[1] for p in pieces], axis=1))
        return lookup_one(x)
    return lookup


def install_lookup(lookup, coeff):
    original_rho, original_tile, original_sink = fk._variant_rho, fk._tile_grads, fk._sink_mass_and_slope
    original_forward, original_fused, original_resident = fk._forward, fk._backward_fused, fk._backward_resident
    def variant_rho(x, var):
        return lookup(x) if var.n == 1 else original_rho(x, var)
    def tile(q, k, v, do, stat, di, i, j, *, var, bq, bk, masked):
        if var.n != 1:
            return original_tile(q, k, v, do, stat, di, i, j, var=var, bq=bq, bk=bk, masked=masked)
        x = lax.dot_general(q, k, fk.TRANS_B, preferred_element_type=jnp.float32)
        w, dw = lookup(x)
        p = w * stat
        dp = lax.dot_general(do, v, fk.TRANS_B, preferred_element_type=jnp.float32)
        ds = dw * stat * (dp - di)
        if masked:
            mask = fk._causal_mask(i, j, bq, bk)
            p, ds = jnp.where(mask, p, 0.), jnp.where(mask, ds, 0.)
        return p, ds
    def sink(s, stat, var):
        if var.n != 1:
            return original_sink(s, stat, var)
        w, dw = lookup(s)
        return w[None, :, None] * stat, dw / w

    def forward(q, k, v, s, *, var, bq, bk, has_sink):
        if var.n != 1:
            return original_forward(q, k, v, s, var=var, bq=bq, bk=bk, has_sink=has_sink)
        B, H, T, D = q.shape
        nq, nk = T // bq, T // bk
        def kv_map(b, h, i, j, *_):
            return b, h, jnp.minimum(j, fk._last_kblock(i, bq, bk, nk)), 0
        def kernel(s_ref, q_ref, k_ref, v_ref, c_ref, o_ref, stat_ref, m_sc, l_sc, acc_sc):
            global COEFFICIENT_REF
            COEFFICIENT_REF = c_ref
            try:
                fk._fwd_kernel(s_ref, q_ref, k_ref, v_ref, o_ref, stat_ref, m_sc, l_sc, acc_sc,
                               var=var, bq=bq, bk=bk, nk=nk, has_sink=has_sink, head_dim=D)
            finally:
                COEFFICIENT_REF = None
        o, stat = pl.pallas_call(kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1, grid=(B, H, nq, nk),
                in_specs=[pl.BlockSpec((1, 1, bq, D), lambda b,h,i,j,*_: (b,h,i,0)),
                          pl.BlockSpec((1, 1, bk, D), kv_map), pl.BlockSpec((1, 1, bk, D), kv_map),
                          pl.BlockSpec(coeff.shape, lambda *ids: (0,0))],
                out_specs=[pl.BlockSpec((1, 1, bq, D), lambda b,h,i,j,*_: (b,h,i,0)),
                           pl.BlockSpec((1, 1, 1, bq), lambda b,h,i,j,*_: (b,h,0,i))],
                scratch_shapes=[pltpu.VMEM((bq, 128), jnp.float32), pltpu.VMEM((bq, 128), jnp.float32),
                                pltpu.VMEM((bq, D), jnp.float32)]),
            out_shape=[jax.ShapeDtypeStruct(q.shape, q.dtype),
                       jax.ShapeDtypeStruct((B,H,1,T), jnp.float32)],
            compiler_params=fk._params())(s, q, k, v, jnp.asarray(coeff))
        return o, stat[:, :, 0, :]

    def backward_lut(q, k, v, s, stat, di4, do, di, *, var, bq=512, bk=512):
        B, H, T, D = q.shape
        bq, bk = (128 if q.dtype == jnp.float32 else 512), 512
        nq, nk = T // bq, T // bk
        q_map = lambda b,h,j,i,*_: (b,h,i,0)
        kv_map = lambda b,h,j,i,*_: (b,h,0,0)
        row_map = lambda b,h,j,i,*_: (b,h,0,i)
        def kernel(s_ref, q_ref, k_ref, v_ref, do_ref, stat_ref, di_ref, c_ref,
                   dk_ref, dv_ref, dq_ref, dk_sc, dv_sc, dq_sc):
            global COEFFICIENT_REF
            COEFFICIENT_REF = c_ref
            try:
                fk._resident_bwd_kernel(s_ref, q_ref, k_ref, v_ref, do_ref, stat_ref, di_ref,
                    dk_ref, dv_ref, dq_ref, dk_sc, dv_sc, dq_sc, var=var, bq=bq, bk=bk, nq=nq, nk=nk)
            finally:
                COEFFICIENT_REF = None
        dk, dv, dq = pl.pallas_call(kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1, grid=(B,H,1,nq),
                in_specs=[pl.BlockSpec((1,1,bq,D), q_map), pl.BlockSpec((1,1,T,D),kv_map),
                          pl.BlockSpec((1,1,T,D),kv_map), pl.BlockSpec((1,1,bq,D),q_map),
                          pl.BlockSpec((1,1,1,bq),row_map), pl.BlockSpec((1,1,1,bq),row_map),
                          pl.BlockSpec(coeff.shape, lambda *ids: (0,0))],
                out_specs=[pl.BlockSpec((1,1,T,D),kv_map), pl.BlockSpec((1,1,T,D),kv_map),
                           pl.BlockSpec((1,1,bq,D),q_map)],
                scratch_shapes=[pltpu.VMEM((T,D),jnp.float32), pltpu.VMEM((T,D),jnp.float32),
                                pltpu.VMEM((bq,D),jnp.float32)]),
            out_shape=[jax.ShapeDtypeStruct(k.shape,k.dtype),jax.ShapeDtypeStruct(v.shape,v.dtype),
                       jax.ShapeDtypeStruct(q.shape,q.dtype)],
            compiler_params=fk._params())(s,q,k,v,do,stat,di4,jnp.asarray(coeff))
        return dq,dk,dv,di
    def fused(*args, var, bq, bk):
        return (backward_lut(*args,var=var,bq=bq,bk=bk) if var.n == 1 else
                original_fused(*args,var=var,bq=bq,bk=bk))
    def resident(*args, var):
        return backward_lut(*args,var=var) if var.n == 1 else original_resident(*args,var=var)
    fk._variant_rho, fk._tile_grads, fk._sink_mass_and_slope = variant_rho, tile, sink
    fk._forward, fk._backward_fused, fk._backward_resident = forward, fused, resident
    return original_rho, original_tile, original_sink, original_forward, original_fused, original_resident


def attention(q, k, v, sink, block=512):
    qs = (q.astype(jnp.float32) / (math.sqrt(q.shape[-1]) * 8.)).astype(q.dtype)
    s = sink.reshape(-1) / 8.
    q2 = jnp.max(jnp.sum(qs.astype(jnp.float32) ** 2, -1))
    k2 = jnp.max(jnp.sum(k.astype(jnp.float32) ** 2, -1))
    bounded = (q2 * k2 <= (HIGH / 1.01) ** 2) & (jnp.max(jnp.abs(s)) <= HIGH / 1.01)
    bounded &= jnp.max(jnp.abs(v.astype(jnp.float32))) <= 1e8
    # n=1 is a private dispatch tag: lookup supplies the entire n=16 weight
    # and derivative, so the shared kernel's extra power/slope is identity.
    var = fk.Variant("asinh", n=1, shift=False, fast_rho=True, inv_stat=True,
                     compact_stat=True, bounded_scores=True,
                     resident_bwd=q.dtype == jnp.bfloat16 and q.shape[-2] == 2048 and block == 512)
    return lax.cond(bounded,
        lambda _: fk._flash(qs, k, v, s, var, block, block, True),
        lambda _: fk.flash_attention(q, k, v, sink, kind="asinh", n=16, c=8., block=block, range_guard=True),
        operand=None)


def validation(dtype):
    shape = (1, 2, 512, 64)
    q, k, v, g = [jax.random.normal(key, shape, dtype) for key in jax.random.split(jax.random.PRNGKey(109), 4)]
    sink = jnp.array([-.35, .125], jnp.float32).reshape(1, 2, 1, 1)
    ref = lambda q, k, v, s: A.asinh_attention(q, k, v, n=16, c=8., sink_logit=s)
    with jax.default_matmul_precision("highest" if dtype == jnp.float32 else "default"):
        got, want = jax.jit(attention)(q, k, v, sink), jax.jit(ref)(q, k, v, sink)
        output_error = float(jnp.max(jnp.abs(got.astype(jnp.float32) - want.astype(jnp.float32))))
        loss = lambda f, q, k, v, s: jnp.sum(f(q, k, v, s).astype(jnp.float32) * g.astype(jnp.float32))
        ga = jax.jit(jax.grad(partial(loss, attention), argnums=(0, 1, 2, 3)))(q, k, v, sink)
        gb = jax.jit(jax.grad(partial(loss, ref), argnums=(0, 1, 2, 3)))(q, k, v, sink)
        errors = [float(jnp.linalg.norm(a.astype(jnp.float32) - b.astype(jnp.float32)) /
                        jnp.maximum(jnp.linalg.norm(b.astype(jnp.float32)), 1e-6)) for a, b in zip(ga, gb)]
    return {"dtype": str(jnp.dtype(dtype)), "shape": shape, "output_max_abs_error": output_error,
            "q_k_v_sink_relative_l2_errors": errors,
            "passed": output_error < (.003 if dtype == jnp.float32 else .02) and
                      max(errors) < (.005 if dtype == jnp.float32 else .02)}


def numerical_grid(coeff):
    x = np.linspace(LOW, HIGH, 524289, dtype=np.float64)
    coord = (x - LOW) / WIDTH
    index = np.minimum(coord.astype(np.int32), BINS - 1)
    u = coord - index
    root = np.sqrt(1 + x * x)
    t = np.abs(x) + root
    exact = np.where(x >= 0, t, 1 / t) ** 16
    w, dw = np.zeros_like(x), np.zeros_like(x)
    for power in range(len(coeff) - 1, -1, -1):
        w = w * u + coeff[power, index]
        if power:
            dw = dw * u + power * coeff[power, index]
    return {"points": len(x), "domain": [LOW, HIGH], "bins": BINS,
            "coefficient_sha256": hashlib.sha256(coeff.tobytes()).hexdigest(),
            "weight_max_relative_error": float(np.max(np.abs(w / exact - 1))),
            "derivative_max_relative_error": float(np.max(np.abs((dw / WIDTH) / (exact * 16 / root) - 1))),
            "minimum_weight": float(w.min()), "minimum_derivative": float((dw / WIDTH).min()),
            "scope": "float64 evaluation of float32 coefficients versus radical oracle; finite grid, not certificate"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--degrees", nargs="+", type=int, default=[3, 5])
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    shape = (8, 12, 2048, 64)
    q, k, v, g = [jax.random.normal(key, shape, jnp.bfloat16) for key in jax.random.split(jax.random.PRNGKey(109), 4)]
    sink = jnp.zeros((12,), jnp.float32)
    report = {"utc": datetime.now(timezone.utc).isoformat(), "shape": shape, "seed": 109,
              "rounds": args.rounds, "iters": args.iters, "jax": jax.__version__,
              "devices": [str(x) for x in jax.devices()], "prototype": True, "model_adopted": False,
              "kernel_sha256": hashlib.sha256((ROOT / "src/sota/kernels/flash.py").read_bytes()).hexdigest(),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "results": {}, "validation": {}, "grid_error": {}, "failures": {}}
    functions = {}
    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    def compile_case(name, fn):
        for mode in ("forward", "training"):
            start = time.perf_counter()
            loss = lambda q, k, v: jnp.sum(fn(q, k, v).astype(jnp.float32) * g.astype(jnp.float32))
            f = (jax.jit(fn) if mode == "forward" else jax.jit(jax.grad(loss, argnums=(0, 1, 2)))).lower(q, k, v).compile()
            for _ in range(3):
                jax.block_until_ready(f(q, k, v))
            functions[name, mode] = f
            report["results"].setdefault(name, {})[mode] = {"samples_ms": [], "compile_s": time.perf_counter() - start}
            print(f"compiled {name} {mode}", flush=True)
            save()
    compile_case("standard_resident", lambda q, k, v: fk.flash_attention(q, k, v))
    compile_case("asinh_resident", lambda q, k, v: fk.flash_attention(q, k, v, sink, kind="asinh", n=16, c=8., range_guard=True))
    for degree in args.degrees:
        name = f"hermite{degree}"
        coeff = coefficients(degree)
        report["grid_error"][name] = numerical_grid(coeff)
        originals = install_lookup(make_lookup(coeff), coeff)
        jax.clear_caches()
        try:
            checks = [validation(jnp.float32), validation(jnp.bfloat16)]
            report["validation"][name] = checks
            print(name, report["grid_error"][name], checks, flush=True)
            if all(x["passed"] for x in checks):
                compile_case(name, lambda q, k, v: attention(q, k, v, sink))
        except Exception as exc:
            report["failures"][name] = {"type": type(exc).__name__, "message": str(exc)}
            print(f"FAILED {name}: {str(exc).splitlines()[0]}", flush=True)
        finally:
            (fk._variant_rho, fk._tile_grads, fk._sink_mass_and_slope,
             fk._forward, fk._backward_fused, fk._backward_resident) = originals
        save()
    cases = list(functions)
    rng = np.random.default_rng(109)
    for rid in range(args.rounds):
        for idx in rng.permutation(len(cases)):
            name, mode = cases[int(idx)]
            f = functions[name, mode]
            jax.block_until_ready(f(q, k, v))
            start = time.perf_counter()
            for _ in range(args.iters):
                out = f(q, k, v)
            jax.block_until_ready(out)
            report["results"][name][mode]["samples_ms"].append((time.perf_counter() - start) * 1000 / args.iters)
        print(f"round {rid+1}/{args.rounds}", flush=True)
        save()
    for name, result in report["results"].items():
        for mode, timing in result.items():
            timing["median_ms"] = float(np.median(timing["samples_ms"]))
            timing["ratio_to_standard"] = ratio_ci(timing["samples_ms"], report["results"]["standard_resident"][mode]["samples_ms"])
        print(name, result["forward"]["median_ms"], result["training"]["median_ms"], flush=True)
    save()


if __name__ == "__main__":
    main()
