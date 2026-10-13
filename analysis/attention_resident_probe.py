#!/usr/bin/env python3
"""Measure an exact backward schedule that keeps all K/V resident in VMEM.

The deployed kernel is unchanged. Compile and retain baseline executables
before installing this process-local alternative to ``_backward_fused``.
The alternative accumulates dQ inside each query block and writes it once,
instead of writing a float32 partial for every key block and reducing in XLA.
"""

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np

from analysis.attention_performance import ratio_ci
from algebraic.kernels import flash as fk


def ordered_tile(q, k, v, do, stat, di, i, j, *, var, bq, bk, masked, order):
    """Issue the independent do@V.T earlier to test MXU/VPU overlap."""
    if order == "dpfirst":
        dp = lax.dot_general(do, v, fk.TRANS_B, preferred_element_type=jnp.float32)
    s = lax.dot_general(q, k, fk.TRANS_B, preferred_element_type=jnp.float32)
    if order == "early":
        dp = lax.dot_general(do, v, fk.TRANS_B, preferred_element_type=jnp.float32)
    mask = fk._causal_mask(i, j, bq, bk) if masked else None
    if var.kind == "softmax":
        p, r = jnp.exp(s - stat), None
    else:
        if not var.shift:
            x = s if var.bounded_scores else jnp.minimum(s, var.clamp)
        else:
            x = jnp.where(mask, s, 0.) if masked else s
        rho, r = fk._variant_rho(x, var)
        p = fk._ipow(rho, var.n) * stat if var.inv_stat else fk._ipow(rho * stat, var.n)
    if masked:
        p = jnp.where(mask, p, 0.)
    ds = p * (dp - di)
    if r is not None:
        ds = ds * r
        if not var.shift and not var.bounded_scores:
            ds = jnp.where(s < var.clamp, ds, 0.)
    return p, ds


def resident_kernel(sink_ref, q_ref, k_ref, v_ref, do_ref, stat_ref, di_ref,
                    dk_ref, dv_ref, dq_ref, dk_sc, dv_sc, dq_sc, *, var,
                    bq, chunk, nq, nk):
    i = pl.program_id(3)

    @pl.when(i == 0)
    def init_kv():
        dk_sc[...] = jnp.zeros(dk_sc.shape, jnp.float32)
        dv_sc[...] = jnp.zeros(dv_sc.shape, jnp.float32)

    dq_sc[...] = jnp.zeros(dq_sc.shape, jnp.float32)

    def chunk_body(j, unused):
        key_slice = pl.ds(j * chunk, chunk)

        def step(masked):
            q, do = q_ref[0, 0], do_ref[0, 0]
            k, v = k_ref[0, 0, key_slice, :], v_ref[0, 0, key_slice, :]
            stat = jnp.expand_dims(stat_ref[0, 0, 0], -1)
            di = jnp.expand_dims(di_ref[0, 0, 0], -1)
            p, ds = fk._tile_grads(q, k, v, do, stat, di, i, j,
                                   var=var, bq=bq, bk=chunk, masked=masked)
            dsb = ds.astype(q.dtype)
            dv_sc[key_slice, :] += lax.dot_general(
                p.astype(do.dtype), do, fk.TRANS_A, preferred_element_type=jnp.float32)
            dk_sc[key_slice, :] += lax.dot_general(
                dsb, q, fk.TRANS_A, preferred_element_type=jnp.float32)
            dq_sc[...] += lax.dot(dsb, k, preferred_element_type=jnp.float32)

        diag = fk._crosses_diagonal(i, j, bq, chunk)

        @pl.when(jnp.logical_not(diag))
        def full():
            step(False)

        @pl.when(diag)
        def masked():
            step(True)
        return unused

    lax.fori_loop(0, fk._last_kblock(i, bq, chunk, nk) + 1, chunk_body, None)
    slope = 1. if var.kind == "softmax" else float(var.n)
    dq_ref[0, 0] = (dq_sc[...] * slope).astype(dq_ref.dtype)

    @pl.when(i == nq - 1)
    def finish_kv():
        dk_ref[0, 0] = (dk_sc[...] * slope).astype(dk_ref.dtype)
        dv_ref[0, 0] = dv_sc[...].astype(dv_ref.dtype)


def resident_backward(q, k, v, sink, stat4, di4, do, di, *, var, bq, bk, chunk,
                      query_block=None):
    B, H, T, D = q.shape
    bq = min(T, query_block or bq)
    if T % chunk:
        raise ValueError("sequence length must divide compute chunk")
    nq, nk = T // bq, T // chunk
    q_map = lambda b, h, j, i, *_: (b, h, i, 0)
    kv_map = lambda b, h, j, i, *_: (b, h, 0, 0)
    row_map = lambda b, h, j, i, *_: (b, h, 0, i)
    dk, dv, dq = pl.pallas_call(
        partial(resident_kernel, var=var, bq=bq, chunk=chunk, nq=nq, nk=nk),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(B, H, 1, nq),
            in_specs=[pl.BlockSpec((1, 1, bq, D), q_map),
                      pl.BlockSpec((1, 1, T, D), kv_map),
                      pl.BlockSpec((1, 1, T, D), kv_map),
                      pl.BlockSpec((1, 1, bq, D), q_map),
                      pl.BlockSpec((1, 1, 1, bq), row_map),
                      pl.BlockSpec((1, 1, 1, bq), row_map)],
            out_specs=[pl.BlockSpec((1, 1, T, D), kv_map),
                       pl.BlockSpec((1, 1, T, D), kv_map),
                       pl.BlockSpec((1, 1, bq, D), q_map)],
            scratch_shapes=[pltpu.VMEM((T, D), jnp.float32),
                            pltpu.VMEM((T, D), jnp.float32),
                            pltpu.VMEM((bq, D), jnp.float32)]),
        out_shape=[jax.ShapeDtypeStruct(k.shape, k.dtype),
                   jax.ShapeDtypeStruct(v.shape, v.dtype),
                   jax.ShapeDtypeStruct(q.shape, q.dtype)],
        compiler_params=fk._params(),
    )(sink, q, k, v, do, stat4, di4)
    return dq, dk, dv, di


def packed_kernel(sink_ref, q_ref, k_ref, v_ref, do_ref, stat_ref, di_ref,
                  dkv_ref, dq_ref, dkv_sc, dq_sc, *, var, bq, chunk, nq, nk):
    """Pack dK/dV into one 128-column product instead of two 64-column products."""
    i = pl.program_id(3)
    slope = 1. if var.kind == "softmax" else float(var.n)

    @pl.when(i == 0)
    def init():
        dkv_sc[...] = jnp.zeros(dkv_sc.shape, jnp.float32)
    dq_sc[...] = jnp.zeros(dq_sc.shape, jnp.float32)

    def chunk_body(j, unused):
        key_slice = pl.ds(j * chunk, chunk)
        def step(masked):
            q, do = q_ref[0, 0], do_ref[0, 0]
            k, v = k_ref[0, 0, key_slice, :], v_ref[0, 0, key_slice, :]
            stat, di = jnp.expand_dims(stat_ref[0, 0, 0], -1), jnp.expand_dims(di_ref[0, 0, 0], -1)
            p, ds = fk._tile_grads(q, k, v, do, stat, di, i, j,
                                   var=var, bq=bq, bk=chunk, masked=masked)
            dsb = ds.astype(q.dtype)
            coefficients = jnp.concatenate((dsb, p.astype(q.dtype)), axis=0)
            zeros = jnp.zeros_like(q)
            q_pad = jnp.concatenate(((q * slope).astype(q.dtype), zeros), axis=1)
            do_pad = jnp.concatenate((zeros, do), axis=1)
            rhs = jnp.concatenate((q_pad, do_pad), axis=0)
            dkv_sc[key_slice, :] += lax.dot_general(coefficients, rhs, fk.TRANS_A,
                                                   preferred_element_type=jnp.float32)
            dq_sc[...] += lax.dot(dsb, k, preferred_element_type=jnp.float32)
        diag = fk._crosses_diagonal(i, j, bq, chunk)
        @pl.when(jnp.logical_not(diag))
        def full():
            step(False)
        @pl.when(diag)
        def masked():
            step(True)
        return unused
    lax.fori_loop(0, fk._last_kblock(i, bq, chunk, nk) + 1, chunk_body, None)
    dq_ref[0, 0] = (dq_sc[...] * slope).astype(dq_ref.dtype)
    @pl.when(i == nq - 1)
    def finish():
        dkv_ref[0, 0] = dkv_sc[...].astype(dkv_ref.dtype)


def packed_backward(q, k, v, sink, stat4, di4, do, di, *, var, bq, bk, chunk, query_block=None):
    B, H, T, D = q.shape
    bq = min(T, query_block or bq)
    nq, nk = T // bq, T // chunk
    q_map = lambda b, h, j, i, *_: (b, h, i, 0)
    kv_map = lambda b, h, j, i, *_: (b, h, 0, 0)
    row_map = lambda b, h, j, i, *_: (b, h, 0, i)
    dkv, dq = pl.pallas_call(
        partial(packed_kernel, var=var, bq=bq, chunk=chunk, nq=nq, nk=nk),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1, grid=(B, H, 1, nq),
            in_specs=[pl.BlockSpec((1, 1, bq, D), q_map),
                      pl.BlockSpec((1, 1, T, D), kv_map), pl.BlockSpec((1, 1, T, D), kv_map),
                      pl.BlockSpec((1, 1, bq, D), q_map),
                      pl.BlockSpec((1, 1, 1, bq), row_map), pl.BlockSpec((1, 1, 1, bq), row_map)],
            out_specs=[pl.BlockSpec((1, 1, T, 2 * D), kv_map), pl.BlockSpec((1, 1, bq, D), q_map)],
            scratch_shapes=[pltpu.VMEM((T, 2 * D), jnp.float32), pltpu.VMEM((bq, D), jnp.float32)]),
        out_shape=[jax.ShapeDtypeStruct((B, H, T, 2 * D), q.dtype),
                   jax.ShapeDtypeStruct(q.shape, q.dtype)],
        compiler_params=fk._params(),
    )(sink, q, k, v, do, stat4, di4)
    return dq, dkv[..., :D], dkv[..., D:], di


def validate(original, candidate, kind, dtype, *, shifted=False, seq=512):
    shape = (1, 2, seq, 64)
    q, k, v, g = [jax.random.normal(key, shape, dtype) for key in
                  jax.random.split(jax.random.PRNGKey(83), 4)]
    sink = jnp.array([-.25, .125], jnp.float32)
    opts = {"block": 128, "kind": kind, "resident_bwd": False}
    if kind == "asinh":
        opts.update(n=16, c=8., shift=shifted, fast_rho=not shifted)
    fn = lambda q, k, v, s: fk.flash_attention(q, k, v, s, **opts)
    def loss(q, k, v, s):
        return jnp.sum(fn(q, k, v, s).astype(jnp.float32) * g.astype(jnp.float32))
    grad = jax.grad(loss, argnums=(0, 1, 2, 3))
    with jax.default_matmul_precision("highest" if dtype == jnp.float32 else "default"):
        fk._backward_fused = original
        jax.clear_caches()
        expected = jax.jit(grad)(q, k, v, sink)
        fk._backward_fused = candidate
        jax.clear_caches()
        got = jax.jit(grad)(q, k, v, sink)
        relative = [float(jnp.linalg.norm(a.astype(jnp.float32) - b.astype(jnp.float32)) /
                          jnp.maximum(jnp.linalg.norm(b.astype(jnp.float32)), 1e-6))
                    for a, b in zip(got, expected)]
        max_abs = [float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))
                   for a, b in zip(got, expected)]
    tolerance = 1e-4 if dtype == jnp.float32 else .01
    return {"shape": shape, "dtype": str(jnp.dtype(dtype)), "kind": kind,
            "shifted": shifted, "q_k_v_sink_relative_l2_errors": relative,
            "q_k_v_sink_max_abs_errors": max_abs,
            "reference": "validated deployed backward, identical forward",
            "passed": max(relative) < tolerance}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--chunks", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--query-blocks", type=int, nargs="+", default=[512])
    p.add_argument("--validation-seq", type=int, default=512)
    p.add_argument("--validation-dtypes", nargs="+", choices=["float32", "bfloat16"],
                   default=["float32", "bfloat16"])
    p.add_argument("--tile-orders", nargs="+", choices=["original", "early", "dpfirst"],
                   default=["original"])
    p.add_argument("--packing", nargs="+", choices=["separate", "together"], default=["separate"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    shape = (8, args.heads, 2048, 64)
    q, k, v, g = [jax.random.normal(key, shape, jnp.bfloat16) for key in
                  jax.random.split(jax.random.PRNGKey(37), 4)]
    sink = jnp.zeros((args.heads,), jnp.float32)
    report = {"utc": datetime.now(timezone.utc).isoformat(), "shape": shape,
              "jax": jax.__version__, "devices": [str(x) for x in jax.devices()],
              "rounds": args.rounds, "iters": args.iters, "seed": 37,
              "validation_seq": args.validation_seq, "validation_dtypes": args.validation_dtypes,
              "query_blocks": args.query_blocks, "chunks": args.chunks,
              "tile_orders": args.tile_orders,
              "packing": args.packing,
              "prototype": True, "operator_changed": False,
              "kernel_sha256": hashlib.sha256((ROOT / "src/algebraic/kernels/flash.py").read_bytes()).hexdigest(),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "results": {}, "validation": {}, "failures": {}}
    functions = {}
    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def compile_case(name, kind):
        opts = {"kind": kind, "compact_stat": True, "resident_bwd": False}
        if kind == "asinh":
            opts.update(n=16, c=8., range_guard=True)
        # Preserve the same sink convention as the paired baseline report.
        fn = lambda q, k, v: fk.flash_attention(q, k, v, sink if kind == "asinh" else None, **opts)
        start = time.perf_counter()
        def loss(q, k, v):
            return jnp.sum(fn(q, k, v).astype(jnp.float32) * g.astype(jnp.float32))
        f = jax.jit(jax.grad(loss, argnums=(0, 1, 2))).lower(q, k, v).compile()
        for _ in range(3):
            jax.block_until_ready(f(q, k, v))
        functions[name] = f
        report["results"][name] = {"samples_ms": [], "compile_s": time.perf_counter() - start}
        print(f"compiled {name}", flush=True)
        save()

    original = fk._backward_fused
    original_tile = fk._tile_grads
    try:
        for kind in ("softmax", "asinh"):
            compile_case(f"{kind}_baseline", kind)
        for query_block, chunk, order, packing in [(b, c, o, p) for b in args.query_blocks
                                          for c in args.chunks for o in args.tile_orders for p in args.packing]:
            backend = resident_backward if packing == "separate" else packed_backward
            candidate = partial(backend, chunk=chunk, query_block=query_block)
            fk._tile_grads = original_tile if order == "original" else partial(ordered_tile, order=order)
            for kind in ("softmax", "asinh"):
                name = f"{kind}_resident_q{query_block}_k{chunk}_{order}_{packing}"
                checks = []
                try:
                    for dtype in [jnp.dtype(name).type for name in args.validation_dtypes]:
                        checks.append(validate(original, candidate, kind, dtype, seq=args.validation_seq))
                    if kind == "asinh":
                        shifted_dtype = jnp.dtype(args.validation_dtypes[0]).type
                        checks.append(validate(original, candidate, kind, shifted_dtype, shifted=True,
                                               seq=args.validation_seq))
                    report["validation"][name] = checks
                    print(name, checks, flush=True)
                    if all(check["passed"] for check in checks):
                        fk._backward_fused = candidate
                        jax.clear_caches()
                        compile_case(name, kind)
                except Exception as exc:
                    report["failures"][name] = {"type": type(exc).__name__, "message": str(exc)}
                    print(f"FAILED {name}: {type(exc).__name__}: {str(exc).splitlines()[0]}", flush=True)
                save()
    finally:
        fk._backward_fused = original
        fk._tile_grads = original_tile

    cases = list(functions)
    rng = np.random.default_rng(37)
    for rid in range(args.rounds):
        for idx in rng.permutation(len(cases)):
            name = cases[int(idx)]
            f = functions[name]
            jax.block_until_ready(f(q, k, v))
            start = time.perf_counter()
            for _ in range(args.iters):
                out = f(q, k, v)
            jax.block_until_ready(out)
            report["results"][name]["samples_ms"].append((time.perf_counter() - start) * 1000 / args.iters)
        print(f"round {rid + 1}/{args.rounds}", flush=True)
        save()
    for name, result in report["results"].items():
        result["median_ms"] = float(np.median(result["samples_ms"]))
        base = "softmax_baseline" if name.startswith("softmax") else "asinh_baseline"
        result["ratio_to_same_operator_baseline"] = ratio_ci(result["samples_ms"], report["results"][base]["samples_ms"])
        result["ratio_to_standard"] = ratio_ci(result["samples_ms"], report["results"]["softmax_baseline"]["samples_ms"])
        print(name, result["median_ms"], flush=True)
    best_standard = min((n for n in functions if n.startswith("softmax")),
                        key=lambda n: report["results"][n]["median_ms"])
    report["fastest_standard"] = best_standard
    for result in report["results"].values():
        result["ratio_to_fastest_standard"] = ratio_ci(result["samples_ms"],
                                                       report["results"][best_standard]["samples_ms"])
    save()


if __name__ == "__main__":
    main()
