#!/usr/bin/env python3
"""Measure score-row segmentation without changing the attention operator.

The MXU still computes a full score/cotangent tile. Explicit VMEM buffers
bound the live nonlinear intermediates to a small group of query rows; only
the already-rounded matrix operands are loaded together for the three VJP
contractions. The independent dense reference checks q/k/v and sink VJPs.
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
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np

from algebraic.kernels import flash as fk
from algebraic import algebraic as A, standard as S
from analysis.attention_performance import ratio_ci


def segmented_tile(q, k, v, do, stat, di, i, j, *, var, bq, bk, masked,
                   rows=64, operand_dtype=None):
    rows = min(rows, bq)
    if bq % rows:
        raise ValueError("nonlinear row segment must divide the query tile")
    dtype = operand_dtype or q.dtype

    def evaluate(s_ref, dp_ref, p_ref, ds_ref, stat_ref, di_ref):
        s_ref[...] = lax.dot_general(q, k, fk.TRANS_B, preferred_element_type=jnp.float32)
        dp_ref[...] = lax.dot_general(do, v, fk.TRANS_B, preferred_element_type=jnp.float32)
        stat_ref[...] = stat
        di_ref[...] = di

        def segment(part, unused):
            rs = pl.ds(part * rows, rows)
            s = s_ref[rs, :]
            row_stat = stat_ref[rs, :]
            row_di = di_ref[rs, :]
            mask = None
            if masked:
                query = i * bq + part * rows + lax.broadcasted_iota(jnp.int32, (rows, bk), 0)
                key = j * bk + lax.broadcasted_iota(jnp.int32, (rows, bk), 1)
                mask = key <= query
            if var.kind == "softmax":
                p = jnp.exp(s - row_stat)
                r = None
            else:
                x = s if var.shift or var.bounded_scores else jnp.minimum(s, var.clamp)
                if masked and var.shift:
                    x = jnp.where(mask, x, 0.)
                rho, r = fk._variant_rho(x, var)
                p = (fk._ipow(rho, var.n) * row_stat if var.inv_stat else
                     fk._ipow(rho * row_stat, var.n))
            if masked:
                p = jnp.where(mask, p, 0.)
            ds = p * (dp_ref[rs, :] - row_di)
            if r is not None:
                ds *= r
                if not var.shift and not var.bounded_scores:
                    ds = jnp.where(s < var.clamp, ds, 0.)
            p_ref[rs, :] = p.astype(dtype)
            ds_ref[rs, :] = ds.astype(dtype)
            return unused

        lax.fori_loop(0, bq // rows, segment, None)
        return p_ref[...], ds_ref[...]

    return pl.run_scoped(evaluate, pltpu.VMEM((bq, bk), jnp.float32),
                         pltpu.VMEM((bq, bk), jnp.float32),
                         pltpu.VMEM((bq, bk), dtype), pltpu.VMEM((bq, bk), dtype),
                         pltpu.VMEM(stat.shape, jnp.float32), pltpu.VMEM(di.shape, jnp.float32))


def static_segmented_tile(q, k, v, do, stat, di, i, j, *, var, bq, bk, masked,
                          rows=64):
    """Expose small independent nonlinear chains without additional VMEM refs."""
    rows = min(rows, bq)
    if bq % rows:
        raise ValueError("nonlinear row segment must divide the query tile")
    scores = lax.dot_general(q, k, fk.TRANS_B, preferred_element_type=jnp.float32)
    dp = lax.dot_general(do, v, fk.TRANS_B, preferred_element_type=jnp.float32)
    probabilities, score_vjps = [], []
    for start in range(0, bq, rows):
        s, row_stat, row_di = scores[start:start+rows], stat[start:start+rows], di[start:start+rows]
        if masked:
            query = i * bq + start + lax.broadcasted_iota(jnp.int32, (rows, bk), 0)
            key = j * bk + lax.broadcasted_iota(jnp.int32, (rows, bk), 1)
            mask = key <= query
        if var.kind == "softmax":
            p, r = jnp.exp(s - row_stat), None
        else:
            x = s if var.shift or var.bounded_scores else jnp.minimum(s, var.clamp)
            if masked and var.shift:
                x = jnp.where(mask, x, 0.)
            rho, r = fk._variant_rho(x, var)
            p = (fk._ipow(rho, var.n) * row_stat if var.inv_stat else
                 fk._ipow(rho * row_stat, var.n))
        if masked:
            p = jnp.where(mask, p, 0.)
        ds = p * (dp[start:start+rows] - row_di)
        if r is not None:
            ds *= r
            if not var.shift and not var.bounded_scores:
                ds = jnp.where(s < var.clamp, ds, 0.)
        probabilities.append(p.astype(q.dtype))
        score_vjps.append(ds.astype(q.dtype))
    return jnp.concatenate(probabilities), jnp.concatenate(score_vjps)


def diagonal_tile(q, k, v, do, stat, di, i, j, *, var, bq, bk, masked,
                  rows=128, original):
    """Skip complete subtiles above the causal diagonal before score evaluation."""
    if not masked or bq != bk or bq <= rows:
        return original(q, k, v, do, stat, di, i, j, var=var, bq=bq, bk=bk, masked=masked)
    if bq % rows:
        raise ValueError("diagonal subtile must divide the square compute tile")
    count = bq // rows
    p_rows, ds_rows = [], []
    for qi in range(count):
        qs = slice(qi*rows, (qi+1)*rows)
        p_cols, ds_cols = [], []
        for kj in range(count):
            if kj > qi:
                p = ds = jnp.zeros((rows, rows), q.dtype)
            else:
                ks = slice(kj*rows, (kj+1)*rows)
                p, ds = original(q[qs], k[ks], v[ks], do[qs], stat[qs], di[qs],
                                 i*count+qi, j*count+kj, var=var, bq=rows, bk=rows,
                                 masked=kj == qi)
                p, ds = p.astype(q.dtype), ds.astype(q.dtype)
            p_cols.append(p)
            ds_cols.append(ds)
        p_rows.append(jnp.concatenate(p_cols, axis=1))
        ds_rows.append(jnp.concatenate(ds_cols, axis=1))
    return jnp.concatenate(p_rows), jnp.concatenate(ds_rows)


def validate(candidate, kind, dtype, *, seq=512, shifted=False):
    original = fk._tile_grads
    shape = (1, 2, seq, 64)
    keys = jax.random.split(jax.random.PRNGKey(313), 4)
    q, k, v, g = [jax.random.normal(key, shape, dtype) for key in keys]
    sink = jnp.array([-.4, .6], jnp.float32)[None, :, None, None]
    options = dict(kind=kind, n=16, c=8., block=512, compact_stat=True,
                   resident_bwd=dtype == jnp.bfloat16, range_guard=not shifted,
                   shift=shifted)
    fn = lambda q, k, v, s: fk.flash_attention(q, k, v, s, **options)
    ref = (lambda q, k, v, s: S.softmax_attention(q, k, v, sink_logit=s)) if kind == "softmax" else (
        lambda q, k, v, s: A.asinh_attention(q, k, v, n=16, c=8., sink_logit=s))
    def loss(fn, q, k, v, s):
        return jnp.sum(fn(q, k, v, s).astype(jnp.float32) * g.astype(jnp.float32))
    try:
        fk._tile_grads = candidate
        jax.clear_caches()
        with jax.default_matmul_precision("highest" if dtype == jnp.float32 else "default"):
            got = jax.jit(jax.grad(partial(loss, fn), argnums=(0, 1, 2, 3)))(q, k, v, sink)
            expected = jax.jit(jax.grad(partial(loss, ref), argnums=(0, 1, 2, 3)))(q, k, v, sink)
        errors = [float(jnp.linalg.norm(a.astype(jnp.float32) - b.astype(jnp.float32)) /
                        jnp.maximum(jnp.linalg.norm(b.astype(jnp.float32)), 1e-6))
                  for a, b in zip(got, expected)]
        return dict(shape=shape, dtype=str(jnp.dtype(dtype)), kind=kind, shifted=shifted,
                    reference="independent dense attention", q_k_v_sink_relative_l2=errors,
                    passed=max(errors) < (.02 if dtype == jnp.bfloat16 else .005))
    finally:
        fk._tile_grads = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--method", choices=["scoped", "static", "diagonal"], default="scoped")
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=317)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    shape = (8, args.heads, 2048, 64)
    q, k, v, g = [jax.random.normal(key, shape, jnp.bfloat16) for key in
                  jax.random.split(jax.random.PRNGKey(args.seed), 4)]
    sink = jnp.zeros((args.heads,), jnp.float32)
    report = dict(utc=datetime.now(timezone.utc).isoformat(), shape=shape, seed=args.seed,
                  jax=jax.__version__, devices=[str(d) for d in jax.devices()],
                  rows=args.rows, method=args.method, rounds=args.rounds, iters=args.iters,
                  prototype=True, operator_changed=False,
                  kernel_sha256=hashlib.sha256(Path(fk.__file__).read_bytes()).hexdigest(),
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  results={}, validation={}, failures={})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    functions = {}
    original = fk._tile_grads
    try:
        for rows in [None] + args.rows:
            backend = {"scoped": segmented_tile, "static": static_segmented_tile,
                       "diagonal": partial(diagonal_tile, original=original)}[args.method]
            candidate = original if rows is None else partial(backend, rows=rows)
            for kind in ("softmax", "asinh"):
                name = f"{kind}_" + ("baseline" if rows is None else f"rows{rows}")
                try:
                    if rows is not None:
                        checks = [validate(candidate, kind, dtype) for dtype in [jnp.float32, jnp.bfloat16]]
                        if kind == "asinh":
                            checks.append(validate(candidate, kind, jnp.bfloat16, shifted=True))
                        report["validation"][name] = checks
                        print(name, "validation", [x["q_k_v_sink_relative_l2"] for x in checks], flush=True)
                        if not all(x["passed"] for x in checks):
                            raise ValueError("dense VJP check failed")
                    fk._tile_grads = candidate
                    jax.clear_caches()
                    opts = dict(kind=kind, n=16, c=8., range_guard=True)
                    def loss(q, k, v):
                        return jnp.sum(fk.flash_attention(q, k, v, sink if kind == "asinh" else None,
                                                         **opts).astype(jnp.float32) * g.astype(jnp.float32))
                    start = time.perf_counter()
                    f = jax.jit(jax.grad(loss, argnums=(0, 1, 2))).lower(q, k, v).compile()
                    for _ in range(3):
                        jax.block_until_ready(f(q, k, v))
                    functions[name] = f
                    report["results"][name] = dict(samples_ms=[], compile_s=time.perf_counter() - start)
                    print("compiled", name, flush=True)
                except Exception as exc:
                    report["failures"][name] = dict(type=type(exc).__name__, message=str(exc))
                    print("FAILED", name, str(exc).splitlines()[0], flush=True)
                finally:
                    fk._tile_grads = original
                    save()
    finally:
        fk._tile_grads = original
    names = list(functions)
    rng = np.random.default_rng(args.seed)
    for rid in range(args.rounds):
        for idx in rng.permutation(len(names)):
            name = names[int(idx)]
            f = functions[name]
            jax.block_until_ready(f(q, k, v))
            start = time.perf_counter()
            for _ in range(args.iters):
                out = f(q, k, v)
            jax.block_until_ready(out)
            report["results"][name]["samples_ms"].append((time.perf_counter() - start) * 1000 / args.iters)
        print("round", rid + 1, flush=True)
        save()
    for name, row in report["results"].items():
        row["median_ms"] = float(np.median(row["samples_ms"]))
        row["ratio_to_standard"] = ratio_ci(row["samples_ms"], report["results"]["softmax_baseline"]["samples_ms"])
        print(name, row["median_ms"], flush=True)
    save()


if __name__ == "__main__":
    main()
