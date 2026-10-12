#!/usr/bin/env python3
"""Isolated Pallas rational/affine-power probe; does not change the deployed model.

Compile baseline executables first, replace two module functions locally,
clear trace caches, then compile candidates. Cached executables are retained
as AOT callables so changed Python globals cannot change the baseline.
"""

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
import numpy as np

from analysis.attention_performance import ratio_ci
from src.sota.kernels import flash as fk


def powers_pair(x, n):
    """Return x^n,x^(n-1), sharing an addition chain with fewer live arrays."""
    if n == 16:
        x2 = x * x
        x3 = x2 * x
        x6 = x3 * x3
        x12 = x6 * x6
        x15 = x12 * x3
        return x15 * x, x15
    if n == 64:
        x2 = x * x
        x3 = x2 * x
        x6 = x3 * x3
        x7 = x6 * x
        x14 = x7 * x7
        x21 = x14 * x7
        x42 = x21 * x21
        x63 = x42 * x21
        return x63 * x, x63
    if n == 32:
        x2 = x * x
        x3 = x2 * x
        x5 = x3 * x2
        x10 = x5 * x5
        x15 = x10 * x5
        x30 = x15 * x15
        x31 = x30 * x
        return x31 * x, x31
    return fk._ipow(x, n), fk._ipow(x, n - 1)


def install_link(kind, lo, hi, mode):
    original_tile = fk._tile_grads

    def parts(x):
        xc = jnp.clip(x, lo, hi)
        if kind == "quadratic":
            rho = 1. + xc + .5 * xc * xc
            numerator = 1. + xc
        elif kind == "binomial":
            rho = 1. + xc
            numerator = 1.
        else:
            inv = 1. / (1. - xc)
            rho = (1. + xc) * inv
            slope = 2. / (1. - xc * xc)
        if kind in ("quadratic", "binomial"):
            if mode == "rsqrt":
                root_inv = lax.rsqrt(rho)
                inverse = root_inv * root_inv
            elif mode in ("reciprocal", "approx", "newton"):
                inverse = pl.reciprocal(rho, approx=mode != "reciprocal")
                if mode == "newton":
                    inverse = inverse * (2. - rho * inverse)
            else:
                inverse = 1. / rho
            slope = numerator * inverse
        slope = jnp.where((x > lo) & (x < hi), slope, 0.)
        return rho, slope

    fk._rho_exact = parts
    fk._rho = lambda x, fast: parts(x)
    if mode in ("product", "chain"):
        def tile(q, k, v, do, stat, di, i, j, *, var, bq, bk, masked):
            if var.kind == "softmax":
                return original_tile(q, k, v, do, stat, di, i, j,
                                     var=var, bq=bq, bk=bk, masked=masked)
            s = lax.dot_general(q, k, fk.TRANS_B, preferred_element_type=jnp.float32)
            x = jnp.clip(s, lo, hi)
            if kind == "quadratic":
                rho = 1. + x + .5 * x * x
                factor = 1. + x
            elif kind == "binomial":
                rho = 1. + x
                factor = 1.
            else:
                inv = 1. / (1. - x)
                rho = (1. + x) * inv
                factor = 2. * (inv * inv)
            if mode == "chain":
                w, dwbase = powers_pair(rho, var.n)
            else:
                w, dwbase = fk._ipow(rho, var.n), fk._ipow(rho, var.n - 1)
            p = w * stat
            dp = lax.dot_general(do, v, fk.TRANS_B, preferred_element_type=jnp.float32)
            dw = factor * dwbase * stat
            ds = dw * (dp - di)
            ds = jnp.where((s > lo) & (s < hi), ds, 0.)
            if masked:
                mask = fk._causal_mask(i, j, bq, bk)
                p = jnp.where(mask, p, 0.)
                ds = jnp.where(mask, ds, 0.)
            return p, ds
        fk._tile_grads = tile


def probe_attention(q, k, v, sink, n, c, block):
    # These replacement links have their own intrinsic clipping bounds. The
    # asinh-specific outer overflow cap is redundant, so skip it in this
    # isolated adapter; no runtime score-bound claim is made for the new link.
    qs = (q.astype(jnp.float32) / (math.sqrt(q.shape[-1]) * c)).astype(q.dtype)
    var = fk.make_variant("asinh", n, c, q.shape[-2], shift=False,
                          fast_rho=True, inv_stat=True, compact_stat=True)._replace(bounded_scores=True)
    return fk._flash(qs, k, v, sink.reshape((-1,)) / c, var, block, block, True)


def reference(q, k, v, sink, n, c, kind, lo, hi):
    qs = (q.astype(jnp.float32) / (math.sqrt(q.shape[-1]) * c)).astype(q.dtype)
    x = jnp.einsum("bhqd,bhkd->bhqk", qs, k, preferred_element_type=jnp.float32)
    x = jnp.clip(x, lo, hi)
    base = (1. + x + .5 * x * x if kind == "quadratic" else
            1. + x if kind == "binomial" else (1. + x) / (1. - x))
    w = fk._ipow(base, n)
    T = q.shape[-2]
    w = jnp.where(jnp.tril(jnp.ones((T, T), bool))[None, None], w, 0.)
    b = jnp.clip(sink.reshape((1, q.shape[1], 1, 1)) / c, lo, hi)
    bbase = (1. + b + .5 * b * b if kind == "quadratic" else
             1. + b if kind == "binomial" else (1. + b) / (1. - b))
    mass = fk._ipow(bbase, n)
    p = w / (w.sum(-1, keepdims=True) + mass)
    return jnp.einsum("bhqk,bhkd->bhqd", p.astype(v.dtype), v,
                      preferred_element_type=jnp.float32).astype(q.dtype)


def validate(n, c, kind, lo, hi, dtype=jnp.float32):
    keys = jax.random.split(jax.random.PRNGKey(93), 4)
    shape = (1, 2, 512, 64)
    q, k, v, g = [jax.random.normal(key, shape, dtype) for key in keys]
    sink = jnp.zeros((2,), jnp.float32)
    got = lambda q, k, v, s: probe_attention(q, k, v, s, n, c, 128)
    ref = lambda q, k, v, s: reference(q, k, v, s, n, c, kind, lo, hi)
    with jax.default_matmul_precision("highest" if dtype == jnp.float32 else "default"):
        o = jax.jit(got)(q, k, v, sink)
        expected = jax.jit(ref)(q, k, v, sink)
        output_error = float(jnp.max(jnp.abs(o - expected)))
        loss = lambda fn, q, k, v, s: jnp.sum(fn(q, k, v, s).astype(jnp.float32) * g.astype(jnp.float32))
        ga = jax.jit(jax.grad(lambda q, k, v, s: loss(got, q, k, v, s), argnums=(0, 1, 2, 3)))(q, k, v, sink)
        gb = jax.jit(jax.grad(lambda q, k, v, s: loss(ref, q, k, v, s), argnums=(0, 1, 2, 3)))(q, k, v, sink)
        errors = [float(jnp.linalg.norm(a.astype(jnp.float32) - b.astype(jnp.float32)) /
                        jnp.maximum(jnp.linalg.norm(b.astype(jnp.float32)), 1e-6)) for a, b in zip(ga, gb)]
    passed = output_error < (.003 if dtype == jnp.float32 else .02) and max(errors) < (.005 if dtype == jnp.float32 else .02)
    return {"shape": shape, "dtype": str(jnp.dtype(dtype)), "output_max_abs_error": output_error,
            "q_k_v_sink_relative_l2_errors": errors, "passed": passed}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--resident-chunk", type=int, default=0)
    p.add_argument("--resident-query-block", type=int, default=512)
    p.add_argument("--links", nargs="+", choices=["pade", "binomial", "quadratic"],
                   default=["pade", "binomial"])
    p.add_argument("--modes", nargs="+", choices=["slope", "product", "chain", "rsqrt", "reciprocal", "approx", "newton"])
    p.add_argument("--validate-bf16", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.resident_chunk:
        from analysis.attention_resident_probe import resident_backward
        fk._backward_fused = partial(resident_backward, chunk=args.resident_chunk,
                                    query_block=args.resident_query_block)
    shape = (8, args.heads, 2048, 64)
    keys = jax.random.split(jax.random.PRNGKey(37), 4)
    q, k, v, g = [jax.random.normal(key, shape, jnp.bfloat16) for key in keys]
    sink = jnp.zeros((args.heads,), jnp.float32)
    report = {"utc": datetime.now(timezone.utc).isoformat(), "shape": shape, "jax": jax.__version__,
              "devices": [str(x) for x in jax.devices()], "seed": 37, "rounds": args.rounds,
              "iters": args.iters, "prototype": True, "model_adopted": False,
              "kernel_sha256": hashlib.sha256((ROOT / "src/sota/kernels/flash.py").read_bytes()).hexdigest(),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "backward": "resident" if args.resident_chunk else "deployed_partials",
              "resident_chunk": args.resident_chunk,
              "resident_query_block": args.resident_query_block,
              "resident_probe_sha256": hashlib.sha256((ROOT / "analysis/attention_resident_probe.py").read_bytes()).hexdigest(),
              "results": {}, "validation": {}}
    functions = {}
    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    def compile_case(name, fn):
        for mode in ("forward", "training"):
            start = time.perf_counter()
            if mode == "forward":
                f = jax.jit(fn).lower(q, k, v).compile()
            else:
                def loss(q, k, v):
                    return jnp.sum(fn(q, k, v).astype(jnp.float32) * g.astype(jnp.float32))
                f = jax.jit(jax.grad(loss, argnums=(0, 1, 2))).lower(q, k, v).compile()
            for _ in range(3):
                jax.block_until_ready(f(q, k, v))
            functions[name, mode] = f
            report["results"].setdefault(name, {})[mode] = {"samples_ms": [], "compile_s": time.perf_counter() - start}
            print(f"compiled {name} {mode}", flush=True)
            save()
    # Freeze baselines before any Python global is replaced.
    compile_case("standard_compact", lambda q, k, v: fk.flash_attention(q, k, v, compact_stat=True, resident_bwd=False))
    compile_case("asinh_guarded", lambda q, k, v: fk.flash_attention(q, k, v, sink, kind="asinh", n=16,
                                                                   c=8., compact_stat=True, range_guard=True, resident_bwd=False))
    original_rho, original_exact, original_tile = fk._rho, fk._rho_exact, fk._tile_grads
    try:
        for kind, n, c, lo, hi in (("pade", 8, 16., -.75, .75),
                                  ("binomial", 32, 32., -.75, .75),
                                  ("binomial", 64, 64., -.5, .75),
                                  ("binomial", 64, 32., -.5, .75),
                                  ("quadratic", 16, 8., -1., 8.)):
            if kind not in args.links:
                continue
            modes = ("slope", "product") if kind == "pade" else ("slope", "product", "chain")
            if kind == "binomial":
                modes += ("rsqrt",)
            if args.modes:
                modes = tuple(args.modes)
            for mode in modes:
                fk._rho, fk._rho_exact, fk._tile_grads = original_rho, original_exact, original_tile
                install_link(kind, lo, hi, mode)
                jax.clear_caches()
                name = f"{kind}{n}_c{c:g}_{lo:g}_{hi:g}_{mode}"
                report["validation"][name] = validate(n, c, kind, lo, hi)
                if args.validate_bf16:
                    check = validate(n, c, kind, lo, hi, jnp.bfloat16)
                    report["validation"][name]["bfloat16"] = check
                    report["validation"][name]["passed"] &= check["passed"]
                print(name, report["validation"][name], flush=True)
                if not report["validation"][name]["passed"]:
                    save()
                    continue
                fn = lambda q, k, v, n=n, c=c: probe_attention(q, k, v, sink, n, c, 512)
                compile_case(name, fn)
    finally:
        fk._rho, fk._rho_exact, fk._tile_grads = original_rho, original_exact, original_tile
    cases = list(functions)
    rng = np.random.default_rng(37)
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
        print(f"round {rid + 1}/{args.rounds}", flush=True)
        save()
    for name in report["results"]:
        for mode in ("forward", "training"):
            a = report["results"][name][mode]["samples_ms"]
            report["results"][name][mode]["median_ms"] = float(np.median(a))
            report["results"][name][mode]["ratio_to_standard"] = ratio_ci(a, report["results"]["standard_compact"][mode]["samples_ms"])
        print(name, report["results"][name]["forward"]["median_ms"], report["results"][name]["training"]["median_ms"], flush=True)
    save()


if __name__ == "__main__":
    main()
