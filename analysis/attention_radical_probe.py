#!/usr/bin/env python3
"""Keep exact asinh semantics; measure alternative radical evaluations."""

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
import numpy as np

from analysis.attention_performance import ratio_ci
from algebraic import algebraic as A
from algebraic.kernels import flash as fk


def make_radical(original, mode, bound=1.5):
    coefficients = None
    sqrt_coefficients = None
    if mode.startswith("poly"):
        t = np.linspace(0., bound * bound, 4097)
        fit = np.polynomial.Chebyshev.fit(t, 1/np.sqrt(1+t), int(mode[4]))
        coefficients = fit.convert(kind=np.polynomial.Polynomial).coef.astype(np.float32)
        if mode.endswith("direct") or mode.endswith("sqrt_reciprocal"):
            sqrt_coefficients = np.polynomial.Chebyshev.fit(
                t, np.sqrt(1+t), int(mode[4])).convert(kind=np.polynomial.Polynomial).coef.astype(np.float32)
        if mode.endswith("anchored_reciprocal"):
            # (sqrt(1+t)-1)/t = 1/(sqrt(1+t)+1), including t=0.
            residual = np.polynomial.Chebyshev.fit(
                t, 1/(np.sqrt(1+t)+1), int(mode[4])-1).convert(kind=np.polynomial.Polynomial).coef
            sqrt_coefficients = np.asarray([1., *residual], np.float32)
    def rho(x, fast):
        if not fast:
            return original(x, False)
        rad = 1. + x * x
        if mode == "sqrt_reciprocal":
            root = lax.sqrt(rad)
            return x + root, pl.reciprocal(root, approx=True)
        if coefficients is not None:
            t = x*x
            inv = jnp.full_like(t,float(coefficients[-1]))
            for coefficient in reversed(coefficients[:-1]):
                inv = inv*t+float(coefficient)
            if sqrt_coefficients is not None:
                root = jnp.full_like(t, float(sqrt_coefficients[-1]))
                for coefficient in reversed(sqrt_coefficients[:-1]):
                    root = root*t+float(coefficient)
                return x + root, (pl.reciprocal(root, approx=True) if mode.endswith("reciprocal") else inv)
            if mode.endswith("shared"):
                return x + rad * inv, inv
        elif mode.startswith("bit_"):
            bits = lax.bitcast_convert_type(rad, jnp.uint32)
            seed_bits = jnp.uint32(0x5f375a86) - lax.shift_right_logical(bits, jnp.uint32(1))
            inv = lax.bitcast_convert_type(seed_bits, jnp.float32)
        else:
            inv = lax.rsqrt(rad.astype(jnp.bfloat16)).astype(jnp.float32)
        iterations = 3 if "nr3" in mode else 2 if "nr2" in mode else 1
        for _ in range(iterations):
            inv = inv * (.5 * (3. - rad * inv * inv))
        root = rad * inv
        if "stable" in mode:
            t = jnp.abs(x) + root
            return jnp.where(x >= 0., t, pl.reciprocal(t, approx=True)), inv
        return x + root, inv
    rho.coefficients = None if coefficients is None else coefficients.tolist()
    rho.sqrt_coefficients = None if sqrt_coefficients is None else sqrt_coefficients.tolist()
    return rho


def bounded_attention(q,k,v,sink, *, bound=1.5):
    """Outer guard for polynomial evaluation on a specified score interval."""
    B,H,T,D=q.shape
    qs=(q.astype(jnp.float32)/(8*np.sqrt(D))).astype(q.dtype)
    si=jnp.broadcast_to(jnp.reshape(sink,(-1,)),(H,))/8
    q2=jnp.max(jnp.sum(qs.astype(jnp.float32)**2,-1))
    k2=jnp.max(jnp.sum(k.astype(jnp.float32)**2,-1))
    safe=(q2*k2 <= (bound/1.01)**2) & (jnp.max(jnp.abs(si)) <= bound/1.01)
    safe=safe & (jnp.max(jnp.abs(v.astype(jnp.float32))) <= 1e8)
    var=fk.make_variant("asinh",16,8.,T,False,True,True,True,False)
    var=var._replace(resident_bwd=q.dtype==jnp.bfloat16 and T in (512,1024,2048) and D==64)
    low=var._replace(shift=False,inv_stat=True,bounded_scores=True)
    high=var._replace(shift=True,inv_stat=False,fast_rho=False,bounded_scores=False)
    return lax.cond(safe,lambda _:fk._flash(qs,k,v,si,low,512,512,True),
                    lambda _:fk._flash(qs,k,v,si,high,512,512,True),None)


def validate(dtype, seq=512, attention=None):
    shape = (1, 2, seq, 64)
    q, k, v, g = [jax.random.normal(key, shape, dtype) for key in
                  jax.random.split(jax.random.PRNGKey(91), 4)]
    sink = jnp.array([-.35, .125], jnp.float32).reshape(1, 2, 1, 1)
    if attention is not None:
        q,k = A.rms_norm(q,None),A.rms_norm(k,None)
    fn = attention or (lambda q, k, v, s: fk.flash_attention(q, k, v, s, kind="asinh", n=16,
                                             c=8., block=512, range_guard=True, head_resident=False))
    ref = lambda q, k, v, s: A.asinh_attention(q, k, v, n=16, c=8., sink_logit=s)
    loss = lambda f, q, k, v, s: jnp.sum(f(q, k, v, s).astype(jnp.float32) * g.astype(jnp.float32))
    with jax.default_matmul_precision("highest" if dtype == jnp.float32 else "default"):
        got, want = jax.jit(fn)(q, k, v, sink), jax.jit(ref)(q, k, v, sink)
        oa = np.asarray(got, np.float32)
        ob = np.asarray(want, np.float32)
        ga = jax.jit(jax.grad(lambda q, k, v, s: loss(fn, q, k, v, s), argnums=(0, 1, 2, 3)))(q, k, v, sink)
        gb = jax.jit(jax.grad(lambda q, k, v, s: loss(ref, q, k, v, s), argnums=(0, 1, 2, 3)))(q, k, v, sink)
        errors = [float(np.linalg.norm(np.asarray(a, np.float32) - np.asarray(b, np.float32)) /
                        max(np.linalg.norm(np.asarray(b, np.float32)), 1e-6)) for a, b in zip(ga, gb)]
    output_error = float(np.max(np.abs(oa - ob)))
    return {"shape": shape, "dtype": str(jnp.dtype(dtype)), "sink": [-.35, .125],
            "qk_normalized":attention is not None,
            "output_max_abs_error": output_error, "q_k_v_sink_relative_l2_errors": errors,
            "passed": output_error < (.003 if dtype == jnp.float32 else .02) and
                      max(errors) < (.005 if dtype == jnp.float32 else .02)}


def grid_error(bound=15.8):
    xs = np.linspace(-bound, bound, 65536, dtype=np.float32).reshape(512, 128)
    def kernel(x_ref, rho_ref, inv_ref):
        rho, inv = fk._rho(x_ref[...], True)
        rho_ref[...] = rho
        inv_ref[...] = inv
    out = pl.pallas_call(kernel, out_shape=(jax.ShapeDtypeStruct(xs.shape, jnp.float32),
                                         jax.ShapeDtypeStruct(xs.shape, jnp.float32)))(jnp.asarray(xs))
    rho, inv = [np.asarray(x, np.float64) for x in out]
    x64 = xs.astype(np.float64)
    root = np.sqrt(1 + x64 * x64)
    t = np.abs(x64) + root
    exact = np.where(x64 >= 0, t, 1. / t)
    weight_relative = np.abs((rho / exact) ** 16 - 1.)
    return {"points": xs.size, "range": [-bound, bound],
            "rho_max_relative_error": float(np.max(np.abs(rho / exact - 1.))),
            "inverse_root_max_relative_error": float(np.max(np.abs(inv * root - 1.))),
            "weight_max_relative_error": float(weight_relative.max()),
            "weight_q99_relative_error": float(np.quantile(weight_relative, .99)),
            "nonpositive_rho": int(np.sum(rho <= 0)),
            "scope": "finite TPU grid versus float64 oracle, not a uniform certificate"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--qk-normalized", action="store_true")
    p.add_argument("--score-bound", type=float, default=1.5,
                   help="Norm-guard domain for polynomial candidates; error certification is separate.")
    p.add_argument("--modes", nargs="+", default=["sqrt_reciprocal", "bit_nr2_stable", "bit_nr3_fast"],
                   choices=["sqrt_reciprocal", "bf16_nr1_fast", "bf16_nr2_fast",
                            "bf16_nr1_stable", "bf16_nr2_stable", "bit_nr2_fast", "bit_nr2_stable",
                            "bit_nr3_fast", "bit_nr3_stable", "poly2_nr2", "poly3_nr1", "poly4_nr1",
                            "poly2_direct", "poly3_direct", "poly4_direct", "poly5_direct",
                            "poly3_shared", "poly4_shared", "poly5_shared", "poly6_shared",
                            "poly2_anchored_reciprocal", "poly3_anchored_reciprocal", "poly4_anchored_reciprocal",
                            "poly3_sqrt_reciprocal", "poly4_sqrt_reciprocal"],
                   help="The installed TPU v4 compiler aborts on bfloat16 rsqrt; those modes require another target.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    shape = (8, 12, 2048, 64)
    q, k, v, g = [jax.random.normal(key, shape, jnp.bfloat16) for key in
                  jax.random.split(jax.random.PRNGKey(97), 4)]
    if args.qk_normalized:
        q,k = A.rms_norm(q,None),A.rms_norm(k,None)
    sink = jnp.zeros((12,), jnp.float32)
    report = {"utc": datetime.now(timezone.utc).isoformat(), "shape": shape,
              "jax": jax.__version__, "devices": [str(x) for x in jax.devices()],
              "rounds": args.rounds, "iters": args.iters, "seed": 97,
              "qk_normalized":args.qk_normalized,
              "score_bound": args.score_bound,
              "modes": args.modes,
              "prototype": True, "model_adopted": False,
              "kernel_sha256": hashlib.sha256((ROOT / "src/algebraic/kernels/flash.py").read_bytes()).hexdigest(),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "results": {}, "validation": {}, "grid_error": {}, "failures": {}, "approximation": {}}
    functions = {}
    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    def compile_case(name, fn):
        for mode in ("forward", "training"):
            start = time.perf_counter()
            f = (jax.jit(fn).lower(q, k, v).compile() if mode == "forward" else
                 jax.jit(jax.grad(lambda q, k, v: jnp.sum(fn(q, k, v).astype(jnp.float32) * g.astype(jnp.float32)),
                                  argnums=(0, 1, 2))).lower(q, k, v).compile())
            for _ in range(3):
                jax.block_until_ready(f(q, k, v))
            functions[name, mode] = f
            report["results"].setdefault(name, {})[mode] = {"samples_ms": [], "compile_s": time.perf_counter() - start}
            print(f"compiled {name} {mode}", flush=True)
            save()
    asinh = lambda q, k, v: fk.flash_attention(q, k, v, sink, kind="asinh", n=16, c=8., range_guard=True)
    original = fk._rho
    try:
        compile_case("standard_resident", lambda q, k, v: fk.flash_attention(q, k, v))
        compile_case("asinh_resident", asinh)
        for name in args.modes:
            fk._rho = make_radical(original, name, args.score_bound)
            jax.clear_caches()
            try:
                attention = partial(bounded_attention, bound=args.score_bound) if name.startswith("poly") else None
                report["approximation"][name] = {"rsqrt_seed_coefficients_f32":fk._rho.coefficients,
                                                  "sqrt_coefficients_f32": fk._rho.sqrt_coefficients,
                                                  "domain_abs_x":args.score_bound if attention else 15.8}
                if attention:
                    qs=(q.astype(jnp.float32)/64).astype(q.dtype)
                    bound2=float(jnp.max(jnp.sum(qs.astype(jnp.float32)**2,-1))*
                                 jnp.max(jnp.sum(k.astype(jnp.float32)**2,-1)))
                    report["approximation"][name]["timing_score_bound_squared"]=bound2
                    report["approximation"][name]["timing_uses_polynomial_branch"]=bound2<=(args.score_bound/1.01)**2
                report["grid_error"][name] = grid_error(args.score_bound if attention else 15.8)
                checks = [validate(jnp.float32,attention=attention), validate(jnp.bfloat16, seq=2048,attention=attention)]
                report["validation"][name] = checks
                print(name, report["grid_error"][name], checks, flush=True)
                if all(x["passed"] for x in checks):
                    compile_case(name, (lambda q,k,v,attention=attention:attention(q,k,v,sink)) if attention else asinh)
            except Exception as exc:
                report["failures"][name] = {"type": type(exc).__name__, "message": str(exc)}
                print(f"FAILED {name}: {str(exc).splitlines()[0]}", flush=True)
            save()
    finally:
        fk._rho = original
    cases = list(functions)
    rng = np.random.default_rng(97)
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
