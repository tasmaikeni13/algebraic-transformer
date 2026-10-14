#!/usr/bin/env python3
"""Measure whole-head VMEM residency and static causal schedules.

These are process-local substitutions for the existing exact operator. Both
attention families receive the same schedules. No language model is trained.
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
from analysis.attention_performance import ratio_ci
from analysis.attention_radical_probe import validate


def validate_softmax(seed=541):
    from algebraic import standard as S
    shape = (1, 2, 2048, 64)
    q, k, v, g = [jax.random.normal(key, shape, jnp.bfloat16) for key in
                  jax.random.split(jax.random.PRNGKey(seed), 4)]
    results = []
    for sink in (None, jnp.array([-.35, .125], jnp.float32)):
        fn = lambda q,k,v: fk.flash_attention(q,k,v,sink,kind="softmax",head_resident=False)
        ref = lambda q,k,v: S.softmax_attention(q,k,v,sink_logit=None if sink is None else sink[None,:,None,None])
        outputs = [jax.jit(f)(q,k,v) for f in (fn,ref)]
        gradients = [jax.jit(jax.grad(lambda q,k,v,f=f: jnp.sum(f(q,k,v).astype(jnp.float32)*g.astype(jnp.float32)),
                                     argnums=(0,1,2)))(q,k,v) for f in (fn,ref)]
        errors = [float(np.linalg.norm(np.asarray(a,np.float32)-np.asarray(b,np.float32))/max(np.linalg.norm(np.asarray(b,np.float32)),1e-6)) for a,b in zip(*gradients)]
        output_error = float(jnp.max(jnp.abs(outputs[0].astype(jnp.float32)-outputs[1].astype(jnp.float32))))
        results.append(dict(seed=seed,shape=shape,has_sink=sink is not None,output_max_abs_error=output_error,
                            q_k_v_relative_l2_errors=errors,passed=output_error<.02 and max(errors)<.02))
    return results


def head_forward_kernel(sink_ref, q_ref, k_ref, v_ref, o_ref, stat_ref,
                        m_ref, l_ref, acc_ref, *, var, has_sink, T, D, static):
    bq = bk = 512
    count = T // bq
    h = pl.program_id(1)

    def query(i, unused):
        qs = pl.ds(i * bq, bq)
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
        unshifted = var.kind == "asinh" and not var.shift
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

        def key(j, unused):
            ks = pl.ds(j * bk, bk)
            q, k, v = q_ref[0, 0, qs, :], k_ref[0, 0, ks, :], v_ref[0, 0, ks, :]

            def step(masked):
                if unshifted and masked and var.diagonal_fwd:
                    p, row_sum = fk._diagonal_forward_weights(q, k, i, j, var=var)
                    l_ref[...] += row_sum
                else:
                    scores = lax.dot_general(q, k, fk.TRANS_B, preferred_element_type=jnp.float32)
                    mask = fk._causal_mask(i, j, bq, bk) if masked else None
                    if unshifted:
                        x = scores if var.bounded_scores else jnp.minimum(scores, var.clamp)
                        p = fk._ipow(fk._variant_rho(x, var)[0], var.n)
                        if masked:
                            p = jnp.where(mask, p, 0.)
                        l_ref[...] += jnp.sum(p, axis=1, keepdims=True)
                    else:
                        previous = m_ref[...]
                        tile_max = jnp.max(jnp.where(mask, scores, -jnp.inf) if masked else scores,
                                          axis=1, keepdims=True)
                        maximum = jnp.maximum(previous, tile_max)
                        if not has_sink:
                            maximum = jnp.where(j == 0, tile_max, maximum)
                            previous = jnp.where(j == 0, maximum, previous)
                        if var.kind == "softmax":
                            p = jnp.exp(scores - fk._bcast(maximum, bk))
                            alpha = jnp.exp(previous - maximum)
                        else:
                            shift = fk._rho_exact(-maximum)[0]
                            x = jnp.where(mask, scores, 0.) if masked else scores
                            p = fk._ipow(fk._rho(x, var.fast_rho)[0] * fk._bcast(shift, bk), var.n)
                            alpha = fk._ipow(fk._rho_exact(previous)[0] * shift, var.n)
                        if masked:
                            p = jnp.where(mask, p, 0.)
                        l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=1, keepdims=True)
                        acc_ref[...] *= fk._bcast(alpha, D)
                        m_ref[...] = maximum
                acc_ref[...] += lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32)

            if static:
                step(j == i)
            else:
                @pl.when(j < i)
                def full():
                    step(False)
                @pl.when(j == i)
                def diagonal():
                    step(True)
            return unused

        if static:
            for j in range(i + 1):
                key(j, None)
        else:
            lax.fori_loop(0, i + 1, key, None)
        inv = 1. / l_ref[...]
        o_ref[0, 0, qs, :] = (acc_ref[...] * fk._bcast(inv, D)).astype(o_ref.dtype)
        if var.kind == "softmax":
            stat = m_ref[...] + jnp.log(l_ref[...])
        elif unshifted:
            stat = inv if var.inv_stat else fk._inv_root(l_ref[...], var.n)
        else:
            stat = fk._rho_exact(-m_ref[...])[0] * fk._inv_root(l_ref[...], var.n)
        stat_ref[0, 0, 0, qs] = jnp.transpose(stat)[:1, :][0]
        return unused

    if static:
        for i in range(count):
            query(i, None)
    else:
        lax.fori_loop(0, count, query, None)


def head_forward(q, k, v, sink, *, var, bq, bk, has_sink, static, compute_q=512):
    B, H, T, D = q.shape
    spec = pl.BlockSpec((1, 1, T, D), lambda b, h, x, y, *_: (b, h, 0, 0))
    stat_spec = pl.BlockSpec((1, 1, 1, T), lambda b, h, x, y, *_: (b, h, 0, 0))
    o, stat = pl.pallas_call(
        partial(head_forward_kernel, var=var, has_sink=has_sink, T=T, D=D, static=static),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1, grid=(B, H, 1, 1), in_specs=[spec, spec, spec],
            out_specs=[spec, stat_spec],
            scratch_shapes=[pltpu.VMEM((compute_q, fk.LANES), jnp.float32),
                            pltpu.VMEM((compute_q, fk.LANES), jnp.float32),
                            pltpu.VMEM((compute_q, D), jnp.float32)]),
        out_shape=[jax.ShapeDtypeStruct(q.shape, q.dtype), jax.ShapeDtypeStruct((B, H, 1, T), jnp.float32)],
        compiler_params=fk._params())(sink, q, k, v)
    return o, stat[:, :, 0, :]


def head_backward_kernel(sink_ref, q_ref, k_ref, v_ref, do_ref, stat_ref, di_ref,
                         dk_ref, dv_ref, dq_ref, dk_sc, dv_sc, dq_sc, *, var, T, static):
    block = 512
    dk_sc[...] = jnp.zeros(dk_sc.shape, jnp.float32)
    dv_sc[...] = jnp.zeros(dv_sc.shape, jnp.float32)

    def query(i, unused):
        qs = pl.ds(i * block, block)
        dq_sc[...] = jnp.zeros(dq_sc.shape, jnp.float32)
        q, do = q_ref[0, 0, qs, :], do_ref[0, 0, qs, :]
        stat = stat_ref[0, 0, 0, qs][:, None]
        di = di_ref[0, 0, 0, qs][:, None]

        def key(j, unused):
            ks = pl.ds(j * block, block)
            k, v = k_ref[0, 0, ks, :], v_ref[0, 0, ks, :]

            def step(masked):
                tile = fk._diagonal_tile_grads if masked and var.diagonal_bwd else fk._tile_grads
                p, ds = tile(q, k, v, do, stat, di, i, j, var=var, bq=block, bk=block, masked=masked)
                ds = ds.astype(q.dtype)
                dv_sc[ks, :] += lax.dot_general(p.astype(do.dtype), do, fk.TRANS_A, preferred_element_type=jnp.float32)
                dk_sc[ks, :] += lax.dot_general(ds, q, fk.TRANS_A, preferred_element_type=jnp.float32)
                dq_sc[...] += lax.dot(ds, k, preferred_element_type=jnp.float32)

            if static:
                step(j == i)
            else:
                @pl.when(j < i)
                def full():
                    step(False)
                @pl.when(j == i)
                def diagonal():
                    step(True)
            return unused

        if static:
            for j in range(i + 1):
                key(j, None)
        else:
            lax.fori_loop(0, i + 1, key, None)
        dq_ref[0, 0, qs, :] = (dq_sc[...] * (1. if var.kind == "softmax" else float(var.n))).astype(dq_ref.dtype)
        return unused

    if static:
        for i in range(T // block):
            query(i, None)
    else:
        lax.fori_loop(0, T // block, query, None)
    dk_ref[0, 0] = (dk_sc[...] * (1. if var.kind == "softmax" else float(var.n))).astype(dk_ref.dtype)
    dv_ref[0, 0] = dv_sc[...].astype(dv_ref.dtype)


def head_backward(q, k, v, sink, stat4, di4, do, di, *, var, static, compute_q=512):
    B, H, T, D = q.shape
    spec = pl.BlockSpec((1, 1, T, D), lambda b, h, x, y, *_: (b, h, 0, 0))
    stat_spec = pl.BlockSpec((1, 1, 1, T), lambda b, h, x, y, *_: (b, h, 0, 0))
    dk, dv, dq = pl.pallas_call(
        partial(head_backward_kernel, var=var, T=T, static=static),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1, grid=(B, H, 1, 1),
            in_specs=[spec, spec, spec, spec, stat_spec, stat_spec], out_specs=[spec, spec, spec],
            scratch_shapes=[pltpu.VMEM((T, D), jnp.float32), pltpu.VMEM((T, D), jnp.float32),
                            pltpu.VMEM((compute_q, D), jnp.float32)]),
        out_shape=[jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype),
                   jax.ShapeDtypeStruct(q.shape, q.dtype)], compiler_params=fk._params())(sink, q, k, v, do, stat4, di4)
    return dq, dk, dv, di


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--seed", type=int, default=503)
    p.add_argument("--schedules", nargs="+", default=["baseline", "fwd_dynamic", "fwd_static", "bwd_dynamic", "bwd_static", "both_static"],
                   choices=["baseline", "fwd_dynamic", "fwd_static", "bwd_dynamic", "bwd_static", "both_static"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    shape = (args.batch, args.heads, 2048, 64)
    q, k, v, g = [jax.random.normal(key, shape, jnp.bfloat16) for key in jax.random.split(jax.random.PRNGKey(args.seed), 4)]
    sink = jnp.zeros((args.heads,), jnp.float32)
    original_fwd, original_bwd = fk._forward, fk._backward_resident
    report = dict(utc=datetime.now(timezone.utc).isoformat(), shape=shape, jax=jax.__version__,
                  devices=[str(d) for d in jax.devices()], seed=args.seed, rounds=args.rounds, iters=args.iters,
                  source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(fk.__file__)]},
                  results={}, validation={}, failures={}, comparisons={})
    compiled = {}

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    def compile_case(name, fn):
        report["results"][name] = {}
        for mode in ("forward", "training"):
            loss = lambda q,k,v: jnp.sum(fn(q,k,v).astype(jnp.float32) * g.astype(jnp.float32))
            start = time.perf_counter()
            try:
                exe = jax.jit(fn if mode == "forward" else jax.grad(loss, argnums=(0,1,2))).lower(q,k,v).compile()
                for _ in range(3):
                    jax.block_until_ready(exe(q,k,v))
                compiled[name,mode] = exe
                report["results"][name][mode] = dict(compile_s=time.perf_counter()-start,samples_ms=[])
                print("compiled",name,mode,flush=True)
            except Exception as exc:
                report["results"][name][mode] = dict(error=str(exc))
                print("failed",name,mode,str(exc).splitlines()[0],flush=True)
            save()

    try:
        for schedule in args.schedules:
            fk._forward = original_fwd if schedule in ("baseline","bwd_dynamic","bwd_static") else partial(head_forward,static=schedule != "fwd_dynamic")
            fk._backward_resident = original_bwd if schedule in ("baseline","fwd_dynamic","fwd_static") else partial(head_backward,static=schedule != "bwd_dynamic")
            jax.clear_caches()
            for kind in ("softmax", "asinh"):
                name = kind + "_" + schedule
                try:
                    # The existing reference helper checks the deployed algebraic
                    # operator, all input gradients, and the learned sink.
                    if schedule != "baseline":
                        checks = ([validate(jnp.bfloat16,seq=2048)] if kind == "asinh" else validate_softmax())
                        report["validation"][name] = checks
                        if not all(check["passed"] for check in checks):
                            raise ValueError(str(checks))
                    fn = lambda q,k,v,kind=kind: fk.flash_attention(q,k,v,sink if kind == "asinh" else None,
                                                                  kind=kind,n=16,c=8.,range_guard=kind == "asinh",
                                                                  head_resident=False)
                    compile_case(name,fn)
                except Exception as exc:
                    report["failures"][name] = str(exc)
                    print("failed",name,str(exc).splitlines()[0],flush=True)
                    save()
    finally:
        fk._forward, fk._backward_resident = original_fwd, original_bwd
    rng = np.random.default_rng(args.seed+1)
    for rnd in range(args.rounds):
        cases = list(compiled)
        rng.shuffle(cases)
        for case in cases:
            exe = compiled[case]
            jax.block_until_ready(exe(q,k,v))
            start = time.perf_counter()
            for _ in range(args.iters):
                output = exe(q,k,v)
            jax.block_until_ready(output)
            report["results"][case[0]][case[1]]["samples_ms"].append((time.perf_counter()-start)*1000/args.iters)
        save()
        print("round",rnd+1,flush=True)
    for mode in ("forward","training"):
        names = [name for name, m in compiled if m == mode]
        for name in names:
            report["results"][name][mode]["median_ms"] = float(np.median(report["results"][name][mode]["samples_ms"]))
            print(name,mode,report["results"][name][mode]["median_ms"],flush=True)
        best = min((n for n in names if n.startswith("softmax")),key=lambda n:report["results"][n][mode]["median_ms"])
        report["comparisons"][mode] = dict(reference=best,ratios={name:ratio_ci(report["results"][name][mode]["samples_ms"],report["results"][best][mode]["samples_ms"]) for name in names})
    save()


if __name__ == "__main__":
    main()
