#!/usr/bin/env python3
"""Move row normalization through matrix products on the algebraic fast path.

The real identities are exact. BF16 operand rounding changes, so independent
output and q/k/v/sink VJP checks precede every timing. The second experiment
uses an extra value column of ones to reduce rounded weights on the MXU.
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

from algebraic import algebraic as A
from algebraic.kernels import flash as fk
from analysis.attention_performance import ratio_ci


def projective_backward_kernel(sink_ref, q_ref, k_ref, v_ref, do_ref, stat_ref, di_ref,
                              dk_ref, dv_ref, dq_ref, dk_sc, dv_sc, dq_sc, *,
                              var, bq, bk, nq, nk):
    i = pl.program_id(3)
    @pl.when(i == 0)
    def init():
        dk_sc[...] = jnp.zeros(dk_sc.shape, jnp.float32)
        dv_sc[...] = jnp.zeros(dv_sc.shape, jnp.float32)
    dq_sc[...] = jnp.zeros(dq_sc.shape, jnp.float32)
    q = q_ref[0, 0]
    inv = stat_ref[0, 0, 0][:, None]
    do = (do_ref[0, 0].astype(jnp.float32) * inv).astype(do_ref.dtype)
    di = di_ref[0, 0, 0][:, None] * inv
    unit = jnp.ones_like(inv)

    def key(j, unused):
        ks = pl.ds(j * bk, bk)
        k, v = k_ref[0, 0, ks, :], v_ref[0, 0, ks, :]
        def step(masked):
            tile = fk._diagonal_tile_grads if masked and var.diagonal_bwd else fk._tile_grads
            w, ds = tile(q, k, v, do, unit, di, i, j, var=var, bq=bq, bk=bk, masked=masked)
            ds = ds.astype(q.dtype)
            dv_sc[ks, :] += lax.dot_general(w.astype(do.dtype), do, fk.TRANS_A, preferred_element_type=jnp.float32)
            dk_sc[ks, :] += lax.dot_general(ds, q, fk.TRANS_A, preferred_element_type=jnp.float32)
            dq_sc[...] += lax.dot(ds, k, preferred_element_type=jnp.float32)
        @pl.when(j < i)
        def full():
            step(False)
        @pl.when(j == i)
        def diagonal():
            step(True)
        return unused
    lax.fori_loop(0, i + 1, key, None)
    dq_ref[0, 0] = (dq_sc[...] * var.n).astype(dq_ref.dtype)
    @pl.when(i == nq - 1)
    def finish():
        dk_ref[0, 0] = (dk_sc[...] * var.n).astype(dk_ref.dtype)
        dv_ref[0, 0] = dv_sc[...].astype(dv_ref.dtype)


def projective_backward(q,k,v,sink,stat4,di4,do,di, *, var):
    B,H,T,D = q.shape
    bq = bk = 512
    qm = lambda b,h,j,i,*_: (b,h,i,0)
    km = lambda b,h,j,i,*_: (b,h,0,0)
    rm = lambda b,h,j,i,*_: (b,h,0,i)
    dk,dv,dq = pl.pallas_call(
        partial(projective_backward_kernel,var=var,bq=bq,bk=bk,nq=T//bq,nk=T//bk),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,grid=(B,H,1,T//bq),
            in_specs=[pl.BlockSpec((1,1,bq,D),qm),pl.BlockSpec((1,1,T,D),km),
                      pl.BlockSpec((1,1,T,D),km),pl.BlockSpec((1,1,bq,D),qm),
                      pl.BlockSpec((1,1,1,bq),rm),pl.BlockSpec((1,1,1,bq),rm)],
            out_specs=[pl.BlockSpec((1,1,T,D),km),pl.BlockSpec((1,1,T,D),km),pl.BlockSpec((1,1,bq,D),qm)],
            scratch_shapes=[pltpu.VMEM((T,D),jnp.float32),pltpu.VMEM((T,D),jnp.float32),pltpu.VMEM((bq,D),jnp.float32)]),
        out_shape=[jax.ShapeDtypeStruct(k.shape,k.dtype),jax.ShapeDtypeStruct(v.shape,v.dtype),
                   jax.ShapeDtypeStruct(q.shape,q.dtype)],compiler_params=fk._params())(sink,q,k,v,do,stat4,di4)
    return dq,dk,dv,di


def augmented_forward_kernel(sink_ref,q_ref,k_ref,v_ref,o_ref,stat_ref,acc_ref, *, var,bq,bk,nk,has_sink,D):
    h,i,j = pl.program_id(1),pl.program_id(2),pl.program_id(3)
    last = fk._last_kblock(i,bq,bk,nk)
    @pl.when(j == 0)
    def init():
        acc_ref[...] = jnp.zeros(acc_ref.shape,jnp.float32)
        if has_sink:
            s = jnp.full((bq,1),sink_ref[h],jnp.float32)
            if not var.bounded_scores:
                s = jnp.minimum(s,var.clamp)
            mass = fk._ipow(fk._variant_rho(s,var)[0],var.n)
            acc_ref[:,D:D+1] = mass

    def step(masked):
        q,k,v = q_ref[0,0],k_ref[0,0],v_ref[0,0]
        if masked and var.diagonal_fwd:
            w,_ = fk._diagonal_forward_weights(q,k,i,j,var=var)
        else:
            s = lax.dot_general(q,k,fk.TRANS_B,preferred_element_type=jnp.float32)
            x = s if var.bounded_scores else jnp.minimum(s,var.clamp)
            w = fk._ipow(fk._variant_rho(x,var)[0],var.n)
            if masked:
                w = jnp.where(fk._causal_mask(i,j,bq,bk),w,0.)
        augmented = jnp.concatenate((v,jnp.ones((bk,1),v.dtype),jnp.zeros((bk,127-D),v.dtype)),axis=1)
        acc_ref[...] += lax.dot(w.astype(v.dtype),augmented,preferred_element_type=jnp.float32)
    @pl.when(j < i)
    def full():
        step(False)
    @pl.when(j == i)
    def diagonal():
        step(True)
    @pl.when(j == last)
    def finish():
        acc = acc_ref[...]
        inv = 1. / acc[:,D:D+1]
        o_ref[0,0] = (acc[:,:D]*inv).astype(o_ref.dtype)
        stat_ref[0,0] = jnp.transpose(inv)


def augmented_forward(q,k,v,sink, *, var,bq,bk,has_sink):
    B,H,T,D=q.shape
    qm = lambda b,h,i,j,*_: (b,h,i,0)
    km = lambda b,h,i,j,*_: (b,h,jnp.minimum(j,fk._last_kblock(i,bq,bk,T//bk)),0)
    rm = lambda b,h,i,j,*_: (b,h,0,i)
    out,stat = pl.pallas_call(
        partial(augmented_forward_kernel,var=var,bq=bq,bk=bk,nk=T//bk,has_sink=has_sink,D=D),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,grid=(B,H,T//bq,T//bk),
            in_specs=[pl.BlockSpec((1,1,bq,D),qm),pl.BlockSpec((1,1,bk,D),km),pl.BlockSpec((1,1,bk,D),km)],
            out_specs=[pl.BlockSpec((1,1,bq,D),qm),pl.BlockSpec((1,1,1,bq),rm)],
            scratch_shapes=[pltpu.VMEM((bq,128),jnp.float32)]),
        out_shape=[jax.ShapeDtypeStruct(q.shape,q.dtype),jax.ShapeDtypeStruct((B,H,1,T),jnp.float32)],
        compiler_params=fk._params())(sink,q,k,v)
    return out,stat[:,:,0,:]


def validate(seed, gain=1.):
    shape=(1,2,2048,64)
    q,k,v,g = [jax.random.normal(key,shape,jnp.bfloat16) for key in jax.random.split(jax.random.PRNGKey(seed),4)]
    q,k = A.rms_norm(q,None)*gain,A.rms_norm(k,None)*gain
    sink=jnp.array([-.35,.125],jnp.float32)
    fn=lambda q,k,v,s:fk.flash_attention(q,k,v,s,kind="asinh",n=16,c=8.,range_guard=True,head_resident=False)
    ref=lambda q,k,v,s:A.asinh_attention(q,k,v,n=16,c=8.,sink_logit=s[None,:,None,None])
    outputs=[jax.jit(f)(q,k,v,sink) for f in (fn,ref)]
    loss=lambda f,q,k,v,s:jnp.sum(f(q,k,v,s).astype(jnp.float32)*g.astype(jnp.float32))
    grads=[jax.jit(jax.grad(partial(loss,f),argnums=(0,1,2,3)))(q,k,v,sink) for f in (fn,ref)]
    errors=[float(np.linalg.norm(np.asarray(a,np.float32)-np.asarray(b,np.float32))/max(np.linalg.norm(np.asarray(b,np.float32)),1e-6)) for a,b in zip(*grads)]
    output_error=float(jnp.max(jnp.abs(outputs[0].astype(jnp.float32)-outputs[1].astype(jnp.float32))))
    return dict(seed=seed,gain=gain,shape=shape,output_max_abs_error=output_error,
                q_k_v_sink_relative_l2_errors=errors,passed=output_error<.02 and max(errors)<.02)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds",type=int,default=7)
    p.add_argument("--iters",type=int,default=25)
    p.add_argument("--seed",type=int,default=509)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    shape=(8,12,2048,64)
    q,k,v,g=[jax.random.normal(key,shape,jnp.bfloat16) for key in jax.random.split(jax.random.PRNGKey(args.seed),4)]
    sink=jnp.zeros((12,),jnp.float32)
    original_fwd,original_bwd=fk._forward,fk._backward_resident
    report=dict(utc=datetime.now(timezone.utc).isoformat(),shape=shape,jax=jax.__version__,
                devices=[str(d) for d in jax.devices()],seed=args.seed,rounds=args.rounds,iters=args.iters,
                source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(fk.__file__)]},
                results={},validation={},failures={},comparisons={})
    compiled={}
    def save():
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    def compile_case(name,fn):
        report["results"][name]={}
        for mode in ("forward","training"):
            loss=lambda q,k,v:jnp.sum(fn(q,k,v).astype(jnp.float32)*g.astype(jnp.float32))
            start=time.perf_counter()
            try:
                exe=jax.jit(fn if mode=="forward" else jax.grad(loss,argnums=(0,1,2))).lower(q,k,v).compile()
                for _ in range(3):jax.block_until_ready(exe(q,k,v))
                compiled[name,mode]=exe
                report["results"][name][mode]=dict(compile_s=time.perf_counter()-start,samples_ms=[])
                print("compiled",name,mode,flush=True)
            except Exception as exc:
                report["results"][name][mode]=dict(error=str(exc))
                print("failed",name,mode,str(exc).splitlines()[0],flush=True)
            save()
    try:
        compile_case("standard",lambda q,k,v:fk.flash_attention(q,k,v,head_resident=False))
        for case in ("baseline","projective_bwd","augmented_fwd","both"):
            def backward(*a,**kw):
                var=kw["var"]
                return (projective_backward if var.kind=="asinh" and not var.shift and var.inv_stat else original_bwd)(*a,**kw)
            def forward(*a,**kw):
                var=kw["var"]
                return (augmented_forward if var.kind=="asinh" and not var.shift and var.inv_stat else original_fwd)(*a,**kw)
            fk._backward_resident=backward if case in ("projective_bwd","both") else original_bwd
            fk._forward=forward if case in ("augmented_fwd","both") else original_fwd
            jax.clear_caches()
            try:
                checks=[validate(seed,gain) for seed,gain in [(511,1.),(521,2.)]]
                report["validation"][case]=checks
                if not all(c["passed"] for c in checks):raise ValueError(str(checks))
                compile_case(case,lambda q,k,v:fk.flash_attention(q,k,v,sink,kind="asinh",n=16,c=8.,range_guard=True,head_resident=False))
            except Exception as exc:
                report["failures"][case]=str(exc)
                print("failed",case,str(exc).splitlines()[0],flush=True)
                save()
    finally:
        fk._forward,fk._backward_resident=original_fwd,original_bwd
    rng=np.random.default_rng(args.seed+1)
    for rnd in range(args.rounds):
        cases=list(compiled);rng.shuffle(cases)
        for case in cases:
            exe=compiled[case];jax.block_until_ready(exe(q,k,v));start=time.perf_counter()
            for _ in range(args.iters):out=exe(q,k,v)
            jax.block_until_ready(out)
            report["results"][case[0]][case[1]]["samples_ms"].append((time.perf_counter()-start)*1000/args.iters)
        save();print("round",rnd+1,flush=True)
    for name,mode in compiled:
        row=report["results"][name][mode];row["median_ms"]=float(np.median(row["samples_ms"]))
        report["comparisons"][name+"_"+mode]=ratio_ci(row["samples_ms"],report["results"]["standard"][mode]["samples_ms"])
        print(name,mode,row["median_ms"],flush=True)
    save()


if __name__=="__main__":
    main()
