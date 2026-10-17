#!/usr/bin/env python3
"""Reduce live backward intermediates and test independent MXU/VPU overlap.

All candidates retain the original algebraic weight and real VJP formulas.
Prefactor variants move a multiplication before centering and round the dV
operand early. Independent dense VJPs and preceding-kernel VJPs are checked
before randomized paired timings. Neither the deployed kernel nor the standard
dispatch is mutated outside this process.
"""

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from algebraic import algebraic as A
from algebraic.kernels import flash as fk
from algebraic.kernels import head_resident as head
from analysis.attention_performance import ratio_ci


def make_tile(original,prefactor=False,order="late"):
    def tile(q,k,v,do,stat,di,i,j,*,var,bq,bk,masked):
        if not (var.head_resident and var.kind=="asinh" and var.bounded_scores and var.inv_stat):
            return original(q,k,v,do,stat,di,i,j,var=var,bq=bq,bk=bk,masked=masked)
        if order=="dpfirst":
            dp=lax.dot_general(do,v,fk.TRANS_B,preferred_element_type=jnp.float32)
        s=lax.dot_general(q,k,fk.TRANS_B,preferred_element_type=jnp.float32)
        if order=="early":
            dp=lax.dot_general(do,v,fk.TRANS_B,preferred_element_type=jnp.float32)
        rho,r=fk._variant_rho(s,var)
        p=fk._ipow(rho,var.n)*stat
        if masked:p=jnp.where(fk._causal_mask(i,j,bq,bk),p,0.)
        if prefactor:
            slope_factor=p*r
            p=p.astype(q.dtype)
        if order=="late":
            dp=lax.dot_general(do,v,fk.TRANS_B,preferred_element_type=jnp.float32)
        ds=slope_factor*(dp-di) if prefactor else (p*(dp-di))*r
        return p,ds
    return tile


def attention(q,k,v,sink):
    return fk.flash_attention(q,k,v,sink,kind="asinh",n=16,c=8.,range_guard=True)


def validate(seed,gain=1.,negative=False,baseline=None):
    shape=(1,2,2048,64)
    q,k,v,g=[jax.random.normal(key,shape,jnp.bfloat16) for key in
             jax.random.split(jax.random.PRNGKey(seed),4)]
    q,k=A.rms_norm(q,None)*gain,A.rms_norm(k,None)*gain
    sink=jnp.array([-.35,.125],jnp.float32)
    if negative:
        q=(1+.05*q.astype(jnp.float32)).astype(q.dtype)
        k=(-1+.05*k.astype(jnp.float32)).astype(k.dtype)
        sink=None
    ref=lambda q,k,v,s:A.asinh_attention(q,k,v,n=16,c=8.,sink_logit=None if s is None else s[None,:,None,None])
    loss=lambda f,q,k,v,s:jnp.sum(f(q,k,v,s).astype(jnp.float32)*g.astype(jnp.float32))
    argnums=(0,1,2) if sink is None else (0,1,2,3)
    output=jax.jit(attention)(q,k,v,sink)
    expected=jax.jit(ref)(q,k,v,sink)
    got=jax.jit(jax.grad(partial(loss,attention),argnums=argnums))(q,k,v,sink)
    want=jax.jit(jax.grad(partial(loss,ref),argnums=argnums))(q,k,v,sink)
    errors=[float(np.linalg.norm(np.asarray(a,np.float32)-np.asarray(b,np.float32))/max(np.linalg.norm(np.asarray(b,np.float32)),1e-6)) for a,b in zip(got,want)]
    joint=float(np.sqrt(sum(np.sum((np.asarray(a,np.float32)-np.asarray(b,np.float32))**2) for a,b in zip(got,want)) /
                        max(sum(np.sum(np.asarray(b,np.float32)**2) for b in want),1e-12)))
    output_error=float(jnp.max(jnp.abs(output.astype(jnp.float32)-expected.astype(jnp.float32))))
    result=dict(seed=seed,gain=gain,negative=negative,output_max_abs_error=output_error,
                q_k_v_sink_relative_l2=errors,joint_vjp_relative_l2=joint,
                passed=output_error<.02 and joint<.025 and (negative or max(errors)<.025))
    if baseline is not None:
        previous=baseline(q,k,v,sink)
        changes=[float(np.linalg.norm(np.asarray(a,np.float32)-np.asarray(b,np.float32))/max(np.linalg.norm(np.asarray(b,np.float32)),1e-6)) for a,b in zip(got,previous)]
        result["preceding_vjp_relative_l2"]=changes
        result["passed"] &= max(changes)<.003
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds",type=int,default=9)
    p.add_argument("--iters",type=int,default=30)
    p.add_argument("--seed",type=int,default=617)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    shape=(8,12,2048,64)
    q,k,v,g=[jax.random.normal(key,shape,jnp.bfloat16) for key in jax.random.split(jax.random.PRNGKey(args.seed),4)]
    sink=jnp.zeros((12,),jnp.float32)
    original=fk._tile_grads
    report=dict(utc=datetime.now(timezone.utc).isoformat(),shape=shape,jax=jax.__version__,
                devices=[str(d) for d in jax.devices()],seed=args.seed,rounds=args.rounds,iters=args.iters,
                source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                               [Path(__file__),Path(fk.__file__),Path(head.__file__)]},
                results={},validation={},failures={},comparisons={})
    compiled={}
    def save():
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    def compile_case(name,fn):
        start=time.perf_counter()
        exe=jax.jit(jax.grad(lambda q,k,v:jnp.sum(fn(q,k,v).astype(jnp.float32)*g.astype(jnp.float32)),argnums=(0,1,2))).lower(q,k,v).compile()
        for _ in range(3):jax.block_until_ready(exe(q,k,v))
        compiled[name]=exe
        report["results"][name]=dict(compile_s=time.perf_counter()-start,samples_ms=[])
        print("compiled",name,flush=True);save()
    try:
        compile_case("standard_head",lambda q,k,v:fk.flash_attention(q,k,v,head_resident=True))
        compile_case("algebraic_head",lambda q,k,v:attention(q,k,v,sink))
        # Each reference executable is compiled while the original Python
        # function is installed; later substitutions cannot change its VJP.
        references={}
        for negative in (False,True):
            test_shape=(1,2,2048,64)
            a=jax.ShapeDtypeStruct(test_shape,jnp.bfloat16)
            s=None if negative else jax.ShapeDtypeStruct((2,),jnp.float32)
            cg=jnp.ones(test_shape,jnp.bfloat16)
            f=lambda q,k,v,s,cg=cg:jnp.sum(attention(q,k,v,s).astype(jnp.float32)*cg.astype(jnp.float32))
            # The validation function uses an arbitrary cotangent, so take
            # it as an explicit argument in the retained reference instead.
            f=lambda q,k,v,s,g:jnp.sum(attention(q,k,v,s).astype(jnp.float32)*g.astype(jnp.float32))
            references[negative]=jax.jit(jax.grad(f,argnums=(0,1,2) if negative else (0,1,2,3))).lower(a,a,a,s,a).compile()
        for prefactor in (False,True):
            for order in ("late","early","dpfirst"):
                if not prefactor and order=="late":continue
                name=("prefactor_" if prefactor else "original_")+order
                fk._tile_grads=make_tile(original,prefactor,order)
                jax.clear_caches()
                try:
                    checks=[]
                    for seed,gain,negative in [(619,1.,False),(631,2.,False),(641,1.,True)]:
                        gcheck=jax.random.normal(jax.random.split(jax.random.PRNGKey(seed),4)[3],(1,2,2048,64),jnp.bfloat16)
                        old=lambda q,k,v,s,negative=negative,gcheck=gcheck:references[negative](q,k,v,s,gcheck)
                        checks.append(validate(seed,gain,negative,old))
                    report["validation"][name]=checks
                    if not all(check["passed"] for check in checks):raise ValueError(str(checks))
                    compile_case(name,lambda q,k,v:attention(q,k,v,sink))
                except Exception as exc:
                    report["failures"][name]=str(exc)
                    print("failed",name,str(exc).splitlines()[0],flush=True);save()
    finally:
        fk._tile_grads=original
    rng=np.random.default_rng(args.seed+1)
    for rnd in range(args.rounds):
        names=list(compiled);rng.shuffle(names)
        for name in names:
            exe=compiled[name];jax.block_until_ready(exe(q,k,v));start=time.perf_counter()
            for _ in range(args.iters):out=exe(q,k,v)
            jax.block_until_ready(out)
            report["results"][name]["samples_ms"].append((time.perf_counter()-start)*1000/args.iters)
        save();print("round",rnd+1,flush=True)
    for name,row in report["results"].items():
        row["median_ms"]=float(np.median(row["samples_ms"]))
        report["comparisons"][name]=dict(over_standard=ratio_ci(row["samples_ms"],report["results"]["standard_head"]["samples_ms"]),
                                       over_previous=ratio_ci(row["samples_ms"],report["results"]["algebraic_head"]["samples_ms"]))
        print(name,row["median_ms"],flush=True)
    save()


if __name__=="__main__":main()
