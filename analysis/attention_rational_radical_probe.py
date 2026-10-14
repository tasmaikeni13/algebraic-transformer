#!/usr/bin/env python3
"""Benchmark certified rational root candidates in the current whole-head path.

The fast branch evaluates the original rho-power weights and original slope
approximately. The score norm guard proves its domain; outside it the current
exact guarded kernel runs. Neither approximation nor dispatch is model code.
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
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np

from algebraic import algebraic as A
from algebraic.kernels import flash as fk
from algebraic.kernels import head_resident as head
from analysis.attention_performance import ratio_ci
from analysis.attention_radical_probe import validate
from analysis.attention_rational_radical import fit,certificate


def horner(t,coefficients):
    out=jnp.full_like(t,float(coefficients[-1]))
    for c in reversed(coefficients[:-1]):out=out*t+float(c)
    return out


def radical(x,p,q):
    t=x*x
    numerator,denominator=horner(t,p),horner(t,q)
    reciprocal=getattr(pltpu,"reciprocal",None) or pl.reciprocal
    root=numerator*reciprocal(denominator,approx=True)
    slope=denominator*reciprocal(numerator,approx=True)
    return x+root,slope


def guarded(q,k,v,sink,*,bound):
    B,H,T,D=q.shape
    qs=(q.astype(jnp.float32)/(8*np.sqrt(D))).astype(q.dtype)
    si=jnp.reshape(sink,(-1,))/8
    score_bound_squared=jnp.max(jnp.sum(qs.astype(jnp.float32)**2,-1))*jnp.max(jnp.sum(k.astype(jnp.float32)**2,-1))
    safe=(score_bound_squared<=(bound/1.01)**2)&(jnp.max(jnp.abs(si))<=bound/1.01)
    safe &= jnp.max(jnp.abs(v.astype(jnp.float32)))<=1e8
    block=512 if q.dtype==jnp.bfloat16 else 256
    var=fk.make_variant("asinh",16,8.,T,False,True,True,True,True)._replace(bounded_scores=True)
    resident=q.dtype==jnp.bfloat16 and T==2048 and D==64
    var=var._replace(resident_bwd=resident,head_resident=resident)
    return lax.cond(safe,lambda _:fk._flash(qs,k,v,si,var,block,block,True),
                    lambda _:fk.flash_attention(q,k,v,sink,kind="asinh",n=16,c=8.,range_guard=True),None)


def grid(p,q,bound):
    x=np.linspace(-bound,bound,65536,dtype=np.float32).reshape(512,128)
    def kernel(xref,rhoref,sloperef):
        rho,slope=radical(xref[...],p,q)
        rhoref[...]=rho;sloperef[...]=slope
    rho,slope=[np.asarray(a,np.float64) for a in pl.pallas_call(kernel,out_shape=[
        jax.ShapeDtypeStruct(x.shape,jnp.float32),jax.ShapeDtypeStruct(x.shape,jnp.float32)])(jnp.asarray(x))]
    x=x.astype(np.float64);root=np.sqrt(1+x*x)
    positive=np.abs(x)+root;truth=np.where(x>=0,positive,1/positive)
    return dict(points=x.size,weight_max_relative_error=float(np.max(abs((rho/truth)**16-1))),
                slope_max_relative_error=float(np.max(abs(slope*root-1))),
                nonpositive_rho=int(np.sum(rho<=0)),scope="finite TPU grid against float64; no uniform machine claim")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound",type=float,default=33/32)
    parser.add_argument("--degrees",nargs="+",default=["11","12","21","22"])
    parser.add_argument("--rounds",type=int,default=9)
    parser.add_argument("--iters",type=int,default=30)
    parser.add_argument("--seed",type=int,default=727)
    parser.add_argument("--unnormalized",action="store_true")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    shape=(8,12,2048,64)
    q,k,v,g=[jax.random.normal(key,shape,jnp.bfloat16) for key in jax.random.split(jax.random.PRNGKey(args.seed),4)]
    if not args.unnormalized:q,k=A.rms_norm(q,None),A.rms_norm(k,None)
    sink=jnp.zeros((12,),jnp.float32)
    qs=(q.astype(jnp.float32)/64).astype(q.dtype)
    bound2=float(jnp.max(jnp.sum(qs.astype(jnp.float32)**2,-1))*jnp.max(jnp.sum(k.astype(jnp.float32)**2,-1)))
    report=dict(utc=datetime.now(timezone.utc).isoformat(),shape=shape,jax=jax.__version__,
                devices=[str(d) for d in jax.devices()],seed=args.seed,rounds=args.rounds,iters=args.iters,
                qk_normalized=not args.unnormalized,domain_abs_x=args.bound,
                timing_score_bound_squared=bound2,timing_uses_approximation=bound2<=(args.bound/1.01)**2,
                prototype=True,model_adopted=False,source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                [Path(__file__),ROOT/"analysis/attention_rational_radical.py",Path(fk.__file__),Path(head.__file__)]},
                results={},validation={},coefficients={},certificates={},grid={},failures={},comparisons={})
    compiled={}
    def save():
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    def compile_case(name,fn):
        for mode in ("forward","training"):
            start=time.perf_counter()
            f=fn if mode=="forward" else jax.grad(lambda q,k,v:jnp.sum(fn(q,k,v).astype(jnp.float32)*g.astype(jnp.float32)),argnums=(0,1,2))
            exe=jax.jit(f).lower(q,k,v).compile()
            for _ in range(3):jax.block_until_ready(exe(q,k,v))
            compiled[name,mode]=exe
            report["results"].setdefault(name,{})[mode]=dict(compile_s=time.perf_counter()-start,samples_ms=[])
            print("compiled",name,mode,flush=True);save()
    original=fk._variant_rho
    try:
        compile_case("standard_head",lambda q,k,v:fk.flash_attention(q,k,v,head_resident=True))
        compile_case("algebraic_head",lambda q,k,v:fk.flash_attention(q,k,v,sink,kind="asinh",n=16,c=8.,range_guard=True))
        for degree in args.degrees:
            p,qcoeff=fit(args.bound,int(degree[0]),int(degree[1]))
            name="rational_"+degree
            def variant_rho(x,var,p=p,qcoeff=qcoeff):
                return radical(x,p,qcoeff) if var.poly_rho and var.bounded_scores else original(x,var)
            fk._variant_rho=variant_rho
            jax.clear_caches()
            try:
                report["coefficients"][name]=dict(p=p,q=qcoeff)
                report["certificates"][name]=certificate(p,qcoeff,args.bound)
                report["grid"][name]=grid(p,qcoeff,args.bound)
                fn=partial(guarded,bound=args.bound)
                checks=[validate(jnp.float32,attention=fn),validate(jnp.bfloat16,seq=2048,attention=fn)]
                report["validation"][name]=checks
                if not all(check["passed"] for check in checks):raise ValueError(str(checks))
                compile_case(name,lambda q,k,v,fn=fn:fn(q,k,v,sink))
            except Exception as exc:
                report["failures"][name]=str(exc)
                print("failed",name,str(exc).splitlines()[0],flush=True);save()
    finally:
        fk._variant_rho=original
    rng=np.random.default_rng(args.seed+1)
    for rnd in range(args.rounds):
        cases=list(compiled);rng.shuffle(cases)
        for name,mode in cases:
            exe=compiled[name,mode];jax.block_until_ready(exe(q,k,v));start=time.perf_counter()
            for _ in range(args.iters):out=exe(q,k,v)
            jax.block_until_ready(out)
            report["results"][name][mode]["samples_ms"].append((time.perf_counter()-start)*1000/args.iters)
        save();print("round",rnd+1,flush=True)
    for (name,mode),exe in compiled.items():
        row=report["results"][name][mode];row["median_ms"]=float(np.median(row["samples_ms"]))
        report["comparisons"][name+"_"+mode]=dict(over_standard=ratio_ci(row["samples_ms"],report["results"]["standard_head"][mode]["samples_ms"]),
                                                over_previous=ratio_ci(row["samples_ms"],report["results"]["algebraic_head"][mode]["samples_ms"]))
        print(name,mode,row["median_ms"],flush=True)
    save()


if __name__=="__main__":main()
