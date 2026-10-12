#!/usr/bin/env python3
"""Measure algebraic vocabulary-head VJPs that retain the logit gradient.

The n=8,c=2 likelihood and clipping policy match the deployed lean head.
Float32 caching changes only the saved residual. BF16 caching is a separate
approximate candidate whose matrix-product gradients must be checked.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from analysis.attention_performance import ratio_ci
from src.sota import algebraic as A, standard as S


def cached_likelihood(cache_dtype):
    def parts(logits,targets):
        z=logits.astype(jnp.float32)
        w,r=A._lean_parts(z,8,2.,A.likelihood_clamp(z.shape[-1],8))
        den=jnp.sum(w,axis=-1,keepdims=True)
        p=w/den
        pt=jnp.take_along_axis(p,targets[:,None],axis=-1)[:,0]
        y=(jnp.arange(z.shape[-1])[None,:]==targets[:,None]).astype(jnp.float32)
        gradient=((4./z.shape[0])*r*(p-y)).astype(cache_dtype)
        return (jnp.mean(1.-pt),pt),gradient
    @jax.custom_vjp
    def likelihood(logits,targets):
        return parts(logits,targets)[0]
    def fwd(logits,targets):
        out,dz=parts(logits,targets)
        return out,(dz,)
    def bwd(res,cot):
        return (cot[0]*res[0].astype(jnp.float32)),None
    likelihood.defvjp(fwd,bwd)
    return likelihood


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens",type=int,default=16384)
    p.add_argument("--dim",type=int,default=768)
    p.add_argument("--vocab",type=int,default=50257)
    p.add_argument("--rounds",type=int,default=9)
    p.add_argument("--iters",type=int,default=10)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    keys=jax.random.split(jax.random.PRNGKey(149),3)
    h=jax.random.normal(keys[0],(a.tokens,a.dim),jnp.bfloat16)
    e=.02*jax.random.normal(keys[1],(a.vocab,a.dim),jnp.float32)
    t=jax.random.randint(keys[2],(a.tokens,),0,a.vocab)
    heads={"standard_ce":S.cross_entropy,
           "algebraic_lean":lambda z,t:A.algebraic_likelihood_lean(z,t,8,2.),
           "algebraic_cache_f32":cached_likelihood(jnp.float32),
           "algebraic_cache_bf16":cached_likelihood(jnp.bfloat16)}
    report={"schema":1,"utc":datetime.now(timezone.utc).isoformat(),"seed":149,
            "shape":[a.tokens,a.dim,a.vocab],"rounds":a.rounds,"iters":a.iters,
            "jax":jax.__version__,"devices":[str(d) for d in jax.devices()],
            "prototype":True,"model_adopted":False,"head_n":8,"head_c":2.,
            "sources":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                       ("analysis/head_gradient_cache_probe.py","src/sota/algebraic.py","src/sota/standard.py")},
            "validation":{},"results":{},"failures":{}}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    def save():
        a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    functions={}
    for name,head in heads.items():
        # Independent small checks include nonunit objective cotangents.
        if name.startswith("algebraic_cache"):
            shape=(128,2053)
            z=5*jax.random.normal(keys[0],shape,jnp.float32)
            target=jax.random.randint(keys[1],(128,),0,shape[-1])
            checks=[]
            for scale in (1.,.375):
                dz=jax.jit(jax.grad(lambda z:scale*head(z,target)[0]))(z)
                ref=jax.jit(jax.grad(lambda z:scale*A.algebraic_likelihood_lean(z,target,8,2.)[0]))(z)
                error=float(jnp.linalg.norm(dz-ref)/jnp.maximum(jnp.linalg.norm(ref),1e-6))
                checks.append({"cotangent":scale,"logit_relative_l2_error":error,
                               "passed":error<(1e-6 if name.endswith("f32") else .005)})
            report["validation"][name]=checks
            print(name,checks,flush=True)
            if not all(c["passed"] for c in checks):
                save()
                continue
        def loss(h,e,t,head=head):
            z=jnp.matmul(h,e.astype(h.dtype).T,preferred_element_type=jnp.float32)
            return head(z,t)
        try:
            start=time.perf_counter()
            f=jax.jit(jax.value_and_grad(loss,argnums=(0,1),has_aux=True)).lower(h,e,t).compile()
            for _ in range(3):
                jax.block_until_ready(f(h,e,t))
            functions[name]=f
            report["results"][name]={"compile_s":time.perf_counter()-start,"samples_ms":[]}
            print("compiled "+name,flush=True)
        except Exception as exc:
            report["failures"][name]={"type":type(exc).__name__,"message":str(exc)}
        save()
    rng=np.random.default_rng(149)
    names=list(functions)
    for rid in range(a.rounds):
        for ix in rng.permutation(len(names)):
            name=names[int(ix)]
            f=functions[name]
            jax.block_until_ready(f(h,e,t))
            start=time.perf_counter()
            for _ in range(a.iters):
                out=f(h,e,t)
            jax.block_until_ready(out)
            report["results"][name]["samples_ms"].append((time.perf_counter()-start)*1000/a.iters)
        print(f"round {rid+1}/{a.rounds}",flush=True)
        save()
    for name,item in report["results"].items():
        item["median_ms"]=float(np.median(item["samples_ms"]))
        item["ratio_to_standard"]=ratio_ci(item["samples_ms"],report["results"]["standard_ce"]["samples_ms"])
        print(name,item["median_ms"],flush=True)
    save()


if __name__=="__main__":
    main()
