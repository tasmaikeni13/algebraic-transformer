#!/usr/bin/env python3
"""Measure equivalent RMS VJPs in pure algebraic attention.

Process-local prototypes only. The forward formula and eps are unchanged.
The analytic VJP uses tau*(g - y*mean(g*y)), where y=x*tau before gain.
Gain gradients sum the original, unrounded normalized values.
"""

import argparse
from datetime import datetime, timezone
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
import numpy as np

from analysis.attention_performance import ratio_ci
from analysis.qk_rms_kernel import make_norm as make_pallas_norm
from algebraic import algebraic as A, standard as S
from algebraic.kernels.flash import flash_attention


def make_norm(store_normalized):
    @jax.custom_vjp
    def norm(x, gain):
        return A.rms_norm(x, gain)
    def fwd(x, gain):
        xf = x.astype(jnp.float32)
        tau = lax.rsqrt(jnp.mean(xf*xf, -1, keepdims=True)+1e-5)
        y = xf*tau
        return (y*gain).astype(x.dtype), (y if store_normalized else x, tau, gain)
    def bwd(res, do):
        saved, tau, gain = res
        y = saved if store_normalized else saved.astype(jnp.float32)*tau
        dg = do.astype(jnp.float32)*gain
        dx = tau*(dg-y*jnp.mean(dg*y, -1, keepdims=True))
        dgain = jnp.sum(do.astype(jnp.float32)*y, tuple(range(y.ndim-1)))
        return dx.astype(do.dtype), dgain.astype(gain.dtype)
    norm.defvjp(fwd, bwd)
    return norm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    shape = (8, 12, 2048, 64)
    q,k,v,g = [jax.random.normal(key,shape,jnp.bfloat16) for key in jax.random.split(jax.random.PRNGKey(137),4)]
    q,k = (q*.55).astype(q.dtype),(k*.55).astype(k.dtype)
    gains = jnp.ones((64,),jnp.float32)
    sink = jnp.zeros((12,),jnp.float32)
    trig,cayley = S.rope_table(64,2048),A.cayley_table_geometric(64,2048)
    report = {"schema":1,"utc":datetime.now(timezone.utc).isoformat(),"shape":shape,
              "seed":137,"rounds":args.rounds,"iters":args.iters,"prototype":True,
              "model_adopted":False,"jax":jax.__version__,"devices":[str(d) for d in jax.devices()],
              "sources":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                  ("analysis/qk_normalization_probe.py","analysis/qk_rms_kernel.py","src/algebraic/algebraic.py","src/algebraic/kernels/flash.py")},
              "validation":{},"results":{},"failures":{}}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    norms = {"autodiff":A.rms_norm,"analytic_save_x":make_norm(False),
             "analytic_save_y":make_norm(True), "pallas256":make_pallas_norm(256),
             "pallas512":make_pallas_norm(512), "pallas1024":make_pallas_norm(1024),
             "pallas1024_nativeforward":make_pallas_norm(1024,True,True),
             "pallas1024_gainpartials":make_pallas_norm(1024,gain_partials_only=True)}
    # Validate the norm directly, before placing it in a custom attention VJP.
    for name,norm in norms.items():
        checks=[]
        for dtype in (jnp.float32,jnp.bfloat16):
            keys=jax.random.split(jax.random.PRNGKey(139),3)
            x=jax.random.normal(keys[0],(2,3,512,64),dtype)*.55
            gain=.5+2*jax.random.uniform(keys[1],(64,),jnp.float32)
            cot=jax.random.normal(keys[2],x.shape,dtype)
            loss=lambda fn,x,ga:jnp.sum(fn(x,ga).astype(jnp.float32)*cot.astype(jnp.float32))
            try:
                got=jax.jit(jax.grad(lambda x,ga:loss(norm,x,ga),argnums=(0,1)))(x,gain)
                ref=jax.jit(jax.grad(lambda x,ga:loss(A.rms_norm,x,ga),argnums=(0,1)))(x,gain)
                oa,ob=jax.jit(norm)(x,gain),jax.jit(A.rms_norm)(x,gain)
                output_ok=np.allclose(np.asarray(oa,np.float32),np.asarray(ob,np.float32),
                                     atol=2e-6 if dtype==jnp.float32 else .01,
                                     rtol=2e-6 if dtype==jnp.float32 else .01)
            except Exception as exc:
                checks.append({"dtype":str(jnp.dtype(dtype)),"passed":False,
                               "error":f"{type(exc).__name__}: {exc}"})
                continue
            errors=[float(jnp.linalg.norm(a.astype(jnp.float32)-b.astype(jnp.float32))/
                          jnp.maximum(jnp.linalg.norm(b.astype(jnp.float32)),1e-6)) for a,b in zip(got,ref)]
            checks.append({"dtype":str(jnp.dtype(dtype)),"x_gain_relative_l2_errors":errors,
                           "output_max_abs_error":float(jnp.max(jnp.abs(oa.astype(jnp.float32)-ob.astype(jnp.float32)))),
                           "output_bitwise_equal":bool(jnp.array_equal(oa,ob)),
                           "passed":bool(output_ok) and max(errors)<(2e-6 if dtype==jnp.float32 else .005)})
        report["validation"][name]=checks
        print(name,checks,flush=True)
        save()
    compiled={}
    inputs=(q,k,v,gains,gains)
    for arch in ("algebraic",):
        for name,norm in norms.items():
            if not all(c["passed"] for c in report["validation"][name]):
                continue
            def attn(q,k,v,qg,kg,arch=arch,norm=norm):
                q,k=norm(q,qg),norm(k,kg)
                tables=cayley if arch=="algebraic" else trig
                q,k=A.apply_rotary(q,*tables),A.apply_rotary(k,*tables)
                return (flash_attention(q,k,v,sink,kind="asinh",n=16,c=8.,range_guard=True)
                        if arch=="algebraic" else flash_attention(q,k,v))
            loss=lambda *a,attn=attn:jnp.sum(attn(*a).astype(jnp.float32)*g.astype(jnp.float32))
            case=arch+"_"+name
            try:
                f=jax.jit(jax.grad(loss,argnums=(0,1,2,3,4))).lower(*inputs).compile()
                for _ in range(3):
                    jax.block_until_ready(f(*inputs))
                compiled[case]=f
                report["results"][case]={"samples_ms":[]}
                print("compiled "+case,flush=True)
            except Exception as exc:
                report["failures"][case]={"type":type(exc).__name__,"message":str(exc)}
            save()
    rng=np.random.default_rng(137)
    names=list(compiled)
    for rid in range(args.rounds):
        for ix in rng.permutation(len(names)):
            name=names[int(ix)]
            f=compiled[name]
            jax.block_until_ready(f(*inputs))
            start=time.perf_counter()
            for _ in range(args.iters):
                out=f(*inputs)
            jax.block_until_ready(out)
            report["results"][name]["samples_ms"].append((time.perf_counter()-start)*1000/args.iters)
        print(f"round {rid+1}/{args.rounds}",flush=True)
        save()
    for name,item in report["results"].items():
        item["median_ms"]=float(np.median(item["samples_ms"]))
        ref=name.split("_",1)[0]+"_autodiff"
        item["ratio_to_same_architecture"]=ratio_ci(item["samples_ms"],report["results"][ref]["samples_ms"])
        print(name,item["median_ms"],flush=True)
    save()


if __name__=="__main__":
    main()
