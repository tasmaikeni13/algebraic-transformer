#!/usr/bin/env python3
"""Wide-cap affine attention, checked clip-free branch and native reciprocal.

The deployed model is unchanged. All baselines and candidates use resident
backward, and the same QK-normalized arrays. Timing across gain products
records when the deterministic guard stops allowing the unclipped branch.
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
from jax.experimental.pallas import tpu as pltpu
import numpy as np

from analysis.attention_performance import ratio_ci
from src.sota import algebraic as A
from src.sota.kernels import flash as fk

LINKS = {"32:16":(32,16.,-.75,3.), "64:32":(64,32.,-.5,2.),
         "16:2":(16,2.,-.9375,32.), "8:1":(8,1.,-.999,64.),
         "8:2":(8,2.,-.999,32.)}


def install(lo, hi, round_stat):
    originals = fk._rho, fk._rho_exact, fk._forward, fk._tile_grads, fk._sink_mass_and_slope
    def parts(x, bounded):
        xc = x if bounded else jnp.clip(x, lo, hi)
        base = 1. + xc
        inverse = pl.reciprocal(base, approx=True)
        if not bounded:
            inverse = jnp.where((x > lo) & (x < hi), inverse, 0.)
        return base, inverse
    fk._rho = lambda x, fast: parts(x, fast)
    fk._rho_exact = lambda x: parts(x, False)

    def sink(s, stat, var):
        x = jnp.clip(s, lo, hi)
        base = 1. + x
        w = fk._ipow(base, var.n)
        slope = jnp.where((s > lo) & (s < hi), var.n / base, 0.)
        return w[None,:,None] * stat, slope
    fk._sink_mass_and_slope = sink
    if round_stat:
        def kernel(s_ref, q_ref, k_ref, v_ref, o_ref, stat_ref, m_sc, l_sc, acc_sc,
                   *, var, bq, bk, nk, has_sink, head_dim):
            h, i, j = pl.program_id(1), pl.program_id(2), pl.program_id(3)
            last = fk._last_kblock(i,bq,bk,nk)
            @pl.when(j == 0)
            def init():
                b = jnp.full(l_sc.shape,s_ref[h],jnp.float32)
                l_sc[...] = fk._ipow(fk._rho(b,var.fast_rho)[0],var.n)
                acc_sc[...] = jnp.zeros(acc_sc.shape,jnp.float32)
            def step(masked):
                s = lax.dot_general(q_ref[0,0],k_ref[0,0],fk.TRANS_B,preferred_element_type=jnp.float32)
                w = fk._ipow(fk._rho(s,var.fast_rho)[0],var.n)
                if masked:
                    w = jnp.where(fk._causal_mask(i,j,bq,bk),w,0.)
                wb = w.astype(v_ref.dtype)
                l_sc[...] += jnp.sum(wb.astype(jnp.float32),axis=1,keepdims=True)
                acc_sc[...] += lax.dot(wb,v_ref[0,0],preferred_element_type=jnp.float32)
            diag = fk._crosses_diagonal(i,j,bq,bk)
            @pl.when((j<=last)&~diag)
            def full():
                step(False)
            @pl.when((j<=last)&diag)
            def masked():
                step(True)
            @pl.when(j == last)
            def finish():
                inv = 1. / l_sc[...]
                o_ref[0,0] = (acc_sc[...] * fk._bcast(inv,head_dim)).astype(o_ref.dtype)
                stat_ref[0,0] = jnp.transpose(inv)[:1,:]
        def forward(q,k,v,s,*,var,bq,bk,has_sink):
            B,H,T,D = q.shape
            nq,nk = T//bq,T//bk
            def kv_map(b,h,i,j,*_):
                return b,h,jnp.minimum(j,fk._last_kblock(i,bq,bk,nk)),0
            o,stat = pl.pallas_call(partial(kernel,var=var,bq=bq,bk=bk,nk=nk,has_sink=has_sink,head_dim=D),
                grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1,grid=(B,H,nq,nk),
                    in_specs=[pl.BlockSpec((1,1,bq,D),lambda b,h,i,j,*_:(b,h,i,0)),
                              pl.BlockSpec((1,1,bk,D),kv_map),pl.BlockSpec((1,1,bk,D),kv_map)],
                    out_specs=[pl.BlockSpec((1,1,bq,D),lambda b,h,i,j,*_:(b,h,i,0)),
                               pl.BlockSpec((1,1,1,bq),lambda b,h,i,j,*_:(b,h,0,i))],
                    scratch_shapes=[pltpu.VMEM((bq,128),jnp.float32),pltpu.VMEM((bq,128),jnp.float32),
                                    pltpu.VMEM((bq,D),jnp.float32)]),
                out_shape=[jax.ShapeDtypeStruct(q.shape,q.dtype),jax.ShapeDtypeStruct((B,H,1,T),jnp.float32)],
                compiler_params=fk._params())(s,q,k,v)
            return o,stat[:,:,0,:]
        def tile(q,k,v,do,stat,di,i,j,*,var,bq,bk,masked):
            x = lax.dot_general(q,k,fk.TRANS_B,preferred_element_type=jnp.float32)
            base,r = parts(x,var.fast_rho)
            w = fk._ipow(base,var.n)
            p_grad = w*stat
            p_value = w.astype(v.dtype).astype(jnp.float32)*stat
            dp = lax.dot_general(do,v,fk.TRANS_B,preferred_element_type=jnp.float32)
            ds = p_grad*(dp-di)*r
            if masked:
                mask = fk._causal_mask(i,j,bq,bk)
                p_value,ds = jnp.where(mask,p_value,0.),jnp.where(mask,ds,0.)
            return p_value,ds
        fk._forward,fk._tile_grads = forward,tile
    return originals


def affine(q,k,v,sink,n,c,lo,hi):
    qs = (q.astype(jnp.float32)/(math.sqrt(q.shape[-1])*c)).astype(q.dtype)
    s = sink.reshape(-1)/c
    bound = min(-lo,hi)/1.01
    q2,k2 = jnp.max(jnp.sum(qs.astype(jnp.float32)**2,-1)),jnp.max(jnp.sum(k.astype(jnp.float32)**2,-1))
    in_range = (q2*k2 <= bound*bound)&(jnp.max(jnp.abs(s))<=bound)
    # This is an intrinsic clipped link; the guard only skips clipping where
    # Cauchy-Schwarz proves it has no effect. Both branches share its formula.
    var = fk.Variant("asinh",n=n,shift=False,fast_rho=True,inv_stat=True,compact_stat=True,
                     bounded_scores=True,resident_bwd=q.dtype==jnp.bfloat16)
    return lax.cond(in_range,
                    lambda _:fk._flash(qs,k,v,s,var,512,512,True),
                    lambda _:fk._flash(qs,k,v,s,var._replace(fast_rho=False),512,512,True),None)


def oracle(q,k,v,sink,n,c,lo,hi):
    qs = (q.astype(jnp.float32)/(math.sqrt(q.shape[-1])*c)).astype(q.dtype)
    x = jnp.einsum("bhqd,bhkd->bhqk",qs,k,preferred_element_type=jnp.float32)
    T = q.shape[-2]
    logw = n*jnp.log1p(jnp.clip(x,lo,hi))
    logw = jnp.where(jnp.tril(jnp.ones((T,T),bool))[None,None],logw,-jnp.inf)
    logsink = n*jnp.log1p(jnp.clip(sink.reshape(1,q.shape[1],1,1)/c,lo,hi))
    logsink = jnp.broadcast_to(logsink,logw.shape[:-1]+(1,))
    p = jax.nn.softmax(jnp.concatenate((logw,logsink),-1),-1)[...,:T]
    return jnp.einsum("bhqk,bhkd->bhqd",p.astype(v.dtype),v,preferred_element_type=jnp.float32).astype(q.dtype)


def validate(n,c,lo,hi,dtype,gain):
    shape = (1,2,512,64)
    q,k,v,g = [jax.random.normal(key,shape,dtype) for key in jax.random.split(jax.random.PRNGKey(113),4)]
    q,k = A.rms_norm(q,None),A.rms_norm(k,None)
    q,k = (q*math.sqrt(gain)).astype(dtype),(k*math.sqrt(gain)).astype(dtype)
    s = jnp.array([-.35,.125],jnp.float32).reshape(1,2,1,1)
    fn = lambda q,k,v,s:affine(q,k,v,s,n,c,lo,hi)
    ref = lambda q,k,v,s:oracle(q,k,v,s,n,c,lo,hi)
    with jax.default_matmul_precision("highest" if dtype==jnp.float32 else "default"):
        oa,ob = jax.jit(fn)(q,k,v,s),jax.jit(ref)(q,k,v,s)
        error = float(jnp.max(jnp.abs(oa.astype(jnp.float32)-ob.astype(jnp.float32))))
        loss = lambda f,q,k,v,s:jnp.sum(f(q,k,v,s).astype(jnp.float32)*g.astype(jnp.float32))
        ga = jax.jit(jax.grad(partial(loss,fn),argnums=(0,1,2,3)))(q,k,v,s)
        gb = jax.jit(jax.grad(partial(loss,ref),argnums=(0,1,2,3)))(q,k,v,s)
        errors = [float(jnp.linalg.norm(a.astype(jnp.float32)-b.astype(jnp.float32))/
                        jnp.maximum(jnp.linalg.norm(b.astype(jnp.float32)),1e-6)) for a,b in zip(ga,gb)]
        output_ok = np.allclose(np.asarray(oa,np.float32),np.asarray(ob,np.float32),
                                atol=.003 if dtype==jnp.float32 else .012,
                                rtol=.003 if dtype==jnp.float32 else .012)
    return {"dtype":str(jnp.dtype(dtype)),"gain_product":gain,"output_max_abs_error":error,
            "q_k_v_sink_relative_l2_errors":errors,"oracle":"stable temperature-exact log1p/softmax; no squared-denominator overflow",
            "passed":bool(output_ok) and max(errors)<(.005 if dtype==jnp.float32 else .02)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gains",nargs="+",type=float,default=[1.,1.5,2.,4.,8.])
    p.add_argument("--links",nargs="+",choices=list(LINKS),default=["32:16","64:32"])
    p.add_argument("--rounded-statistics",nargs="+",choices=["false","true"],default=["false","true"])
    p.add_argument("--rounds",type=int,default=9)
    p.add_argument("--iters",type=int,default=20)
    p.add_argument("--out",type=Path,required=True)
    args = p.parse_args()
    shape = (8,12,2048,64)
    q,k,v,g = [jax.random.normal(key,shape,jnp.bfloat16) for key in jax.random.split(jax.random.PRNGKey(113),4)]
    q,k = A.rms_norm(q,None),A.rms_norm(k,None)
    data = {gain:((q*math.sqrt(gain)).astype(q.dtype),(k*math.sqrt(gain)).astype(k.dtype),v) for gain in args.gains}
    sink = jnp.zeros((12,),jnp.float32)
    report = {"utc":datetime.now(timezone.utc).isoformat(),"shape":shape,"qk_normalized":True,"gains":args.gains,
              "seed":113,"rounds":args.rounds,"iters":args.iters,"jax":jax.__version__,"devices":[str(x) for x in jax.devices()],
              "prototype":True,"model_adopted":False,"kernel_sha256":hashlib.sha256((ROOT/"src/sota/kernels/flash.py").read_bytes()).hexdigest(),
              "probe_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"links":[LINKS[x] for x in args.links],
              "validation":{},"results":{},"failures":{},"domains":{}}
    functions = {}
    def save():
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    def compile_case(name,fn):
        for mode in ("forward","training"):
            loss = lambda q,k,v:jnp.sum(fn(q,k,v).astype(jnp.float32)*g.astype(jnp.float32))
            f = (jax.jit(fn) if mode=="forward" else jax.jit(jax.grad(loss,argnums=(0,1,2)))).lower(q,k,v).compile()
            jax.block_until_ready(f(q,k,v))
            functions[name,mode] = f
            report["results"].setdefault(name,{})[mode] = {str(gain):{"samples_ms":[]} for gain in args.gains}
            print(f"compiled {name} {mode}",flush=True)
            save()
    compile_case("standard_resident",lambda q,k,v:fk.flash_attention(q,k,v))
    compile_case("asinh_resident",lambda q,k,v:fk.flash_attention(q,k,v,sink,kind="asinh",n=16,c=8.,range_guard=True))
    for n,c,lo,hi in (LINKS[x] for x in args.links):
        for rounded in (x=="true" for x in args.rounded_statistics):
            name = f"affine{n}_c{c:g}_wide_guard_roundstat{rounded}"
            vmax=min(1e8,1e37/((shape[-2]+1)*(1+hi)**n))
            report["domains"][name]={"score_clip":[lo,hi],"value_magnitude_bound":vmax,
                "values_in_domain":bool(jnp.max(jnp.abs(v.astype(jnp.float32)))<=vmax),
                "gain_guard":{}}
            for gain,inputs in data.items():
                qs=(inputs[0].astype(jnp.float32)/(math.sqrt(shape[-1])*c)).astype(q.dtype)
                bound2=float(jnp.max(jnp.sum(qs.astype(jnp.float32)**2,-1))*
                             jnp.max(jnp.sum(inputs[1].astype(jnp.float32)**2,-1)))
                report["domains"][name]["gain_guard"][str(gain)]={"score_bound_squared":bound2,
                    "unclipped_branch":bound2 <= (min(-lo,hi)/1.01)**2}
            originals = install(lo,hi,rounded)
            jax.clear_caches()
            try:
                checks = [validate(n,c,lo,hi,jnp.float32,1.),validate(n,c,lo,hi,jnp.bfloat16,1.),
                          validate(n,c,lo,hi,jnp.bfloat16,8.)]
                report["validation"][name] = checks
                print(name,checks,flush=True)
                if all(x["passed"] for x in checks):
                    compile_case(name,lambda q,k,v,n=n,c=c,lo=lo,hi=hi:affine(q,k,v,sink,n,c,lo,hi))
            except Exception as exc:
                report["failures"][name] = {"type":type(exc).__name__,"message":str(exc)}
                print(f"FAILED {name}: {str(exc).splitlines()[0]}",flush=True)
            finally:
                fk._rho,fk._rho_exact,fk._forward,fk._tile_grads,fk._sink_mass_and_slope = originals
            save()
    cases = [(key,gain) for key in functions for gain in args.gains]
    rng = np.random.default_rng(113)
    for rid in range(args.rounds):
        for ix in rng.permutation(len(cases)):
            (name,mode),gain = cases[int(ix)]
            f = functions[name,mode]
            inp = data[gain]
            jax.block_until_ready(f(*inp))
            start = time.perf_counter()
            for _ in range(args.iters):
                out = f(*inp)
            jax.block_until_ready(out)
            report["results"][name][mode][str(gain)]["samples_ms"].append((time.perf_counter()-start)*1000/args.iters)
        print(f"round {rid+1}/{args.rounds}",flush=True)
        save()
    for name,result in report["results"].items():
        for mode,by_gain in result.items():
            for gain,timing in by_gain.items():
                timing["median_ms"] = float(np.median(timing["samples_ms"]))
                timing["ratio_to_standard"] = ratio_ci(timing["samples_ms"],report["results"]["standard_resident"][mode][gain]["samples_ms"])
        print(name,{g:result["training"][g]["median_ms"] for g in result["training"]},flush=True)
    save()


if __name__ == "__main__":
    main()
