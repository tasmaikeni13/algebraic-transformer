#!/usr/bin/env python3
"""Run simultaneously on all four hosts of an idle 16-chip TPU v4 slice.

Uses JAX distributed initialization and explicit NamedSharding on a physical
mesh. Each timing sample ends with block_until_ready; compile/warmup, transfers,
host barriers, and gathering are outside the timed interval.
"""

import argparse
from pathlib import Path
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ["JAX_PLATFORMS"]="tpu,cpu"
os.environ["JAX_ENABLE_X64"]="0"

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from src.primitives import alu, avn
from scripts.primitives_experiments import deep_trials, gelu, rmsnorm, layernorm, summary
from scripts.primitives_records import environment, source_hashes, write_json
from tests.reference_primitives import alu_reference, alu_derivative, avn_reference, avn_vjp_reference


def comparative_benchmarks(place, replicated, repetitions=100):
    rows=[]
    raw={}
    for dtype in (jnp.float32,jnp.bfloat16):
        dtype_name=str(jnp.dtype(dtype))
        rng=np.random.default_rng(100+jax.process_index())
        host=rng.normal(size=(1024,4096)).astype(np.float32)
        x=place(host).astype(dtype)
        g=place(rng.normal(size=host.shape).astype(np.float32)).astype(dtype)
        gamma=jax.device_put(np.ones(4096,np.float32),replicated)
        beta=jax.device_put(np.zeros(4096,np.float32),replicated)
        arms={"alu":(alu,(x,)),"gelu":(gelu,(x,)),"swish":(jax.nn.silu,(x,)),
              "avn":(avn,(x,)),"rmsnorm":(rmsnorm,(x,gamma)),"layernorm":(layernorm,(x,gamma,beta))}
        compiled={}
        for name,(fn,args) in arms.items():
            forward=jax.jit(fn).lower(*args).compile()
            def both(*a,fn=fn):
                y,back=jax.vjp(fn,*a[:-1])
                return y,back(a[-1])
            backward=jax.jit(both).lower(*args,g).compile()
            for mode,executable,inputs in [("forward",forward,args),("forward_backward",backward,(*args,g))]:
                key=f"{name}_{mode}"
                compiled[key]=(executable,inputs)
                for _ in range(10):
                    jax.block_until_ready(executable(*inputs))
        latencies={key:[] for key in compiled}
        order_rng=np.random.default_rng(123)  # Identical order on every host.
        for rep in range(repetitions):
            for key in order_rng.permutation(list(compiled)):
                fn,args=compiled[key]
                multihost_utils.sync_global_devices(f"timing-{dtype_name}-{rep}-{key}")
                start=time.perf_counter_ns()
                jax.block_until_ready(fn(*args))
                elapsed=(time.perf_counter_ns()-start)*1e-9
                all_hosts=np.asarray(multihost_utils.process_allgather(np.array(elapsed)))
                latencies[key].append(float(all_hosts.max()))
        gates={}
        for mode in ("forward","forward_backward"):
            for alg,base,threshold in [("alu","gelu",.90),("avn","rmsnorm",.95)]:
                ratio=np.median(latencies[f"{base}_{mode}"])/np.median(latencies[f"{alg}_{mode}"])
                gates[f"{alg}_{mode}"]={"throughput_ratio":float(ratio),"minimum":threshold,"passed":bool(ratio>=threshold)}
        raw[dtype_name]=latencies
        rows.append({"dtype":dtype_name,"global_shape":list(x.shape),"repetitions":repetitions,"warmups":10,
                     "latency_seconds":{key:summary(value) for key,value in latencies.items()},
                     "gates":gates,"passed":all(g["passed"] for g in gates.values()),
                     "parameter_bytes":{"avn":0,"rmsnorm":gamma.nbytes,"layernorm":gamma.nbytes+beta.nbytes}})
        if jax.process_index()==0:
            print(dtype_name,{key:round(g["throughput_ratio"],3) for key,g in gates.items()},flush=True)
    return {"rows":rows,"passed":all(r["passed"] for r in rows)},raw


def device_parity(place):
    rows=[]
    rng=np.random.default_rng(142+jax.process_index())
    for dtype in (jnp.float32,jnp.bfloat16):
        for scale in (.1,1.,10.,1e15):
            host=(rng.normal(size=(16,256))*scale).astype(np.float32)
            x=place(host).astype(dtype)
            g=jnp.ones_like(x)
            # Each local shard's oracle uses its actual rounded input.
            for name,fn,ref,deriv in [("alu",alu,alu_reference,lambda a:alu_derivative(a)),
                                      ("avn",avn,avn_reference,lambda a:avn_vjp_reference(a,np.ones_like(a)))]:
                y,back=jax.vjp(fn,x)
                dx=jax.block_until_ready(back(g)[0])
                local=[]
                for xs,ys,gs in zip(x.addressable_shards,y.addressable_shards,dx.addressable_shards):
                    a=np.asarray(xs.data,dtype=np.float64)
                    yy=np.asarray(ys.data,dtype=np.float64); gg=np.asarray(gs.data,dtype=np.float64)
                    oracle=ref(a); oracle_g=deriv(a)
                    tol=2e-5 if dtype==jnp.float32 else .016
                    yerr=np.max(np.abs(yy-oracle)/(1+np.abs(oracle)))
                    gerr=np.max(np.abs(gg-oracle_g)/(1+np.abs(oracle_g)))
                    local.append([yerr,gerr,float(np.isfinite(yy).all() and np.isfinite(gg).all())])
                values=np.asarray(multihost_utils.process_allgather(np.asarray(local)))
                rows.append({"dtype":str(jnp.dtype(dtype)),"scale":scale,"primitive":name,
                             "forward_scaled_error":float(values[...,0].max()),"vjp_scaled_error":float(values[...,1].max()),
                             "tolerance":tol,"passed":bool(np.all(values[...,:2]<=tol) and np.all(values[...,2]==1))})
    return {"rows":rows,"passed":all(r["passed"] for r in rows)}


def tpu_monte_carlo(place):
    """1e6 scalars/scale, distributed across the 16 chips; fp64 host oracle."""
    rng=np.random.default_rng(242+jax.process_index())
    rows=[]
    for scale in (.1,.2,.5,1.,2.,5.,10.):
        # 4 rows/host x 62500 features x 4 hosts = 1e6 scalars per scale.
        host=(rng.normal(size=(4,62500))*scale).astype(np.float32)
        for eps in (1e-5,0.):
            x=place(host)
            y=jax.block_until_ready(avn(x,eps))
            local=[]
            for xs,ys in zip(x.addressable_shards,y.addressable_shards):
                xx=np.asarray(xs.data,dtype=np.float64); yy=np.asarray(ys.data,dtype=np.float64)
                m2=np.mean(xx*xx,axis=-1); v=np.var(xx,axis=-1)
                local.extend(zip(np.var(yy,axis=-1), np.abs(np.mean(yy*yy,axis=-1)-m2/(m2+eps)),
                                 np.abs(np.var(yy,axis=-1)-v/(m2+eps))))
            all_values=np.asarray(multihost_utils.process_allgather(np.asarray(local))).reshape(-1,3)
            stats=summary(all_values[:,0])
            unit_gate=.9999<=stats["ci95"][0] and stats["ci95"][1]<=1.0001
            identity_error=float(all_values[:,1:].max())
            rows.append({"scale":scale,"eps":eps,"variance":stats,"identity_error":identity_error,
                         "fp32_tolerance":2e-5,"ideal_unit_variance_gate":unit_gate if eps==0 else None,
                         "passed":identity_error<=2e-5 and (eps>0 or unit_gate)})
    return {"samples_per_scale":1_000_000,"rows":rows,"passed":all(r["passed"] for r in rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/primitives/tpu")
    args=parser.parse_args()
    jax.distributed.initialize(initialization_timeout=120)
    devices=jax.devices()
    if len(devices)!=16 or jax.process_count()!=4 or any("TPU v4" not in d.device_kind for d in devices):
        raise RuntimeError("Primitive study requires exactly 16 TPU v4 chips on four processes.")
    mesh=Mesh(mesh_utils.create_device_mesh((2,2,4),devices,allow_split_physical_axes=True),
              ("data","fsdp","model"))
    sharding=NamedSharding(mesh,P(("data","fsdp","model")))
    replicated=NamedSharding(mesh,P())
    def place(a):
        return jax.make_array_from_process_local_data(sharding,a)
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    records={"study":1,"gate_version":2,"environment":environment(),"device_count":len(devices),
             "process_count":jax.process_count(),"mesh_shape":dict(mesh.shape),"command":sys.argv,
             "devices":[{"id":d.id,"kind":d.device_kind,"process":d.process_index,"coords":list(d.coords)} for d in devices]}
    records["parity"]=device_parity(place)
    records["monte_carlo"]=tpu_monte_carlo(place)
    records["benchmarks"],latencies=comparative_benchmarks(place,replicated)
    records["deep"],raw=deep_trials(place=place,gather=multihost_utils.process_allgather,
        process_index=jax.process_index(),process_count=jax.process_count(),batch_size=100,
        progress=(lambda msg:print(msg,flush=True)) if jax.process_index()==0 else lambda msg:None)
    records["passed"]=all(records[k]["passed"] for k in ("parity","monte_carlo","benchmarks","deep"))
    if source_hashes()!=records["environment"]["source_sha256"]:
        records["passed"]=False
        records["error"]="Source changed during execution."
    if jax.process_index()==0:
        write_json(out/"metrics.json",records)
        write_json(out/"latencies.json",latencies)
        np.savez_compressed(out/"deep-trials.npz",**raw)
        print("TPU PASS" if records["passed"] else "TPU FAIL",flush=True)
    multihost_utils.sync_global_devices("primitives-complete")
    jax.distributed.shutdown()
    return 0 if records["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
