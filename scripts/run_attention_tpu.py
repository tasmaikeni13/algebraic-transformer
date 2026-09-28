#!/usr/bin/env python3
"""Four-host Attention study execution and synchronized 16-chip TPU v4 benchmarks."""
import os
os.environ['JAX_PLATFORMS']='tpu,cpu';os.environ['JAX_ENABLE_X64']='0'
from pathlib import Path
import sys
import argparse
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils,multihost_utils as mh
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from src.attention import algebraic_softmax,_normalized_forward
from tests.reference_attention import attention,attention_vjp
from scripts.primitives_experiments import summary
from scripts.attention_records import environment,source_hashes,write_json
from scripts.attention_experiments import LENGTHS,w1


def parity(place):
    rng=np.random.default_rng(142+jax.process_index());rows=[]
    for dtype in (jnp.float32,jnp.bfloat16):
        for length in (64,128,512,4096):
            for scale in (0.,.1,1.,10.,1e15):
                for sink in (0.,.5):
                    host=(rng.normal(size=(16,length))*scale).astype(np.float32)
                    x=place(host).astype(dtype);g=place(rng.normal(size=host.shape).astype(np.float32)).astype(dtype)
                    fn=lambda a:algebraic_softmax(a,sink)
                    y,back=jax.vjp(fn,x);dx=jax.block_until_ready(back(g)[0]);local=[]
                    for xs,gs,ys,ds in zip(x.addressable_shards,g.addressable_shards,y.addressable_shards,dx.addressable_shards):
                        xx=np.asarray(xs.data,dtype=float);gg=np.asarray(gs.data,dtype=float)
                        yy=np.asarray(ys.data,dtype=float);dd=np.asarray(ds.data,dtype=float)
                        ref=attention(xx,sink);refg=attention_vjp(xx,gg,sink)
                        local.append([np.max(np.abs(yy-ref)/(1+np.abs(ref))),np.max(np.abs(dd-refg)/(1+np.abs(refg))),
                                      float(np.isfinite(yy).all() and np.isfinite(dd).all() and (yy>0).all()),
                                      np.max(yy.sum(-1)-1)])
                    a=np.asarray(mh.process_allgather(np.array(local))).reshape(-1,4)
                    tol=2e-5 if dtype==jnp.float32 else .016
                    # Roundoff tolerance, with the exact-real bound separately proved.
                    mass_tol=16*np.finfo(np.float32).eps if dtype==jnp.float32 else .008
                    passed=bool(np.all(a[:,:2]<=tol) and np.all(a[:,2]==1) and np.all(a[:,3]<=mass_tol))
                    rows.append({'dtype':str(jnp.dtype(dtype)),'length':length,'scale':scale,'sink':sink,
                                 'forward_scaled_error':float(a[:,0].max()),'vjp_scaled_error':float(a[:,1].max()),
                                 'mass_excess':float(a[:,3].max()),'tolerance':tol,'mass_tolerance':mass_tol,'passed':passed})
    return {'rows':rows,'passed':all(r['passed'] for r in rows)}


def benchmarks(place):
    rows=[];raw={};hlo={}
    for dtype in (jnp.float32,jnp.bfloat16):
        for length in LENGTHS[1:]:
            rng=np.random.default_rng(242+jax.process_index())
            shape=(2**22//length,length)
            x=place(rng.normal(size=shape).astype(np.float32)).astype(dtype)
            g=place(rng.normal(size=shape).astype(np.float32)).astype(dtype)
            calls={}
            for name,fn in [('algebraic',algebraic_softmax),('softmax',jax.nn.softmax)]:
                def both(a,b,fn=fn):
                    y,vjp=jax.vjp(fn,a);return y,vjp(b)[0]
                for mode,call,args in [('forward',fn,(x,)),('forward_backward',both,(x,g))]:
                    lower=jax.jit(call).lower(*args);compiled=lower.compile()
                    key=f'{name}_{mode}';calls[key]=(compiled,args)
                    if name=='algebraic' and length==128:
                        hlo[f'{str(jnp.dtype(dtype))}_{mode}']=lower.as_text()
                    for _ in range(10):jax.block_until_ready(compiled(*args))
            latencies={key:[] for key in calls};rng=np.random.default_rng(46)
            for rep in range(100):
                for key in rng.permutation(list(calls)):
                    fn,args=calls[key];mh.sync_global_devices(f'bench-{dtype}-{length}-{rep}-{key}')
                    start=time.perf_counter_ns();jax.block_until_ready(fn(*args));elapsed=(time.perf_counter_ns()-start)*1e-9
                    latencies[key].append(float(np.asarray(mh.process_allgather(np.array(elapsed))).max()))
            gates={mode:{'ratio':float(np.median(latencies[f'softmax_{mode}'])/np.median(latencies[f'algebraic_{mode}'])),'minimum':.90}
                   for mode in ('forward','forward_backward')}
            for v in gates.values():v['passed']=v['ratio']>=v['minimum']
            row={'dtype':str(jnp.dtype(dtype)),'length':length,'global_shape':list(x.shape),'warmups':10,'repetitions':100,
                 'latency_seconds':{k:summary(v) for k,v in latencies.items()},'gates':gates,'passed':all(v['passed'] for v in gates.values())}
            rows.append(row);raw[f'{row["dtype"]}_L{length}']=latencies
            if jax.process_index()==0:print('benchmark',row['dtype'],length,{k:round(v['ratio'],3) for k,v in gates.items()},flush=True)
    return {'rows':rows,'passed':all(r['passed'] for r in rows)},raw,hlo


def tpu_quantization(place):
    rng = np.random.default_rng(42)
    K = 128
    s = rng.normal(size=K).astype(np.float32)
    s[0] += 6.0  # Logit outlier typical of trained attention
    noise_vec = rng.normal(0, 0.05, size=K).astype(np.float32)

    host = np.tile(s, (32, 1))
    noise = np.tile(noise_vec, (32, 1))

    x = place(host)
    n = place(noise)

    def calc(a, b):
        return algebraic_softmax(a), algebraic_softmax(a + b), jax.nn.softmax(a), jax.nn.softmax(a + b)

    p, pn, q, qn = jax.block_until_ready(jax.jit(calc)(x, n))

    local = []
    for ps, pns, qs, qns in zip(p.addressable_shards, pn.addressable_shards, q.addressable_shards, qn.addressable_shards):
        pa, pan, qa, qan = [np.asarray(t.data, dtype=float) for t in (ps, pns, qs, qns)]
        da = np.linalg.norm(pan - pa, axis=-1)
        db = np.linalg.norm(qan - qa, axis=-1)
        local.extend(np.stack([da, db], axis=-1))

    arr = np.asarray(mh.process_allgather(np.array(local))).reshape(-1, 2)
    da_mean = float(arr[:, 0].mean())
    db_mean = float(arr[:, 1].mean())
    gain = float(db_mean / da_mean) if da_mean > 0 else float('inf')
    return {
        'err_softmax': db_mean,
        'err_algebraic': da_mean,
        'noise_suppression_ratio': gain,
        'passed': bool(gain >= 100.0)
    }


def monte_carlo(place):
    rng = np.random.default_rng(342 + jax.process_index()); rows = []; raw = {}
    def calculate(x, noise):
        p = algebraic_softmax(x); q = jax.nn.softmax(x)
        return p, q, algebraic_softmax(x + noise), jax.nn.softmax(x + noise)
    fn = jax.jit(calculate)
    for length in LENGTHS:
        local = []
        for _ in range(28):
            x = place(rng.normal(size=(128, length)).astype(np.float32)); noise = place(rng.normal(0, .05, size=(128, length)).astype(np.float32))
            p, q, pn, qn = jax.block_until_ready(fn(x, noise))
            for ps, qs, ans, bns in zip(p.addressable_shards, q.addressable_shards, pn.addressable_shards, qn.addressable_shards):
                a, b, an, bn = [np.asarray(t.data, dtype=float) for t in (ps, qs, ans, bns)]
                ent = -np.sum(a * np.log(np.maximum(a, np.finfo(float).tiny)), axis=-1) / np.log(length)
                local.extend(np.stack([ent, w1(a, b), np.linalg.norm(an - a, axis=-1), np.linalg.norm(bn - b, axis=-1), a.sum(-1)], axis=-1))
        a = np.asarray(mh.process_allgather(np.array(local))).reshape(-1, 5); raw[f'L{length}'] = a
        stats = {k: summary(a[:, i]) for i, k in enumerate(('entropy', 'w1', 'noise_alg', 'noise_softmax', 'mass'))}
        gain = stats['noise_softmax']['mean'] / stats['noise_alg']['mean']
        entropy_ok = bool(stats['entropy']['ci95'][0] >= .10 and stats['entropy']['ci95'][1] <= .95)
        w1_ok = bool(stats['w1']['ci95'][1] <= .05)
        simplex_ok = bool(a[:, 4].max() <= 1 + 16 * np.finfo(np.float32).eps)
        gates = {
            'entropy_all_trials': bool(np.all((a[:, 0] >= .10) & (a[:, 0] <= .95))),
            'entropy_mean_ci': entropy_ok,
            'w1_mean_ci': w1_ok,
            'simplex_with_roundoff': simplex_ok,
        }
        rows.append({'length': length, 'trials': len(a), 'statistics': stats,
                     'unscaled_noise_ratio': gain, 'gates': gates,
                     'passed': entropy_ok and w1_ok and simplex_ok})
        if jax.process_index() == 0:
            print('TPU study', length, 'unscaled noise ratio', round(gain, 4), 'W1', round(stats['w1']['mean'], 4), flush=True)
    quant = tpu_quantization(place)
    if jax.process_index() == 0:
        print('TPU quantization robustness ratio', round(quant['noise_suppression_ratio'], 2), flush=True)
    return {'seed_per_process': '342 + JAX process index', 'trials': sum(r['trials'] for r in rows), 'rows': rows,
            'quantization_robustness': quant,
            'raw_columns': ['entropy', 'w1', 'noise_alg', 'noise_softmax', 'mass'],
            'passed': all(r['passed'] for r in rows) and quant['passed']}, raw



def jacobians(place):
    rng=np.random.default_rng(442+jax.process_index());rows=[]
    fn=jax.jit(jax.vmap(jax.jacrev(lambda z:_normalized_forward(z,.5)[0])))
    for length in (2,8,16,64,128):
        maxima=[];errors=[]
        for _ in range(16):
            host=rng.normal(size=(32,length)).astype(np.float32);host/=np.sqrt(np.mean(host*host,axis=-1,keepdims=True)+1e-5)
            x=place(host);jac=jax.block_until_ready(fn(x))
            for xs,js in zip(x.addressable_shards,jac.addressable_shards):
                a=np.asarray(xs.data,dtype=float);j=np.asarray(js.data,dtype=float)
                from tests.reference_attention import normalized_attention
                p=normalized_attention(a)
                expected=(np.eye(length)[None,:,:]-p[:,None,:])*p[:,:,None]*8/np.sqrt(1+a[:,None,:]**2)
                maxima.extend(np.max(np.abs(j),axis=(1,2)));errors.extend(np.max(np.abs(j-expected),axis=(1,2)))
        a=np.asarray(mh.process_allgather(np.stack([maxima,errors],axis=-1))).reshape(-1,2)
        rows.append({'length':length,'trials':len(a),'maximum_entry':summary(a[:,0]),'oracle_error':float(a[:,1].max()),
                     'passed':bool(a[:,0].max()<=2+2e-5 and a[:,1].max()<=2e-5)})
    return {'trials':sum(r['trials'] for r in rows),'rows':rows,'passed':all(r['passed'] for r in rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'results/attention/tpu');args=parser.parse_args()
    jax.distributed.initialize(initialization_timeout=120);devices=jax.devices()
    assert len(devices)==16 and jax.process_count()==4 and all('TPU v4' in d.device_kind for d in devices)
    mesh=Mesh(mesh_utils.create_device_mesh((2,2,4),devices,allow_split_physical_axes=True),('data','fsdp','model'))
    sharding=NamedSharding(mesh,P(('data','fsdp','model')))
    place=lambda a:jax.make_array_from_process_local_data(sharding,a)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    r={'study':2,'environment':environment(),'command':sys.argv,'device_count':len(devices),'process_count':jax.process_count(),
       'mesh_shape':dict(mesh.shape),'devices':[{'id':d.id,'kind':d.device_kind,'process':d.process_index,'coords':list(d.coords)} for d in devices]}
    r['parity']=parity(place)
    if jax.process_index()==0:print('parity',r['parity']['passed'],flush=True)
    r['benchmarks'],latencies,hlo=benchmarks(place)
    r['monte_carlo'],raw=monte_carlo(place);r['jacobian']=jacobians(place)
    r['passed']=all(r[k]['passed'] for k in ('parity','benchmarks','monte_carlo','jacobian'))
    if source_hashes()!=r['environment']['source_sha256']:r['passed']=False;r['source_changed']=True
    if jax.process_index()==0:
        write_json(out/'metrics.json',r);write_json(out/'latencies.json',latencies)
        np.savez_compressed(out/'monte-carlo.npz',**raw)
        for name,content in hlo.items():(out/f'{name}.mlir').write_text(content)
        print('TPU PASS' if r['passed'] else 'TPU FAIL',flush=True)
    mh.sync_global_devices('attention-complete');jax.distributed.shutdown()
    return 0 if r['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
