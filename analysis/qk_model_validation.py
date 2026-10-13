#!/usr/bin/env python3
"""Independent complete-model checks of the isolated QK RMS prototype."""

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from analysis.qk_rms_kernel import make_norm
from algebraic import algebraic as A, config, model
from algebraic.model import init_params, forward_hidden, head_loss, position_tables


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forward-native",action="store_true")
    p.add_argument("--faithful-backward",action="store_true")
    p.add_argument("--gain-partials-only",action="store_true")
    p.add_argument("--production",action="store_true")
    p.add_argument("--out",type=Path,default=ROOT/"results/analysis/qk_model_validation.json")
    args=p.parse_args()
    original=A.rms_norm
    original_qk=model._qk_norm
    if args.production:
        from algebraic.kernels.qk_norm import qk_rms_norm
        candidate=qk_rms_norm
    else:
        candidate=make_norm(1024,args.forward_native,args.faithful_backward,args.gain_partials_only)
    report={"schema":1,"utc":datetime.now(timezone.utc).isoformat(),
            "jax":jax.__version__,"devices":[str(d) for d in jax.devices()],
            "prototype":True,"model_adopted":False,
            "forward_native":args.forward_native,"faithful_backward":args.faithful_backward,
            "gain_partials_only":args.gain_partials_only,
            "public_kernel":args.production,
            "scope":"numerical equivalence of initialized complete models; not trained quality",
            "sources":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                       ("analysis/qk_model_validation.py","analysis/qk_rms_kernel.py","src/algebraic/kernels/qk_norm.py","src/algebraic/model.py",
                        "src/algebraic/algebraic.py","src/algebraic/kernels/flash.py")},"cases":[]}
    path=args.out
    path.parent.mkdir(parents=True,exist_ok=True)
    def save():
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    cases=[("hybrid",163,512,2,4096,1.,"125m"),("hybrid",167,512,2,4096,4.,"125m"),
           ("algebraic",163,512,2,4096,1.,"125m"),("hybrid",173,2048,12,50257,1.,"125m"),
           ("hybrid",179,2048,10,50257,1.,"60m")]
    for arch,seed,T,L,V,gain,size in cases:
        cfg=config.build(arch,size,seq_len=T,n_layers=L,vocab_size=V,attn_impl="flash")
        keys=jax.random.split(jax.random.PRNGKey(seed),3)
        params=init_params(cfg,keys[0])
        params["layers"]["q_gain"]=gain*params["layers"]["q_gain"]
        params["layers"]["k_gain"]=gain*params["layers"]["k_gain"]
        x=jax.random.randint(keys[1],(1,T),0,V)
        y=jax.random.randint(keys[2],(1,T),0,V)
        tables=position_tables(cfg)
        def loss(params,x,y):
            return head_loss(params,forward_hidden(params,x,cfg,tables),y,cfg)
        model._qk_norm=lambda x,gain,cfg:original(x,gain,cfg.eps)
        jax.clear_caches()
        baseline=jax.jit(jax.value_and_grad(loss,has_aux=True)).lower(params,x,y).compile()
        bo,bg=baseline(params,x,y)
        # Retain the compiled baseline before substituting the prototype.
        def norm(x,gain,cfg):
            return candidate(x,gain) if x.ndim==4 and x.shape[-1]==64 and gain is not None and cfg.eps==1e-5 else original(x,gain,cfg.eps)
        model._qk_norm=norm
        jax.clear_caches()
        try:
            alternative=jax.jit(jax.value_and_grad(loss,has_aux=True)).lower(params,x,y).compile()
            ao,ag=alternative(params,x,y)
            jax.block_until_ready(ag)
            field_errors=[]
            for (field,a),(_,b) in zip(jax.tree_util.tree_flatten_with_path(ag)[0],
                                      jax.tree_util.tree_flatten_with_path(bg)[0]):
                av,bv=np.asarray(a,np.float32),np.asarray(b,np.float32)
                abs_error=float(np.max(np.abs(av-bv)))
                rel_error=float(np.linalg.norm(av-bv)/max(np.linalg.norm(bv),1e-6))
                field_errors.append({"field":jax.tree_util.keystr(field),"max_abs_error":abs_error,
                                     "relative_l2_error":rel_error,"finite":bool(np.isfinite(av).all())})
            telem_error=float(jnp.max(jnp.abs(ao[1]-bo[1])))
            case={"architecture":arch,"size":size,"seed":seed,"seq_len":T,"layers":L,"vocab":V,
                  "q_and_k_gain":gain,"objective_abs_error":float(jnp.abs(ao[0]-bo[0])),
                  "target_probability_max_abs_error":telem_error,"gradient_fields":field_errors,
                  "passed":all(z["finite"] and z["relative_l2_error"]<.01 for z in field_errors)
                           and bool(jnp.allclose(ao[1],bo[1],atol=1e-6,rtol=.01))}
            report["cases"].append(case)
            print(arch,seed,T,case["passed"],max(z["relative_l2_error"] for z in field_errors),flush=True)
            save()
        finally:
            model._qk_norm=original_qk
        del baseline,alternative,bo,bg,ao,ag,params
    report["passed"]=all(c["passed"] for c in report["cases"])
    save()


if __name__=="__main__":
    main()
