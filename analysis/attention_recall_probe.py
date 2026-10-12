#!/usr/bin/env python3
"""Paired learned associative recall for the faster affine attention links.

All students have learned QK RMS gains and the same toy softmax classifier.
Stable conceptual logits are an independent training oracle for the links;
this does not measure the Pallas kernels or language-model quality.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from analysis.attention_analysis import recall_trial

LINKS={"standard":("softmax",1,1.,0.,0.),"asinh16_c8":("asinh",16,8.,0.,0.),
       "affine32_c16":("affine",32,16.,-.75,3.),
       "affine16_c2":("affine",16,2.,-.9375,32.),
       "affine8_c1":("affine",8,1.,-.999,64.)}


def weights(spec):
    kind,n,c,lo,hi=spec
    def fn(s):
        if kind=="softmax":
            logits=s
        elif kind=="asinh":
            logits=n*jnp.arcsinh(s/c)
        else:
            logits=n*jnp.log1p(jnp.clip(s/c,lo,hi))
        return jax.nn.softmax(logits,axis=-1)
    return fn


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--links",nargs="+",choices=list(LINKS),default=list(LINKS))
    p.add_argument("--seeds",nargs="+",type=int,default=[211,223,227])
    p.add_argument("--items",nargs="+",type=int,default=[128,512])
    p.add_argument("--steps",type=int,default=1500)
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    report={"schema":1,"utc":datetime.now(timezone.utc).isoformat(),"steps":a.steps,
            "seeds":a.seeds,"items":a.items,"batch":a.batch,"d":64,"lr":.003,
            "qk_normalized":True,"learned_qk_gains":True,"links":{k:LINKS[k] for k in a.links},
            "jax":jax.__version__,"devices":[str(x) for x in jax.devices()],
            "scope":"learned toy associative recall; stable log/exp oracle and shared softmax output classifier; not an algebraic-kernel audit, language-model result, or speed estimate",
            "sources":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                       ("analysis/attention_recall_probe.py","analysis/attention_analysis.py")},
            "trials":[],"summary":[]}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    def save():
        a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for items in a.items:
        for seed in a.seeds:
            for name in a.links:
                with jax.default_matmul_precision("highest"):
                    detail=recall_trial(None,items,seed,steps=a.steps,batch=a.batch,
                        weight_fn=weights(LINKS[name]),qk_normalize=True,return_details=True)
                row={"items":items,"seed":seed,"link":name,**detail}
                report["trials"].append(row)
                print(row,flush=True)
                save()
                jax.clear_caches()
    for items in a.items:
        for name in a.links:
            rows=[x for x in report["trials"] if x["items"]==items and x["link"]==name]
            report["summary"].append({"items":items,"link":name,"seeds":len(rows),
                "nll_mean":float(np.mean([x["nll"] for x in rows])),
                "accuracy_mean":float(np.mean([x["accuracy"] for x in rows])),
                "accuracy_min":float(np.min([x["accuracy"] for x in rows])),
                "accuracy_max":float(np.max([x["accuracy"] for x in rows]))})
    save()


if __name__=="__main__":
    main()
