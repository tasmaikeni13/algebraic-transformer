#!/usr/bin/env python3
"""Randomize real optimizer-step profile jobs on one pinned TPU chip.

Each job has fresh parameters/AdamW moments and a seed-matched data stream.
Its first step compiles; the following steps include real updates and I/O.
Use chiprun once on this driver; the child processes inherit chip pinning.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from analysis.attention_performance import ratio_ci


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size",default="125m",choices=["60m","125m"])
    p.add_argument("--seeds",nargs="+",type=int,default=[191,193,197])
    p.add_argument("--profile-steps",type=int,default=30)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    choices={"standard":("standard",False),"hybrid_native":("hybrid",True),
             "hybrid_tiled_gain":("hybrid",False),"algebraic":("algebraic",False)}
    report={"schema":1,"utc":datetime.now(timezone.utc).isoformat(),"size":a.size,
            "batch":8,"seq":2048,"seeds":a.seeds,"hostname":platform.node(),
            "profile_steps":a.profile_steps,"order_seed":191,
            "protocol":"fresh process per seed/case; first step excluded; real AdamW updates and input I/O; randomized case order; paired seeds",
            "scope":"single-chip training speed; no held-out quality or slice throughput claim",
            "sources":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                       ("analysis/model_step_performance.py","analysis/model_profile.py","src/algebraic/train.py",
                        "src/algebraic/model.py","src/algebraic/kernels/qk_norm.py","src/algebraic/kernels/flash.py")},
            "jobs":[],"results":{},"comparisons":{}}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    root=ROOT/"runs"/(a.out.stem+"_jobs")
    root.mkdir(parents=True,exist_ok=True)
    def save():
        a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    rng=np.random.default_rng(191)
    for seed in a.seeds:
        for ix in rng.permutation(len(choices)):
            name=list(choices)[int(ix)]
            preset,native=choices[name]
            out=root/f"{name}_s{seed}"
            cmd=[sys.executable,str(ROOT/"analysis/model_profile.py"),"--preset",preset,
                 "--size",a.size,"--seq-len","2048","--batch","8","--micro","8",
                 "--steps","1000","--profile-steps",str(a.profile_steps),"--log-every","1000",
                 "--eval-tokens","16384","--eval-batch","8","--seed",str(seed),
                 "--set","attn_impl=flash","--out",str(out)]
            if native:
                cmd.append("--native-qk")
            job={"case":name,"seed":seed,"command":cmd,"output_dir":str(out)}
            log=root/f"{name}_s{seed}.log"
            with log.open("w") as f:
                run=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            job["exit_code"]=run.returncode
            if run.returncode:
                job["error_tail"]=log.read_text()[-4000:]
            else:
                job["profile"]=json.loads((out/"profile.json").read_text())
                job["meta"]=json.loads((out/"meta.json").read_text())
                job["implementation"]=json.loads((out/"prototype.json").read_text())
                print(name,seed,job["profile"]["step_time"]*1000,flush=True)
            report["jobs"].append(job)
            save()
    for name in choices:
        jobs=[x for x in report["jobs"] if x["case"]==name and not x["exit_code"]]
        report["results"][name]={"seeds":[x["seed"] for x in jobs],
            "samples_ms":[x["profile"]["step_time"]*1000 for x in jobs]}
        if jobs:
            report["results"][name]["median_ms"]=float(np.median(report["results"][name]["samples_ms"]))
    std=report["results"]["standard"]
    for name,item in report["results"].items():
        if item["seeds"]==std["seeds"] and item["seeds"]:
            report["comparisons"][name]=ratio_ci(item["samples_ms"],std["samples_ms"])
    old,new=report["results"]["hybrid_native"],report["results"]["hybrid_tiled_gain"]
    if old["seeds"]==new["seeds"] and old["seeds"]:
        report["hybrid_change"]=ratio_ci(new["samples_ms"],old["samples_ms"])
    save()


if __name__=="__main__":
    main()
