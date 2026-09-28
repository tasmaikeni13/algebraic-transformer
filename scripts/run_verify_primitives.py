#!/usr/bin/env python3
"""Check primitive behavior against numerical references and TPU records.

The ``--cpu-only`` option runs the numerical checks without reading TPU
measurements.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")

import numpy as np
import jax

from scripts.audit_primitives import audit
from scripts.primitives_experiments import numerical, monte_carlo, deep_trials
from scripts.primitives_records import environment, source_hashes, write_json

CPU_MEASUREMENT_FILES = ("src/__init__.py", "src/primitives.py",
    "tests/reference_primitives.py", "scripts/primitives_experiments.py", "requirements.txt")


def reusable_cpu_evidence(record, current):
    """Reuse expensive samples only when all numerical dependencies are identical.

    Hardware/metadata-only repairs do not change these measurements. Unit tests,
    formal proofs, purity checks and hardware gates are always checked afresh.
    """
    old = record.get("environment", {})
    if any(old.get(k) != current.get(k) for k in ("jax", "jaxlib", "numpy", "scipy")):
        return False
    if any(old.get("source_sha256", {}).get(p) != current["source_sha256"].get(p)
           for p in CPU_MEASUREMENT_FILES):
        return False
    return (record.get("status") in {"PASS", "CPU_VERIFIED_TPU_PENDING"}
            and all(record.get(k, {}).get("passed") for k in ("numerical", "monte_carlo", "deep"))
            and record["monte_carlo"].get("samples_per_scale") == 1_000_000
            and record["deep"].get("trials_per_depth") == 10_000
            and record["deep"].get("width") == 128)


def hardware_evidence(path, hashes):
    if not path.exists():
        return {"passed":False,"reason":"Required 16-chip TPU v4 measurements are missing."}
    record = json.loads(path.read_text())
    if record.get("environment",{}).get("source_sha256") != hashes:
        return {"passed":False,"reason":"TPU evidence was produced from different source files."}
    if record.get("device_count") != 16 or record.get("process_count") != 4:
        return {"passed":False,"reason":"Measurements do not cover the target four-host, 16-chip slice."}
    if not all("TPU v4" in d.get("kind","") for d in record.get("devices",[])) or len(record.get("devices",[])) != 16:
        return {"passed":False,"reason":"TPU v4 device fingerprint is incomplete."}
    if not record.get("passed"):
        return {"passed":False,"reason":"At least one hardware gate failed.","record":record}
    # Re-evaluate the complete inventory, rather than trusting a top-level flag.
    try:
        parity=record["parity"]["rows"]
        parity_ok=len(parity)==16 and all(r["passed"] and r["forward_scaled_error"]<=r["tolerance"]
            and r["vjp_scaled_error"]<=r["tolerance"] for r in parity)
        mc=record["monte_carlo"]
        mc_ok=mc["samples_per_scale"]>=1_000_000 and len(mc["rows"])==14 and all(r["passed"] for r in mc["rows"])
        bench=record["benchmarks"]["rows"]
        bench_ok={r["dtype"] for r in bench}=={"float32","bfloat16"} and all(
            r["repetitions"]>=100 and r["parameter_bytes"]["avn"]==0 and
            all(r["gates"][f"{arm}_{mode}"]["throughput_ratio"]>=threshold
                for arm,threshold in [("alu",.90),("avn",.95)] for mode in ["forward","forward_backward"])
            for r in bench)
        deep=record["deep"]
        deep_ok=deep["trials_per_depth"]>=10_000 and {r["depth"] for r in deep["rows"]}=={8,16,24,32} and all(
            r["arms"]["alu"]["gradient_ratio"]["n"]>=10_000 and
            r["arms"]["alu"]["gradient_ratio"]["min"]>=.2 and r["arms"]["alu"]["gradient_ratio"]["max"]<=5 and
            r["gradient_variance_relative_delta"]<=.05 and
            (r["depth"]!=32 or (r["arms"]["alu"]["activation_variance_ratio"]["min"]>=.5 and
                                 r["arms"]["alu"]["activation_variance_ratio"]["max"]<=2)) for r in deep["rows"])
        if not (parity_ok and mc_ok and bench_ok and deep_ok):
            return {"passed":False,"reason":"Hardware evidence fails the complete Primitive study gate inventory."}
    except (KeyError,TypeError,ValueError):
        return {"passed":False,"reason":"Hardware evidence has an incomplete gate inventory."}
    return {"passed":True,"path":str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),"record":record}


def plot_results(records, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    for eps,label in [(1e-5,"Default epsilon = 1e-5"),(0.,"Ideal epsilon = 0")]:
        rows = [r for r in records["monte_carlo"]["rows"] if r["eps"]==eps]
        axes[0].errorbar([r["sigma"] for r in rows],[r["variance"]["mean"] for r in rows],
                         yerr=[r["variance"]["ci95"][1]-r["variance"]["mean"] for r in rows],marker="o",label=label)
    axes[0].axhspan(.9999,1.0001,color="green",alpha=.1,label="Unit second-moment reference band")
    axes[0].set(xscale="log",xlabel="Input standard deviation",ylabel="Centered variance after AVN")
    axes[0].legend(fontsize=8)
    for arm in ("alu","gelu","swish"):
        rows=records["deep"]["rows"]
        stats=[r["arms"][arm]["gradient_ratio"] for r in rows]
        axes[1].errorbar([r["depth"] for r in rows],[s["mean"] for s in stats],
                        yerr=[s["ci95"][1]-s["mean"] for s in stats],marker="o",label=arm.upper())
    axes[1].set(xlabel="Residual block depth",ylabel="Input / terminal gradient norm")
    axes[1].legend()
    fig.suptitle("Primitive study CPU study (fp64 moments, fp32 depth); 95% confidence intervals")
    fig.savefig(out/"verification.png",dpi=300)
    fig.savefig(out/"verification.pdf")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only",action="store_true")
    parser.add_argument("--output",type=Path,default=ROOT/"results/primitives")
    parser.add_argument("--tpu-results",type=Path,default=ROOT/"results/primitives/tpu/metrics.json")
    parser.add_argument("--reuse-cpu",type=Path,help="Existing metrics with identical numerical dependencies; rechecks tests/proofs/purity")
    args=parser.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    records={"study":1,"gate_version":2,"environment":environment(),"command":sys.argv,"cpu_devices":[str(d) for d in jax.devices()]}
    lake=shutil.which("lake") or str(Path.home()/".elan/bin/lake")
    proof_scan=[]
    import re
    for p in (ROOT/"formal").rglob("*.lean"):
        if ".lake" not in p.parts and re.search(r"\b(sorry|admit|axiom)\b",p.read_text()):
            proof_scan.append(str(p.relative_to(ROOT)))
    try:
        build=subprocess.run([lake,"build"],cwd=ROOT/"formal",capture_output=True,text=True)
        build_log=build.stdout+build.stderr
        formal_ok=build.returncode==0 and not re.search(r"\bwarning\b",build_log,re.I) and not proof_scan
    except OSError as exc:
        build_log=str(exc); formal_ok=False
    (out/"lean-build.log").write_text(build_log)
    records["formal"]={"passed":formal_ok,"command":[lake,"build"],"proof_scan_violations":proof_scan,"log":"lean-build.log"}
    tests=subprocess.run([sys.executable,"-m","pytest","-q"],cwd=ROOT,capture_output=True,text=True)
    (out/"pytest.log").write_text(tests.stdout+tests.stderr)
    records["unit_tests"]={"passed":tests.returncode==0,"log":"pytest.log"}
    previous = json.loads(args.reuse_cpu.read_text()) if args.reuse_cpu else None
    if previous is not None and not reusable_cpu_evidence(previous, records["environment"]):
        raise ValueError("CPU evidence is incomplete or its numerical dependencies changed.")
    for name,fn in [("purity",audit),("numerical",numerical),("monte_carlo",monte_carlo)]:
        if previous is not None and name != "purity":
            records[name] = previous[name]
            continue
        records[name]=fn()
        print(f"{name}: {'PASS' if records[name]['passed'] else 'FAIL'}",flush=True)
    if previous is None:
        records["deep"],raw=deep_trials(progress=lambda msg: print(msg,flush=True))
        np.savez_compressed(out/"deep-trials.npz",**raw)
    else:
        records["deep"] = previous["deep"]
        records["cpu_measurement_provenance"] = {"path":str(args.reuse_cpu.resolve()),
            "environment":previous["environment"],"command":previous["command"],
            "unchanged_dependencies":list(CPU_MEASUREMENT_FILES)}
        raw_path = args.reuse_cpu.parent/"deep-trials.npz"
        if raw_path.resolve() != (out/"deep-trials.npz").resolve():
            shutil.copy2(raw_path, out/"deep-trials.npz")
    records["hardware"]=hardware_evidence(args.tpu_results,records["environment"]["source_sha256"])
    cpu_ok=all(records[k]["passed"] for k in ("formal","unit_tests","purity","numerical","monte_carlo","deep"))
    records["passed"]=cpu_ok and records["hardware"]["passed"]
    records["status"]="PASS" if records["passed"] else "CPU_VERIFIED_TPU_PENDING" if cpu_ok else "FAIL"
    if source_hashes()!=records["environment"]["source_sha256"]:
        records.update(passed=False,status="FAIL_SOURCE_CHANGED_DURING_RUN")
    write_json(out/"metrics.json",records)
    plot_results(records,out)
    (out/"STATUS.md").write_text(f"# Primitive study: {records['status']}\n\n"
        "Direct evidence: [metrics.json](metrics.json), [Lean build](lean-build.log), "
        "[tests](pytest.log), [raw deep trials](deep-trials.npz), [figure](verification.png).\n\n"
        +("All execution gates passed. See PASS.md for the audited completion record.\n" if records["passed"] else
          "Primitive study is incomplete. CPU evidence does not establish TPU parity. " + records["hardware"].get("reason","")+"\n"))
    if not records["passed"] and (out/"PASS.md").exists():
        (out/"PASS.md").rename(out/"PASS.superseded.md")
    print(records["status"],flush=True)
    return 0 if records["passed"] or (args.cpu_only and cpu_ok) else 1


if __name__=="__main__":
    raise SystemExit(main())
