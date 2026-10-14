#!/usr/bin/env python3
"""Recompute paired ratios from the whole-head confirmation's raw samples."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/"src")]

import numpy as np
from analysis.attention_performance import ratio_ci


def summarize(path):
    record = json.loads(path.read_text())
    results = record["results"]
    algebraic = "asinh_guarded_512x512"
    previous = "asinh_guarded_resident_512x512"
    summary = dict(record=str(path.relative_to(ROOT)),record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   shape=record["shape"],jax=record["jax"],seed=record["seed"],
                   kernel_sha256=record["kernel_sha256"],head_resident_sha256=record["head_resident_sha256"],
                   modes={},failed_baselines={})
    for mode in ("forward","training"):
        successful = {name:np.asarray(row[mode]["samples_ms"],np.float64) for name,row in results.items()
                      if "samples_ms" in row.get(mode,{})}
        for name,samples in successful.items():
            if len(samples) != record["rounds"] or not np.isfinite(samples).all() or np.any(samples<=0):
                raise ValueError(f"invalid raw samples: {path}, {name}, {mode}")
            if not np.isclose(np.median(samples),results[name][mode]["median_ms"],rtol=1e-12,atol=1e-12):
                raise ValueError(f"median does not match raw samples: {path}, {name}, {mode}")
        standard = [name for name in successful if name.startswith(("softmax","splash","tokamax","jax_flash"))]
        best = min(standard,key=lambda name:np.median(successful[name]))
        ratio = ratio_ci(successful[algebraic],successful[best])
        improvement = ratio_ci(successful[algebraic],successful[previous])
        summary["modes"][mode] = dict(reference=best,algebraic_ms=float(np.median(successful[algebraic])),
                                    standard_ms=float(np.median(successful[best])),previous_ms=float(np.median(successful[previous])),
                                    algebraic_over_standard=ratio,new_over_previous=improvement,
                                    measured_parity=ratio["ci95"][1]<=1.)
        for name,row in results.items():
            if "error" in row.get(mode,{}):
                summary["failed_baselines"][name+"_"+mode]=row[mode]["error"]
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records",nargs="+",type=Path,default=[ROOT/"results/kernels"/name for name in
                   ["attention_head_precise_confirm_125m.json","attention_head_precise_modern_confirm_125m.json",
                    "attention_head_confirm_350m_shape.json","attention_head_confirm_1b_shape.json"]])
    p.add_argument("--out",type=Path,default=ROOT/"results/analysis/attention_head_statistics.json")
    args=p.parse_args()
    report=dict(utc=datetime.now(timezone.utc).isoformat(),records=[summarize(path) for path in args.records],
                scope="within-process paired measurement rounds; excludes between-device, model, and compilation variability",
                full_model_scope="five-step optimizer smokes have no per-step samples or statistical parity claim",
                parity_achieved=False)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for row in report["records"]:
        print(row["record"],{mode:value["algebraic_over_standard"]["ratio"] for mode,value in row["modes"].items()})


if __name__=="__main__":main()
