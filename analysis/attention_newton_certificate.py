#!/usr/bin/env python3
"""Exact real-arithmetic bounds for rounded polynomial inverse-root seeds.

Uses exact Bernstein bounds for E=(1+t)*seed(t)^2-1 and the Lean-checked
Newton residual identity. This does not certify machine evaluation.
"""

from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.attention_interpolation_certificate import bernstein_min, sqrt_interval


def product(a,b):
    out=[F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            out[i+j]+=x*y
    return out


def upper_float(x):
    return math.nextafter(float(x),math.inf)


def main():
    source=ROOT/"results/kernels/attention_radical_polynomial_seed_normalized_probe.json"
    raw=json.loads(source.read_text())
    report={"schema":1,"domain_abs_x":1.5,"subintervals":1024,
            "probe_record_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "certificate_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "arithmetic":"exact rational Bernstein bounds and 90-bit radical enclosures",
            "machine_evaluation_certified":False,"coefficient_bounds_imported_into_lean":False,
            "newton_residual_identity_checked_in_lean":True,"modes":{}}
    for mode,spec in raw["approximation"].items():
        if not mode.startswith("poly"):
            continue
        coeff=[F.from_float(x) for x in spec["rsqrt_seed_coefficients_f32"]]
        residual=product([F(1),F(1)],product(coeff,coeff))
        residual[0]-=1
        magnitude=F(0)
        min_seed=None
        for i in range(1024):
            a,b=F(9*i,4096),F(9*(i+1),4096)
            lo=bernstein_min(residual,a,b)
            hi=-bernstein_min([-x for x in residual],a,b)
            magnitude=max(magnitude,abs(lo),abs(hi))
            seed_lower=bernstein_min(coeff,a,b)
            min_seed=seed_lower if min_seed is None else min(min_seed,seed_lower)
        if min_seed <= 0 or magnitude >= 1:
            raise ArithmeticError("positive Newton seed domain certificate failed")
        initial=magnitude
        iterations=2 if "nr2" in mode else 1
        for _ in range(iterations):
            magnitude=magnitude*magnitude*(3+magnitude)/4
        inverse_error=magnitude/(1+sqrt_interval(1-magnitude).lo)
        rho_error=inverse_error*(F(13,4)+F(3,2)*sqrt_interval(F(13,4)).hi)
        weight_error=(1+rho_error)**16-1
        report["modes"][mode]={"coefficients_f32":spec["rsqrt_seed_coefficients_f32"],
            "iterations":iterations,"seed_positive":True,
            "initial_residual_abs_bound":upper_float(initial),
            "refined_residual_abs_bound":upper_float(magnitude),
            "inverse_root_relative_bound":upper_float(inverse_error),
            "rho_relative_bound":upper_float(rho_error),
            "weight_relative_bound":upper_float(weight_error)}
    (ROOT/"results/analysis/attention_newton_certificate.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n")
    print({k:v["weight_relative_bound"] for k,v in report["modes"].items()})


if __name__=="__main__":
    main()
