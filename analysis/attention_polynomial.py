#!/usr/bin/env python3
"""Exact rational interval certificates for the rejected degree-four experiment.

Certifies the actual float32 coefficients over the entire interval [0,1],
using Bernstein convex hull bounds on each rational subinterval. No grid of
sampled function values is used as proof. These bounds concern real arithmetic;
TPU rounding and the timing experiment are separate evidence.
"""

import argparse
from fractions import Fraction as Q
import json
from math import comb
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from algebraic.kernels.radical import SQRT_COEFFICIENTS, RSQRT_COEFFICIENTS


def product(a, b):
    c = [Q(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i + j] += x * y
    return c


def bernstein(poly, a, b):
    """Convert p(a+(b-a)u) to Bernstein basis, exactly, with u in [0,1]."""
    degree = len(poly) - 1
    power = [Q(0)] * (degree + 1)
    for i, ci in enumerate(poly):
        for j in range(i + 1):
            power[j] += ci * comb(i, j) * a ** (i - j) * (b - a) ** j
    return [sum((power[j] * Q(comb(k, j), comb(degree, j)) for j in range(k + 1)), Q(0))
            for k in range(degree + 1)]


def certify(poly, intervals):
    rows = []
    for i in range(intervals):
        a, b = Q(i, intervals), Q(i + 1, intervals)
        c = bernstein(poly, a, b)
        rows.append({"a": str(a), "b": str(b), "lower": str(min(c)), "upper": str(max(c))})
    lo = min(Q(r["lower"]) for r in rows)
    hi = max(Q(r["upper"]) for r in rows)
    return {"lower": str(lo), "upper": str(hi), "max_abs": str(max(abs(lo), abs(hi))), "intervals": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--intervals", type=int, default=256)
    p.add_argument("--out", type=Path, default=ROOT / "results/analysis/attention_polynomial.json")
    args = p.parse_args()
    ps = [Q(float(np.float32(x))) for x in SQRT_COEFFICIENTS]
    pr = [Q(float(np.float32(x))) for x in RSQRT_COEFFICIENTS]
    sqrt_residual = product(ps, ps)
    sqrt_residual[0] -= 1
    sqrt_residual[1] -= 1
    rsqrt_residual = product(product(pr, pr), [Q(1), Q(1)])
    rsqrt_residual[0] -= 1
    certificates = {"P": certify(ps, args.intervals), "Q": certify(pr, args.intervals),
                    "P_squared_minus_1_minus_t": certify(sqrt_residual, args.intervals),
                    "Q_squared_times_1_plus_t_minus_1": certify(rsqrt_residual, args.intervals)}
    pmin, qmin = Q(certificates["P"]["lower"]), Q(certificates["Q"]["lower"])
    assert pmin > 0 and qmin > 0
    ep = Q(certificates["P_squared_minus_1_minus_t"]["max_abs"])
    er = Q(certificates["Q_squared_times_1_plus_t_minus_1"]["max_abs"])
    sqrt_abs_error = ep / (pmin + 1)  # |P-sqrt(1+t)| = |P²-(1+t)|/(P+sqrt(1+t))
    rho_rel_error = sqrt_abs_error / Q(414, 1000)  # sqrt(2)-1 > 0.414
    slope_rel_error = er / (qmin + 1)  # |Q*sqrt(1+t)-1| = |Q²*(1+t)-1|/(Q*sqrt(1+t)+1)
    weight_error = max((1 + rho_rel_error) ** 16 - 1, 1 - (1 - rho_rel_error) ** 16)
    # Normalization gives TV <= eps/(1-eps) by the entrywise ratio theorem.
    tv_error = weight_error / (1 - weight_error)
    report = {"schema": 1, "domain": {"x": [-1, 1], "t": [0, 1]},
              "method": "exact Fraction arithmetic; subdivided Bernstein convex hull",
              "float32_sqrt_coefficients": [str(x) for x in ps],
              "float32_rsqrt_coefficients": [str(x) for x in pr],
              "certificates": certificates,
              "real_arithmetic_bounds": {"sqrt_absolute_error": float(sqrt_abs_error),
                "rho_relative_error": float(rho_rel_error), "log_slope_relative_error": float(slope_rel_error),
                "weight_n16_relative_error": float(weight_error), "normalized_TV_error": float(tv_error)},
              "status": "accurate but slower on TPU v4; opt-in research variant",
              "coverage": "real polynomial evaluation; no claim of Lean-checked coefficients or IEEE rounding"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["real_arithmetic_bounds"], indent=2))


if __name__ == "__main__":
    main()
