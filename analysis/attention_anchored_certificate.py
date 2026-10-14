#!/usr/bin/env python3
"""Uniform real bounds and row-error studies for the anchored root candidate.

Exact rational Bernstein bounds cover the stored float32 root coefficients.
The TPU approximate reciprocal and floating-point evaluation are checked in
the probe, not certified here. No exponential/logarithm is used to construct
or evaluate this approximation. The model does not adopt the candidate.
"""

import argparse
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
from analysis.attention_polynomial import bernstein, certify, product


def exact_certificate(coefficients, bound, intervals):
    root_poly = [Q(float(np.float32(c))) for c in coefficients]
    squared = product(root_poly, root_poly)
    squared[0] -= 1
    squared[1] -= 1
    domain = Q(bound) ** 2
    root = certify(root_poly, intervals, domain)
    residual = certify(squared, intervals, domain)
    pmin = Q(root["lower"])
    if pmin <= 0:
        raise ValueError("root polynomial must be positive over the domain")
    abs_error = Q(residual["max_abs"]) / (pmin + 1)
    # sqrt(1+B²) <= 1+B, so rho(-B) >= 1/(1+2B).
    rho_lower = 1 / (1 + 2 * Q(bound))
    relative = abs_error / rho_lower
    if relative >= 1:
        raise ValueError("error bound does not prove positive approximate rho")
    weight_lower, weight_upper = (1-relative)**16, (1+relative)**16
    # The sharp likelihood-ratio bound is (sqrt(u)-sqrt(l))/(sqrt(u)+sqrt(l)).
    # All roots simplify exactly because the rho powers are even.
    a, b = (1-relative)**8, (1+relative)**8
    sharp_tv = (b-a)/(b+a)
    naive_weight_error = max(weight_upper - 1, 1-weight_lower)
    naive_tv = (naive_weight_error / (1-naive_weight_error) if naive_weight_error < 1 else None)
    return {"method": "exact rational arithmetic; subdivided Bernstein convex hull",
            "coefficients_as_exact_float32_fractions": [str(c) for c in root_poly],
            "root_polynomial": root, "root_squared_residual": residual,
            "real_bounds": {"sqrt_absolute_error": float(abs_error),
                "rho_relative_error": float(relative),
                "weight_ratio_lower": float(weight_lower), "weight_ratio_upper": float(weight_upper),
                "TV_sharp_ratio_bound": float(sharp_tv),
                "TV_previous_relative_bound": float(naive_tv) if naive_tv is not None else None},
            "coverage": "exact real polynomial evaluation and normalization; excludes TPU reciprocal/rounding"}


def row_errors(coefficients, x, sink_x=0.):
    root = np.sqrt(1+x*x)
    pos = np.abs(x)+root
    rho = np.where(x >= 0, pos, 1/pos)
    exact_weight = rho**16
    p_root = np.polynomial.polynomial.polyval(x*x, coefficients)
    approx_weight = (x+p_root)**16
    sr = math.sqrt(1+sink_x*sink_x)
    sink_exact = (sink_x+sr)**16
    sink_root = np.polynomial.polynomial.polyval(sink_x*sink_x, coefficients)
    sink_approx = (sink_x+sink_root)**16
    exact_weight = np.concatenate((exact_weight, np.full((len(x), 1), sink_exact)), axis=1)
    approx_weight = np.concatenate((approx_weight, np.full((len(x), 1), sink_approx)), axis=1)
    p = exact_weight / exact_weight.sum(axis=1, keepdims=True)
    ph = approx_weight / approx_weight.sum(axis=1, keepdims=True)
    tv = .5*np.abs(ph-p).sum(axis=1)
    slope = 16 / np.concatenate((root, np.full((len(x), 1), sr)), axis=1)
    approx_slope = 16 / np.concatenate((p_root, np.full((len(x), 1), sink_root)), axis=1)
    # Each arbitrary row cotangent includes the zero-value sink's cotangent.
    rng = np.random.default_rng(359)
    g = rng.normal(size=p.shape)
    g[:, -1] = 0.
    exact_vjp = slope * p * (g-(p*g).sum(axis=1, keepdims=True))
    approx_vjp = approx_slope * ph * (g-(ph*g).sum(axis=1, keepdims=True))
    vjp_l2 = np.linalg.norm(approx_vjp-exact_vjp, axis=1) / np.maximum(np.linalg.norm(exact_vjp, axis=1), 1e-100)
    return {"rows": len(x), "keys_plus_sink": p.shape[1],
            "TV_mean": float(tv.mean()), "TV_max": float(tv.max()),
            "TV_quantiles": np.quantile(tv, [.5, .95, .99]).tolist(),
            "original_VJP_relative_l2_mean": float(vjp_l2.mean()),
            "original_VJP_relative_l2_max": float(vjp_l2.max())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path, default=ROOT / "results/kernels/attention_polynomial_reciprocal_probe.json")
    p.add_argument("--candidate", default="poly3_anchored_reciprocal")
    p.add_argument("--intervals", type=int, default=256)
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--keys", type=int, default=2048)
    p.add_argument("--out", type=Path, default=ROOT / "results/analysis/attention_anchored_certificate.json")
    args = p.parse_args()
    probe = json.loads(args.probe.read_text())
    spec = probe["approximation"][args.candidate]
    coefficients = np.asarray(spec["sqrt_coefficients_f32"], np.float64)
    bound = spec["domain_abs_x"]
    report = {"schema": 1, "candidate": args.candidate, "bound": bound,
              "model_adopted": False, "parity_achieved": False,
              "probe": str(args.probe.relative_to(ROOT)),
              "probe_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest(),
              "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "certificate": exact_certificate(coefficients, bound, args.intervals),
              "trials": [], "adversarial": [],
              "statistical_scope": "synthetic independent seeds, no learned-score or language-model quality claim"}
    for seed in [359, 367, 373, 379, 383, 389, 397, 401]:
        rng = np.random.default_rng(seed)
        for std in [.125, .25, .5, 1.]:
            x = np.clip(rng.normal(0., std, (args.rows, args.keys)), -bound, bound)
            report["trials"].append(dict(seed=seed, kernel_score_std=std, **row_errors(coefficients, x)))
    for mean in [-bound, -.75, 0., .75, bound]:
        x = np.clip(np.linspace(mean-.1, mean+.1, args.keys), -bound, bound)[None, :]
        for sink in [0., -bound, bound]:
            report["adversarial"].append(dict(score_center=mean, sink_kernel_score=sink,
                **row_errors(coefficients, x, sink)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report["certificate"]["real_bounds"], indent=2))
    print("max synthetic row TV", max(r["TV_max"] for r in report["trials"]))
    print("max adversarial row TV", max(r["TV_max"] for r in report["adversarial"]))


if __name__ == "__main__":
    main()
