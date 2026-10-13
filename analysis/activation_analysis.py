#!/usr/bin/env python3
"""The algebraic activation against SiLU (numerical analysis).

For ALU_beta(x) = x/2 (1 + beta x / sqrt(1 + beta^2 x^2)):
* Gaussian-weighted L2 distance to SiLU and GELU over beta (best match).
* Mean-field statistics under z ~ N(0, q): E[phi], E[phi^2], E[phi'^2]
  (gain of the forward signal and of back-propagated gradients), and the
  GLU output second moment E[(g phi(u))^2] for independent g, u ~ N(0, q).
* Tails: ALU's negative tail decays like -1/(4 beta^2 |x|) (algebraic),
  SiLU's like -|x| e^{-|x|}.

Writes results/analysis/activation_analysis.json.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis"

x, w = np.polynomial.hermite_e.hermegauss(161)  # integrates against exp(-x^2/2)
w = w / math.sqrt(2 * math.pi)


def alu(z, b):
    return 0.5 * z * (1 + b * z / np.sqrt(1 + (b * z) ** 2))


def alu_d(z, b):
    u = b * z / np.sqrt(1 + (b * z) ** 2)
    r2 = 1 / (1 + (b * z) ** 2)
    return 0.5 * (1 + u) + 0.5 * u * r2


def silu(z):
    return z * special.expit(z)


def silu_d(z):
    s = special.expit(z)
    return s * (1 + z * (1 - s))


def gelu(z):
    return 0.5 * z * (1 + special.erf(z / math.sqrt(2)))


def gauss(f, q=1.0):
    return float(np.sum(w * f(math.sqrt(q) * x)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    betas = np.round(np.arange(0.3, 1.31, 0.05), 3)
    dist = {}
    for name, ref in (("silu", silu), ("gelu", gelu)):
        rows = []
        for b in betas:
            rows.append(math.sqrt(gauss(lambda z: (alu(z, b) - ref(z)) ** 2, 4.0)))
        k = int(np.argmin(rows))
        dist[name] = {"best_beta": float(betas[k]), "l2_at_best": rows[k],
                      "l2_at_beta1": rows[int(np.where(betas == 1.0)[0][0])],
                      "curve": dict(zip(map(float, betas), rows))}
    mean_field = {}
    for q in (0.25, 1.0, 4.0):
        row = {}
        for name, f, fd in (("silu", silu, silu_d), ("alu_b1", lambda z: alu(z, 1.0), lambda z: alu_d(z, 1.0)),
                            ("alu_b0.5", lambda z: alu(z, 0.5), lambda z: alu_d(z, 0.5)),
                            ("alu_b0.8", lambda z: alu(z, 0.8), lambda z: alu_d(z, 0.8))):
            row[name] = {"E_phi": gauss(f, q), "E_phi2": gauss(lambda z: f(z) ** 2, q),
                         "E_dphi2": gauss(lambda z: fd(z) ** 2, q),
                         "glu_out_second_moment": q * gauss(lambda z: f(z) ** 2, q)}
        mean_field[str(q)] = row
    tails = {}
    for z in (-2.0, -4.0, -8.0, -16.0):
        tails[str(z)] = {"silu": float(silu(z)), "alu_b1": float(alu(z, 1.0)), "alu_b0.5": float(alu(z, 0.5))}
    report = {"distance_to_reference": dist, "mean_field": mean_field, "negative_tail_values": tails,
              "slope_of_gate_at_zero": {"silu": 0.25, "gelu": 1 / math.sqrt(2 * math.pi), "alu_beta": "beta/2"}}
    (OUT / "activation_analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in dist.items()}, indent=1))
    print(json.dumps(mean_field["1.0"], indent=1))
    print(json.dumps(tails, indent=1))


if __name__ == "__main__":
    main()
