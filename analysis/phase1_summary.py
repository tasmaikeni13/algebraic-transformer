#!/usr/bin/env python3
"""Paired-by-seed statistics for the Phase 1 headline comparisons.

Runs with the same seed share initialization and data order, so differences
are paired. Writes results/analysis/phase1_summary.json.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ["p1b1", "p1b2", "p1b3", "p1b4", "p1b5"]

# name -> {seed: run directory name}
ARMS = {
    "standard": {0: "std_lr0.002", 1: "std_lr0.002_s1", 2: "std_lr0.002_s2", 3: "std_flash_s3"},
    "v1 (as published)": {0: "v1_lr0.002"},
    "v2 provisional": {0: "v2prov", 1: "v2prov_s1", 2: "v2prov_s2"},
    "v2b": {0: "v2b", 1: "v2b_s1"},
    "v2c": {0: "v2b_qknorm", 1: "v2b_qk", 2: "v2c_s2", 3: "v2c_s3"},
    "hybrid (v2c + softmax attention)": {0: "v2b_qk_attnsoftmax", 1: "hyb_s1", 2: "hyb_s2"},
    "v2d (final algebraic)": {0: "v2d_s0", 1: "v2d_s1", 2: "v2d_s2", 3: "v2d_s3"},
    "standard + QK-norm + algebraic head": {0: "std+qk+head", 1: "std+qk+head_s1", 2: "std+qk+head_s2"},
}


def nll_of(name):
    for g in GROUPS:
        p = ROOT / "runs" / g / name / "result.json"
        if p.exists():
            return json.loads(p.read_text())["final_eval_nll"]
    raise FileNotFoundError(name)


def describe(vals):
    v = np.asarray(vals)
    se = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
    return {"n": len(v), "mean": float(v.mean()), "se": se, "values": v.tolist()}


def paired(a, b):
    seeds = sorted(set(ARMS[a]) & set(ARMS[b]))
    d = np.asarray([nll_of(ARMS[a][s]) - nll_of(ARMS[b][s]) for s in seeds])
    out = {"seeds": seeds, "differences": d.tolist(), "mean": float(d.mean())}
    if len(d) > 1:
        se = d.std(ddof=1) / math.sqrt(len(d))
        t = d.mean() / se
        out.update({"se": float(se), "t": float(t),
                    "p_two_sided": float(2 * (1 - stats.t.cdf(abs(t), len(d) - 1))),
                    "ci95": [float(d.mean() - stats.t.ppf(0.975, len(d) - 1) * se),
                             float(d.mean() + stats.t.ppf(0.975, len(d) - 1) * se)]})
    return out


def main():
    arms = {k: describe([nll_of(v) for v in seeds.values()]) for k, seeds in ARMS.items()}
    comps = {
        "v2d - standard": paired("v2d (final algebraic)", "standard"),
        "v2c - standard": paired("v2c", "standard"),
        "hybrid - v2c": paired("hybrid (v2c + softmax attention)", "v2c"),
        "v2d - (standard + QK-norm + algebraic head)": paired("v2d (final algebraic)",
                                                              "standard + QK-norm + algebraic head"),
        "(standard + QK-norm + algebraic head) - standard": paired("standard + QK-norm + algebraic head",
                                                                   "standard"),
    }
    out = {"arms": arms, "paired": comps}
    path = ROOT / "results" / "analysis" / "phase1_summary.json"
    path.write_text(json.dumps(out, indent=1))
    for k, v in arms.items():
        print(f"{k:40s} n={v['n']} {v['mean']:.4f} ± {v['se']:.4f}")
    for k, v in comps.items():
        extra = f" ± {v['se']:.4f} (t={v['t']:.1f}, p={v['p_two_sided']:.4f})" if "se" in v else ""
        print(f"{k:50s} {v['mean']:+.4f}{extra}")


if __name__ == "__main__":
    main()
