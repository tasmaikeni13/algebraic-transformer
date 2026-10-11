#!/usr/bin/env python3
"""Summarize independent Monte Carlo seeds and paired hardware rounds.

Seed-level bootstrap for numerical errors avoids treating correlated keys or
implementation variants as independent samples. Hardware intervals describe
variation inside one run; they are not cross-device/generalization intervals.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def mean_ci(values, seed=2026, draws=20000):
    a = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    boot = a[rng.integers(len(a), size=(draws, len(a)))].mean(1)
    return {"mean": float(a.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(), "seeds": len(a)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    root = args.root
    mc = json.loads((root / "results/analysis/attention_numerics.json").read_text())
    result = {"schema": 1, "monte_carlo": [], "hardware": [],
              "hardware_interval_scope": "paired within-run rounds, fixed data seed; not across machines/workloads",
              "objective_status": "not achieved: algebraic training latency still exceeds the fastest measured standard"}
    for distribution in sorted({x["distribution"] for x in mc["trials"]}):
        for scale in sorted({x["score_sd_target"] for x in mc["trials"]}):
            rows = sorted((x for x in mc["trials"] if x["distribution"] == distribution and
                           x["score_sd_target"] == scale), key=lambda x: x["seed"])
            group = {"distribution": distribution, "score_sd_target": scale,
                     "rows": len(rows) * mc["rows_per_seed"], "by_mode": {}}
            for mode in rows[0]["metrics"]:
                metrics = [x["metrics"][mode] for x in rows]
                group["by_mode"][mode] = {
                    "tv_seed_means": mean_ci([x["tv_error"]["mean"] for x in metrics]),
                    "maximum_observed_tv": max(x["tv_error"]["max"] for x in metrics),
                    "nonfinite_rows": sum(x["nonfinite_rows"] for x in metrics),
                    "clamped_scores": sum(x["clamped_scores"] for x in metrics),
                    "gradient_relative_l2_seed_means": mean_ci([x["gradient_relative_l2_error"]["mean"] for x in metrics]),
                }
            root_error = [x["metrics"]["fast_root"]["tv_error"]["mean"] for x in rows]
            inv_error = [x["metrics"]["fast_inverseZ"]["tv_error"]["mean"] for x in rows]
            group["paired_root_minus_inverseZ_tv"] = mean_ci(np.asarray(root_error) - inv_error)
            result["monte_carlo"].append(group)
    for size in ("60m", "125m"):
        path = root / f"results/kernels/attention_confirm_{size}.json"
        bench = json.loads(path.read_text())
        reference = bench["comparisons"]["training"]["reference"]
        candidate = "asinh_guarded_512x512"
        result["hardware"].append({"size": size, "shape": bench["shape"],
            "kernel_sha256": bench["kernel_sha256"], "rounds": bench["rounds"],
            "reference": reference, "candidate": candidate,
            "standard_training_median_ms": bench["results"][reference]["training"]["median_ms"],
            "algebraic_training_median_ms": bench["results"][candidate]["training"]["median_ms"],
            "algebraic_over_standard_latency": bench["comparisons"]["training"]["latency_ratios"][candidate],
            "parity_proved": False})
    out = root / "results/analysis/attention_statistics.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for h in result["hardware"]:
        print(h["size"], h["algebraic_over_standard_latency"])


if __name__ == "__main__":
    main()
