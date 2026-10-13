#!/usr/bin/env python3
"""Seed-level routing summaries and paired timing for experimental links."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.attention_statistics import mean_ci
from analysis.attention_performance import ratio_ci


def main():
    mc = json.loads((ROOT / "results/analysis/attention_link_analysis.json").read_text())
    metrics = ("cue_probability_mean", "KL_to_temperature_matched_softmax_mean",
               "KL_to_deployed_asinh_mean", "TV_to_deployed_asinh_mean",
               "fraction_scores_clipped", "entropy_mean")
    result = {"schema": 1, "routing": [], "hardware": [],
              "routing_scope": "bootstrap over eight independent synthetic seeds; static cue probability, not trained accuracy",
              "hardware_scope": "paired rounds within one chip/process; not cross-device or model generalization",
              "model_adopted": False, "parity_achieved": False}
    for kernel in mc["specs"]:
        for gain in sorted({x["gain_product"] for x in mc["trials"]}):
            rows = sorted((x for x in mc["trials"] if x["kernel"] == kernel and
                           x["gain_product"] == gain), key=lambda x: x["seed"])
            result["routing"].append({"kernel": kernel, "gain_product": gain,
                "rows": len(rows) * mc["rows_per_seed"],
                "metrics": {m: mean_ci([x[m] for x in rows]) for m in metrics},
                "cue_tie_rows": sum(x["cue_tie_rows"] for x in rows)})
    for filename in ("attention_link_probe.json", "attention_link_resident.json",
                     "attention_quadratic_probe.json", "attention_quadratic_reciprocal.json"):
        path = ROOT / "results/kernels" / filename
        bench = json.loads(path.read_text())
        standard = bench["results"]["standard_compact"]["training"]["samples_ms"]
        for kernel, timings in bench["results"].items():
            result["hardware"].append({"artifact": filename, "kernel": kernel,
                "shape": bench["shape"], "kernel_sha256": bench["kernel_sha256"],
                "probe_sha256": bench["probe_sha256"],
                "validation": bench["validation"].get(kernel),
                "forward_median_ms": timings["forward"]["median_ms"],
                "training_median_ms": timings["training"]["median_ms"],
                "paired_training_ratio": ratio_ci(timings["training"]["samples_ms"], standard)})
    out = ROOT / "results/analysis/attention_link_statistics.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}: {len(result['routing'])} routing groups, {len(result['hardware'])} hardware cases")


if __name__ == "__main__":
    main()
