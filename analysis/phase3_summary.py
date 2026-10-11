#!/usr/bin/env python3
"""Summarize Phase 3 (60M parameters, 1B tokens, one seed, 16 chips).

Reads runs/p3/*/result.json and log.jsonl; writes results/phase3/summary.json
with each run's final held-out NLL/perplexity, validation curve, throughput,
and the best learning rate per architecture.
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(group="p3"):
    rows = {}
    for d in sorted((ROOT / "runs" / group).iterdir()):
        res = d / "result.json"
        if not (d.is_dir() and res.exists()):
            continue
        r = json.loads(res.read_text())
        curve = []
        for line in (d / "log.jsonl").read_text().splitlines():
            e = json.loads(line)
            if "eval_nll" in e:
                curve.append([e["step"], e["eval_nll"]])
        rows[d.name] = {
            "preset": r["args"]["preset"], "lr": r["args"]["lr"], "overrides": r["args"]["set"],
            "params": r["n_params"], "tokens": r["total_steps"] * r["tokens_per_step"],
            "final_eval_nll": r["final_eval_nll"], "final_eval_ppl": r["final_eval_ppl"],
            "eval_tokens": r["eval_tokens"], "tokens_per_s": r["mean_tok_per_s"],
            "train_minutes": r["train_seconds"] / 60, "curve": curve,
            "nll_by_log2_position": r.get("eval_nll_by_log2_position"),
        }
    best = {}
    for name, r in rows.items():
        arch = name.split("_")[0]
        if arch in ("std", "alg", "hyb") and (arch not in best or r["final_eval_nll"] < rows[best[arch]]["final_eval_nll"]):
            best[arch] = name
    out = {"runs": rows, "best": best}
    if "std" in best and "alg" in best:
        a, s = rows[best["alg"]], rows[best["std"]]
        out["comparison"] = {
            "algebraic": best["alg"], "standard": best["std"],
            "nll_difference_alg_minus_std": a["final_eval_nll"] - s["final_eval_nll"],
            "ppl_ratio_alg_over_std": a["final_eval_ppl"] / s["final_eval_ppl"],
            "throughput_ratio_alg_over_std": a["tokens_per_s"] / s["tokens_per_s"],
        }
    if "std" in best and "hyb" in best:
        h, s = rows[best["hyb"]], rows[best["std"]]
        out["comparison_hybrid"] = {
            "hybrid": best["hyb"], "standard": best["std"],
            "nll_difference_hyb_minus_std": h["final_eval_nll"] - s["final_eval_nll"],
            "ppl_ratio_hyb_over_std": h["final_eval_ppl"] / s["final_eval_ppl"],
            "throughput_ratio_hyb_over_std": h["tokens_per_s"] / s["tokens_per_s"],
        }
    path = ROOT / "results" / "phase3" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"{'run':24s} {'lr':>8s} {'NLL':>8s} {'PPL':>8s} {'Mtok/s':>7s}")
    for name, r in sorted(rows.items()):
        print(f"{name:24s} {r['lr']:8.4g} {r['final_eval_nll']:8.4f} {r['final_eval_ppl']:8.3f} "
              f"{r['tokens_per_s']/1e6:7.2f}")
    for key in ("comparison", "comparison_hybrid"):
        if key in out:
            print(json.dumps(out[key], indent=1))


if __name__ == "__main__":
    main()
