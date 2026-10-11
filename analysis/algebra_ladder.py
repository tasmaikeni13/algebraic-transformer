#!/usr/bin/env python3
"""How much algebra do we really need?

Climbs from the standard model to the fully algebraic one, adding one
algebraic piece per rung (proxy scale, 3-4 seeds per rung, flash kernels):

  L0 standard
  L1 + algebraic likelihood head          (L1b: QK-norm alone, for reference)
  L2 + QK-normalization       = the hybrid
  L3 + learned attention sink
  L4 + ALU-GLU (beta 1/2)
  L5 + geometric Cayley rotations         (everything but the attention kernel)
  L6 + asinh attention        = the fully algebraic model (v2d)

Reports mean held-out NLL ± SE, the paired difference to the standard model
(seeds 0-2), the share of the full gain each rung recovers, and single-chip
training throughput. Writes results/analysis/algebra_ladder.json and
results/figures/algebra_ladder.png.
"""

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ["p1b1", "p1b2", "p1b3", "p1b4", "p1b5", "ladder"]

RUNGS = [
    ("L0 standard", {0: "std_flash_s0", 1: "std_lr0.002_s1", 2: "std_lr0.002_s2", 3: "std_flash_s3"}),
    ("L1 + algebraic head", {0: "L1_head_s0", 1: "L1_head_s1", 2: "L1_head_s2"}),
    ("L1b QK-norm only", {0: "L1b_qk_s0", 1: "L1b_qk_s1", 2: "L1b_qk_s2"}),
    ("L2 + QK-norm (hybrid)", {0: "std+qk+head", 1: "std+qk+head_s1", 2: "std+qk+head_s2"}),
    ("L3 + sink", {0: "L3_sink_s0", 1: "L3_sink_s1", 2: "L3_sink_s2"}),
    ("L4 + ALU-GLU", {0: "L4_alu_s0", 1: "L4_alu_s1", 2: "L4_alu_s2"}),
    ("L5 + Cayley rotations", {0: "L5_cayley_s0", 1: "L5_cayley_s1", 2: "L5_cayley_s2"}),
    ("L6 + asinh attention (algebraic)", {0: "v2d_s0", 1: "v2d_s1", 2: "v2d_s2", 3: "v2d_s3"}),
]


def result(name):
    for g in GROUPS:
        p = ROOT / "runs" / g / name / "result.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


def main():
    table = []
    base = {s: result(n)["final_eval_nll"] for s, n in RUNGS[0][1].items()}
    full = None
    for label, seeds in RUNGS:
        res = {s: result(n) for s, n in seeds.items()}
        res = {s: r for s, r in res.items() if r is not None}
        if not res:
            continue
        nll = np.array([r["final_eval_nll"] for r in res.values()])
        paired = np.array([res[s]["final_eval_nll"] - base[s] for s in res if s in base])
        flash_tps = [r["mean_tok_per_s"] for r in res.values()
                     if dict(kv.split("=") for kv in r["args"]["set"]).get("attn_impl") == "flash"]
        row = {"rung": label, "seeds": sorted(res), "mean_nll": float(nll.mean()),
               "se": float(nll.std(ddof=1) / math.sqrt(len(nll))) if len(nll) > 1 else None,
               "paired_vs_standard": float(paired.mean()) if len(paired) else None,
               "paired_se": float(paired.std(ddof=1) / math.sqrt(len(paired))) if len(paired) > 1 else None,
               "tokens_per_s_one_chip": float(np.mean(flash_tps)) if flash_tps else None}
        table.append(row)
        if label.startswith("L6"):
            full = row["paired_vs_standard"]
    for row in table:
        if full and row["paired_vs_standard"] is not None:
            row["share_of_full_gain"] = row["paired_vs_standard"] / full
    out = {"rungs": table}
    (ROOT / "results/analysis/algebra_ladder.json").write_text(json.dumps(out, indent=1))
    print(f"{'rung':36s} {'NLL':>8s} {'±SE':>7s} {'Δ std':>8s} {'±SE':>7s} {'share':>6s} {'tok/s':>7s}")
    for r in table:
        se = f"{r['se']:.4f}" if r["se"] else "   -  "
        pse = f"{r['paired_se']:.4f}" if r["paired_se"] else "   -  "
        sh = f"{r.get('share_of_full_gain', 0) * 100:5.0f}%" if r.get("share_of_full_gain") is not None else "   -  "
        tps = f"{r['tokens_per_s_one_chip'] / 1e3:6.0f}K" if r["tokens_per_s_one_chip"] else "    - "
        print(f"{r['rung']:36s} {r['mean_nll']:8.4f} {se:>7s} {r['paired_vs_standard']:+8.4f} {pse:>7s} {sh} {tps}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_figures import SURFACE, INK2, BASE, S1, S2  # noqa: E402  (shared style)
    rows = [r for r in table if not r["rung"].startswith("L1b")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        err = 1.96 * (r["paired_se"] or 0)
        a.errorbar(r["paired_vs_standard"], y, xerr=err, fmt="o", color=S2, ms=6, mec=SURFACE,
                   mew=1.2, ecolor=S2, elinewidth=1.4, capsize=3, zorder=3)
        a.annotate(f"{r['paired_vs_standard']:+.3f}", (r["paired_vs_standard"], y), xytext=(0, 9),
                   textcoords="offset points", ha="center", color=INK2, fontsize=9)
    a.axvline(0, color=BASE, lw=1.0)
    a.set_yticks(ys)
    a.set_yticklabels([r["rung"] for r in rows], color=INK2)
    a.grid(axis="y", visible=False)
    a.set_title("How much algebra is needed? Quality by rung")
    a.set_xlabel("Δ validation NLL vs standard (nats; proxy, paired seeds, 95% CI)")
    tps = [r["tokens_per_s_one_chip"] / 1e3 if r["tokens_per_s_one_chip"] else 0 for r in rows]
    b.barh(ys, tps, height=0.6, color=S1, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for y, t in zip(ys, tps):
        b.annotate(f"{t:.0f}K", (t, y), xytext=(4, 0), textcoords="offset points", va="center",
                   color=INK2, fontsize=9)
    b.set_yticks(ys)
    b.set_yticklabels([])
    b.grid(axis="y", visible=False)
    b.set_xlim(0, max(tps) * 1.2)
    b.set_title("Training throughput (one chip)")
    b.set_xlabel("tokens / s (proxy model, context 1024)")
    out_png = ROOT / "results/figures/algebra_ladder.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print("wrote", out_png)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
