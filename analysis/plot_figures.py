#!/usr/bin/env python3
"""Figures for the reports (results/figures/*.png) from recorded JSON outputs.

Palette: the validated reference categorical slots (blue, orange, aqua,
yellow) on a light chart surface; every series is direct-labeled in text ink
because aqua and yellow sit below 3:1 contrast. One measure per axis.
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.labelcolor": INK2,
})


def line(ax, x, y, color, label, marker=True):
    ax.plot(x, y, color=color, lw=1.6, label=label, zorder=3,
            marker="o" if marker else None, ms=5.5, mec=SURFACE, mew=1.4)


def end_label(ax, x, y, text, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center",
                color=INK2, fontsize=9)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def phase3():
    s = json.loads((ROOT / "results/phase3/summary.json").read_text())
    runs = s["runs"]
    std, alg, ctl = runs[s["best"]["std"]], runs[s["best"]["alg"]], runs.get("ctrl_qk_head_lr0.003")
    tok = 524288 / 1e9
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.0))
    for r, c, name in ((std, S1, "standard"), (alg, S2, "algebraic"), (ctl, S3, "std + QK-norm + alg. head")):
        if r is None:
            continue
        xs = [p[0] * tok for p in r["curve"]]
        ys = [p[1] for p in r["curve"]]
        line(a, xs, ys, c, name, marker=False)
        dy = {"standard": 9, "algebraic": -10, "std + QK-norm + alg. head": 0}[name]
        end_label(a, xs[-1], ys[-1], f"{name.split(' ')[0] if name != 'std + QK-norm + alg. head' else 'control'} {ys[-1]:.3f}", dy)
    a.set_xlim(right=1.3)
    a.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    a.set_title("Held-out NLL during training (60M, best LR each)")
    a.set_xlabel("training tokens (billions)")
    a.set_ylabel("validation NLL (nats)")
    a.legend(loc="upper right")
    base = {p[0]: p[1] for p in std["curve"]}
    for r, c, name, dy in ((alg, S2, "algebraic", -6), (ctl, S3, "std + QK-norm + alg. head", 6)):
        if r is None:
            continue
        pts = [(p[0] * tok, p[1] - base[p[0]]) for p in r["curve"] if p[0] in base]
        line(b, [p[0] for p in pts], [p[1] for p in pts], c, name)
        end_label(b, pts[-1][0], pts[-1][1], f"{name} {pts[-1][1]:+.3f}", dy)
    b.axhline(0, color=BASE, lw=1.0, zorder=2)
    b.set_title("Difference to the standard model")
    b.set_xlabel("training tokens (billions)")
    b.set_ylabel("Δ validation NLL (nats)")
    b.set_xlim(right=1.45)
    b.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    b.legend(loc="lower left")
    save(fig, "phase3_curves.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for arch, c, name in (("std", S1, "standard"), ("alg", S2, "algebraic")):
        pts = sorted((r["lr"], r["final_eval_nll"]) for n, r in runs.items() if n.startswith(arch + "_"))
        line(ax, [p[0] for p in pts], [p[1] for p in pts], c, name)
        end_label(ax, pts[-1][0], pts[-1][1], name)
    if ctl is not None:
        ax.plot([ctl["lr"]], [ctl["final_eval_nll"]], "D", color=S3, ms=6, mec=SURFACE, mew=1.2,
                label="std + QK-norm + alg. head", zorder=4)
        ax.annotate("std + QK-norm + alg. head", (ctl["lr"], ctl["final_eval_nll"]), xytext=(0, -14),
                    textcoords="offset points", ha="center", va="top", color=INK2, fontsize=9)
        ax.set_ylim(bottom=ctl["final_eval_nll"] - 0.018)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1.5e-3, 3e-3, 6e-3])
    ax.set_xticklabels(["1.5e-3", "3e-3", "6e-3"])
    ax.set_xlim(1.3e-3, 9e-3)
    ax.set_title("Learning-rate sensitivity (60M, 1B tokens)")
    ax.set_xlabel("peak learning rate")
    ax.set_ylabel("final validation NLL (nats)")
    ax.legend(loc="upper left")
    save(fig, "phase3_lr.png")


def attribution():
    rows = json.loads((ROOT / "results/analysis/phase1_batch1_2.json").read_text())
    ref = sum(rows[k]["nll"] for k in ("std_lr0.002", "std_lr0.002_s1", "std_lr0.002_s2")) / 3
    items = [("OACE head", "std+v1head"), ("ARDS schedule + wd 0.01", "std_v1recipe"),
             ("no norm gains (AVN)", "std+v1norm"), ("octic attention", "std+v1attn"),
             ("ALU activation", "std+v1ffn"), ("first-study Cayley grid", "std+v1pos")]
    vals = [(name, rows[key]["nll"] - ref) for name, key in items]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ys = list(range(len(vals)))[::-1]
    ax.barh(ys, [v for _, v in vals], height=0.6, color=S1, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for y, (_, v) in zip(ys, vals):
        ax.annotate(f"{v:+.3f}", (v, y), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if v >= 0 else "right", va="center", color=INK2, fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([n for n, _ in vals], color=INK2)
    ax.axvline(0, color=BASE, lw=1.0, zorder=2)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.02, 0.1)
    ax.set_title("First-study components, one at a time on the standard model")
    ax.set_xlabel("Δ validation NLL vs standard (nats; proxy, 197M tokens)")
    save(fig, "v1_attribution.png")


def head_and_bottleneck():
    rows = json.loads((ROOT / "results/analysis/phase1_batch1_2.json").read_text())
    heads = [("softmax CE", "v2_headce"), ("n=32, c=32", "v2_head32x32"), ("n=16, c=16", "v2prov"),
             ("n=8, c=2", "v2_head8x2")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.0))
    xs = list(range(len(heads)))
    vals = [rows[k]["nll"] for _, k in heads]
    line(a, xs, vals, S2, "v2 backbone, head varied")
    for x, v in zip(xs, vals):
        a.annotate(f"{v:.4f}", (x, v), xytext=(0, 8), textcoords="offset points", ha="center",
                   color=INK2, fontsize=9)
    a.set_xticks(xs)
    a.set_xticklabels([h for h, _ in heads], color=INK2)
    a.set_xlim(-0.4, len(heads) - 0.6)
    a.set_ylim(min(vals) - 0.006, max(vals) + 0.008)
    a.set_title("Output head: stronger asinh compression is better")
    a.set_ylabel("validation NLL (nats; proxy, one seed)")

    c = json.loads((ROOT / "results/analysis/head_analysis_c.json").read_text())["C_rank_bottleneck"]["rows"]
    ranks = sorted({r["rank"] for r in c})
    series = (("kl_softmax", S1, "softmax", 6), ("kl_alg16", S2, "asinh n=16, c=16", 0),
              ("kl_alg_c1", S3, "asinh n=8, c=1", -6))
    for key, col, name, dy in series:
        ys = [sum(r[key] for r in c if r["rank"] == d) / sum(1 for r in c if r["rank"] == d) for d in ranks]
        line(b, ranks, ys, col, name)
        end_label(b, ranks[-1], ys[-1], f"{name} {ys[-1]:.3f}", dy)
    b.set_xscale("log", base=2)
    b.set_xticks(ranks)
    b.set_xticklabels([str(d) for d in ranks])
    b.set_xlim(6.5, 180)
    b.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ticks = [0.25, 0.5, 1.0, 1.5]
    b.yaxis.set_major_locator(FixedLocator(ticks))
    b.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    b.yaxis.set_minor_locator(NullLocator())
    b.set_title("Rank bottleneck: best KL to a full-rank teacher")
    b.set_xlabel("logit rank d")
    b.set_ylabel("KL (nats, mean of 6 trials, log scale)")
    b.legend(loc="lower left")
    save(fig, "head_and_bottleneck.png")


def recall():
    rows = json.loads((ROOT / "results/analysis/attention_analysis.json").read_text())["B_associative_recall"]
    kernels = (("softmax", S1, "softmax"), ("v1_octic_n8_c1_sink1", S2, "first-study kernel n=8, c=1"),
               ("asinh_n16_c16", S3, "asinh n=16, c=16"), ("asinh_n8_c2", S4, "asinh n=8, c=2"))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    offsets = {"softmax": 7, "v1_octic_n8_c1_sink1": 0, "asinh_n16_c16": -7, "asinh_n8_c2": 0}
    for key, col, name in kernels:
        pts = sorted((r["items"], 100 * (1 - r["acc_mean"])) for r in rows if r["kernel"] == key)
        line(ax, [p[0] for p in pts], [max(p[1], 0.01) for p in pts], col, name)
        end_label(ax, pts[-1][0], pts[-1][1], f"{pts[-1][1]:.2f}%", offsets[key])
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 128, 512])
    ax.set_xticklabels(["32", "128", "512"])
    ax.set_xlim(24, 900)
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ticks = [0.03, 0.1, 0.3, 1, 3, 10, 30]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_title("Associative recall: error rate vs distractors")
    ax.set_xlabel("stored items (distractors)")
    ax.set_ylabel("retrieval error (%, mean of 3 seeds)")
    ax.legend(loc="upper left")
    save(fig, "recall.png")


if __name__ == "__main__":
    phase3()
    attribution()
    head_and_bottleneck()
    recall()
