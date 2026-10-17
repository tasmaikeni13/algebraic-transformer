#!/usr/bin/env python3
"""Build publication figures from recorded comparison JSON; never invent runs."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"standard": "#3269b8", "algebraic": "#d65b37"}
INK, SURFACE, GRID = "#182438", "#fbfcfe", "#dfe5ed"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK,
                     "font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "legend.frameon": False})


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    svg = out / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(out / f"{name}.png")


def architecture(out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, arch, blocks in [
        (axes[0], "standard", ["Tied token embedding", "Learned RMSNorm", "RoPE + softmax attention", "RMSNorm + SwiGLU", "Tied projection + cross-entropy"]),
        (axes[1], "algebraic", ["Tied token embedding", "Learned RMSNorm", "QK norm + Cayley rotations\nAlgebraic attention + learned sink", "RMSNorm + ALU-GLU", "Tied projection + algebraic likelihood\nExact log-loss gradient"]),
    ]:
        ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis("off")
        ax.text(5, 10.5, f"{arch.capitalize()} Transformer", ha="center", fontsize=17, weight="bold", color=COLORS[arch])
        for index, label in enumerate(blocks):
            y = 8.9 - 1.8*index
            ax.add_patch(FancyBboxPatch((.8, y-.65), 8.4, 1.3, boxstyle="round,pad=0.12", ec=COLORS[arch], fc="white", lw=1.6))
            ax.text(5, y, label, ha="center", va="center", fontsize=11)
            if index < len(blocks)-1:
                ax.add_patch(FancyArrowPatch((5, y-.82), (5, y-1.02), arrowstyle="-|>", mutation_scale=14, color=INK))
        ax.text(5, .3, "Attention and feed-forward blocks repeat across layers.\nBoth models share tokenization, shape, batch, seed, and data.", ha="center", va="center", fontsize=9, color="#59677b")
    fig.tight_layout()
    save(fig, out, "architecture")


def overview(summary, out):
    archs = summary["architectures"]
    title = experiment_title(summary)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    for ax, key, scale, label in [(axes[0], "final_eval_ppl", 1, "Held-out perplexity (lower is better)"),
                                  (axes[1], "mean_tok_per_s", 1e-6, "Training throughput (million tokens/s)")]:
        for i, arch in enumerate(archs):
            stat = summary["training_metrics"][key][arch]
            mean, sd = stat["mean"]*scale, (stat["sd"] or 0)*scale
            ax.barh(i, mean, xerr=sd, color=COLORS[arch], height=.58, capsize=3)
            ax.text(mean + (.2 if scale == 1 else .025), i, f"{mean:.2f}", va="center", color=INK)
        ax.set_yticks(range(len(archs)), archs); ax.invert_yaxis()
        ax.set_xlabel(label); ax.set_xlim(0, max(s["mean"]*scale for s in summary["training_metrics"][key].values())*1.18)
        ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    uncertainty = "Error bars: sample SD across training seeds" if len(summary["training_seeds"])>1 else "One training seed; seed variance is not estimated"
    fig.suptitle(title + "\n" + uncertainty, fontsize=12, weight="bold")
    save(fig, out, "results_overview")


def public_tasks(summary, out):
    rows = [r for r in summary["metrics"] if r["suite"] == "zero_shot"]
    archs = summary["architectures"]
    fig, ax = plt.subplots(figsize=(10, 6.2), layout="constrained")
    positions = np.arange(len(rows))
    width = .75/len(archs)
    for index, arch in enumerate(archs):
        vals = [100*r["architectures"][arch]["mean"] for r in rows]
        errors = [100*(r["architectures"][arch]["sd"] or 0) for r in rows]
        ax.barh(positions + (index-(len(archs)-1)/2)*width, vals, height=width*.9,
                xerr=errors, capsize=2, label=arch, color=COLORS[arch])
    ax.set_yticks(positions, [r["task"].replace("_", " ").title() for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%) · project scoring protocol"); ax.set_xlim(0, 75)
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    seeds = summary["training_seeds"]
    uncertainty = ("Mean and sample SD across seeds " if len(seeds)>1 else "Training seed ") + ", ".join(map(str, seeds))
    training = summary["training"]
    title = f"Public tasks · {training['size'].upper()} / {training['tokens']/1e9:g}B tokens"
    ax.set_title(title + "\n" + uncertainty, loc="left", pad=14)
    ax.legend(loc="lower right")
    save(fig, out, "public_tasks")


def experiment_title(summary):
    training = summary["training"]
    size = training["size"].upper()
    tokens = training["tokens"] / 1e9
    count = len(summary["training_seeds"])
    return f"{size} parameters · {tokens:g}B FineWeb-Edu tokens · {count} paired seed" + ("s" if count>1 else "")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, default=ROOT / "results/main")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/figures")
    a = p.parse_args()
    source = a.results_root / "comparison.json"
    summary = json.loads(source.read_text())
    architecture(a.out_dir)
    overview(summary, a.out_dir)
    public_tasks(summary, a.out_dir)
    (a.out_dir / "metadata.json").write_text(json.dumps({"source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "training_seeds": summary["training_seeds"], "generator": "analysis/plot_figures.py"}, indent=2)+"\n")


if __name__ == "__main__":
    main()
