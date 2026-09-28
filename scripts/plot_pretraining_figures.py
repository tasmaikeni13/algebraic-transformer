#!/usr/bin/env python3
"""Plot the recorded pretraining and evaluation measurements."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def plot_loss_curves(ledger_data: dict, output_dir: Path):
    """Show each training objective on its own axis; their scales differ."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    runs = ledger_data.get("runs", [])
    grouped = {"algebraic": [], "baseline": []}
    for run in runs:
        arch = run.get("architecture")
        losses = run.get("losses_sample", [])
        if arch in grouped and losses:
            grouped[arch].append(losses)

    colors = {"algebraic": "#1f77b4", "baseline": "#e66101"}
    labels = {"algebraic": "Algebraic Transformer", "baseline": "Standard Baseline"}

    for ax, (arch, loss_lists) in zip(axes, grouped.items()):
        if not loss_lists:
            continue
        min_len = min(len(l) for l in loss_lists)
        arr = np.asarray([l[:min_len] for l in loss_lists], dtype=np.float64)
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)
        steps = np.linspace(0, 2385, min_len)

        for l in arr:
            ax.plot(steps, l, color=colors[arch], alpha=0.2, linewidth=1.0)

        ax.plot(steps, mean_curve, color=colors[arch], label=labels[arch], linewidth=2.2)
        ax.fill_between(
            steps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=colors[arch],
            alpha=0.15,
        )

        ax.set_xlabel("Optimization step")
        ax.set_ylabel("Training objective")
        ax.set_title(labels[arch])
    fig.suptitle("Training traces (different objective scales)")
    plt.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=300)
    plt.close(fig)


def plot_validation_perplexity(ledger_data: dict, output_dir: Path):
    """Plot mean validation perplexity with standard-error bars."""
    summary = ledger_data.get("summary", {})
    fig, ax = plt.subplots(figsize=(6, 5))

    architectures = ["algebraic", "baseline"]
    names = ["Algebraic\nTransformer", "Standard\nBaseline"]
    colors = ["#1f77b4", "#e66101"]

    means = [summary.get(f"{arch}_mean_perplexity", 0.0) for arch in architectures]
    sems = [summary.get(f"{arch}_perplexity_sem", 0.0) for arch in architectures]

    bars = ax.bar(names, means, yerr=sems, capsize=6, color=colors, alpha=0.85, width=0.55, edgecolor="black", linewidth=1.2)

    for bar, mean, sem in zip(bars, means, sems):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height / 2.0,
            f"{mean:.2f}\n±{sem:.2f}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=12,
        )

    ax.set_title("Validation perplexity across three seeds")

    ax.set_ylabel("Held-Out FineWeb-Edu Perplexity")
    ax.set_ylim(0, max(means) * 1.35 if means else 100)
    plt.tight_layout()
    fig.savefig(output_dir / "validation_perplexity.png", dpi=300)
    plt.close(fig)


def plot_downstream_benchmarks(ledger_data: dict, output_dir: Path):
    """Plot zero-shot reasoning benchmarks comparison."""
    benchmark_summary = ledger_data.get("benchmark_summary", {})
    if not benchmark_summary:
        return

    benchmarks = ["arc_easy", "hellaswag", "piqa", "lambada"]
    display_names = ["ARC-Easy", "HellaSwag", "PIQA", "LAMBADA"]

    alg_means = [benchmark_summary.get(b, {}).get("algebraic", 0.0) * 100 for b in benchmarks]
    alg_sems = [benchmark_summary.get(b, {}).get("algebraic_sem", 0.0) * 100 for b in benchmarks]
    base_means = [benchmark_summary.get(b, {}).get("baseline", 0.0) * 100 for b in benchmarks]
    base_sems = [benchmark_summary.get(b, {}).get("baseline_sem", 0.0) * 100 for b in benchmarks]

    x = np.arange(len(benchmarks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width / 2, alg_means, width, yerr=alg_sems, capsize=5, label="Algebraic Transformer", color="#1f77b4", alpha=0.85, edgecolor="black")
    rects2 = ax.bar(x + width / 2, base_means, width, yerr=base_sems, capsize=5, label="Standard Baseline", color="#e66101", alpha=0.85, edgecolor="black")

    ax.set_ylabel("Zero-Shot Accuracy (%)")
    ax.set_title("Zero-shot accuracy after 2.5B training tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.legend(frameon=True, loc="upper right")
    ax.set_ylim(0, 100)

    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "downstream_benchmarks.png", dpi=300)
    plt.close(fig)


def plot_layer_second_moments(ledger_data: dict, output_dir: Path):
    """Plot layer-input second moment tracking showing unit-variance stability."""
    runs = [r for r in ledger_data.get("runs", []) if r.get("architecture") == "algebraic"]
    if not runs:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    layers = np.arange(1, 14)  # Twelve blocks and final normalization.

    for run in runs:
        moments = run.get("normalization_second_moments", [])
        if len(moments) >= 13:
            ax.plot(layers, moments[:13], marker="o", linewidth=1.8, alpha=0.7, label=f"Seed {run.get('seed')}")

    ax.axhspan(0.8, 1.3, color="green", alpha=0.12, label="Reference band [0.8, 1.3]")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Target E[h²] = 1.0")

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Layer Input Second Moment E[h²]")
    ax.set_title("Normalized layer-input second moments")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}" for i in range(1, 13)] + ["Final"])
    ax.set_ylim(0.7, 1.4)
    ax.legend(frameon=True, loc="lower left")
    plt.tight_layout()
    fig.savefig(output_dir / "layer_second_moments.png", dpi=300)
    plt.close(fig)


def plot_hardware_throughput(ledger_data: dict, output_dir: Path):
    """Plot throughput and step latency comparison on TPU v4."""
    runs = ledger_data.get("runs", [])
    if not runs:
        return

    grouped = {"algebraic": [], "baseline": []}
    for r in runs:
        arch = r.get("architecture")
        t = r.get("throughput_tokens_sec")
        if arch in grouped and t:
            grouped[arch].append(t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    architectures = ["algebraic", "baseline"]
    names = ["Algebraic\nTransformer", "Standard\nBaseline"]
    colors = ["#1f77b4", "#e66101"]

    throughputs = [np.mean(grouped[a]) if grouped[a] else 0.0 for a in architectures]
    step_latencies = [
        (512 * 2048) / t * 1000 if t > 0 else 0.0 for t in throughputs
    ]

    ax1.bar(names, throughputs, color=colors, alpha=0.85, width=0.5, edgecolor="black")
    ax1.set_ylabel("Throughput (Tokens / Second)")
    ax1.set_title("Sustained TPU v4 Throughput")
    for i, v in enumerate(throughputs):
        ax1.text(i, v * 0.9, f"{v:,.0f} tok/s", ha="center", va="top", color="white", fontweight="bold")

    ax2.bar(names, step_latencies, color=colors, alpha=0.85, width=0.5, edgecolor="black")
    ax2.set_ylabel("Step Latency (ms)")
    ax2.set_title("Per-Step Latency (Batch 512, Seq 2048)")
    for i, v in enumerate(step_latencies):
        ax2.text(i, v * 0.9, f"{v:.1f} ms", ha="center", va="top", color="white", fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "hardware_throughput.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=ROOT / "results/pretraining/pretraining_ledger.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/figures")
    args = parser.parse_args()

    if not args.ledger.exists():
        print(f"Ledger file not found at {args.ledger}. Skipping figure generation.", flush=True)
        return 1

    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.ledger.read_text())

    plot_loss_curves(data, args.output_dir)
    plot_validation_perplexity(data, args.output_dir)
    plot_downstream_benchmarks(data, args.output_dir)
    plot_layer_second_moments(data, args.output_dir)
    plot_hardware_throughput(data, args.output_dir)

    print(f"Successfully generated all publication PNG figures in {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
