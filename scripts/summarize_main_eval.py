#!/usr/bin/env python3
"""Compare matched post-training evaluations across architectures and seeds."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sota.benchmarking import (RECALL_PROTOCOL, ZERO_SHOT_PROTOCOL, load_jsonl,
                                   sha256_file, write_json)


def mean_sd(values):
    return {"mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "seed_values": values}


def read_records(directory, archs, seeds):
    records, predictions = {}, {}
    for suite, protocol in (("zero_shot", ZERO_SHOT_PROTOCOL), ("recall", RECALL_PROTOCOL)):
        reference = None
        for arch in archs:
            for seed in seeds:
                path = directory / f"{arch}_s{seed}_{suite}.json"
                record = json.loads(path.read_text())
                if record["protocol"] != protocol or record["limited"]:
                    raise ValueError(f"incompatible or limited evaluation: {path}")
                args = record["checkpoint"]["training_args"]
                if args["preset"] != arch or args["seed"] != seed:
                    raise ValueError(f"mislabeled training seed/architecture: {path}")
                prediction_path = path.with_suffix(".predictions.jsonl")
                if sha256_file(prediction_path) != record["predictions_sha256"]:
                    raise ValueError(f"prediction hash mismatch: {prediction_path}")
                details = load_jsonl(prediction_path)
                ids = [(r["task"], r["id"]) for r in details]
                if len(ids) != len(set(ids)):
                    raise ValueError(f"duplicate predictions: {path}")
                signature = {"datasets": {t: d["sha256"] for t, d in
                                          record["dataset_metadata"]["datasets"].items()},
                             "protocol": protocol, "source": record["source"]["sha256"],
                             "backend": record["backend"], "batch": record["batch"],
                             "dtype": record["config"]["dtype"], "ids": set(ids)}
                if reference is None:
                    reference = signature
                elif signature != reference:
                    raise ValueError(f"evaluations use different examples, protocol, source, or execution: {path}")
                records[(suite, arch, seed)] = record
                predictions[(suite, arch, seed)] = details
        # The two suites must come from the same saved weights for each seed.
    for arch in archs:
        for seed in seeds:
            if records[("zero_shot", arch, seed)]["checkpoint"] != records[("recall", arch, seed)]["checkpoint"]:
                raise ValueError(f"different public/recall checkpoints: {arch}, seed {seed}")
    return records, predictions


def summarize(records, predictions, archs, seeds):
    metrics = defaultdict(dict)
    cells = defaultdict(dict)
    for arch in archs:
        for seed in seeds:
            for task, result in records[("zero_shot", arch, seed)]["results"].items():
                metrics[("zero_shot", task, "all", "accuracy")][(arch, seed)] = result["accuracy"]
            groups = defaultdict(list)
            for row in predictions[("recall", arch, seed)]:
                groups[(row["task"], row["condition"])].append(row)
            for (task, condition), rows in groups.items():
                primary = "unsupported_target_hit" if condition == "absent" else "exact_retrieval"
                metrics[("recall", task, condition, primary)][(arch, seed)] = statistics.mean(r["exact_match"] for r in rows)
                metrics[("recall", task, condition, "recognition_accuracy")][(arch, seed)] = statistics.mean(r["recognition_correct"] for r in rows)
                metrics[("recall", task, condition, "needle_recall")][(arch, seed)] = sum(r["needle_hits"] for r in rows) / sum(r["needles"] for r in rows)
            for task, task_cells in records[("recall", arch, seed)]["results"].items():
                for cell in task_cells:
                    key = (task, cell["condition"], cell["context_budget"], cell["requested_depth"])
                    cells[key][(arch, seed)] = cell["exact_match"]

    def comparison(values):
        if set(values) != {(arch, seed) for arch in archs for seed in seeds}:
            raise ValueError("missing matched architecture/seed measurements")
        by_arch = {arch: mean_sd([values[(arch, seed)] for seed in seeds]) for arch in archs}
        delta = {}
        if "standard" in archs:
            delta = {arch: mean_sd([100 * (values[(arch, seed)] - values[("standard", seed)])
                                   for seed in seeds]) for arch in archs if arch != "standard"}
        return {"architectures": by_arch, "paired_delta_vs_standard_pp": delta}

    rows = [{"suite": suite, "task": task, "condition": condition, "metric": metric,
             **comparison(values)} for (suite, task, condition, metric), values in sorted(metrics.items())]
    cell_rows = [{"task": task, "condition": condition, "context_budget": budget,
                  "requested_depth": depth, **comparison(values)}
                 for (task, condition, budget, depth), values in sorted(cells.items())]
    training_metrics = {}
    for name in ("final_eval_nll", "final_eval_ppl", "train_seconds", "mean_tok_per_s"):
        if all(name in records[("zero_shot", arch, seed)].get("training_metrics", {})
               for arch in archs for seed in seeds):
            training_metrics[name] = {arch: mean_sd([records[("zero_shot", arch, seed)]["training_metrics"][name]
                                                    for seed in seeds]) for arch in archs}
    return {"architectures": archs, "training_seeds": seeds, "training_metrics": training_metrics,
            "complete_three_arm_three_seed_comparison": set(archs) == {"standard", "algebraic", "hybrid"}
                                                        and set(seeds) == {42, 43, 44},
            "uncertainty": "sample SD across training seeds; paired deltas use the same seed and examples; descriptive only",
            "metrics": rows, "recall_cells": cell_rows}


def write_tables(directory, summary):
    archs, seeds = summary["architectures"], summary["training_seeds"]

    def format_stat(stat, percentage=True):
        scale = 100 if percentage else 1
        mean = f"{scale*stat['mean']:.2f}"
        return mean if stat["sd"] is None else f"{mean} ± {scale*stat['sd']:.2f}"

    lines = ["# Post-training comparison", "", f"Training seeds: {', '.join(map(str, seeds))}.", "",
             "Scores are percentages, mean ± sample SD across training seeds. "
             "Paired differences use the same seed and examples. No pooled item confidence interval is implied.", ""]
    if not summary["complete_three_arm_three_seed_comparison"]:
        lines += ["**Subset comparison: the full three-architecture, three-seed result is incomplete.**", ""]
    if summary["training_metrics"]:
        lines += ["## Held-out language modeling and training telemetry", "",
                  "NLL is in nats/token; perplexity is exp(NLL). Training times and throughput are trainer telemetry.", "",
                  "| metric | " + " | ".join(archs) + " |", "|---|" + "---|" * len(archs)]
        for metric, stats in summary["training_metrics"].items():
            lines.append("| " + metric + " | " + " | ".join(format_stat(stats[a], False) for a in archs) + " |")
        lines.append("")
    for title, selector in (
        ("Public tasks", lambda r: r["suite"] == "zero_shot"),
        ("Recall with evidence", lambda r: r["condition"] == "present" and r["metric"] == "exact_retrieval"),
        ("Recall controls and diagnostics", lambda r: r["suite"] == "recall" and
         (r["condition"] != "present" or r["metric"] != "exact_retrieval")),
    ):
        selected = [r for r in summary["metrics"] if selector(r)]
        lines += [f"## {title}", "", "| task / condition / metric | " + " | ".join(archs) + " |",
                  "|---|" + "---|" * len(archs)]
        for row in selected:
            label = row["task"] if row["suite"] == "zero_shot" else f"{row['task']} / {row['condition']} / {row['metric']}"
            lines.append("| " + label + " | " + " | ".join(format_stat(row["architectures"][a]) for a in archs) + " |")
        lines.append("")
    if "standard" in archs and len(archs) > 1:
        others = [a for a in archs if a != "standard"]
        lines += ["## Paired difference from standard", "", "Percentage points, mean ± sample SD across matched training seeds.", "",
                  "| task | " + " | ".join(others) + " |", "|---|" + "---|" * len(others)]
        for row in summary["metrics"]:
            if row["suite"] == "zero_shot" or (row["condition"] == "present" and row["metric"] == "exact_retrieval"):
                lines.append("| " + row["task"] + " | " + " | ".join(format_stat(row["paired_delta_vs_standard_pp"][a], False) for a in others) + " |")
        lines.append("")
    lines += ["The longest recall budget includes answer space: 2,032 prompt tokens + 16 generated tokens.", "",
              "Absent-evidence target hits are unsupported guesses, not successful retrieval. "
              "Recognition has four choices (25% chance); short controls check whether the answer format is usable.", ""]
    (directory / "comparison.md").write_text("\n".join(lines))
    with (directory / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["suite", "task", "condition", "metric", "architecture", "mean", "sample_sd",
                         "paired_delta_vs_standard_pp", "paired_delta_sample_sd_pp"])
        for row in summary["metrics"]:
            for arch in archs:
                stat = row["architectures"][arch]
                delta = row["paired_delta_vs_standard_pp"].get(arch, {})
                writer.writerow([row["suite"], row["task"], row["condition"], row["metric"], arch,
                                 stat["mean"], stat["sd"], delta.get("mean"), delta.get("sd")])
    with (directory / "recall_cells.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "condition", "context_budget", "depth", "architecture", "mean", "sample_sd"])
        for row in summary["recall_cells"]:
            for arch in archs:
                stat = row["architectures"][arch]
                writer.writerow([row["task"], row["condition"], row["context_budget"], row["requested_depth"],
                                 arch, stat["mean"], stat["sd"]])


def plot_needle(directory, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cells = [c for c in summary["recall_cells"] if c["task"] == "needle_passkey" and c["condition"] == "present"]
    if not cells:
        return
    budgets = sorted({c["context_budget"] for c in cells})
    depths = sorted({c["requested_depth"] for c in cells})
    archs = summary["architectures"]
    fig, axes = plt.subplots(1, len(archs), figsize=(4.2 * len(archs), 3.7), squeeze=False, layout="constrained")
    for ax, arch in zip(axes[0], archs):
        matrix = np.full((len(budgets), len(depths)), np.nan)
        for cell in cells:
            matrix[budgets.index(cell["context_budget"]), depths.index(cell["requested_depth"])] = 100 * cell["architectures"][arch]["mean"]
        plotted = ax.imshow(matrix, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        ax.set_title(arch)
        ax.set_xticks(range(len(depths)), [f"{d:.0%}" for d in depths])
        ax.set_yticks(range(len(budgets)), budgets)
        ax.set_xlabel("Needle position in distractor region")
        ax.set_ylabel("Context budget including answer")
        for row in range(len(budgets)):
            for col in range(len(depths)):
                value = matrix[row, col]
                ax.text(col, row, f"{value:.1f}", ha="center", va="center", color="white" if value < 50 else "black")
    fig.colorbar(plotted, ax=list(axes[0]), label="Exact retrieval (%) across training seeds", shrink=.85)
    fig.savefig(directory / "needle_recall.png", dpi=200)
    fig.savefig(directory / "needle_recall.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/main")
    p.add_argument("--archs", default="standard,algebraic,hybrid")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()
    archs = a.archs.split(",")
    if len(set(archs)) != len(archs) or len(set(a.seeds)) != len(a.seeds):
        p.error("architectures and seeds must be unique")
    records, predictions = read_records(a.out_dir, archs, a.seeds)
    summary = summarize(records, predictions, archs, a.seeds)
    write_json(a.out_dir / "comparison.json", summary)
    write_tables(a.out_dir, summary)
    if not a.no_plots:
        plot_needle(a.out_dir, summary)
    print(f"Comparison: {a.out_dir / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
