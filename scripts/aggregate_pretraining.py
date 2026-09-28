#!/usr/bin/env python3
"""Combine the six training runs and their zero-shot benchmark records."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.tuning_records import write_json
from scripts.pretraining_experiments import summarize_pretraining
from scripts.pretraining_records import environment, run_evidence, source_hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    expected_hashes = source_hashes()
    records = []
    validations = []
    for path in sorted(args.input_dir.glob("*/run_metrics.json")):
        validation = run_evidence(path, expected_hashes)
        validations.append({"path": str(path), **validation})
        if validation["passed"]:
            records.append(json.loads(path.read_text())["run"])

    summary = summarize_pretraining(records)
    pretraining_passed = (
        len(validations) == 6
        and all(item["passed"] for item in validations)
        and summary["pretraining_safety_passed"]
        and summary["perplexity_parity_passed"]
    )
    benchmark_rows = []
    for path in sorted(args.input_dir.glob("*/benchmarks.json")):
        item = json.loads(path.read_text())
        results = item.get("results", {})
        metadata = item.get("dataset_metadata", {}).get("datasets", {})
        datasets_valid = all(
            name in results
            and name in metadata
            and metadata[name].get("limited") is False
            and isinstance(metadata[name].get("sha256"), str)
            and len(metadata[name]["sha256"]) == 64
            and results[name].get("total") == metadata[name].get("rows")
            and results[name].get("total", 0) > 0
            and 0 <= results[name].get("correct", -1) <= results[name]["total"]
            and np.isclose(
                results[name].get("accuracy", -1.0),
                results[name]["correct"] / results[name]["total"],
            )
            for name in ("arc_easy", "hellaswag", "piqa", "lambada")
        )
        if (
            item.get("scope") == "zero_shot_benchmarks"
            and item.get("environment", {}).get("source_sha256") == expected_hashes
            and datasets_valid
        ):
            benchmark_rows.append(item)
    benchmark_pairs = {(row.get("architecture"), row.get("seed")) for row in benchmark_rows}
    expected_pairs = {
        (architecture, seed)
        for architecture in ("algebraic", "baseline")
        for seed in (42, 43, 44)
    }
    benchmark_summary = {}
    metadata_fingerprints = {
        json.dumps(row.get("dataset_metadata"), sort_keys=True)
        for row in benchmark_rows
    }
    benchmark_parity = (
        len(benchmark_rows) == 6
        and benchmark_pairs == expected_pairs
        and len(metadata_fingerprints) == 1
    )
    for benchmark in ("arc_easy", "hellaswag", "piqa", "lambada"):
        means = {}
        sems = {}
        for architecture in ("algebraic", "baseline"):
            values = [
                row["results"][benchmark]["accuracy"]
                for row in benchmark_rows if row["architecture"] == architecture
            ]
            means[architecture] = float(np.mean(values)) if len(values) == 3 else None
            sems[architecture] = (
                float(np.std(values, ddof=1) / np.sqrt(3.0))
                if len(values) == 3 else None
            )
        delta = (
            means["algebraic"] - means["baseline"]
            if means["algebraic"] is not None and means["baseline"] is not None
            else None
        )
        benchmark_summary[benchmark] = {
            **means,
            "algebraic_sem": sems["algebraic"],
            "baseline_sem": sems["baseline"],
            "algebraic_minus_baseline": delta,
        }
        benchmark_parity = (
            benchmark_parity
            and np.isfinite(means["algebraic"])
            and means["algebraic"] >= means["baseline"] - 0.02
        )

    pretraining_passed = (
        pretraining_passed
        and summary["algebraic_perplexity_sem"] is not None
        and summary["algebraic_perplexity_sem"] <= 0.15
        and summary["baseline_perplexity_sem"] is not None
        and summary["baseline_perplexity_sem"] <= 0.15
        and benchmark_parity
    )
    ledger = {
        "study": 9,
        "environment": environment(),
        "runs": records,
        "run_validations": validations,
        "summary": summary,
        "benchmark_runs": benchmark_rows,
        "benchmark_summary": benchmark_summary,
        "benchmark_parity_passed": benchmark_parity,
        "pretraining_passed": pretraining_passed,
        "pretraining_passed": pretraining_passed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "pretraining_ledger.json", ledger)
    print(json.dumps({
        "pretraining_passed": pretraining_passed,
        "pretraining_passed": pretraining_passed,
        "summary": summary,
    }, indent=2))
    return 0 if pretraining_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
