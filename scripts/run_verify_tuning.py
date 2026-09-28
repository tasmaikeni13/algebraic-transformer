#!/usr/bin/env python3
"""Verify Tuning study evidence and create PASS.md only after every gate passes."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pilot_records import hardware_evidence as pilot_hardware_evidence
from scripts.pilot_records import source_hashes as pilot_source_hashes
from scripts.tuning_records import environment, hardware_evidence, source_hashes, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", type=Path, default=ROOT / "results/tuning/metrics.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/tuning"
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    expected_hashes = source_hashes()
    hardware = hardware_evidence(args.metrics, expected_hashes)
    pilot_metrics = ROOT / "results/pilot/metrics.json"
    pilot_aggregate = (
        json.loads(pilot_metrics.read_text()) if pilot_metrics.exists() else {}
    )
    inherited_pilot = {
        "metrics_exist": pilot_metrics.exists(),
        "pass_record_exists": (ROOT / "results/pilot/PASS.md").exists(),
        "aggregate_status_pass": (
            pilot_aggregate.get("passed") is True
            and pilot_aggregate.get("status") == "PASS"
            and pilot_aggregate.get("environment", {}).get("source_sha256")
            == pilot_source_hashes()
        ),
        "hardware_passed": pilot_hardware_evidence(
            ROOT / "results/pilot/tpu/metrics.json", pilot_source_hashes()
        ).get("passed", False),
    }
    inherited_pilot["passed"] = all(inherited_pilot.values())

    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    (output / "pytest.log").write_text(tests.stdout + tests.stderr)

    lake = shutil.which("lake") or str(Path.home() / ".elan/bin/lake")
    formal = subprocess.run(
        [lake, "build"], cwd=ROOT / "formal", capture_output=True, text=True
    )
    (output / "lean-build.log").write_text(formal.stdout + formal.stderr)
    proof_violations = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "formal").rglob("*.lean")
        if ".lake" not in path.parts
        and re.search(r"\b(sorry|admit|axiom)\b", path.read_text())
    ]
    formal_passed = (
        formal.returncode == 0
        and not re.search("warning", formal.stdout + formal.stderr, re.I)
        and not proof_violations
    )

    verification = {
        "study": 8,
        "gate_version": 2,
        "environment": environment(),
        "hardware": hardware,
        "inherited_pilot": inherited_pilot,
        "tests_passed": tests.returncode == 0,
        "formal_passed": formal_passed,
        "proof_scan_violations": proof_violations,
    }
    verification["passed"] = bool(
        hardware.get("passed")
        and inherited_pilot["passed"]
        and verification["tests_passed"]
        and formal_passed
        and source_hashes() == verification["environment"]["source_sha256"]
    )
    verification["status"] = "PASS" if verification["passed"] else "FAIL"
    write_json(output / "verification.json", verification)

    pass_path = output / "PASS.md"
    if verification["passed"]:
        record = json.loads(args.metrics.read_text())
        summary = record["sweep_summary"]
        pass_path.write_text(
            "# Tuning study PASS — 125M Hyperparameter Sweep\n\n"
            f"Validated all {summary['completed_runs']} preregistered candidate/seed "
            "runs on four hosts and 16 TPU v4 chips.\n\n"
            f"- Minimum tokens per run: {summary['minimum_actual_tokens_per_run']:,}.\n"
            f"- Algebraic/baseline mean perplexity ratio: "
            f"{summary['mean_perplexity_ratio']:.6f}.\n"
            f"- Selected candidates: "
            f"{summary['selected_algebraic_candidate']} (algebraic), "
            f"{summary['selected_baseline_candidate']} (baseline).\n"
            "- Source hashes, winner artifacts, inherited Pilot study, tests, Lean, "
            "topology, stability, and normalization gates passed.\n\n"
            "Authoritative evidence: `metrics.json`, `verification.json`, "
            "`sweep_ledger.json`, the two `*_optimal.json` files, `pytest.log`, "
            "and `lean-build.log`.\n"
        )
    elif pass_path.exists():
        pass_path.unlink()
    return 0 if verification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
