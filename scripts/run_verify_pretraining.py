#!/usr/bin/env python3
"""Verify aggregate Pretraining study evidence and create PASS.md only when every gate passes."""

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

from scripts.tuning_records import write_json
from scripts.pretraining_experiments import load_verified_tuning_configs
from scripts.pretraining_records import environment, source_hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/pretraining")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    expected_hashes = source_hashes()
    ledger = json.loads(args.ledger.read_text())
    ledger_valid = (
        ledger.get("environment", {}).get("source_sha256") == expected_hashes
        and ledger.get("pretraining_passed") is True
    )
    tuning_valid = True
    tuning_error = None
    try:
        load_verified_tuning_configs(ROOT / "results/tuning")
    except Exception as error:
        tuning_valid = False
        tuning_error = str(error)

    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
        capture_output=True, text=True,
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

    passed = ledger_valid and tuning_valid and tests.returncode == 0 and formal_passed
    metrics = {
        "study": 9,
        "gate_version": 1,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "environment": environment(),
        "ledger_valid": ledger_valid,
        "tuning_valid": tuning_valid,
        "tuning_error": tuning_error,
        "tests_passed": tests.returncode == 0,
        "formal_passed": formal_passed,
        "proof_scan_violations": proof_violations,
        "summary": ledger.get("summary", {}),
        "benchmark_summary": ledger.get("benchmark_summary", {}),
    }
    if source_hashes() != metrics["environment"]["source_sha256"]:
        metrics["passed"] = False
        metrics["status"] = "FAIL"
        metrics["source_changed"] = True
    write_json(output / "metrics.json", metrics)

    pass_path = output / "PASS.md"
    if metrics["passed"]:
        summary = metrics["summary"]
        pass_path.write_text(
            "# Pretraining study PASS — 125M FineWeb-Edu Pretraining\n\n"
            f"Validated all six 2.5B-token runs. Algebraic/baseline perplexity ratio: "
            f"{summary['perplexity_ratio']:.6f}.\n\n"
            "Zero-shot means:\n\n"
            + "\n".join(
                f"- **{name}**: algebraic {values['algebraic']:.4f}, "
                f"baseline {values['baseline']:.4f}"
                for name, values in metrics["benchmark_summary"].items()
            )
            + "\n\nEvidence: `metrics.json`, `pretraining_ledger.json`, "
            "`pytest.log`, and `lean-build.log`.\n"
        )
    elif pass_path.exists():
        pass_path.unlink()
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
