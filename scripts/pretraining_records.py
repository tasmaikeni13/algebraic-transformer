"""Pretraining study source fingerprints and hardware evidence validation."""

import hashlib
import json
from pathlib import Path

from scripts.tuning_records import environment as tuning_environment
from scripts.tuning_records import source_hashes as tuning_hashes

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    result = tuning_hashes()
    names = [
        "scripts/pretraining_experiments.py",
        "scripts/pretraining_records.py",
        "scripts/run_pretrain_125m.py",
        "scripts/aggregate_pretraining.py",
        "scripts/evaluate_benchmarks.py",
        "scripts/prepare_eval_benchmarks.py",
        "scripts/smoke_pretraining.py",
        "scripts/launch_pretraining_tpu.py",
        "scripts/run_verify_pretraining.py",
        "tests/test_pretraining_contracts.py",
    ]
    for name in names:
        path = ROOT / name
        if path.exists():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def environment():
    result = tuning_environment()
    result["source_sha256"] = source_hashes()
    return result


def run_evidence(path: Path, expected_hashes):
    path = Path(path)
    if not path.exists():
        return {"passed": False, "reason": f"missing run evidence: {path}"}
    record = json.loads(path.read_text())
    run = record.get("run", {})
    hardware = record.get("hardware", {})
    matched = record.get("environment", {}).get("source_sha256") == expected_hashes
    normalization_passed = (
        run.get("architecture") != "algebraic"
        or (
            run.get("normalization_second_moment_min", 0.0) >= 0.8
            and run.get("normalization_second_moment_max", 999.0) <= 1.3
        )
    )
    passed = (
        matched
        and record.get("passed") is True
        and hardware.get("platform") == "tpu"
        and hardware.get("device_count") == 16
        and hardware.get("process_count") == 4
        and run.get("total_tokens", 0) >= 2_500_000_000
        and run.get("nan_or_inf_count") == 0
        and run.get("loss_spike_count") == 0
        and run.get("peak_gradient_norm", 999.0) <= 5.0
        and normalization_passed
    )
    return {
        "passed": passed,
        "source_matches": matched,
        "architecture": run.get("architecture"),
        "seed": run.get("seed"),
        "tokens": run.get("total_tokens", 0),
        "platform": hardware.get("platform"),
        "device_count": hardware.get("device_count"),
        "process_count": hardware.get("process_count"),
        "normalization_second_moment_passed": normalization_passed,
    }
