"""Tests for Loss study empirical records and hardware evidence validation."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _check_hardware(path, expected_hashes):
    if not path.exists():
        return {"passed": False, "reason": "Missing TPU evidence"}
    h = json.loads(path.read_text())
    matched = h.get("environment", {}).get("source_sha256") == expected_hashes
    inventory = (
        h.get("device_count") == 16
        and h.get("process_count") == 4
        and len(h.get("devices", [])) == 16
        and all("TPU v4" in d.get("kind", "") for d in h.get("devices", []))
    )
    inventory = (
        inventory
        and len(h.get("parity", {}).get("rows", [])) == 16
        and all(v["passed"] for v in h["parity"]["rows"])
    )
    inventory = (
        inventory
        and len(h.get("boundary_stability", {}).get("rows", [])) == 9
        and all(v["passed"] for v in h["boundary_stability"]["rows"])
    )
    inventory = (
        inventory
        and len(h.get("fisher_equivalence", {}).get("rows", [])) == 1
        and all(v["passed"] for v in h["fisher_equivalence"]["rows"])
    )
    inventory = (
        inventory
        and len(h.get("benchmarks", {}).get("rows", [])) == 8
        and all(
            v["repetitions"] >= 100
            and all(g["ratio"] >= 0.90 for g in v["gates"].values())
            for v in h["benchmarks"]["rows"]
        )
    )
    inventory = inventory and all(
        h.get(k, {}).get("passed")
        for k in (
            "parity",
            "boundary_stability",
            "fisher_equivalence",
            "benchmarks",
        )
    )
    return {
        "passed": bool(matched and inventory and h.get("passed")),
        "source_matches": matched,
        "inventory_matches": inventory,
        "path": str(path),
    }


def test_missing_tpu_hardware_cannot_pass(tmp_path):
    assert not _check_hardware(tmp_path / "missing.json", {})["passed"]


def test_stale_tpu_hardware_cannot_pass(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"passed": True, "environment": {"source_sha256": {"src/loss.py": "old"}}}))
    assert not _check_hardware(path, {"src/loss.py": "new"})["passed"]


def test_incomplete_tpu_inventory_cannot_pass(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps({
            "passed": True,
            "environment": {"source_sha256": {}},
            "device_count": 8,  # Only 8 chips instead of 16
            "process_count": 2,
            "devices": [{"kind": "TPU v4"}] * 8,
        })
    )
    assert not _check_hardware(path, {})["passed"]


def test_committed_tpu_metrics_validates_cleanly():
    tpu_path = ROOT / "results/loss/tpu/metrics.json"
    if not tpu_path.exists():
        pytest.skip("TPU evidence not yet produced")
    h = json.loads(tpu_path.read_text())
    hashes = h["environment"]["source_sha256"]
    res = _check_hardware(tpu_path, hashes)
    assert res["passed"] is True, f"Committed TPU evidence failed validation: {res}"
    assert res["inventory_matches"] is True
    assert res["source_matches"] is True
