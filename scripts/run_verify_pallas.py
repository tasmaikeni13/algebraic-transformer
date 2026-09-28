#!/usr/bin/env python3
"""Verification entrypoint for TPU kernel study Hardware-Fused Kernels & Algebraic FlashAttention."""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
import time
from pathlib import Path
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.tpu_kernels_experiments import run_all_experiments
from scripts.tpu_kernels_records import hardware_evidence, source_hashes, write_json
from scripts.optimizer_records import source_hashes as optimizer_source_hashes


def main():
    print("================================================================================")
    print("TPU KERNEL VERIFICATION: HARDWARE-FUSED ALGEBRAIC FLASHATTENTION (AFA)")
    print("================================================================================")
    t0 = time.time()

    metrics = run_all_experiments()
    local_passed = metrics["passed"]

    optimizer_path = ROOT / "results/optimizer/metrics.json"
    optimizer = json.loads(optimizer_path.read_text()) if optimizer_path.exists() else {}
    metrics["inherited_optimizer"] = {
        "metrics_exist": optimizer_path.exists(),
        "pass_record_exists": (ROOT / "results/optimizer/PASS.md").exists(),
        "status_pass": optimizer.get("status") == "PASS" and optimizer.get("passed") is True,
        "source_matches": optimizer.get("environment", {}).get("source_sha256") == optimizer_source_hashes(),
    }
    metrics["inherited_optimizer"]["passed"] = all(metrics["inherited_optimizer"].values())

    lake = shutil.which("lake") or str(Path.home() / ".elan/bin/lake")
    build = subprocess.run([lake, "build"], cwd=ROOT / "formal", capture_output=True, text=True)
    (ROOT / "results/tpu_kernels/lean-build.log").write_text(build.stdout + build.stderr)
    proof_violations = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "formal").rglob("*.lean")
        if ".lake" not in path.parts and re.search(r"\b(sorry|admit|axiom)\b", path.read_text())
    ]
    metrics["formal"] = {
        "passed": build.returncode == 0
        and not re.search("warning", build.stdout + build.stderr, re.I)
        and not proof_violations,
        "proof_scan_violations": proof_violations,
    }

    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    (ROOT / "results/tpu_kernels/pytest.log").write_text(tests.stdout + tests.stderr)
    metrics["unit_tests"] = {"passed": tests.returncode == 0}
    metrics["hardware"] = hardware_evidence(
        ROOT / "results/tpu_kernels/tpu/metrics.json", source_hashes()
    )
    metrics["local_cpu"] = {"passed": local_passed}
    metrics["passed"] = all(
        metrics[key]["passed"]
        for key in ("local_cpu", "inherited_optimizer", "formal", "unit_tests", "hardware")
    )
    if source_hashes() != metrics["environment"]["source_sha256"]:
        metrics["source_changed"] = True
        metrics["passed"] = False
    metrics["status"] = "PASS" if metrics["passed"] else "FAIL"
    metrics["elapsed_seconds"] = time.time() - t0
    output_path = ROOT / "results/tpu_kernels/metrics.json"
    write_json(output_path, metrics)

    print("\n================================================================================")
    print(f"TPU KERNEL AGGREGATE VERIFICATION COMPLETE: {metrics['status']}")
    print(f"Total elapsed: {metrics['elapsed_seconds']:.2f}s")
    print(f"Metrics written to: {output_path}")
    print("================================================================================")

    if not metrics["passed"]:
        print("ERROR: One or more TPU kernel study CPU, formal, inherited, or TPU gates failed.")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
