#!/usr/bin/env python3
"""Validate the 20M language-model run against code and hardware records."""

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

from scripts.pilot_records import hardware_evidence, source_hashes, write_json, environment
from scripts.pilot_experiments import audit_pilot_ast
from scripts.tpu_kernels_records import source_hashes as tpu_kernels_source_hashes


def main():
    print("================================================================================")
    print("PILOT VERIFICATION: FULL ARCHITECTURE ASSEMBLY & PILOT PRETRAINING")
    print("================================================================================")
    t0 = time.time()

    # Check the recorded TPU kernel measurements against the current source.
    tpu_kernels_path = ROOT / "results/tpu_kernels/metrics.json"
    tpu_kernels = json.loads(tpu_kernels_path.read_text()) if tpu_kernels_path.exists() else {}
    inherited_tpu_kernels = {
        "metrics_exist": tpu_kernels_path.exists(),
        "pass_record_exists": (ROOT / "results/tpu_kernels/PASS.md").exists(),
        "status_pass": tpu_kernels.get("status") == "PASS" and tpu_kernels.get("passed") is True,
        "source_matches": tpu_kernels.get("environment", {}).get("source_sha256") == tpu_kernels_source_hashes(),
    }
    inherited_tpu_kernels["passed"] = all(inherited_tpu_kernels.values())

    # Compile the formal library and scan project proofs for placeholders.
    lake = shutil.which("lake") or str(Path.home() / ".elan/bin/lake")
    build = subprocess.run([lake, "build"], cwd=ROOT / "formal", capture_output=True, text=True)
    res_dir = ROOT / "results/pilot"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "lean-build.log").write_text(build.stdout + build.stderr)

    proof_violations = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "formal").rglob("*.lean")
        if ".lake" not in path.parts and re.search(r"\b(sorry|admit|axiom)\b", path.read_text())
    ]
    formal_metrics = {
        "passed": build.returncode == 0
        and not re.search("warning", build.stdout + build.stderr, re.I)
        and not proof_violations,
        "proof_scan_violations": proof_violations,
    }

    # Audit the algebraic training modules.
    ast_audit = audit_pilot_ast()

    # Run CPU numerical and integration tests.
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    (res_dir / "pytest.log").write_text(tests.stdout + tests.stderr)
    unit_tests = {"passed": tests.returncode == 0}

    # Check the complete TPU training record.
    tpu_metrics_path = res_dir / "tpu/metrics.json"
    hardware = hardware_evidence(tpu_metrics_path, source_hashes())

    metrics = {
        "study": 7,
        "gate_version": 1,
        "environment": environment(),
        "inherited_tpu_kernels": inherited_tpu_kernels,
        "formal": formal_metrics,
        "ast_audit": ast_audit,
        "unit_tests": unit_tests,
        "hardware": hardware,
        "passed": (
            inherited_tpu_kernels["passed"]
            and formal_metrics["passed"]
            and ast_audit["passed"]
            and unit_tests["passed"]
            and hardware["passed"]
        ),
    }

    if source_hashes() != metrics["environment"]["source_sha256"]:
        metrics["source_changed"] = True
        metrics["passed"] = False

    metrics["status"] = "PASS" if metrics["passed"] else "FAIL"
    metrics["elapsed_seconds"] = time.time() - t0
    output_path = res_dir / "metrics.json"
    write_json(output_path, metrics)
    pass_path = res_dir / "PASS.md"
    if metrics["passed"]:
        tpu = json.loads(tpu_metrics_path.read_text())
        pilot = tpu["pilot_pretraining"]
        pass_path.write_text(
            "# Pilot study PASS — 20M WikiText-103 Pilot\n\n"
            f"Verified the full {tpu['total_steps']:,}-step run for each architecture "
            "on four hosts and 16 TPU v4 chips.\n\n"
            f"- Algebraic/baseline validation perplexity ratio: "
            f"{pilot['perplexity_ratio']:.6f} (gate: at most 1.08).\n"
            f"- Algebraic/baseline throughput ratio: "
            f"{pilot['throughput_ratio']:.6f} (gate: at least 0.90).\n"
            f"- Peak algebraic gradient norm: {pilot['peak_gradient_norm']:.6f}.\n"
            f"- Algebraic non-finite iterations / loss spikes: "
            f"{pilot['nan_or_inf_count']} / {pilot['loss_spike_count']}.\n"
            "- Inherited TPU kernel study, Lean, source audit, tests, source hashes, and "
            "hardware inventory all passed.\n\n"
            "Authoritative evidence: `metrics.json`, `tpu/metrics.json`, "
            "`tpu/losses.npz`, `pytest.log`, and `lean-build.log`.\n"
        )
    else:
        pass_path.write_text(
            "# Pilot study Evidence Invalidated\n\n"
            "**Current status: NOT PASSED.** See `metrics.json` for the failing "
            "source, inherited, formal, test, or hardware gate.\n"
        )

    print("\n================================================================================")
    print(f"PILOT AGGREGATE VERIFICATION COMPLETE: {metrics['status']}")
    print(f"Total elapsed: {metrics['elapsed_seconds']:.2f}s")
    print(f"Metrics written to: {output_path}")
    print("================================================================================")

    if not metrics["passed"]:
        print("Details:")
        for k in ("inherited_tpu_kernels", "formal", "ast_audit", "unit_tests", "hardware"):
            print(f"  {k:20s}: {metrics[k].get('passed', False)}")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
