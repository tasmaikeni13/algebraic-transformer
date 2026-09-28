"""Pilot study evidence fingerprints, including the inherited dependency closure."""
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts.tpu_kernels_records import source_hashes as tpu_kernels_hashes, environment as tpu_kernels_environment

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PILOT_STEPS = 100_000


def write_json(path, value):
    """Preserve NumPy scalar values as JSON scalars; reject NaN and infinity."""
    def scalar(item):
        if isinstance(item, (np.generic, np.ndarray)):
            if isinstance(item, np.ndarray) and item.ndim == 0:
                return item.item()
            elif isinstance(item, np.ndarray):
                return item.tolist()
            return item.item()
        raise TypeError(f"Unsupported evidence value: {type(item).__name__}")

    encoded = json.dumps(value, indent=2, allow_nan=False, default=scalar) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def source_hashes():
    result = tpu_kernels_hashes()
    names = [
        'src/model.py',
        'src/baseline.py',
        'src/mesh.py',
        'src/dataset.py',
        'src/kernels/pallas_afa.py',
        'src/kernels/pallas_oace.py',
        'src/kernels/pallas_flash_attention.py',
        'src/kernels/fused_cross_entropy.py',
        'tests/test_model.py',
        'tests/test_pallas_afa.py',
        'tests/test_pallas_oace.py',
        'tests/test_standard_kernels.py',
        'formal/AlgebraicTheory/Composition.lean',
    ]
    names += [
        str(p.relative_to(ROOT))
        for pattern in ('pilot_*.py', '*pilot*.py', 'run_pilot_20m.py')
        for p in (ROOT / 'scripts').glob(pattern)
    ]
    for name in sorted(set(names)):
        p = ROOT / name
        if p.exists():
            result[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def environment():
    env = tpu_kernels_environment()
    env['source_sha256'] = source_hashes()
    return env


def hardware_evidence(path, expected_hashes):
    """Validate that Pilot study evidence exercised the real pilot pretraining on TPU."""
    path = Path(path)
    if not path.exists():
        return {"passed": False, "reason": "Missing TPU evidence"}
    record = json.loads(path.read_text())
    matched = record.get("environment", {}).get("source_sha256") == expected_hashes

    pilot = record.get("pilot_pretraining", {})
    ppl_ratio = pilot.get("perplexity_ratio", 999.0)
    nan_count = pilot.get("nan_or_inf_count", 999)
    spike_count = pilot.get("loss_spike_count", 999)
    peak_grad_norm = pilot.get("peak_gradient_norm", 999.0)
    throughput_ratio = pilot.get("throughput_ratio", 0.0)
    total_steps = record.get("total_steps", 0)
    algebraic = pilot.get("algebraic_results", {})
    baseline = pilot.get("baseline_results", {})
    alg_steps = algebraic.get("total_steps", 0)
    base_steps = baseline.get("total_steps", 0)
    device_count = record.get("device_count", 0)
    devices = record.get("devices", [])
    gates = pilot.get("gates", {})
    embedded_gates_pass = bool(gates) and all(
        gate.get("passed", False) for gate in gates.values()
    )
    stable_arms = all(
        result.get("nan_or_inf_count") == 0
        and result.get("loss_spike_count") == 0
        and np.isfinite(result.get("valid_perplexity", np.nan))
        and np.isfinite(result.get("final_loss", np.nan))
        for result in (algebraic, baseline)
    )

    passed = (
        matched
        and record.get("passed") is True
        and total_steps >= REQUIRED_PILOT_STEPS
        and alg_steps >= REQUIRED_PILOT_STEPS
        and base_steps >= REQUIRED_PILOT_STEPS
        and device_count == 16
        and record.get("process_count") == 4
        and len(devices) == 16
        and all("TPU v4" in item.get("kind", "") for item in devices)
        and record.get("batch_size_sequences") == 64
        and record.get("context_length_tokens") == 512
        and record.get("global_batch_size_tokens") == 64 * 512
        and ppl_ratio <= 1.08
        and nan_count == 0
        and spike_count == 0
        and peak_grad_norm <= 5.0
        and throughput_ratio >= 0.90
        and record.get("ast_audit", {}).get("passed", False)
        and embedded_gates_pass
        and stable_arms
    )

    return {
        "passed": passed,
        "matched_hashes": matched,
        "perplexity_ratio": ppl_ratio,
        "nan_count": nan_count,
        "spike_count": spike_count,
        "peak_gradient_norm": peak_grad_norm,
        "throughput_ratio": throughput_ratio,
        "total_steps": total_steps,
        "algebraic_steps": alg_steps,
        "baseline_steps": base_steps,
        "required_steps": REQUIRED_PILOT_STEPS,
        "device_count": device_count,
        "process_count": record.get("process_count", 0),
        "embedded_gates_pass": embedded_gates_pass,
        "stable_arms": stable_arms,
    }
