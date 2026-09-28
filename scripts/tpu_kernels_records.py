"""TPU kernel study evidence fingerprints, including the inherited dependency closure."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.optimizer_records import source_hashes as optimizer_hashes, environment as optimizer_environment

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    """Preserve NumPy scalar values as JSON scalars; reject NaN and infinity."""
    def scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported evidence value: {type(item).__name__}")
    encoded = json.dumps(value, indent=2, allow_nan=False, default=scalar) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def source_hashes():
    result = optimizer_hashes()
    names = [
        'src/kernels/__init__.py',
        'src/kernels/pallas_afa.py',
        'tests/reference_attention.py',
        'tests/test_pallas_afa.py',
        'formal/AlgebraicTheory/Kernel.lean',
        'formal/AlgebraicTheory/Gate.lean',
    ]
    names += [
        str(p.relative_to(ROOT))
        for pattern in ('tpu_kernels_*.py', '*tpu_kernels*.py', 'run_verify_pallas.py', 'audit_xla_hlo.py')
        for p in (ROOT / 'scripts').glob(pattern)
    ]
    for name in sorted(set(names)):
        p = ROOT / name
        if p.exists():
            result[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def environment():
    env = optimizer_environment()
    env['source_sha256'] = source_hashes()
    return env


def hardware_evidence(path, expected_hashes):
    """Validate that TPU kernel study evidence exercised the real Pallas implementation."""
    path = Path(path)
    if not path.exists():
        return {"passed": False, "reason": "Missing TPU evidence"}
    record = json.loads(path.read_text())
    matched = record.get("environment", {}).get("source_sha256") == expected_hashes
    inventory = (
        record.get("device_count") == 16
        and record.get("process_count") == 4
        and len(record.get("devices", [])) == 16
        and all("TPU v4" in d.get("kind", "") for d in record.get("devices", []))
    )

    parity_rows = record.get("parity", {}).get("rows", [])
    inventory = inventory and len(parity_rows) == 8 and all(
        row.get("passed") and row.get("implementation") == "pallas_afa_forward"
        for row in parity_rows
    )

    benchmark_rows = record.get("benchmarks", {}).get("rows", [])
    inventory = inventory and len(benchmark_rows) >= 2 and all(
        row.get("passed")
        and row.get("repetitions", 0) >= 50
        and row.get("implementation") == "pallas_afa_forward"
        and row.get("baseline") == "jax.experimental.pallas.ops.tpu.flash_attention"
        and row.get("throughput_ratio", 0.0) >= row.get("gate_bound", 1.0)
        for row in benchmark_rows
    )

    streaming = record.get("streaming_contract", {})
    inventory = (
        inventory
        and streaming.get("passed", False)
        and streaming.get("metric_kind") == "bounded_tile_storage"
        and streaming.get("physical_hbm_utilization_claimed") is False
        and streaming.get("conservative_tile_working_set_bytes", 1)
        <= streaming.get("vmem_budget_bytes", 0)
    )

    ring = record.get("ring_attention", {})
    hlo = record.get("hlo_audit", {})
    inventory = (
        inventory
        and ring.get("passed", False)
        and ring.get("rel_error", 1.0) <= 1.0e-6
        and hlo.get("passed", False)
        and hlo.get("transcendental_opcodes_count", 1) == 0
        and hlo.get("audited_implementation") == "pallas_afa_forward"
        and bool(hlo.get("pallas_lowering_markers"))
    )
    inventory = inventory and all(
        record.get(key, {}).get("passed")
        for key in ("parity", "benchmarks", "streaming_contract", "ring_attention", "hlo_audit")
    )
    return {
        "passed": bool(matched and inventory and record.get("passed")),
        "source_matches": matched,
        "inventory_matches": inventory,
        "path": str(path),
    }
