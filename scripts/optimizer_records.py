"""Optimizer study evidence fingerprints, including the inherited dependency closure."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.loss_records import source_hashes as loss_hashes, environment as loss_environment

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
    result = loss_hashes()
    names = [
        'src/optimizer.py',
        'tests/reference_optimizer.py',
        'tests/test_optimizer.py',
        'formal/AlgebraicTheory/Curvature.lean',
    ]
    names += [
        str(p.relative_to(ROOT))
        for pattern in ('optimizer_*.py', '*optimizer*.py', 'run_verify_optimizer.py')
        for p in (ROOT / 'scripts').glob(pattern)
    ]
    for name in sorted(set(names)):
        p = ROOT / name
        if p.exists():
            result[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def environment():
    env = loss_environment()
    env['source_sha256'] = source_hashes()
    return env
