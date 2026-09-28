"""Reproducible source fingerprints and artifact serialization."""

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def source_hashes():
    # Hash the source files that determine primitive behavior and its checks.
    names = ["src/__init__.py", "src/primitives.py", "scripts/__init__.py",
             "scripts/audit_primitives.py", "scripts/launch_primitives_tpu.py",
             "scripts/primitives_experiments.py", "scripts/primitives_records.py",
             "scripts/run_primitives_tpu.py", "scripts/run_verify_primitives.py",
             "tests/__init__.py", "tests/conftest.py", "tests/reference_primitives.py",
             "tests/test_primitives.py", "tests/test_primitives_records.py", "pytest.ini",
             "formal/AlgebraicTheory/Gate.lean", "formal/AlgebraicTheory/Variance.lean",
             "requirements.txt", "requirements-tpu.txt", "formal/lean-toolchain",
             "formal/lake-manifest.json", "formal/lakefile.toml",]
    paths = [ROOT/name for name in names]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def environment():
    import jax, jaxlib, numpy, scipy
    def git(*args):
        return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
    return {"python":sys.version,"platform":platform.platform(),"hostname":platform.node(),
            "jax":jax.__version__,"jaxlib":jaxlib.__version__,"numpy":numpy.__version__,"scipy":scipy.__version__,
            "git_commit":git("rev-parse","HEAD"),"git_dirty":bool(git("status","--porcelain")),
            "source_sha256":source_hashes()}
