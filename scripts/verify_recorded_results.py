"""Check packaged numerical records against their original Git blobs."""

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def strict_json(data: bytes):
    return json.loads(data, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def numbers(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from numbers(child)
    elif isinstance(value, list):
        for child in value:
            yield from numbers(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield repr(value)


def main() -> int:
    manifest = strict_json((ROOT / "results/record_origins.json").read_bytes())
    records = sorted(
        path for path in (ROOT / "results").rglob("*.json")
        if path.name != "record_origins.json"
    )
    if {path.relative_to(ROOT).as_posix() for path in records} != set(manifest):
        raise ValueError("record origin manifest does not cover the result files")
    by_commit = {}
    checked = 0
    for path in records:
        current = strict_json(path.read_bytes())
        origin = manifest[path.relative_to(ROOT).as_posix()]
        commit = origin["git_commit"]
        if commit not in by_commit:
            paths = subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", commit, "results"],
                cwd=ROOT, text=True,
            ).splitlines()
            originals = {}
            for original_path in paths:
                if not original_path.endswith(".json"):
                    continue
                raw = subprocess.check_output(
                    ["git", "show", f"{commit}:{original_path}"], cwd=ROOT,
                )
                originals[hashlib.sha256(raw).hexdigest()] = raw
            by_commit[commit] = originals
        digest = origin["sha256"]
        raw = by_commit[commit].get(digest)
        if raw is None:
            raise ValueError(f"original record not found for {path}")
        original = strict_json(raw)
        if list(numbers(original)) != list(numbers(current)):
            raise ValueError(f"numerical content differs from original for {path}")
        checked += 1
    print(f"Verified {checked} recorded JSON files against their Git originals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
