#!/usr/bin/env python3
"""Execute and retain the Loss study scientific gates; failures remain failures."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from pathlib import Path
import sys
import subprocess
import shutil
import json
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scripts.loss_records import environment, source_hashes, write_json
from scripts.loss_experiments import (
    audit,
    monte_carlo_label_noise_study,
    simplex_boundary_stability_study,
    fisher_information_ratio_study,
    strict_propriety_and_monotonicity_study,
    oace_vs_cross_entropy_benchmark,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/loss')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    r = {
        'study': 4,
        'gate_version': 1,
        'environment': environment(),
        'command': sys.argv,
    }

    # Require the primitive, attention, and position measurements.
    p1_metrics_path = ROOT / 'results/primitives/metrics.json'
    p2_metrics_path = ROOT / 'results/attention/metrics.json'
    p3_metrics_path = ROOT / 'results/positions/metrics.json'
    p1_pass_path = ROOT / 'results/primitives/PASS.md'
    p2_pass_path = ROOT / 'results/attention/PASS.md'
    p3_pass_path = ROOT / 'results/positions/PASS.md'

    r['dependencies'] = {
        'primitives_metrics_exist': p1_metrics_path.exists(),
        'primitives_pass_exists': p1_pass_path.exists(),
        'attention_metrics_exist': p2_metrics_path.exists(),
        'attention_pass_exists': p2_pass_path.exists(),
        'positions_metrics_exist': p3_metrics_path.exists(),
        'positions_pass_exists': p3_pass_path.exists(),
    }
    r['dependencies']['passed'] = all(r['dependencies'].values())

    # Formal Lean verification
    lake = shutil.which('lake') or str(Path.home() / '.elan/bin/lake')
    build = subprocess.run([lake, 'build'], cwd=ROOT / 'formal', capture_output=True, text=True)
    (out / 'lean-build.log').write_text(build.stdout + build.stderr)
    violations = [
        str(f.relative_to(ROOT))
        for f in (ROOT / 'formal').rglob('*.lean')
        if '.lake' not in f.parts and re.search(r'\b(sorry|admit|axiom)\b', f.read_text())
    ]
    r['formal'] = {
        'passed': build.returncode == 0 and not re.search('warning', build.stdout + build.stderr, re.I) and not violations,
        'proof_scan_violations': violations,
    }

    # Full unit test suite
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, capture_output=True, text=True)
    (out / 'pytest.log').write_text(test.stdout + test.stderr)
    r['unit_tests'] = {'passed': test.returncode == 0}

    # Zero-transcendental AST and token audit
    r['purity'] = audit()

    # Empirical scientific studies
    r['monte_carlo_label_noise'] = monte_carlo_label_noise_study()
    r['simplex_boundary_stability'] = simplex_boundary_stability_study()
    r['fisher_information_ratio'] = fisher_information_ratio_study()
    r['strict_propriety_and_monotonicity'] = strict_propriety_and_monotonicity_study()
    r['benchmark'] = oace_vs_cross_entropy_benchmark()

    # Hardware evidence check
    tpu = out / 'tpu/metrics.json'
    r['hardware'] = {'passed': False, 'reason': 'Missing TPU evidence'}
    if tpu.exists():
        h = json.loads(tpu.read_text())
        matched = h.get('environment', {}).get('source_sha256') == r['environment']['source_sha256']
        inventory = (
            h.get('device_count') == 16
            and h.get('process_count') == 4
            and len(h.get('devices', [])) == 16
            and all('TPU v4' in d.get('kind', '') for d in h.get('devices', []))
        )
        inventory = (
            inventory
            and len(h.get('parity', {}).get('rows', [])) == 16
            and all(v['passed'] for v in h['parity']['rows'])
        )
        inventory = (
            inventory
            and len(h.get('boundary_stability', {}).get('rows', [])) == 9
            and all(v['passed'] for v in h['boundary_stability']['rows'])
        )
        inventory = (
            inventory
            and len(h.get('fisher_equivalence', {}).get('rows', [])) == 1
            and all(v['passed'] for v in h['fisher_equivalence']['rows'])
        )
        inventory = (
            inventory
            and len(h.get('benchmarks', {}).get('rows', [])) == 8
            and all(
                v['repetitions'] >= 100
                and all(g['ratio'] >= 0.90 for g in v['gates'].values())
                for v in h['benchmarks']['rows']
            )
        )
        inventory = inventory and all(
            h.get(k, {}).get('passed')
            for k in (
                'parity',
                'boundary_stability',
                'fisher_equivalence',
                'benchmarks',
            )
        )
        r['hardware'] = {
            'passed': bool(matched and inventory and h.get('passed')),
            'source_matches': matched,
            'inventory_matches': inventory,
            'path': str(tpu),
        }

    gates = (
        'dependencies',
        'formal',
        'unit_tests',
        'purity',
        'monte_carlo_label_noise',
        'simplex_boundary_stability',
        'fisher_information_ratio',
        'strict_propriety_and_monotonicity',
        'benchmark',
        'hardware',
    )
    r['passed'] = all(r[k]['passed'] for k in gates)
    if source_hashes() != r['environment']['source_sha256']:
        r['passed'] = False
        r['source_changed'] = True
    r['status'] = 'PASS' if r['passed'] else 'FAIL'

    write_json(out / 'metrics.json', r)
    print(r['status'], flush=True)
    return 0 if r['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
