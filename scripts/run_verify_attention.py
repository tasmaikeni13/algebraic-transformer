#!/usr/bin/env python3
"""Execute and retain the Attention study scientific gates; failures remain failures."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['JAX_ENABLE_X64']='1'
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from pathlib import Path
import sys
import subprocess
import shutil
import json
import re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.attention_records import environment,source_hashes,write_json
from scripts.attention_experiments import audit,scalar_checks,mc_study,jacobian_study,sink_sweep
from scripts.run_verify_primitives import hardware_evidence,reusable_cpu_evidence
from scripts.primitives_records import environment as primitives_environment,source_hashes as primitives_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'results/attention')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    r={'study':2,'gate_version':2,'environment':environment(),'command':sys.argv}
    inherited=json.loads((ROOT/'results/primitives/metrics.json').read_text())
    r['inherited_primitives']={'cpu_dependencies_unchanged':reusable_cpu_evidence(inherited,primitives_environment()),
       'hardware':hardware_evidence(ROOT/'results/primitives/tpu/metrics.json',primitives_hashes())['passed'],
       'pass_record_exists':(ROOT/'results/primitives/PASS.md').exists()}
    r['inherited_primitives']['passed']=all(r['inherited_primitives'].values())
    lake=shutil.which('lake') or str(Path.home()/'.elan/bin/lake')
    build=subprocess.run([lake,'build'],cwd=ROOT/'formal',capture_output=True,text=True)
    (out/'lean-build.log').write_text(build.stdout+build.stderr)
    violations=[str(f.relative_to(ROOT)) for f in (ROOT/'formal').rglob('*.lean') if '.lake' not in f.parts and re.search(r'\b(sorry|admit|axiom)\b',f.read_text())]
    r['formal']={'passed':build.returncode==0 and not re.search('warning',build.stdout+build.stderr,re.I) and not violations,'proof_scan_violations':violations}
    test=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,capture_output=True,text=True)
    (out/'pytest.log').write_text(test.stdout+test.stderr);r['unit_tests']={'passed':test.returncode==0}
    r['purity']=audit();r['scalar']=scalar_checks()
    r['monte_carlo'],raw=mc_study();np.savez_compressed(out/'monte-carlo.npz',**raw)
    r['jacobian'],raw=jacobian_study();np.savez_compressed(out/'jacobians.npz',**raw)
    r['sink_sweep']=sink_sweep()
    tpu=out/'tpu/metrics.json'
    r['hardware']={'passed':False,'reason':'Missing TPU evidence'}
    if tpu.exists():
        h=json.loads(tpu.read_text())
        matched=h['environment']['source_sha256']==r['environment']['source_sha256']
        inventory=h['device_count']==16 and h['process_count']==4 and len(h['devices'])==16 and all('TPU v4' in d['kind'] for d in h['devices'])
        inventory=inventory and len(h['parity']['rows'])==80 and all(v['passed'] for v in h['parity']['rows'])
        inventory=inventory and len(h['benchmarks']['rows'])==12 and all(v['repetitions']>=100 and all(g['ratio']>=.90 for g in v['gates'].values()) for v in h['benchmarks']['rows'])
        inventory=inventory and h['monte_carlo']['trials']>=100000 and h['jacobian']['trials']>=10000
        inventory=inventory and all(h[k]['passed'] for k in ('monte_carlo','jacobian'))
        r['hardware']={'passed':bool(matched and inventory and h['passed']),'source_matches':matched,'inventory_matches':inventory,'path':str(tpu)}
    gates=('inherited_primitives','formal','unit_tests','purity','scalar','monte_carlo','jacobian','hardware')
    r['passed']=all(r[k]['passed'] for k in gates)
    if source_hashes()!=r['environment']['source_sha256']:r['passed']=False;r['source_changed']=True
    r['status']='PASS' if r['passed'] else 'FAIL'
    write_json(out/'metrics.json',r)
    print(r['status'],flush=True)
    return 0 if r['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
