#!/usr/bin/env python3
"""Run training jobs one after another on the whole TPU slice (all hosts).

Each job is ``{"name": ..., "args": [...]}`` (arguments for
``python -m src.sota.train`` without ``--out``/``--distributed``). The same
command starts on every host (w0..w3 via ~/.ssh/config); process 0 writes
``runs/<group>/<name>/`` on this host. Code is synced first with
scripts/setup_workers.sh.

  python scripts/run_slice.py jobs.json --group phase3
"""

import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jobs", type=Path)
    p.add_argument("--group", required=True)
    p.add_argument("--hosts", default="w0,w1,w2,w3")
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--no-sync", action="store_true")
    args = p.parse_args()
    hosts = args.hosts.split(",")
    if not args.no_sync:
        subprocess.check_call(["bash", str(ROOT / "scripts/setup_workers.sh")])
    root = ROOT / "runs" / args.group
    root.mkdir(parents=True, exist_ok=True)
    for job in json.loads(args.jobs.read_text()):
        name = job["name"]
        out = f"~/sota-algebra/runs/{args.group}/{name}"
        if args.skip_done and (root / name / "result.json").exists():
            continue
        cmd = (f"mkdir -p ~/sota-algebra/runs/{args.group} && cd ~/sota-algebra && "
               f"JAX_PLATFORMS=tpu ~/venv/bin/python -m src.sota.train --distributed "
               f"{' '.join(job['args'])} --out {out} > {out}.$(hostname).log 2>&1")
        t0 = time.time()
        procs = [subprocess.Popen(["ssh", h, cmd]) for h in hosts]
        codes = [pr.wait() for pr in procs]
        res = root / name / "result.json"
        summary = ""
        if res.exists():
            r = json.loads(res.read_text())
            summary = (f"nll {r['final_eval_nll']:.4f} ppl {r['final_eval_ppl']:.3f} "
                       f"{r['mean_tok_per_s']/1e6:.2f}M tok/s")
        print(f"{name}: codes {codes} {summary} ({(time.time()-t0)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
