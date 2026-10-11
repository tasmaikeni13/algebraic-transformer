#!/usr/bin/env python3
"""Run a list of single-chip training jobs across the slice's 16 chips.

Each job is ``{"name": ..., "args": [...]}`` (arguments for
``python -m src.sota.train`` without ``--out``). Slots are (host, chip) pairs;
jobs start on free slots, write to ``~/sota-algebra/runs/<group>/<name>`` on
their host, and their output directory is copied back to this host when they
finish. Hosts are reached as ``w0..w3`` through ~/.ssh/config, and every host
needs ~/bin/chiprun (see scripts/setup_workers.sh).

  python scripts/run_queue.py jobs.json --group phase1
"""

import argparse
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jobs", type=Path)
    p.add_argument("--group", required=True)
    p.add_argument("--hosts", default="w0,w1,w2,w3")
    p.add_argument("--chips", default="0,1,2,3")
    p.add_argument("--skip-done", action="store_true", help="skip jobs with a result.json")
    p.add_argument("--slots", default="", help="explicit host:chip list, e.g. w0:1,w2:3")
    args = p.parse_args()

    jobs = json.loads(args.jobs.read_text())
    local_root = ROOT / "runs" / args.group
    local_root.mkdir(parents=True, exist_ok=True)
    work = queue.Queue()
    for job in jobs:
        if args.skip_done and (local_root / job["name"] / "result.json").exists():
            continue
        work.put(job)
    total = work.qsize()
    lock = threading.Lock()
    done = []

    def worker(host, chip):
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            name = job["name"]
            remote = f"~/sota-algebra/runs/{args.group}/{name}"
            cmd = ("mkdir -p {r} && ~/bin/chiprun {c} -m src.sota.train {a} --out {r} "
                   "> {r}.log 2>&1").format(r=remote, c=chip, a=" ".join(job["args"]))
            t0 = time.time()
            rc = subprocess.call(["ssh", host, cmd])
            subprocess.call(["scp", "-q", "-r", f"{host}:{remote}", str(local_root)])
            subprocess.call(["scp", "-q", f"{host}:{remote}.log", str(local_root / f"{name}.log")])
            with lock:
                done.append(name)
                res = local_root / name / "result.json"
                status = "ok" if res.exists() else f"FAILED rc={rc}"
                summary = ""
                if res.exists():
                    r = json.loads(res.read_text())
                    summary = f"nll {r['final_eval_nll']:.4f}"
                print(f"[{len(done)}/{total}] {name} on {host}:{chip} {status} {summary} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)

    if args.slots:
        slots = [(s.split(":")[0], int(s.split(":")[1])) for s in args.slots.split(",")]
    else:
        slots = [(h, int(c)) for c in args.chips.split(",") for h in args.hosts.split(",")]
    threads = [threading.Thread(target=worker, args=slot) for slot in slots]
    for t in threads:
        t.start()
        time.sleep(2)
    for t in threads:
        t.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
