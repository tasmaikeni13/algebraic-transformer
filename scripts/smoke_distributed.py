#!/usr/bin/env python3
"""One synthetic optimizer step per full model on the actual TPU slice.

This is a verification command, not a corpus training run. Run from worker
zero after configuring SSH aliases. It syncs code and tiny deterministic
synthetic token arrays, uses the production batch and accumulation settings,
and writes a verification report without exporting large model weights.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.run_comparison import parameter_count, training_jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", choices=("125m", "350m", "1b"), default="350m")
    p.add_argument("--hosts", default="w0,w1,w2,w3")
    p.add_argument("--arches",nargs="+",choices=("standard","algebraic"),default=["standard","algebraic"])
    p.add_argument("--group",help="Use a fresh run directory for this verification.")
    p.add_argument("--out",type=Path,help="Preserve historical verification reports at a separate output path.")
    a = p.parse_args()
    import numpy as np
    from algebraic.benchmarking import source_fingerprint, write_json
    config = json.loads((ROOT / f"configs/comparison_{a.scale}.json").read_text())
    config["training"]["tokens"] = config["training"]["batch"] * config["training"]["seq_len"]
    config["training"].update(eval_tokens=65536, eval_every=0, ckpt_every=0, log_every=1)
    group = a.group or f"smoke_slice_{a.scale}"
    if not group or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in group):
        p.error("group must be a simple directory name")
    data_dir = ROOT / "runs" / group / "synthetic"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    vocab_size = config["training"].get("vocab_size", 32768)
    for name in (config["training"]["train_file"], config["training"]["valid_file"]):
        np.save(data_dir / name, rng.integers(0, vocab_size, 1_048_576+1, dtype=np.uint16))
    hosts = a.hosts.split(",")
    subprocess.run(["bash", str(ROOT / "scripts/setup_workers.sh"), *hosts], cwd=ROOT, check=True)
    for host in hosts:
        if subprocess.check_output(["ssh", host, "hostname"], text=True).strip() == subprocess.check_output(["hostname"], text=True).strip():
            continue
        subprocess.run(["ssh", host, f"mkdir -p {shlex.quote(str(data_dir))}"], check=True)
        subprocess.run(["rsync", "-a", str(data_dir)+"/", f"{host}:{data_dir}/"], check=True)
    jobs = training_jobs(config, data_dir, save_params=False)
    jobs = [job for job in jobs if job["name"].split("_s")[0] in a.arches]
    if not jobs:
        p.error("no requested architecture appears in this comparison")
    for job in jobs:
        job["args"] += ["--steps", "1"]
    job_path = ROOT / "runs" / group / "jobs.json"
    write_json(job_path, jobs)
    subprocess.run([sys.executable, str(ROOT / "scripts/run_slice.py"), str(job_path), "--group", group,
                    "--hosts", a.hosts, "--no-sync"], cwd=ROOT, check=True)
    report = {"purpose": "one-step distributed synthetic verification", "full_training_launched": False,
              "scale": a.scale, "completed_steps": 1, "processes": len(hosts),
              "source": source_fingerprint(), "hosts": hosts, "runs": {}}
    for job in jobs:
        result = json.loads((ROOT / "runs" / group / job["name"] / "result.json").read_text())
        arch = job["name"].split("_s")[0]
        if (result["completed_steps"] != 1 or result["processes"] != len(hosts)
                or result["n_params"] != parameter_count(config["training"], arch)
                or not np.isfinite(result["final_eval_nll"])):
            raise AssertionError("distributed smoke shape or execution mismatch")
        report["runs"][arch] = {"passed": True, "completed_steps": 1, "parameters": result["n_params"],
                                 "devices": result["devices"], "processes": result["processes"],
                                 "global_batch": result["args"]["batch"], "micro_batch": result["args"]["micro"],
                                 "accumulation": result["accum"], "attn_impl": result["config"]["attn_impl"],
                                 "head_impl": result["config"]["head_impl"]}
    report["passed"] = True
    path = a.out or ROOT / f"results/verification/distributed_{a.scale}.json"
    write_json(path, report)
    print(f"Distributed smoke passed: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
