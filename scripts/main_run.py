#!/usr/bin/env python3
"""Budget sweep and main-run job lists for the 125M comparison (16 chips).

Stages (each prints a job list for scripts/run_slice.py):

* ``sweep``: every candidate in ``configs/main_sweep.json`` for each
  architecture in ``--archs`` (standard, algebraic, hybrid), seeds 42/43/44,
  600M tokens from ``fineweb_sweep_600M.npy``.
* ``select``: reads ``runs/<group>/*/result.json`` and writes
  ``configs/main_selected.json`` with the lowest mean held-out NLL candidate
  per architecture (all three seeds required and finite).
* ``main``: the main run with the selected hyperparameters: 125M models,
  2.5B tokens, seeds 42/43/44, checkpoints every 200 steps (resumable with
  --resume), validation every 250 steps. ``--archs standard,algebraic``
  restricts it to the two original arms.

All arms use AdamW with a cosine schedule. Batch 512 x 2048 tokens
(1,048,576 per step): 573 steps for 600M tokens and 2,385 for 2.5B, as in the
first study.

  python scripts/main_run.py sweep > sweep.json
  python scripts/run_slice.py sweep.json --group main_sweep
  python scripts/main_run.py select --group main_sweep,main_sweep_hybrid
  python scripts/main_run.py main > main.json
  python scripts/run_slice.py main.json --group main   # add --resume via args if interrupted
"""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "data" / "fw25"
SEEDS = (42, 43, 44)
ARCHS = {"standard": "standard", "algebraic": "algebraic", "hybrid": "hybrid"}


def common(tokens, train_file):
    return ["--size", "125m", "--seq-len", "2048", "--batch", "512", "--micro", "8",
            "--tokens", str(tokens), "--b2", "0.95", "--schedule", "cosine",
            "--train", str(DATA / train_file), "--valid", str(DATA / "fineweb_valid.npy"),
            "--eval-tokens", "8e6", "--eval-batch", "8", "--log-every", "25",
            "--set", "attn_impl=flash"]


def hp_args(hp):
    return ["--lr", str(hp["lr"]), "--wd", str(hp["wd"]), "--warmup", str(hp["warmup"]),
            "--min-lr-ratio", str(hp["min_lr_ratio"])]


def sweep(archs):
    cands = json.loads((ROOT / "configs/main_sweep.json").read_text())
    jobs = []
    for seed in SEEDS:
        for idx in range(len(cands["standard"])):
            for arch in archs:
                preset = ARCHS[arch]
                cand = cands[arch][idx]
                jobs.append({"name": f"{arch}_{cand['name']}_s{seed}",
                             "args": ["--preset", preset, *common(6e8, "fineweb_sweep_600M.npy"),
                                      *hp_args(cand), "--seed", str(seed), "--eval-every", "100"]})
    return jobs


def select(group, archs):
    cands = json.loads((ROOT / "configs/main_sweep.json").read_text())
    roots = [ROOT / "runs" / g for g in group.split(",")]
    chosen, table = {}, {}
    for arch in archs:
        rows = []
        for cand in cands[arch]:
            vals = []
            for seed in SEEDS:
                found = [r / f"{arch}_{cand['name']}_s{seed}" / "result.json" for r in roots]
                found = [f for f in found if f.exists()]
                res = found[0] if found else None
                if res is not None:
                    v = json.loads(res.read_text())["final_eval_nll"]
                    if math.isfinite(v):
                        vals.append(v)
            if len(vals) == len(SEEDS):
                mean = sum(vals) / len(vals)
                sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                rows.append((mean, cand, vals, sd))
                table[f"{arch}/{cand['name']}"] = {"nll": vals, "mean": mean, "sd": sd}
        if not rows:
            raise SystemExit(f"no complete candidate for {arch}")
        mean, cand, vals, sd = min(rows, key=lambda r: r[0])
        chosen[arch] = {**cand, "sweep_mean_nll": mean, "sweep_nll": vals}
    path = ROOT / "configs/main_selected.json"
    previous = json.loads(path.read_text()) if path.exists() else {"selected": {}, "table": {}}
    previous["selected"].update(chosen)
    previous["table"].update(table)
    path.write_text(json.dumps(previous, indent=1))
    return previous


def main_jobs(archs):
    sel = json.loads((ROOT / "configs/main_selected.json").read_text())["selected"]
    jobs = []
    for seed in SEEDS:
        for arch in archs:
            preset = ARCHS[arch]
            jobs.append({"name": f"{arch}_s{seed}",
                         "args": ["--preset", preset, *common(2.5e9, "fineweb_train_2_5B.npy"),
                                  *hp_args(sel[arch]), "--seed", str(seed), "--eval-every", "250",
                                  "--ckpt-every", "200", "--resume", "--save-params"]})
    return jobs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("sweep", "select", "main"))
    p.add_argument("--group", default="main_sweep",
                   help="run group(s) holding the sweep results, comma separated")
    p.add_argument("--archs", default="standard,algebraic,hybrid",
                   help="architectures to include, comma separated")
    a = p.parse_args()
    archs = [x for x in a.archs.split(",") if x]
    if a.stage == "sweep":
        print(json.dumps(sweep(archs), indent=1))
    elif a.stage == "select":
        print(json.dumps(select(a.group, archs), indent=1))
    else:
        print(json.dumps(main_jobs(archs), indent=1))
