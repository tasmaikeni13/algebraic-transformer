#!/usr/bin/env python3
"""Summarize a group of runs: final held-out NLL, differences to a reference,
and validation curves.

  python analysis/summarize_runs.py runs/p1b1 --ref std_lr0.002 [--json out.json]
"""

import argparse
import json
import math
from pathlib import Path


def load(group: Path):
    rows = {}
    for d in sorted(group.iterdir()):
        res = d / "result.json"
        if not res.is_dir() and res.exists():
            r = json.loads(res.read_text())
            curve = []
            log = d / "log.jsonl"
            if log.exists():
                for line in log.read_text().splitlines():
                    e = json.loads(line)
                    if "eval_nll" in e:
                        curve.append((e["step"], e["eval_nll"]))
            rows[d.name] = {"nll": r["final_eval_nll"], "ppl": r["final_eval_ppl"],
                            "tok_s": r.get("mean_tok_per_s"), "minutes": r.get("train_seconds", 0) / 60,
                            "params": r.get("n_params"), "curve": curve,
                            "by_pos": r.get("eval_nll_by_log2_position"),
                            "train_loss": r.get("final_train_loss")}
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("groups", nargs="+", type=Path)
    p.add_argument("--ref", default=None)
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    rows = {}
    for g in args.groups:
        rows.update(load(g))
    ref = rows.get(args.ref, {}).get("nll") if args.ref else None
    print(f"{'run':28s} {'nll':>8s} {'ppl':>8s} {'d_ref':>8s} {'tok/s':>8s} {'min':>6s}  curve(step:nll)")
    for name, r in sorted(rows.items(), key=lambda kv: kv[1]["nll"]):
        d = f"{r['nll'] - ref:+.4f}" if ref is not None else ""
        curve = " ".join(f"{s}:{v:.3f}" for s, v in r["curve"][-4:])
        print(f"{name:28s} {r['nll']:8.4f} {r['ppl']:8.2f} {d:>8s} {(r['tok_s'] or 0)/1e3:7.0f}K "
              f"{r['minutes']:6.1f}  {curve}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
