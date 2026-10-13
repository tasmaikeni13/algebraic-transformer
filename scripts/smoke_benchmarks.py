#!/usr/bin/env python3
"""Exercise all public and recall evaluator routes with tiny random models.

Uses one real public example per task and present/absent/short recall
controls from prepared development data. Scores are limited smoke evidence,
never publication benchmark results. On TPU, forward uses fused attention.
"""

from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmarks", type=Path, required=True)
    p.add_argument("--recall-benchmarks", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "results/verification/benchmark_smoke.json")
    a = p.parse_args()
    import jax
    from algebraic.benchmarking import PUBLIC_TASKS, Scorer, read_dataset, source_fingerprint, write_json
    from algebraic.config import build
    from algebraic.model import init_params
    from algebraic.recall_benchmarks import TASKS
    from scripts.evaluate_zero_shot import evaluate as public_evaluate
    from scripts.evaluate_recall import evaluate as recall_evaluate
    report = {"purpose": "limited evaluator smoke", "limited": True, "full_training_launched": False,
              "backend": jax.default_backend(), "source": source_fingerprint(), "runs": {}}
    with tempfile.TemporaryDirectory(prefix="algebraic-benchmark-smoke-") as directory:
        subset = Path(directory)
        meta = json.loads((a.recall_benchmarks / "metadata.json").read_text())
        meta["datasets"] = {}
        for task in TASKS:
            rows, _ = read_dataset(a.recall_benchmarks, task)
            examples = [next(row for row in rows if row["condition"] == condition)
                        for condition in ("present", "absent", "short")]
            encoded = "".join(json.dumps(row) + "\n" for row in examples)
            (subset / f"{task}.jsonl").write_text(encoded)
            meta["datasets"][task] = {"rows": len(examples), "limited": True,
                                       "sha256": hashlib.sha256(encoded.encode()).hexdigest()}
        write_json(subset / "metadata.json", meta)
        for arch in ("standard", "algebraic"):
            cfg = build(arch, "350m", d_model=128, n_layers=1, n_heads=2, d_ff=256,
                        seq_len=2048, dtype="bfloat16", attn_impl="flash" if jax.default_backend()=="tpu" else "dense")
            scorer = Scorer(cfg, init_params(cfg, jax.random.PRNGKey(42)), batch=1)
            public, public_predictions = public_evaluate(scorer, a.benchmarks, PUBLIC_TASKS, limit=1)
            recall, recall_predictions = recall_evaluate(scorer, subset, TASKS)
            if (set(public["results"]) != set(PUBLIC_TASKS) or len(public_predictions) != len(PUBLIC_TASKS)
                    or set(recall["results"]) != set(TASKS) or len(recall_predictions) != 3*len(TASKS)
                    or any(row["evidence_truncated"] for row in recall_predictions)
                    or {row["condition"] for row in recall_predictions} != {"present", "absent", "short"}):
                raise AssertionError("benchmark routing or bounded-context smoke failed")
            report["runs"][arch] = {"passed": True, "public_tasks": list(public["results"]),
                                     "recall_tasks": list(recall["results"]), "public_examples": len(public_predictions),
                                     "recall_examples": len(recall_predictions), "controls": ["present", "absent", "short"],
                                     "evidence_truncated": False, "attn_impl": cfg.attn_impl}
            del scorer
            jax.clear_caches()
    report["passed"] = True
    write_json(a.out, report)
    print(f"Benchmark smoke passed: {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
