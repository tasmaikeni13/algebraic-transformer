#!/usr/bin/env python3
"""Smoke-test full model presets with one synthetic AdamW step per arm.

Run on one isolated TPU chip:
  bash scripts/tpu_python.sh 0 scripts/smoke_comparison.py --out results/verification/smoke_350m.json

This uses the actual width, depth, 2,048-token context, 32,768 vocabulary,
rematerialization, Pallas attention, and fused head. No FineWeb corpus is
opened. Temporary model weights are removed after checkpoint loading checks.
--tiny keeps the kernel paths and vocabulary but reduces width/depth/context.
"""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", choices=("125m", "350m", "1b"), default="350m")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    a.out = a.out or ROOT / f"results/verification/smoke_{a.scale}.json"
    import jax
    import numpy as np
    from algebraic.benchmarking import load_model, source_fingerprint, write_json
    from algebraic.model import count_params
    from algebraic.train import main as train_main
    from scripts.audit_algebraic import audit
    from scripts.run_comparison import build_plan
    if jax.default_backend() != "tpu":
        p.error("fused smoke checks require a TPU; use scripts/tpu_python.sh")
    jax.config.update("jax_enable_x64", False)
    seq, batch = (128, 1) if a.tiny else (2048, 2 if a.scale == "350m" else 1)
    settings = ["attn_impl=flash", "head_impl=fused"]
    if a.tiny:
        settings += ["d_model=128", "n_layers=1", "n_heads=2", "d_ff=256", "flash_block=128"]
    plan = build_plan(ROOT / f"configs/comparison_{a.scale}.json", Path("unused"), Path("unused"), Path("unused"), "smoke", 2)
    report = {"purpose": "synthetic smoke verification", "full_training_launched": False,
              "full_model_shape": not a.tiny, "scale": a.scale, "backend": jax.default_backend(), "jax_version": jax.__version__,
              "source": source_fingerprint(), "plan": {k: plan[k] for k in
              ("parameter_counts", "optimizer_steps", "effective_training_tokens")}, "runs": {}}
    with tempfile.TemporaryDirectory(prefix="algebraic-smoke-") as directory:
        root = Path(directory)
        rng = np.random.default_rng(42)
        for name in ("train", "valid"):
            np.save(root / f"{name}.npy", rng.integers(0, 32768, seq * batch * 8 + 1, dtype=np.uint16))
        for arch in ("standard", "algebraic"):
            run = root / arch
            command = [sys.executable, "-m", "algebraic.train", "--preset", arch, "--size", a.scale,
                       "--seq-len", str(seq), "--steps", "1", "--batch", str(batch), "--micro", str(batch),
                       "--eval-batch", str(batch), "--eval-tokens", str(seq * batch),
                       "--seed", "42", "--data-seed", "42", "--lr", str(plan["config"]["training"]["lr"]), "--warmup", "1",
                       "--train", str(root / "train.npy"), "--valid", str(root / "valid.npy"),
                       "--log-every", "1", "--out", str(run), "--save-params"]
            for setting in settings:
                command += ["--set", setting]
            started = time.monotonic()
            train_main(command[3:])
            result = json.loads((run / "result.json").read_text())
            cfg, params = load_model(run)
            n_params = count_params(params)
            if result["completed_steps"] != 1 or n_params != result["n_params"]:
                raise AssertionError("synthetic optimizer/checkpoint validation failed")
            if not a.tiny and n_params != plan["parameter_counts"][arch]:
                raise AssertionError("full preset does not match the planned parameter budget")
            if not all(np.isfinite(result[k]) for k in ("final_eval_nll", "final_train_loss")):
                raise AssertionError("non-finite synthetic training")
            report["runs"][arch] = {"passed": True, "parameters": n_params, "completed_steps": 1,
                                     "seq_len": cfg.seq_len, "micro_batch": batch,
                                     "attn_impl": cfg.attn_impl, "head_impl": cfg.head_impl,
                                     "final_eval_nll": result["final_eval_nll"],
                                     "gradient_norm": json.loads((run / "log.jsonl").read_text().splitlines()[0])["gnorm"],
                                     "seconds_including_compile_and_io": time.monotonic()-started}
            del params
            jax.clear_caches()
            write_json(a.out, report)
        report["algebraic_primitive_audit"] = audit("algebraic", {
            "d_model": 128, "n_heads": 2, "d_ff": 256, "attn_impl": "flash", "head_impl": "fused",
            "flash_block": 128}, seq_len=128)
        if not report["algebraic_primitive_audit"]["passed"]:
            raise AssertionError("forbidden primitive in fused model gradient")
    report["passed"] = True
    write_json(a.out, report)
    print(f"Smoke verification passed: {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
