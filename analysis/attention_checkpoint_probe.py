#!/usr/bin/env python3
"""Synthetic optimizer speed smoke for selectively saved attention residuals.

Each subprocess performs one compile step and at most ten timed AdamW steps.
No corpus is opened and no model/checkpoint weights are saved. Both standard
and algebraic receive the same policy candidates and synthetic token stream.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import jax
from jax import ad_checkpoint, lax
import jax.numpy as jnp
import numpy as np

from algebraic import config, model, train
from algebraic.kernels import flash as fk

NAMES = tuple("flash_" + x for x in ("q", "k", "v", "sink", "output", "stat"))


@contextmanager
def save_attention(enabled):
    original_hidden, original_fwd = model.forward_hidden, fk._flash_fwd
    if enabled:
        def cached_flash(q, k, v, sink, var, bq, bk, has_sink):
            output, residuals = original_fwd(q, k, v, sink, var, bq, bk, has_sink)
            residuals = tuple(ad_checkpoint.checkpoint_name(a, name) for a, name in zip(residuals, NAMES))
            return output, residuals
        fk._flash.defvjp(cached_flash, fk._flash_bwd)

        def hidden(params, tokens, cfg, tables):
            cos, sin = tables
            x = params["embed"][tokens].astype(cfg.compute_dtype)
            x = model._norm(x, params.get("embed_gain"), cfg)
            def body(carry, lp):
                return model._block(cfg, cos, sin, carry, lp), None
            if cfg.remat:
                body = jax.checkpoint(body, prevent_cse=False,
                                      policy=jax.checkpoint_policies.save_only_these_names(*NAMES))
            x, _ = lax.scan(body, x, params["layers"])
            return model._norm(x, params.get("final_gain"), cfg)
        model.forward_hidden = hidden
    try:
        yield
    finally:
        model.forward_hidden = original_hidden
        fk._flash.defvjp(original_fwd, fk._flash_bwd)


def validate(arch):
    cfg = config.build(arch, "proxy", d_model=128, n_heads=2, n_layers=2, d_ff=256,
                       vocab_size=256, seq_len=2048, attn_impl="flash")
    params = model.init_params(cfg, jax.random.PRNGKey(409))
    keys = jax.random.split(jax.random.PRNGKey(419), 2)
    tokens, targets = [jax.random.randint(key, (1, cfg.seq_len), 0, cfg.vocab_size) for key in keys]
    tables = model.position_tables(cfg)
    def loss(params):
        return model.head_loss(params, model.forward_hidden(params, tokens, cfg, tables), targets, cfg)[0]
    original_attention = model._attention
    def resident_attention(q,k,v,sink,cfg):
        return fk.flash_attention(q,k,v,sink,kind=cfg.attn,n=cfg.attn_n,c=cfg.attn_c,
                                  block=cfg.flash_block,range_guard=cfg.attn == "asinh",head_resident=False)
    outputs, gradients = [], []
    try:
        for enabled, attention in [(False,resident_attention),(False,original_attention),(True,original_attention)]:
            model._attention = attention
            with save_attention(enabled):
                jax.clear_caches()
                outputs.append(jax.jit(lambda p: model.forward_hidden(p,tokens,cfg,tables))(params))
                gradients.append(jax.jit(jax.grad(loss))(params))
    finally:
        model._attention = original_attention
    checks = []
    for name,output,gradient in zip(["whole_head","saved"],outputs[1:],gradients[1:]):
        output_error = float(jnp.max(jnp.abs(outputs[0].astype(jnp.float32)-output.astype(jnp.float32))))
        relative = [float(jnp.linalg.norm(a-b)/jnp.maximum(jnp.linalg.norm(a),1e-8)) for a,b in
                    zip(jax.tree_util.tree_leaves(gradients[0]),jax.tree_util.tree_leaves(gradient))]
        check = dict(case=name,output_max_abs_error=output_error,max_parameter_relative_l2=max(relative),
                     passed=output_error <= .005 and max(relative) <= .01)
        checks.append(check)
    if not all(c["passed"] for c in checks):
        raise ValueError(f"whole-head/checkpoint model VJP check failed: {checks}")
    return dict(config=cfg.__dict__,reference="query-streamed resident causal complete-model output and parameter VJP",
                checks=checks,passed=True)


def child(args):
    check = validate(args.arch)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="algebraic-speed-smoke-") as path:
        root = Path(path)
        rng = np.random.default_rng(args.seed)
        for name in ["train", "valid"]:
            np.save(root / f"{name}.npy", rng.integers(0, 50257, 8*2048*8+1, dtype=np.uint16))
        run = root / "profile"
        # The trainer exits immediately after its profile; the explicit budget
        # also limits work if its profile behavior changes.
        argv = ["--preset", args.arch, "--size", args.size, "--seq-len", "2048",
                "--batch", "8", "--micro", "8", "--steps", str(args.steps+1),
                "--profile-steps", str(args.steps), "--train", str(root / "train.npy"),
                "--valid", str(root / "valid.npy"), "--eval-batch", "8", "--eval-tokens", "16384",
                "--log-every", str(args.steps+2), "--seed", str(args.seed), "--set", "attn_impl=flash",
                "--out", str(run)]
        if args.head == "fused":
            argv += ["--set", "head_impl=fused"]
        original_attention = model._attention
        if args.case in ("full_tiles","resident_tiles","whole_head"):
            def full_tiles(q, k, v, sink, cfg):
                if cfg.attn_impl != "flash":
                    return original_attention(q, k, v, sink, cfg)
                opts = {"head_resident":args.case == "whole_head"}
                if args.case == "full_tiles":
                    opts.update(diagonal_bwd=False,diagonal_fwd=False)
                return fk.flash_attention(q,k,v,sink,kind=cfg.attn,n=cfg.attn_n,
                    c=cfg.attn_c,block=cfg.flash_block,range_guard=cfg.attn == "asinh",**opts)
            model._attention = full_tiles
        try:
            with save_attention(args.case == "saved"):
                train.main(argv)
        finally:
            model._attention = original_attention
        report = dict(case=args.case, arch=args.arch, size=args.size, seed=args.seed,
                      devices=[str(d) for d in jax.devices()],
                      visible_chips=os.environ.get("TPU_VISIBLE_CHIPS"),
                      validation=check, profile=json.loads((run / "profile.json").read_text()),
                      meta=json.loads((run / "meta.json").read_text()),
                      synthetic_tokens=True, full_training_launched=False, checkpoints_saved=False,
                      source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                     [Path(__file__),Path(fk.__file__),ROOT/"src/algebraic/kernels/head_resident.py",
                                      Path(model.__file__),Path(train.__file__)]})
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--child", action="store_true")
    p.add_argument("--arch", choices=["standard", "algebraic"], default="algebraic")
    p.add_argument("--case", choices=["baseline","saved","full_tiles","resident_tiles","whole_head"], default="saved")
    p.add_argument("--cases", nargs="+", choices=["baseline","saved","full_tiles","resident_tiles","whole_head"], default=["baseline", "saved"])
    p.add_argument("--head-comparison",action="store_true",help="Compare the two attention backends in both architectures.")
    p.add_argument("--arches", nargs="+", choices=["standard", "algebraic"], default=["standard", "algebraic"])
    p.add_argument("--size", choices=["60m", "125m"], default="125m")
    p.add_argument("--head", choices=["dense", "fused"], default="dense")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=421)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.steps <= 10:
        p.error("this smoke probe supports one to ten timed steps")
    if args.child:
        return child(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(utc=datetime.now(timezone.utc).isoformat(), size=args.size, steps=args.steps,
                  seed=args.seed, head=args.head, purpose="synthetic speed smoke",
                  synthetic_tokens=True, full_training_launched=False, cases={},
                  jax=jax.__version__)
    order = [(a, c) for a in args.arches for c in args.cases]
    if args.head_comparison:
        order = [("standard","baseline"),("standard","whole_head"),("algebraic","resident_tiles"),("algebraic","baseline")]
    np.random.default_rng(args.seed).shuffle(order)
    with tempfile.TemporaryDirectory(prefix="algebraic-checkpoint-probe-") as path:
        root = Path(path)
        for arch, case in order:
            out = root / f"{arch}_{case}.json"
            command = [sys.executable, str(Path(__file__)), "--child", "--arch", arch, "--case", case,
                       "--size", args.size, "--steps", str(args.steps), "--seed", str(args.seed),
                       "--head", args.head, "--out", str(out)]
            started = time.monotonic()
            job = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            record = dict(exit_code=job.returncode, seconds_including_compile=time.monotonic()-started)
            if job.returncode:
                record["error_tail"] = job.stderr[-5000:]+job.stdout[-2000:]
                print("FAILED", arch, case, record["error_tail"][-1500:], flush=True)
            else:
                record.update(json.loads(out.read_text()))
                print(arch, case, record["profile"]["step_time"]*1000, "ms", flush=True)
            report["cases"][f"{arch}_{case}"] = record
            args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return 0


if __name__ == "__main__":
    main()
