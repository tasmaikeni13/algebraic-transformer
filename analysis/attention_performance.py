#!/usr/bin/env python3
"""Compile once, then compare causal forward and training kernels in random order.

Training means forward plus all q/k/v VJPs, with a fresh arbitrary cotangent.
Repeated paired rounds supply raw samples and a bootstrap CI for latency ratios.
The maintained Splash baseline pads *only values* to its required 128 lanes;
the original q/k dimension and score scale stay unchanged. Failed baselines
are retained as errors rather than silently dropped.
"""

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from src.sota.kernels.flash import flash_attention


def ratio_ci(a, b, seed=19, samples=10000):
    """Paired geometric latency ratio, bootstrap over whole measurement rounds."""
    delta = np.log(np.asarray(a)) - np.log(np.asarray(b))
    rng = np.random.default_rng(seed)
    boot = np.exp(delta[rng.integers(len(delta), size=(samples, len(delta)))].mean(1))
    return {"ratio": float(np.exp(delta.mean())),
            "ci95": np.quantile(boot, [.025, .975]).tolist(), "rounds": len(delta)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--blocks", default="128,256,512,128x256,256x128,256x512,512x256")
    p.add_argument("--families", default="softmax,asinh,asinh_inv")
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--baseline-blocks", default="128,256,512")
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--source-revision", help="revision of a checkout copied to a worker without .git")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    B, H, T, D = shape = (args.batch, args.heads, args.seq, args.dim)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), 4)
    q, k, v, g = (jax.random.normal(key, shape, jnp.bfloat16) for key in keys)
    sink = jnp.zeros((H,), jnp.float32)
    functions = {}
    for spec in args.blocks.split(","):
        bq, _, bk = spec.partition("x")
        bq, bk = int(bq), int(bk or bq)
        if T % bq or T % bk:
            continue
        for family in args.families.split(","):
            kw = dict(block_q=bq, block_k=bk)
            if family in ("softmax", "softmax_compact", "softmax_partial"):
                kw.update(kind="softmax", compact_stat=family != "softmax")
                if family == "softmax_partial":
                    kw.update(resident_bwd=False)
                s = None
            elif family in ("asinh", "asinh_inv", "asinh_exact", "asinh_compact", "asinh_poly", "asinh_poly_compact", "asinh_guarded", "asinh_guarded_partial"):
                kw.update(kind="asinh", n=16, c=8., shift=False,
                          fast_rho=family != "asinh_exact", inv_stat=family != "asinh",
                          compact_stat=family in ("asinh_compact", "asinh_poly_compact", "asinh_guarded", "asinh_guarded_partial"),
                          poly_rho=family in ("asinh_poly", "asinh_poly_compact"),
                          range_guard=family in ("asinh_guarded", "asinh_guarded_partial"))
                if family == "asinh_guarded_partial":
                    kw.update(resident_bwd=False)
                s = sink
            else:
                raise ValueError(f"unknown family {family}")
            functions[f"{family}_{bq}x{bk}"] = partial(flash_attention, sink_logit=s, **kw)
    if args.baselines:
        from jax.experimental.pallas.ops.tpu import flash_attention as jfa
        functions["jax_flash_default"] = partial(jfa.flash_attention, causal=True, sm_scale=1 / math.sqrt(D))
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sm
        mask = sm.MultiHeadMask([sm.CausalMask((T, T)) for _ in range(H)])
        for block in map(int, args.baseline_blocks.split(",")):
            for fused in (False, True):
                bs = sk.BlockSizes(block_q=block, block_kv=block, block_kv_compute=block,
                                   block_q_dkv=block, block_kv_dkv=block,
                                   block_kv_dkv_compute=block, block_q_dq=None if fused else block,
                                   block_kv_dq=None if fused else block, use_fused_bwd_kernel=fused)
                kernel = jax.vmap(sk.make_splash_mha(mask, head_shards=1, q_seq_shards=1, block_sizes=bs))
                def splash(q, k, v, kernel=kernel):
                    pad = (-v.shape[-1]) % 128
                    vp = jnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, pad)))
                    return kernel(q * (1 / math.sqrt(D)), k, vp)[..., :D]
                functions[f"splash_{block}_fused{fused}"] = splash
    revision = args.source_revision
    if revision is None:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                             capture_output=True, check=False)
        revision = git.stdout.strip() if git.returncode == 0 else None
    report = {"schema": 1, "utc": datetime.now(timezone.utc).isoformat(), "shape": shape,
              "dtype": "bfloat16", "causal": True, "asinh_n": 16, "asinh_c": 8.,
              "asinh_sink": 0., "softmax_sink": None, "jax": jax.__version__,
              "devices": [str(d) for d in jax.devices()], "hostname": platform.node(),
              "rounds": args.rounds, "iters_per_round": args.iters, "seed": args.seed,
              "git_revision": revision,
              "kernel_sha256": hashlib.sha256((ROOT / "src/sota/kernels/flash.py").read_bytes()).hexdigest(),
              "radical_sha256": hashlib.sha256((ROOT / "src/sota/kernels/radical.py").read_bytes()).hexdigest(),
              "results": {}, "comparisons": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    compiled = {}
    for name, fn in functions.items():
        report["results"][name] = {}
        for mode in ("forward", "training"):
            try:
                if mode == "forward":
                    f = jax.jit(fn)
                else:
                    def loss(q, k, v, fn=fn):
                        return jnp.sum(fn(q, k, v).astype(jnp.float32) * g.astype(jnp.float32))
                    f = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
                start = time.perf_counter()
                jax.block_until_ready(f(q, k, v))
                compile_s = time.perf_counter() - start
                for _ in range(3):
                    jax.block_until_ready(f(q, k, v))
                compiled[name, mode] = f
                report["results"][name][mode] = {"compile_s": compile_s, "samples_ms": []}
                print(f"compiled {name} {mode} ({compile_s:.1f}s)", flush=True)
            except Exception as exc:
                report["results"][name][mode] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"ERROR {name} {mode}: {type(exc).__name__}: {exc}", flush=True)
            save()
    rng = np.random.default_rng(args.seed)
    cases = list(compiled)
    for round_id in range(args.rounds):
        order = rng.permutation(len(cases))
        for idx in order:
            name, mode = cases[int(idx)]
            f = compiled[name, mode]
            jax.block_until_ready(f(q, k, v))
            start = time.perf_counter()
            for _ in range(args.iters):
                out = f(q, k, v)
            jax.block_until_ready(out)
            ms = (time.perf_counter() - start) * 1000 / args.iters
            report["results"][name][mode]["samples_ms"].append(ms)
        print(f"round {round_id + 1}/{args.rounds}", flush=True)
        save()
    for name in functions:
        for mode in ("forward", "training"):
            result = report["results"][name][mode]
            if "samples_ms" in result:
                a = result["samples_ms"]
                result["median_ms"] = float(np.median(a))
                print(f"{name:30s} {mode:8s} {np.median(a):.4f} ms", flush=True)
    for mode in ("forward", "training"):
        standard = [n for n in functions if (n.startswith("softmax") or n.startswith("splash"))
                    and (n, mode) in compiled]
        if standard:
            best = min(standard, key=lambda n: report["results"][n][mode]["median_ms"])
            comparisons = {}
            for name in functions:
                if (name, mode) in compiled:
                    comparisons[name] = ratio_ci(report["results"][name][mode]["samples_ms"],
                                                  report["results"][best][mode]["samples_ms"])
            report["comparisons"][mode] = {"reference": best, "selection": "fastest measured standard; exploratory",
                                            "latency_ratios": comparisons}
    save()


if __name__ == "__main__":
    main()
