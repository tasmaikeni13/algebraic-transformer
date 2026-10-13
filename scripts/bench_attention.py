#!/usr/bin/env python3
"""Time causal attention forward+backward on one TPU chip.

Compares this repository's Pallas kernels (softmax and asinh) with JAX's
maintained TPU flash attention and splash attention at a training shape.

  python scripts/bench_attention.py --batch 8 --heads 12 --seq 2048 --dim 64
"""

import argparse
import json
import math
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp


def timeit(fn, *args, iters=20):
    out = fn(*args)
    jax.block_until_ready(out)
    t = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--blocks", default="512,512x1024,1024x512,1024")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    from algebraic.kernels.flash import flash_attention
    B, H, T, D = args.batch, args.heads, args.seq, args.dim
    ks = jax.random.split(jax.random.PRNGKey(0), 4)
    q, k, v, g = (jax.random.normal(kk, (B, H, T, D), jnp.bfloat16) for kk in ks)
    sink = jnp.zeros((1, H, 1, 1), jnp.float32)
    flops_fwd = 4 * B * H * T * T * D / 2  # causal

    results = {}

    def bench(name, attn):
        def fwd(q, k, v):
            return attn(q, k, v)

        def fwdbwd(q, k, v):
            return jax.grad(lambda q, k, v: jnp.sum(attn(q, k, v).astype(jnp.float32) * g),
                            argnums=(0, 1, 2))(q, k, v)
        tf = timeit(jax.jit(fwd), q, k, v)
        tb = timeit(jax.jit(fwdbwd), q, k, v)
        results[name] = {"fwd_ms": tf * 1e3, "fwd_bwd_ms": tb * 1e3,
                         "fwd_tflops": flops_fwd / tf / 1e12,
                         "fwd_bwd_tflops": 3.5 * flops_fwd / tb / 1e12}
        print(f"{name:34s} fwd {tf*1e3:7.2f} ms  fwd+bwd {tb*1e3:7.2f} ms  "
              f"({flops_fwd/tf/1e12:5.1f} / {3.5*flops_fwd/tb/1e12:5.1f} TFLOP/s)", flush=True)

    for spec in args.blocks.split(","):
        bq, _, bk = spec.partition("x")
        bq = int(bq); bk = int(bk or bq)
        if T % bq or T % bk:
            continue
        tag = f"{bq}x{bk}"
        bench(f"ours softmax {tag}", lambda q, k, v, bq=bq, bk=bk: flash_attention(q, k, v, None, kind="softmax", block_q=bq, block_k=bk))
        bench(f"ours softmax+sink {tag}", lambda q, k, v, bq=bq, bk=bk: flash_attention(q, k, v, sink, kind="softmax", block_q=bq, block_k=bk))
        for shift, fast in ((True, False), (True, True), (False, True)):
            name = f"ours asinh16/16 {'shift' if shift else 'noshift'}{'+fast' if fast else ''} {tag}"
            bench(name, lambda q, k, v, bq=bq, bk=bk, shift=shift, fast=fast: flash_attention(
                q, k, v, sink, kind="asinh", n=16, c=16.0, block_q=bq, block_k=bk, shift=shift, fast_rho=fast))

    from jax.experimental.pallas.ops.tpu import flash_attention as jfa
    bench("jax flash_attention (defaults)", lambda q, k, v: jfa.flash_attention(q, k, v, causal=True, sm_scale=1 / math.sqrt(D)))
    try:
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sm
        mask = sm.MultiHeadMask([sm.CausalMask((T, T)) for _ in range(H)])
        kernel = sk.make_splash_mha(mask, head_shards=1, q_seq_shards=1,
                                    block_sizes=sk.BlockSizes(block_q=512, block_kv=512, block_kv_compute=512,
                                                              block_q_dkv=512, block_kv_dkv=512,
                                                              block_kv_dkv_compute=512, block_q_dq=512,
                                                              block_kv_dq=512))
        splash = jax.vmap(kernel)
        bench("jax splash_attention b512", lambda q, k, v: splash(q * (1 / math.sqrt(D)), k, v))
    except Exception as e:  # pragma: no cover - API differences across versions
        print("splash unavailable:", type(e).__name__, e)

    def dense(q, k, v):
        s = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32) / math.sqrt(D)
        s = jnp.where(jnp.tril(jnp.ones((T, T), bool)), s, -1e30)
        return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(s, -1).astype(v.dtype), v)
    bench("xla dense softmax", dense)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"shape": [B, H, T, D], "results": results}, indent=1))


if __name__ == "__main__":
    main()
