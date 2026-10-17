#!/usr/bin/env python3
"""Benchmark and record execution speed and throughput for Standard vs Algebraic architectures.

Measures forward and backward latency and token throughput (tokens/sec) on Cloud TPU v4
for:
1. FlashAttention Pallas Kernels: Standard causal softmax vs Pure Algebraic (rho(s/8)^16)
   across sequence lengths T in {512, 1024, 2048} with H=12, D=64.
2. Full Transformer Model Steps: 125M scale (d_model=768, n_layers=12, V=32,768)
   standard vs algebraic end-to-end forward+backward AdamW step.

Outputs structured benchmark results to results/verification/benchmark_throughput.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
from algebraic.benchmarking import source_fingerprint, write_json
from algebraic.kernels.flash import flash_attention
from algebraic import config as cfgs
from algebraic.model import LMConfig, count_params, init_params, forward_hidden, head_loss, position_tables


def timeit(fn, *args, warmup: int = 5, iters: int = 20) -> float:
    """Accurately time a JAX function using warmup and block_until_ready."""
    # Warmup runs
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
        jax.block_until_ready(out)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def benchmark_attention_kernels(seq_lengths: list[int] = [512, 1024, 2048], batch: int = 8, heads: int = 12, dim: int = 64):
    """Benchmark Standard Softmax vs Algebraic rho(s/8)^16 attention kernels on TPU."""
    print("=" * 80)
    print("BENCHMARK 1: Pallas FlashAttention Kernels (Standard Softmax vs Algebraic rho(s/8)^16)")
    print(f"Hardware: {jax.devices()[0].device_kind} | Batch: {batch} | Heads: {heads} | Head Dim: {dim}")
    print("=" * 80)

    kernel_results = {}
    rng = jax.random.PRNGKey(42)

    for T in seq_lengths:
        print(f"\n--- Sequence Length T = {T} ---")
        k1, k2, k3, k4 = jax.random.split(rng, 4)
        rng = k4
        q = jax.random.normal(k1, (batch, heads, T, dim), dtype=jnp.bfloat16)
        k = jax.random.normal(k2, (batch, heads, T, dim), dtype=jnp.bfloat16)
        v = jax.random.normal(k3, (batch, heads, T, dim), dtype=jnp.bfloat16)
        g = jax.random.normal(k4, (batch, heads, T, dim), dtype=jnp.bfloat16)
        sink = jnp.zeros((1, heads, 1, 1), jnp.float32)

        # Causal FLOPs: 4 * B * H * T^2 * D / 2
        flops_fwd = 4.0 * batch * heads * T * T * dim / 2.0
        total_tokens = batch * T

        seq_res = {}
        for arch in ["standard", "algebraic"]:
            if arch == "standard":
                attn_fn = lambda q, k, v: flash_attention(
                    q, k, v, sink_logit=None, kind="softmax", block=512
                )
            else:
                attn_fn = lambda q, k, v: flash_attention(
                    q, k, v, sink_logit=sink, kind="asinh", n=16, c=8.0, block=512, range_guard=True
                )

            @jax.jit
            def fwd(q, k, v):
                return attn_fn(q, k, v)

            @jax.jit
            def fwd_bwd(q, k, v):
                def loss(q, k, v):
                    out = attn_fn(q, k, v)
                    return jnp.sum(out.astype(jnp.float32) * g)
                return jax.grad(loss, argnums=(0, 1, 2))(q, k, v)

            tf = timeit(fwd, q, k, v, warmup=5, iters=20)
            tb = timeit(fwd_bwd, q, k, v, warmup=5, iters=20)
            t_bwd_only = max(0.0, tb - tf)

            fwd_tok_s = total_tokens / tf
            step_tok_s = total_tokens / tb
            fwd_tflops = (flops_fwd / tf) / 1e12
            step_tflops = (3.5 * flops_fwd / tb) / 1e12

            res = {
                "fwd_latency_ms": round(tf * 1000.0, 3),
                "bwd_latency_ms": round(t_bwd_only * 1000.0, 3),
                "fwd_bwd_latency_ms": round(tb * 1000.0, 3),
                "fwd_throughput_tokens_per_s": round(fwd_tok_s, 1),
                "fwd_bwd_throughput_tokens_per_s": round(step_tok_s, 1),
                "fwd_tflops": round(fwd_tflops, 2),
                "fwd_bwd_tflops": round(step_tflops, 2),
            }
            seq_res[arch] = res

            print(
                f"  [{arch.upper():9s}] Fwd: {res['fwd_latency_ms']:6.2f} ms ({res['fwd_throughput_tokens_per_s']:10.1f} tok/s) | "
                f"Fwd+Bwd: {res['fwd_bwd_latency_ms']:6.2f} ms ({res['fwd_bwd_throughput_tokens_per_s']:10.1f} tok/s) | "
                f"TFLOP/s: {res['fwd_tflops']:5.1f} fwd, {res['fwd_bwd_tflops']:5.1f} step"
            )

        # Compute algebraic / standard ratios
        std = seq_res["standard"]
        alg = seq_res["algebraic"]
        seq_res["algebraic_relative_to_standard"] = {
            "fwd_latency_ratio": round(alg["fwd_latency_ms"] / std["fwd_latency_ms"], 3),
            "step_latency_ratio": round(alg["fwd_bwd_latency_ms"] / std["fwd_bwd_latency_ms"], 3),
            "step_throughput_ratio": round(alg["fwd_bwd_throughput_tokens_per_s"] / std["fwd_bwd_throughput_tokens_per_s"], 3),
        }
        print(f"  Ratio (Alg/Std): Step Latency = {seq_res['algebraic_relative_to_standard']['step_latency_ratio']:.2f}x | "
              f"Throughput = {seq_res['algebraic_relative_to_standard']['step_throughput_ratio'] * 100:.1f}%")

        kernel_results[f"seq_{T}"] = seq_res

    return kernel_results


def benchmark_model_steps(scale: str = "125m", batch: int = 4, seq_len: int = 2048):
    """Benchmark full forward + backward step time and throughput for 125M model."""
    print("\n" + "=" * 80)
    print(f"BENCHMARK 2: Full Transformer 1-Step Training Benchmark ({scale.upper()} Scale)")
    print(f"Batch: {batch} | Seq Len: {seq_len} | Total Tokens/Step: {batch * seq_len}")
    print("=" * 80)

    model_results = {}
    rng = jax.random.PRNGKey(101)

    for arch in ["standard", "algebraic"]:
        cfg_builder = cfgs.standard if arch == "standard" else cfgs.algebraic
        cfg = cfg_builder(scale, vocab_size=32768, seq_len=seq_len, attn_impl="flash", head_impl="fused")

        rng, r1, r2 = jax.random.split(rng, 3)
        params = init_params(cfg, r1)
        tokens = jax.random.randint(r2, (batch, seq_len), 0, 32768, dtype=jnp.uint16)
        tables = position_tables(cfg, seq_len)

        # Loss function with forward hidden + fused head
        def compute_loss(p, tok):
            h = forward_hidden(p, tok, cfg, tables)
            loss, _ = head_loss(p, h, tok, cfg)
            return loss

        @jax.jit
        def train_step(p, tok):
            val, grads = jax.value_and_grad(compute_loss)(p, tok)
            return val, grads

        step_time = timeit(train_step, params, tokens, warmup=3, iters=10)
        tok_s = (batch * seq_len) / step_time

        model_results[arch] = {
            "parameters": count_params(params),
            "step_latency_ms": round(step_time * 1000.0, 2),
            "throughput_tokens_per_s": round(tok_s, 1),
        }
        print(
            f"  [{arch.upper():9s}] Params: {model_results[arch]['parameters']:,} | "
            f"Step Time: {model_results[arch]['step_latency_ms']:7.2f} ms | "
            f"Throughput: {model_results[arch]['throughput_tokens_per_s']:8.1f} tokens/s"
        )

    std = model_results["standard"]
    alg = model_results["algebraic"]
    model_results["comparison"] = {
        "latency_ratio_algebraic_over_standard": round(alg["step_latency_ms"] / std["step_latency_ms"], 3),
        "throughput_ratio_algebraic_over_standard": round(alg["throughput_tokens_per_s"] / std["throughput_tokens_per_s"], 3),
    }
    print(
        f"  Full Model Ratio (Alg/Std): Step Latency = {model_results['comparison']['latency_ratio_algebraic_over_standard']:.2f}x | "
        f"Throughput = {model_results['comparison']['throughput_ratio_algebraic_over_standard'] * 100:.1f}%\n"
    )

    return model_results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "results/verification/benchmark_throughput.json")
    args = p.parse_args()

    t_start = time.time()
    devices = jax.devices()
    print(f"JAX Devices: {devices}")

    attn_results = benchmark_attention_kernels()
    model_results = benchmark_model_steps()

    report = {
        "purpose": "Execution speed and token throughput benchmark: Standard Softmax vs Pure Algebraic attention on Cloud TPU v4",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source_fingerprint(),
        "hardware": {
            "device_kind": str(devices[0].device_kind),
            "device_count": len(devices),
            "platform": str(jax.default_backend()),
        },
        "attention_kernel_benchmark": attn_results,
        "full_model_step_benchmark": model_results,
        "elapsed_seconds": round(time.time() - t_start, 2),
    }

    write_json(args.out, report)
    print(f"Benchmark report written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
