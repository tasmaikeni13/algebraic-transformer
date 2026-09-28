#!/usr/bin/env python3
"""Pilot study: End-to-end distributed pilot pretraining on 16 TPU v4 Pod slice.

Trains AlgebraicTransformerLM vs StandardTransformerLM (about 20.45M parameters).

The configured tied embedding and projection matrices total
20,445,984/20,450,016 trainable scalars.
on WikiText-103 across 10^5 steps under identical data order and budget.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple

if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "tpu,cpu"
if "JAX_ENABLE_X64" not in os.environ:
    os.environ["JAX_ENABLE_X64"] = "0"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

from src.model import AlgebraicTransformerLM, ModelConfig, count_parameters
from src.baseline import StandardTransformerLM, BaselineConfig
from src.optimizer import algebraic_adamw, ards_schedule
from src.attention import build_cayley_rotary_matrix
from src.baseline import _build_standard_rope
from src.dataset import ensure_wikitext103_ready, ShardedTokenLoader, evaluate_perplexity
from src.mesh import compile_data_parallel_step, create_tpu_mesh, ModelSharding
from scripts.pilot_records import REQUIRED_PILOT_STEPS, environment, source_hashes, write_json
from scripts.pilot_experiments import (
    audit_pilot_ast,
    create_cosine_schedule,
    train_step_algebraic_fn,
    train_step_baseline_fn,
)


def run_training_arm(
    model_name: str,
    model,
    optimizer_tx,
    train_step_fn,
    loader: ShardedTokenLoader,
    mesh: Any,
    total_steps: int,
    log_every: int,
    valid_path: Path,
    is_algebraic: bool,
    seed: int = 42,
):
    """Executes a single model pretraining arm across total_steps."""
    proc_idx = jax.process_index()
    if proc_idx == 0:
        print(f"\n{'='*20} Starting Pretraining: {model_name} ({total_steps} steps) {'='*20}", flush=True)

    sharding = ModelSharding(mesh)
    from jax.sharding import NamedSharding, PartitionSpec as P
    data_sharding = NamedSharding(mesh, P(('data', 'fsdp', 'model'), None))

    init_key = jax.random.PRNGKey(seed)
    params = model.init_params(init_key)
    opt_state = optimizer_tx.init(params)

    params = jax.device_put(params, sharding.replicated)
    opt_state = jax.device_put(opt_state, sharding.replicated)

    # Explicit shard_map is required for Mosaic/Pallas kernels nested in the
    # model; an outer SPMD jit cannot partition those custom calls itself.
    jitted_step = compile_data_parallel_step(
        train_step_fn,
        mesh,
        P(("data", "fsdp", "model"), None),
    )

    # Metrics trackers
    losses = []
    grad_norms = []
    nan_or_inf_count = 0
    loss_spike_count = 0
    prev_loss = None
    step_times = []

    # Compile with step 0 data, but discard the result so the recorded budget
    # remains exactly ``total_steps`` optimizer updates.
    x_init_np, y_init_np = loader.get_batch(0)
    x_init = jax.make_array_from_process_local_data(
        data_sharding, x_init_np, (loader.batch_size, loader.seq_len)
    )
    y_init = jax.make_array_from_process_local_data(
        data_sharding, y_init_np, (loader.batch_size, loader.seq_len)
    )
    warmup_params, warmup_opt_state, _ = jitted_step(
        params, opt_state, x_init, y_init
    )
    jax.block_until_ready(warmup_params)
    del warmup_params, warmup_opt_state

    start_time = time.perf_counter()
    tokens_per_step = loader.batch_size * loader.seq_len  # 64 * 512 = 32,768

    for step in range(total_steps):
        t0 = time.perf_counter()
        x_np, y_np = loader.get_batch(step)
        x_jax = jax.make_array_from_process_local_data(
            data_sharding, x_np, (loader.batch_size, loader.seq_len)
        )
        y_jax = jax.make_array_from_process_local_data(
            data_sharding, y_np, (loader.batch_size, loader.seq_len)
        )

        params, opt_state, metrics = jitted_step(params, opt_state, x_jax, y_jax)
        # Block until ready for accurate timing
        loss_val = float(jax.device_get(metrics["loss"]))
        grad_norm_val = float(jax.device_get(metrics["grad_norm"]))
        t1 = time.perf_counter()

        step_duration = t1 - t0
        step_times.append(step_duration)

        # Check numerical stability
        if not (math.isfinite(loss_val) and math.isfinite(grad_norm_val)):
            nan_or_inf_count += 1

        # Check loss spike anomaly: Delta L > 1.5
        if prev_loss is not None:
            delta_l = loss_val - prev_loss
            if delta_l > 1.5:
                loss_spike_count += 1
        prev_loss = loss_val

        losses.append(loss_val)
        grad_norms.append(grad_norm_val)

        if (step + 1) % log_every == 0 and proc_idx == 0:
            avg_step_time = np.mean(step_times[-log_every:])
            tok_per_sec = tokens_per_step / max(1e-6, avg_step_time)
            print(
                f"[{model_name}] Step {step+1:6d}/{total_steps} | "
                f"Loss: {loss_val:.4f} | "
                f"Grad Norm: {grad_norm_val:.3f} | "
                f"Throughput: {tok_per_sec:,.0f} tok/s | "
                f"Step time: {avg_step_time*1000:.1f}ms",
                flush=True,
            )

    total_duration = time.perf_counter() - start_time
    # Steady-state throughput excludes transient startup steps
    warmup_cutoff = min(10, max(1, len(step_times) // 10))
    steady_steps = step_times[warmup_cutoff:] if len(step_times) > warmup_cutoff else step_times
    steady_step_time = float(np.mean(steady_steps)) if steady_steps else (total_duration / max(1, total_steps))
    if jax.process_count() > 1:
        # Throughput is bounded by the slowest host in synchronous training.
        host_times = np.asarray(
            multihost_utils.process_allgather(np.asarray(steady_step_time))
        )
        steady_step_time = float(host_times.max())
    steady_state_tok_per_sec = tokens_per_step / max(1e-6, steady_step_time)
    peak_grad_norm = float(max(grad_norms)) if grad_norms else 0.0

    if proc_idx == 0:
        print(f"[{model_name}] Training Complete in {total_duration:.1f}s. Evaluating held-out perplexity...", flush=True)

    # Extract local replica for evaluation on worker 0
    params_eval = jax.tree_util.tree_map(
        lambda p: jnp.asarray(p.addressable_data(0)) if hasattr(p, "addressable_data") else p,
        params,
    )
    # Every controller executes the same local validation program before the
    # next distributed arm.  This keeps multi-host program order consistent.
    valid_ppl = evaluate_perplexity(
        model,
        params_eval,
        valid_path,
        seq_len=loader.seq_len,
        batch_size=32,
        max_eval_batches=20,
        is_algebraic=is_algebraic,
    )
    if proc_idx == 0:
        print(f"[{model_name}] Validation Perplexity: {valid_ppl:.2f}", flush=True)

    return {
        "model_name": model_name,
        "total_steps": total_steps,
        "final_loss": float(losses[-1]) if losses else 0.0,
        "valid_perplexity": float(valid_ppl),
        "peak_gradient_norm": peak_grad_norm,
        "nan_or_inf_count": int(nan_or_inf_count),
        "loss_spike_count": int(loss_spike_count),
        "steady_state_throughput_tokens_per_sec": float(steady_state_tok_per_sec),
        "losses": losses,
        "grad_norms": grad_norms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/pilot/tpu")
    parser.add_argument("--steps", type=int, default=100_000, help="Total pretraining steps (default: 100,000)")
    parser.add_argument("--log-every", type=int, default=500, help="Logging frequency in steps")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Initialize distributed JAX if in multi-host TPU environment
    is_multi_host = False
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        try:
            jax.distributed.initialize(initialization_timeout=120)
            is_multi_host = True
        except Exception as e:
            print(f"Warning: jax.distributed.initialize failed or already initialized: {e}", flush=True)

    devices = jax.devices()
    num_devices = len(devices)
    proc_idx = jax.process_index()
    proc_count = jax.process_count()

    if devices[0].platform != "tpu" or num_devices != 16 or proc_count != 4:
        raise RuntimeError(
            "Pilot study requires four processes and 16 TPU devices; "
            f"found {proc_count} processes and {num_devices} {devices[0].platform} devices"
        )

    if proc_idx == 0:
        print(f"Hardware: {num_devices} devices across {proc_count} processes. Platform: {devices[0].platform}", flush=True)

    # 2. Verify AST zero-transcendental purity of algebraic stack
    ast_audit = audit_pilot_ast()
    if proc_idx == 0:
        print(f"AST Zero-Transcendental Audit: {'PASS' if ast_audit['passed'] else 'FAIL'}", flush=True)

    # 3. Create Distributed Mesh
    mesh = create_tpu_mesh(devices)
    if proc_idx == 0:
        print(f"Distributed Mesh Topology: {mesh.shape}", flush=True)

    # 4. Prepare Dataset
    train_npy, valid_npy = ensure_wikitext103_ready(args.data_dir)
    loader = ShardedTokenLoader(
        token_path=train_npy,
        batch_size=64,
        seq_len=512,
        process_index=proc_idx,
        process_count=proc_count,
        seed=args.seed,
    )

    # 5. Configurations & Models
    cfg_alg = ModelConfig(
        vocab_size=50257,
        d_model=288,
        num_layers=6,
        num_heads=6,
        d_ff=768,
        max_seq_len=512,
        eps=1e-5,
        eps_vocab=100.0,
        sink_omega=0.5,
        gamma=2.0,
        tie_embeddings=True,
        dtype=jnp.bfloat16 if devices[0].platform == "tpu" else jnp.float32,
    )
    cfg_base = BaselineConfig(
        vocab_size=50257,
        d_model=288,
        num_layers=6,
        num_heads=6,
        d_ff=768,
        max_seq_len=512,
        eps=1e-5,
        tie_embeddings=True,
        dtype=jnp.bfloat16 if devices[0].platform == "tpu" else jnp.float32,
    )

    alg_model = AlgebraicTransformerLM(cfg_alg)
    base_model = StandardTransformerLM(cfg_base)

    head_dim = cfg_alg.d_model // cfg_alg.num_heads
    rotary_params = build_cayley_rotary_matrix(head_dim, cfg_alg.max_seq_len, dtype=cfg_alg.dtype)
    cos_angles, sin_angles = _build_standard_rope(head_dim, cfg_base.max_seq_len)

    # Schedulers & Optimizers
    decay_steps = int(args.steps * 0.8)
    actual_warmup = min(args.warmup_steps, max(1, int(args.steps * 0.05)))
    alg_schedule = ards_schedule(
        learning_rate=args.lr,
        warmup_steps=actual_warmup,
        decay_steps=decay_steps,
        alpha=1.0,
    )
    # Cosine annealing for baseline model
    base_schedule = create_cosine_schedule(
        learning_rate=args.lr,
        warmup_steps=actual_warmup,
        total_steps=args.steps,
    )

    # Optimizer: Algebraic AdamW holds optimizer constant across both architectures
    opt_alg = algebraic_adamw(learning_rate=alg_schedule, weight_decay=1e-2)
    opt_base = algebraic_adamw(learning_rate=base_schedule, weight_decay=1e-2)

    step_fn_alg = train_step_algebraic_fn(
        alg_model,
        opt_alg,
        rotary_params,
        max_grad_norm=1.0,
        data_axis_names=mesh.axis_names,
    )
    step_fn_base = train_step_baseline_fn(
        base_model,
        opt_base,
        cos_angles,
        sin_angles,
        max_grad_norm=1.0,
        data_axis_names=mesh.axis_names,
    )

    # 7. Run Head-to-Head Training
    # Run Baseline first
    base_res = run_training_arm(
        model_name="StandardTransformerLM",
        model=base_model,
        optimizer_tx=opt_base,
        train_step_fn=step_fn_base,
        loader=loader,
        mesh=mesh,
        total_steps=args.steps,
        log_every=args.log_every,
        valid_path=valid_npy,
        is_algebraic=False,
        seed=args.seed,
    )

    # Run Algebraic Model under identical data stream
    alg_res = run_training_arm(
        model_name="AlgebraicTransformerLM",
        model=alg_model,
        optimizer_tx=opt_alg,
        train_step_fn=step_fn_alg,
        loader=loader,
        mesh=mesh,
        total_steps=args.steps,
        log_every=args.log_every,
        valid_path=valid_npy,
        is_algebraic=True,
        seed=args.seed,
    )

    # 8. Compute Parity & Acceptance Gates
    ppl_ratio = alg_res["valid_perplexity"] / max(1e-6, base_res["valid_perplexity"])
    tp_ratio = (
        alg_res["steady_state_throughput_tokens_per_sec"]
        / max(1e-6, base_res["steady_state_throughput_tokens_per_sec"])
    )

    gates = {
        "execution_budget": {
            "steps": args.steps,
            "required_steps": REQUIRED_PILOT_STEPS,
            "passed": bool(args.steps >= REQUIRED_PILOT_STEPS),
        },
        "perplexity_parity": {
            "algebraic_ppl": alg_res["valid_perplexity"],
            "baseline_ppl": base_res["valid_perplexity"],
            "ratio": float(ppl_ratio),
            "threshold": 1.08,
            "passed": bool(ppl_ratio <= 1.08),
        },
        "numerical_stability": {
            "nan_or_inf_count": alg_res["nan_or_inf_count"],
            "threshold": 0,
            "passed": bool(alg_res["nan_or_inf_count"] == 0),
        },
        "loss_spike_anomaly": {
            "spike_count": alg_res["loss_spike_count"],
            "threshold": 0,
            "passed": bool(alg_res["loss_spike_count"] == 0),
        },
        "peak_gradient_norm": {
            "peak_norm": alg_res["peak_gradient_norm"],
            "threshold": 5.0,
            "passed": bool(alg_res["peak_gradient_norm"] <= 5.0),
        },
        "steady_state_throughput": {
            "algebraic_tok_sec": alg_res["steady_state_throughput_tokens_per_sec"],
            "baseline_tok_sec": base_res["steady_state_throughput_tokens_per_sec"],
            "ratio": float(tp_ratio),
            "threshold": 0.90,
            "passed": bool(tp_ratio >= 0.90),
        },
        "ast_audit": {
            "passed": bool(ast_audit["passed"]),
        },
    }

    all_passed = all(g["passed"] for g in gates.values())

    # 9. Assemble Machine-Readable Evidence
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    records = {
        "study": 7,
        "gate_version": 1,
        "environment": environment(),
        "device_count": num_devices,
        "process_count": proc_count,
        "devices": [
            {"id": d.id, "kind": d.device_kind, "process": getattr(d, "process_index", 0)}
            for d in devices
        ],
        "mesh_shape": dict(mesh.shape),
        "total_steps": args.steps,
        "batch_size_sequences": loader.batch_size,
        "context_length_tokens": loader.seq_len,
        "global_batch_size_tokens": loader.batch_size * loader.seq_len,
        "ast_audit": ast_audit,
        "pilot_pretraining": {
            "perplexity_ratio": float(ppl_ratio),
            "throughput_ratio": float(tp_ratio),
            "peak_gradient_norm": alg_res["peak_gradient_norm"],
            "nan_or_inf_count": alg_res["nan_or_inf_count"],
            "loss_spike_count": alg_res["loss_spike_count"],
            "algebraic_results": {
                k: v for k, v in alg_res.items() if k not in ("losses", "grad_norms")
            },
            "baseline_results": {
                k: v for k, v in base_res.items() if k not in ("losses", "grad_norms")
            },
            "gates": gates,
            "passed": all_passed,
        },
        "passed": all_passed,
    }

    if source_hashes() != records["environment"]["source_sha256"]:
        records["passed"] = False
        records["error"] = "Source changed during execution."

    if proc_idx == 0:
        write_json(out / "metrics.json", records)
        np.savez_compressed(
            out / "losses.npz",
            alg_losses=np.asarray(alg_res["losses"], dtype=np.float32),
            alg_grad_norms=np.asarray(alg_res["grad_norms"], dtype=np.float32),
            base_losses=np.asarray(base_res["losses"], dtype=np.float32),
            base_grad_norms=np.asarray(base_res["grad_norms"], dtype=np.float32),
        )
        print(f"\n{'='*20} Pilot study Summary {'='*20}", flush=True)
        for k, v in gates.items():
            print(f"Gate {k:25s}: {'PASS' if v['passed'] else 'FAIL'} | {v}", flush=True)
        print(f"\nOverall Pilot study Result: {'TPU PASS' if records['passed'] else 'TPU FAIL'}", flush=True)

    if is_multi_host:
        multihost_utils.sync_global_devices("pilot-complete")
        jax.distributed.shutdown()

    return 0 if records["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
