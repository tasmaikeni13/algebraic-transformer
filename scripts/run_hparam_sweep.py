#!/usr/bin/env python3
"""Tuning study: Systematic Hyperparameter Sweeping & Architecture Tuning on 16 TPU v4 Pod slice.

Executes the equal-budget multi-seed hyperparameter search protocol:
- 2 Architectures: AlgebraicTransformerLM and StandardTransformerLM (125M scale)
- 3 Random Seeds: Seed 42, Seed 43, Seed 44
- Token Budget: 600M tokens per run drawn from FineWeb-Edu
- Emits:
  - results/tuning/sweep_ledger.json
  - results/tuning/algebraic_optimal.json
  - results/tuning/baseline_optimal.json
  - results/tuning/metrics.json
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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

from src.model import AlgebraicTransformerLM, count_parameters
from src.baseline import StandardTransformerLM, _build_standard_rope
from src.optimizer import algebraic_adamw, ards_schedule
from src.attention import build_cayley_rotary_matrix
from src.dataset import ShardedTokenLoader
from src.mesh import compile_data_parallel_step, create_tpu_mesh, ModelSharding
from scripts.tuning_records import environment, source_hashes, write_json
from scripts.tuning_experiments import (
    HparamConfig,
    load_hparam_candidates,
    select_best_candidate,
    training_steps_for_budget,
    get_125m_algebraic_config,
    get_125m_baseline_config,
    create_cosine_schedule,
    train_step_algebraic_fn,
    train_step_baseline_fn,
    evaluate_perplexity_fast,
)
from scripts.audit_primitives import source_audit


def _host_tree(tree):
    """Copy a replicated JAX pytree to plain NumPy arrays for checkpointing."""
    return jax.tree_util.tree_map(
        lambda value: np.asarray(value.addressable_data(0))
        if hasattr(value, "addressable_data") else np.asarray(value),
        tree,
    )


def save_training_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically serialize a trusted local training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_training_checkpoint(path: Path) -> Dict[str, Any]:
    """Load a checkpoint created by ``save_training_checkpoint``."""
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def audit_tuning_ast() -> Dict[str, Any]:
    """Audit production files in the algebraic stack for zero transcendentals."""
    files_to_audit = [
        "src/model.py",
        "src/primitives.py",
        "src/attention.py",
        "src/loss.py",
        "src/optimizer.py",
        "src/mesh.py",
        "src/kernels/pallas_afa.py",
        "src/kernels/pallas_oace.py",
    ]
    violations = {}
    for rel_path in files_to_audit:
        full = ROOT / rel_path
        if full.exists():
            v = source_audit(full.read_text())
            if v:
                violations[rel_path] = v
    return {
        "audited_files": files_to_audit,
        "violations": violations,
        "passed": len(violations) == 0,
    }


def run_sweep_arm(
    architecture: str,
    candidate_name: str,
    seed: int,
    hparams: HparamConfig,
    token_path: Path,
    valid_path: Path,
    total_tokens: int,
    batch_size: int,
    seq_len: int,
    mesh: Any,
    log_every: int = 50,
    is_multi_host: bool = False,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_every_tokens: Optional[int] = None,
    resume_checkpoint: Optional[Path] = None,
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
    max_checkpoints: int = 2,
) -> Dict[str, Any]:
    """Runs a single 600M token pretraining sweep arm."""
    proc_idx = jax.process_index()
    num_procs = jax.process_count()
    is_alg = (architecture == "algebraic")

    if proc_idx == 0:
        print(
            f"\n{'='*20} Sweep Arm: {architecture.upper()} | {candidate_name} | Seed {seed} {'='*20}\n"
            f"  Hyperparameters: lr={hparams.learning_rate}, warmup={hparams.warmup_steps}, "
            f"wd={hparams.weight_decay}, beta1={hparams.beta1}, beta2={hparams.beta2}, "
            f"sink={hparams.sink_omega}, gamma={hparams.gamma}\n"
            f"  Requested Budget: {total_tokens:,} tokens",
            flush=True,
        )

    tokens_per_step = batch_size * seq_len
    total_steps, actual_tokens = training_steps_for_budget(total_tokens, tokens_per_step)
    if proc_idx == 0:
        print(
            f"  Full-batch execution: {total_steps:,} steps, {actual_tokens:,} tokens "
            f"(+{actual_tokens - total_tokens:,} bounded overrun)",
            flush=True,
        )

    loader = ShardedTokenLoader(
        token_path=token_path,
        batch_size=batch_size,
        seq_len=seq_len,
        process_index=proc_idx,
        process_count=num_procs,
        seed=seed,
    )

    valid_tokens = np.load(valid_path, mmap_mode="r")

    accum_steps = 16
    assert batch_size % accum_steps == 0, f"batch_size {batch_size} must be divisible by accum_steps {accum_steps}"
    micro_batch_size = batch_size // accum_steps
    local_micro_batch = loader.local_batch_size // accum_steps
    global_batch_shape = (accum_steps, micro_batch_size, seq_len)

    sharding = ModelSharding(mesh)
    from jax.sharding import NamedSharding, PartitionSpec as P
    data_sharding = NamedSharding(mesh, P(None, ('data', 'fsdp', 'model'), None))

    # Initialize model
    if is_alg:
        model_cfg = get_125m_algebraic_config(sink_omega=hparams.sink_omega, gamma=hparams.gamma)
        model = AlgebraicTransformerLM(model_cfg)
        rotary = build_cayley_rotary_matrix(model.head_dim, seq_len, dtype=model_cfg.dtype)
        rotary_dev = jax.device_put(rotary, sharding.replicated)

        schedule_fn = ards_schedule(
            learning_rate=hparams.learning_rate,
            warmup_steps=hparams.warmup_steps,
            decay_steps=int(total_steps * 0.8),
        )
        decay_mask = lambda p: jax.tree_util.tree_map(lambda x: x.ndim >= 2, p)
        optimizer_tx = algebraic_adamw(
            learning_rate=schedule_fn,
            beta1=hparams.beta1,
            beta2=hparams.beta2,
            weight_decay=hparams.weight_decay,
            mask=decay_mask,
        )
        step_fn = train_step_algebraic_fn(
            model=model,
            optimizer_tx=optimizer_tx,
            rotary_params=rotary_dev,
            max_grad_norm=hparams.max_grad_norm,
            accum_steps=accum_steps,
            data_axis_names=mesh.axis_names,
        )
    else:
        model_cfg = get_125m_baseline_config()
        model = StandardTransformerLM(model_cfg)
        cos_angles, sin_angles = _build_standard_rope(model.head_dim, seq_len)
        cos_dev = jax.device_put(cos_angles, sharding.replicated)
        sin_dev = jax.device_put(sin_angles, sharding.replicated)

        schedule_fn = create_cosine_schedule(
            learning_rate=hparams.learning_rate,
            warmup_steps=hparams.warmup_steps,
            total_steps=total_steps,
            min_lr=hparams.min_lr,
        )
        decay_mask = lambda p: jax.tree_util.tree_map(lambda x: x.ndim >= 2, p)
        optimizer_tx = algebraic_adamw(
            learning_rate=schedule_fn,
            beta1=hparams.beta1,
            beta2=hparams.beta2,
            weight_decay=hparams.weight_decay,
            mask=decay_mask,
        )
        step_fn = train_step_baseline_fn(
            model=model,
            optimizer_tx=optimizer_tx,
            cos_angles=cos_dev,
            sin_angles=sin_dev,
            max_grad_norm=hparams.max_grad_norm,
            accum_steps=accum_steps,
            data_axis_names=mesh.axis_names,
        )

    # Initialize weights
    init_key = jax.random.PRNGKey(seed)
    params = model.init_params(init_key)
    opt_state = optimizer_tx.init(params)

    start_step = 0
    losses: List[float] = []
    grad_norms: List[float] = []
    nan_or_inf_count = 0
    loss_spike_count = 0
    prev_loss = None
    elapsed_before = 0.0
    if resume_checkpoint is not None:
        saved = load_training_checkpoint(resume_checkpoint)
        identity = (architecture, candidate_name, seed, total_steps)
        saved_identity = (
            saved.get("architecture"), saved.get("candidate"),
            saved.get("seed"), saved.get("total_steps"),
        )
        if saved_identity != identity or saved.get("hparams") != asdict(hparams):
            raise ValueError(
                f"checkpoint identity/config mismatch: expected {identity}, got {saved_identity}"
            )
        if checkpoint_metadata is not None and saved.get("metadata") != checkpoint_metadata:
            raise ValueError("checkpoint source metadata does not match the current run")
        params = saved["params"]
        opt_state = saved["opt_state"]
        start_step = int(saved["completed_steps"])
        losses = list(saved.get("losses", []))
        grad_norms = list(saved.get("grad_norms", []))
        nan_or_inf_count = int(saved.get("nan_or_inf_count", 0))
        loss_spike_count = int(saved.get("loss_spike_count", 0))
        prev_loss = saved.get("prev_loss")
        elapsed_before = float(saved.get("elapsed_seconds", 0.0))
        if not 0 <= start_step <= total_steps:
            raise ValueError(f"invalid checkpoint step {start_step} for {total_steps} steps")
        if proc_idx == 0:
            print(f"  Resuming from step {start_step:,}: {resume_checkpoint}", flush=True)

    params = jax.device_put(params, sharding.replicated)
    opt_state = jax.device_put(opt_state, sharding.replicated)

    # Keep Mosaic/Pallas calls inside a per-device program and explicitly
    # average gradients in the step function before optimizer updates.
    jitted_step = compile_data_parallel_step(
        step_fn,
        mesh,
        P(None, ("data", "fsdp", "model"), None),
    )

    # Compile with step 0
    x_init_np, y_init_np = loader.get_batch(start_step)
    x_init_local = x_init_np.reshape(accum_steps, local_micro_batch, seq_len)
    y_init_local = y_init_np.reshape(accum_steps, local_micro_batch, seq_len)
    x_init = jax.make_array_from_process_local_data(data_sharding, x_init_local, global_batch_shape)
    y_init = jax.make_array_from_process_local_data(data_sharding, y_init_local, global_batch_shape)
    # Warmup JIT compilation on step 0 data without mutating initial training parameters
    warmup_params, warmup_opt, _ = jitted_step(params, opt_state, x_init, y_init)
    jax.block_until_ready(warmup_params)
    del warmup_params, warmup_opt

    step_times: List[float] = []

    checkpoint_interval = None
    if checkpoint_dir is not None:
        if checkpoint_every_tokens is None or checkpoint_every_tokens <= 0:
            raise ValueError("checkpoint_every_tokens must be positive when checkpoint_dir is set")
        checkpoint_interval, _ = training_steps_for_budget(
            checkpoint_every_tokens, tokens_per_step
        )
        if max_checkpoints <= 0:
            raise ValueError("max_checkpoints must be positive")

    t_start = time.perf_counter()

    for step in range(start_step, total_steps):
        t0 = time.perf_counter()
        x_np, y_np = loader.get_batch(step)
        x_local = x_np.reshape(accum_steps, local_micro_batch, seq_len)
        y_local = y_np.reshape(accum_steps, local_micro_batch, seq_len)
        x_jax = jax.make_array_from_process_local_data(data_sharding, x_local, global_batch_shape)
        y_jax = jax.make_array_from_process_local_data(data_sharding, y_local, global_batch_shape)

        params, opt_state, metrics = jitted_step(params, opt_state, x_jax, y_jax)
        loss_val = float(jax.device_get(metrics["loss"]))
        grad_norm_val = float(jax.device_get(metrics["grad_norm"]))
        t1 = time.perf_counter()

        step_duration = t1 - t0
        step_times.append(step_duration)

        if not (math.isfinite(loss_val) and math.isfinite(grad_norm_val)):
            nan_or_inf_count += 1

        # Monitor steady-state loss spikes (step >= warmup, excluding initial warmup)
        warmup_cutoff = max(10, hparams.warmup_steps)
        if step >= warmup_cutoff and prev_loss is not None and (loss_val - prev_loss) > 1.5:
            loss_spike_count += 1
        prev_loss = loss_val

        losses.append(loss_val)
        grad_norms.append(grad_norm_val)

        if (step + 1) % log_every == 0 and proc_idx == 0:
            avg_time = np.mean(step_times[-log_every:])
            cur_thru = tokens_per_step / avg_time
            print(
                f"[{architecture.capitalize():9s}|Seed {seed}] Step {step+1:4d}/{total_steps:4d} | "
                f"Loss: {loss_val:7.4f} | Grad: {grad_norm_val:.3f} | "
                f"Throughput: {cur_thru:,.0f} tok/s | Step: {avg_time*1000:.1f}ms",
                flush=True,
            )

        should_checkpoint = (
            checkpoint_interval is not None
            and ((step + 1) % checkpoint_interval == 0 or step + 1 == total_steps)
        )
        if should_checkpoint:
            checkpoint_path = Path(checkpoint_dir) / f"step-{step + 1:06d}.pkl"
            save_training_checkpoint(checkpoint_path, {
                "format_version": 1,
                "metadata": checkpoint_metadata,
                "architecture": architecture,
                "candidate": candidate_name,
                "seed": seed,
                "hparams": asdict(hparams),
                "total_steps": total_steps,
                "completed_steps": step + 1,
                "params": _host_tree(params),
                "opt_state": _host_tree(opt_state),
                "losses": losses,
                "grad_norms": grad_norms,
                "nan_or_inf_count": nan_or_inf_count,
                "loss_spike_count": loss_spike_count,
                "prev_loss": prev_loss,
                "elapsed_seconds": elapsed_before + time.perf_counter() - t_start,
            })
            older = sorted(Path(checkpoint_dir).glob("step-*.pkl"))[:-max_checkpoints]
            for old_checkpoint in older:
                old_checkpoint.unlink()
            if proc_idx == 0:
                print(f"  Saved checkpoint: {checkpoint_path}", flush=True)
            if is_multi_host:
                multihost_utils.sync_global_devices(
                    f"checkpoint-{architecture}-{candidate_name}-{seed}-{step + 1}"
                )

    jax.block_until_ready(params)
    total_duration = elapsed_before + time.perf_counter() - t_start
    if is_multi_host:
        host_durations = np.asarray(
            multihost_utils.process_allgather(np.asarray(total_duration))
        )
        total_duration = float(host_durations.max())
    overall_throughput = (total_steps * tokens_per_step) / total_duration

    params_eval = jax.tree_util.tree_map(
        lambda p: jnp.asarray(p.addressable_data(0)) if hasattr(p, "addressable_data") else p,
        params,
    )

    if proc_idx == 0:
        print(f"[{architecture.capitalize()}|Seed {seed}] Evaluating validation perplexity on held-out FineWeb-Edu...", flush=True)
    # All controllers execute validation in the same program order.  Running
    # evaluation only on controller 0 can deadlock the other hosts when they
    # enter the next distributed training compilation.
    ppl, val_loss = evaluate_perplexity_fast(
        model=model,
        params=params_eval,
        valid_tokens=valid_tokens,
        seq_len=seq_len,
        batch_size=4,
        num_eval_batches=10,
        is_algebraic=is_alg,
        rotary_or_angles=None,
    )
    if proc_idx == 0:
        print(f"[{architecture.capitalize()}|Seed {seed}] Result: Val Loss = {val_loss:.4f}, Perplexity = {ppl:.2f}", flush=True)

    probe = jnp.asarray(
        np.asarray(valid_tokens[:seq_len], dtype=np.int32)[None, :]
    )
    moment_fn = jax.jit(
        lambda current_params, current_tokens: model.normalization_second_moments(
            current_params, current_tokens
        )
    )
    normalization_moments = np.asarray(
        jax.device_get(moment_fn(params_eval, probe)), dtype=np.float64
    ).tolist()

    local_peak_hbm = 0
    memory_stats_available = False
    for device in jax.local_devices():
        try:
            stats = device.memory_stats()
        except (AttributeError, NotImplementedError):
            stats = None
        if stats:
            memory_stats_available = True
            local_peak_hbm = max(
                local_peak_hbm,
                int(stats.get("peak_bytes_in_use", stats.get("peak_pool_bytes", 0))),
            )
    if is_multi_host:
        peak_values = np.asarray(
            multihost_utils.process_allgather(np.asarray(local_peak_hbm))
        )
        availability = np.asarray(
            multihost_utils.process_allgather(np.asarray(memory_stats_available))
        )
        local_peak_hbm = int(peak_values.max())
        memory_stats_available = bool(availability.all())

    if is_multi_host:
        multihost_utils.sync_global_devices(f"{architecture}-{candidate_name}-{seed}-done")

    run_record = {
        "architecture": architecture,
        "candidate": candidate_name,
        "seed": seed,
        "hparams": asdict(hparams),
        "requested_tokens": total_tokens,
        "total_tokens": actual_tokens,
        "token_overrun": actual_tokens - total_tokens,
        "token_budget_satisfied": actual_tokens >= total_tokens,
        "total_steps": total_steps,
        "resumed_from_step": start_step,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "final_train_loss": losses[-1] if losses else None,
        "validation_loss": val_loss,
        "validation_perplexity": ppl,
        "peak_gradient_norm": float(np.max(grad_norms)) if grad_norms else 0.0,
        "nan_or_inf_count": nan_or_inf_count,
        "loss_spike_count": loss_spike_count,
        "throughput_tokens_sec": overall_throughput,
        "elapsed_seconds": total_duration,
        "normalization_second_moments": normalization_moments,
        "normalization_second_moment_min": float(min(normalization_moments)),
        "normalization_second_moment_max": float(max(normalization_moments)),
        "peak_hbm_bytes": local_peak_hbm if memory_stats_available else None,
        "memory_stats_available": memory_stats_available,
        "losses_sample": losses[::max(1, len(losses)//20)],
    }
    return run_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tuning")
    parser.add_argument("--tokens-per-run", type=int, default=600_000_000, help="Tokens evaluated per sweep run")
    parser.add_argument("--batch-size", type=int, default=512, help="Global batch size in sequences")
    parser.add_argument("--seq-len", type=int, default=2048, help="Context length")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "configs/tuning_candidates.json",
        help="Preregistered candidate matrix; every candidate is evaluated on every seed",
    )
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    if sorted(args.seeds) != [42, 43, 44] or len(args.seeds) != 3:
        raise ValueError("Tuning study evidence requires exactly seeds 42, 43, and 44")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Initialize distributed TPU if running in multi-host mode
    is_multi_host = False
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        try:
            jax.distributed.initialize(initialization_timeout=120)
            is_multi_host = True
        except Exception as e:
            print(f"Warning: jax.distributed.initialize failed or already initialized: {e}", flush=True)

    devices = jax.devices()
    platform = devices[0].platform
    proc_idx = jax.process_index()
    num_procs = jax.process_count()
    if platform != "tpu" or len(devices) != 16 or num_procs != 4:
        raise RuntimeError(
            "Tuning study requires four processes and 16 TPU devices; "
            f"found {num_procs} processes and {len(devices)} {platform} devices"
        )
    if proc_idx == 0:
        print(f"Hardware Platform: {platform}, Total Devices: {len(devices)}, Processes: {num_procs}", flush=True)

    mesh = create_tpu_mesh(devices=devices)
    if proc_idx == 0:
        print(f"Created distributed mesh: {mesh}", flush=True)

    # Verify AST Zero-Transcendental audit
    ast_res = audit_tuning_ast()
    if proc_idx == 0:
        print(f"AST Zero-Transcendental Audit: {'PASS' if ast_res['passed'] else 'FAIL'}", flush=True)
    if not ast_res["passed"]:
        raise RuntimeError(f"AST audit failed: {ast_res['violations']}")

    # Check datasets
    sweep_data = args.data_dir / "fineweb_sweep_600M.npy"
    if not sweep_data.exists():
        sweep_data = args.data_dir / "fineweb_train_2_5B.npy"
    valid_data = args.data_dir / "fineweb_valid.npy"

    if not (sweep_data.exists() and valid_data.exists()):
        raise FileNotFoundError(
            f"Dataset files not found in {args.data_dir}. Run scripts/prepare_fineweb_edu.py first."
        )

    candidates = load_hparam_candidates(args.candidates)
    expected_runs = sum(len(rows) for rows in candidates.values()) * len(args.seeds)
    all_run_records = []

    # Evaluate every preregistered candidate on the same seed set and token budget.
    architectures = ["algebraic", "baseline"]
    for arch in architectures:
        for candidate_name, hp in candidates[arch]:
            for seed in args.seeds:
                record = run_sweep_arm(
                    architecture=arch,
                    candidate_name=candidate_name,
                    seed=seed,
                    hparams=hp,
                    token_path=sweep_data,
                    valid_path=valid_data,
                    total_tokens=args.tokens_per_run,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    mesh=mesh,
                    log_every=args.log_every,
                    is_multi_host=is_multi_host,
                )
                all_run_records.append(record)

    metrics_record = None
    if proc_idx == 0:
        # Save structured artifacts
        sweep_ledger = {
            "title": "Tuning study Equal-Budget Hyperparameter Sweep Ledger",
            "protocol_version": 2,
            "token_budget_per_run": args.tokens_per_run,
            "total_tokens_evaluated": sum(r["total_tokens"] for r in all_run_records),
            "seeds": args.seeds,
            "candidate_count": sum(len(rows) for rows in candidates.values()),
            "expected_runs": expected_runs,
            "runs": all_run_records,
        }
        write_json(out / "sweep_ledger.json", sweep_ledger)

        # Select and freeze only candidates with complete, stable multi-seed evidence.
        alg_hparams, alg_runs, alg_std_pct = select_best_candidate(
            all_run_records, "algebraic", args.seeds
        )
        base_hparams, base_runs, base_std_pct = select_best_candidate(
            all_run_records, "baseline", args.seeds
        )
        write_json(out / "algebraic_optimal.json", asdict(alg_hparams))
        write_json(out / "baseline_optimal.json", asdict(base_hparams))

        # Compute summary metrics for the selected candidates across seeds.

        alg_val_losses = [r["validation_loss"] for r in alg_runs]
        base_val_losses = [r["validation_loss"] for r in base_runs]
        alg_ppls = [r["validation_perplexity"] for r in alg_runs]
        base_ppls = [r["validation_perplexity"] for r in base_runs]

        alg_mean_loss = float(np.mean(alg_val_losses))
        base_mean_loss = float(np.mean(base_val_losses))

        alg_mean_ppl = float(np.mean(alg_ppls))
        base_mean_ppl = float(np.mean(base_ppls))
        ppl_ratio = alg_mean_ppl / base_mean_ppl if base_mean_ppl > 0 else 999.0

        selected_runs = alg_runs + base_runs
        max_grad = max(r["peak_gradient_norm"] for r in selected_runs)
        total_nans = sum(r["nan_or_inf_count"] for r in selected_runs)
        total_spikes = sum(r["loss_spike_count"] for r in selected_runs)
        token_budget_satisfied = all(r["token_budget_satisfied"] for r in all_run_records)
        algebraic_moments = [
            value
            for run in alg_runs
            for value in run["normalization_second_moments"]
        ]
        normalization_passed = bool(
            algebraic_moments
            and min(algebraic_moments) >= 0.8
            and max(algebraic_moments) <= 1.3
        )

        summary = {
            "completed_runs": len(all_run_records),
            "expected_runs": expected_runs,
            "candidates_evaluated": sum(len(rows) for rows in candidates.values()),
            "algebraic_candidates_evaluated": len(candidates["algebraic"]),
            "baseline_candidates_evaluated": len(candidates["baseline"]),
            "seed_count": len(args.seeds),
            "requested_tokens_per_run": args.tokens_per_run,
            "minimum_actual_tokens_per_run": min(r["total_tokens"] for r in all_run_records),
            "maximum_actual_tokens_per_run": max(r["total_tokens"] for r in all_run_records),
            "maximum_token_overrun": max(r["token_overrun"] for r in all_run_records),
            "tokens_per_step": args.batch_size * args.seq_len,
            "token_budget_satisfied": token_budget_satisfied,
            "selected_algebraic_candidate": alg_runs[0]["candidate"],
            "selected_baseline_candidate": base_runs[0]["candidate"],
            "algebraic_mean_val_loss": alg_mean_loss,
            "algebraic_seed_std_pct": alg_std_pct,
            "algebraic_mean_ppl": alg_mean_ppl,
            "baseline_mean_val_loss": base_mean_loss,
            "baseline_seed_std_pct": base_std_pct,
            "baseline_mean_ppl": base_mean_ppl,
            "mean_perplexity_ratio": ppl_ratio,
            "peak_gradient_norm": max_grad,
            "nan_or_inf_count": total_nans,
            "loss_spike_count": total_spikes,
            "all_candidate_nan_or_inf_count": sum(
                r["nan_or_inf_count"] for r in all_run_records
            ),
            "all_candidate_loss_spike_count": sum(
                r["loss_spike_count"] for r in all_run_records
            ),
            "algebraic_normalization_second_moment_min": min(algebraic_moments),
            "algebraic_normalization_second_moment_max": max(algebraic_moments),
            "normalization_second_moment_passed": normalization_passed,
            "peak_hbm_bytes": max(
                (r["peak_hbm_bytes"] or 0) for r in all_run_records
            ) or None,
        }

        is_pass = (
            len(all_run_records) == expected_runs
            and len(candidates["algebraic"]) >= 2
            and len(candidates["baseline"]) >= 2
            and token_budget_satisfied
            and platform == "tpu"
            and len(devices) == 16
            and num_procs == 4
            and ppl_ratio <= 1.08
            and total_nans == 0
            and total_spikes == 0
            and max_grad <= 5.0
            and normalization_passed
            and alg_std_pct < 2.0
            and base_std_pct < 2.0
            and ast_res["passed"]
        )

        metrics_record = {
            "study": "tuning",
            "gate_version": 1,
            "status": "PASS" if is_pass else "FAIL",
            "passed": is_pass,
            "environment": environment(),
            "hardware": {
                "platform": platform,
                "device_count": len(devices),
                "process_count": num_procs,
            },
            "sweep_summary": summary,
            "ast_audit": ast_res,
            "artifacts": {
                name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                for name in (
                    "sweep_ledger.json",
                    "algebraic_optimal.json",
                    "baseline_optimal.json",
                )
            },
        }
        write_json(out / "metrics.json", metrics_record)

        print("\n" + "="*20 + " Tuning study Sweep Summary " + "="*20)
        print(f"Runs Completed: {len(all_run_records)}/{expected_runs}")
        print(f"Algebraic PPL:  {alg_mean_ppl:.2f} (Loss std: {alg_std_pct:.2f}%)")
        print(f"Baseline PPL:   {base_mean_ppl:.2f} (Loss std: {base_std_pct:.2f}%)")
        print(f"PPL Ratio:      {ppl_ratio:.4f} (Threshold <= 1.08)")
        print(f"Peak Grad Norm: {max_grad:.4f} (Threshold <= 5.0)")
        print(f"NaN / Inf:      {total_nans}")
        print(f"Loss Spikes:    {total_spikes}")
        print(f"Overall Result: {'PASS' if metrics_record['passed'] else 'FAIL'}")

    if is_multi_host:
        multihost_utils.sync_global_devices("tuning-complete")
        jax.distributed.shutdown()

    return 0 if (metrics_record is None or metrics_record.get("passed", False)) else 1


if __name__ == "__main__":
    main()
