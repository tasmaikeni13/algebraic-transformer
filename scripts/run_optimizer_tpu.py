#!/usr/bin/env python3
"""Four-host Optimizer study execution and synchronized 16-chip TPU v4 benchmarks for Algebraic AdamW & ARDS."""
import os
os.environ['JAX_PLATFORMS'] = 'tpu,cpu'
os.environ['JAX_ENABLE_X64'] = '0'
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils, multihost_utils as mh
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from src.optimizer import algebraic_adamw, ards_schedule
from tests.reference_optimizer import (
    adamw_fp64_reference,
    ards_schedule_fp64,
    cosine_schedule_fp64,
    quadratic_fp64,
    quadratic_grad_fp64,
)
from scripts.primitives_experiments import summary
from scripts.optimizer_records import environment, source_hashes, write_json


def parity(place):
    """Verify Algebraic AdamW numerical parity on TPU against float64 CPU oracle."""
    rng = np.random.default_rng(542 + jax.process_index())
    rows = []
    for dtype in (jnp.float32, jnp.bfloat16):
        tol = 2.0e-4 if dtype == jnp.float32 else 0.04
        for shape in ((512, 512), (1024, 1024), (2048, 1024), (4096, 512)):
            w_host = rng.normal(size=shape).astype(np.float32)
            g_host = rng.normal(size=shape).astype(np.float32)

            w = place(w_host).astype(dtype)
            g = place(g_host).astype(dtype)

            opt = algebraic_adamw(learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2)
            state = opt.init(w)

            updates, new_state = opt.update(g, state, w)
            updates = jax.block_until_ready(updates)

            local = []
            for ws, gs, us in zip(w.addressable_shards, g.addressable_shards, updates.addressable_shards):
                wa = np.asarray(ws.data, dtype=np.float64)
                ga = np.asarray(gs.data, dtype=np.float64)
                ua = np.asarray(us.data, dtype=np.float64)

                _, u_ref, _, _ = adamw_fp64_reference(
                    wa, ga, np.zeros_like(wa), np.zeros_like(wa), t=1, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2
                )

                err = float(np.max(np.abs(ua - u_ref) / (1.0 + np.abs(u_ref))))
                finite_check = float(np.all(np.isfinite(ua)))
                local.append([err, finite_check])

            arr = np.asarray(mh.process_allgather(np.array(local))).reshape(-1, 2)
            max_err = float(arr[:, 0].max())
            all_finite = bool(np.all(arr[:, 1] == 1.0))
            passed = bool(max_err <= tol and all_finite)
            rows.append({
                "dtype": str(jnp.dtype(dtype)),
                "shape": list(shape),
                "max_err": max_err,
                "all_finite": all_finite,
                "tolerance": tol,
                "passed": passed,
            })
    return {"rows": rows, "passed": all(r["passed"] for r in rows)}


def quadratic_sweep_tpu(place):
    """Run ill-conditioned quadratic sweep on TPU v4 hardware."""
    rng = np.random.default_rng(642 + jax.process_index())
    trials = 10000
    dim = 8
    log_kappas = rng.uniform(2.0, 6.0, size=trials)
    kappas = 10.0 ** log_kappas

    diag_A = np.zeros((trials, dim), dtype=np.float32)
    for i in range(trials):
        diag_A[i] = np.geomspace(1.0, kappas[i], dim).astype(np.float32)

    x0 = rng.normal(size=(trials, dim)).astype(np.float32)

    diag_A_tpu = place(diag_A)
    x_tpu = place(x0)

    init_loss = 0.5 * jnp.sum(diag_A_tpu * (x_tpu ** 2), axis=-1)

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr = 0.1

    def step_fn(carry, step_idx):
        x, m, v, b1_p, b2_p = carry
        g = diag_A_tpu * x
        b1_p = b1_p * beta1
        b2_p = b2_p * beta2
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - b1_p)
        v_hat = v / (1.0 - b2_p)
        u = m_hat / (jnp.sqrt(v_hat) + eps)
        x = x - lr * u
        return (x, m, v, b1_p, b2_p), None

    init_carry = (x_tpu, jnp.zeros_like(x_tpu), jnp.zeros_like(x_tpu), 1.0, 1.0)
    (x_final, _, _, _, _), _ = jax.lax.scan(step_fn, init_carry, jnp.arange(1, 301))

    final_loss = 0.5 * jnp.sum(diag_A_tpu * (x_final ** 2), axis=-1)
    reduction = 1.0 - (final_loss / init_loss)
    reduction = jax.block_until_ready(reduction)

    local_reds = []
    for shard in reduction.addressable_shards:
        ra = np.asarray(shard.data)
        local_reds.append([float(np.min(ra)), float(np.mean(ra))])

    arr = np.asarray(mh.process_allgather(np.array(local_reds))).reshape(-1, 2)
    min_red = float(arr[:, 0].min())
    mean_red = float(arr[:, 1].mean())
    passed = bool(min_red > 0.9999)

    return {
        "trials": trials,
        "min_reduction": min_red,
        "mean_reduction": mean_red,
        "passed": passed,
    }


def benchmarks(place):
    """Benchmark Algebraic AdamW with ARDS vs AdamW with Cosine Annealing on 16 TPU v4 chips."""
    rows = []
    raw = {}
    hlo = {}

    for dtype in (jnp.float32, jnp.bfloat16):
        for shape in ((1024, 1024), (2048, 2048)):
            rng = np.random.default_rng(742 + jax.process_index())
            w_host = rng.normal(size=shape).astype(np.float32)
            g_host = rng.normal(size=shape).astype(np.float32)

            w = place(w_host).astype(dtype)
            g = place(g_host).astype(dtype)

            # Define ARDS and Cosine step functions
            ards_sched = ards_schedule(learning_rate=1e-3, warmup_steps=100, decay_steps=1000, alpha=1.0)
            cos_sched = lambda t: 1e-4 + 0.5 * (1e-3 - 1e-4) * (1.0 + jnp.cos(jnp.pi * jnp.clip(t / 1000.0, 0.0, 1.0)))

            opt_ards = algebraic_adamw(learning_rate=ards_sched, weight_decay=1e-2)
            opt_cos = algebraic_adamw(learning_rate=cos_sched, weight_decay=1e-2)

            state_ards = opt_ards.init(w)
            state_cos = opt_cos.init(w)

            def step_ards(params, state, grads):
                updates, new_state = opt_ards.update(grads, state, params)
                new_params = params + updates
                return new_params, new_state

            def step_cos(params, state, grads):
                updates, new_state = opt_cos.update(grads, state, params)
                new_params = params + updates
                return new_params, new_state

            lower_ards = jax.jit(step_ards).lower(w, state_ards, g)
            comp_ards = lower_ards.compile()

            lower_cos = jax.jit(step_cos).lower(w, state_cos, g)
            comp_cos = lower_cos.compile()

            if shape == (2048, 2048):
                hlo[f"{str(jnp.dtype(dtype))}_ards_step"] = lower_ards.as_text()
                hlo[f"{str(jnp.dtype(dtype))}_cos_step"] = lower_cos.as_text()

            calls = {
                "ards_step": (comp_ards, (w, state_ards, g)),
                "cos_step": (comp_cos, (w, state_cos, g)),
            }

            # Warmup
            for _ in range(10):
                for fn, args in calls.values():
                    out_p, out_s = fn(*args)
                    jax.block_until_ready(out_p)
                    jax.block_until_ready(out_s.mu)

            latencies = {key: [] for key in calls}
            perm_rng = np.random.default_rng(846)
            for rep in range(100):
                for key in perm_rng.permutation(list(calls)):
                    fn, args = calls[key]
                    mh.sync_global_devices(f"bench-opt-{dtype}-{shape[0]}-{rep}-{key}")
                    start = time.perf_counter_ns()
                    out_p, out_s = fn(*args)
                    jax.block_until_ready(out_p)
                    jax.block_until_ready(out_s.mu)
                    elapsed = (time.perf_counter_ns() - start) * 1e-9
                    latencies[key].append(float(np.asarray(mh.process_allgather(np.array(elapsed))).max()))

            med_ards = float(np.median(latencies["ards_step"]))
            med_cos = float(np.median(latencies["cos_step"]))
            ratio = float(med_cos / med_ards)  # Higher is better for ARDS

            gates = {
                "throughput": {
                    "ratio": ratio,
                    "minimum": 0.90,
                }
            }
            row = {
                "dtype": str(jnp.dtype(dtype)),
                "shape": list(shape),
                "repetitions": 100,
                "ards_latency_ms": med_ards * 1e3,
                "cos_latency_ms": med_cos * 1e3,
                "throughput_ratio": ratio,
                "gates": gates,
                "passed": ratio >= 0.90,
            }
            rows.append(row)
            raw[f"{dtype}_{shape[0]}x{shape[1]}"] = {k: summary(v) for k, v in latencies.items()}
            if jax.process_index() == 0:
                print("TPU optimizer benchmark", row["dtype"], f"shape={shape}", f"ratio={ratio:.3f}", flush=True)

    return {"rows": rows, "passed": all(r["passed"] for r in rows)}, raw, hlo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/optimizer/tpu")
    args = parser.parse_args()

    jax.distributed.initialize(initialization_timeout=120)
    devices = jax.devices()
    assert len(devices) == 16 and jax.process_count() == 4 and all("TPU v4" in d.device_kind for d in devices)

    mesh = Mesh(mesh_utils.create_device_mesh((2, 2, 4), devices, allow_split_physical_axes=True), ("data", "fsdp", "model"))
    sharding = NamedSharding(mesh, P(("data", "fsdp", "model")))
    place = lambda a: jax.make_array_from_process_local_data(sharding, a)

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    r = {
        "study": 5,
        "environment": environment(),
        "command": sys.argv,
        "device_count": len(devices),
        "process_count": jax.process_count(),
        "mesh_shape": dict(mesh.shape),
        "devices": [{"id": d.id, "kind": d.device_kind, "process": d.process_index, "coords": list(d.coords)} for d in devices],
    }

    r["parity"] = parity(place)
    if jax.process_index() == 0:
        print("TPU parity:", r["parity"]["passed"], flush=True)

    r["quadratic_sweep"] = quadratic_sweep_tpu(place)
    if jax.process_index() == 0:
        print("TPU quadratic sweep:", r["quadratic_sweep"]["passed"], flush=True)

    r["benchmarks"], latencies, hlo = benchmarks(place)
    if jax.process_index() == 0:
        print("TPU benchmarks:", r["benchmarks"]["passed"], flush=True)

    hlo_rows = []
    for name, content in hlo.items():
        lower = content.lower()
        is_ards = "ards_step" in name
        row = {
            "name": name,
            "raw_sqrt_count": lower.count("stablehlo.sqrt "),
            "rsqrt_count": lower.count("stablehlo.rsqrt "),
        }
        row["passed"] = (not is_ards) or (
            row["raw_sqrt_count"] == 0 and row["rsqrt_count"] >= 2
        )
        hlo_rows.append(row)
    r["hlo_audit"] = {
        "rows": hlo_rows,
        "passed": bool(hlo_rows) and all(row["passed"] for row in hlo_rows),
    }

    r["passed"] = all(
        r[k]["passed"]
        for k in (
            "parity",
            "quadratic_sweep",
            "benchmarks",
            "hlo_audit",
        )
    )

    if source_hashes() != r["environment"]["source_sha256"]:
        r["passed"] = False
        r["source_changed"] = True

    if jax.process_index() == 0:
        write_json(out / "metrics.json", r)
        write_json(out / "latencies.json", latencies)
        for name, content in hlo.items():
            (out / f"{name}.mlir").write_text(content)
        print("TPU OPTIMIZER PASS" if r["passed"] else "TPU OPTIMIZER FAIL", flush=True)

    mh.sync_global_devices("optimizer-complete")
    jax.distributed.shutdown()
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
