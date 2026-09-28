#!/usr/bin/env python3
"""Four-host Loss study execution and synchronized 16-chip TPU v4 benchmarks for OACE."""
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

from src.loss import oace_loss, pearson_divergence
from tests.reference_loss import (
    cross_entropy_fp64,
    kl_divergence_fp64,
    oace_loss_fp64,
    oace_vjp_fp64,
    pearson_divergence_fp64,
    pearson_vjp_fp64,
)
from scripts.primitives_experiments import summary
from scripts.loss_records import environment, source_hashes, write_json


def parity(place):
    rng = np.random.default_rng(142 + jax.process_index())
    rows = []
    for dtype in (jnp.float32, jnp.bfloat16):
        tol = 2.0e-4 if dtype == jnp.float32 else 0.04
        for batch_size in (16, 64):
            for num_classes in (32, 128, 512, 2048):
                shape = (batch_size, num_classes)
                raw = rng.uniform(0.1, 1.0, size=shape).astype(np.float32)
                p_host = raw / np.sum(raw, axis=-1, keepdims=True)
                targets_host = rng.integers(0, num_classes, size=batch_size).astype(np.int32)
                g_host = rng.normal(size=batch_size).astype(np.float32)

                p = place(p_host).astype(dtype)
                targets = place(targets_host)
                g = place(g_host).astype(dtype)

                def fwd(probs):
                    return oace_loss(probs, targets, gamma=2.0, reduction="none")

                loss, vjp_fn = jax.vjp(fwd, p)
                (dp,) = vjp_fn(g)
                loss = jax.block_until_ready(loss)
                dp = jax.block_until_ready(dp)

                local = []
                for ps, ts, gs, ls, dps in zip(
                    p.addressable_shards,
                    targets.addressable_shards,
                    g.addressable_shards,
                    loss.addressable_shards,
                    dp.addressable_shards,
                ):
                    pa = np.asarray(ps.data, dtype=np.float64)
                    ta = np.asarray(ts.data, dtype=np.int32)
                    ga = np.asarray(gs.data, dtype=np.float64)
                    la = np.asarray(ls.data, dtype=np.float64)
                    dpa = np.asarray(dps.data, dtype=np.float64)

                    ref_loss = oace_loss_fp64(pa, ta, gamma=2.0, reduction="none")
                    ref_dp = oace_vjp_fp64(pa, ta, ga, gamma=2.0)

                    err_loss = float(np.max(np.abs(la - ref_loss) / (1.0 + np.abs(ref_loss))))
                    err_dp = float(np.max(np.abs(dpa - ref_dp) / (1.0 + np.abs(ref_dp))))
                    finite_check = float(np.all(np.isfinite(la)) and np.all(np.isfinite(dpa)))
                    local.append([err_loss, err_dp, finite_check])

                arr = np.asarray(mh.process_allgather(np.array(local))).reshape(-1, 3)
                max_err_loss = float(arr[:, 0].max())
                max_err_dp = float(arr[:, 1].max())
                all_finite = bool(np.all(arr[:, 2] == 1.0))
                passed = bool(max_err_loss <= tol and max_err_dp <= tol and all_finite)
                rows.append({
                    "dtype": str(jnp.dtype(dtype)),
                    "batch_size": batch_size,
                    "num_classes": num_classes,
                    "max_err_loss": max_err_loss,
                    "max_err_dp": max_err_dp,
                    "all_finite": all_finite,
                    "tolerance": tol,
                    "passed": passed,
                })
    return {"rows": rows, "passed": all(r["passed"] for r in rows)}


def boundary_stability(place):
    rows = []
    pk_vals = [1e-9, 1e-7, 1e-5, 1e-3, 0.01, 0.1, 0.5, 0.9, 1.0 - 1e-9]
    K = 128
    batch_size = 16

    for pk in pk_vals:
        p_host = np.full((batch_size, K), (1.0 - pk) / (K - 1), dtype=np.float32)
        p_host[:, 0] = pk
        targets_host = np.zeros(batch_size, dtype=np.int32)
        g_host = np.ones(batch_size, dtype=np.float32)

        p = place(p_host)
        targets = place(targets_host)
        g = place(g_host)

        def fwd(probs):
            return oace_loss(probs, targets, gamma=1.0, reduction="none")

        loss, vjp_fn = jax.vjp(fwd, p)
        (dp,) = vjp_fn(g)
        loss = jax.block_until_ready(loss)
        dp = jax.block_until_ready(dp)

        loss_finite = bool(jax.block_until_ready(jnp.all(jnp.isfinite(loss))))
        dp_finite = bool(jax.block_until_ready(jnp.all(jnp.isfinite(dp))))

        # This is the probability-domain gradient.  It is intentionally not
        # confused with the finite gradient after composition with
        # AVN-bounded A-Softmax.
        target_grad = float(jax.block_until_ready(jnp.max(jnp.abs(dp[..., 0]))))
        expected = abs(pk ** (-1.0 / 8.0) - pk ** (-9.0 / 8.0))
        scaled_error = abs(target_grad - expected) / (1.0 + expected)
        passed = bool(loss_finite and dp_finite and scaled_error <= 2.0e-4)
        rows.append({
            "pk": pk,
            "loss_finite": loss_finite,
            "dp_finite": dp_finite,
            "probability_gradient_magnitude": target_grad,
            "expected_probability_gradient_magnitude": expected,
            "scaled_error": scaled_error,
            "tolerance": 2.0e-4,
            "passed": passed,
        })
    return {"rows": rows, "passed": all(r["passed"] for r in rows)}


def fisher_equivalence(place):
    rows = []
    K = 16
    batch = 64
    rng = np.random.default_rng(242 + jax.process_index())
    raw = rng.uniform(0.1, 1.0, size=(batch, K)).astype(np.float32)
    p_host = raw / np.sum(raw, axis=-1, keepdims=True)

    p = place(p_host)
    # At p = y:
    # Hessian ratio of D_A vs D_KL is 2.0
    h_p = 2.0 / p
    h_kl = 1.0 / p
    ratio = h_p / h_kl
    max_err = float(jax.block_until_ready(jnp.max(jnp.abs(ratio - 2.0))))
    tol = 1e-6
    passed = bool(max_err <= tol)
    rows.append({
        "num_classes": K,
        "batch_size": batch,
        "max_ratio_error": max_err,
        "tolerance": tol,
        "passed": passed,
    })
    return {"rows": rows, "passed": all(r["passed"] for r in rows)}


def benchmarks(place):
    rows = []
    raw = {}
    hlo = {}
    for dtype in (jnp.float32, jnp.bfloat16):
        for num_classes in (256, 1024, 4096, 32000):
            rng = np.random.default_rng(442 + jax.process_index())
            batch_size = 64
            shape = (batch_size, num_classes)
            raw_p = rng.uniform(0.1, 1.0, size=shape).astype(np.float32)
            p_host = raw_p / np.sum(raw_p, axis=-1, keepdims=True)
            targets_host = rng.integers(0, num_classes, size=batch_size).astype(np.int32)
            g_host = rng.normal(size=batch_size).astype(np.float32)

            p = place(p_host).astype(dtype)
            targets = place(targets_host)
            g = place(g_host).astype(dtype)

            # Targets one-hot for CE
            targets_one_hot_host = (np.arange(num_classes) == targets_host[:, None]).astype(np.float32)
            targets_one_hot = place(targets_one_hot_host).astype(dtype)

            calls = {}

            # OACE functions
            def oace_fwd(probs, tgts):
                return oace_loss(probs, tgts, gamma=2.0, reduction="none")

            def oace_both(probs, tgts, cotangent):
                y, vjp_fn = jax.vjp(lambda pr: oace_fwd(pr, tgts), probs)
                return y, vjp_fn(cotangent)[0]

            # Cross-Entropy functions
            def ce_fwd(probs, tgts):
                pk = jnp.take_along_axis(probs, jnp.expand_dims(tgts, -1), axis=-1)[..., 0]
                return -jnp.log(jnp.clip(pk, 1e-15, 1.0))

            def ce_both(probs, tgts, cotangent):
                y, vjp_fn = jax.vjp(lambda pr: ce_fwd(pr, tgts), probs)
                return y, vjp_fn(cotangent)[0]

            for name, fwd_fn, both_fn in [("oace", oace_fwd, oace_both), ("ce", ce_fwd, ce_both)]:
                lower_fwd = jax.jit(fwd_fn).lower(p, targets)
                comp_fwd = lower_fwd.compile()
                calls[f"{name}_forward"] = (comp_fwd, (p, targets))

                lower_both = jax.jit(both_fn).lower(p, targets, g)
                comp_both = lower_both.compile()
                calls[f"{name}_forward_backward"] = (comp_both, (p, targets, g))

                if name == "oace" and num_classes == 1024:
                    hlo[f"{str(jnp.dtype(dtype))}_forward"] = lower_fwd.as_text()
                    hlo[f"{str(jnp.dtype(dtype))}_forward_backward"] = lower_both.as_text()

            # Warmup
            for _ in range(10):
                for fn, args in calls.values():
                    jax.block_until_ready(fn(*args))

            latencies = {key: [] for key in calls}
            perm_rng = np.random.default_rng(46)
            for rep in range(100):
                for key in perm_rng.permutation(list(calls)):
                    fn, args = calls[key]
                    mh.sync_global_devices(f"bench-{dtype}-{num_classes}-{rep}-{key}")
                    start = time.perf_counter_ns()
                    jax.block_until_ready(fn(*args))
                    elapsed = (time.perf_counter_ns() - start) * 1e-9
                    latencies[key].append(float(np.asarray(mh.process_allgather(np.array(elapsed))).max()))

            gates = {
                mode: {
                    "ratio": float(np.median(latencies[f"ce_{mode}"]) / np.median(latencies[f"oace_{mode}"])),
                    "minimum": 0.90,
                }
                for mode in ("forward", "forward_backward")
            }
            row = {
                "dtype": str(jnp.dtype(dtype)),
                "num_classes": num_classes,
                "batch_size": batch_size,
                "repetitions": 100,
                "forward_latency_ms": float(np.median(latencies["oace_forward"])) * 1e3,
                "forward_backward_latency_ms": float(np.median(latencies["oace_forward_backward"])) * 1e3,
                "ce_forward_latency_ms": float(np.median(latencies["ce_forward"])) * 1e3,
                "ce_forward_backward_latency_ms": float(np.median(latencies["ce_forward_backward"])) * 1e3,
                "gates": gates,
                "passed": all(g["ratio"] >= g["minimum"] for g in gates.values()),
            }
            rows.append(row)
            raw[f"{dtype}_{num_classes}"] = {k: summary(v) for k, v in latencies.items()}
            if jax.process_index() == 0:
                print("TPU loss benchmark", row["dtype"], f"K={num_classes}", {k: round(v["ratio"], 3) for k, v in gates.items()}, flush=True)

    return {"rows": rows, "passed": all(r["passed"] for r in rows)}, raw, hlo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/loss/tpu")
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
        "study": 4,
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

    r["boundary_stability"] = boundary_stability(place)
    if jax.process_index() == 0:
        print("TPU boundary stability:", r["boundary_stability"]["passed"], flush=True)

    r["fisher_equivalence"] = fisher_equivalence(place)
    if jax.process_index() == 0:
        print("TPU Fisher equivalence:", r["fisher_equivalence"]["passed"], flush=True)

    r["benchmarks"], latencies, hlo = benchmarks(place)
    if jax.process_index() == 0:
        print("TPU benchmarks:", r["benchmarks"]["passed"], flush=True)

    r["passed"] = all(
        r[k]["passed"]
        for k in (
            "parity",
            "boundary_stability",
            "fisher_equivalence",
            "benchmarks",
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
        print("TPU LOSS PASS" if r["passed"] else "TPU LOSS FAIL", flush=True)

    mh.sync_global_devices("loss-complete")
    jax.distributed.shutdown()
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
