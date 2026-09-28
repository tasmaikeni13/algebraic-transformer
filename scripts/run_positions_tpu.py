#!/usr/bin/env python3
"""Four-host Position study execution and synchronized 16-chip TPU v4 benchmarks for AGO."""
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

from src.attention import apply_ago_rotations, build_cayley_rotary_matrix
from tests.reference_ago import (
    apply_ago_rotations_fp64,
    build_cayley_rotary_matrix_fp64,
    cayley_rotation_matrix_fp64,
)
from scripts.primitives_experiments import summary
from scripts.positions_records import environment, source_hashes, write_json


def parity(place):
    rng = np.random.default_rng(142 + jax.process_index())
    rows = []
    for dtype in (jnp.float32, jnp.bfloat16):
        tol = 2.0e-4 if dtype == jnp.float32 else 0.04
        for length in (64, 128, 512, 2048, 4096):
            for head_dim in (32, 64):
                batch_size = 16
                num_heads = 4
                shape = (batch_size, length, num_heads, head_dim)
                q_host = rng.normal(size=shape).astype(np.float32)
                k_host = rng.normal(size=shape).astype(np.float32)
                g_host = rng.normal(size=shape).astype(np.float32)

                q = place(q_host).astype(dtype)
                k = place(k_host).astype(dtype)
                g = place(g_host).astype(dtype)

                rot = build_cayley_rotary_matrix(head_dim, length, dtype=dtype)

                def fwd(a, b):
                    return apply_ago_rotations(a, b, rotary_params=rot)

                (qr, kr), vjp_fn = jax.vjp(fwd, q, k)
                dq, dk = vjp_fn((g, g))
                qr = jax.block_until_ready(qr)
                kr = jax.block_until_ready(kr)
                dq = jax.block_until_ready(dq)
                dk = jax.block_until_ready(dk)

                local = []
                for qs, ks, gs, qrs, krs, dqs, dks in zip(
                    q.addressable_shards,
                    k.addressable_shards,
                    g.addressable_shards,
                    qr.addressable_shards,
                    kr.addressable_shards,
                    dq.addressable_shards,
                    dk.addressable_shards,
                ):
                    qa = np.asarray(qs.data, dtype=np.float64)
                    ka = np.asarray(ks.data, dtype=np.float64)
                    qra = np.asarray(qrs.data, dtype=np.float64)
                    kra = np.asarray(krs.data, dtype=np.float64)
                    dqa = np.asarray(dqs.data, dtype=np.float64)
                    dka = np.asarray(dks.data, dtype=np.float64)

                    ref_qr, ref_kr = apply_ago_rotations_fp64(qa, ka)
                    err_qr = np.max(np.abs(qra - ref_qr))
                    err_kr = np.max(np.abs(kra - ref_kr))
                    finite_grad = float(np.all(np.isfinite(dqa)) and np.all(np.isfinite(dka)))
                    local.append([err_qr, err_kr, finite_grad])

                arr = np.asarray(mh.process_allgather(np.array(local))).reshape(-1, 3)
                max_err_q = float(arr[:, 0].max())
                max_err_k = float(arr[:, 1].max())
                all_finite = bool(np.all(arr[:, 2] == 1.0))
                passed = bool(max_err_q <= tol and max_err_k <= tol and all_finite)
                rows.append({
                    "dtype": str(jnp.dtype(dtype)),
                    "length": length,
                    "head_dim": head_dim,
                    "max_err_q": max_err_q,
                    "max_err_k": max_err_k,
                    "all_finite": all_finite,
                    "tolerance": tol,
                    "passed": passed,
                })
    return {"rows": rows, "passed": all(r["passed"] for r in rows)}


def unimodularity_and_orthogonality(place):
    rows = []
    for dim in (32, 64):
        for length in (128, 512, 2048, 4096, 8192):
            rot = build_cayley_rotary_matrix(dim, length, dtype=jnp.float32)
            c, s = rot.c, rot.s
            # det(R) = c^2 + s^2
            det_diff = jnp.max(jnp.abs(c**2 + s**2 - 1.0))
            # Column orthogonality: c * (-s) + s * c = 0
            ortho = jnp.max(jnp.abs(c * (-s) + s * c))

            det_err = float(jax.block_until_ready(det_diff))
            ortho_err = float(jax.block_until_ready(ortho))
            # In FP32 on TPU, machine epsilon is 1.19e-7
            tol = 1.0e-6
            passed = bool(det_err <= tol and ortho_err <= tol)
            rows.append({
                "dim": dim,
                "length": length,
                "det_error": det_err,
                "ortho_error": ortho_err,
                "tolerance": tol,
                "passed": passed,
            })
    return {"rows": rows, "passed": all(r["passed"] for r in rows)}


def shift_equivariance(place):
    rows = []
    dim = 64
    for length in (128, 512, 2048, 4096):
        rot = build_cayley_rotary_matrix(dim, length + 1, dtype=jnp.float32)
        c, s = rot.c, rot.s
        delta = 7
        m = jnp.arange(0, length - delta)
        n = m + delta
        c_prod = c[m] * c[n] + s[m] * s[n]
        s_prod = c[m] * s[n] - s[m] * c[n]
        c_diff = jnp.max(jnp.abs(c_prod - c[delta:delta+1]))
        s_diff = jnp.max(jnp.abs(s_prod - s[delta:delta+1]))
        max_err = float(jax.block_until_ready(jnp.maximum(c_diff, s_diff)))
        tol = 1.0e-6
        passed = bool(max_err <= tol)
        rows.append({
            "length": length,
            "dim": dim,
            "max_shift_error": max_err,
            "tolerance": tol,
            "passed": passed,
        })
    return {"rows": rows, "passed": all(r["passed"] for r in rows)}


def norm_conservation(place):
    rng = np.random.default_rng(342 + jax.process_index())
    rows = []
    dim = 64
    rot = build_cayley_rotary_matrix(dim, 8192, dtype=jnp.float32)
    c_table = rot.c
    s_table = rot.s

    batch = 64
    v_host = rng.normal(size=(batch, dim)).astype(np.float32)
    v_host /= np.linalg.norm(v_host, axis=-1, keepdims=True)
    v = place(v_host)

    # Test norm preservation across positions m in [1, 8192]
    # Rotate v by position m
    curr_batch = v.shape[0]
    vp = v.reshape(curr_batch, 1, dim // 2, 2)
    for length in (128, 512, 2048, 4096, 8192):
        c_m = c_table[length - 1:length, :]
        s_m = s_table[length - 1:length, :]
        v0 = c_m * vp[..., 0] - s_m * vp[..., 1]
        v1 = s_m * vp[..., 0] + c_m * vp[..., 1]
        v_rot = jnp.stack([v0, v1], axis=-1).reshape(curr_batch, dim)
        norms = jnp.linalg.norm(v_rot, axis=-1)
        max_drift = float(jax.block_until_ready(jnp.max(jnp.abs(norms - 1.0))))
        tol = 1.0e-6
        passed = bool(max_drift <= tol)
        rows.append({
            "m": length,
            "dim": dim,
            "max_norm_drift": max_drift,
            "tolerance": tol,
            "passed": passed,
        })
    return {"rows": rows, "passed": all(r["passed"] for r in rows)}


def benchmarks(place):
    rows = []
    raw = {}
    hlo = {}
    for dtype in (jnp.float32, jnp.bfloat16):
        for length in (512, 1024, 2048, 4096):
            rng = np.random.default_rng(442 + jax.process_index())
            batch_size = 16
            num_heads = 8
            head_dim = 64
            shape = (batch_size, length, num_heads, head_dim)

            q_host = rng.normal(size=shape).astype(np.float32)
            k_host = rng.normal(size=shape).astype(np.float32)
            g_host = rng.normal(size=shape).astype(np.float32)

            q = place(q_host).astype(dtype)
            k = place(k_host).astype(dtype)
            g = place(g_host).astype(dtype)

            rot_ago = build_cayley_rotary_matrix(head_dim, length, dtype=dtype)
            c_ago = rot_ago.c
            s_ago = rot_ago.s

            # Standard RoPE baseline
            k_idx = np.arange(head_dim // 2, dtype=np.float32)
            theta = 10000.0 ** (-2.0 * k_idx / head_dim)
            m_pos = np.arange(length, dtype=np.float32)[:, None]
            angles = m_pos * theta[None, :]
            c_rope = jnp.asarray(np.cos(angles), dtype=dtype)
            s_rope = jnp.asarray(np.sin(angles), dtype=dtype)

            def ago_fwd(a, b, c, s):
                return apply_ago_rotations(a, b, rotary_params=(c, s))

            def rope_fwd(a, b, c, s):
                ap = a.reshape(a.shape[:-1] + (head_dim // 2, 2))
                bp = b.reshape(b.shape[:-1] + (head_dim // 2, 2))
                c_b = c[: a.shape[1], None, :]
                s_b = s[: a.shape[1], None, :]
                a0 = c_b * ap[..., 0] - s_b * ap[..., 1]
                a1 = s_b * ap[..., 0] + c_b * ap[..., 1]
                b0 = c_b * bp[..., 0] - s_b * bp[..., 1]
                b1 = s_b * bp[..., 0] + c_b * bp[..., 1]
                return jnp.stack([a0, a1], axis=-1).reshape(a.shape), jnp.stack([b0, b1], axis=-1).reshape(b.shape)

            def ago_both(a, b, g_in, c, s):
                (qr, kr), vjp_fn = jax.vjp(lambda x, y: apply_ago_rotations(x, y, rotary_params=(c, s)), a, b)
                return (qr, kr), vjp_fn((g_in, g_in))

            def rope_both(a, b, g_in, c, s):
                (qr, kr), vjp_fn = jax.vjp(lambda x, y: rope_fwd(x, y, c, s), a, b)
                return (qr, kr), vjp_fn((g_in, g_in))

            calls = {}
            for name, fwd_fn, both_fn, c_in, s_in in [
                ("ago", ago_fwd, ago_both, c_ago, s_ago),
                ("rope", rope_fwd, rope_both, c_rope, s_rope),
            ]:
                for mode, fn, args in [
                    ("forward", fwd_fn, (q, k, c_in, s_in)),
                    ("forward_backward", both_fn, (q, k, g, c_in, s_in)),
                ]:
                    lower = jax.jit(fn).lower(*args)
                    compiled = lower.compile()
                    key = f"{name}_{mode}"
                    calls[key] = (compiled, args)
                    if name == "ago" and length == 2048:
                        hlo[f"{str(jnp.dtype(dtype))}_{mode}"] = lower.as_text()
                    for _ in range(10):
                        jax.block_until_ready(compiled(*args))

            latencies = {key: [] for key in calls}
            perm_rng = np.random.default_rng(77)
            for rep in range(100):
                for key in perm_rng.permutation(list(calls)):
                    fn, args = calls[key]
                    mh.sync_global_devices(f"bench-{dtype}-{length}-{rep}-{key}")
                    start = time.perf_counter_ns()
                    jax.block_until_ready(fn(*args))
                    elapsed = (time.perf_counter_ns() - start) * 1e-9
                    latencies[key].append(float(np.asarray(mh.process_allgather(np.array(elapsed))).max()))

            gates = {}
            for mode in ("forward", "forward_backward"):
                med_rope = float(np.median(latencies[f"rope_{mode}"]))
                med_ago = float(np.median(latencies[f"ago_{mode}"]))
                ratio = med_rope / med_ago if med_ago > 0 else 1.0
                gates[mode] = {
                    "ratio": ratio,
                    "minimum": 0.90,
                    "passed": ratio >= 0.90,
                    "med_ago_ms": med_ago * 1000.0,
                    "med_rope_ms": med_rope * 1000.0,
                }

            passed = all(v["passed"] for v in gates.values())
            row = {
                "dtype": str(jnp.dtype(dtype)),
                "length": length,
                "global_shape": list(shape),
                "warmups": 10,
                "repetitions": 100,
                "latency_seconds": {k: summary(v) for k, v in latencies.items()},
                "gates": gates,
                "passed": passed,
            }
            rows.append(row)
            raw[f"{row['dtype']}_L{length}"] = latencies
            if jax.process_index() == 0:
                print("TPU benchmark", row["dtype"], f"L={length}", {k: round(v["ratio"], 3) for k, v in gates.items()}, flush=True)

    return {"rows": rows, "passed": all(r["passed"] for r in rows)}, raw, hlo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/positions/tpu")
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
        "study": 3,
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

    r["unimodularity_and_orthogonality"] = unimodularity_and_orthogonality(place)
    if jax.process_index() == 0:
        print("TPU unimodularity & orthogonality:", r["unimodularity_and_orthogonality"]["passed"], flush=True)

    r["shift_equivariance"] = shift_equivariance(place)
    if jax.process_index() == 0:
        print("TPU shift equivariance:", r["shift_equivariance"]["passed"], flush=True)

    r["norm_conservation"] = norm_conservation(place)
    if jax.process_index() == 0:
        print("TPU norm conservation:", r["norm_conservation"]["passed"], flush=True)

    r["benchmarks"], latencies, hlo = benchmarks(place)
    if jax.process_index() == 0:
        print("TPU benchmarks:", r["benchmarks"]["passed"], flush=True)

    r["passed"] = all(
        r[k]["passed"]
        for k in (
            "parity",
            "unimodularity_and_orthogonality",
            "shift_equivariance",
            "norm_conservation",
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
        print("TPU POSITION PASS" if r["passed"] else "TPU POSITION FAIL", flush=True)

    mh.sync_global_devices("positions-complete")
    jax.distributed.shutdown()
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
