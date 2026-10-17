#!/usr/bin/env python3
"""Paired TPU timing of deployed model components, including algebraic attention.

The normalization/rotation timing differentiates raw Q/K and their gains.
Head timing uses the deployed n=8,c=2 algebraic likelihood, including its
telemetry. These are component measurements, not optimizer-step throughput.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from analysis.attention_performance import ratio_ci
from algebraic import algebraic as A, standard as S
from algebraic import model
from algebraic.kernels.flash import flash_attention


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--vocab", type=int, default=32768)
    p.add_argument("--groups", nargs="+", choices=("attention", "head"), default=["attention", "head"])
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=131)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    B, H, T, D = shape = (args.batch, args.heads, args.seq, args.dim)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), 8)
    q, k, v, g = [jax.random.normal(kk, shape, jnp.bfloat16) for kk in keys[:4]]
    # Projection outputs have this approximate RMS at initialization.
    q, k = (q * .55).astype(q.dtype), (k * .55).astype(k.dtype)
    gains = jnp.ones((D,), jnp.float32)
    trig = S.rope_table(D, T)
    cayley = A.cayley_table_geometric(D, T)
    sink = jnp.zeros((H,), jnp.float32)
    hidden = jax.random.normal(keys[4], (B*T, H*D), jnp.bfloat16)
    embed = .02 * jax.random.normal(keys[5], (args.vocab, H*D), jnp.float32)
    targets = jax.random.randint(keys[6], (B*T,), 0, args.vocab)
    report = {"schema": 1, "utc": datetime.now(timezone.utc).isoformat(),
              "seed": args.seed, "attention_shape": shape,
              "head_shape": [B*T, H*D, args.vocab], "dtype": "bfloat16",
              "qk_input_rms": .55, "qk_gains": 1., "jax": jax.__version__,
              "hostname": platform.node(), "devices": [str(x) for x in jax.devices()],
              "rounds": args.rounds, "iters": args.iters,
              "scope": "component forward/VJP; no optimizer, model, data loading, or quality estimate",
              "source_sha256": {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                  for f in ("analysis/model_component_performance.py", "src/algebraic/algebraic.py",
                            "src/algebraic/standard.py", "src/algebraic/kernels/flash.py",
                            "src/algebraic/model.py", "src/algebraic/kernels/qk_norm.py")},
              "results": {}, "comparisons": {}, "failures": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    cases = {}
    def compile_case(name, group, mode, fn, inputs):
        try:
            start = time.perf_counter()
            f = jax.jit(fn).lower(*inputs).compile()
            jax.block_until_ready(f(*inputs))
            seconds = time.perf_counter()-start
            for _ in range(3):
                jax.block_until_ready(f(*inputs))
            cases[name] = (f, inputs, group, mode)
            report["results"][name] = {"group": group, "mode": mode,
                                      "compile_s": seconds, "samples_ms": []}
            print(f"compiled {name} ({seconds:.1f}s)", flush=True)
        except Exception as exc:
            report["failures"][name] = {"type": type(exc).__name__, "message": str(exc)}
            print(f"FAILED {name}: {str(exc).splitlines()[0]}", flush=True)
        save()
    if "attention" in args.groups:
        for arch in ("standard", "standard_qknorm", "algebraic"):
            cfg = model.LMConfig(d_model=H*D,n_heads=H,seq_len=T,attn_impl="flash",
                                 attn="asinh" if arch=="algebraic" else "softmax")
            def attn(q, k, v, qg, kg, arch=arch,cfg=cfg):
                if arch != "standard":
                    q,k=model._qk_norm(q,qg,cfg),model._qk_norm(k,kg,cfg)
                tables = cayley if arch == "algebraic" else trig
                q, k = A.apply_rotary(q, *tables), A.apply_rotary(k, *tables)
                return (flash_attention(q, k, v, sink, kind="asinh", n=16, c=8., range_guard=True)
                        if arch == "algebraic" else flash_attention(q, k, v))
            inputs = (q, k, v, gains, gains)
            compile_case(arch+"_attention_forward", "attention", "forward", attn, inputs)
            def loss(*inputs, attn=attn):
                return jnp.sum(attn(*inputs).astype(jnp.float32)*g.astype(jnp.float32))
            compile_case(arch+"_attention_training", "attention", "training",
                         jax.grad(loss, argnums=(0, 1, 2, 3, 4)), inputs)
    if "head" in args.groups:
        for kind in ("ce", "alg_ce"):
            def loss(h, e, t, kind=kind):
                z = jnp.matmul(h, e.astype(h.dtype).T, preferred_element_type=jnp.float32)
                return S.cross_entropy(z, t) if kind == "ce" else A.algebraic_likelihood_lean(z, t, 8, 2.)
            inputs = (hidden, embed, targets)
            compile_case(kind+"_head_forward", "head", "forward", loss, inputs)
            compile_case(kind+"_head_training", "head", "training",
                         jax.value_and_grad(loss, argnums=(0, 1), has_aux=True), inputs)
    rng = np.random.default_rng(args.seed)
    names = list(cases)
    for rid in range(args.rounds):
        for ix in rng.permutation(len(names)):
            name = names[int(ix)]
            f, inputs, _, _ = cases[name]
            jax.block_until_ready(f(*inputs))
            start = time.perf_counter()
            for _ in range(args.iters):
                out = f(*inputs)
            jax.block_until_ready(out)
            report["results"][name]["samples_ms"].append((time.perf_counter()-start)*1000/args.iters)
        print(f"round {rid+1}/{args.rounds}", flush=True)
        save()
    for name, item in report["results"].items():
        item["median_ms"] = float(np.median(item["samples_ms"]))
        print(f"{name}: {item['median_ms']:.4f} ms", flush=True)
        group, mode = item["group"], item["mode"]
        reference = f"standard_attention_{mode}" if group == "attention" else f"ce_head_{mode}"
        if reference in report["results"]:
            report["comparisons"][name] = {"reference": reference, **ratio_ci(
                item["samples_ms"], report["results"][reference]["samples_ms"])}
    save()


if __name__ == "__main__":
    main()
