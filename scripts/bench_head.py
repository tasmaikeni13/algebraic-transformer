#!/usr/bin/env python3
"""Time the fused vocabulary head (forward+backward) against dense XLA heads."""
import argparse, json, time, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax, jax.numpy as jnp
from src.sota import algebraic as A, standard as S
from src.sota.kernels.fused_head import fused_head


def timeit(f, *a, iters=10):
    jax.block_until_ready(f(*a))
    t = time.perf_counter()
    for _ in range(iters):
        out = f(*a)
    jax.block_until_ready(out)
    return (time.perf_counter() - t) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--vocab", type=int, default=50257)
    p.add_argument("--blocks", default="512x1024,512x2048,1024x1024,256x2048")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    N, d, V = a.tokens, a.dim, a.vocab
    ks = jax.random.split(jax.random.PRNGKey(0), 3)
    h = jax.random.normal(ks[0], (N, d), jnp.bfloat16)
    E = jax.random.normal(ks[1], (V, d), jnp.float32) * 0.05
    t = jax.random.randint(ks[2], (N,), 0, V)
    flops = 3 * 2 * N * d * V
    res = {}

    def run(name, loss):
        f = jax.jit(jax.grad(lambda h, E: loss(h, E)[0], argnums=(0, 1)))
        dt = timeit(f, h, E)
        res[name] = {"ms": dt * 1e3, "tflops": flops / dt / 1e12}
        print(f"{name:32s} {dt*1e3:8.2f} ms  {flops/dt/1e12:6.1f} TFLOP/s (3 matmuls)", flush=True)

    def dense(kind):
        def f(h, E):
            z = jnp.matmul(h, E.astype(h.dtype).T, preferred_element_type=jnp.float32)
            return S.cross_entropy(z, t) if kind == "ce" else A.algebraic_likelihood(z, t, 16, 16.0)
        return f
    run("dense ce", dense("ce"))
    run("dense alg_ce", dense("alg"))
    run("dense alg_ce lean", lambda h, E: A.algebraic_likelihood_lean(
        jnp.matmul(h, E.astype(h.dtype).T, preferred_element_type=jnp.float32), t, 16, 16.0))
    for spec in a.blocks.split(","):
        bn, bv = map(int, spec.split("x"))
        for kind in ("ce", "alg_ce"):
            run(f"fused {kind} {bn}x{bv}", lambda h, E, kind=kind, bn=bn, bv=bv:
                fused_head(h, E, t, kind=kind, block_n=bn, block_v=bv))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"shape": [N, d, V], "results": res}, indent=1))


if __name__ == "__main__":
    main()
