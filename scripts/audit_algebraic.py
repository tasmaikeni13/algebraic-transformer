#!/usr/bin/env python3
"""Audit the algebraic training path at the level of traced primitives.

Traces the gradient of the training objective (forward and backward,
including the bodies of Pallas kernels) for a configuration and lists every
primitive. The algebraic path passes when none of the forbidden primitives
appears. Telemetry (the logged NLL) is computed by the trainer outside this
graph and is not part of the training path.

  python scripts/audit_algebraic.py --preset algebraic --set attn_impl=flash
"""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp

FORBIDDEN = {"exp", "exp2", "expm1", "log", "log1p", "logistic", "sin", "cos", "tan", "tanh",
             "asinh", "acosh", "atanh", "sinh", "cosh", "atan2", "erf", "erfc", "erf_inv",
             "pow", "lgamma", "digamma"}


def primitives(jaxpr) -> Counter:
    counts = Counter()
    stack = [jaxpr]
    while stack:
        j = stack.pop()
        j = getattr(j, "jaxpr", j)
        for eqn in getattr(j, "eqns", []):
            counts[eqn.primitive.name] += 1
            for value in eqn.params.values():
                values = value if isinstance(value, (list, tuple)) else [value]
                for v in values:
                    if hasattr(v, "eqns") or hasattr(v, "jaxpr"):
                        stack.append(v)
    return counts


def audit(preset: str, overrides: dict, seq_len: int = 256, *, size="proxy", batch=2) -> dict:
    from algebraic import config as cfgs
    from algebraic.model import forward_hidden, head_loss, init_params, position_tables
    cfg = cfgs.build(preset, size, **{**overrides, "seq_len": seq_len})
    # Abstract parameter shapes trace the complete model without allocating weights.
    params = jax.eval_shape(lambda: init_params(cfg, jax.random.PRNGKey(0)))
    tables = position_tables(cfg, seq_len)
    tok = jax.ShapeDtypeStruct((batch, seq_len), jnp.int32)

    def objective(p, tokens):
        return head_loss(p, forward_hidden(p, tokens, cfg, tables), tokens, cfg)[0]

    counts = primitives(jax.make_jaxpr(jax.grad(objective))(params, tok))
    bad = sorted(set(counts) & FORBIDDEN)
    return {"preset": preset, "size": size, "batch": batch, "config": asdict(cfg),
            "overrides": overrides, "seq_len": seq_len,
            "kernel_sha256": hashlib.sha256((ROOT / "src/algebraic/kernels/flash.py").read_bytes()).hexdigest(),
            "fused_head_sha256": hashlib.sha256((ROOT / "src/algebraic/kernels/fused_head.py").read_bytes()).hexdigest(),
            "model_sha256": hashlib.sha256((ROOT / "src/algebraic/model.py").read_bytes()).hexdigest(),
            "qk_norm_sha256": hashlib.sha256((ROOT / "src/algebraic/kernels/qk_norm.py").read_bytes()).hexdigest(),
            "forbidden_found": bad,
            "passed": not bad, "primitive_counts": dict(sorted(counts.items()))}


def main():
    from algebraic.train import parse_overrides
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="algebraic")
    p.add_argument("--size", default="proxy")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    res = audit(a.preset, parse_overrides(a.set), seq_len=a.seq_len, size=a.size, batch=a.batch)
    print(json.dumps({k: v for k, v in res.items() if k != "primitive_counts"}, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
