#!/usr/bin/env python3
"""Independent float64 check of held-out NLL for a saved model.

Loads ``params.npz`` and ``meta.json`` from a run directory, recomputes final
hidden states with the model in float32, then forms the predictive
distribution in float64 NumPy directly from its definition
(``p_j ~ rho(z_j / c)^n`` for algebraic heads, softmax for "ce"), checks that
it sums to one over the full vocabulary, and recomputes the mean NLL on the
first validation sequences. Agreement with the trainer's evaluation rules out
an evaluation artifact.

  python analysis/verify_head_eval.py runs/p1b4/v2c_s0_save --sequences 16
"""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--sequences", type=int, default=16)
    p.add_argument("--valid", default=str(Path.home() / "data/tok/fineweb1b_valid.npy"))
    a = p.parse_args()

    import jax
    import jax.numpy as jnp
    from algebraic.model import LMConfig, forward_hidden, position_tables, eval_logprobs

    meta = json.loads((a.run / "meta.json").read_text())
    cfg = LMConfig(**{**meta["config"], "dtype": "float32", "attn_impl": "dense"})
    flat = np.load(a.run / "params.npz")
    params = {"layers": {}}
    for key in flat.files:
        parts = [x.strip("[]'\"") for x in key.split("]") if x]
        if parts[0] == "layers":
            params["layers"][parts[1]] = jnp.asarray(flat[key])
        else:
            params[parts[0]] = jnp.asarray(flat[key])
    T = meta["args"]["seq_len"]
    valid = np.load(a.valid, mmap_mode="r")
    seqs = np.stack([np.asarray(valid[i * T:i * T + T + 1]) for i in range(a.sequences)]).astype(np.int32)
    x, y = seqs[:, :-1], seqs[:, 1:]
    tables = position_tables(cfg, T)

    hidden = np.asarray(jax.jit(lambda p, t: forward_hidden(p, t, cfg, tables))(params, x), np.float64)
    E = np.asarray(params["embed"], np.float64)
    nll_sum, count, max_dev = 0.0, 0, 0.0
    for b in range(hidden.shape[0]):
        z = hidden[b] @ E.T                                    # (T, V) float64
        if cfg.head == "ce":
            a_ = z
        else:
            n, c = cfg.head_n, cfg.head_c
            a_ = n * np.arcsinh(z / c)                         # log rho(z/c)^n
        a_ = a_ - a_.max(axis=1, keepdims=True)
        w = np.exp(a_)
        prob = w / w.sum(axis=1, keepdims=True)
        max_dev = max(max_dev, float(np.abs(prob.sum(axis=1) - 1.0).max()))
        nll_sum += float(-np.log(prob[np.arange(T), y[b]]).sum())
        count += T
    nll64 = nll_sum / count
    lp = jax.jit(lambda p, t, g: eval_logprobs(p, t, g, cfg, tables))(params, x, y)
    nll_model = float(-jnp.mean(lp))
    res = {"run": str(a.run), "head": cfg.head, "head_n": cfg.head_n, "head_c": cfg.head_c,
           "tokens": count, "nll_float64_numpy": nll64, "nll_model_eval_float32": nll_model,
           "abs_difference": abs(nll64 - nll_model), "max_abs_deviation_of_sum_from_one": max_dev}
    print(json.dumps(res, indent=1))
    out = ROOT / "results" / "analysis" / "verify_head_eval.json"
    existing = json.loads(out.read_text()) if out.exists() else []
    existing.append(res)
    out.write_text(json.dumps(existing, indent=1))


if __name__ == "__main__":
    main()
