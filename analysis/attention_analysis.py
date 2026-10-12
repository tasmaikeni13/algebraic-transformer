#!/usr/bin/env python3
"""Phase 1c: attention kernels and positional frequency grids.

A. Attention entropy at initialization (Monte Carlo): random queries/keys
   with per-coordinate scale sigma, d = 64, T = 1024 visible keys. Softmax,
   the first study's octic kernel (n=8, c=1, sink mass 1), and asinh kernels.

B. Associative recall (Monte Carlo over seeds): one attention layer must
   retrieve the value stored with the queried key among N key/value items.
   Kernels compete on equal training budgets; we report final accuracy and
   loss. This isolates retrieval precision from everything else.

C. Positional resolution (linear programs): for a frequency grid theta_k,
   the score of a head as a function of relative offset is
   s(D) = sum_k a_k cos(theta_k D) + b_k sin(theta_k D). Maximize the margin
   s(D*) - max_{D != D*, D < L} s(D) subject to sum |a_k| + |b_k| <= 1. A
   small margin means position alone cannot single out "exactly D* back".

Writes results/analysis/attention_analysis.json.
"""

import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linprog

from src.sota import algebraic as A

OUT = ROOT / "results" / "analysis"

KERNELS = {
    "softmax": dict(kind="softmax"),
    "v1_octic_n8_c1_sink1": dict(kind="asinh", n=8, c=1.0, sink=0.0),
    "asinh_n16_c16": dict(kind="asinh", n=16, c=16.0, sink=None),
    "asinh_n16_c4": dict(kind="asinh", n=16, c=4.0, sink=None),
    "asinh_n32_c32": dict(kind="asinh", n=32, c=32.0, sink=None),
    "asinh_n8_c2": dict(kind="asinh", n=8, c=2.0, sink=None),
}


def weights(scores, spec):
    """scores (..., T) -> attention weights over T keys (all visible)."""
    if spec["kind"] == "softmax":
        return jax.nn.softmax(scores, -1)
    mask = jnp.ones(scores.shape, bool)
    sink = None if spec.get("sink") is None else jnp.full(scores.shape[:-1] + (1,), spec["sink"])
    return A.asinh_power_weights(scores, mask, spec["n"], spec["c"], sink)


# ---------------------------------------------------------------------------
# A
# ---------------------------------------------------------------------------

def part_a(T=1024, d=64, samples=256):
    rows = []
    key = jax.random.PRNGKey(0)
    for sigma in (0.25, 0.5, 1.0, 2.0, 4.0):
        kq, kk = jax.random.split(jax.random.fold_in(key, int(sigma * 100)))
        q = jax.random.normal(kq, (samples, d)) * sigma
        k = jax.random.normal(kk, (samples, T, d)) * sigma
        s = jnp.einsum("nd,ntd->nt", q, k) / math.sqrt(d)
        row = {"sigma": sigma, "score_sd": float(s.std())}
        for name, spec in KERNELS.items():
            w = weights(s, spec)
            mass = w.sum(-1)
            p = w / mass[:, None]
            ent = -(p * jnp.log(jnp.maximum(p, 1e-30))).sum(-1)
            row[name] = {"entropy_over_logT": float(ent.mean() / math.log(T)),
                         "max_weight": float(w.max(-1).mean()),
                         "visible_mass": float(mass.mean())}
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# B
# ---------------------------------------------------------------------------

def recall_trial(spec, n_items, seed, steps=1500, n_keys=None, d=64, batch=128, lr=3e-3,
                 weight_fn=None, qk_normalize=False, return_details=False):
    n_keys = n_keys or max(256, 2 * n_items)
    key = jax.random.PRNGKey(seed)
    ks = jax.random.split(key, 8)
    p = {
        "ek": jax.random.normal(ks[0], (n_keys, d)) * 0.5,
        "ev": jax.random.normal(ks[1], (n_keys, d)) * 0.5,
        "wq": jax.random.normal(ks[2], (d, d)) / math.sqrt(d),
        "wk": jax.random.normal(ks[3], (d, d)) / math.sqrt(d),
        "wv": jax.random.normal(ks[4], (d, d)) / math.sqrt(d),
        "wo": jax.random.normal(ks[5], (d, n_keys)) / math.sqrt(d),
    }
    if qk_normalize:
        p["q_gain"] = jnp.ones((d,))
        p["k_gain"] = jnp.ones((d,))

    def sample(k):
        k1, k2, k3 = jax.random.split(k, 3)
        keys = jax.vmap(lambda kk: jax.random.permutation(kk, n_keys)[:n_items])(jax.random.split(k1, batch))
        vals = jax.random.randint(k2, (batch, n_items), 0, n_keys)
        pick = jax.random.randint(k3, (batch,), 0, n_items)
        return keys, vals, pick

    def loss_fn(p, keys, vals, pick):
        items = p["ek"][keys] + p["ev"][vals]                  # (B, N, d)
        qk = p["ek"][jnp.take_along_axis(keys, pick[:, None], 1)[:, 0]]  # (B, d)
        q = qk @ p["wq"]
        k = items @ p["wk"]
        if qk_normalize:
            q,k = A.rms_norm(q,p["q_gain"]),A.rms_norm(k,p["k_gain"])
        v = items @ p["wv"]
        s = jnp.einsum("bd,bnd->bn", q, k) / math.sqrt(d)
        w = weights(s, spec) if weight_fn is None else weight_fn(s)
        o = jnp.einsum("bn,bnd->bd", w, v)
        logits = o @ p["wo"]
        target = jnp.take_along_axis(vals, pick[:, None], 1)[:, 0]
        lp = jax.nn.log_softmax(logits, -1)
        nll = -jnp.take_along_axis(lp, target[:, None], -1)[:, 0]
        acc = (jnp.argmax(logits, -1) == target).mean()
        return nll.mean(), acc

    grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    m = jax.tree_util.tree_map(jnp.zeros_like, p)
    v = jax.tree_util.tree_map(jnp.zeros_like, p)

    @jax.jit
    def step(i, p, m, v, k):
        keys, vals, pick = sample(k)
        (l, acc), g = grad(p, keys, vals, pick)
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        c1, c2 = 1 - 0.9 ** (i + 1), 1 - 0.999 ** (i + 1)
        p = jax.tree_util.tree_map(lambda a, b, c: a - lr * (b / c1) / (jnp.sqrt(c / c2) + 1e-8), p, m, v)
        return p, m, v, l, acc

    k = ks[6]
    for i in range(steps):
        k, sub = jax.random.split(k)
        p, m, v, l, acc = step(i, p, m, v, sub)
    # evaluation on fresh samples
    accs, losses = [], []
    for j in range(8):
        keys, vals, pick = sample(jax.random.fold_in(ks[7], j))
        l, a = loss_fn(p, keys, vals, pick)
        accs.append(float(a)); losses.append(float(l))
    if return_details:
        detail = {"nll":float(np.mean(losses)),"accuracy":float(np.mean(accs))}
        if qk_normalize:
            detail["gain_statistics"] = {name:{"rms":float(jnp.sqrt(jnp.mean(p[name]**2))),
                "min":float(jnp.min(p[name])),"max":float(jnp.max(p[name]))} for name in ("q_gain","k_gain")}
        return detail
    return float(np.mean(losses)), float(np.mean(accs))


def part_b(seeds=(0, 1, 2), items=(32, 128, 512)):
    rows = []
    for n_items in items:
        for name, spec in KERNELS.items():
            res = [recall_trial(spec, n_items, s) for s in seeds]
            row = {"items": n_items, "kernel": name,
                   "loss_mean": float(np.mean([r[0] for r in res])),
                   "loss_se": float(np.std([r[0] for r in res], ddof=1) / math.sqrt(len(res))),
                   "acc_mean": float(np.mean([r[1] for r in res]))}
            print(row, flush=True)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# C
# ---------------------------------------------------------------------------

def grids(half=32, base=1e4):
    k = np.arange(half)
    alpha = (math.sqrt(base) - 1) / (half - 1)
    return {
        "cayley_v1": 2 * np.arctan(1.0 / (1 + alpha * k) ** 2),
        "rope": base ** (-k / half),
        "cayley_geometric": 2 * np.arctan(0.5 * base ** (-k / half)),
    }


def margin_lp(theta, target, L):
    offsets = np.arange(L)
    feats = np.concatenate([np.cos(np.outer(offsets, theta)), np.sin(np.outer(offsets, theta))], 1)
    nf = feats.shape[1]
    # variables: a+ (nf), a- (nf), m ; maximize m
    others = [o for o in offsets if o != target]
    A_ub, b_ub = [], []
    for o in others:
        diff = feats[o] - feats[target]          # (f_o - f_t) . a + m <= 0
        A_ub.append(np.concatenate([diff, -diff, [1.0]]))
        b_ub.append(0.0)
    A_ub.append(np.concatenate([np.ones(nf), np.ones(nf), [0.0]]))
    b_ub.append(1.0)
    c = np.zeros(2 * nf + 1)
    c[-1] = -1.0
    bounds = [(0, None)] * (2 * nf) + [(None, None)]
    res = linprog(c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub), bounds=bounds, method="highs")
    return float(-res.fun) if res.success else float("nan")


def part_c():
    rows = []
    for L in (256, 2048):
        for target in (1, 2, 3, 8, 32, 100):
            row = {"context": L, "offset": target}
            for name, theta in grids().items():
                row[name] = margin_lp(theta, target, L)
            print(row, flush=True)
            rows.append(row)
    spectra = {}
    for name, theta in grids().items():
        wl = 2 * np.pi / theta
        spectra[name] = {f"pairs_wavelength_lt_{L}": int((wl < L).sum()) for L in (8, 16, 64, 256, 2048)}
    return {"margins": rows, "spectra": spectra}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["a", "b", "c"]
    path = OUT / "attention_analysis.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    if "a" in which:
        report["A_init_entropy"] = part_a()
        print(json.dumps(report["A_init_entropy"], indent=1))
    if "c" in which:
        report["C_positional_margins"] = part_c()
    if "b" in which:
        report["B_associative_recall"] = part_b()
    path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
