#!/usr/bin/env python3
"""Phase 1b: mathematics and Monte Carlo of the output head and objective.

A. Representational floor. For Zipf-like target distributions q over the
   GPT-2 vocabulary, minimize KL(q || p(z)) over unconstrained logits z for
   (i) softmax, (ii) the first study's head (vocabulary AVN with eps=100 and
   rho^8), (iii) the second-generation algebraic head rho(z/c)^n. Softmax and
   (iii) represent any q exactly; (ii) cannot when q's dynamic range is too
   large for the RMS constraint (proof sketch in paper/theory.md, section 4).

B. Objective under misspecification (Monte Carlo over random problems).
   Contexts x in R^d, true conditionals q(.|x) from a nonlinear teacher, a
   linear student z = W x trained with full-batch Adam on (a) log-loss and
   (b) the OACE power score, through the same algebraic link. Report the test
   log-loss gap: the price of training on OACE but evaluating perplexity.

C. Rank bottleneck. With z = W x of rank d, softmax log-probabilities lie in a
   (d+1)-dimensional affine family; n asinh(z/c) is an entrywise nonlinear
   map whose image is not. Fit a high-rank teacher with rank-d students and
   compare the achievable log-loss for softmax and asinh links.

Writes results/analysis/head_analysis.json.
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

from src.sota import algebraic as A

if jax.default_backend() == "cpu":
    jax.config.update("jax_enable_x64", True)
OUT = ROOT / "results" / "analysis"


def log_probs(z, link, n=16, c=16.0, eps_vocab=100.0):
    if link == "softmax":
        a = z
    elif link == "v1":
        zn = z / jnp.sqrt(jnp.mean(z * z, -1, keepdims=True) + eps_vocab)
        a = 8.0 * jnp.arcsinh(zn)
    else:
        a = n * jnp.arcsinh(z / c)
    return jax.nn.log_softmax(a, -1)


def adam_minimize(loss, x0, steps=3000, lr=0.05):
    m = jnp.zeros_like(x0)
    v = jnp.zeros_like(x0)
    grad = jax.jit(jax.value_and_grad(loss))

    @jax.jit
    def step(i, x, m, v):
        val, g = grad(x)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** (i + 1))
        vh = v / (1 - 0.999 ** (i + 1))
        return x - lr * mh / (jnp.sqrt(vh) + 1e-12), m, v, val

    x = x0
    val = None
    for i in range(steps):
        x, m, v, val = step(i, x, m, v)
    return x, float(loss(x))


# ---------------------------------------------------------------------------
# A. representational floor
# ---------------------------------------------------------------------------

def part_a(V=50257, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s_exp in (0.8, 1.0, 1.2, 1.5):
        for top in (0.0, 0.5, 0.9):
            ranks = np.arange(1, V + 1, dtype=np.float64)
            q = ranks ** (-s_exp)
            q /= q.sum()
            if top:
                q = (1 - top) * q
                q[0] += top
            q = q[rng.permutation(V)]
            q = jnp.asarray(q)
            ent = float(-(q * jnp.log(q)).sum())
            row = {"zipf_exponent": s_exp, "top_mass": top, "entropy_nats": ent,
                   "log_dynamic_range": float(jnp.log(q.max() / q.min()))}
            for link in ("softmax", "v1", "alg16"):
                def kl(z):
                    return jnp.sum(q * (jnp.log(q) - log_probs(z, link)))
                z0 = jnp.zeros(V)
                _, val = adam_minimize(kl, z0, steps=1500, lr=0.5 if link != "softmax" else 0.1)
                row[f"min_kl_{link}"] = val
            rows.append(row)
            print(row, flush=True)
    return rows


# ---------------------------------------------------------------------------
# B. objective under misspecification
# ---------------------------------------------------------------------------

def oace_score(z, y, eps_vocab, gamma=2.0):
    loss, _ = A.oace(z, y, eps_vocab, gamma)
    return loss


def part_b(trials=12, d=24, V=512, n_train=32768, n_test=8192):
    out = []
    for t in range(trials):
        key = jax.random.PRNGKey(t)
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        # nonlinear teacher: log q = 3 * tanh-free quadratic features, Zipf bias
        U = jax.random.normal(k1, (2 * d, V)) / math.sqrt(2 * d)
        bias = -1.1 * jnp.log(jnp.arange(1, V + 1, dtype=jnp.float64))
        bias = bias[jax.random.permutation(k5, V)]

        def teacher_logits(x):
            feats = jnp.concatenate([x, (x * x - 1.0) / math.sqrt(2)], -1)
            return 1.5 * feats @ U + bias

        xtr = jax.random.normal(k2, (n_train, d))
        xte = jax.random.normal(k3, (n_test, d))
        qtr = jax.nn.softmax(teacher_logits(xtr), -1)
        ytr = jax.random.categorical(k4, jnp.log(qtr), axis=-1)
        qte = jax.nn.softmax(teacher_logits(xte), -1)
        res = {"trial": t, "teacher_test_entropy": float(-(qte * jnp.log(qte)).sum(-1).mean())}

        def test_ce(W, link):
            return float(-(qte * log_probs(xte @ W[:-1] + W[-1], link)).sum(-1).mean())

        W0 = jnp.zeros((d + 1, V))
        fits = {
            "softmax_ce": (lambda W: -jnp.mean(jnp.take_along_axis(
                log_probs(xtr @ W[:-1] + W[-1], "softmax"), ytr[:, None], -1)), "softmax"),
            "alg16_ce": (lambda W: -jnp.mean(jnp.take_along_axis(
                log_probs(xtr @ W[:-1] + W[-1], "alg16"), ytr[:, None], -1)), "alg16"),
            "v1_ce": (lambda W: -jnp.mean(jnp.take_along_axis(
                log_probs(xtr @ W[:-1] + W[-1], "v1", eps_vocab=1.0), ytr[:, None], -1)), "v1"),
            "v1_oace": (lambda W: oace_score(xtr @ W[:-1] + W[-1], ytr, 1.0), "v1"),
        }
        for name, (loss, link) in fits.items():
            W, _ = adam_minimize(loss, W0, steps=2000, lr=0.01)
            if link == "v1":
                res[name] = float(-(qte * log_probs(xte @ W[:-1] + W[-1], "v1", eps_vocab=1.0)).sum(-1).mean())
            else:
                res[name] = test_ce(W, link)
        print(res, flush=True)
        out.append(res)
    keys = [k for k in out[0] if k not in ("trial",)]
    summary = {k: {"mean": float(np.mean([r[k] for r in out])),
                   "se": float(np.std([r[k] for r in out], ddof=1) / math.sqrt(len(out)))} for k in keys}
    diffs = {
        "oace_minus_ce_same_link_v1": [r["v1_oace"] - r["v1_ce"] for r in out],
        "v1link_minus_softmax_ce": [r["v1_ce"] - r["softmax_ce"] for r in out],
        "alg16_minus_softmax_ce": [r["alg16_ce"] - r["softmax_ce"] for r in out],
    }
    summary["paired_differences"] = {k: {"mean": float(np.mean(v)), "se": float(np.std(v, ddof=1) / math.sqrt(len(v)))}
                                     for k, v in diffs.items()}
    return {"trials": out, "summary": summary,
            "setup": {"d": d, "V": V, "n_train": n_train, "n_test": n_test,
                      "note": "eps_vocab scaled to 1 for the 512-class problem (logit scale ~ O(1))"}}


# ---------------------------------------------------------------------------
# C. rank bottleneck
# ---------------------------------------------------------------------------

def part_c(trials=6, N=1024, V=1024, ranks=(8, 16, 32, 64)):
    rows = []
    for t in range(trials):
        key = jax.random.PRNGKey(100 + t)
        k1, k2 = jax.random.split(key)
        # full-rank teacher log-probabilities with LM-like spread
        T = jax.nn.log_softmax(3.0 * jax.random.normal(k1, (N, V)) - 0.9 * jnp.log(
            jnp.arange(1, V + 1, dtype=jnp.float64))[jax.random.permutation(k2, V)], -1)
        q = jnp.exp(T)
        ent = float(-(q * T).sum(-1).mean())
        for d in ranks:
            row = {"trial": t, "rank": d, "teacher_entropy": ent}
            for link, kw in (("softmax", {}), ("alg16", {}), ("alg_c4", {"n": 16, "c": 4.0}),
                             ("alg_c1", {"n": 8, "c": 1.0})):
                kk = jax.random.PRNGKey(7)
                H0 = jax.random.normal(kk, (N, d)) * 0.1
                E0 = jax.random.normal(jax.random.split(kk)[0], (d, V)) * 0.1
                x0 = jnp.concatenate([H0.ravel(), E0.ravel()])
                lk = "softmax" if link == "softmax" else "alg"

                def kl(x):
                    H = x[:N * d].reshape(N, d)
                    E = x[N * d:].reshape(d, V)
                    return jnp.mean(jnp.sum(q * (T - log_probs(H @ E, lk, **kw)), -1))

                _, val = adam_minimize(kl, x0, steps=2500, lr=0.02)
                row[f"kl_{link}"] = val
            print(row, flush=True)
            rows.append(row)
    summary = {}
    for d in ranks:
        sub = [r for r in rows if r["rank"] == d]
        summary[str(d)] = {k: float(np.mean([r[k] for r in sub])) for k in sub[0] if k.startswith("kl_")}
    return {"rows": rows, "mean_kl_by_rank": summary}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["a", "b", "c"]
    path = OUT / ("head_analysis_" + "".join(which) + ".json")
    report = json.loads(path.read_text()) if path.exists() else {}
    if "a" in which:
        report["A_representational_floor"] = part_a()
    if "b" in which:
        report["B_objective_misspecification"] = part_b()
    if "c" in which:
        report["C_rank_bottleneck"] = part_c()
    path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
