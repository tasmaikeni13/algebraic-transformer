#!/usr/bin/env python3
"""Reproducible float64-oracle analysis of the actual n=16,c=8 attention.

Separates model differences (compressed logits versus temperature-matched
softmax) from evaluation error (fast rho and backward statistics). Tests
Gaussian, sphere QK-normalized, heavy-tail, shifted, and adversarial rows.
The oracle uses exp/asinh for independent verification only; these are not
part of the algebraic training path.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def oracle(x, n, sink):
    lw = n * np.arcsinh(np.asarray(x, np.float64))
    if sink is not None:
        lw = np.concatenate([lw, np.full((len(x), 1), n * np.arcsinh(sink))], axis=-1)
    return np.exp(lw - logsumexp(lw, axis=-1, keepdims=True))


def evaluated(x, n, sink, fast, inv):
    x = np.asarray(x, np.float32)
    root = (1e37 / (x.shape[-1] + 1)) ** (1 / n)
    clamp = np.float32(.5 * (root - 1 / root))
    if sink is not None:
        x = np.concatenate([x, np.full((len(x), 1), sink, np.float32)], axis=-1)
    unclipped = x < clamp
    x = np.minimum(x, clamp)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        rad = np.float32(1) + x * x
        r = np.float32(1) / np.sqrt(rad)
        if fast:
            rho = x + rad * r
            # The TPU implementation uses the stable expression for the sink.
            if sink is not None:
                t = np.abs(x[:, -1]) + rad[:, -1] * r[:, -1]
                rho[:, -1] = np.where(x[:, -1] >= 0, t, 1 / t)
        else:
            t = np.abs(x) + rad * r
            rho = np.where(x >= 0, t, 1 / t)
        w = rho.copy()
        for _ in range(int(math.log2(n))):
            w = w * w
        z = w.sum(-1, keepdims=True, dtype=np.float32)
        if inv:
            p = w * (np.float32(1) / z)
        else:
            stat = np.float32(1) / np.sqrt(z)
            for _ in range(int(math.log2(n)) - 1):
                stat = stat / np.sqrt(stat)
            p = rho * stat
            for _ in range(int(math.log2(n))):
                p = p * p
    return p, r * unclipped, clamp


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"finite_count": 0}
    return {"mean": float(a.mean()), "q99": float(np.quantile(a, .99)),
            "max": float(a.max()), "finite_count": len(a)}


def rows_metrics(x, n, sink, rng):
    ref = oracle(x, n, sink)
    derivative = n / np.sqrt(1 + np.asarray(x, np.float64) ** 2)
    if sink is not None:
        derivative = np.concatenate([derivative, np.full((len(x), 1), n / math.sqrt(1 + sink ** 2))], -1)
    dp = rng.standard_normal(ref.shape)
    if sink is not None:
        dp[:, -1] = 0
    ds = derivative * ref * (dp - (dp * ref).sum(-1, keepdims=True))
    out = {}
    for fast in (False, True):
        for inv in (False, True):
            p, r, clamp = evaluated(x, n, sink, fast, inv)
            gs = n * r * p * (dp - (dp * p).sum(-1, keepdims=True))
            with np.errstate(invalid="ignore", divide="ignore"):
                err = np.linalg.norm(gs - ds, axis=-1) / np.maximum(np.linalg.norm(ds, axis=-1), 1e-30)
            out[f"{'fast' if fast else 'stable'}_{'inverseZ' if inv else 'root'}"] = {
                "nonfinite_rows": int((~np.isfinite(p).all(-1)).sum()),
                "zero_denominator_rows": int((p.sum(-1) == 0).sum()),
                "negative_weights": int((p < 0).sum()),
                "tv_error": summary(.5 * np.abs(p - ref).sum(-1)),
                "max_probability_error": summary(np.abs(p - ref).max(-1)),
                "probability_sum_error": summary(np.abs(p.sum(-1) - 1)),
                "gradient_relative_l2_error": summary(err),
                "clamped_scores": int((x > clamp).sum()), "clamp": float(clamp),
            }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/analysis/attention_numerics.json"))
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--seq", type=int, default=2048)
    args = p.parse_args()
    n, c, d, T = 16, 8., 64, args.seq
    report = {"schema": 1, "utc": datetime.now(timezone.utc).isoformat(),
              "n": n, "c": c, "head_dim": d, "seq": T,
              "seeds": list(range(args.seeds)), "rows_per_seed": args.rows,
              "oracle": "float64 softmax(n*asinh(x)); x=s/c",
              "caveat": "CPU float32 arithmetic; does not certify TPU rounding or accumulation",
              "trials": [], "adversarial": [], "model_distortion": []}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        for scale in (1., 4., 16.):
            x = rng.standard_normal((args.rows, T)) * (scale / c)
            cases = {"gaussian": x,
                     "student_t3": rng.standard_t(3, size=(args.rows, T)) * (scale / (c * math.sqrt(3)))}
            # Independent points on the RMS-normalized sphere, not an assumed Gaussian tail.
            scores = []
            for start in range(0, args.rows, 16):
                batch = min(16, args.rows - start)
                q = rng.standard_normal((batch, d))
                k = rng.standard_normal((batch, T, d))
                q *= math.sqrt(d) / np.linalg.norm(q, axis=-1, keepdims=True)
                k *= math.sqrt(d) / np.linalg.norm(k, axis=-1, keepdims=True)
                scores.append(np.einsum("bd,btd->bt", q, k) * scale / (c * math.sqrt(d)))
            cases["qk_sphere"] = np.concatenate(scores)
            for distribution, x in cases.items():
                item = {"seed": seed, "score_sd_target": scale, "distribution": distribution,
                        "x_min": float(x.min()), "x_max": float(x.max()),
                        "metrics": rows_metrics(x, n, 0., rng)}
                report["trials"].append(item)
                if distribution == "qk_sphere":
                    algebraic = oracle(x, n, None)
                    logsoft = n * x - logsumexp(n * x, axis=-1, keepdims=True)
                    refsoft = np.exp(logsoft)
                    kl = (algebraic * (np.log(np.maximum(algebraic, 1e-300)) - logsoft)).sum(-1)
                    report["model_distortion"].append({"seed": seed, "scale": scale,
                        "temperature_matched_softmax_inverse_temperature": n / c,
                        "kl_algebraic_to_softmax": summary(kl),
                        "tv": summary(.5 * np.abs(algebraic - refsoft).sum(-1)),
                        "algebraic_entropy": float(-(algebraic * np.log(np.maximum(algebraic, 1e-300))).sum(-1).mean()),
                        "softmax_entropy": float(-(refsoft * logsoft).sum(-1).mean())})
            print(f"seed {seed}, score SD {scale:g} done", flush=True)
    rng = np.random.default_rng(991)
    for center in (-1e4, -1e3, -100., -10., 0., 10., 100.):
        x = rng.uniform(center - .125, center + .125, size=(16, T))
        for sink in (None, 0.):
            report["adversarial"].append({"center_kernel_units": center, "sink_kernel_units": sink,
                                          "metrics": rows_metrics(x, n, sink, rng)})
    # At learned gain gq,gk the deterministic Cauchy-Schwarz bound is sqrt(d)*max|gq|*max|gk|/c.
    report["qk_bound"] = {"unit_gains_max_abs_kernel_score": math.sqrt(d) / c,
        "formula": "sqrt(d)*max_abs(q_gain)*max_abs(k_gain)/c; RMS eps only reduces norms",
        "note": "unit-gain bound is not preserved when learned gains grow"}
    # An independent check of the additive log-sum-exp statistic used by
    # standard flash backward. Very large offsets can erase log(l) in float32.
    report["softmax_lse_stress"] = []
    for offset in (-1e8, -1e5, -1e3, 0., 1e3, 1e5, 1e8):
        scores = (rng.standard_normal((16, T)) + offset).astype(np.float32)
        ref_logits = scores.astype(np.float64)
        ref = np.exp(ref_logits - logsumexp(ref_logits, axis=-1, keepdims=True))
        m = scores.max(-1, keepdims=True)
        w = np.exp(scores - m)
        l = w.sum(-1, keepdims=True, dtype=np.float32)
        pf = w / l
        lse = m + np.log(l)
        pb = np.exp(scores - lse)
        report["softmax_lse_stress"].append({"score_offset": offset,
            "forward_tv": summary(.5 * np.abs(pf - ref).sum(-1)),
            "backward_tv": summary(.5 * np.abs(pb - ref).sum(-1)),
            "backward_probability_sum_error": summary(np.abs(pb.sum(-1) - 1))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
