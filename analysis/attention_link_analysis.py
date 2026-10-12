#!/usr/bin/env python3
"""Static routing/temperature diagnostics for rational and affine-power links."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


SPECS = {
    "standard": ("softmax", 1., 1., None),
    "standard_temperature2": ("softmax", 2., 1., None),
    "asinh16_c8": ("asinh", 16., 8., None),
    "pade8_c16_a075": ("pade", 8., 16., .75),
    "pade4_c8_a09375": ("pade", 4., 8., .9375),
    "pade4_c4_a09375": ("pade", 4., 4., .9375),
    "binomial32_c32": ("binomial", 32., 32., (-.75, .75)),
    "binomial64_c64": ("binomial", 64., 64., (-.5, .75)),
    "binomial64_c32": ("binomial", 64., 32., (-.5, .75)),
    "quadratic16_c8": ("quadratic", 16., 8., (-1., 8.)),
}


def logits(s, spec):
    kind, n, c, a = spec
    if kind == "softmax":
        return n * s
    if kind == "asinh":
        return n * np.arcsinh(s / c)
    if kind in ("binomial", "quadratic"):
        x = np.clip(s / c, a[0], a[1])
        return n * np.log1p(x + .5 * x * x) if kind == "quadratic" else n * np.log1p(x)
    x = np.clip(s / c, -a, a)
    return n * (np.log1p(x) - np.log1p(-x))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/analysis/attention_link_analysis.json"))
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--seq", type=int, default=2048)
    args = p.parse_args()
    T, d = args.seq, 64
    report = {"schema": 2, "specs": SPECS, "rows_per_seed": args.rows,
              "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "seeds": list(range(args.seeds)), "d": d, "distractors": T,
              "scope": "static normalized-sphere routing; not trained retrieval or language modeling",
              "trials": []}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        chunks = []
        for start in range(0, args.rows, 16):
            B = min(16, args.rows - start)
            q = rng.standard_normal((B, d))
            k = rng.standard_normal((B, T, d))
            q *= math.sqrt(d) / np.linalg.norm(q, axis=-1, keepdims=True)
            k *= math.sqrt(d) / np.linalg.norm(k, axis=-1, keepdims=True)
            chunks.append(np.einsum("bd,btd->bt", q, k) / math.sqrt(d))
        unit = np.concatenate(chunks)
        for scale in (1., 1.5, 2., 4., 8.):
            s = unit * scale
            deployed = logits(s, SPECS["asinh16_c8"])
            deployed -= logsumexp(deployed, -1, keepdims=True)
            deployed_prob = np.exp(deployed)
            for name, spec in SPECS.items():
                lp = logits(s, spec)
                lnorm = logsumexp(lp, -1, keepdims=True)
                prob = np.exp(lp - lnorm)
                beta = spec[1] if spec[0] == "softmax" else spec[1] / spec[2] * (2 if spec[0] == "pade" else 1)
                teacher = beta * s - logsumexp(beta * s, -1, keepdims=True)
                kl = np.sum(prob * (lp - lnorm - teacher), -1)
                cue_lp = logits(np.full((len(s), 1), 8 * scale), spec)
                cue_prob = np.exp(cue_lp - np.logaddexp(cue_lp, lnorm))[:, 0]
                ties = (lp == cue_lp).sum(-1)
                if spec[0] in ("binomial", "quadratic"):
                    fraction = float(((s / spec[2] <= spec[3][0]) | (s / spec[2] >= spec[3][1])).mean())
                else:
                    fraction = float((np.abs(s / spec[2]) >= spec[3]).mean()) if spec[3] else 0.
                report["trials"].append({"seed": seed, "gain_product": scale, "kernel": name,
                    "origin_slope": beta, "KL_to_temperature_matched_softmax_mean": float(kl.mean()),
                    "KL_to_deployed_asinh_mean": float(np.sum(prob * (lp - lnorm - deployed), -1).mean()),
                    "TV_to_deployed_asinh_mean": float((.5 * np.abs(prob - deployed_prob).sum(-1)).mean()),
                    "cue_raw_score": 8 * scale, "cue_probability_mean": float(cue_prob.mean()),
                    "cue_probability_q01": float(np.quantile(cue_prob, .01)),
                    "cue_tie_rows": int((ties > 0).sum()), "mean_other_cue_ties": float(ties.mean()),
                    "uniform_tie_break_correct_probability": float((1 / (1 + ties)).mean()),
                    "fraction_scores_clipped": fraction,
                    "entropy_mean": float(-(prob * (lp - lnorm)).sum(-1).mean())})
        print(f"seed {seed} done", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
