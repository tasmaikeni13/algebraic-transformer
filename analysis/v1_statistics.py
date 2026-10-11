#!/usr/bin/env python3
"""Phase 1a: statistics of the first study's recorded runs.

Reads the six 125M final runs, the 18-run 600M tuning sweep, and the 20M
pilot traces, and writes ``results/analysis/v1_statistics.json``:

1. Paired seed analysis of the final runs (t interval, Bayesian posterior,
   sign test, TOST equivalence, effect size, seeds needed for 80% power).
2. The tuning sweep as an architecture x candidate table and the 600M -> 2.5B
   crossover.
3. Learning-rate-schedule areas and an LR-annealing scaling-law fit
   (L = L0 + A S1^-alpha - C S2; Tissue et al., 2024) on the standard arm's
   cross-entropy traces, used to predict the cost of the first study's
   rational (ARDS) schedule for an architecture that otherwise behaves like
   the standard model.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis"


def find(*candidates):
    for c in candidates:
        p = ROOT / c
        if p.exists():
            return p
    raise FileNotFoundError(candidates)


def load_final_runs():
    rows = {}
    for arch in ("algebraic", "standard"):
        for seed in (42, 43, 44):
            p = find(f"results/v1/pretraining/{arch}-{seed}/run_metrics.json")
            rows[(arch, seed)] = json.loads(p.read_text())["run"]
    return rows


def paired_analysis(alg, std, margin_list=(0.005, 0.01, 0.02)):
    d = np.asarray(alg) - np.asarray(std)
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    t = mean / se
    # Flat prior on the mean, Jeffreys on the variance -> Student-t posterior.
    p_alg_worse = float(stats.t.cdf(mean / se, n - 1))
    sign_p = float(stats.binomtest(int((d > 0).sum()), n, 0.5, alternative="greater").pvalue)
    tost = {}
    for m in margin_list:
        p_low = 1 - stats.t.cdf((mean + m) / se, n - 1)
        p_high = stats.t.cdf((mean - m) / se, n - 1)
        tost[str(m)] = float(max(p_low, p_high))
    dz = mean / sd
    seeds_needed = None
    for k in range(2, 200):
        nc = abs(dz) * math.sqrt(k)
        crit = stats.t.ppf(0.975, k - 1)
        power = 1 - stats.nct.cdf(crit, k - 1, nc) + stats.nct.cdf(-crit, k - 1, nc)
        if power >= 0.8:
            seeds_needed = k
            break
    return {
        "differences": d.tolist(), "mean": mean, "sd": sd, "se": se,
        "ci95": [mean - tcrit * se, mean + tcrit * se],
        "t": t, "p_two_sided": float(2 * (1 - stats.t.cdf(abs(t), n - 1))),
        "posterior_p_algebraic_worse": p_alg_worse,
        "sign_test_p_one_sided": sign_p,
        "tost_p_by_margin": tost,
        "cohens_dz": dz,
        "paired_seeds_for_80pct_power_at_observed_effect": seeds_needed,
    }


def schedule(kind, peak, warm, total, min_lr=None):
    t = np.arange(1, total + 1, dtype=np.float64)
    if kind == "cosine":
        prog = np.clip((t - warm) / max(1, total - warm), 0, 1)
        lr = min_lr + (peak - min_lr) * 0.5 * (1 + np.cos(np.pi * prog))
    else:  # ards, first study: decay scale 0.8 * total
        d = np.maximum(0, t - warm) / (0.8 * total)
        lr = peak / np.sqrt(1 + d * d)
    return np.where(t < warm, peak * t / warm, lr)


def areas(lr, lam=0.999):
    s1 = np.cumsum(lr)
    # only decreases count as annealing (warmup increases are ignored)
    drop = np.maximum(0.0, np.concatenate([[0.0], lr[:-1] - lr[1:]]))
    m = np.zeros_like(lr)
    acc = 0.0
    for i, x in enumerate(drop):
        acc = lam * acc + x
        m[i] = acc
    s2 = np.cumsum(m)
    return s1, s2


def law(theta, s1, s2):
    l0, a, alpha, c = theta
    return l0 + a * s1 ** (-alpha) - c * s2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}

    # 1. Final 125M runs -------------------------------------------------------
    runs = load_final_runs()
    seeds = (42, 43, 44)
    alg_nll = [runs[("algebraic", s)]["validation_loss"] for s in seeds]
    std_nll = [runs[("standard", s)]["validation_loss"] for s in seeds]
    alg_ppl = [runs[("algebraic", s)]["validation_perplexity"] for s in seeds]
    std_ppl = [runs[("standard", s)]["validation_perplexity"] for s in seeds]
    report["final_125m"] = {
        "nll": {"algebraic": alg_nll, "standard": std_nll, "paired": paired_analysis(alg_nll, std_nll)},
        "ppl_paired": paired_analysis(alg_ppl, std_ppl, margin_list=(0.1, 0.25, 0.5)),
        "seed_sd_nll": {"algebraic": float(np.std(alg_nll, ddof=1)),
                        "standard": float(np.std(std_nll, ddof=1)),
                        "variance_ratio_F": float(np.var(alg_nll, ddof=1) / np.var(std_nll, ddof=1)),
                        "F_test_p_two_sided": float(2 * min(
                            stats.f.cdf(np.var(alg_nll, ddof=1) / np.var(std_nll, ddof=1), 2, 2),
                            1 - stats.f.cdf(np.var(alg_nll, ddof=1) / np.var(std_nll, ddof=1), 2, 2)))},
        "validation_positions": 81920,
        "throughput_ratio": float(np.mean([runs[("algebraic", s)]["throughput_tokens_sec"] for s in seeds])
                                  / np.mean([runs[("standard", s)]["throughput_tokens_sec"] for s in seeds])),
        "clipped_every_step": all(abs(runs[k]["peak_gradient_norm"] - 1.0) < 1e-5 for k in runs),
    }

    # 2. Tuning sweep ------------------------------------------------------------
    ledger = json.loads(find("results/v1/tuning/sweep_ledger.json").read_text())
    table = {}
    for r in ledger["runs"]:
        key = f"{r['architecture']}/{r['candidate']}"
        table.setdefault(key, {"lr": r["hparams"]["learning_rate"], "wd": r["hparams"]["weight_decay"],
                               "schedule": r["hparams"]["schedule"], "nll": []})
        table[key]["nll"].append(r["validation_loss"])
    for v in table.values():
        v["mean_nll"] = float(np.mean(v["nll"]))
        v["sd_nll"] = float(np.std(v["nll"], ddof=1))
    best_alg = min((v["mean_nll"] for k, v in table.items() if k.startswith("algebraic")))
    best_std = min((v["mean_nll"] for k, v in table.items() if k.startswith("baseline")))
    report["tuning_600m"] = {
        "table": table,
        "best_gap_nll_alg_minus_std": best_alg - best_std,
        "final_gap_nll_alg_minus_std": float(np.mean(alg_nll) - np.mean(std_nll)),
        "note": "both arms chose the largest learning rate offered (1e-3): the optimum lies at "
                "or beyond the edge of the searched range",
    }

    # 3. Schedule areas and the annealing-law fit --------------------------------
    T_final, T_tune, warm = 2385, 573, 57
    sched = {
        "final_ards": schedule("ards", 1e-3, warm, T_final),
        "final_cosine": schedule("cosine", 1e-3, warm, T_final, 5e-5),
        "tune_ards": schedule("ards", 1e-3, warm, T_tune),
        "tune_cosine": schedule("cosine", 1e-3, warm, T_tune, 5e-5),
    }
    report["schedule_areas"] = {
        k: {"final_lr_ratio": float(v[-1] / 1e-3), "mean_lr_ratio": float(v.mean() / 1e-3),
            "S1": float(areas(v)[0][-1]), "S2": float(areas(v)[1][-1])}
        for k, v in sched.items()}

    # Fit the law to the standard arm's training cross-entropy traces.
    xs1, xs2, ys, groups = [], [], [], []
    def add_run(losses, total, peak, warmup, min_lr, gid):
        lr = schedule("cosine", peak, warmup, total, min_lr)
        s1, s2 = areas(lr)
        stride = max(1, total // 20)
        for j, val in enumerate(losses):
            step = j * stride  # losses_sample = losses[::stride]
            if step < 200 or step >= total:
                continue
            xs1.append(s1[step]); xs2.append(s2[step]); ys.append(val); groups.append(gid)
    for r in ledger["runs"]:
        # same peak LR and warmup as the final runs; only the budget differs
        if r["architecture"] != "baseline" or r["hparams"]["learning_rate"] != 1e-3:
            continue
        h = r["hparams"]
        add_run(r["losses_sample"], r["total_steps"], h["learning_rate"], h["warmup_steps"], h["min_lr"],
                f"tune-{r['candidate']}")
    for s in seeds:
        r = runs[("standard", s)]
        add_run(r["losses_sample"], r["total_steps"], 1e-3, 57, 5e-5, "final")
    xs1, xs2, ys = map(np.asarray, (xs1, xs2, ys))

    def resid(th):
        return law(th, xs1, xs2) - ys
    fit = optimize.least_squares(resid, x0=[2.5, 1.0, 0.5, 100.0],
                                 bounds=([0, 0, 0.01, 0], [10, 100, 3, 1e5]))
    th = fit.x
    rmse = float(np.sqrt(np.mean(fit.fun ** 2)))
    pred = {}
    for name in ("final_ards", "final_cosine"):
        s1, s2 = areas(sched[name])
        pred[name] = float(law(th, s1[-1], s2[-1]))
    # Residual bootstrap (residuals resampled in contiguous blocks of 4 points
    # within each trace to respect autocorrelation) for an interval on the penalty.
    rng = np.random.default_rng(0)
    gid = np.asarray(groups)
    base_fit = law(th, xs1, xs2)
    res = ys - base_fit
    penalties = []
    a1, a2 = areas(sched["final_ards"]); c1, c2 = areas(sched["final_cosine"])
    for _ in range(500):
        boot = np.empty_like(res)
        for g in set(groups):
            idx = np.where(gid == g)[0]
            blocks = [idx[i:i + 4] for i in range(0, len(idx), 4)]
            drawn = []
            while sum(len(b) for b in drawn) < len(idx):
                drawn.append(res[blocks[rng.integers(0, len(blocks))]])
            drawn = np.concatenate(drawn)[:len(idx)]
            boot[idx] = drawn
        yb = base_fit + boot
        f = optimize.least_squares(lambda t: law(t, xs1, xs2) - yb, x0=th,
                                   bounds=([0, 0, 0.01, 0], [10, 100, 3, 1e5]))
        penalties.append(law(f.x, a1[-1], a2[-1]) - law(f.x, c1[-1], c2[-1]))
    report["annealing_law_fit_standard_arm"] = {
        "params": {"L0": th[0], "A": th[1], "alpha": th[2], "C": th[3]},
        "rmse_train_loss": rmse, "points": int(len(ys)),
        "predicted_final_train_ce": pred,
        "predicted_ards_penalty_nats": pred["final_ards"] - pred["final_cosine"],
        "bootstrap_penalty_ci90": [float(np.percentile(penalties, 5)), float(np.percentile(penalties, 95))],
        "observed_gap_nats": float(np.mean(alg_nll) - np.mean(std_nll)),
        "caveat": "identified from two schedule configurations (573 and 2385 steps, both cosine); an extrapolation, not a measurement",
        "interpretation": "predicted cross-entropy cost of the first study's ARDS schedule for a model "
                          "whose loss obeys the standard arm's fitted annealing law",
    }

    # 4. Pilot gradient-norm traces ------------------------------------------------
    pilot = np.load(find("results/v1/pilot/losses.npz"))
    report["pilot_20m"] = {
        "fraction_steps_clipped": {
            "algebraic": float(np.mean(pilot["alg_grad_norms"] >= 0.999)),
            "standard": float(np.mean(pilot["base_grad_norms"] >= 0.999)),
        },
        "note": "recorded norms are post-clipping; a value of 1 means the step was clipped",
        "ppl": {"algebraic": 26.919, "standard": 25.226},
        "gap_nats": math.log(26.919 / 25.226),
    }
    (OUT / "v1_statistics.json").write_text(json.dumps(report, indent=2, default=float))
    f = report["final_125m"]["nll"]["paired"]
    a = report["annealing_law_fit_standard_arm"]
    print(f"125M paired NLL gap {f['mean']:.4f} [{f['ci95'][0]:.4f}, {f['ci95'][1]:.4f}], "
          f"P(alg worse)={f['posterior_p_algebraic_worse']:.3f}, seeds for 80% power: "
          f"{f['paired_seeds_for_80pct_power_at_observed_effect']}")
    print(f"ARDS penalty predicted {a['predicted_ards_penalty_nats']:.4f} nats "
          f"(90% {a['bootstrap_penalty_ci90']}), law rmse {a['rmse_train_loss']:.4f}")
    print(json.dumps(report["schedule_areas"], indent=1))
    print(json.dumps(report["pilot_20m"], indent=1))


if __name__ == "__main__":
    main()
