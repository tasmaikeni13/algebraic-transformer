# sota-algebra: a second-generation algebraic Transformer

The [first study](https://github.com/tasmaikeni13/algebraic-transformer)
(tag `v1-original` here) built a causal language model whose training path
avoids exponentials, logarithms, trigonometric and sigmoid functions, and
compared it with a standard Transformer at 125M parameters on 2.5B
FineWeb-Edu tokens. The algebraic model lost narrowly: 25.774 vs 25.548
validation perplexity (paired gap 0.009 nats, 95% CI [−0.011, 0.028]).

This repository finds out **why it lost**, builds a **stronger algebraic
model**, provides **Pallas TPU kernels for both architectures**, and prepares
the **125M × 3-seed main run** (implementation and hyperparameter sweep; the
main run itself is left to be launched by the author).

| Phase | Content | Status |
|---|---|---|
| 1 | Statistics, mathematics, and Monte Carlo diagnosis of the first study; ablations; a new algebraic design | done (iterating) |
| 2 | Flash attention and fused head kernels for both arms | done |
| 3 | 60M models on FineWeb-Edu-1B (1B tokens), one seed, both arms | in progress |
| Main | 125M, 3 seeds, 2.5B tokens: implementation + budget sweep | pending |

## Why the first algebraic model lost

Proxy experiments (33M parameters, 197M tokens, 2M held-out tokens; seed SD
≈ 0.003 nats) swap each first-study component into the standard model:

| first-study component | Δ held-out NLL |
|---|---|
| OACE power-score head with vocabulary normalization | **+0.080** |
| rational (ARDS) schedule + weight decay 0.01 | **+0.060** |
| normalization without learned gains | **+0.037** |
| octic attention kernel (n=8, sink) | +0.003 |
| ALU activation | +0.000 |
| Cayley rotations (first-study grid) | −0.009 |

The genuinely algebraic substitutions were free. The gap came from the
training objective, the schedule (which ended at 63% of peak learning
rate; an annealing-law fit predicts 0.115 nats lost at 125M), and the missing
gains. Details and every supporting analysis: [analysis/README.md](analysis/README.md);
short mathematics and proofs: [paper/theory.md](paper/theory.md).

## The second-generation algebraic model

* **Algebraic maximum-likelihood head.** Predictive distribution
  `p_j ∝ ρ(z_j/c)^n` with `ρ(x) = x + √(1+x²)`. The log-loss gradient of this
  head is algebraic, `(n/c)(1+(z_j/c)²)^(−1/2)(p_j − y_j)`, so the model
  trains on exact likelihood without ever evaluating a logarithm. With
  strong compression (n=8, c=2) it beats the softmax head, plausibly by
  escaping the softmax rank bottleneck.
* **Asinh-power attention** with a learned per-head sink and an exact,
  log-free max shift (`ρ(a)ρ(−b) = ρ(a ⊖ b)`), plus QK-normalization.
* Learned RMS gains, ALU-GLU with β = ½, Cayley rotations, cosine AdamW.

Current proxy results (held-out NLL, lower is better):

| model | NLL |
|---|---|
| standard (4 seeds) | 3.8182 ± 0.0029 |
| first-study algebraic model | 3.9608 |
| **second-generation algebraic (2 seeds)** | **3.7686 ± 0.0005** |
| standard + algebraic head (control) | 3.7849 |
| standard + QK-norm (control) | 3.7882 |

The controls show what transfers: the algebraic head and QK-norm each help a
standard backbone too. A primitive-level audit
(`scripts/audit_algebraic.py`) confirms that the algebraic model's traced
training gradient, including the Pallas kernel bodies, contains no
exponential, logarithmic, trigonometric, hyperbolic, sigmoid, or error-function
primitive.

## Kernels (Phase 2)

One Pallas TPU flash-attention implementation serves both arms (forward,
fused dK/dV/dQ backward, causal tile skipping, sinks). On one TPU v4 chip,
batch 8 × 12 heads × 2048 tokens × 64 dims, forward + backward:

| kernel | ms |
|---|---|
| this repo, softmax | 4.31 |
| this repo, asinh (algebraic) | 6.32 |
| JAX `pallas.ops.tpu.flash_attention` (used by the first study) | 22.14 |
| dense XLA softmax | 10.80 |

End to end at 60M parameters the kernels give a 2.2× training speedup to both
arms. Vocabulary heads use XLA-dense logits (the fastest option measured,
with a lean algebraic formulation within 14% of softmax cross-entropy); a
fused Pallas head is included as a low-memory alternative.

## Layout

| path | contents |
|---|---|
| `src/sota/` | model (`model.py`), algebraic and standard components, configs, trainer (`train.py`) |
| `src/sota/kernels/` | Pallas flash attention (`flash.py`) and fused head (`fused_head.py`) |
| `analysis/` | Phase 1 statistics, Monte Carlo, ablation and Phase 3 job lists |
| `scripts/` | data preparation, 16-chip job queue (`run_queue.py`), full-slice launcher (`run_slice.py`), benchmarks, audit |
| `results/` | first-study records (`pretraining/`, `tuning/`, `pilot/`) and new analysis outputs (`analysis/`) |
| `paper/theory.md` | concise mathematics, proofs, and measured status of each claim |

The first study's modules (`src/*.py`, `src/kernels/`, old `scripts/`,
`tests/`, `formal/`) are still present and are scheduled for removal; they are
preserved under the `v1-original` tag.

## Reproduce

```bash
python3 -m virtualenv ~/venv && ~/venv/bin/pip install -r requirements-tpu.txt
# data: FineWeb-Edu-1B (Phase 1/3) and the 2.5B FineWeb-Edu sample (main run)
python scripts/prepare_fineweb_edu_1b.py --raw-dir <parquet dir> --output-dir ~/data/tok
python scripts/prepare_fineweb_edu.py --output-dir ~/data/fw25 --threads 128
# tests (CPU; TPU kernel tests skip elsewhere)
JAX_PLATFORMS=cpu python -m pytest -q tests/test_sota_components.py
# one training run on one chip
python -m src.sota.train --preset algebraic --size proxy --steps 3000 --out runs/demo
# Phase 1 ablations over 16 chips, Phase 3 on the whole slice
python analysis/phase1_jobs.py batch4 > jobs.json && python scripts/run_queue.py jobs.json --group p1
python analysis/phase3_jobs.py sweep > p3.json && python scripts/run_slice.py p3.json --group p3
```
