# sota-algebra: a second-generation algebraic Transformer

The [first study](https://github.com/tasmaikeni13/algebraic-transformer)
(tag `v1-original` here) built a causal language model whose training path
avoids exponentials, logarithms, trigonometric and sigmoid functions, and
compared it with a standard Transformer at 125M parameters on 2.5B
FineWeb-Edu tokens. The algebraic model lost narrowly: 25.774 vs 25.548
validation perplexity (paired gap 0.009 nats, 95% CI [−0.011, 0.028]).

This repository finds out **why it lost**, builds a **stronger algebraic
model that beats the standard one**, provides **Pallas TPU kernels for both
architectures**, and prepares the **125M × 3-seed main run** (implementation
and budget hyperparameter sweep; the main run itself is left to the author).

| Phase | Content | Status |
|---|---|---|
| 1 | Statistics, mathematics, and Monte Carlo diagnosis of the first study; ablations; a new algebraic design | done |
| 2 | Flash attention and fused head kernels for both arms | done |
| 3 | 60M models, FineWeb-Edu-1B, 1B tokens, one seed, both arms | done: **algebraic wins** |
| Main | 125M, 3 seeds, 2.5B tokens | implementation done; budget sweep running; main run left to the author |

## Headline results

**Phase 3 (60M parameters, 1B tokens, 16 TPU v4 chips, best of four learning
rates per arm):**

| model | held-out NLL | perplexity | tokens/s |
|---|---|---|---|
| standard (RMSNorm, SwiGLU, RoPE, softmax, cross-entropy) | 3.4170 | 30.48 | 2.70M |
| **algebraic (v2d)** | **3.3697** | **29.07** | 1.94M |

Every algebraic run (each of its four learning rates) beat the standard's
best run.

![Phase 3 training curves](results/figures/phase3_curves.png)
![Learning-rate sensitivity](results/figures/phase3_lr.png) Proxy experiments (33M parameters, 4 paired seeds) agree:
3.7574 ± 0.0016 vs 3.8182 ± 0.0029, paired difference −0.061 ± 0.004 nats
(t = −14).

**Why the first model lost** — first-study components swapped one at a time
into the standard model (proxy, Δ held-out NLL):

| first-study component | Δ NLL |
|---|---|
| OACE power-score head with vocabulary normalization | **+0.080** |
| rational (ARDS) schedule + weight decay 0.01 | **+0.060** |
| normalization without learned gains | **+0.037** |
| octic attention kernel (n=8, sink) | +0.003 |
| ALU activation | +0.001 |
| Cayley rotations (first-study grid) | −0.008 |

![First-study attribution](results/figures/v1_attribution.png)

The genuinely algebraic substitutions were free; the gap came from the
training objective, the schedule (it ended at 63% of peak learning rate), and
the missing gains. Details: [analysis/README.md](analysis/README.md); short
mathematics and proofs: [paper/theory.md](paper/theory.md).

## The second-generation algebraic model (v2d)

* **Algebraic maximum-likelihood head.** `p_j ∝ ρ(z_j/c)^n` with
  `ρ(x) = x + √(1+x²)`, n=8, c=2. Its log-loss gradient is algebraic,
  `(n/c)(1+(z_j/c)²)^(−1/2)(p_j − y_j)`, so the model trains on exact
  likelihood without evaluating a logarithm. The entrywise asinh compression
  escapes the softmax rank bottleneck (Monte Carlo: up to 18% lower KL at
  equal rank) and is the largest single source of the win.
![Head compression and the rank bottleneck](results/figures/head_and_bottleneck.png)

* **Asinh-power attention** `ρ(s/8)^16` with a learned per-head sink,
  QK-normalization, and an exact log-free max shift (`ρ(a)ρ(−b) = ρ(a ⊖ b)`);
  at 512 distractors it retrieves at least as precisely as softmax.
* Geometric Cayley rotations, ALU-GLU with β = ½, learned RMS gains,
  cosine AdamW (as specified for the new study).

A primitive-level audit (`scripts/audit_algebraic.py`) confirms the traced
training gradient — including the Pallas kernel bodies — contains no
exponential, logarithmic, trigonometric, hyperbolic, sigmoid or
error-function primitive.

**What transfers.** A standard backbone given the same head and QK-norm
matches the algebraic model (proxy −0.004 ± 0.005; 60M: 3.3716 vs 3.3697):
the algebraic backbone now ties the standard one, and the margin over the
standard baseline is carried by the algebraic head and QK-normalization.

## Kernels (Phase 2)

One Pallas TPU flash-attention implementation serves both arms (forward;
fused dK/dV/dQ backward; causal tile skipping with diagonal-only masking;
sinks). Forward + backward on one TPU v4 chip, 2048 tokens, head dim 64:

| kernel | 125M shape (8 × 12 heads) | 60M shape (8 × 8 heads) |
|---|---|---|
| this repo, softmax | 4.28 ms | 2.72 ms |
| this repo, asinh (algebraic) | 6.32 ms | 4.09 ms |
| JAX `pallas.ops.tpu.flash_attention` (used by the first study) | 22.14 ms | 14.55 ms |
| dense XLA softmax | 10.80 ms | 7.20 ms |

End to end on 16 chips the kernels make both arms 2–2.6× faster than the
first study (125M: standard 1.58M vs 0.60M tokens/s; algebraic 1.14M vs
0.57M). The vocabulary head uses XLA-dense logits (the fastest measured; the
lean algebraic head is within 14% of softmax cross-entropy); a fused Pallas
head is included as a low-memory option. Raw numbers: `results/kernels/`.

## Running the 125M main run

The budget sweep selects learning rates per architecture (three candidates ×
three seeds × 600M tokens, `configs/main_sweep.json`) and writes
`configs/main_selected.json`. Then, on a TPU v4-32 with hosts `w0..w3`
reachable over SSH (see `scripts/setup_workers.sh`):

```bash
python scripts/prepare_fineweb_edu.py --output-dir ~/data/fw25 --threads 128   # 2.5B + 600M + 10M tokens
python scripts/prepare_eval_benchmarks.py --output-dir ~/data/evals            # zero-shot tasks
python scripts/main_run.py main > main.json         # 3 seeds x {standard, algebraic}, 2,385 steps each
python scripts/run_slice.py main.json --group main  # rerun the same command to resume (checkpoints every 200 steps)
python scripts/evaluate_zero_shot.py runs/main/algebraic_s42 --out results/main/algebraic_s42_zero_shot.json
```

Expected wall time per run: about 30 minutes (standard) and 40 minutes
(algebraic) on 16 chips.

## Layout

| path | contents |
|---|---|
| `src/sota/` | model (`model.py`), algebraic and standard components, presets (`config.py`), trainer with checkpoint/resume (`train.py`) |
| `src/sota/kernels/` | Pallas flash attention (`flash.py`) and fused head (`fused_head.py`) |
| `analysis/` | Phase 1 statistics and Monte Carlo, ablation job lists, Phase 3 jobs and summary |
| `scripts/` | data preparation, 16-chip job queue (`run_queue.py`), full-slice launcher (`run_slice.py`), main run (`main_run.py`), zero-shot evaluation, benchmarks, audit |
| `configs/` | budget-sweep candidates and the selected main-run hyperparameters |
| `results/` | first-study records (`pretraining/`, `tuning/`, `pilot/`), `analysis/`, `phase3/`, `kernels/` |
| `paper/theory.md` | concise mathematics, proofs, and measured status of each claim |

The first study's modules (`src/*.py`, `src/kernels/`, its `scripts/`,
`tests/`, `formal/`) are still present and scheduled for removal; they are
preserved under the `v1-original` tag.

## Reproduce

```bash
python3 -m virtualenv ~/venv && ~/venv/bin/pip install -r requirements-tpu.txt
python scripts/prepare_fineweb_edu_1b.py --raw-dir <parquet dir> --output-dir ~/data/tok
JAX_PLATFORMS=cpu python -m pytest -q tests/test_sota_components.py   # TPU kernel tests: tests/test_*_tpu.py
python -m src.sota.train --preset algebraic --size proxy --steps 3000 --out runs/demo   # one chip
python analysis/phase1_jobs.py batch5 > jobs.json && python scripts/run_queue.py jobs.json --group p1
python analysis/phase3_jobs.py sweep > p3.json && python scripts/run_slice.py p3.json --group p3
```
