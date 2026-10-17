# Algebraic Transformer

Causal language models with an algebraic attention kernel, Cayley positional rotations, and an output head whose exact log-loss gradient uses arithmetic and reciprocal square roots. The repository includes a matched standard Transformer, fused Pallas TPU kernels, reproducible training configurations, and public-task and recall evaluation.

The pure algebraic **model objective and its gradient** avoid exponential, logarithmic, trigonometric, sigmoid, and hyperbolic primitives. Initialization, the shared cosine learning-rate schedule, and metric reporting are outside that claim. See [the mathematics](paper/theory.md) and [primitive audit](scripts/audit_algebraic.py).

## Models and kernels

| Component | Standard | Pure algebraic |
|---|---|---|
| Normalization | Learned RMS gains | Learned RMS gains, plus QK normalization |
| Feed-forward | SwiGLU | ALU-GLU, β = ½ |
| Positions | RoPE | Geometric Cayley rotations |
| Attention | Causal softmax | `ρ(s/8)^16`, learned per-head sink |
| Output distribution | Softmax | `ρ(z/2)^8` |
| Objective gradient | Cross-entropy | Exact algebraic log-loss gradient |

Here `ρ(x) = x + √(1 + x²)`. Token embeddings and the output projection are tied.

Pallas provides tiled attention forward/backward, a vocabulary projection/loss kernel, and a QK gain-gradient reduction. The 350M and 1B configurations enable **fused attention and the fused head** for both arms. The head uses 128-token × 512-vocabulary tiles, verified at both widths. Dense components remain available for CPU checks. [Kernel measurements](paper/attention_performance.md), [numerical identities](paper/attention_exact.md), and [Lean certificates](formal/README.md) document scope and limitations.

## Install

Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
JAX_PLATFORMS=cpu python -m pytest -q tests/test_components.py tests/test_experiments.py
```

For TPU workers, install `requirements-tpu.txt` instead of `requirements.txt`. The pinned environment uses JAX 0.6.2. `scripts/tpu_python.sh` isolates one local TPU chip and uses `.venv/bin/python`, `~/venv/bin/python`, or `ALGEBRAIC_PYTHON`.

## 125M, 350M, and 1B experiments

All comparisons use **standard vs pure algebraic**, **one paired seed (42)**, a 32,768 TPU-aligned BPE tokenizer, a 2,048-token context, AdamW, a cosine schedule, and the same data seed. Models are matched in width, depth, feed-forward width, and tied vocabulary embedding; QK gains and learned sinks account for the small parameter difference.

| Configuration | Standard parameters | Algebraic parameters | Training budget | Steps | Per-device micro-batch |
|---|---:|---:|---:|---:|---:|
| [125M](configs/comparison_125m.json) | 110,120,448 | 110,122,128 | 2.5B tokens | 2,385 | 4 |
| [350M](configs/comparison_350m.json) | 332,449,792 | 332,453,248 | 7B tokens | 6,676 | 2 |
| [1B](configs/comparison_1b.json) | 975,277,056 | 975,281,920 | 20B tokens | 19,074 | 1 |

Global batch is 512 sequences. Budgets round up to complete optimizer steps: **7,000,293,376** and **20,000,538,624** token presentations. Corpus arrays contain exactly 7B and 20B tokens. The trainer samples deterministic contiguous windows, so token presentations are a compute budget rather than a claim of one pass through every corpus token.

**These full runs have not been launched here.** The implementations were smoke-tested with one synthetic optimizer step at each full model shape, including fused kernels and checkpoint reload. One-step checks also verify the production batch and gradient accumulation across all 16 TPU chips. Synthetic loss values are execution checks, not benchmark results. Reports are in `results/verification/`.

### Prepare data

The preparer discovers all required parquet shards at an immutable source revision, separates validation and training documents, inserts end-of-text separators, and publishes arrays only after every quota is filled. It fails if the source is exhausted.

```bash
# 350M corpus: exactly 7B training tokens, plus 10M held-out tokens.
python scripts/prepare_fineweb_edu.py --output-dir ~/data/fw7b \
  --train-tokens 7000000000 --threads 32

# 1B corpus: exactly 20B training tokens from the larger sample.
python scripts/prepare_fineweb_edu.py --output-dir ~/data/fw20b \
  --dataset-path sample/100BT --train-tokens 20000000000 --threads 32

python scripts/prepare_eval_benchmarks.py --output-dir ~/data/evals
python scripts/prepare_recall_benchmarks.py --config configs/comparison_350m.json \
  --output-dir ~/data/recall_350m
python scripts/prepare_recall_benchmarks.py --config configs/comparison_1b.json \
  --output-dir ~/data/recall_1b
```

Use the same absolute repository, environment, and data paths on all four TPU workers. Configure SSH aliases `w0,w1,w2,w3`; copy token arrays to every worker. Run the launchers from worker zero. Code synchronization is handled by [setup_workers.sh](scripts/setup_workers.sh). Token arrays require about **14GB** and **40GB** per worker, plus validation, raw-shard downloads, and optimizer checkpoints.

### Plan, train, and evaluate

Commands without `--run` print plans and start no training.

```bash
# 350M: inspect the plan and run comparison training and evaluation.
python scripts/run_comparison.py --scale 350m
python scripts/run_comparison.py --scale 350m --run

# 1B: inspect the plan and run comparison training and evaluation.
python scripts/run_comparison.py --scale 1b
python scripts/run_comparison.py --scale 1b --run
```

Full checkpoints go to `runs/comparison_<scale>/`; evaluation outputs go to `results/comparison_<scale>/`. Repeating the commands resumes checkpoints and reuses only validated, matching completed outputs. The pipeline trains both models before running all benchmarks. [Training details](paper/training.md) cover configuration, resume, and separate evaluation commands.

## Benchmark protocol

Every completed model receives **ARC-Easy, ARC-Challenge, HellaSwag, PIQA, OpenBookQA, WinoGrande, BoolQ, and LAMBADA**, plus **needle passkey retrieval, associative recall, multiple-needle retrieval, and latest-value recall**. The 350M and 1B runs retain the 125M prompts and scoring rules so results can be compared across scales.

Recall covers 256/512/1,024/2,048-token total budgets, five evidence positions, 32 trials per cell, 16 generated answer tokens, absent-evidence controls, and short-context controls. Evidence is never truncated. Outputs include predictions, checkpoint/dataset/source hashes, matched-model differences, per-cell intervals, and heatmaps. Single-seed comparisons report individual scores and paired differences; they do not estimate variance across training seeds. See the [evaluation protocol](paper/post_training_evaluation.md).

## Execution speed and throughput benchmark (TPU v4)

Execution speed and token throughput were empirically measured on Cloud TPU v4 (chip 0) for both Standard Causal Softmax and Pure Algebraic Attention ($\rho(s/8)^{16}$), recorded in `results/verification/benchmark_throughput.json`.

### Pallas FlashAttention Kernels (Batch 8, Heads 12, Head Dim 64)

| Sequence Length | Architecture | Forward Latency | Fwd+Bwd Latency | Step Throughput | Step TFLOP/s | Relative Throughput |
|---:|---|---:|---:|---:|---:|---:|
| 512 | Standard Softmax | 0.31 ms | 0.59 ms | 6,883,838 tok/s | 18.9 | 100.0% |
| 512 | Pure Algebraic | 0.36 ms | 0.88 ms | 4,640,130 tok/s | 12.8 | 67.4% |
| 1,024 | Standard Softmax | 0.65 ms | 1.48 ms | 5,526,717 tok/s | 30.4 | 100.0% |
| 1,024 | Pure Algebraic | 0.87 ms | 2.25 ms | 3,643,887 tok/s | 20.1 | 65.9% |
| 2,048 | Standard Softmax | 1.71 ms | 3.59 ms | 4,563,751 tok/s | 50.2 | 100.0% |
| 2,048 | Pure Algebraic | 1.77 ms | 4.55 ms | 3,596,646 tok/s | 39.6 | 78.8% |

At context length $T = 2048$, forward pass latency is within 4% (1.77 ms vs 1.71 ms).

### Full 125M Transformer Training Step (Batch 4, Context 2048, Vocab 32,768)

| Architecture | Parameters | Step Latency (Fwd+Bwd) | Token Throughput | Relative Speed |
|---|---:|---:|---:|---:|
| Standard Transformer | 110,120,448 | 74.37 ms | 110,155.3 tokens/s | 1.00x (100.0%) |
| Pure Algebraic Transformer | 110,122,128 | 79.99 ms | 102,406.9 tokens/s | 1.08x (93.0%) |

The Pure Algebraic Transformer achieves **93.0%** of standard transformer end-to-end training throughput while entirely eliminating transcendental exponentiation, log-sum-exp, and trigonometric functions.

## Verification and reproduction

```bash
# Actual full model shapes, one synthetic optimizer step each; no corpus training.
bash scripts/tpu_python.sh 0 scripts/smoke_comparison.py --scale 125m
bash scripts/tpu_python.sh 0 scripts/smoke_comparison.py --scale 350m
bash scripts/tpu_python.sh 0 scripts/smoke_comparison.py --scale 1b
bash scripts/tpu_python.sh 0 -m pytest -q \
  tests/test_flash_kernel_tpu.py tests/test_fused_head_tpu.py tests/test_qk_norm_tpu.py

# Distributed multi-worker synthetic verification across all 16 TPU v4 chips (4 workers):
python scripts/smoke_distributed.py --scale 125m
python scripts/smoke_distributed.py --scale 350m
python scripts/smoke_distributed.py --scale 1b

# Measure kernel speed and model throughput on TPU v4:
bash scripts/tpu_python.sh 0 scripts/benchmark_throughput.py
```

| Path | Contents |
|---|---|
| `src/algebraic/` | Model presets, trainer, numerical components, scoring, and recall generator |
| `src/algebraic/kernels/` | Pallas attention, fused vocabulary head, and QK reduction |
| `configs/` | Frozen comparisons and evaluation matrices |
| `scripts/` | Data preparation, experiment launchers, audits, and benchmark drivers |
| `analysis/` | Performance probes and numerical studies |
| `results/` | Verification and evaluation outputs |
| `paper/`, `formal/` | Mathematical notes, protocols, and Lean certificates |

TPU experiments were supported by Google's TPU Research Cloud (TRC).
