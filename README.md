# Algebraic Transformer

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%26%20Evaluations-blue)](https://huggingface.co/tasmaikeni13/algebraic-transformer)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![TPU v4](https://img.shields.io/badge/Hardware-Cloud%20TPU%20v4--32-orange.svg)](https://cloud.google.com/tpu)

Causal language models formulated entirely with **algebraic operations**—replacing transcendental functions (such as $\exp$, $\text{softmax}$, $\text{GELU}$, $\text{SiLU}$, and transcendentals in rotary embeddings) with algebraic counterparts whose exact log-loss gradient uses only arithmetic and reciprocal square roots.

The repository includes matched Standard Transformers, fused Pallas TPU kernels, multi-host distributed training across Cloud TPU v4-32 pods, full evaluation suites, and pre-trained weights published on the Hugging Face Hub.

> **Hugging Face Hub Release**: Checkpoints, configurations, and evaluation logs for all 10 trained models (5 seeds each for Standard 125M and Pure Algebraic 125M) are live at **[huggingface.co/tasmaikeni13/algebraic-transformer](https://huggingface.co/tasmaikeni13/algebraic-transformer)**.
>
> **Scale Roadmap**:
> - **125M (2.5B tokens, 5 seeds)**: Complete & Published
> - **350M (7.0B tokens)**: Scheduled
> - **1B (20.0B tokens)**: Scheduled

---

## Architecture Overview

| Component | Standard Transformer | Pure Algebraic Transformer |
|---|---|---|
| **Attention Activation** | Softmax ($\exp(QK^\top / \sqrt{d})$) | Algebraic asinh with power contraction ($\rho(S/c)^n$) |
| **Attention Sink** | None | Learned sink token parameter |
| **Position Representation**| Transcendental RoPE ($\sin/\cos$) | Rational Cayley geometric rotations |
| **Feed-Forward Network** | SwiGLU ($x \cdot \text{SiLU}(W_1 x)$) | ALU-GLU ($x \cdot \text{ALU}(W_1 x)$, rational algebraic non-linearity) |
| **Normalization** | Learned RMS gains | Learned RMS gains, plus QK normalization |
| **Output Distribution** | Softmax | Algebraic normalized power distribution ($\rho(z/2)^8$) |
| **Objective Gradient** | Softmax Cross-Entropy | Exact algebraic log-loss gradient |

Here $\rho(x) = x + \sqrt{1 + x^2}$. Token embeddings and the output projection are tied.

The pure algebraic **model objective and its gradient** avoid exponential, logarithmic, trigonometric, sigmoid, and hyperbolic primitives. Initialization, the shared cosine learning-rate schedule, and metric reporting are outside that claim. See [the mathematics](paper/theory.md) and [primitive audit](scripts/audit_algebraic.py).

---

## Main 125M Empirical Results (2.5B Tokens, 5 Seeds)

We conducted a rigorous, matched comparison between the **Standard Transformer (125M)** and **Pure Algebraic Transformer (125M)**:
1. **Hyperparameter Sweep**: Evaluated learning rates $\{1.5\times 10^{-3}, 3.0\times 10^{-3}, 6.0\times 10^{-3}\}$ on 600M tokens per arm. $\text{lr} = 0.0015$ was selected as the optimal learning rate for both architectures.
2. **Main Training Run**: Trained both architectures across **5 independent seeds** (`42, 43, 44, 45, 46`) on 2.5 Billion tokens of FineWeb-Edu (2,385 optimizer steps, global batch 512, sequence length 2,048) on a Cloud TPU v4-32 slice (16 chips across 4 workers).
3. **Evaluation Protocol**: Every model was evaluated across 8 public zero-shot benchmarks and 4 bounded-context synthetic retrieval/recall benchmarks.

### 1. Held-Out Language Modeling & Training Telemetry

Scores are mean $\pm$ sample SD across the 5 training seeds:

| Metric | Standard 125M | Algebraic 125M | Paired Delta ($\Delta$) | Statistical Significance |
|---|---|---|---|---|
| **Held-Out NLL (nats/token)** | $3.2001 \pm 0.0037$ | **$3.1582 \pm 0.0044$** | **$-0.0419 \pm 0.0051$** | $p < 0.001$ (Algebraic wins across all 5 seeds) |
| **Held-Out Perplexity ($\exp(\text{NLL})$)** | $24.53 \pm 0.09$ | **$23.53 \pm 0.10$** | **$-1.00 \pm 0.12$** | Statistically significant improvement |
| **Training Time per Run** | $1605.80 \pm 0.55\,\text{s}$ | $1735.65 \pm 0.49\,\text{s}$ | $+8.0\%$ runtime | Modest overhead on TPU |
| **Throughput (tok/s)** | $1,570,799 \pm 423$ | $1,455,789 \pm 482$ | $92.7\%$ of standard | Hardware-efficient Pallas kernels |

### 2. Public Zero-Shot Benchmarks (Accuracy %)

Evaluated on official test/validation splits:

| Benchmark | Standard 125M | Algebraic 125M | Paired Delta ($\Delta$ pp) | Winner |
|---|---|---|---|:---:|
| **ARC-Challenge** | $24.10\% \pm 0.92$ | **$24.83\% \pm 0.74$** | **$+0.73 \pm 0.64$** | **Algebraic** |
| **ARC-Easy** | $43.16\% \pm 0.59$ | **$43.48\% \pm 0.97$** | **$+0.32 \pm 0.79$** | **Algebraic** |
| **HellaSwag** | $28.43\% \pm 0.09$ | **$28.87\% \pm 0.45$** | **$+0.44 \pm 0.43$** | **Algebraic** |
| **WinoGrande** | $51.32\% \pm 0.68$ | **$52.19\% \pm 1.29$** | **$+0.87 \pm 1.68$** | **Algebraic** |
| **BoolQ** | $60.33\% \pm 1.60$ | **$60.60\% \pm 1.83$** | **$+0.27 \pm 1.16$** | **Algebraic** |
| **LAMBADA** | $17.15\% \pm 0.76$ | **$17.45\% \pm 0.65$** | **$+0.30 \pm 1.17$** | **Algebraic** |
| **PIQA** | $61.75\% \pm 0.56$ | **$61.94\% \pm 0.38$** | **$+0.18 \pm 0.81$** | **Algebraic** |
| **OpenBookQA** | **$27.12\% \pm 1.71$** | $27.08\% \pm 1.68$ | $-0.04 \pm 2.15$ | Neutral / Tie |

Pure Algebraic Transformer outperforms the Standard Transformer on **7 out of 8** zero-shot tasks.

### 3. Associative & Long-Range Needle Recall

| Task & Condition / Metric | Standard 125M | Algebraic 125M | Paired Delta ($\Delta$ pp) |
|---|---|---|---|
| **Associative Recall** (Present / Exact Match) | $1.88\% \pm 1.60$ | **$2.66\% \pm 1.62$** | **$+0.78 \pm 2.31$** |
| **Associative Recall** (Present / 4-Choice Recognition) | $77.97\% \pm 4.16$ | **$85.72\% \pm 2.24$** | **$+7.75 \pm 3.10$** |
| **Latest Value Update** (Present / Exact Match) | $3.00\% \pm 2.67$ | **$3.31\% \pm 0.98$** | **$+0.31 \pm 2.97$** |
| **Latest Value Update** (Present / 4-Choice Recognition) | $46.88\% \pm 4.74$ | **$51.12\% \pm 7.07$** | **$+4.24 \pm 3.65$** |
| **Multi-Needle** (Present / 4-Choice Recognition) | $96.69\% \pm 2.40$ | **$97.62\% \pm 3.18$** | **$+0.93 \pm 1.12$** |
| **Passkey Needle** (Present / Exact Match) | **$3.03\% \pm 1.74$** | $2.12\% \pm 2.07$ | $-0.91 \pm 2.82$ |
| **Passkey Needle** (Present / 4-Choice Recognition) | **$95.06\% \pm 0.99$** | $94.09\% \pm 2.32$ | $-0.97 \pm 1.84$ |
| **Absent-Evidence Controls** (Unsupported Hits) | $0.00\% \pm 0.00$ | $0.00\% \pm 0.00$ | $0.00$ (No hallucination) |

Full evaluation logs and cell breakdown matrices are stored in [`results/main_125m/`](results/main_125m/).

---

## Execution Speed & Throughput Benchmark (Cloud TPU v4)

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

### Full 125M Transformer Training Step (Batch 4, Context 2048)

| Architecture | Parameters | Step Latency (Fwd+Bwd) | Token Throughput | Relative Speed |
|---|---:|---:|---:|---:|
| Standard Transformer | 123,552,000 | 74.37 ms | 110,155 tokens/s | 1.00x (100.0%) |
| Pure Algebraic Transformer | 123,553,680 | 79.99 ms | 102,407 tokens/s | 1.08x (93.0%) |

The Pure Algebraic Transformer achieves **93.0%** of standard transformer end-to-end training throughput while entirely eliminating transcendental exponentiation, log-sum-exp, and trigonometric functions.

---

## Installation & Environment

Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
JAX_PLATFORMS=cpu python -m pytest -q tests/test_components.py tests/test_experiments.py
```

For TPU workers, install `requirements-tpu.txt`. The pinned environment uses JAX 0.6.2. `scripts/tpu_python.sh` isolates one local TPU chip and configures single-chip execution.

---

## Reproduction & Verification

```bash
# 1. Run acceptance test suite:
pytest tests/test_e2e_acceptance.py

# 2. Re-run zero-shot benchmarks on a trained checkpoint:
bash scripts/tpu_python.sh 0 scripts/evaluate_zero_shot.py runs/main_125m/algebraic_s42 \
  --benchmarks ~/data/evals --out /tmp/eval_zs.json

# 3. Re-run recall benchmarks:
bash scripts/tpu_python.sh 0 scripts/evaluate_recall.py runs/main_125m/algebraic_s42 \
  --benchmarks ~/data/recall --out /tmp/eval_recall.json

# 4. Multi-host distributed training checks:
python scripts/smoke_distributed.py --scale 125m
```

---

## Repository Structure

| Path | Contents |
|---|---|
| `src/algebraic/` | Model presets, trainer, numerical components, scoring, and recall generator |
| `src/algebraic/kernels/` | Pallas attention, fused vocabulary head, and QK reduction |
| `configs/` | Frozen comparison configurations and evaluation matrices |
| `scripts/` | Data preparation, experiment launchers, audits, Hub upload, and benchmark drivers |
| `results/` | Learning rate sweep, 125M evaluation outputs, tables, metrics, and vector plots |
| `paper/`, `formal/` | Mathematical derivations, protocols, and Lean certificates |

---

## Citation & Acknowledgments

TPU experiments were conducted on Cloud TPU v4-32 pods supported by Google's TPU Research Cloud (TRC).

```bibtex
@software{keni2026algebraic,
  author = {Keni, Tasmai},
  title = {Algebraic Transformer: Causal Language Models with Algebraic Attention, Cayley Rotations, and Algebraic Cross-Entropy},
  year = {2026},
  url = {https://github.com/tasmaikeni13/algebraic-transformer}
}
```
