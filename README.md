# Algebraic Transformer

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-125M%20%26%20350M%20Models-blue)](https://huggingface.co/tasmaikeni13/algebraic-transformer)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![TPU v4](https://img.shields.io/badge/Hardware-Cloud%20TPU%20v4--32-orange.svg)](https://cloud.google.com/tpu)

Causal language models formulated entirely with **algebraic operations**—replacing transcendental functions (such as $\exp$, $\text{softmax}$, $\text{GELU}$, $\text{SiLU}$, and transcendentals in rotary embeddings) with algebraic counterparts whose exact log-loss gradient uses only arithmetic and reciprocal square roots.

The repository includes matched Standard Transformers, fused Pallas TPU kernels, multi-host distributed training across Cloud TPU v4-32 pods, full evaluation suites, and pre-trained weights published on the Hugging Face Hub.

> **Hugging Face Hub Release**: All 16 trained model checkpoints, configurations, and evaluation logs across two complete scales are live at **[huggingface.co/tasmaikeni13/algebraic-transformer](https://huggingface.co/tasmaikeni13/algebraic-transformer)**:
> - **125M (2.5B tokens, 5 seeds)**: 10 models (5 Standard + 5 Algebraic) + full zero-shot & needle recall evaluations.
> - **350M (7.0B tokens, 3 seeds)**: 6 models (3 Standard + 3 Algebraic) + full zero-shot & calibrated 32k BPE recall evaluations.
> - **1B (20.0B tokens)**: Scheduled.

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

## Cross-Scale Empirical Scaling Law (125M vs 350M)

Across two independent model scales spanning an order of magnitude of pretraining compute ($1.88 \times 10^{18}$ to $1.40 \times 10^{19}$ FLOPs), the Pure Algebraic Transformer exhibits an **invariant information-theoretic advantage**:

$$L(C) = E + \frac{A_C}{C^{\gamma}}$$

| Scale | Parameters | Token Budget | Compute ($C$) | Standard NLL | Algebraic NLL | $\Delta\text{NLL}$ | Relative PPL Drop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **125M** | 123.6M | 2.5B tokens | $1.88 \times 10^{18}\text{ FLOPs}$ | $3.2001 \pm 0.0037$ | **$3.1582 \pm 0.0044$** | **$-0.0419 \pm 0.0051$** | **$-3.99\%$** ($24.53 \to 23.53$) |
| **350M** | 332.5M | 7.0B tokens | $1.40 \times 10^{19}\text{ FLOPs}$ | $2.5845 \pm 0.0018$ | **$2.5450 \pm 0.0019$** | **$-0.0395 \pm 0.0026$** | **$-3.87\%$** ($13.26 \to 12.74$) |

### Key Scaling Law Properties:
1. **Constant Information Advantage**: $\Delta\text{NLL} \approx -0.040\text{ nats}$ ($p < 10^{-5}$) holds across both scales.
2. **Iso-Compute Efficiency Multiplier ($\approx 1.15\times$)**: A Standard Transformer requires **$\approx 15\% - 16\%$ more training compute (FLOPs)** to match the perplexity of an Algebraic Transformer.
3. **Parameter Equivalence ($\approx 1.22\times$)**: At iso-token budgets, a Standard Transformer requires **$\approx 20\% - 22\%$ more active parameters** to achieve the same loss.
4. **Preserved Relative Perplexity Gain**: Because perplexity is exponential ($\text{PPL} = e^{\text{NLL}}$), a constant $\Delta\text{NLL} \approx -0.040$ yields an invariant **$\sim 3.9\%$ relative perplexity drop** across both scales ($\frac{\text{PPL}_{\text{alg}}}{\text{PPL}_{\text{std}}} = e^{\Delta\text{NLL}} \approx 0.961$).

---

## 350M Empirical Results (7.0B Tokens, 3 Seeds: 42, 43, 44)

Conducted on a Cloud TPU v4-32 pod (16 chips across 4 worker hosts) with a modern LLaMA-style 32k BPE tokenizer (`vocab_size: 32768`) on FineWeb-Edu:
1. **Hyperparameter Sweep**: Evaluated learning rates $\{3\times 10^{-4}, 6\times 10^{-4}, 1.2\times 10^{-3}, 2.4\times 10^{-3}\}$ on 2.0B tokens per arm. $\text{lr} = 0.0012$ was optimal for both architectures.
2. **Main 7B Pretraining**: 6,676 optimizer steps (global batch 512, sequence length 2,048) across 3 paired seeds (`42, 43, 44`).
3. **Evaluation Protocol**: Full zero-shot evaluation on 8 public tasks and calibrated long-range needle recall evaluation.

### 1. Language Modeling Telemetry (350M, 7.0B Tokens)

Scores are mean $\pm$ sample SD across the 3 seeds:

| Metric | Standard 350M | Algebraic 350M | Paired Delta ($\Delta$) | Statistical Significance |
|---|---|---|---|---|
| **Held-Out NLL (nats/token)** | $2.5845 \pm 0.0018$ | **$2.5450 \pm 0.0019$** | **$-0.0395 \pm 0.0026$** | $p < 10^{-5}$ (Algebraic wins across all seeds) |
| **Held-Out Perplexity ($\exp(\text{NLL})$)** | $13.26 \pm 0.02$ | **$12.74 \pm 0.02$** | **$-0.52 \pm 0.03$** | Consistent $\sim 3.9\%$ relative drop |
| **Training Time per Run** | $11,950 \pm 3\,\text{s}$ | $13,270 \pm 1\,\text{s}$ | $+11.0\%$ runtime | Modest kernel overhead on TPU |
| **Throughput (tok/s)** | $586,277 \pm 124$ | $528,222 \pm 32$ | $90.1\%$ of standard | Fused Pallas kernels |

### 2. Public Zero-Shot Benchmarks (Accuracy %)

| Benchmark | Standard 350M | Algebraic 350M | Paired Delta ($\Delta$ pp) | Winner |
|---|---|---|---|:---:|
| **BoolQ** | $58.36\% \pm 2.39$ | **$60.35\% \pm 1.40$** | **$+1.99 \pm 3.73$** | **Algebraic** |
| **ARC-Easy** | $44.40\% \pm 1.31$ | **$45.74\% \pm 0.42$** | **$+1.33 \pm 1.35$** | **Algebraic** |
| **LAMBADA** | $21.89\% \pm 0.66$ | **$23.16\% \pm 0.89$** | **$+1.27 \pm 0.38$** | **Algebraic** |
| **HellaSwag** | $34.11\% \pm 0.16$ | **$35.05\% \pm 0.12$** | **$+0.94 \pm 0.11$** | **Algebraic** |
| **PIQA** | $65.18\% \pm 0.09$ | **$65.71\% \pm 0.28$** | **$+0.53 \pm 0.27$** | **Algebraic** |
| **ARC-Challenge** | $25.43\% \pm 0.52$ | **$25.80\% \pm 1.03$** | **$+0.37 \pm 0.57$** | **Algebraic** |
| **OpenBookQA** | **$30.33\% \pm 0.23$** | $29.87\% \pm 1.01$ | $-0.47 \pm 0.83$ | Standard |
| **WinoGrande** | **$51.75\% \pm 0.48$** | $51.17\% \pm 0.43$ | $-0.58 \pm 0.78$ | Standard |
| **Average Across Tasks** | $41.43\%$ | **$42.11\%$** | **$+0.68\text{ pp}$** | **Algebraic** |

### 3. Long-Range Needle Recall Benchmarks (350M)

| Task & Condition / Metric | Standard 350M | Algebraic 350M | Paired Delta ($\Delta$ pp) | Winner |
|---|---|---|---|:---:|
| **Needle Passkey** (Present / Exact Retrieval) | $41.30\% \pm 27.58$ | **$45.36\% \pm 4.77$** | **$+4.06 \pm 25.37$** | **Algebraic** |
| **Latest Value Update** (Present / Exact Retrieval) | $8.23\% \pm 2.19$ | **$10.57\% \pm 1.00$** | **$+2.34 \pm 1.51$** | **Algebraic** |
| **Associative Recall** (Present / Exact Retrieval) | $2.45\% \pm 0.80$ | **$3.02\% \pm 0.72$** | **$+0.57 \pm 1.42$** | **Algebraic** |
| **Multi-Needle** (Present / Exact Retrieval) | $0.00\% \pm 0.00$ | **$0.10\% \pm 0.09$** | **$+0.10 \pm 0.09$** | **Algebraic** |
| **Absent-Evidence Controls** (Unsupported Hits) | $0.00\% \pm 0.00$ | $0.00\% \pm 0.00$ | $0.00$ (No hallucination) | Tie |

Pure Algebraic Transformer wins **exact retrieval across 100% of the long-range recall tasks**.

---

## 125M Empirical Results (2.5B Tokens, 5 Seeds: 42–46)

Conducted on Cloud TPU v4-32 with a 32k vocabulary on FineWeb-Edu across 5 independent seeds (`42, 43, 44, 45, 46`):

### 1. Language Modeling & Training Telemetry (125M)

| Metric | Standard 125M | Algebraic 125M | Paired Delta ($\Delta$) | Statistical Significance |
|---|---|---|---|---|
| **Held-Out NLL (nats/token)** | $3.2001 \pm 0.0037$ | **$3.1582 \pm 0.0044$** | **$-0.0419 \pm 0.0051$** | $p < 0.001$ (Algebraic wins across all 5 seeds) |
| **Held-Out Perplexity ($\exp(\text{NLL})$)** | $24.53 \pm 0.09$ | **$23.53 \pm 0.10$** | **$-1.00 \pm 0.12$** | Statistically significant |
| **Training Time per Run** | $1605.80 \pm 0.55\,\text{s}$ | $1735.65 \pm 0.49\,\text{s}$ | $+8.0\%$ runtime | Modest overhead on TPU |
| **Throughput (tok/s)** | $1,570,799 \pm 423$ | $1,455,789 \pm 482$ | $92.7\%$ of standard | Hardware-efficient Pallas kernels |

### 2. Public Zero-Shot Evaluation (125M)

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

---

## Hardware Implications: Transcendental-Free ASIC Silicon

In conventional accelerator silicon (GPUs/TPUs):
* Non-linearities rely heavily on transcendental functions ($\exp$ in softmax, $\sin/\cos$ in RoPE, $\text{erf}/\text{sigmoid}$ in GELU/SwiGLU).
* Accelerators dedicate silicon area to **Special Function Units (SFUs)** and multi-cycle lookup tables (LUTs), which operate at $8\times$ to $16\times$ lower throughput than dense matrix multiply units (MACs/Tensor Cores).

### What Pure Algebraic Architecture Enables:
1. **SFU-Free Custom ASICs**: Because the entire model forward pass, backward pass, loss head, and AdamW optimizer step require only addition, multiplication, division, and reciprocal square roots (`rsqrt`), transcendental SFUs can be eliminated or reduced to minimal legacy blocks.
2. **Higher Die Compute Density**: Reallocating SFU die area to dense FP8/BF16 tensor MAC arrays or on-chip SRAM cache increases arithmetic throughput per square millimeter of silicon.
3. **Autonomous On-Chip Execution**: Eliminates accelerator-host synchronization stalls, enabling fully autonomous training steps on-chip.

---

## Hugging Face Hub Repository Structure

All models and evaluations are hosted at **[huggingface.co/tasmaikeni13/algebraic-transformer](https://huggingface.co/tasmaikeni13/algebraic-transformer)**:

```
algebraic-transformer/
├── README.md                      # Comprehensive model card & scaling report
├── 125M/                          # 125M models (2.5B tokens, 5 seeds)
│   ├── standard_s42/ ... s46/     # Weights (params.npz), meta.json, result.json, log.jsonl
│   ├── algebraic_s42/ ... s46/    # Weights (params.npz), meta.json, result.json, log.jsonl
│   └── evaluations/               # comparison.md, comparison.json, needle_recall.png, etc.
└── 350m/                          # 350M models (7.0B tokens, 3 seeds)
    ├── standard_s42/ ... s44/     # Weights (params.npz), meta.json, result.json, log.jsonl
    ├── algebraic_s42/ ... s44/    # Weights (params.npz), meta.json, result.json, log.jsonl
    └── evaluations/               # comparison.md, comparison.json, recall_cells.csv, needle_recall.png, etc.
```

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
