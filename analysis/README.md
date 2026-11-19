# Reproducible Analysis Suite for the Algebraic Transformer

This directory contains the numerical studies, formal real-arithmetic certificates, hardware performance probes, and publication figure generators for the Algebraic Transformer.

All 38 analysis scripts are organized into six modular subdirectories, orchestrated by a unified CLI dispatcher `run_analysis.py`. Backward-compatible symlinks and dynamic module exports are maintained at the `analysis/` root.

---

## Directory Layout

```text
analysis/
├── README.md                           # Master index and command guide
├── run_analysis.py                     # Unified entry point for all analysis jobs
├── figures/                            # Publication plotting and figure generation
│   ├── plot_figures.py                 # Scaling laws, factorial ablation, zero-shot plots
│   └── summarize_runs.py               # Aggregates run metrics across seeds and sweeps
├── kernels/                            # Pallas TPU FlashAttention & Fused Head benchmarks
│   ├── attention_performance.py        # Causal forward/backward paired benchmark
│   ├── attention_resident_probe.py     # Whole-head VMEM resident K/V schedule
│   ├── attention_head_resident_probe.py# Score-tile sweeps (128x512) and VMEM bounds
│   ├── attention_streamed_probe.py     # Query-streamed backward schedules
│   ├── attention_backward_schedule_probe.py # MXU/VPU overlap and live buffer reduction
│   ├── attention_checkpoint_probe.py   # Full-model synthetic AdamW speed smoke
│   ├── attention_segmented_probe.py    # Score-row segmentation probes
│   ├── attention_statistics.py         # Hardware timing bootstrap statistics
│   ├── attention_head_statistics.py    # Whole-head paired confirmation statistics
│   ├── attention_followup_statistics.py# Wider-link timing summaries
│   ├── head_gradient_cache_probe.py    # Vocabulary head gradient caching
│   ├── qk_rms_kernel.py                # Experimental Pallas RMS normalization
│   ├── qk_normalization_probe.py       # QK normalization VJP verification
│   └── qk_model_validation.py          # Complete-model QK checks
├── certificates/                       # Formal real-arithmetic certificates & numerics
│   ├── activation_analysis.py          # ALU vs SiLU numerical moments and derivatives
│   ├── attention_numerics.py           # Float64 oracle analysis of n=16, c=8 attention
│   ├── attention_polynomial.py         # Degree-4 Bernstein rational certificates
│   ├── attention_interpolation_certificate.py # Quintic Hermite rational error bounds
│   ├── attention_newton_certificate.py # Newton inverse-root polynomial error bounds
│   ├── attention_anchored_certificate.py # Uniform real bounds for anchored roots
│   ├── attention_rational_radical.py   # Certified rational sqrt(1+x^2) approximations
│   └── attention_rational_radical_probe.py # Benchmarks of rational root candidates
├── profiling/                          # Component and full-step profiling
│   ├── model_component_performance.py  # Layer-by-layer TPU timing breakdown
│   ├── model_step_performance.py       # Full optimizer step latency breakdown
│   └── model_profile.py                # JAX/Pallas memory and execution profiler
├── experimental/                       # Opt-in research links (not in main presets)
│   ├── attention_link_analysis.py      # Temperature routing for affine/power links
│   ├── attention_link_probe.py         # Isolated Pallas rational/affine probe
│   ├── attention_link_statistics.py    # Routing and accuracy statistics
│   ├── attention_recall_probe.py       # Toy associative recall probes
│   ├── attention_radical_probe.py      # Alternative radical evaluations
│   ├── attention_interpolation_probe.py# Hermite weight/VJP interpolation
│   ├── attention_projective_probe.py   # Row normalization via matrix products
│   └── attention_affine_guard_probe.py # Range guard and clip-free branches
└── scaling/                            # Multi-scale sweeps and evaluation verification
    ├── scaling_jobs.py                 # Multi-host job generator
    ├── scaling_summary.py              # Power law loss curve extraction
    └── verify_head_eval.py             # Independent float64 evaluation verification
```

---

## Unified CLI Runner (`run_analysis.py`)

The analysis suite provides a central CLI dispatcher supporting modular subcommands, direct script shortcuts, and index listing:

### List all modules and available scripts
```bash
python analysis/run_analysis.py list
```

### Run scripts via module subcommand
```bash
# Publication figure generation
python analysis/run_analysis.py figures plot_figures

# Numerical certificates on CPU
JAX_PLATFORMS=cpu python analysis/run_analysis.py certificates activation_analysis
JAX_PLATFORMS=cpu python analysis/run_analysis.py certificates attention_polynomial
JAX_PLATFORMS=cpu python analysis/run_analysis.py certificates attention_interpolation_certificate

# Generate scaling job queues
python analysis/run_analysis.py scaling scaling_jobs sweep
```

### Run scripts via direct shortcut
```bash
python analysis/run_analysis.py plot_figures
python analysis/run_analysis.py activation_analysis
```

### Module help
```bash
python analysis/run_analysis.py figures --help
python analysis/run_analysis.py kernels --help
python analysis/run_analysis.py certificates --help
```

---

## Modular Subpackage Index

### 1. `analysis.figures`
Generates publication-quality plots and tabular metrics from recorded pretraining JSON outputs:
* `plot_figures.py`: Creates architecture computation graphs (`architecture.png`), scaling overview curves (`results_overview.png`), and zero-shot benchmark comparisons (`public_tasks.png`).
* `summarize_runs.py`: Collects training outputs across runs and computes paired delta statistics against reference baselines.

### 2. `analysis.kernels`
Probes and benchmarks TPU Pallas kernels, VMEM resident memory schedules, and gradient caching:
* `attention_performance.py`: Benchmarks forward and backward causal attention passes against FlashAttention / Splash baselines.
* `attention_resident_probe.py`: Evaluates VMEM-resident K/V schedules avoiding dQ intermediate HBM writes.
* `attention_head_resident_probe.py`: Sweeps score tile dimensions ($128 \times 512$) for whole-head residency.
* `attention_streamed_probe.py`, `attention_backward_schedule_probe.py`: Evaluates MXU/VPU overlap and query-streamed schedules.
* `attention_checkpoint_probe.py`, `attention_segmented_probe.py`: Synthetic optimizer step tests with selective residual saving.
* `attention_statistics.py`, `attention_head_statistics.py`, `attention_followup_statistics.py`: Bootstrap confidence intervals for paired timing ratios.
* `head_gradient_cache_probe.py`: Measures memory and throughput of vocabulary-head VJP gradient caching.
* `qk_rms_kernel.py`, `qk_normalization_probe.py`, `qk_model_validation.py`: Verifies Pallas QK RMS normalization and gradient consistency.

### 3. `analysis.certificates`
Formal numerical error bounds and real-arithmetic certificates:
* `activation_analysis.py`: Gaussian-weighted $L_2$ distance and derivative moments comparing ALU-GLU against SiLU/GELU over $\beta$.
* `attention_numerics.py`: Float64 oracle verification for $n=16, c=8$ attention across gradient stress regimes.
* `attention_polynomial.py`: Degree-4 Bernstein rational certificates bounding approximation error.
* `attention_interpolation_certificate.py`: Quintic Hermite rational error bounds with certified interval arithmetic.
* `attention_newton_certificate.py`: Newton-Raphson inverse-root polynomial error bounds.
* `attention_anchored_certificate.py`: Uniform real bounds and row-error bounds for anchored radical evaluations.
* `attention_rational_radical.py`, `attention_rational_radical_probe.py`: Certified rational approximations of $\sqrt{1 + x^2}$ for purely algebraic execution.

### 4. `analysis.profiling`
Execution latency and memory profiling across training steps:
* `model_component_performance.py`: Layer-by-layer TPU execution breakdown across attention, FFN, and normalization.
* `model_step_performance.py`: Measures end-to-end optimizer step latency on pinned Cloud TPU chips.
* `model_profile.py`: JAX and Pallas memory profiler capturing VMEM and HBM buffer residency.

### 5. `analysis.experimental`
Exploratory research investigations into candidate attention links:
* `attention_link_analysis.py`, `attention_link_probe.py`, `attention_link_statistics.py`: Evaluates temperature routing for candidate affine and power links.
* `attention_recall_probe.py`: Toy associative retrieval studies under synthetic key-value pairs.
* `attention_radical_probe.py`: Alternative radical approximations retaining exact asinh semantics.
* `attention_interpolation_probe.py`: Bounded Hermite interpolation probes.
* `attention_projective_probe.py`, `attention_affine_guard_probe.py`: Explores row normalization rearrangements and range guards.

### 6. `analysis.scaling`
Multi-scale pretraining sweeps and independent validation:
* `scaling_jobs.py`: Generates parallel multi-host job configurations for TPU cluster execution.
* `scaling_summary.py`: Extracts power-law scaling exponents from multi-scale training trajectories.
* `verify_head_eval.py`: Independent float64 validation of held-out validation cross-entropy for saved checkpoints.

---

## Execution Environment

* **CPU Execution**: Certificate scripts and summary tools execute cleanly on standard CPUs using `JAX_PLATFORMS=cpu`:
  ```bash
  JAX_PLATFORMS=cpu python analysis/run_analysis.py certificates activation_analysis
  ```
* **TPU Execution**: Kernel performance probes targeting Cloud TPU v4 hardware use the repository's launcher:
  ```bash
  bash scripts/tpu_python.sh 0 analysis/kernels/attention_performance.py --blocks 512 --out results/verification/tpu_bench.json
  ```

---

## Compatibility Note

For compatibility with historical scripts and external references, all 38 scripts remain importable via both legacy (`from analysis.<script> import ...`) and modular (`from analysis.<module>.<script> import ...`) paths. Symlinks at the `analysis/` root preserve command-line compatibility with existing pipelines.
