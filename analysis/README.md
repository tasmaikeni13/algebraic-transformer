# Reproducible analysis

The public comparison is in [results/main/comparison.md](../results/main/comparison.md). This directory contains numerical studies, exact attention certificates, and performance probes for the implemented components.

| Script | Purpose |
|---|---|
| `plot_figures.py` | Rebuild architecture, held-out-quality, and benchmark figures from the published 125M results |
| `activation_analysis.py` | Compare ALU and SiLU activation values, derivatives, and moments |
| `attention_numerics.py` | Float32 attention normalization and gradient stress cases |
| `attention_statistics.py`, `attention_followup_statistics.py` | Paired kernel timing statistics |
| `attention_performance.py`, `attention_resident_probe.py` | Tiled and resident attention performance |
| `attention_head_resident_probe.py`, `attention_streamed_probe.py`, `attention_projective_probe.py` | Whole-head scheduling, score-tile sweeps, and normalization rearrangements |
| `attention_head_statistics.py`, `attention_checkpoint_probe.py` | Paired confirmation statistics and bounded synthetic optimizer smokes |
| `attention_link_analysis.py`, `attention_link_probe.py`, `attention_link_statistics.py` | Candidate link accuracy and latency |
| `attention_interpolation_certificate.py`, `attention_newton_certificate.py` | Exact rational error certificates |
| `attention_interpolation_probe.py`, `attention_polynomial.py`, `attention_radical_probe.py`, `attention_affine_guard_probe.py` | Experimental approximation probes |
| `model_component_performance.py`, `model_step_performance.py`, `model_profile.py` | Component and optimizer-step timings |
| `qk_model_validation.py`, `qk_normalization_probe.py`, `qk_rms_kernel.py` | QK normalization checks and reduction experiments |
| `head_gradient_cache_probe.py`, `verify_head_eval.py` | Head gradient and evaluation checks |
| `attention_recall_probe.py` | Toy attention retrieval study |
| `scaling_jobs.py`, `scaling_summary.py` | 60M / 1B-token scaling comparison |
| `summarize_runs.py` | Collect completed trainer outputs |

```bash
python analysis/plot_figures.py
JAX_PLATFORMS=cpu python analysis/activation_analysis.py
python scripts/audit_algebraic.py --set attn_impl=flash --set head_impl=fused
```

TPU probes run through `bash scripts/tpu_python.sh 0 <script> ...`. Experimental approximations are opt-in and are not selected by the language-model presets. Performance records describe their measured shapes, dtypes, and code fingerprints; use the current smoke reports under `results/verification/` for execution checks of the published training configurations. Toy retrieval studies do not replace post-training recall evaluation.
