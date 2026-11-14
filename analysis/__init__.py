"""Modular analysis package for the Algebraic Transformer.

Organized into 6 submodules:
- analysis.figures: Publication plotting and figure generation
- analysis.kernels: TPU Pallas FlashAttention and Fused Head probes/benchmarks
- analysis.certificates: Formal real-arithmetic certificates and numerical bounds
- analysis.profiling: Component and step performance profiling scripts
- analysis.experimental: Exploratory attention link probes
- analysis.scaling: Scaling jobs and power-law summaries
"""

import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

__all__ = [
    "figures",
    "kernels",
    "certificates",
    "profiling",
    "experimental",
    "scaling",
]

# Legacy script name to modular subpackage mapping
_LEGACY_MAP = {
    # figures
    "plot_figures": "analysis.figures.plot_figures",
    "summarize_runs": "analysis.figures.summarize_runs",
    # kernels
    "attention_performance": "analysis.kernels.attention_performance",
    "attention_resident_probe": "analysis.kernels.attention_resident_probe",
    "attention_head_resident_probe": "analysis.kernels.attention_head_resident_probe",
    "attention_streamed_probe": "analysis.kernels.attention_streamed_probe",
    "attention_backward_schedule_probe": "analysis.kernels.attention_backward_schedule_probe",
    "attention_checkpoint_probe": "analysis.kernels.attention_checkpoint_probe",
    "attention_segmented_probe": "analysis.kernels.attention_segmented_probe",
    "attention_statistics": "analysis.kernels.attention_statistics",
    "attention_head_statistics": "analysis.kernels.attention_head_statistics",
    "attention_followup_statistics": "analysis.kernels.attention_followup_statistics",
    "head_gradient_cache_probe": "analysis.kernels.head_gradient_cache_probe",
    "qk_rms_kernel": "analysis.kernels.qk_rms_kernel",
    "qk_normalization_probe": "analysis.kernels.qk_normalization_probe",
    "qk_model_validation": "analysis.kernels.qk_model_validation",
    # certificates
    "activation_analysis": "analysis.certificates.activation_analysis",
    "attention_numerics": "analysis.certificates.attention_numerics",
    "attention_polynomial": "analysis.certificates.attention_polynomial",
    "attention_interpolation_certificate": "analysis.certificates.attention_interpolation_certificate",
    "attention_newton_certificate": "analysis.certificates.attention_newton_certificate",
    "attention_anchored_certificate": "analysis.certificates.attention_anchored_certificate",
    "attention_rational_radical": "analysis.certificates.attention_rational_radical",
    "attention_rational_radical_probe": "analysis.certificates.attention_rational_radical_probe",
    # profiling
    "model_component_performance": "analysis.profiling.model_component_performance",
    "model_step_performance": "analysis.profiling.model_step_performance",
    "model_profile": "analysis.profiling.model_profile",
    # experimental
    "attention_link_analysis": "analysis.experimental.attention_link_analysis",
    "attention_link_probe": "analysis.experimental.attention_link_probe",
    "attention_link_statistics": "analysis.experimental.attention_link_statistics",
    "attention_recall_probe": "analysis.experimental.attention_recall_probe",
    "attention_radical_probe": "analysis.experimental.attention_radical_probe",
    "attention_interpolation_probe": "analysis.experimental.attention_interpolation_probe",
    "attention_projective_probe": "analysis.experimental.attention_projective_probe",
    "attention_affine_guard_probe": "analysis.experimental.attention_affine_guard_probe",
    # scaling
    "scaling_jobs": "analysis.scaling.scaling_jobs",
    "scaling_summary": "analysis.scaling.scaling_summary",
    "verify_head_eval": "analysis.scaling.verify_head_eval",
}


def __getattr__(name: str):
    if name in _LEGACY_MAP:
        mod = importlib.import_module(_LEGACY_MAP[name])
        sys.modules[f"analysis.{name}"] = mod
        return mod
    raise AttributeError(f"module 'analysis' has no attribute '{name}'")
