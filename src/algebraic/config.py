"""Matched standard and pure algebraic Transformer presets."""

from dataclasses import replace
from typing import Dict

from algebraic.model import LMConfig

SIZES: Dict[str, Dict[str, int]] = {
    "proxy": dict(d_model=384, n_layers=8, n_heads=6, d_ff=1024),
    "125m": dict(d_model=768, n_layers=12, n_heads=12, d_ff=2048),
    # Tied 32k BPE embedding; both arms have approximately 332.45M parameters.
    "350m": dict(d_model=1024, n_layers=24, n_heads=16, d_ff=2688),
    "1b": dict(d_model=1536, n_layers=32, n_heads=24, d_ff=4224),
}


def standard(size: str = "proxy", **overrides) -> LMConfig:
    """Learned RMS gains, SwiGLU, RoPE, softmax attention, cross-entropy."""
    cfg = LMConfig(**SIZES[size], norm="rms", ffn="swiglu", pos="rope",
                   attn="softmax", sink="none", head="ce")
    return replace(cfg, **overrides)


def algebraic(size: str = "proxy", **overrides) -> LMConfig:
    """Pure algebraic model: arithmetic, comparisons, and reciprocal roots.

    Learned RMS gains; ALU-GLU with beta=1/2; geometric Cayley rotations;
    QK normalization; rho(s/8)^16 attention with a learned per-head sink;
    rho(z/2)^8 likelihood with the algebraic gradient of exact log loss.
    The primitive audit covers the model objective and its gradient.
    """
    cfg = LMConfig(**SIZES[size], norm="rms", ffn="alu_glu", alu_beta=0.5,
                   pos="cayley_geo", qk_norm=True, attn="asinh", attn_n=16, attn_c=8.0,
                   sink="learned", sink_init=0.0, head="alg_ce", head_n=8, head_c=2.0)
    return replace(cfg, **overrides)


PRESETS = {"standard": standard, "algebraic": algebraic}


def build(preset: str, size: str, **overrides) -> LMConfig:
    return PRESETS[preset](size, **overrides)
