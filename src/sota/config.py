"""Named model configurations.

* ``standard``: learned-gain RMSNorm, SwiGLU, RoPE, softmax attention,
  cross-entropy (the reference arm of both studies).
* ``algebraic_v1``: the first study's algebraic arm (AVN without gains,
  ALU-GLU, original Cayley grid, octic attention with sink mass 1, OACE).
* ``algebraic``: the second-generation algebraic arm (v2d), chosen by the
  Phase 1 ablations; ``paper/theory.md`` and ``analysis/README.md`` record why.
"""

from dataclasses import replace
from typing import Dict

from src.sota.algebraic import kernel_sink_logit
from src.sota.model import LMConfig

SIZES: Dict[str, Dict[str, int]] = {
    # ~33M parameters (14M non-embedding): Phase 1 proxy.
    "proxy": dict(d_model=384, n_layers=8, n_heads=6, d_ff=1024),
    # ~60M parameters (34M non-embedding): Phase 3.
    "60m": dict(d_model=512, n_layers=10, n_heads=8, d_ff=1536),
    # ~124M parameters (85M non-embedding): main run.
    "125m": dict(d_model=768, n_layers=12, n_heads=12, d_ff=2048),
}


def standard(size: str = "proxy", **overrides) -> LMConfig:
    cfg = LMConfig(**SIZES[size], norm="rms", ffn="swiglu", pos="rope",
                   attn="softmax", sink="none", head="ce")
    return replace(cfg, **overrides)


def algebraic_v1(size: str = "proxy", **overrides) -> LMConfig:
    cfg = LMConfig(**SIZES[size], norm="avn", ffn="alu_glu", alu_beta=1.0,
                   pos="cayley_orig", attn="asinh", attn_n=8, attn_c=1.0,
                   sink="const", sink_init=kernel_sink_logit(1.0, 8, 1.0),
                   head="oace", eps_vocab=100.0, gamma=2.0)
    return replace(cfg, **overrides)


def algebraic(size: str = "proxy", **overrides) -> LMConfig:
    """Second-generation algebraic arm (v2d, selected in Phase 1).

    Learned RMS gains; ALU-GLU with beta = 1/2; geometric Cayley rotations;
    QK-normalization; asinh-power attention rho(s/8)^16 with a learned
    per-head sink; algebraic maximum-likelihood head rho(z/2)^8 trained with
    the exact log-loss gradient. Every training-path operation is
    arithmetic, comparison, or rsqrt (see scripts/audit_algebraic.py).
    """
    cfg = LMConfig(**SIZES[size], norm="rms", ffn="alu_glu", alu_beta=0.5,
                   pos="cayley_geo", qk_norm=True, attn="asinh", attn_n=16, attn_c=8.0,
                   sink="learned", sink_init=0.0, head="alg_ce", head_n=8, head_c=2.0)
    return replace(cfg, **overrides)


PRESETS = {"standard": standard, "algebraic_v1": algebraic_v1, "algebraic": algebraic}


def build(preset: str, size: str, **overrides) -> LMConfig:
    return PRESETS[preset](size, **overrides)
