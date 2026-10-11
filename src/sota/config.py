"""Named model configurations.

* ``standard``: learned-gain RMSNorm, SwiGLU, RoPE, softmax attention,
  cross-entropy (the reference arm of both studies).
* ``algebraic_v1``: the first study's algebraic arm (AVN without gains,
  ALU-GLU, original Cayley grid, octic attention with sink mass 1, OACE).
* ``algebraic``: the current second-generation algebraic arm. It is updated
  as Phase 1/3 iterations conclude; ``paper/theory.md`` records why.
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
    """Second-generation algebraic arm (provisional until Phase 1 concludes)."""
    cfg = LMConfig(**SIZES[size], norm="rms", ffn="alu_glu", alu_beta=1.0,
                   pos="cayley_geo", attn="asinh", attn_n=8, attn_c=1.0,
                   sink="learned", sink_init=0.0, head="alg_ce", head_n=16,
                   head_c=16.0)
    return replace(cfg, **overrides)


PRESETS = {"standard": standard, "algebraic_v1": algebraic_v1, "algebraic": algebraic}


def build(preset: str, size: str, **overrides) -> LMConfig:
    return PRESETS[preset](size, **overrides)
