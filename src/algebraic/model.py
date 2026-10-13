"""Configurable pre-norm causal Transformer language model.

``LMConfig`` selects each component independently; ``src/algebraic/config.py``
names the compared architectures. Parameters are stacked over layers and the
blocks run under ``lax.scan``. This file performs no transcendental calls:
standard components come from ``standard.py`` and algebraic ones from
``algebraic.py``.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

import jax
from jax import lax
import jax.numpy as jnp

from algebraic import algebraic as A
from algebraic import standard as S

F32 = jnp.float32


@dataclass(frozen=True)
class LMConfig:
    vocab_size: int = 50257
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    d_ff: int = 1024
    seq_len: int = 1024
    # normalization: "rms" (learned gain) or "avn" (parameter free)
    norm: str = "rms"
    eps: float = 1e-5
    # feed-forward: "swiglu" or "alu_glu"
    ffn: str = "swiglu"
    alu_beta: float = 1.0
    # positions: "rope", "cayley_geo", "none"
    pos: str = "rope"
    rope_base: float = 10000.0
    rope_fraction: float = 1.0
    # attention: "softmax" or "asinh" (rho(s/c)^n)
    attn: str = "softmax"
    attn_n: int = 8
    attn_c: float = 1.0
    # sink: "none", "const" (fixed logit sink_init) or "learned" (per head)
    sink: str = "none"
    sink_init: float = 0.0
    qk_norm: bool = False
    # training head: "ce" or "alg_ce"
    head: str = "ce"
    head_n: int = 16
    head_c: float = 16.0
    # parameters and execution
    init_std: float = 0.02
    out_init_scale: float = 1.0  # multiplies the init std of w_o and w_d
    dtype: str = "bfloat16"
    remat: bool = True
    attn_impl: str = "dense"     # "dense" or "flash" (Pallas kernels)
    flash_block: int = 512
    head_impl: str = "dense"     # "dense" (XLA, fastest) or "fused" (Pallas, low memory)
    head_block_n: int = 128
    head_block_v: int = 512

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def compute_dtype(self):
        return jnp.dtype(self.dtype)

    @property
    def algebraic_path(self) -> bool:
        """True when no transcendental op is used in training."""
        return (self.ffn == "alu_glu" and self.pos.startswith("cayley")
                and self.attn == "asinh" and self.head == "alg_ce")


def count_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def init_params(cfg: LMConfig, key: jax.Array) -> Dict[str, Any]:
    if cfg.d_model % cfg.n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    keys = jax.random.split(key, 8)
    L, d, f, V, H, D = (cfg.n_layers, cfg.d_model, cfg.d_ff, cfg.vocab_size,
                        cfg.n_heads, cfg.head_dim)

    def normal(k, shape, scale=1.0):
        return jax.random.normal(k, shape, F32) * (cfg.init_std * scale)

    layers = {
        "w_q": normal(keys[1], (L, d, d)),
        "w_k": normal(keys[2], (L, d, d)),
        "w_v": normal(keys[3], (L, d, d)),
        "w_o": normal(keys[4], (L, d, d), cfg.out_init_scale),
        "w_g": normal(keys[5], (L, d, f)),
        "w_u": normal(keys[6], (L, d, f)),
        "w_d": normal(keys[7], (L, f, d), cfg.out_init_scale),
    }
    params: Dict[str, Any] = {"embed": normal(keys[0], (V, d))}
    if cfg.norm == "rms":
        params["embed_gain"] = jnp.ones((d,), F32)
        params["final_gain"] = jnp.ones((d,), F32)
        layers["ln1"] = jnp.ones((L, d), F32)
        layers["ln2"] = jnp.ones((L, d), F32)
    elif cfg.norm != "avn":
        raise ValueError(f"unknown norm {cfg.norm!r}")
    if cfg.qk_norm:
        layers["q_gain"] = jnp.ones((L, D), F32)
        layers["k_gain"] = jnp.ones((L, D), F32)
    if cfg.sink == "learned":
        layers["sink"] = jnp.full((L, H), cfg.sink_init, F32)
    params["layers"] = layers
    return params


def decay_mask(params) -> Any:
    """Decay weight matrices and the embedding; not gains or sink logits."""
    def rule(path, _):
        name = getattr(path[-1], "key", str(path[-1]))
        return name.startswith("w_") or name == "embed"
    return jax.tree_util.tree_map_with_path(rule, params)


def position_tables(cfg: LMConfig, seq_len: Optional[int] = None):
    T = seq_len or cfg.seq_len
    if cfg.pos == "rope":
        return S.rope_table(cfg.head_dim, T, cfg.rope_base)
    if cfg.pos == "cayley_geo":
        return A.cayley_table_geometric(cfg.head_dim, T, cfg.rope_base,
                                        fraction=cfg.rope_fraction)
    if cfg.pos == "none":
        half = cfg.head_dim // 2
        return jnp.ones((T, half), F32), jnp.zeros((T, half), F32)
    raise ValueError(f"unknown position kind {cfg.pos!r}")


def _norm(x, gain, cfg):
    return A.rms_norm(x, gain if cfg.norm == "rms" else None, cfg.eps)


def _attention(q, k, v, sink_logit, cfg: LMConfig):
    if cfg.attn_impl == "flash":
        from algebraic.kernels.flash import flash_attention
        return flash_attention(q, k, v, sink_logit, kind=cfg.attn, n=cfg.attn_n,
                               c=cfg.attn_c, block=cfg.flash_block,
                               range_guard=cfg.attn == "asinh")
    if cfg.attn == "softmax":
        return S.softmax_attention(q, k, v, sink_logit=sink_logit)
    if cfg.attn == "asinh":
        return A.asinh_attention(q, k, v, n=cfg.attn_n, c=cfg.attn_c, sink_logit=sink_logit)
    raise ValueError(f"unknown attention {cfg.attn!r}")


def _qk_norm(x, gain, cfg: LMConfig):
    # Measured hybrid training shape. Other paths retain native RMS; the
    # algebraic attention's different compilation pipeline is timed separately.
    if (cfg.attn_impl == "flash" and cfg.attn == "softmax" and x.dtype == jnp.bfloat16
            and x.shape[-2:] == (2048, 64) and x.size//64 % 1024 == 0
            and gain.dtype == F32 and cfg.eps > 0):
        from algebraic.kernels.qk_norm import qk_rms_norm
        return qk_rms_norm(x, gain, cfg.eps)
    return A.rms_norm(x, gain, cfg.eps)


def _block(cfg: LMConfig, cos, sin, x, lp):
    B, T, d = x.shape
    H, D = cfg.n_heads, cfg.head_dim
    dt = x.dtype
    h = _norm(x, lp.get("ln1"), cfg)
    q = (h @ lp["w_q"].astype(dt)).reshape(B, T, H, D).transpose(0, 2, 1, 3)
    k = (h @ lp["w_k"].astype(dt)).reshape(B, T, H, D).transpose(0, 2, 1, 3)
    v = (h @ lp["w_v"].astype(dt)).reshape(B, T, H, D).transpose(0, 2, 1, 3)
    if cfg.qk_norm:
        q = _qk_norm(q, lp["q_gain"], cfg)
        k = _qk_norm(k, lp["k_gain"], cfg)
    q = A.apply_rotary(q, cos, sin)
    k = A.apply_rotary(k, cos, sin)
    if cfg.sink == "learned":
        sink = lp["sink"].astype(F32)[None, :, None, None]
    elif cfg.sink == "const":
        sink = jnp.full((1, H, 1, 1), cfg.sink_init, F32)
    else:
        sink = None
    a = _attention(q, k, v, sink, cfg)
    a = a.transpose(0, 2, 1, 3).reshape(B, T, d)
    x = x + a @ lp["w_o"].astype(dt)
    h2 = _norm(x, lp.get("ln2"), cfg)
    gate = h2 @ lp["w_g"].astype(dt)
    up = h2 @ lp["w_u"].astype(dt)
    if cfg.ffn == "swiglu":
        act = S.silu(up)
    elif cfg.ffn == "alu_glu":
        act = A.alu(up, cfg.alu_beta)
    else:
        raise ValueError(f"unknown ffn {cfg.ffn!r}")
    return x + (gate * act) @ lp["w_d"].astype(dt)


def forward_hidden(params, tokens: jax.Array, cfg: LMConfig, tables) -> jax.Array:
    """Final normalized hidden states (B, T, d) in the compute dtype."""
    cos, sin = tables
    x = params["embed"][tokens].astype(cfg.compute_dtype)
    x = _norm(x, params.get("embed_gain"), cfg)

    def body(carry, lp):
        return _block(cfg, cos, sin, carry, lp), None

    if cfg.remat:
        body = jax.checkpoint(body, prevent_cse=False)
    x, _ = lax.scan(body, x, params["layers"])
    return _norm(x, params.get("final_gain"), cfg)


def head_loss(params, hidden: jax.Array, targets: jax.Array, cfg: LMConfig):
    """Training objective for one micro-batch.

    Returns ``(objective, telemetry)``: for the "ce" head the telemetry is the
    per-token NLL; for algebraic heads it is the per-token target probability
    (stop-gradient), converted to NLL by the trainer outside the model.
    """
    N = hidden.shape[0] * hidden.shape[1]
    h = hidden.reshape(N, -1)
    tgt = targets.reshape(N)
    if cfg.head_impl == "fused" and cfg.head in ("ce", "alg_ce"):
        from algebraic.kernels.fused_head import fused_head
        return fused_head(h, params["embed"], tgt, kind=cfg.head, n=cfg.head_n, c=cfg.head_c,
                          block_n=cfg.head_block_n, block_v=cfg.head_block_v)
    logits = jnp.matmul(h, params["embed"].astype(h.dtype).T, preferred_element_type=F32)
    if cfg.head == "ce":
        return S.cross_entropy(logits, tgt)
    if cfg.head == "alg_ce":
        obj, pt = A.algebraic_likelihood_lean(logits, tgt, cfg.head_n, cfg.head_c)
        return obj, lax.stop_gradient(pt)
    raise ValueError(f"unknown head {cfg.head!r}")


def telemetry_is_probability(cfg: LMConfig) -> bool:
    return cfg.head != "ce"


def eval_logprobs(params, tokens, targets, cfg: LMConfig, tables) -> jax.Array:
    """Per-token log p(target) under the model's normalized distribution."""
    from algebraic.evaluation import logprob_from_logits
    hidden = forward_hidden(params, tokens, cfg, tables)
    N = hidden.shape[0] * hidden.shape[1]
    logits = jnp.matmul(hidden.reshape(N, -1), params["embed"].astype(hidden.dtype).T,
                        preferred_element_type=F32)
    lp = logprob_from_logits(logits, targets.reshape(N), cfg.head, cfg.head_n,
                             cfg.head_c)
    return lp.reshape(targets.shape)
