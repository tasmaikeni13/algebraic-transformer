"""Standard causal Transformer used for the training comparison.

The model uses learned-scale RMSNorm, SwiGLU, rotary positions, softmax
attention, and a fused linear cross-entropy head."""

from dataclasses import dataclass
from functools import partial
import math
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from src.kernels.fused_cross_entropy import standard_fused_cross_entropy
from src.kernels.pallas_flash_attention import standard_flash_attention


@dataclass(frozen=True)
class BaselineConfig:
    vocab_size: int = 50257
    d_model: int = 288
    num_layers: int = 6
    num_heads: int = 6
    d_ff: int = 768
    max_seq_len: int = 512
    eps: float = 1e-5
    tie_embeddings: bool = True
    dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.float32
    remat: bool = False
    attention_block_size: int = 128
    vocab_chunk_size: int = 4096


def _standard_rmsnorm(x: jax.Array, gamma: jax.Array, eps: float = 1e-5) -> jax.Array:
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(variance + eps)
    return (normed * gamma).astype(x.dtype)


def _build_standard_rope(dim: int, max_seq_len: int, base: float = 10000.0) -> Tuple[jax.Array, jax.Array]:
    half_dim = dim // 2
    theta = 1.0 / (base ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    pos = jnp.arange(max_seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, theta)
    cos_angles = jnp.cos(angles)
    sin_angles = jnp.sin(angles)
    return cos_angles, sin_angles


def _apply_standard_rope(
    q: jax.Array,
    k: jax.Array,
    cos_angles: jax.Array,
    sin_angles: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    seq_len = q.shape[2]
    cos_t = cos_angles[:seq_len][None, None, :, :]
    sin_t = sin_angles[:seq_len][None, None, :, :]

    def rotate_half(x):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return jnp.concatenate([-x2, x1], axis=-1)

    cos_full = jnp.concatenate([cos_t, cos_t], axis=-1).astype(q.dtype)
    sin_full = jnp.concatenate([sin_t, sin_t], axis=-1).astype(q.dtype)

    q_rot = q * cos_full + rotate_half(q) * sin_full
    k_rot = k * cos_full + rotate_half(k) * sin_full
    return q_rot, k_rot


def _standard_swiglu(x: jax.Array, w_g: jax.Array, w_u: jax.Array, w_d: jax.Array) -> jax.Array:
    gate = jnp.matmul(x, w_g)
    up = jnp.matmul(x, w_u)
    # Swish: z * sigmoid(z)
    swish_up = up * jax.nn.sigmoid(up)
    return jnp.matmul(gate * swish_up, w_d)


def _baseline_layer_forward(
    x,
    layer,
    cos_angles,
    sin_angles,
    *,
    eps,
    attention_block_size,
):
    B, T, d_model = x.shape
    dtype = x.dtype
    head_dim = cos_angles.shape[-1] * 2
    num_heads = d_model // head_dim
    h = _standard_rmsnorm(x, layer["norm1_gamma"], eps)
    q = jnp.matmul(h, layer["w_q"].astype(dtype)).reshape(B, T, num_heads, head_dim).swapaxes(1, 2)
    k = jnp.matmul(h, layer["w_k"].astype(dtype)).reshape(B, T, num_heads, head_dim).swapaxes(1, 2)
    v = jnp.matmul(h, layer["w_v"].astype(dtype)).reshape(B, T, num_heads, head_dim).swapaxes(1, 2)

    q_rot, k_rot = _apply_standard_rope(q, k, cos_angles, sin_angles)

    attn_out = standard_flash_attention(
        q_rot,
        k_rot,
        v,
        causal=True,
        block_q=attention_block_size,
        block_k=attention_block_size,
    ).swapaxes(1, 2).reshape(B, T, d_model)
    x = x + jnp.matmul(attn_out, layer["w_o"].astype(dtype))

    h2 = _standard_rmsnorm(x, layer["norm2_gamma"], eps)
    ffn_out = _standard_swiglu(
        h2,
        layer["w_g"].astype(dtype),
        layer["w_u"].astype(dtype),
        layer["w_d"].astype(dtype),
    )
    return x + ffn_out


class StandardTransformerLM:
    """Standard Causal Transformer baseline."""

    def __init__(self, config: Optional[BaselineConfig] = None):
        self.config = config or BaselineConfig()
        if self.config.num_heads <= 0 or self.config.d_model % self.config.num_heads != 0:
            raise ValueError("d_model must be divisible by a positive num_heads")
        if self.config.attention_block_size <= 0:
            raise ValueError("attention_block_size must be positive")
        if self.config.vocab_chunk_size <= 0:
            raise ValueError("vocab_chunk_size must be positive")
        self.head_dim = self.config.d_model // self.config.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")

    def init_params(self, key: jax.Array) -> Dict[str, Any]:
        cfg = self.config
        total_keys = 2 + cfg.num_layers * 7
        keys = jax.random.split(key, total_keys)
        k_idx = 0

        def normal(shape, std=0.02):
            nonlocal k_idx
            k = keys[k_idx]
            k_idx += 1
            return (jax.random.normal(k, shape) * std).astype(cfg.param_dtype)

        params: Dict[str, Any] = {
            "token_embed": normal((cfg.vocab_size, cfg.d_model)),
            "embed_norm_gamma": jnp.ones(cfg.d_model, dtype=cfg.param_dtype),
            "final_norm_gamma": jnp.ones(cfg.d_model, dtype=cfg.param_dtype),
            "layers": [],
        }

        if not cfg.tie_embeddings:
            params["output_head"] = normal((cfg.d_model, cfg.vocab_size))

        for _ in range(cfg.num_layers):
            layer = {
                "norm1_gamma": jnp.ones(cfg.d_model, dtype=cfg.param_dtype),
                "w_q": normal((cfg.d_model, cfg.d_model)),
                "w_k": normal((cfg.d_model, cfg.d_model)),
                "w_v": normal((cfg.d_model, cfg.d_model)),
                "w_o": normal((cfg.d_model, cfg.d_model)),
                "norm2_gamma": jnp.ones(cfg.d_model, dtype=cfg.param_dtype),
                "w_g": normal((cfg.d_model, cfg.d_ff)),
                "w_u": normal((cfg.d_model, cfg.d_ff)),
                "w_d": normal((cfg.d_ff, cfg.d_model)),
            }
            params["layers"].append(layer)

        return params

    def _hidden(
        self,
        params: Dict[str, Any],
        tokens: jax.Array,
        cos_angles: Optional[jax.Array] = None,
        sin_angles: Optional[jax.Array] = None,
    ) -> jax.Array:
        cfg = self.config
        B, T = tokens.shape
        if T > cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {cfg.max_seq_len}")

        if cos_angles is None or sin_angles is None:
            cos_angles, sin_angles = _build_standard_rope(self.head_dim, cfg.max_seq_len)

        x = params["token_embed"][tokens].astype(cfg.dtype)
        x = _standard_rmsnorm(x, params["embed_norm_gamma"], cfg.eps)

        layer_impl = partial(
            _baseline_layer_forward,
            eps=cfg.eps,
            attention_block_size=cfg.attention_block_size,
        )
        layer_fn = jax.checkpoint(layer_impl) if cfg.remat else layer_impl
        for layer in params["layers"]:
            x = layer_fn(x, layer, cos_angles, sin_angles)

        return _standard_rmsnorm(x, params["final_norm_gamma"], cfg.eps)

    def forward(
        self,
        params: Dict[str, Any],
        tokens: jax.Array,
        cos_angles: Optional[jax.Array] = None,
        sin_angles: Optional[jax.Array] = None,
    ) -> jax.Array:
        cfg = self.config
        x_final = self._hidden(params, tokens, cos_angles, sin_angles)

        if cfg.tie_embeddings:
            logits = jnp.matmul(x_final, params["token_embed"].T.astype(cfg.dtype))
        else:
            logits = jnp.matmul(x_final, params["output_head"].astype(cfg.dtype))

        return logits.astype(jnp.float32)

    def normalization_second_moments(
        self,
        params: Dict[str, Any],
        tokens: jax.Array,
        cos_angles: Optional[jax.Array] = None,
        sin_angles: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Measure normalized layer-input second moments for diagnostics."""
        cfg = self.config
        _, seq_len = tokens.shape
        if seq_len > cfg.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {cfg.max_seq_len}"
            )
        if cos_angles is None or sin_angles is None:
            cos_angles, sin_angles = _build_standard_rope(
                self.head_dim, cfg.max_seq_len
            )
        x = params["token_embed"][tokens].astype(cfg.dtype)
        x = _standard_rmsnorm(x, params["embed_norm_gamma"], cfg.eps)
        moments = []
        for layer in params["layers"]:
            normalized = _standard_rmsnorm(x, layer["norm1_gamma"], cfg.eps)
            moments.append(jnp.mean(normalized.astype(jnp.float32) ** 2))
            x = _baseline_layer_forward(
                x,
                layer,
                cos_angles,
                sin_angles,
                eps=cfg.eps,
                attention_block_size=cfg.attention_block_size,
            )
        final = _standard_rmsnorm(x, params["final_norm_gamma"], cfg.eps)
        moments.append(jnp.mean(final.astype(jnp.float32) ** 2))
        return jnp.stack(moments)

    def loss(
        self,
        params: Dict[str, Any],
        tokens: jax.Array,
        targets: jax.Array,
        cos_angles: Optional[jax.Array] = None,
        sin_angles: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, Dict[str, Any]]:
        cfg = self.config
        hidden = self._hidden(params, tokens, cos_angles, sin_angles)
        if cfg.tie_embeddings:
            output_weight = params["token_embed"].T
        else:
            output_weight = params["output_head"]
        loss = standard_fused_cross_entropy(
            hidden,
            output_weight,
            targets,
            chunk_size=cfg.vocab_chunk_size,
        )
        return loss, {"loss": loss}
