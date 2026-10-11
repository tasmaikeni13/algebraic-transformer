"""Standard (transcendental) components of the reference Transformer."""

import math
from typing import Optional

import jax
import jax.numpy as jnp

F32 = jnp.float32


def silu(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)


def rope_table(head_dim: int, seq_len: int, base: float = 10000.0):
    """Trigonometric RoPE cos/sin tables of shape (T, head_dim // 2)."""
    half = head_dim // 2
    theta = 1.0 / (base ** (jnp.arange(half, dtype=F32) / half))
    angles = jnp.arange(seq_len, dtype=F32)[:, None] * theta[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def softmax_attention(q, k, v, *, sink_logit: Optional[jax.Array] = None):
    """Dense causal softmax attention on (B, H, T, D), optional sink logit."""
    d = q.shape[-1]
    t = q.shape[-2]
    s = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=F32)
    s = s * (1.0 / math.sqrt(d))
    mask = jnp.tril(jnp.ones((t, t), bool))[None, None]
    s = jnp.where(mask, s, -1e30)
    if sink_logit is not None:
        b = jnp.broadcast_to(sink_logit, s.shape[:-1] + (1,))
        p = jax.nn.softmax(jnp.concatenate([s, b], axis=-1), axis=-1)[..., :-1]
    else:
        p = jax.nn.softmax(s, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", p.astype(v.dtype), v, preferred_element_type=F32)
    return out.astype(q.dtype)


def cross_entropy(logits: jax.Array, targets: jax.Array):
    """Mean cross-entropy and stop-gradient per-token NLL."""
    z = logits.astype(F32)
    lse = jax.nn.logsumexp(z, axis=-1)
    zt = jnp.take_along_axis(z, targets[:, None], axis=-1)[:, 0]
    nll = lse - zt
    return jnp.mean(nll), jax.lax.stop_gradient(nll)
