"""Held-out log-likelihood under each head's normalized distribution.

Evaluation is outside the algebraic training path, so logarithms are used
here to report negative log-likelihood and perplexity for every model.
"""

import jax
from jax import lax
import jax.numpy as jnp

F32 = jnp.float32


def logprob_from_logits(logits, targets, head: str, n: int = 16, c: float = 16.0,
                        eps_vocab: float = 100.0):
    """Per-token ``log p(target)``; logits (N, V), targets (N,)."""
    z = jnp.asarray(logits)
    z = z.astype(jnp.promote_types(z.dtype, F32))
    if head == "ce":
        a = z
    else:
        if head == "oace":
            z = z * lax.rsqrt(jnp.mean(z * z, axis=-1, keepdims=True) + eps_vocab)
            n, c = 8, 1.0
        a = n * jnp.arcsinh(z / c)
    return jnp.take_along_axis(jax.nn.log_softmax(a, axis=-1), targets[:, None], axis=-1)[:, 0]


def nll_from_target_prob(p_target):
    """Telemetry: per-token NLL from a target probability."""
    return -jnp.log(jnp.maximum(p_target, 1e-30))
