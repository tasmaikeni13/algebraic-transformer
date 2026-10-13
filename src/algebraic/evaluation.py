"""Held-out log-likelihood under each head's normalized distribution.

Evaluation is outside the algebraic training path, so logarithms are used
here to report negative log-likelihood and perplexity for every model.
"""

import jax
import jax.numpy as jnp

F32 = jnp.float32


def logprob_from_logits(logits, targets, head: str, n: int = 16, c: float = 16.0):
    """Per-token ``log p(target)``; logits (N, V), targets (N,)."""
    z = jnp.asarray(logits)
    z = z.astype(jnp.promote_types(z.dtype, F32))
    if head == "ce":
        a = z
    elif head == "alg_ce":
        a = n * jnp.arcsinh(z / c)
    else:
        raise ValueError(f"unknown head {head!r}")
    return jnp.take_along_axis(jax.nn.log_softmax(a, axis=-1), targets[:, None], axis=-1)[:, 0]


def nll_from_target_prob(p_target):
    """Telemetry: per-token NLL from a target probability."""
    return -jnp.log(jnp.maximum(p_target, 1e-30))
