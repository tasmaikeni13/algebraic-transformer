"""Independent NumPy fp64 attention oracle and diagnostic baselines."""
import numpy as np


def kernel(x):
    x = np.asarray(x, dtype=np.float64)
    s = np.sqrt(1 + x*x)
    # Index the branches to avoid evaluating an unstable unused branch.
    rho = np.empty_like(x)
    positive = x >= 0
    rho[positive] = s[positive] + x[positive]
    rho[~positive] = 1 / (s[~positive] - x[~positive])
    return rho**8


def normalized_attention(x, sink=0.5):
    k = kernel(x)
    return k / (np.sum(k, axis=-1, keepdims=True) + sink)


def attention(x, sink=0.5, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    z = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps)
    return normalized_attention(z, sink)


def attention_vjp(x, g, sink=0.5, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    tau = 1 / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps)
    z = tau*x
    k = kernel(z)
    denom = np.sum(k, axis=-1, keepdims=True) + sink
    dk = 8*k / np.sqrt(1+z*z)
    # Quotient rule directly, independent of production's cached-p expression.
    dz = dk * (g / denom - np.sum(g*k, axis=-1, keepdims=True) / denom**2)
    return tau*dz - x*tau**3*np.mean(x*dz, axis=-1, keepdims=True)


def softmax(x):
    e = np.exp(x-np.max(x, axis=-1, keepdims=True))
    return e/e.sum(axis=-1, keepdims=True)


def reference_afa(q, k, v, sink=0.5, causal=False):
    """Independent pure NumPy float64 ground truth for Algebraic FlashAttention."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    head_dim = q.shape[-1]
    scale = 1.0 / np.sqrt(head_dim)
    s = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    p = kernel(s)
    if causal:
        seq_len = q.shape[-2]
        mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
        p = np.where(mask, p, 0.0)
    o = np.matmul(p, v)
    denom = np.sum(p, axis=-1, keepdims=True) + sink
    return o / denom

