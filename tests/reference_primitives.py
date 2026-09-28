"""Independent float64 CPU oracle; never imported by production primitives."""

import numpy as np


def alu_reference(x):
    x = np.asarray(x, dtype=np.float64)
    s = np.hypot(x, 1.)
    # Independent conjugate form for the cancellation-prone negative branch.
    return np.where(x < 0, x / (2 * s * (s - np.minimum(x, 0))), x / 2 + x * x / (2 * s))


def alu_derivative(x):
    x = np.asarray(x, dtype=np.float64)
    s = np.sqrt(1 + x * x)
    return (1 + x / s) / 2 + x / (2 * s * s * s)


def alu_second_derivative(x):
    x = np.asarray(x, dtype=np.float64)
    s = np.sqrt(1 + x * x)
    return (2 - x * x) / (2 * s * s * s * s * s)


def avn_reference(x, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def avn_vjp_reference(x, g, eps=1e-5):
    x, g = np.asarray(x, dtype=np.float64), np.asarray(g, dtype=np.float64)
    v = np.mean(x * x, axis=-1, keepdims=True) + eps
    return g / np.sqrt(v) - x * np.mean(g * x, axis=-1, keepdims=True) / (v * np.sqrt(v))
