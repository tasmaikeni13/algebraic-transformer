"""Independent NumPy float64 loss oracle and diagnostic baselines."""
from typing import Optional, Union
import numpy as np


def oace_loss_fp64(
    probabilities: np.ndarray,
    targets: Union[np.ndarray, int],
    gamma: float = 2.0,
    reduction: Optional[str] = "mean",
) -> np.ndarray:
    """Independent float64 NumPy implementation of the proper OACE score."""
    p = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets)
    gamma = float(gamma)

    if targets.ndim == p.ndim - 1 and targets.shape == p.shape[:-1]:
        # Integer targets
        y = (np.arange(p.shape[-1]) == np.expand_dims(targets, -1)).astype(np.float64)
        target_power_sum = np.ones(p.shape[:-1], dtype=np.float64)
    elif targets.shape == p.shape:
        y = targets.astype(np.float64)
        target_power_sum = np.sum(np.where(y > 0.0, y ** (7.0 / 8.0), 0.0), axis=-1)
    else:
        raise ValueError(f"Incompatible shapes: probs {p.shape}, targets {targets.shape}")

    # p^{-1/8} evaluated via three square roots: (p^{-1/2})^{-1/2})^{-1/2}
    r1 = 1.0 / np.sqrt(p)
    r2 = 1.0 / np.sqrt(r1)
    r3 = 1.0 / np.sqrt(r2)

    loss = gamma * (
        8.0 * np.sum(y * r3, axis=-1)
        + (8.0 / 7.0) * np.sum(p * r3, axis=-1)
        - (64.0 / 7.0) * target_power_sum
    )

    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    elif reduction in ("none", None):
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def oace_vjp_fp64(
    probabilities: np.ndarray,
    targets: Union[np.ndarray, int],
    g: np.ndarray,
    gamma: float = 2.0,
) -> np.ndarray:
    """Independent float64 NumPy analytical gradient of unreduced OACE loss."""
    p = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets)
    gamma = float(gamma)
    upstream = np.asarray(g, dtype=np.float64)

    if targets.ndim == p.ndim - 1 and targets.shape == p.shape[:-1]:
        y = (np.arange(p.shape[-1]) == np.expand_dims(targets, -1)).astype(np.float64)
    elif targets.shape == p.shape:
        y = targets.astype(np.float64)
    else:
        raise ValueError(f"Incompatible shapes: probs {p.shape}, targets {targets.shape}")

    r1 = 1.0 / np.sqrt(p)
    r2 = 1.0 / np.sqrt(r1)
    r3 = 1.0 / np.sqrt(r2)

    upstream = np.expand_dims(upstream, -1)
    # dL/dp_i = gamma * (p_i^-1/8 - y_i p_i^-9/8)
    dp = gamma * r3 * (1.0 - y * (r1 * r1)) * upstream
    return dp


def pearson_divergence_fp64(
    p: np.ndarray,
    q: np.ndarray,
    reduction: Optional[str] = None,
) -> np.ndarray:
    """Independent float64 NumPy implementation of Pearson chi^2 divergence."""
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    diff = p_arr - q_arr
    div = np.sum(diff * diff / q_arr, axis=-1)

    if reduction == "mean":
        return np.mean(div)
    elif reduction == "sum":
        return np.sum(div)
    elif reduction in ("none", None):
        return div
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def pearson_vjp_fp64(
    p: np.ndarray,
    q: np.ndarray,
    g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent float64 NumPy analytical gradient of unreduced Pearson divergence."""
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    upstream = np.expand_dims(np.asarray(g, dtype=np.float64), -1)

    dp = upstream * 2.0 * (p_arr - q_arr) / q_arr
    dq = upstream * (1.0 - (p_arr / q_arr) ** 2)
    return dp, dq


def cross_entropy_fp64(
    probabilities: np.ndarray,
    targets: Union[np.ndarray, int],
    reduction: Optional[str] = "mean",
    eps: float = 1e-15,
) -> np.ndarray:
    """Standard logarithmic cross-entropy baseline in float64."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0)
    targets = np.asarray(targets)

    if targets.ndim == p.ndim - 1 and targets.shape == p.shape[:-1]:
        pk = np.take_along_axis(p, np.expand_dims(targets, -1), axis=-1)[..., 0]
        loss = -np.log(pk)
    elif targets.shape == p.shape:
        loss = -np.sum(targets * np.log(p), axis=-1)
    else:
        raise ValueError(f"Incompatible shapes: probs {p.shape}, targets {targets.shape}")

    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    elif reduction in ("none", None):
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def kl_divergence_fp64(
    p: np.ndarray,
    q: np.ndarray,
    reduction: Optional[str] = None,
    eps: float = 1e-15,
) -> np.ndarray:
    """Standard Kullback-Leibler divergence baseline in float64."""
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.clip(np.asarray(q, dtype=np.float64), eps, 1.0)
    # Sum over last axis
    kl = np.sum(np.where(p_arr > 0, p_arr * np.log(np.where(p_arr > 0, p_arr / q_arr, 1.0)), 0.0), axis=-1)

    if reduction == "mean":
        return np.mean(kl)
    elif reduction == "sum":
        return np.sum(kl)
    elif reduction in ("none", None):
        return kl
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
