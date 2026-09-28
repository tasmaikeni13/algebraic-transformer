"""Independent NumPy float64 optimizer oracle and diagnostic baselines."""
from typing import Tuple, Union
import numpy as np


def adamw_fp64_reference(
    w: np.ndarray,
    g: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    t: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    use_rsqrt: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent float64 NumPy implementation of AdamW step.

    Returns:
        (w_next, update, m_next, v_next)
    """
    w_arr = np.asarray(w, dtype=np.float64)
    g_arr = np.asarray(g, dtype=np.float64)
    m_arr = np.asarray(m, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)

    # 1. Update moments
    m_next = beta1 * m_arr + (1.0 - beta1) * g_arr
    v_next = beta2 * v_arr + (1.0 - beta2) * (g_arr * g_arr)

    # 2. Debiasing factors
    debias1 = 1.0 - (beta1 ** t)
    debias2 = 1.0 - (beta2 ** t)

    m_hat = m_next / debias1
    v_hat = v_next / debias2

    # 3. Preconditioned update
    if use_rsqrt:
        u = m_hat / np.sqrt(v_hat + eps * eps)
    else:
        u = m_hat / (np.sqrt(v_hat) + eps)

    # 4. Decoupled weight decay
    if weight_decay > 0.0:
        u = u + weight_decay * w_arr

    update = -lr * u
    w_next = w_arr + update
    return w_next, update, m_next, v_next


def ards_schedule_fp64(
    step: Union[int, np.ndarray],
    lr_max: float,
    warmup_steps: int,
    decay_steps: int,
    alpha: float = 1.0,
) -> np.ndarray:
    """Independent float64 NumPy implementation of ARDS schedule."""
    s = np.asarray(step, dtype=np.float64)
    if warmup_steps > 0:
        warmup_factor = np.minimum(1.0, s / float(warmup_steps))
        decay_step = np.maximum(0.0, s - float(warmup_steps))
    else:
        warmup_factor = np.ones_like(s)
        decay_step = np.maximum(0.0, s)

    norm_decay = decay_step / float(max(1, decay_steps))
    decay_factor = 1.0 / np.sqrt(1.0 + float(alpha) * (norm_decay * norm_decay))
    return float(lr_max) * warmup_factor * decay_factor


def cosine_schedule_fp64(
    step: Union[int, np.ndarray],
    lr_max: float,
    lr_min: float,
    total_steps: int,
) -> np.ndarray:
    """Standard transcendental Cosine Annealing baseline schedule in float64."""
    s = np.asarray(step, dtype=np.float64)
    progress = np.clip(s / float(max(1, total_steps)), 0.0, 1.0)
    return float(lr_min) + 0.5 * (float(lr_max) - float(lr_min)) * (1.0 + np.cos(np.pi * progress))


def rosenbrock_fp64(x: np.ndarray) -> np.ndarray:
    """Multi-dimensional Rosenbrock function in float64."""
    x_arr = np.asarray(x, dtype=np.float64)
    return np.sum(
        100.0 * (x_arr[..., 1:] - x_arr[..., :-1] ** 2) ** 2 + (1.0 - x_arr[..., :-1]) ** 2,
        axis=-1,
    )


def rosenbrock_grad_fp64(x: np.ndarray) -> np.ndarray:
    """Analytical gradient of Rosenbrock function in float64."""
    x_arr = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x_arr)
    # df/dx_i = -400 * x_i * (x_{i+1} - x_i^2) - 2 * (1 - x_i) + 200 * (x_i - x_{i-1}^2)
    dim = x_arr.shape[-1]
    for i in range(dim):
        term1 = 0.0
        term2 = 0.0
        if i < dim - 1:
            term1 = -400.0 * x_arr[..., i] * (x_arr[..., i + 1] - x_arr[..., i] ** 2) - 2.0 * (1.0 - x_arr[..., i])
        if i > 0:
            term2 = 200.0 * (x_arr[..., i] - x_arr[..., i - 1] ** 2)
        g[..., i] = term1 + term2
    return g


def rastrigin_fp64(x: np.ndarray) -> np.ndarray:
    """Multi-dimensional Rastrigin benchmark function in float64."""
    x_arr = np.asarray(x, dtype=np.float64)
    d = x_arr.shape[-1]
    return 10.0 * d + np.sum(x_arr ** 2 - 10.0 * np.cos(2.0 * np.pi * x_arr), axis=-1)


def rastrigin_grad_fp64(x: np.ndarray) -> np.ndarray:
    """Analytical gradient of Rastrigin benchmark function in float64."""
    x_arr = np.asarray(x, dtype=np.float64)
    return 2.0 * x_arr + 20.0 * np.pi * np.sin(2.0 * np.pi * x_arr)


def quadratic_fp64(x: np.ndarray, diag_A: np.ndarray) -> np.ndarray:
    """Quadratic surface f(x) = 0.5 * x^T A x with diagonal A in float64."""
    x_arr = np.asarray(x, dtype=np.float64)
    a_arr = np.asarray(diag_A, dtype=np.float64)
    return 0.5 * np.sum(a_arr * (x_arr ** 2), axis=-1)


def quadratic_grad_fp64(x: np.ndarray, diag_A: np.ndarray) -> np.ndarray:
    """Gradient of quadratic surface: grad f(x) = A x in float64."""
    x_arr = np.asarray(x, dtype=np.float64)
    a_arr = np.asarray(diag_A, dtype=np.float64)
    return a_arr * x_arr
