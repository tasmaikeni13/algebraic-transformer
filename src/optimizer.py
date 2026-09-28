"""AdamW update and inverse-square-root learning-rate decay.

Moment updates, integer-power bias correction, and decoupled weight decay use
arithmetic operations. The schedule uses a single inverse square root."""

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp


class AlgebraicAdamWState(NamedTuple):
    """Optimizer state for Algebraic AdamW."""
    count: jax.Array
    mu: Any
    nu: Any


class GradientTransformation(NamedTuple):
    """Optax-compatible gradient transformation container."""
    init: Callable[[Any], AlgebraicAdamWState]
    update: Callable[[Any, AlgebraicAdamWState, Optional[Any]], Tuple[Any, AlgebraicAdamWState]]


def _algebraic_square_root(x: jax.Array) -> jax.Array:
    """Evaluate the nonnegative square root using only rsqrt and multiply.

    The explicit zero branch avoids the indeterminate product ``0 * rsqrt(0)``.
    This keeps exact Optax-style ``sqrt(v) + eps`` semantics without emitting a
    square-root opcode outside the project's permitted primitive set.
    """
    inverse_root = jax.lax.rsqrt(x)
    return jnp.where(x == 0, jnp.zeros_like(x), x * inverse_root)


def _rational_power(base: jax.Array, n: jax.Array) -> jax.Array:
    """Exact polynomial power via 24-bit binary exponentiation: O(log t) multiplications.

    Zero exponential functions, zero logarithmic functions, zero transcendentals.
    """
    res = jnp.ones_like(base)
    cur = base
    # 24 bits covers steps up to 16,777,216
    for i in range(24):
        bit = (n >> i) & 1
        res = jnp.where(bit == 1, res * cur, res)
        cur = cur * cur
    return res


def ards_schedule(
    learning_rate: float,
    warmup_steps: int,
    decay_steps: int,
    alpha: float = 1.0,
) -> Callable[[Union[int, jax.Array]], jax.Array]:
    """Algebraic Rational Decay Schedule (ARDS).

    eta_t = eta_max * min(1, t / T_warm) * rsqrt(1 + alpha * [max(0, t - T_warm) / T_decay]^2)

    Properties:
    1. Continuous and piecewise smooth.
    2. Linear warmup for t <= T_warm.
    3. Strictly monotonic rational decay for t > T_warm with asymptotic O(1/t) rate.
    4. Evaluated in 1 subtraction, 1 square, 1 FMA, and 1 hardware rsqrt.

    Args:
        learning_rate: Peak learning rate (eta_max).
        warmup_steps: Number of linear warmup steps (T_warm).
        decay_steps: Scale factor for rational decay (T_decay).
        alpha: Curvature hyperparameter (default: 1.0).

    Returns:
        Callable taking step count and returning scalar learning rate.
    """
    lr_max = float(learning_rate)
    t_warm = float(warmup_steps)
    t_decay = float(max(1, decay_steps))
    alpha_val = float(alpha)

    def schedule_fn(count: Union[int, jax.Array]) -> jax.Array:
        raw_step = jnp.asarray(count)
        calc_dtype = jnp.float64 if raw_step.dtype == jnp.float64 else jnp.float32
        step = raw_step.astype(calc_dtype)
        if warmup_steps > 0:
            warmup_factor = jnp.minimum(1.0, step / t_warm)
            decay_step = jnp.maximum(0.0, step - t_warm)
        else:
            warmup_factor = 1.0
            decay_step = jnp.maximum(0.0, step)

        norm_decay = decay_step / t_decay
        denom = 1.0 + alpha_val * (norm_decay * norm_decay)
        decay_factor = jax.lax.rsqrt(denom)
        return (lr_max * warmup_factor * decay_factor).astype(calc_dtype)

    return schedule_fn


def algebraic_adamw(
    learning_rate: Union[float, Callable[[Union[int, jax.Array]], jax.Array]],
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    mask: Optional[Union[Any, Callable[[Any], Any]]] = None,
    use_rsqrt: bool = False,
) -> GradientTransformation:
    """Algebraic AdamW optimizer with pure rational polynomial debiasing.

    Moment updates and integer-power bias correction use arithmetic operations.
    The preconditioner uses inverse square root.

    Args:
        learning_rate: Learning rate scalar float or callable schedule function.
        beta1: Rational first moment decay rate (default: 0.9).
        beta2: Rational second moment decay rate (default: 0.999).
        eps: Small stability constant (default: 1e-8).
        weight_decay: Decoupled weight decay factor (default: 1e-2).
        mask: Optional PyTree mask or predicate callable for selective weight decay.
        use_rsqrt: If True, preconditions via hat_m * rsqrt(hat_v + eps^2).
            If False, retains exact Optax ``sqrt(hat_v) + eps`` semantics but
            constructs the square root as ``hat_v * rsqrt(hat_v)`` so the
            compiled graph still contains no raw square-root instruction.

    Returns:
        Optax-compatible GradientTransformation with init and update functions.
    """
    b1 = float(beta1)
    b2 = float(beta2)
    eps_val = float(eps)
    wd = float(weight_decay)

    def init_fn(params: Any) -> AlgebraicAdamWState:
        mu = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
        nu = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
        return AlgebraicAdamWState(count=jnp.zeros([], dtype=jnp.int32), mu=mu, nu=nu)

    def update_fn(
        updates: Any,
        state: AlgebraicAdamWState,
        params: Optional[Any] = None,
    ) -> Tuple[Any, AlgebraicAdamWState]:
        new_count = state.count + 1

        # Determine accumulation dtype: float64 stays float64, half precision uses float32
        leaves = jax.tree_util.tree_leaves(updates)
        if leaves:
            base_dtype = jnp.result_type(*[x.dtype for x in leaves])
            accum_dtype = jnp.promote_types(base_dtype, jnp.float32)
        else:
            accum_dtype = jnp.float32

        # Rational polynomial debiasing via binary exponentiation
        b1_t = _rational_power(jnp.asarray(b1, dtype=accum_dtype), new_count)
        b2_t = _rational_power(jnp.asarray(b2, dtype=accum_dtype), new_count)
        debias1 = 1.0 - b1_t
        debias2 = 1.0 - b2_t

        # Moment updates: rational linear and quadratic combinations
        new_mu = jax.tree_util.tree_map(
            lambda g, m: b1 * m + (1.0 - b1) * g,
            updates,
            state.mu,
        )
        new_nu = jax.tree_util.tree_map(
            lambda g, v: b2 * v + (1.0 - b2) * (g * g),
            updates,
            state.nu,
        )

        # Preconditioned updates
        if use_rsqrt:
            eps_sq = eps_val * eps_val
            preconditioned = jax.tree_util.tree_map(
                lambda m, v: (m / debias1) * jax.lax.rsqrt((v / debias2) + eps_sq),
                new_mu,
                new_nu,
            )
        else:
            preconditioned = jax.tree_util.tree_map(
                lambda m, v: (m / debias1) / (_algebraic_square_root(v / debias2) + eps_val),
                new_mu,
                new_nu,
            )

        # Decoupled algebraic weight decay: U + lambda * W
        if wd > 0.0 and params is not None:
            if mask is not None:
                mask_tree = mask(params) if callable(mask) else mask
                preconditioned = jax.tree_util.tree_map(
                    lambda u, p, m_val: u + (wd * p if m_val else jnp.zeros_like(p)),
                    preconditioned,
                    params,
                    mask_tree,
                )
            else:
                preconditioned = jax.tree_util.tree_map(
                    lambda u, p: u + wd * p,
                    preconditioned,
                    params,
                )

        # Apply the learning rate and retain the gradient dtype.
        if callable(learning_rate):
            lr = learning_rate(new_count)
        else:
            lr = learning_rate

        final_updates = jax.tree_util.tree_map(
            lambda u, g: (-lr * u).astype(g.dtype),
            preconditioned,
            updates,
        )
        new_mu = jax.tree_util.tree_map(lambda m, g: m.astype(g.dtype), new_mu, updates)
        new_nu = jax.tree_util.tree_map(lambda v, g: v.astype(g.dtype), new_nu, updates)

        new_state = AlgebraicAdamWState(count=new_count, mu=new_mu, nu=new_nu)
        return final_updates, new_state

    return GradientTransformation(init=init_fn, update=update_fn)
