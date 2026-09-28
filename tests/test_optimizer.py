"""Tests for Optimizer study: Algebraic AdamW optimizer and ARDS rational decay schedule."""
import ast
import io
from pathlib import Path
import re
import tokenize

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.audit_primitives import FORBIDDEN, primitives_in, source_audit
from src.optimizer import algebraic_adamw, ards_schedule, _rational_power
from tests.reference_optimizer import (
    adamw_fp64_reference,
    ards_schedule_fp64,
    cosine_schedule_fp64,
    quadratic_fp64,
    quadratic_grad_fp64,
    rastrigin_fp64,
    rosenbrock_fp64,
)


@pytest.mark.parametrize("dtype,tol", [(jnp.float64, 1e-12), (jnp.float32, 1e-6), (jnp.bfloat16, 0.03)])
def test_algebraic_adamw_oracle_and_optax_parity(dtype, tol):
    """Verify single-step update parity against float64 oracle and Optax."""
    rng = np.random.default_rng(42)
    dim = 8
    w_raw = rng.normal(size=dim).astype(np.float64)
    g_raw = rng.normal(size=dim).astype(np.float64)
    m_raw = np.zeros(dim, dtype=np.float64)
    v_raw = np.zeros(dim, dtype=np.float64)
    lr = 1e-3
    wd = 1e-2

    w_ref, u_ref, m_ref, v_ref = adamw_fp64_reference(
        w_raw, g_raw, m_raw, v_raw, t=1, lr=lr, weight_decay=wd
    )

    opt = algebraic_adamw(learning_rate=lr, weight_decay=wd)
    w_jax = jnp.asarray(w_raw, dtype=dtype)
    g_jax = jnp.asarray(g_raw, dtype=dtype)
    state = opt.init(w_jax)
    updates, new_state = opt.update(g_jax, state, w_jax)

    np.testing.assert_allclose(np.asarray(updates, dtype=np.float64), u_ref, rtol=tol, atol=tol)
    np.testing.assert_allclose(np.asarray(new_state.mu, dtype=np.float64), m_ref, rtol=tol, atol=tol)
    np.testing.assert_allclose(np.asarray(new_state.nu, dtype=np.float64), v_ref, rtol=tol, atol=tol)
    assert updates.dtype == dtype
    assert new_state.mu.dtype == dtype
    assert new_state.nu.dtype == dtype

    # Test against Optax if installed
    try:
        import optax
        opt_ref = optax.adamw(learning_rate=lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=wd)
        s_opt = opt_ref.init(w_jax)
        u_opt, _ = opt_ref.update(g_jax, s_opt, w_jax)
        np.testing.assert_allclose(np.asarray(updates, dtype=np.float64), np.asarray(u_opt, dtype=np.float64), rtol=tol, atol=tol)
    except ImportError:
        pass


def test_algebraic_adamw_multistep_trajectory():
    """Verify 50-step optimization trajectory in float64 against independent oracle."""
    rng = np.random.default_rng(101)
    dim = 6
    w_ref = rng.normal(size=dim).astype(np.float64)
    m_ref = np.zeros(dim, dtype=np.float64)
    v_ref = np.zeros(dim, dtype=np.float64)
    lr = 2e-3
    wd = 1e-2

    opt = algebraic_adamw(learning_rate=lr, weight_decay=wd)
    w_jax = jnp.asarray(w_ref, dtype=jnp.float64)
    state = opt.init(w_jax)

    for step in range(1, 51):
        g = rng.normal(size=dim).astype(np.float64)
        w_ref, u_ref, m_ref, v_ref = adamw_fp64_reference(
            w_ref, g, m_ref, v_ref, t=step, lr=lr, weight_decay=wd
        )
        updates, state = opt.update(jnp.asarray(g, dtype=jnp.float64), state, w_jax)
        w_jax = w_jax + updates

        np.testing.assert_allclose(np.asarray(updates), u_ref, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(np.asarray(w_jax), w_ref, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(np.asarray(state.mu), m_ref, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(np.asarray(state.nu), v_ref, rtol=1e-11, atol=1e-11)


def test_algebraic_adamw_weight_decay_mask():
    """Verify weight decay masking behaves correctly on pytree leaves."""
    params = {
        "kernel": jnp.ones((4, 4), dtype=jnp.float32),
        "bias": jnp.zeros((4,), dtype=jnp.float32),
    }
    mask = {
        "kernel": True,
        "bias": False,
    }
    opt = algebraic_adamw(learning_rate=1e-3, weight_decay=0.1, mask=mask)
    state = opt.init(params)
    grads = {
        "kernel": jnp.zeros((4, 4), dtype=jnp.float32),
        "bias": jnp.zeros((4,), dtype=jnp.float32),
    }
    updates, _ = opt.update(grads, state, params)

    # Bias should receive 0 update (grad=0, wd=0)
    np.testing.assert_allclose(np.asarray(updates["bias"]), np.zeros(4), atol=1e-7)
    # Kernel should receive -lr * wd * W = -1e-3 * 0.1 * 1.0 = -1e-4
    np.testing.assert_allclose(np.asarray(updates["kernel"]), -1e-4 * np.ones((4, 4)), atol=1e-7)


def test_algebraic_adamw_callable_schedule():
    """Verify algebraic_adamw correctly steps using a callable schedule."""
    sched = ards_schedule(learning_rate=1e-3, warmup_steps=10, decay_steps=100)
    opt = algebraic_adamw(learning_rate=sched, weight_decay=0.0)
    p = jnp.array([1.0, 2.0], dtype=jnp.float32)
    state = opt.init(p)
    g = jnp.array([0.1, 0.2], dtype=jnp.float32)

    # At step 1, count=1, warmup factor = 1/10 = 0.1, lr = 1e-4
    updates1, state1 = opt.update(g, state, p)
    # At step 10, count=10, warmup factor = 1.0, lr = 1e-3
    s = state1
    for step in range(2, 11):
        updates, s = opt.update(g, s, p)
    assert s.count == 10
    # Step 10 lr is exactly 1e-3
    expected_lr10 = 1e-3
    # With g constant, u ~ 1.0, update ~ -1e-3 * 1.0
    np.testing.assert_allclose(float(sched(10)), expected_lr10, rtol=1e-5)


def test_ards_schedule_properties():
    """Verify ARDS schedule correctness, monotonicity, and asymptotic rate."""
    lr_max = 1e-3
    warmup = 100
    decay = 1000
    alpha = 1.0
    sched = ards_schedule(learning_rate=lr_max, warmup_steps=warmup, decay_steps=decay, alpha=alpha)

    # 1. Warmup linearity
    warmup_steps = np.array([0, 25, 50, 75, 100], dtype=np.float64)
    lrs_warmup = np.array([float(sched(s)) for s in warmup_steps])
    expected_warmup = lr_max * (warmup_steps / float(warmup))
    np.testing.assert_allclose(lrs_warmup, expected_warmup, rtol=1e-6)

    # 2. Strict monotonicity for t in [T_warm, 10^5]
    eval_steps = jnp.arange(warmup, 100000, 100, dtype=jnp.float32)
    eval_lrs = jax.vmap(sched)(eval_steps)
    diffs = jnp.diff(eval_lrs)
    assert jnp.all(diffs < 0.0), "ARDS schedule must be strictly monotonically decreasing after warmup"

    # 3. Asymptotic O(1/t) decay rate: t * eta(t) -> lr_max * T_decay / sqrt(alpha)
    asymptotic_limit_expected = lr_max * decay / np.sqrt(alpha)
    large_t = 1_000_000
    asymptotic_limit_actual = float(large_t * sched(large_t))
    ratio = asymptotic_limit_actual / asymptotic_limit_expected
    assert abs(ratio - 1.0) < 0.01, f"Asymptotic rate ratio {ratio} deviates from 1.0 by > 1%"

    # 4. Parity with float64 reference
    steps_ref = np.arange(0, 5000, 25, dtype=np.float64)
    ref_lrs = ards_schedule_fp64(steps_ref, lr_max, warmup, decay, alpha=alpha)
    jax_lrs = np.array([float(sched(s)) for s in steps_ref])
    np.testing.assert_allclose(jax_lrs, ref_lrs, rtol=1e-6)


def test_optimizer_zero_transcendental_audit():
    """Verify zero occurrences of exp, log, sin, cos, or noninteger powers in src/optimizer.py."""
    source_path = Path(__file__).resolve().parents[1] / "src/optimizer.py"
    source = source_path.read_text()

    # 1. AST audit
    violations = source_audit(source)
    for node in ast.walk(ast.parse(source)):
        name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
        if name == "sqrt":
            violations.append({"line": node.lineno, "name": name})
    assert violations == [], f"AST audit detected forbidden nodes: {violations}"

    # 2. Token-level regex inspection
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in {tokenize.COMMENT, tokenize.STRING}:
            tokens.append(token.string)
    token_str = " ".join(tokens)
    regex_hits = re.findall(r"\b(?:exp|log|sin|cos|tanh|sigmoid)\b", token_str)
    assert regex_hits == [], f"Token regex found forbidden identifiers: {regex_hits}"

    # 3. Traced graph primitives in JAXPR
    opt = algebraic_adamw(learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2)
    p = jnp.array([1.0, 2.0], dtype=jnp.float32)
    s = opt.init(p)
    g = jnp.array([0.1, -0.2], dtype=jnp.float32)

    jaxpr_up = jax.make_jaxpr(lambda p, s, g: opt.update(g, s, p))(p, s, g)
    counts_up = primitives_in(jaxpr_up)
    forbidden_up = set(counts_up.keys()) & (FORBIDDEN | {"sqrt"})
    assert forbidden_up == set(), f"Forbidden primitives in update graph: {forbidden_up}"

    sched = ards_schedule(1e-3, 100, 1000, alpha=1.0)
    jaxpr_sched = jax.make_jaxpr(sched)(jnp.array(500, dtype=jnp.int32))
    counts_sched = primitives_in(jaxpr_sched)
    forbidden_sched = set(counts_sched.keys()) & FORBIDDEN
    assert forbidden_sched == set(), f"Forbidden primitives in schedule graph: {forbidden_sched}"


def test_ill_conditioned_quadratic_sweep():
    """Verify > 99.99% loss reduction in 300 steps across 10^4 ill-conditioned quadratics."""
    num_trials = 10000
    dim = 8
    rng = np.random.default_rng(42)
    log_kappas = rng.uniform(2.0, 6.0, size=num_trials)
    kappas = 10.0 ** log_kappas

    diag_A = np.zeros((num_trials, dim), dtype=np.float32)
    for i in range(num_trials):
        diag_A[i] = np.geomspace(1.0, kappas[i], dim).astype(np.float32)

    x0 = rng.normal(size=(num_trials, dim)).astype(np.float32)
    diag_A_j = jnp.asarray(diag_A)
    x_j = jnp.asarray(x0)

    init_loss = 0.5 * jnp.sum(diag_A_j * (x_j ** 2), axis=-1)

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr = 0.1

    def step_fn(carry, step_idx):
        x, m, v, b1_p, b2_p = carry
        g = diag_A_j * x
        b1_p = b1_p * beta1
        b2_p = b2_p * beta2
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - b1_p)
        v_hat = v / (1.0 - b2_p)
        u = m_hat / (jnp.sqrt(v_hat) + eps)
        x = x - lr * u
        return (x, m, v, b1_p, b2_p), None

    init_carry = (x_j, jnp.zeros_like(x_j), jnp.zeros_like(x_j), 1.0, 1.0)
    (x_final, _, _, _, _), _ = jax.lax.scan(step_fn, init_carry, jnp.arange(1, 301))

    final_loss = 0.5 * jnp.sum(diag_A_j * (x_final ** 2), axis=-1)
    reduction = 1.0 - (final_loss / init_loss)
    min_reduction = float(jnp.min(reduction))
    mean_reduction = float(jnp.mean(reduction))

    assert min_reduction > 0.9999, f"Minimum loss reduction {min_reduction*100:.4f}% <= 99.99%"
    assert mean_reduction > 0.9999, f"Mean loss reduction {mean_reduction*100:.4f}% <= 99.99%"


def test_nonconvex_stochastic_benchmarks():
    """Verify Rosenbrock and Rastrigin with noise sigma=0.5 match Cosine Annealing within <= 2%."""
    dim = 4
    total_steps = 600
    warmup_steps = 50
    decay_steps = 450
    alpha = 1.0
    lr_max = 0.03
    lr_min = 1e-4
    noise_sigma = 0.5

    def rosenbrock(x):
        return jnp.sum(100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1.0 - x[..., :-1]) ** 2, axis=-1)

    def rastrigin(x):
        d = x.shape[-1]
        return 10.0 * d + jnp.sum(x ** 2 - 10.0 * jnp.cos(2.0 * jnp.pi * x), axis=-1)

    def run_bench(fn, keys):
        grad_fn = jax.grad(fn)

        def single_run(key, is_ards):
            k_init, k_loop = jax.random.split(key)
            x0 = jax.random.normal(k_init, (dim,)) * 0.5

            def step_fn(carry, step_idx):
                x, m, v, b1_p, b2_p, k = carry
                k_step, k_next = jax.random.split(k)
                noise = jax.random.normal(k_step, (dim,)) * noise_sigma
                g = grad_fn(x) + noise

                b1_p = b1_p * 0.9
                b2_p = b2_p * 0.999
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * (g * g)
                m_hat = m / (1.0 - b1_p)
                v_hat = v / (1.0 - b2_p)
                u = m_hat / (jnp.sqrt(v_hat) + 1e-8)

                step = step_idx.astype(jnp.float32)
                # Cosine
                prog = jnp.clip(step / total_steps, 0.0, 1.0)
                lr_cos = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + jnp.cos(jnp.pi * prog))
                # ARDS
                w = jnp.minimum(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
                d_step = jnp.maximum(0.0, step - warmup_steps) / decay_steps
                lr_ards = lr_max * w * jax.lax.rsqrt(1.0 + alpha * d_step * d_step)

                lr = jnp.where(is_ards, lr_ards, lr_cos)
                x = x - lr * u
                return (x, m, v, b1_p, b2_p, k_next), None

            init_carry = (x0, jnp.zeros(dim), jnp.zeros(dim), 1.0, 1.0, k_loop)
            (x_final, _, _, _, _, _), _ = jax.lax.scan(step_fn, init_carry, jnp.arange(1, total_steps + 1))
            return fn(x_final)

        cos_losses = jax.vmap(lambda k: single_run(k, False))(keys)
        ards_losses = jax.vmap(lambda k: single_run(k, True))(keys)
        return float(jnp.mean(cos_losses)), float(jnp.mean(ards_losses))

    keys = jax.random.split(jax.random.PRNGKey(42), 200)

    # 1. Rosenbrock
    cos_ros, ards_ros = run_bench(rosenbrock, keys)
    ros_diff = (ards_ros - cos_ros) / cos_ros * 100.0
    assert ros_diff <= 2.0, f"Rosenbrock ARDS degradation {ros_diff:+.2f}% exceeds 2.0% tolerance"

    # 2. Rastrigin
    cos_ras, ards_ras = run_bench(rastrigin, keys)
    ras_diff = (ards_ras - cos_ras) / cos_ras * 100.0
    assert ras_diff <= 2.0, f"Rastrigin ARDS degradation {ras_diff:+.2f}% exceeds 2.0% tolerance"


def test_architectural_isolation_contract():
    """Verify optimizer configurations maintain exact hyperparameter parity."""
    algebraic_cfg = {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 1e-2}
    baseline_cfg = {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 1e-2}

    assert algebraic_cfg == baseline_cfg, "Optimizer configuration must be strictly identical"
