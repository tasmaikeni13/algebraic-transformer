import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.audit_primitives import audit, source_audit
from src.primitives import alu, avn
from tests.reference_primitives import (
    alu_reference, alu_derivative, alu_second_derivative, avn_reference, avn_vjp_reference,
)


@pytest.mark.parametrize("shape", [(), (19,), (3, 17), (2, 3, 7)])
def test_alu_oracle(shape):
    x = np.random.default_rng(42).normal(size=shape)
    np.testing.assert_allclose(alu(x), alu_reference(x), atol=5e-16, rtol=5e-15)
    actual = jax.grad(lambda z: alu(z).sum())(jnp.asarray(x))
    np.testing.assert_allclose(actual, alu_derivative(x), atol=5e-16, rtol=0)


def test_horner_against_independent_autodiff():
    x = jnp.linspace(-30, 30, 100_000, dtype=jnp.float64)
    native = jax.grad(lambda z: jnp.sum(z / 2 * (1 + z / jnp.sqrt(1 + z * z))))(x)
    actual = jax.grad(lambda z: alu(z).sum())(x)
    assert float(jnp.max(jnp.abs(actual - native))) <= 5e-16


def test_inflections_and_global_extrema():
    second = jax.grad(jax.grad(alu))
    for sign in (-1, 1):
        x = sign * np.sqrt(2.)
        assert abs(float(second(x))) <= 1e-15
        assert float(second(x - .01) * second(x + .01)) < 0
    x = np.linspace(-100, 100, 100_000)
    np.testing.assert_allclose(jax.vmap(second)(x), alu_second_derivative(x), atol=8e-16)
    bound = .5 + 2 * np.sqrt(6) / 9
    assert np.max(np.abs(alu_derivative(x))) <= bound + 2e-16 < 1.05


@pytest.mark.parametrize("eps", [0., 1e-8, 1e-5, .1])
@pytest.mark.parametrize("shape", [(7,), (4, 13), (2, 3, 17)])
def test_avn_oracle_and_vjp(shape, eps):
    rng = np.random.default_rng(43)
    x, g = rng.normal(size=shape), rng.normal(size=shape)
    np.testing.assert_allclose(avn(x, eps), avn_reference(x, eps), rtol=2e-15, atol=2e-15)
    _, pullback = jax.vjp(lambda z: avn(z, eps), x)
    np.testing.assert_allclose(pullback(g)[0], avn_vjp_reference(x, g, eps), rtol=4e-14, atol=2e-15)


def test_regularized_moments_and_mean_are_not_confused():
    x = np.array([[.1, -.1], [3., 3.], [0., 0.]])
    y = np.asarray(avn(x))
    m2 = np.mean(x*x, axis=-1)
    np.testing.assert_allclose(np.mean(y*y, axis=-1), m2/(m2+1e-5), atol=3e-16)
    np.testing.assert_allclose(np.var(y, axis=-1), np.var(x, axis=-1)/(m2+1e-5), atol=3e-16)
    assert np.var(y[0]) < .9999  # Regularization keeps this moment below one.
    assert np.var(y[1]) == 0
    np.testing.assert_allclose(jax.jacrev(avn)(jnp.zeros(2)), np.eye(2)/np.sqrt(1e-5))


def test_scale_and_gate_coupling():
    x = np.random.default_rng(44).normal(size=(3, 16))
    for alpha in (.01, .5, 2., 100.):
        np.testing.assert_allclose(avn(alpha*x, 0.), avn(x, 0.), atol=1e-15)
        np.testing.assert_allclose(avn(alpha*x, alpha*alpha*1e-5), avn(x), atol=1e-15)
    v = np.mean(x*x, axis=-1, keepdims=True)+1e-5
    y = np.asarray(avn(x))
    np.testing.assert_allclose(.5*(1+x/np.sqrt(x*x+v)), .5*(1+y/np.sqrt(y*y+1)), atol=3e-16)
    assert not np.allclose(avn(x*.001), avn(x))


@pytest.mark.parametrize("dtype,tol", [(jnp.float64, 2e-14), (jnp.float32, 2e-6), (jnp.bfloat16, .012)])
def test_mixed_precision_boundaries(dtype, tol):
    x = jnp.array([[-1e15, -1e6, -1e-15, 0., 1e-15, 1e6, 1e15]], dtype=dtype)
    # Oracle sees the actual quantized input, not its unrounded precursor.
    x64 = np.asarray(x, dtype=np.float64)
    g = jnp.ones_like(x)
    for fn, ref, derivative in [(alu, alu_reference, lambda z,g: g*alu_derivative(z)),
                                (avn, avn_reference, avn_vjp_reference)]:
        y, back = jax.vjp(fn, x)
        dx = back(g)[0]
        assert y.dtype == dtype and dx.dtype == dtype
        assert np.isfinite(np.asarray(y,dtype=float)).all()
        assert np.isfinite(np.asarray(dx,dtype=float)).all()
        np.testing.assert_allclose(np.asarray(y,dtype=float), ref(x64), rtol=tol, atol=tol)
        np.testing.assert_allclose(np.asarray(dx,dtype=float), derivative(x64,np.ones_like(x64)), rtol=tol, atol=tol)


def test_avn_directional_derivative():
    rng = np.random.default_rng(45)
    x, g, direction = [rng.normal(size=(4, 9)) for _ in range(3)]
    _, back = jax.vjp(avn, x)
    analytic = np.sum(back(g)[0]*direction)
    delta = 1e-5
    finite_difference = np.sum((avn(x+delta*direction)-avn(x-delta*direction))*g)/(2*delta)
    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-8, atol=1e-8)


def test_invalid_inputs():
    for x in (jnp.array(1.), jnp.empty((2, 0))):
        with pytest.raises(ValueError):
            avn(x)
    with pytest.raises(ValueError):
        avn(jnp.ones(3), -1.)
    with pytest.raises(TypeError):
        alu(jnp.ones(3, dtype=jnp.int32))


def test_zero_transcendentals_and_cached_backwards():
    assert audit()["passed"]


@pytest.mark.parametrize("source", ["jnp.exp(x)", "from math import sin as f\ny=f(x)", "jnp.tanh(x)", "x**0.25"])
def test_audit_rejects_forbidden_operations(source):
    assert source_audit(source)
