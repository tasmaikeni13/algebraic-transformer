from pathlib import Path
import ast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.attention import octic_kernel, algebraic_softmax, _normalized_forward
from tests.reference_attention import kernel, attention, attention_vjp
from scripts.audit_primitives import source_audit, primitives_in, FORBIDDEN


@pytest.mark.parametrize('dtype,tol', [(jnp.float64,2e-13),(jnp.float32,2e-5),(jnp.bfloat16,.016)])
@pytest.mark.parametrize('sink', [0.,.5,1.])
def test_oracle_and_vjp(dtype,tol,sink):
    rng=np.random.default_rng(42)
    a=jnp.asarray(rng.normal(size=(4,128)),dtype=dtype)
    g=jnp.asarray(rng.normal(size=a.shape),dtype=dtype)
    y,back=jax.vjp(lambda x:algebraic_softmax(x,sink),a)
    ref=attention(np.asarray(a,dtype=float),sink)
    refg=attention_vjp(np.asarray(a,dtype=float),np.asarray(g,dtype=float),sink)
    np.testing.assert_allclose(np.asarray(y,dtype=float),ref,rtol=tol,atol=tol)
    np.testing.assert_allclose(np.asarray(back(g)[0],dtype=float),refg,rtol=tol,atol=tol)
    assert y.dtype==a.dtype and back(g)[0].dtype==a.dtype


def test_kernel_chain_tail_and_derivative():
    x=jnp.asarray(np.r_[np.linspace(-64,64,1001),-1e3,1e3],dtype=jnp.float64)
    np.testing.assert_allclose(octic_kernel(x),kernel(x),rtol=2e-13)
    grad=jax.grad(lambda a:octic_kernel(a).sum())(x)
    np.testing.assert_allclose(grad,8*kernel(x)/np.sqrt(1+np.asarray(x)**2),rtol=2e-13)
    assert float(octic_kernel(jnp.array(3.))/octic_kernel(jnp.array(-3.)))>1e5
    np.testing.assert_allclose(octic_kernel(jnp.array(2.)),51841+23184*np.sqrt(5),rtol=2e-15)


def test_jacobian_coordinates_and_chain_rule():
    rng=np.random.default_rng(14)
    for sink in (0.,.5):
        fn=lambda z:_normalized_forward(z,sink)[0]
        for x in (np.zeros(2),rng.normal(size=64)):
            jac=jax.jacrev(fn)(jnp.asarray(x))
            assert np.max(np.abs(jac))<=2+1e-14
    # Entrywise 2 is not a spectral-norm or raw-score bound.
    normalized=jax.jacrev(lambda z:_normalized_forward(z,0.)[0])(jnp.zeros(2))
    np.testing.assert_allclose(normalized,[[2,-2],[-2,2]])
    assert np.linalg.norm(normalized,2)==4
    raw=jax.jacrev(lambda z:algebraic_softmax(z,0.))(jnp.zeros(2))
    np.testing.assert_allclose(raw,normalized/np.sqrt(1e-5))
    x=jnp.asarray(rng.normal(size=16))
    g=jnp.asarray(rng.normal(size=16))
    direct=lambda z:_normalized_forward(z*jax.lax.rsqrt(jnp.mean(z*z)+1e-5),.5)[0]
    np.testing.assert_allclose(jax.vjp(direct,x)[1](g)[0],jax.vjp(algebraic_softmax,x)[1](g)[0],rtol=2e-13,atol=2e-13)


@pytest.mark.parametrize('length',[1,64,4096])
def test_simplex_extremes(length):
    rng=np.random.default_rng(9)
    for x in (np.zeros((2,length)),np.full((2,length),-1e15),rng.normal(size=(2,length))*1e15):
        for sink in (0.,.5,1.):
            p=np.asarray(algebraic_softmax(jnp.asarray(x),sink))
            assert np.isfinite(p).all() and np.all(p>0)
            assert np.max(p.sum(axis=-1))<=1+16*np.finfo(p.dtype).eps


@pytest.mark.parametrize('kwargs',[{'sink_omega':-1.},{'sink_omega':float('nan')},{'eps':0.},{'eps':-1.}])
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError): algebraic_softmax(jnp.ones(4),**kwargs)


def test_invalid_input():
    with pytest.raises(TypeError): algebraic_softmax(jnp.arange(4))
    with pytest.raises(ValueError): algebraic_softmax(jnp.ones(()))
    with pytest.raises(ValueError): algebraic_softmax(jnp.ones((2,0)))


def test_purity_and_three_squarings():
    source=Path('src/attention.py').read_text()
    assert not source_audit(source)
    assert not any(isinstance(n,ast.Attribute) and n.attr in {'softmax','sqrt'} for n in ast.walk(ast.parse(source)))
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_kernel')
    squares=[n for n in ast.walk(fn) if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Mult)
             and isinstance(n.left,ast.Name) and isinstance(n.right,ast.Name) and n.left.id==n.right.id and n.left.id in {'rho','k2','k4'}]
    assert len(squares)==3
    x=jnp.ones((2,64))
    for f in (octic_kernel,algebraic_softmax,jax.grad(lambda z:algebraic_softmax(z).sum())):
        assert not set(primitives_in(jax.make_jaxpr(f)(x))) & (FORBIDDEN|{'sqrt'})


def test_jacobian_evidence_is_json_serializable():
    import json
    from scripts.attention_experiments import jacobian_study
    record,raw=jacobian_study(trials=10,progress=lambda _:None)
    assert json.loads(json.dumps(record,allow_nan=False))['passed'] is True
    assert sum(a.shape[0] for a in raw.values())==10


def test_numpy_evidence_scalars_and_nonfinite_rejection(tmp_path):
    import json
    from scripts.attention_records import write_json
    path=tmp_path/'record.json'
    write_json(path,{'passed':np.bool_(False),'tol':np.float32(.016),'trials':np.int64(10000)})
    record=json.loads(path.read_text())
    assert record['passed'] is False and record['trials']==10000
    assert record['tol']==float(np.float32(.016))
    with pytest.raises(ValueError):write_json(path,{'error':np.float32(np.nan)})
    assert json.loads(path.read_text())==record


def test_w1_attention_distribution_parity():
    from scripts.attention_experiments import w1
    from tests.reference_attention import attention, softmax
    rng = np.random.default_rng(42)
    for length in (64, 128, 256, 512, 1024):
        x = rng.normal(size=(50, length))
        p = attention(x)
        q = softmax(x)
        dist = w1(p, q)
        assert np.mean(dist) <= 0.05, f"Wasserstein-1 delta {np.mean(dist)} exceeded 0.05 at L={length}"


def test_quantization_noise_suppression():
    from scripts.attention_experiments import quantization_robustness
    res = quantization_robustness(seed=42, trials=50)
    assert res['passed'] is True
    assert res['benchmark_trial']['noise_suppression_ratio'] >= 100.0

