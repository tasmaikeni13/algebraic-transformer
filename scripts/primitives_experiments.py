"""Prespecified Primitive study experiments, shared by CPU and TPU runners.

Random input generation, statistical summaries, and transcendental comparison
baselines live here; none are part of the algebraic production graph.
"""

from functools import partial
import numpy as np
from scipy.stats import t
import jax
import jax.numpy as jnp

from src.primitives import alu, avn
from tests.reference_primitives import alu_derivative, alu_second_derivative, avn_reference


def summary(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.
    radius = float(t.ppf(.975, values.size-1) * sem) if values.size > 1 else 0.
    return {"n": values.size, "mean": mean, "sem": sem, "ci95": [mean-radius, mean+radius],
            "min": float(values.min()), "max": float(values.max())}


def numerical(seed=42):
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.uniform(-30, 30, 100_000), [-np.sqrt(2), 0, np.sqrt(2)]])
    z = jnp.asarray(x, dtype=jnp.float64)
    u = x / np.sqrt(1 + x*x)
    actual = np.asarray(jax.grad(lambda a: alu(a).sum())(z))
    automatic = np.asarray(jax.grad(lambda a: jnp.sum(.5*a*(1+a/jnp.sqrt(1+a*a))))(z))
    errors = {"reflection": float(np.max(np.abs(.5*(1+u)+.5*(1-u)-1))),
              "backward_autodiff": float(np.max(np.abs(actual-automatic))),
              "backward_numpy": float(np.max(np.abs(actual-alu_derivative(x)))),
              "lipschitz": float(np.max(np.abs(actual))),
              "inflection": float(np.max(np.abs(jax.vmap(jax.grad(jax.grad(alu)))(jnp.array([-np.sqrt(2),np.sqrt(2)])))))}
    limits = {"reflection": 1e-15, "backward_autodiff": 5e-16, "backward_numpy": 5e-16,
              "lipschitz": 1.05, "inflection": 1e-15}
    return {"samples": len(x), "metrics": errors, "limits": limits,
            "passed": all(errors[k] <= limits[k] for k in limits)}


def monte_carlo(seed=42, samples=1_000_000):
    """20 independent vectors per scale, 1e6 scalar samples per scale.

    CI units are independent vectors, since coordinates within a vector are
    dependent. Check the regularized moment identities and the unregularized
    unit-variance interval separately.
    """
    rng = np.random.default_rng(seed)
    repetitions = 20
    if samples % repetitions:
        raise ValueError("samples must be divisible by 20")
    rows = []
    for sigma in (.1, .2, .5, 1., 2., 5., 10.):
        x = rng.normal(size=(repetitions, samples//repetitions))*sigma
        m2, var = np.mean(x*x,axis=-1), np.var(x,axis=-1)
        for eps in (1e-5, 0.):
            y = np.asarray(avn(jnp.asarray(x, dtype=jnp.float64), eps))
            empirical_m2, empirical_var = np.mean(y*y,axis=-1), np.var(y,axis=-1)
            m2_error = float(np.max(np.abs(empirical_m2-m2/(m2+eps))))
            var_error = float(np.max(np.abs(empirical_var-var/(m2+eps))))
            stats = summary(empirical_var)
            near_unit = .9999 <= stats["ci95"][0] and stats["ci95"][1] <= 1.0001
            rows.append({"sigma": sigma, "eps": eps, "variance": stats, "second_moment": summary(empirical_m2),
                         "moment_identity_error": m2_error, "variance_identity_error": var_error,
                         "historical_unit_variance_gate": near_unit,
                         "passed": m2_error <= 1e-12 and var_error <= 1e-12 and (eps > 0 or near_unit)})
    # Nonzero means distinguish centered variance from the raw second moment.
    x = rng.normal(size=(20, 1000))*.1 + 3.
    y = np.asarray(avn(x))
    expected = np.var(x,axis=-1)/(np.mean(x*x,axis=-1)+1e-5)
    noncentered_error = float(np.max(np.abs(np.var(y,axis=-1)-expected)))
    return {"gate_version": 2, "samples_per_scale": samples, "rows": rows,
            "noncentered_variance_identity_error": noncentered_error,
            "passed": all(row["passed"] for row in rows) and noncentered_error <= 1e-12}


def gelu(x):
    return jax.nn.gelu(x, approximate=False)


def rmsnorm(x, gamma, eps=1e-5):
    z = x.astype(jnp.promote_types(x.dtype, jnp.float32))
    return (z*jax.lax.rsqrt(jnp.mean(z*z,axis=-1,keepdims=True)+eps)*gamma).astype(x.dtype)


def layernorm(x, gamma, beta, eps=1e-5):
    z = x.astype(jnp.promote_types(x.dtype, jnp.float32))
    centered = z-jnp.mean(z,axis=-1,keepdims=True)
    return (centered*jax.lax.rsqrt(jnp.mean(centered*centered,axis=-1,keepdims=True)+eps)*gamma+beta).astype(x.dtype)


def deep_function(activation, depth, residual=True):
    """Independent He-initialized two-matrix blocks, identical across arms.

    h[l+1] = AVN(h[l] + rsqrt(2D) Wdown act(Wup AVN(h[l]))).
    Baseline normalization uses RMSNorm with gamma initialized to one.
    Unattenuated, non-residual composition is retained as a mechanism ablation.
    """
    norm = avn if activation is alu else lambda z: rmsnorm(z, jnp.ones(z.shape[-1],dtype=z.dtype))

    def single(x, weights, terminal):
        def network(h):
            def layer(h, w):
                branch = activation(norm(h) @ w[0]) @ w[1]
                next_h = h + jax.lax.rsqrt(jnp.asarray(2.*depth,dtype=h.dtype))*branch if residual else branch
                return norm(next_h), None
            return jax.lax.scan(layer,h,weights)[0]
        y, backward = jax.vjp(network, x)
        g = backward(terminal)[0]
        ratio = jnp.sqrt(jnp.sum(g*g)/jnp.sum(terminal*terminal))
        variance_ratio = jnp.var(y)/jnp.var(x)
        gradient_variance = jnp.var(g)
        return jnp.stack([ratio,variance_ratio,gradient_variance])
    return jax.jit(jax.vmap(single))


def deep_trials(seed=42, trials=10_000, width=128, batch_size=100, residual=True,
                place=None, gather=None, process_index=0, process_count=1, progress=print):
    place = place or jnp.asarray
    gather = gather or np.asarray
    rows, raw = [], {}
    # Each trial gets an independent input, terminal cotangent, and every weight.
    # Process streams are independent; data are paired exactly between all arms.
    rng = np.random.default_rng(np.random.SeedSequence([seed,process_index]))
    local_trials = (trials+process_count-1)//process_count
    for depth in (8,16,24,32):
        funcs = {"alu": deep_function(alu,depth,residual), "gelu": deep_function(gelu,depth,residual),
                 "swish": deep_function(jax.nn.silu,depth,residual)}
        chunks = {name: [] for name in funcs}
        for start in range(0,local_trials,batch_size):
            x = rng.normal(size=(batch_size,width)).astype(np.float32)
            terminal = rng.normal(size=x.shape).astype(np.float32)
            terminal /= np.sqrt(np.sum(terminal*terminal,axis=-1,keepdims=True))
            weights = (rng.normal(size=(batch_size,depth,2,width,width))*np.sqrt(2/width)).astype(np.float32)
            args = [place(a) for a in (x,weights,terminal)]
            for name, fn in funcs.items():
                values = np.asarray(gather(jax.block_until_ready(fn(*args))))
                # All hosts have equal padding; full production trial counts divide four.
                keep = min(batch_size,local_trials-start)*process_count
                chunks[name].append(values[:keep])
        arrays = {name: np.concatenate(values,axis=0)[:trials] for name,values in chunks.items()}
        for name, values in arrays.items():
            raw[f"{name}_depth{depth}"] = values
        base = arrays["gelu"][:,2]
        delta = abs(float(arrays["alu"][:,2].mean()/base.mean()-1))
        # Paired delta uncertainty (delta method on the ratio of means).
        ratio = arrays["alu"][:,2].mean()/base.mean()
        influence = (arrays["alu"][:,2]-ratio*base)/base.mean()
        delta_sem = float(influence.std(ddof=1)/np.sqrt(trials))
        arms = {name:{"gradient_ratio":summary(v[:,0]),"activation_variance_ratio":summary(v[:,1]),
                      "gradient_variance":summary(v[:,2])} for name,v in arrays.items()}
        alu_values = arrays["alu"]
        bounded = bool(np.all((alu_values[:,0]>=.2)&(alu_values[:,0]<=5)))
        activation_ok = bool(np.all((alu_values[:,1]>=.5)&(alu_values[:,1]<=2))) if depth==32 else True
        row = {"depth":depth,"arms":arms,"gradient_variance_relative_delta":delta,
               "gradient_variance_delta_sem":delta_sem,
               "gradient_variance_delta_ci95":[float(ratio-1-1.96*delta_sem),float(ratio-1+1.96*delta_sem)],
               "bounded_gradient_gate":bounded,"activation_gate":activation_ok,"parity_gate":delta<=.05,
               "passed":bounded and activation_ok and delta<=.05}
        rows.append(row)
        progress(f"depth={depth}, ALU gradient range={arms['alu']['gradient_ratio']['min']:.4f}..{arms['alu']['gradient_ratio']['max']:.4f}, variance delta={delta:.3%}")
    return {"seed":seed,"trials_per_depth":trials,"width":width,"dtype":"float32","batch_size_per_process":batch_size,
            "residual":residual,"rows":rows,"passed":all(row["passed"] for row in rows)},raw
