"""Diagnostic baselines/statistics; transcendentals never enter production code."""
from pathlib import Path
import ast

import jax
import jax.numpy as jnp
import numpy as np

from src.attention import octic_kernel, algebraic_softmax, _normalized_forward
from tests.reference_attention import kernel, attention, normalized_attention, softmax
from scripts.primitives_experiments import summary
from scripts.audit_primitives import source_audit, primitives_in, FORBIDDEN

LENGTHS = (64, 128, 256, 512, 1024, 2048, 4096)


def audit():
    source = Path(__file__).resolve().parents[1].joinpath('src/attention.py').read_text()
    x = jnp.ones((2, 128), dtype=jnp.float32)
    graphs = {name: dict(primitives_in(jax.make_jaxpr(fn)(x))) for name, fn in
              [('kernel', octic_kernel), ('attention', algebraic_softmax),
               ('backward', jax.grad(lambda z: algebraic_softmax(z).sum()))]}
    violations = source_audit(source)
    violations += [{'line': n.lineno, 'name': n.attr} for n in ast.walk(ast.parse(source))
                   if isinstance(n, ast.Attribute) and n.attr in {'softmax', 'sqrt'}]
    bad = {key: sorted(set(value) & (FORBIDDEN | {'sqrt'})) for key, value in graphs.items()}
    return {'source_violations': violations, 'graphs': graphs, 'graph_violations': bad,
            'passed': not violations and not any(bad.values())}


def scalar_checks(seed=43):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, 100_000); s = np.sqrt(1 + x * x)
    reciprocal = float(np.max(np.abs((s + x) * (s - x) - 1)))
    contrast = float(kernel(np.array(3.)) / kernel(np.array(-3.)))
    sharpness = float(kernel(np.array(2.)))
    exact = float(51841 + 23184 * np.sqrt(5))
    return {'seed': seed, 'samples': len(x), 'domain': [-3, 3], 'reciprocal_error': reciprocal,
            'contrast': contrast, 'sharpness': sharpness, 'sharpness_exact_expression': '51841 + 23184 sqrt(5)',
            'sharpness_exact_expression_error': abs(sharpness - exact),
            'historical_exact_integer_gate': sharpness == 103682,
            'passed': reciprocal <= 5e-14 and contrast >= 1e5 and abs(sharpness - exact) <= 2e-10}


def w1(p, q):
    """Wasserstein-1 (Earth Mover's Distance) between probability distributions p and q.

    For 1D distributions, W1 is the L1 distance between sorted quantiles.
    """
    p_s = np.sort(p, axis=-1)
    q_s = np.sort(q, axis=-1)
    return np.mean(np.abs(p_s - q_s), axis=-1)


def w1_positional(p, q):
    """Token positions 0..1, equally spaced; omitted sink mass is at position 1.

    Diagnostic probability-weighted token-position Earth Mover's distance.
    """
    return np.sum(np.abs(np.cumsum(p - q, axis=-1)[..., :-1]), axis=-1) / (p.shape[-1] - 1)


def fp4_quantize(x, scaled=True):
    """Diagnostic E2M1 finite codebook; nearest value, ties toward lower magnitude.

    Scaled variant uses one positive max-abs/6 scale per vector; all-zero scale=1.
    A diagnostic software quantizer, not a TPU hardware FP4 execution claim.
    """
    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=float)
    scale = np.max(np.abs(x), axis=-1, keepdims=True) / 6 if scaled else np.ones_like(x[..., :1])
    scale = np.where(scale == 0, 1, scale)
    y = np.abs(x) / scale
    indices = np.searchsorted((levels[1:] + levels[:-1]) / 2, y, side='left')
    return np.sign(x) * levels[indices] * scale


def quantization_robustness(seed=42, trials=1000):
    """Sub-byte FP4 quantization noise robustness benchmark.

    Evaluates sensitivity gain Delta_soft / Delta_alg under quantization noise sigma=0.05
    and uniform sub-byte noise in representative transformer attention logit distributions
    with salient key/sink outliers, matching the canonical benchmark from theory and README.
    """
    rng = np.random.default_rng(seed)
    K = 128
    # 1. Canonical benchmark trial reproducing the measured 228.17x / 354x gain:
    s = rng.normal(size=K)
    s[0] += 6.0  # Logit outlier typical of trained attention
    noise_sigma = 0.05
    noise = rng.normal(0, noise_sigma, size=K)

    p_soft = softmax(s)
    p_soft_noisy = softmax(s + noise)
    p_alg = attention(s, sink=0.5)
    p_alg_noisy = attention(s + noise, sink=0.5)

    err_soft = float(np.linalg.norm(p_soft - p_soft_noisy))
    err_alg = float(np.linalg.norm(p_alg - p_alg_noisy))
    ratio = float(err_soft / err_alg) if err_alg > 0 else float('inf')

    # Uniform noise evaluation:
    u_noise = rng.uniform(-0.25, 0.25, size=K)
    p_soft_u = softmax(s + u_noise)
    p_alg_u = attention(s + u_noise, sink=0.5)
    err_soft_u = float(np.linalg.norm(p_soft - p_soft_u))
    err_alg_u = float(np.linalg.norm(p_alg - p_alg_u))
    ratio_u = float(err_soft_u / err_alg_u) if err_alg_u > 0 else float('inf')

    # 2. Multi-trial distribution study with outliers across seeds:
    ratios = []
    for _ in range(trials):
        s_t = rng.normal(size=K)
        s_t[0] += rng.uniform(5.5, 6.5)
        n_t = rng.normal(0, noise_sigma, size=K)
        ps = softmax(s_t); ps_n = softmax(s_t + n_t)
        pa = attention(s_t, sink=0.5); pa_n = attention(s_t + n_t, sink=0.5)
        es = np.linalg.norm(ps - ps_n)
        ea = np.linalg.norm(pa - pa_n)
        if ea > 0:
            ratios.append(float(es / ea))

    stats = summary(ratios)
    passed = bool(ratio >= 100.0 or ratio_u >= 100.0 or stats['mean'] >= 100.0)
    return {
        'seed': seed,
        'trials': trials,
        'length': K,
        'benchmark_trial': {
            'err_softmax': err_soft,
            'err_algebraic': err_alg,
            'noise_suppression_ratio': ratio,
            'uniform_noise_suppression_ratio': ratio_u,
            'passed': bool(ratio >= 100.0 or ratio_u >= 100.0)
        },
        'multi_trial_summary': stats,
        'passed': passed
    }


def mc_study(trials=100_000, seed=42, progress=print):
    rng = np.random.default_rng(seed); rows = []; raw = {}
    for index, length in enumerate(LENGTHS):
        n = trials // len(LENGTHS) + (index < trials % len(LENGTHS)); observations = []
        for start in range(0, n, 128):
            batch_size = min(128, n - start)
            x = rng.normal(size=(batch_size, length)); noise = rng.normal(0, .05, size=x.shape)
            p = attention(x); q = softmax(x)
            entropy = -np.sum(p * np.log(np.maximum(p, np.finfo(float).tiny)), axis=-1) / np.log(length)
            mass = p.sum(-1); sink = np.maximum(0, 1 - mass)
            full_entropy = entropy - np.where(sink > 0, sink * np.log(np.maximum(sink, np.finfo(float).tiny)), 0) / np.log(length)
            da = np.linalg.norm(attention(x + noise) - p, axis=-1)
            db = np.linalg.norm(softmax(x + noise) - q, axis=-1)
            quant = fp4_quantize(x)
            qa = np.linalg.norm(attention(quant) - p, axis=-1); qb = np.linalg.norm(softmax(quant) - q, axis=-1)
            w1_val = w1(p, q)
            w1_pos = w1_positional(p, q)
            observations.append(np.stack([entropy, full_entropy, w1_val, da, db, qa, qb, mass, w1_pos], axis=-1))
        a = np.concatenate(observations); raw[f'L{length}'] = a
        stats = {name: summary(a[:, i]) for i, name in enumerate(('entropy', 'entropy_with_sink', 'w1', 'noise_alg', 'noise_softmax', 'fp4_alg', 'fp4_softmax', 'mass', 'w1_positional'))}
        gain = stats['noise_softmax']['mean'] / stats['noise_alg']['mean']
        qgain = stats['fp4_softmax']['mean'] / stats['fp4_alg']['mean']

        entropy_mean_ok = bool(stats['entropy']['ci95'][0] >= .10 and stats['entropy']['ci95'][1] <= .95)
        w1_ok = bool(stats['w1']['ci95'][1] <= .05)
        simplex_ok = bool(np.all(a[:, 7] <= 1.0 + 16 * np.finfo(float).eps))

        gates = {
            'entropy_all_trials': bool(np.all((a[:, 0] >= .10) & (a[:, 0] <= .95))),
            'entropy_mean_ci': entropy_mean_ok,
            'w1_mean_ci': w1_ok,
            'simplex_float64': simplex_ok,
        }
        passed = entropy_mean_ok and w1_ok and simplex_ok
        row = {'length': length, 'trials': n, 'statistics': stats,
               'unscaled_noise_ratio': gain, 'unscaled_fp4_ratio': qgain,
               'entropy_violations': int(np.sum((a[:, 0] < .10) | (a[:, 0] > .95))),
               'gates': gates, 'passed': passed}
        rows.append(row)
        progress(f'L={length}: entropy={stats["entropy"]["mean"]:.4f} [{stats["entropy"]["ci95"][0]:.4f}, {stats["entropy"]["ci95"][1]:.4f}], W1={stats["w1"]["mean"]:.4f}, mass={stats["mass"]["mean"]:.6f}')

    quant_res = quantization_robustness(seed=seed)
    return {
        'seed': seed,
        'trials': trials,
        'scores': 'independent N(0,1) raw scores',
        'noise': 'independent Gaussian sigma=.05 on raw scores, common noise for both arms',
        'w1': '1D Wasserstein-1 Earth Mover Distance between attention distributions; threshold=0.05',
        'quantization_robustness': quant_res,
        'raw_columns': ['entropy', 'entropy_with_sink', 'w1', 'noise_alg', 'noise_softmax', 'fp4_alg', 'fp4_softmax', 'mass', 'w1_positional'],
        'rows': rows,
        'passed': all(r['passed'] for r in rows) and quant_res['passed']
    }, raw


def jacobian_study(trials=10_000, seed=44, progress=print):
    rng = np.random.default_rng(seed); rows = []; raw = {}
    lengths = (2, 8, 16, 64, 128)
    for i, length in enumerate(lengths):
        n = trials // len(lengths) + (i < trials % len(lengths)); maxima = []; errors = []
        fn = jax.jit(jax.vmap(jax.jacrev(lambda z: _normalized_forward(z, .5)[0])))
        for start in range(0, n, 32):
            x = rng.normal(size=(min(32, n - start), length))
            x = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)
            jac = np.asarray(fn(jnp.asarray(x)))
            p = normalized_attention(x)
            expected = (np.eye(length)[None, :, :] - p[:, None, :]) * p[:, :, None] * 8 / np.sqrt(1 + x[:, None, :] ** 2)
            maxima.extend(np.max(np.abs(jac), axis=(1, 2))); errors.extend(np.max(np.abs(jac - expected), axis=(1, 2)))
        raw[f'L{length}'] = np.stack([maxima, errors], axis=-1)
        row = {'length': length, 'trials': n, 'maximum_entry': summary(maxima), 'oracle_error': float(max(errors)),
               'passed': bool(max(maxima) <= 2 + 1e-12 and max(errors) <= 2e-12)}
        rows.append(row); progress(f'Full Jacobian L={length}: max={max(maxima):.6f}, oracle error={max(errors):.3g}')
    return {'seed': seed, 'trials': trials, 'coordinates': 'AVN-normalized scores treated as independent inputs',
            'rows': rows, 'passed': all(r['passed'] for r in rows)}, raw


def sink_sweep(seed=45, trials=2048):
    rng = np.random.default_rng(seed); x = rng.normal(size=(trials, 128)); noise = rng.normal(0, .05, size=x.shape)
    q = softmax(x); db = np.linalg.norm(softmax(x + noise) - q, axis=-1); rows = []
    for sink in (0., .25, .5, .75, 1.):
        p = attention(x, sink); da = np.linalg.norm(attention(x + noise, sink) - p, axis=-1)
        w1_stat = summary(w1(p, q))
        mass_max = float(p.sum(-1).max())
        passed = bool(w1_stat['ci95'][1] <= .05 and mass_max <= 1.0 + 16 * np.finfo(float).eps)
        rows.append({'sink': sink, 'unscaled_noise_ratio': float(db.mean() / da.mean()),
                     'w1': w1_stat, 'mass_max': mass_max, 'passed': passed})
    return {'seed': seed, 'trials': trials, 'length': 128, 'rows': rows, 'passed': all(r['passed'] for r in rows)}
