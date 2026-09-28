"""Position study scientific experiments and verification benchmarks for AGO."""
import ast
import io
import re
import time
import tokenize
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts.audit_primitives import FORBIDDEN, primitives_in, source_audit
from scripts.primitives_experiments import summary
from src.attention import apply_ago_rotations, build_cayley_rotary_matrix
from tests.reference_ago import (
    apply_ago_rotations_fp64,
    apply_rope_rotations_fp64,
    build_cayley_rotary_matrix_fp64,
    cayley_rotation_matrix_fp64,
)

ROOT = Path(__file__).resolve().parents[1]


def audit():
    """Static AST, token, and traced primitive graph audit for zero transcendentals."""
    source_path = ROOT / "src/attention.py"
    source = source_path.read_text()
    violations = source_audit(source)

    # Check for forbidden attributes including sin, cos, sqrt, pow
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in (FORBIDDEN | {"sqrt", "sin", "cos"}):
            violations.append({"line": node.lineno, "name": node.attr})
        if isinstance(node, ast.Name) and node.id in (FORBIDDEN | {"sin", "cos"}):
            violations.append({"line": node.lineno, "name": node.id})

    # Regex audit over non-comment/non-string tokens
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in {tokenize.COMMENT, tokenize.STRING}:
            tokens.append(token.string)
    regex_hits = re.findall(r"\b(?:exp|log|sin|cos|tanh|sigmoid)\b", " ".join(tokens))

    # Primitive tracer audit over JAX graphs
    q = jnp.ones((2, 16, 4, 16), dtype=jnp.float32)
    k = jnp.ones((2, 16, 4, 16), dtype=jnp.float32)
    rot = build_cayley_rotary_matrix(16, 16)

    traces = {
        "build_cayley": jax.make_jaxpr(lambda: build_cayley_rotary_matrix(16, 16))(),
        "apply_rotations": jax.make_jaxpr(apply_ago_rotations)(q, k, rot),
        "rotations_grad": jax.make_jaxpr(
            jax.grad(lambda a, b: jnp.sum(apply_ago_rotations(a, b, rot)[0]), argnums=0)
        )(q, k),
    }
    graphs = {name: dict(primitives_in(trace)) for name, trace in traces.items()}
    bad = {key: sorted(set(val) & (FORBIDDEN | {"sqrt", "sin", "cos"})) for key, val in graphs.items()}

    passed = not violations and not regex_hits and not any(bad.values())
    return {
        "source_violations": violations,
        "regex_hits": regex_hits,
        "graphs": graphs,
        "graph_violations": bad,
        "passed": passed,
    }


def determinant_and_orthogonality_study(trials=100_000, seed=42):
    """Verifies unimodularity det(R(w))=1 and orthogonality c1 . c2 = 0 to 1e-15."""
    rng = np.random.default_rng(seed)
    # Log-spaced frequencies across [1e-5, 1e2]
    log_w = rng.uniform(-5.0, 2.0, size=trials)
    w = 10.0 ** log_w

    R = cayley_rotation_matrix_fp64(w) # (trials, 2, 2)
    det = R[:, 0, 0] * R[:, 1, 1] - R[:, 0, 1] * R[:, 1, 0]
    det_err = np.abs(det - 1.0)
    max_det_err = float(np.max(det_err))

    col1 = R[:, :, 0]
    col2 = R[:, :, 1]
    ortho = col1[:, 0] * col2[:, 0] + col1[:, 1] * col2[:, 1]
    ortho_err = np.abs(ortho)
    max_ortho_err = float(np.max(ortho_err))

    col1_norm_sq = col1[:, 0] ** 2 + col1[:, 1] ** 2
    col2_norm_sq = col2[:, 0] ** 2 + col2[:, 1] ** 2
    max_col1_norm_err = float(np.max(np.abs(col1_norm_sq - 1.0)))
    max_col2_norm_err = float(np.max(np.abs(col2_norm_sq - 1.0)))

    passed = (max_det_err <= 1.0e-15) and (max_ortho_err <= 1.0e-15)
    return {
        "trials": trials,
        "domain": [1.0e-5, 100.0],
        "max_determinant_error": max_det_err,
        "max_orthogonality_error": max_ortho_err,
        "max_col1_norm_error": max_col1_norm_err,
        "max_col2_norm_error": max_col2_norm_err,
        "det_stats": summary(det_err),
        "passed": passed,
    }


def shift_equivariance_study(max_seq_len=4096, dim=64, stride=16):
    """Verifies matrix shift equivariance ||R_m^T R_n - R_{n-m}||_inf <= 1.0e-6."""
    c_table, s_table = build_cayley_rotary_matrix_fp64(dim, max_seq_len + 1)
    # Check across channel pairs and sampled position pairs
    positions = np.arange(0, max_seq_len + 1, stride, dtype=np.int32)
    M, N = np.meshgrid(positions, positions, indexing="ij")
    M_flat = M.ravel()
    N_flat = N.ravel()

    diff = N_flat - M_flat
    abs_diff = np.abs(diff)
    sgn = np.sign(diff)

    # For each channel pair k:
    # R_m^T R_n top-left: C_m C_n + S_m S_n vs C_{|n-m|}
    # R_m^T R_n bottom-left: C_m S_n - S_m C_n vs sgn * S_{|n-m|}
    max_c_err = 0.0
    max_s_err = 0.0

    for k in range(dim // 2):
        c_k = c_table[:, k]
        s_k = s_table[:, k]

        c_prod = c_k[M_flat] * c_k[N_flat] + s_k[M_flat] * s_k[N_flat]
        s_prod = c_k[M_flat] * s_k[N_flat] - s_k[M_flat] * c_k[N_flat]

        target_c = c_k[abs_diff]
        target_s = sgn * s_k[abs_diff]

        c_err = np.max(np.abs(c_prod - target_c))
        s_err = np.max(np.abs(s_prod - target_s))
        if c_err > max_c_err:
            max_c_err = float(c_err)
        if s_err > max_s_err:
            max_s_err = float(s_err)

    max_shift_err = max(max_c_err, max_s_err)
    passed = max_shift_err <= 1.0e-6
    return {
        "max_seq_len": max_seq_len,
        "dim": dim,
        "pairs_evaluated": len(M_flat) * (dim // 2),
        "max_shift_equivariance_error": max_shift_err,
        "max_cosine_block_error": max_c_err,
        "max_sine_block_error": max_s_err,
        "passed": passed,
    }


def relative_dot_product_study(trials=100_000, dim=64, max_seq_len=4096, seed=42):
    """Verifies relative attention dot product error <= 1.0e-6 across 10^5 pairs."""
    rng = np.random.default_rng(seed)
    c_table, s_table = build_cayley_rotary_matrix_fp64(dim, max_seq_len + 1)
    num_pairs = dim // 2

    m_idx = rng.integers(0, max_seq_len, size=trials)
    n_idx = rng.integers(0, max_seq_len, size=trials)

    q = rng.standard_normal(size=(trials, dim), dtype=np.float64)
    k = rng.standard_normal(size=(trials, dim), dtype=np.float64)

    # Reshape to 2D pairs
    qp = q.reshape(trials, num_pairs, 2)
    kp = k.reshape(trials, num_pairs, 2)

    # 1. Direct rotation at m and n
    c_m = c_table[m_idx]
    s_m = s_table[m_idx]
    c_n = c_table[n_idx]
    s_n = s_table[n_idx]

    q_rot_0 = c_m * qp[..., 0] - s_m * qp[..., 1]
    q_rot_1 = s_m * qp[..., 0] + c_m * qp[..., 1]
    q_rot = np.stack([q_rot_0, q_rot_1], axis=-1)

    k_rot_0 = c_n * kp[..., 0] - s_n * kp[..., 1]
    k_rot_1 = s_n * kp[..., 0] + c_n * kp[..., 1]
    k_rot = np.stack([k_rot_0, k_rot_1], axis=-1)

    dot_direct = np.sum(q_rot * k_rot, axis=(-2, -1))

    # 2. Relative shift rotation: R_{n-m} k
    diff = n_idx - m_idx
    abs_diff = np.abs(diff)
    sgn = np.sign(diff)

    c_rel = c_table[abs_diff]
    s_rel = sgn[:, None] * s_table[abs_diff]

    k_rel_0 = c_rel * kp[..., 0] - s_rel * kp[..., 1]
    k_rel_1 = s_rel * kp[..., 0] + c_rel * kp[..., 1]
    k_rel = np.stack([k_rel_0, k_rel_1], axis=-1)

    dot_rel = np.sum(qp * k_rel, axis=(-2, -1))

    dot_err = np.abs(dot_direct - dot_rel)
    max_dot_err = float(np.max(dot_err))
    passed = max_dot_err <= 1.0e-6
    return {
        "trials": trials,
        "max_seq_len": max_seq_len,
        "dim": dim,
        "max_relative_dot_product_error": max_dot_err,
        "dot_error_stats": summary(dot_err),
        "passed": passed,
    }


def norm_conservation_study(max_seq_len=8192, dim=64, seed=42):
    """Verifies cumulative norm conservation drift | ||R^m v||_2 - ||v||_2 | <= 1.0e-6."""
    rng = np.random.default_rng(seed)
    c_table, s_table = build_cayley_rotary_matrix_fp64(dim, max_seq_len + 1)
    num_pairs = dim // 2

    # Unit vector v in R^dim
    v = rng.standard_normal(size=(dim,), dtype=np.float64)
    v = v / np.linalg.norm(v)
    vp = v.reshape(num_pairs, 2)

    # Apply rotation across all m in [1, max_seq_len]
    c_m = c_table[1:] # (max_seq_len, num_pairs)
    s_m = s_table[1:]

    v0 = vp[:, 0]
    v1 = vp[:, 1]

    v_rot_0 = c_m * v0 - s_m * v1
    v_rot_1 = s_m * v0 + c_m * v1

    norms = np.sqrt(np.sum(v_rot_0 ** 2 + v_rot_1 ** 2, axis=-1))
    drift = np.abs(norms - 1.0)
    max_drift = float(np.max(drift))

    passed = max_drift <= 1.0e-6
    return {
        "max_seq_len": max_seq_len,
        "dim": dim,
        "max_norm_conservation_drift": max_drift,
        "drift_stats": summary(drift),
        "passed": passed,
    }


def associative_recall_study(seed=42):
    """Trains on L=256, tests on L=1024 and L=2048. Verifies retrieval accuracy >= 95.0%."""
    rng = np.random.default_rng(seed)
    num_keys = 8
    num_vals = 8
    d_model = 64
    num_heads = 2
    head_dim = 32
    num_pairs = 4

    # Build AGO rotary parameters up to 2048 with calibrated base for length generalization
    rot_ago = build_cayley_rotary_matrix(head_dim, 2048, base=100000.0, dtype=jnp.float32)

    def init_params(key):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        scale = 0.2
        return {
            "emb_k": jax.random.normal(k1, (num_keys + 1, d_model)) * scale,
            "emb_v": jax.random.normal(k2, (num_vals + 1, d_model)) * scale,
            "wq": jax.random.normal(k3, (d_model, d_model)) * scale,
            "wk": jax.random.normal(k4, (d_model, d_model)) * scale,
            "wv": jax.random.normal(k5, (d_model, d_model)) * scale,
            "head": jax.random.normal(k6, (d_model, num_vals)) * scale,
        }

    def forward_model(params, k_seq, v_seq, q_key, rot):
        B, M = k_seq.shape
        ctx_emb = params["emb_k"][k_seq] + params["emb_v"][v_seq]
        q_emb = params["emb_k"][q_key][:, None, :]
        seq = jnp.concatenate([ctx_emb, q_emb], axis=1)
        L = M + 1

        q = (seq @ params["wq"]).reshape(B, L, num_heads, head_dim)
        k = (seq @ params["wk"]).reshape(B, L, num_heads, head_dim)
        v = (seq @ params["wv"]).reshape(B, L, num_heads, head_dim)

        q, k = apply_ago_rotations(q, k, rotary_params=rot)

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        q_last = q[:, :, -1:, :]
        k_ctx = k[:, :, :-1, :]
        v_ctx = v[:, :, :-1, :]

        scores = (q_last @ jnp.swapaxes(k_ctx, -1, -2)) * (1.0 / jnp.sqrt(head_dim))
        weights = jax.nn.softmax(scores, axis=-1)

        out = jnp.transpose(weights @ v_ctx, (0, 2, 1, 3)).reshape(B, d_model)
        logits = out @ params["head"]
        return logits

    def loss_fn(params, k_seq, v_seq, q_key, target, rot):
        logits = forward_model(params, k_seq, v_seq, q_key, rot)
        log_p = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(jnp.take_along_axis(log_p, target[:, None], axis=-1))
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == target)
        return loss, acc

    def generate_batch(batch_size, seq_len, rng_gen):
        M = seq_len - 1
        k_seq = np.full((batch_size, M), num_keys, dtype=np.int32)
        v_seq = np.full((batch_size, M), num_vals, dtype=np.int32)
        q_key = np.zeros(batch_size, dtype=np.int32)
        target = np.zeros(batch_size, dtype=np.int32)
        for b in range(batch_size):
            keys = rng_gen.choice(num_keys, size=num_pairs, replace=False)
            vals = rng_gen.choice(num_vals, size=num_pairs, replace=False)
            positions = rng_gen.choice(M, size=num_pairs, replace=False)
            k_seq[b, positions] = keys
            v_seq[b, positions] = vals
            idx = rng_gen.integers(0, num_pairs)
            q_key[b] = keys[idx]
            target[b] = vals[idx]
        return jnp.asarray(k_seq), jnp.asarray(v_seq), jnp.asarray(q_key), jnp.asarray(target)

    # Train loop
    params = init_params(jax.random.PRNGKey(seed))
    m_opt = jax.tree_util.tree_map(jnp.zeros_like, params)
    v_opt = jax.tree_util.tree_map(jnp.zeros_like, params)

    @jax.jit
    def train_step(params, m_opt, v_opt, k_seq, v_seq, q_key, target, step, lr):
        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, k_seq, v_seq, q_key, target, rot_ago
        )
        beta1, beta2, eps = 0.9, 0.999, 1.0e-8
        m_opt = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1.0 - beta1) * g, m_opt, grads)
        v_opt = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), v_opt, grads)
        m_hat = jax.tree_util.tree_map(lambda m: m / (1.0 - beta1 ** step), m_opt)
        v_hat = jax.tree_util.tree_map(lambda v: v / (1.0 - beta2 ** step), v_opt)
        params = jax.tree_util.tree_map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat
        )
        return params, m_opt, v_opt, loss, acc

    best_params = params
    best_loss = 1.0e9

    for step in range(1, 151):
        lr = 0.02 * (0.985 ** step)
        k_s, v_s, q_k, tgt = generate_batch(64, 256, rng)
        params, m_opt, v_opt, loss, acc = train_step(
            params, m_opt, v_opt, k_s, v_s, q_k, tgt, step, lr
        )
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_params = params

    # Evaluate across contexts
    results = {}
    for test_L in (256, 1024, 2048):
        accs = []
        for _ in range(25):
            k_s, v_s, q_k, tgt = generate_batch(64, test_L, rng)
            _, acc = loss_fn(best_params, k_s, v_s, q_k, tgt, rot_ago)
            accs.append(float(acc))
        results[f"acc_L{test_L}"] = float(np.mean(accs))

    passed = (results["acc_L1024"] >= 0.95) and (results["acc_L2048"] >= 0.95)
    return {
        "seed": seed,
        "train_length": 256,
        "eval_lengths": [256, 1024, 2048],
        "results": results,
        "passed": passed,
    }


def benchmark_ago_vs_rope(seq_len=2048, batch_size=16, num_heads=4, head_dim=32, repetitions=100):
    """Direct head-to-head microbenchmark of AGO vs RoPE latency and throughput."""
    rot_ago = build_cayley_rotary_matrix(head_dim, seq_len)

    # Standard RoPE frequencies
    k_idx = np.arange(head_dim // 2, dtype=np.float32)
    theta = 10000.0 ** (-2.0 * k_idx / head_dim)
    m_pos = np.arange(seq_len, dtype=np.float32)[:, None]
    angles = m_pos * theta[None, :]
    cos_rope = jnp.asarray(np.cos(angles))
    sin_rope = jnp.asarray(np.sin(angles))

    @jax.jit
    def apply_rope_jit(q, k):
        qp = q.reshape(q.shape[:-1] + (head_dim // 2, 2))
        kp = k.reshape(k.shape[:-1] + (head_dim // 2, 2))
        c = cos_rope[: q.shape[1], None, :]
        s = sin_rope[: q.shape[1], None, :]
        q0 = c * qp[..., 0] - s * qp[..., 1]
        q1 = s * qp[..., 0] + c * qp[..., 1]
        k0 = c * kp[..., 0] - s * kp[..., 1]
        k1 = s * kp[..., 0] + c * kp[..., 1]
        qr = jnp.stack([q0, q1], axis=-1).reshape(q.shape)
        kr = jnp.stack([k0, k1], axis=-1).reshape(k.shape)
        return qr, kr

    @jax.jit
    def apply_ago_jit(q, k):
        return apply_ago_rotations(q, k, rotary_params=rot_ago)

    rng = jax.random.PRNGKey(42)
    q = jax.random.normal(rng, (batch_size, seq_len, num_heads, head_dim))
    k = jax.random.normal(rng, (batch_size, seq_len, num_heads, head_dim))

    # Warmup
    _ = apply_ago_jit(q, k)[0].block_until_ready()
    _ = apply_rope_jit(q, k)[0].block_until_ready()

    # Time AGO
    t0 = time.perf_counter()
    for _ in range(repetitions):
        apply_ago_jit(q, k)[0].block_until_ready()
    t_ago = (time.perf_counter() - t0) / repetitions

    # Time RoPE
    t0 = time.perf_counter()
    for _ in range(repetitions):
        apply_rope_jit(q, k)[0].block_until_ready()
    t_rope = (time.perf_counter() - t0) / repetitions

    throughput_ratio = float(t_rope / t_ago) if t_ago > 0 else 1.0
    passed = throughput_ratio >= 0.90
    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "repetitions": repetitions,
        "ago_latency_ms": t_ago * 1000.0,
        "rope_latency_ms": t_rope * 1000.0,
        "throughput_ratio": throughput_ratio,
        "gate_threshold": 0.90,
        "passed": passed,
    }
