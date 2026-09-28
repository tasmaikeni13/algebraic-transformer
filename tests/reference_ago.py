"""Independent NumPy float64 reference oracle for AGO and baseline trigonometric RoPE."""
import numpy as np


def cayley_rotation_matrix_fp64(w):
    """Computes exact 2x2 rational Cayley rotation matrix in float64.

    R(w) = 1/(1 + w^2) * [[1 - w^2, -2w], [2w, 1 - w^2]]
    """
    w = np.asarray(w, dtype=np.float64)
    denom = 1.0 + w * w
    c = (1.0 - w * w) / denom
    s = (2.0 * w) / denom
    if w.ndim == 0:
        return np.array([[c, -s], [s, c]], dtype=np.float64)
    row0 = np.stack([c, -s], axis=-1)
    row1 = np.stack([s, c], axis=-1)
    return np.stack([row0, row1], axis=-2)


def build_cayley_rotary_matrix_fp64(dim, max_seq_len, freqs=None, base=10000.0):
    """Computes reference (C, S) rotary tables across sequence context in float64."""
    if dim <= 0 or dim % 2 != 0:
        raise ValueError("dim must be a positive even integer")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer")

    num_pairs = dim // 2
    if freqs is not None:
        w = np.asarray(freqs, dtype=np.float64)
    else:
        k = np.arange(num_pairs, dtype=np.float64)
        alpha = (np.sqrt(base) - 1.0) / max(float(num_pairs - 1), 1.0)
        lin = 1.0 + alpha * k
        w = 1.0 / (lin * lin)

    w_sq = w * w
    c_base = (1.0 - w_sq) / (1.0 + w_sq)
    s_base = (2.0 * w) / (1.0 + w_sq)

    c_table = [np.ones(num_pairs, dtype=np.float64)]
    s_table = [np.zeros(num_pairs, dtype=np.float64)]

    for _ in range(max_seq_len - 1):
        c_prev = c_table[-1]
        s_prev = s_table[-1]
        c_next = c_base * c_prev - s_base * s_prev
        s_next = s_base * c_prev + c_base * s_prev
        r = 1.0 / np.sqrt(c_next * c_next + s_next * s_next)
        c_table.append(c_next * r)
        s_table.append(s_next * r)

    return np.array(c_table, dtype=np.float64), np.array(s_table, dtype=np.float64)


def apply_ago_rotations_fp64(q, k, rotary_params=None, seq_axis=None):
    """Applies AGO Cayley rotation to query and key arrays in float64."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    if q.shape != k.shape:
        raise ValueError("q and k shapes must match")
    dim = q.shape[-1]
    num_pairs = dim // 2

    ndim = q.ndim
    if seq_axis is None:
        seq_axis = 0 if ndim == 2 else 1 if ndim == 3 else (1 if q.shape[1] > q.shape[2] else 2)
    seq_len = q.shape[seq_axis]

    if rotary_params is None:
        c, s = build_cayley_rotary_matrix_fp64(dim, seq_len)
    else:
        c, s = np.asarray(rotary_params[0], dtype=np.float64), np.asarray(rotary_params[1], dtype=np.float64)

    c_sliced = c[:seq_len]
    s_sliced = s[:seq_len]

    bcast_shape = [1] * (ndim - 1) + [num_pairs]
    bcast_shape[seq_axis] = seq_len
    c_b = c_sliced.reshape(bcast_shape)
    s_b = s_sliced.reshape(bcast_shape)

    qp = q.reshape(q.shape[:-1] + (num_pairs, 2))
    kp = k.reshape(k.shape[:-1] + (num_pairs, 2))

    q0_rot = c_b * qp[..., 0] - s_b * qp[..., 1]
    q1_rot = s_b * qp[..., 0] + c_b * qp[..., 1]

    k0_rot = c_b * kp[..., 0] - s_b * kp[..., 1]
    k1_rot = s_b * kp[..., 0] + c_b * kp[..., 1]

    q_rot = np.stack([q0_rot, q1_rot], axis=-1).reshape(q.shape)
    k_rot = np.stack([k0_rot, k1_rot], axis=-1).reshape(k.shape)
    return q_rot, k_rot


def apply_rope_rotations_fp64(q, k, base=10000.0, seq_axis=None):
    """Standard trigonometric RoPE baseline using sin and cos."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    dim = q.shape[-1]
    num_pairs = dim // 2

    ndim = q.ndim
    if seq_axis is None:
        seq_axis = 0 if ndim == 2 else 1 if ndim == 3 else (1 if q.shape[1] > q.shape[2] else 2)
    seq_len = q.shape[seq_axis]

    freqs = base ** (-2.0 * np.arange(num_pairs, dtype=np.float64) / dim)
    m = np.arange(seq_len, dtype=np.float64)
    angles = np.outer(m, freqs) # (seq_len, num_pairs)

    cos = np.cos(angles)
    sin = np.sin(angles)

    bcast_shape = [1] * (ndim - 1) + [num_pairs]
    bcast_shape[seq_axis] = seq_len
    cos_b = cos.reshape(bcast_shape)
    sin_b = sin.reshape(bcast_shape)

    qp = q.reshape(q.shape[:-1] + (num_pairs, 2))
    kp = k.reshape(k.shape[:-1] + (num_pairs, 2))

    q0_rot = cos_b * qp[..., 0] - sin_b * qp[..., 1]
    q1_rot = sin_b * qp[..., 0] + cos_b * qp[..., 1]

    k0_rot = cos_b * kp[..., 0] - sin_b * kp[..., 1]
    k1_rot = sin_b * kp[..., 0] + cos_b * kp[..., 1]

    q_rot = np.stack([q0_rot, q1_rot], axis=-1).reshape(q.shape)
    k_rot = np.stack([k0_rot, k1_rot], axis=-1).reshape(k.shape)
    return q_rot, k_rot
