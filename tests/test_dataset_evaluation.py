"""Bounded-memory evaluation must match full vocabulary projection."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.attention import algebraic_softmax, build_cayley_rotary_matrix
from src.baseline import BaselineConfig, StandardTransformerLM, _build_standard_rope
from src.dataset import chunked_argmax_tokens, chunked_target_log_probs
from src.model import AlgebraicTransformerLM, ModelConfig


@pytest.mark.parametrize("architecture", ["algebraic", "standard"])
def test_chunked_evaluation_matches_full_logits(architecture):
    common = dict(
        vocab_size=19,
        d_model=8,
        num_layers=0,
        num_heads=2,
        d_ff=16,
        max_seq_len=4,
        dtype=jnp.float32,
        vocab_chunk_size=5,
        remat=False,
    )
    if architecture == "algebraic":
        model = AlgebraicTransformerLM(ModelConfig(**common, eps_vocab=3.0))
        position = build_cayley_rotary_matrix(model.head_dim, 4)
        is_algebraic = True
    else:
        model = StandardTransformerLM(BaselineConfig(**common))
        position = _build_standard_rope(model.head_dim, 4)
        is_algebraic = False

    params = model.init_params(jax.random.PRNGKey(17))
    tokens = jnp.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
    targets = jnp.asarray([[2, 3, 4, 5], [6, 7, 8, 9]], dtype=jnp.int32)
    if is_algebraic:
        logits = model.forward(params, tokens, rotary_params=position)
        full_log_probs = jnp.log(
            jnp.maximum(
                algebraic_softmax(
                    logits, sink_omega=0.0, eps=model.config.eps_vocab
                ),
                1e-30,
            )
        )
    else:
        cosine, sine = position
        logits = model.forward(
            params, tokens, cos_angles=cosine, sin_angles=sine
        )
        full_log_probs = jax.nn.log_softmax(logits, axis=-1)

    expected = jnp.take_along_axis(full_log_probs, targets[..., None], axis=-1)[..., 0]
    actual = chunked_target_log_probs(
        model,
        params,
        tokens,
        targets,
        is_algebraic=is_algebraic,
        rotary_or_angles=position,
        chunk_size=5,
    )
    predicted = chunked_argmax_tokens(
        model,
        params,
        tokens,
        is_algebraic=is_algebraic,
        rotary_or_angles=position,
        chunk_size=5,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(predicted, jnp.argmax(logits, axis=-1))
