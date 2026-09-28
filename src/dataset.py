"""WikiText-103 loading, GPT-2 tokenization, and distributed batch sharding.

Tokenized data is stored in memory-mapped arrays so training workers can read
batch slices without loading the whole corpus into RAM."""

from pathlib import Path
import urllib.request
import zipfile
import math
from typing import Any, Generator, Optional, Tuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import tiktoken

WIKITEXT_URL = "https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/resolve/main/wikitext-103-raw-v1.zip?download=true"
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "data"


def ensure_wikitext103_ready(data_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Ensures tokenized WikiText-103 binary arrays exist, downloading if necessary.

    Returns:
        Tuple of (train_npy_path, valid_npy_path).
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_npy = data_dir / "train.npy"
    valid_npy = data_dir / "valid.npy"

    if train_npy.exists() and valid_npy.exists():
        return train_npy, valid_npy

    raw_dir = data_dir / "wikitext-103-raw"
    train_raw = raw_dir / "wiki.train.raw"
    valid_raw = raw_dir / "wiki.valid.raw"

    if not (train_raw.exists() and valid_raw.exists()):
        zip_path = data_dir / "wikitext-103-raw-v1.zip"
        if not zip_path.exists():
            print(f"Downloading WikiText-103 from {WIKITEXT_URL}...")
            urllib.request.urlretrieve(WIKITEXT_URL, zip_path)
        print("Extracting WikiText-103...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)

    enc = tiktoken.get_encoding("gpt2")
    if not valid_npy.exists():
        print("Tokenizing wiki.valid.raw...")
        valid_text = valid_raw.read_text(encoding="utf-8")
        valid_tokens = np.array(enc.encode(valid_text), dtype=np.uint16)
        np.save(valid_npy, valid_tokens)

    if not train_npy.exists():
        print("Tokenizing wiki.train.raw...")
        train_text = train_raw.read_text(encoding="utf-8")
        train_tokens = np.array(enc.encode(train_text), dtype=np.uint16)
        np.save(train_npy, train_tokens)

    return train_npy, valid_npy


def ensure_fineweb_ready(data_dir: Optional[Path] = None) -> Tuple[Path, Path, Path]:
    """Ensures tokenized FineWeb-Edu binary arrays exist.

    Returns:
        Tuple of (sweep_npy_path, valid_npy_path, train_2_5b_npy_path).
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)
    sweep_path = data_dir / "fineweb_sweep_600M.npy"
    valid_path = data_dir / "fineweb_valid.npy"
    train_path = data_dir / "fineweb_train_2_5B.npy"
    return sweep_path, valid_path, train_path


class ShardedTokenLoader:
    """SPMD data-parallel sequence batch generator for distributed pretraining."""

    def __init__(
        self,
        token_path: Path,
        batch_size: int = 64,
        seq_len: int = 512,
        process_index: int = 0,
        process_count: int = 1,
        seed: int = 42,
    ):
        self.tokens = np.load(token_path, mmap_mode="r")
        self.total_tokens = len(self.tokens)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.process_index = process_index
        self.process_count = process_count
        self.seed = seed

        assert batch_size % process_count == 0, (
            f"batch_size ({batch_size}) must be divisible by process_count ({process_count})"
        )
        self.local_batch_size = batch_size // process_count
        self.span = seq_len + 1  # tokens + target shifted by 1
        self.num_possible_sequences = (self.total_tokens - 1) // seq_len
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.num_possible_sequences // self.batch_size

    def get_batch(self, step: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generates process-local slice of the global batch for the given step.

        Returns:
            x: input token ids of shape (local_batch_size, seq_len)
            y: target token ids of shape (local_batch_size, seq_len)
        """
        # Deterministic sequence offset calculation across all hosts
        step_rng = np.random.default_rng(10007 * (step + 1) + self.seed)
        global_indices = step_rng.integers(
            0, self.total_tokens - self.span, size=self.batch_size
        )

        # Slice the process-local chunk
        start_idx = self.process_index * self.local_batch_size
        end_idx = start_idx + self.local_batch_size
        local_indices = global_indices[start_idx:end_idx]

        x_batch = np.empty((self.local_batch_size, self.seq_len), dtype=np.int32)
        y_batch = np.empty((self.local_batch_size, self.seq_len), dtype=np.int32)

        for i, idx in enumerate(local_indices):
            chunk = self.tokens[idx : idx + self.span].astype(np.int32)
            x_batch[i] = chunk[:self.seq_len]
            y_batch[i] = chunk[1 : self.span]

        return x_batch, y_batch


def _output_weight(model, params):
    if model.config.tie_embeddings:
        return params["token_embed"].T
    return params["output_head"]


def chunked_target_log_probs(
    model,
    params,
    tokens: jax.Array,
    targets: jax.Array,
    *,
    is_algebraic: bool,
    rotary_or_angles: Optional[Any] = None,
    chunk_size: Optional[int] = None,
) -> jax.Array:
    """Return target log-probabilities without materializing ``(..., vocab)``.

    The algebraic branch exactly matches ``algebraic_softmax(..., sink=0)``:
    it first AVN-normalizes the vocabulary logits and then normalizes the octic
    kernel.  The standard branch performs an online log-sum-exp reduction.
    """
    if tokens.shape != targets.shape:
        raise ValueError("tokens and targets must have identical shapes")
    if is_algebraic:
        hidden = model._hidden(params, tokens, rotary_params=rotary_or_angles)
    else:
        if rotary_or_angles is None:
            cosine = sine = None
        else:
            cosine, sine = rotary_or_angles
        hidden = model._hidden(params, tokens, cos_angles=cosine, sin_angles=sine)

    weight = _output_weight(model, params)
    vocab_size = weight.shape[-1]
    width = chunk_size or int(getattr(model.config, "vocab_chunk_size", 4096))
    if width <= 0:
        raise ValueError("chunk_size must be positive")
    calc_dtype = jnp.float64 if hidden.dtype == jnp.float64 else jnp.float32
    flat_hidden = hidden.reshape(-1, hidden.shape[-1]).astype(calc_dtype)
    flat_targets = targets.reshape(-1)
    rows = flat_hidden.shape[0]

    if is_algebraic:
        sum_squares = jnp.zeros((rows,), dtype=calc_dtype)
        for start in range(0, vocab_size, width):
            stop = min(start + width, vocab_size)
            logits = flat_hidden @ weight[:, start:stop].astype(calc_dtype)
            sum_squares += jnp.sum(logits * logits, axis=-1)
        eps = float(getattr(model.config, "eps_vocab", 100.0))
        tau = lax.rsqrt(sum_squares / float(vocab_size) + eps)

        partition = jnp.zeros((rows,), dtype=calc_dtype)
        target_weight = jnp.zeros((rows,), dtype=calc_dtype)
        for start in range(0, vocab_size, width):
            stop = min(start + width, vocab_size)
            normalized = (flat_hidden @ weight[:, start:stop].astype(calc_dtype)) * tau[:, None]
            rad = 1.0 + normalized * normalized
            inv_root = lax.rsqrt(rad)
            unit = normalized * inv_root
            rho = jnp.where(
                normalized < 0,
                inv_root / (1.0 - unit),
                normalized + rad * inv_root,
            )
            rho2 = rho * rho
            rho4 = rho2 * rho2
            kernel = rho4 * rho4
            partition += jnp.sum(kernel, axis=-1)
            in_chunk = (flat_targets >= start) & (flat_targets < stop)
            local_index = jnp.clip(flat_targets - start, 0, stop - start - 1)
            selected = jnp.take_along_axis(kernel, local_index[:, None], axis=-1)[:, 0]
            target_weight = jnp.where(in_chunk, selected, target_weight)
        result = jnp.log(jnp.maximum(target_weight, 1e-30)) - jnp.log(
            jnp.maximum(partition, 1e-30)
        )
    else:
        running_max = jnp.full((rows,), -jnp.inf, dtype=calc_dtype)
        running_sum = jnp.zeros((rows,), dtype=calc_dtype)
        target_logit = jnp.zeros((rows,), dtype=calc_dtype)
        for start in range(0, vocab_size, width):
            stop = min(start + width, vocab_size)
            logits = flat_hidden @ weight[:, start:stop].astype(calc_dtype)
            block_max = jnp.max(logits, axis=-1)
            new_max = jnp.maximum(running_max, block_max)
            running_sum = (
                running_sum * jnp.exp(running_max - new_max)
                + jnp.sum(jnp.exp(logits - new_max[:, None]), axis=-1)
            )
            running_max = new_max
            in_chunk = (flat_targets >= start) & (flat_targets < stop)
            local_index = jnp.clip(flat_targets - start, 0, stop - start - 1)
            selected = jnp.take_along_axis(logits, local_index[:, None], axis=-1)[:, 0]
            target_logit = jnp.where(in_chunk, selected, target_logit)
        result = target_logit - running_max - jnp.log(jnp.maximum(running_sum, 1e-30))
    return result.reshape(targets.shape)


def chunked_argmax_tokens(
    model,
    params,
    tokens: jax.Array,
    *,
    is_algebraic: bool,
    rotary_or_angles: Optional[Any] = None,
    chunk_size: Optional[int] = None,
) -> jax.Array:
    """Return vocabulary argmax indices using bounded vocabulary tiles."""
    if is_algebraic:
        hidden = model._hidden(params, tokens, rotary_params=rotary_or_angles)
    else:
        if rotary_or_angles is None:
            cosine = sine = None
        else:
            cosine, sine = rotary_or_angles
        hidden = model._hidden(params, tokens, cos_angles=cosine, sin_angles=sine)
    weight = _output_weight(model, params)
    vocab_size = weight.shape[-1]
    width = chunk_size or int(getattr(model.config, "vocab_chunk_size", 4096))
    if width <= 0:
        raise ValueError("chunk_size must be positive")
    calc_dtype = jnp.float64 if hidden.dtype == jnp.float64 else jnp.float32
    flat_hidden = hidden.reshape(-1, hidden.shape[-1]).astype(calc_dtype)
    best_value = jnp.full((flat_hidden.shape[0],), -jnp.inf, dtype=calc_dtype)
    best_index = jnp.zeros((flat_hidden.shape[0],), dtype=jnp.int32)
    for start in range(0, vocab_size, width):
        stop = min(start + width, vocab_size)
        logits = flat_hidden @ weight[:, start:stop].astype(calc_dtype)
        local_index = jnp.argmax(logits, axis=-1)
        local_value = jnp.take_along_axis(logits, local_index[:, None], axis=-1)[:, 0]
        replace = local_value > best_value
        best_value = jnp.where(replace, local_value, best_value)
        best_index = jnp.where(replace, local_index.astype(jnp.int32) + start, best_index)
    return best_index.reshape(tokens.shape)


def evaluate_perplexity_and_nll_tokens(
    model,
    params,
    tokens: np.ndarray,
    seq_len: int = 512,
    batch_size: int = 32,
    max_eval_batches: int = 20,
    is_algebraic: bool = True,
    rotary_or_angles: Optional[Any] = None,
) -> Tuple[float, float]:
    """Return perplexity and mean NLL with a bounded vocabulary working set."""
    total_tokens = len(tokens)
    span = seq_len + 1
    total_nll = 0.0
    total_count = 0
    num_sequences = (total_tokens - 1) // seq_len
    num_batches = min(max_eval_batches, num_sequences // batch_size)
    if num_batches <= 0:
        raise ValueError("validation data does not contain one complete batch")

    def batch_nll(current_params, x, y):
        log_probs = chunked_target_log_probs(
            model,
            current_params,
            x,
            y,
            is_algebraic=is_algebraic,
            rotary_or_angles=rotary_or_angles,
        )
        return -jnp.sum(log_probs, dtype=jnp.float32)

    compiled_nll = jax.jit(batch_nll)

    for b in range(num_batches):
        batch_starts = [i * seq_len for i in range(b * batch_size, (b + 1) * batch_size)]
        x = np.stack([tokens[s : s + seq_len].astype(np.int32) for s in batch_starts])
        y = np.stack([tokens[s + 1 : s + span].astype(np.int32) for s in batch_starts])
        total_nll += float(jax.device_get(compiled_nll(params, x, y)))
        total_count += x.size

    avg_nll = total_nll / total_count
    return float(math.exp(min(avg_nll, 20.0))), float(avg_nll)


def evaluate_perplexity_tokens(
    model,
    params,
    tokens: np.ndarray,
    seq_len: int = 512,
    batch_size: int = 32,
    max_eval_batches: int = 20,
    is_algebraic: bool = True,
    rotary_or_angles: Optional[Any] = None,
) -> float:
    """Evaluate held-out perplexity with a bounded vocabulary working set."""
    perplexity, _ = evaluate_perplexity_and_nll_tokens(
        model,
        params,
        tokens,
        seq_len=seq_len,
        batch_size=batch_size,
        max_eval_batches=max_eval_batches,
        is_algebraic=is_algebraic,
        rotary_or_angles=rotary_or_angles,
    )
    return perplexity


def evaluate_perplexity(
    model,
    params,
    valid_tokens_path: Path,
    seq_len: int = 512,
    batch_size: int = 32,
    max_eval_batches: int = 20,
    is_algebraic: bool = True,
    rotary_or_angles: Optional[Any] = None,
) -> float:
    """Evaluate held-out validation perplexity from a memory-mapped token file."""
    tokens = np.load(valid_tokens_path, mmap_mode="r")
    return evaluate_perplexity_tokens(
        model,
        params,
        tokens,
        seq_len=seq_len,
        batch_size=batch_size,
        max_eval_batches=max_eval_batches,
        is_algebraic=is_algebraic,
        rotary_or_angles=rotary_or_angles,
    )
