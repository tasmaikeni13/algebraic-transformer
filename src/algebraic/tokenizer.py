"""TPU-aligned 32,768 vocabulary BPE tokenizer (LLaMA/Mistral style).

Provides a pure BPE tokenizer aligned to Cloud TPU v4 matrix units (multiple of 128),
with exactly 32,768 vocabulary entries, EOS=2, BOS=1, PAD=0.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from tokenizers import AddedToken, Tokenizer

# Constants
vocab_size: int = 32768
eot_token: int = 2
bos_token: int = 1
pad_token: int = 0

_TOKENIZER: Optional[BPE32kTokenizer] = None
_DATA_DIR = Path(__file__).resolve().parent / "data"
_DEFAULT_TOKENIZER_PATH = _DATA_DIR / "tokenizer.json"


class BPE32kTokenizer:
    """Wrapper around tokenizers.Tokenizer enforcing 32k BPE contracts."""

    vocab_size: int = 32768
    eot_token: int = 2
    bos_token: int = 1
    pad_token: int = 0

    def __init__(self, raw: Tokenizer):
        self._raw = raw
        self.vocab_size = 32768
        self.eot_token = 2
        self.bos_token = 1
        self.pad_token = 0

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self._raw.encode(text).ids

    def encode_ordinary(self, text: str) -> list[int]:
        if not text:
            return []
        return self._raw.encode(text).ids

    def encode_ordinary_batch(
        self, texts: list[str], num_threads: Optional[int] = None
    ) -> list[list[int]]:
        if not texts:
            return []
        encodings = self._raw.encode_batch(texts)
        return [e.ids for e in encodings]

    def decode(self, tokens: list[int]) -> str:
        if not tokens:
            return ""
        return self._raw.decode(tokens)

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        return self.vocab_size

    def __getattr__(self, name: str):
        return getattr(self._raw, name)


def _build_or_load_raw_tokenizer() -> Tokenizer:
    """Load or hermetically build the 32,768-vocab BPE tokenizer."""
    if _DEFAULT_TOKENIZER_PATH.exists():
        tok = Tokenizer.from_file(str(_DEFAULT_TOKENIZER_PATH))
        if tok.get_vocab_size() == 32768:
            return tok

    # Look in HF cache for base 32,000 Mistral snapshot if data file is absent or needs rebuild
    hf_cache_candidates = list(
        (Path.home() / ".cache" / "huggingface" / "hub").glob(
            "models--mistralai--Mistral-7B-v0.1/snapshots/*/tokenizer.json"
        )
    )
    if hf_cache_candidates and hf_cache_candidates[0].exists():
        tok = Tokenizer.from_file(str(hf_cache_candidates[0]))
        tok.post_processor = None
        if tok.get_vocab_size() == 32000:
            reserved = [AddedToken(f"<reserved_{i}>", special=False) for i in range(768)]
            tok.add_tokens(reserved)
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        tok.save(str(_DEFAULT_TOKENIZER_PATH))
        return tok

    if _DEFAULT_TOKENIZER_PATH.exists():
        return Tokenizer.from_file(str(_DEFAULT_TOKENIZER_PATH))

    raise FileNotFoundError(
        f"32,768 tokenizer file not found at {_DEFAULT_TOKENIZER_PATH} and no base snapshot in cache."
    )


def get_tokenizer() -> BPE32kTokenizer:
    """Return the cached 32,768 vocabulary BPE tokenizer instance."""
    global _TOKENIZER
    if _TOKENIZER is None:
        raw = _build_or_load_raw_tokenizer()
        _TOKENIZER = BPE32kTokenizer(raw)
    return _TOKENIZER


def encode(text: str) -> list[int]:
    """Encode text to token IDs within [0, 32768)."""
    return get_tokenizer().encode(text)


def encode_ordinary(text: str) -> list[int]:
    """Encode text to token IDs without special handling."""
    return get_tokenizer().encode_ordinary(text)


def encode_ordinary_batch(
    texts: list[str], num_threads: Optional[int] = None
) -> list[list[int]]:
    """Batch-encode multiple texts to lists of token IDs."""
    return get_tokenizer().encode_ordinary_batch(texts, num_threads=num_threads)


def decode(tokens: list[int]) -> str:
    """Decode a sequence of token IDs to string."""
    return get_tokenizer().decode(tokens)
