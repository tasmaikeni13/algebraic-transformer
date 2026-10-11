#!/usr/bin/env python3
"""Tokenize codelion/fineweb-edu-1B into GPT-2 token arrays.

The dataset is a reservoir sample of HuggingFaceFW/fineweb-edu, so its row
order is already random. Every document is followed by the GPT-2
end-of-text token (50256). The split is made at document level: documents
are taken from the end of the last shard until the validation array holds
exactly ``--valid-tokens`` tokens (the final document is truncated), and
every remaining document goes to training.

Outputs (uint16 .npy):
- fineweb1b_train.npy  all remaining tokens
- fineweb1b_valid.npy  exactly --valid-tokens held-out tokens
- fineweb1b_metadata.json
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import tiktoken

EOT = 50256
SHARDS = [f"train-{i:05d}-of-00010.parquet" for i in range(10)]
HF_URL = "https://huggingface.co/datasets/codelion/fineweb-edu-1B/resolve/main/data/{}"


def tokenize_shard(path: Path, enc, threads: int) -> list:
    table = pq.read_table(str(path), columns=["text"])
    texts = table["text"].to_pylist()
    docs = enc.encode_ordinary_batch(texts, num_threads=threads)
    return [np.asarray(doc + [EOT], dtype=np.uint16) for doc in docs if doc]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="directory holding the 10 parquet shards")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--valid-tokens", type=int, default=10_000_000)
    parser.add_argument("--threads", type=int, default=128)
    args = parser.parse_args()

    missing = [name for name in SHARDS if not (args.raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing shards {missing}; download them from {HF_URL.format('<name>')}"
        )
    enc = tiktoken.get_encoding("gpt2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    shard_docs = []
    for name in SHARDS:
        docs = tokenize_shard(args.raw_dir / name, enc, args.threads)
        count = sum(len(d) for d in docs)
        print(f"{name}: {len(docs):,} documents, {count:,} tokens "
              f"({time.time() - t0:.0f}s)", flush=True)
        shard_docs.append(docs)

    # Validation: whole documents from the end of the last shard.
    last = shard_docs[-1]
    valid_parts, taken = [], 0
    split_index = len(last)
    while taken < args.valid_tokens:
        split_index -= 1
        doc = last[split_index]
        need = args.valid_tokens - taken
        valid_parts.append(doc[:need])
        taken += min(len(doc), need)
    valid = np.concatenate(valid_parts[::-1])
    assert len(valid) == args.valid_tokens
    shard_docs[-1] = last[:split_index]

    train_count = sum(len(d) for docs in shard_docs for d in docs)
    train_path = args.output_dir / "fineweb1b_train.npy"
    valid_path = args.output_dir / "fineweb1b_valid.npy"
    train = np.lib.format.open_memmap(train_path, mode="w+", dtype=np.uint16,
                                      shape=(train_count,))
    offset = 0
    for docs in shard_docs:
        for doc in docs:
            train[offset:offset + len(doc)] = doc
            offset += len(doc)
    train.flush()
    del train
    np.save(valid_path, valid)

    def digest(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 24), b""):
                h.update(block)
        return h.hexdigest()

    meta = {
        "dataset": "codelion/fineweb-edu-1B",
        "tokenizer": "gpt2 (tiktoken)",
        "document_separator": EOT,
        "train_tokens": int(train_count),
        "valid_tokens": int(args.valid_tokens),
        "valid_documents_from_end_of_last_shard": len(valid_parts),
        "train_sha256": digest(train_path),
        "valid_sha256": digest(valid_path),
        "seconds": round(time.time() - t0, 1),
    }
    (args.output_dir / "fineweb1b_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
