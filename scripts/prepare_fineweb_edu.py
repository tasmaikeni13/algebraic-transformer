#!/usr/bin/env python3
"""Pipeline to download and tokenize 2.5B tokens of real FineWeb-Edu.

Streams real FineWeb-Edu parquet shards from HuggingFaceFW/fineweb-edu,
tokenizes using tiktoken GPT-2 BPE, and emits:
- data/fineweb_train_2_5B.npy (exactly 2,500,000,000 tokens)
- data/fineweb_sweep_600M.npy (exactly 600,000,000 tokens for Tuning study sweep)
- data/fineweb_valid.npy (10,000,000 held-out validation tokens)
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Optional
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
HF_BASE_URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT"
SHARDS = [
    "000_00000.parquet",
    "001_00000.parquet",
    "002_00000.parquet",
    "003_00000.parquet",
]


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(".tmp")
    print(f"Downloading {url} to {dest}...", flush=True)
    t0 = time.time()
    
    # Try using curl for fast multi-stream download if available
    cmd = f"curl -L -f -s -S '{url}' -o '{temp_dest}'"
    ret = os.system(cmd)
    if ret != 0:
        print(f"curl failed, falling back to urllib...", flush=True)
        urllib.request.urlretrieve(url, temp_dest)
    
    temp_dest.replace(dest)
    dt = time.time() - t0
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"Downloaded {size_mb:.1f} MB in {dt:.1f}s ({size_mb/dt:.1f} MB/s)", flush=True)


def build_fineweb_dataset(
    output_dir: Path,
    target_train_tokens: int = 2_500_000_000,
    target_sweep_tokens: int = 600_000_000,
    target_val_tokens: int = 10_000_000,
    clean_raw: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "fineweb_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "fineweb_train_2_5B.npy"
    sweep_path = output_dir / "fineweb_sweep_600M.npy"
    valid_path = output_dir / "fineweb_valid.npy"

    if train_path.exists() and sweep_path.exists() and valid_path.exists():
        train_mm = np.load(train_path, mmap_mode="r")
        sweep_mm = np.load(sweep_path, mmap_mode="r")
        valid_mm = np.load(valid_path, mmap_mode="r")
        if len(train_mm) >= target_train_tokens and len(sweep_mm) >= target_sweep_tokens and len(valid_mm) >= target_val_tokens:
            print(f"FineWeb-Edu dataset already complete in {output_dir}:")
            print(f"  Train: {len(train_mm):,} tokens")
            print(f"  Sweep: {len(sweep_mm):,} tokens")
            print(f"  Valid: {len(valid_mm):,} tokens")
            return

    enc = tiktoken.get_encoding("gpt2")
    
    print(f"Pre-allocating binary memmap files...")
    # Pre-allocate valid memmap
    valid_mm = np.lib.format.open_memmap(
        valid_path, mode="w+", dtype=np.uint16, shape=(target_val_tokens,)
    )
    # Pre-allocate train memmap
    train_mm = np.lib.format.open_memmap(
        train_path, mode="w+", dtype=np.uint16, shape=(target_train_tokens,)
    )

    valid_tokens_written = 0
    train_tokens_written = 0
    t_start = time.time()

    for shard_name in SHARDS:
        if train_tokens_written >= target_train_tokens and valid_tokens_written >= target_val_tokens:
            break

        shard_path = raw_dir / shard_name
        shard_url = f"{HF_BASE_URL}/{shard_name}"

        if not shard_path.exists():
            download_file(shard_url, shard_path)

        print(f"Processing shard {shard_name}...", flush=True)
        pf = pq.ParquetFile(str(shard_path))
        num_rg = pf.num_row_groups
        print(f"Shard {shard_name} has {num_rg} row groups ({pf.metadata.num_rows:,} rows)", flush=True)

        t_shard = time.time()
        shard_tokens = 0

        # Read in batches of row groups
        batch_size = 10
        for rg_idx in range(0, num_rg, batch_size):
            if train_tokens_written >= target_train_tokens and valid_tokens_written >= target_val_tokens:
                break

            rg_indices = list(range(rg_idx, min(rg_idx + batch_size, num_rg)))
            texts = []
            for rg in rg_indices:
                tbl = pf.read_row_group(rg, columns=["text"])
                texts.extend(tbl["text"].to_pylist())

            # Tokenize batch
            encoded = enc.encode_ordinary_batch(texts, num_threads=32)
            for doc_toks in encoded:
                doc_len = len(doc_toks)
                if doc_len == 0:
                    continue
                arr = np.array(doc_toks, dtype=np.uint16)

                # First satisfy validation quota
                if valid_tokens_written < target_val_tokens:
                    needed = target_val_tokens - valid_tokens_written
                    take = min(doc_len, needed)
                    valid_mm[valid_tokens_written : valid_tokens_written + take] = arr[:take]
                    valid_tokens_written += take
                    shard_tokens += take
                    if take < doc_len:
                        arr = arr[take:]
                        doc_len -= take
                    else:
                        continue

                # Then satisfy training quota
                if train_tokens_written < target_train_tokens:
                    needed = target_train_tokens - train_tokens_written
                    take = min(doc_len, needed)
                    train_mm[train_tokens_written : train_tokens_written + take] = arr[:take]
                    train_tokens_written += take
                    shard_tokens += take

                if train_tokens_written >= target_train_tokens:
                    break

            if (rg_idx // batch_size) % 5 == 0:
                elapsed = time.time() - t_start
                rate = (train_tokens_written + valid_tokens_written) / elapsed if elapsed > 0 else 0
                print(
                    f"  [{shard_name} RG {rg_idx}/{num_rg}] Written: "
                    f"train={train_tokens_written:,}/{target_train_tokens:,} "
                    f"valid={valid_tokens_written:,}/{target_val_tokens:,} "
                    f"({rate/1e6:.2f} Mtok/s)",
                    flush=True,
                )

        dt_shard = time.time() - t_shard
        print(f"Finished {shard_name}: {shard_tokens:,} tokens in {dt_shard:.1f}s ({shard_tokens/dt_shard/1e6:.2f} Mtok/s)", flush=True)

        if clean_raw and shard_path.exists():
            print(f"Removing processed shard {shard_path} to conserve disk space...", flush=True)
            shard_path.unlink()

    # Flush memmaps
    train_mm.flush()
    valid_mm.flush()
    del train_mm
    del valid_mm

    print(f"Creating {sweep_path} (taking exactly {target_sweep_tokens:,} tokens from train)...", flush=True)
    # Open read-only memmap and copy first 600M tokens to sweep_path
    train_ro = np.load(train_path, mmap_mode="r")
    sweep_mm = np.lib.format.open_memmap(
        sweep_path, mode="w+", dtype=np.uint16, shape=(target_sweep_tokens,)
    )
    chunk_size = 50_000_000
    for i in range(0, target_sweep_tokens, chunk_size):
        end_idx = min(i + chunk_size, target_sweep_tokens)
        sweep_mm[i:end_idx] = train_ro[i:end_idx]
    sweep_mm.flush()
    del sweep_mm
    del train_ro

    total_time = time.time() - t_start
    print(f"\n================ FineWeb-Edu Pipeline Complete ================")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"Train path: {train_path} ({target_train_tokens:,} tokens, {train_path.stat().st_size / 1e9:.2f} GB)")
    print(f"Sweep path: {sweep_path} ({target_sweep_tokens:,} tokens, {sweep_path.stat().st_size / 1e9:.2f} GB)")
    print(f"Valid path: {valid_path} ({target_val_tokens:,} tokens, {valid_path.stat().st_size / 1e6:.2f} MB)")

    # Verification checks
    print("\nRunning verification integrity checks...")
    train_verify = np.load(train_path, mmap_mode="r")
    sweep_verify = np.load(sweep_path, mmap_mode="r")
    valid_verify = np.load(valid_path, mmap_mode="r")

    assert len(train_verify) == target_train_tokens, f"Expected {target_train_tokens}, got {len(train_verify)}"
    assert len(sweep_verify) == target_sweep_tokens, f"Expected {target_sweep_tokens}, got {len(sweep_verify)}"
    assert len(valid_verify) == target_val_tokens, f"Expected {target_val_tokens}, got {len(valid_verify)}"

    min_t, max_t = int(train_verify[:1_000_000].min()), int(train_verify[:1_000_000].max())
    assert 0 <= min_t and max_t < 50257, f"Token range invalid: [{min_t}, {max_t}]"
    
    # Save metadata
    meta = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "tokenizer": "gpt2",
        "vocab_size": 50257,
        "train_tokens": target_train_tokens,
        "sweep_tokens": target_sweep_tokens,
        "valid_tokens": target_val_tokens,
        "train_file": str(train_path.name),
        "sweep_file": str(sweep_path.name),
        "valid_file": str(valid_path.name),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(output_dir / "fineweb_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Verification passed! Metadata saved to data/fineweb_metadata.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--train-tokens", type=int, default=2_500_000_000)
    parser.add_argument("--sweep-tokens", type=int, default=600_000_000)
    parser.add_argument("--valid-tokens", type=int, default=10_000_000)
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    build_fineweb_dataset(
        output_dir=args.output_dir,
        target_train_tokens=args.train_tokens,
        target_sweep_tokens=args.sweep_tokens,
        target_val_tokens=args.valid_tokens,
        clean_raw=not args.no_clean,
    )


if __name__ == "__main__":
    main()
