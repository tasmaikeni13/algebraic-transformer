#!/usr/bin/env python3
"""Download and tokenize a complete FineWeb-Edu corpus with 32k BPE.

The 7B experiment uses --train-tokens 7000000000.
Data is published only after every quota is filled. Validation and training
use separate documents; source shards come from an immutable Hub revision.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow.parquet as pq
import requests

from algebraic.tokenizer import get_tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATASET = "HuggingFaceFW/fineweb-edu"
EOT = 2


def train_filename(tokens):
    label = {2_500_000_000: "2_5B", 7_000_000_000: "7B", 20_000_000_000: "20B"}.get(tokens, str(tokens))
    return f"fineweb_train_{label}.npy"


def resolve_shards(revision="main", dataset_path="sample/10BT"):
    response = requests.get(f"https://huggingface.co/api/datasets/{DATASET}/revision/{revision}", timeout=60)
    response.raise_for_status()
    pinned = response.json()["sha"]
    url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{pinned}/{dataset_path}"
    shards = []
    while url:
        response = requests.get(url, params={"recursive": "true", "limit": 1000}, timeout=60)
        response.raise_for_status()
        shards.extend(row["path"] for row in response.json()
                      if row["type"] == "file" and row["path"].endswith(".parquet"))
        url = response.links.get("next", {}).get("url")
    if not shards:
        raise ValueError(f"no parquet shards in {DATASET}/{dataset_path}@{pinned}")
    return pinned, sorted(shards)


def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        stream.write(chunk)
            temporary.replace(destination)
            return
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sweep_filename(tokens):
    label = {600_000_000: "600M", 2_000_000_000: "2B", 5_000_000_000: "5B"}.get(tokens, str(tokens))
    return f"fineweb_sweep_{label}.npy"


def build_fineweb_dataset(output_dir, target_train_tokens=2_500_000_000,
                          target_sweep_tokens=0, target_val_tokens=10_000_000,
                          clean_raw=True, eot=True, threads=32, revision="main", dataset_path="sample/10BT"):
    output_dir = Path(output_dir)
    if (min(target_train_tokens, target_val_tokens, threads) < 1
            or not 0 <= target_sweep_tokens <= target_train_tokens):
        raise ValueError("positive train/validation quotas and threads; sweep cannot exceed training")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / train_filename(target_train_tokens)
    valid_path = output_dir / "fineweb_valid.npy"
    sweep_path = output_dir / sweep_filename(target_sweep_tokens) if target_sweep_tokens else None
    metadata_path = output_dir / "fineweb_metadata.json"
    requested = {"dataset": DATASET, "dataset_path": dataset_path, "tokenizer": "llama32k", "vocab_size": 32768,
                 "document_separator": EOT if eot else None, "train_tokens": target_train_tokens,
                 "valid_tokens": target_val_tokens, "sweep_tokens": target_sweep_tokens,
                 "train_file": train_path.name, "valid_file": valid_path.name,
                 "sweep_file": sweep_path.name if sweep_path else None}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if all(metadata.get(k) == v for k, v in requested.items()) and metadata.get("complete"):
            paths = [(train_path, target_train_tokens), (valid_path, target_val_tokens)]
            if target_sweep_tokens:
                paths.append((sweep_path, target_sweep_tokens))
            if all(path.is_file() and np.load(path, mmap_mode="r").shape == (count,)
                   and np.load(path, mmap_mode="r").dtype == np.uint16 for path, count in paths):
                if revision != "main" and revision != metadata.get("source_revision"):
                    raise ValueError("existing corpus has a different source revision; use another output directory")
                print(f"Verified complete corpus: {output_dir}", flush=True)
                return metadata
    pinned, shards = resolve_shards(revision, dataset_path)
    # Remove the completion marker before replacing any corpus file.
    metadata_path.unlink(missing_ok=True)
    raw_dir = output_dir / "fineweb_raw" / pinned
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = [(train_path, target_train_tokens), (valid_path, target_val_tokens)]
    required_bytes = sum(count * 2 for _, count in files) + target_sweep_tokens * 2
    if shutil.disk_usage(output_dir).free < required_bytes:
        raise OSError(f"need at least {required_bytes/1e9:.1f} GB free for token arrays plus parquet download space")
    temporary = [path.with_suffix(".npy.incomplete") for path, _ in files]
    arrays = [np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(count,))
              for path, (_, count) in zip(temporary, files)]
    train, valid = arrays
    counts = {"train": 0, "valid": 0}
    enc = get_tokenizer()
    sources = []
    started = time.monotonic()
    for source in shards:
        if counts["train"] == target_train_tokens and counts["valid"] == target_val_tokens:
            break
        local = raw_dir / Path(source).name
        if not local.is_file():
            download_file(f"https://huggingface.co/datasets/{DATASET}/resolve/{pinned}/{source}", local)
        sources.append({"path": source, "sha256": sha256_file(local)})
        for batch in pq.ParquetFile(local).iter_batches(batch_size=2048, columns=["text"]):
            texts = batch.column(0).to_pylist()
            for doc in enc.encode_ordinary_batch(texts, num_threads=threads):
                if not doc:
                    continue
                tokens = np.asarray(doc + ([EOT] if eot else []), dtype=np.uint16)
                # A validation document is never also used for training.
                if counts["valid"] < target_val_tokens:
                    take = min(len(tokens), target_val_tokens - counts["valid"])
                    valid[counts["valid"]:counts["valid"]+take] = tokens[:take]
                    counts["valid"] += take
                else:
                    take = min(len(tokens), target_train_tokens - counts["train"])
                    train[counts["train"]:counts["train"]+take] = tokens[:take]
                    counts["train"] += take
                if counts["train"] == target_train_tokens and counts["valid"] == target_val_tokens:
                    break
            if counts["train"] == target_train_tokens and counts["valid"] == target_val_tokens:
                break
        print(f"{source}: train {counts['train']:,}/{target_train_tokens:,}; valid {counts['valid']:,}/{target_val_tokens:,}", flush=True)
        if clean_raw:
            local.unlink()
    if clean_raw:
        shutil.rmtree(output_dir / "fineweb_raw", ignore_errors=True)
    for array in arrays:
        array.flush()
    del train, valid, arrays, array
    if counts != {"train": target_train_tokens, "valid": target_val_tokens}:
        raise RuntimeError(f"source exhausted before quotas: {counts}; incomplete arrays were not published")
    for pending, (path, _) in zip(temporary, files):
        pending.replace(path)
    if target_sweep_tokens:
        train = np.load(train_path, mmap_mode="r")
        pending = sweep_path.with_suffix(".npy.incomplete")
        sweep = np.lib.format.open_memmap(pending, mode="w+", dtype=np.uint16, shape=(target_sweep_tokens,))
        for start in range(0, target_sweep_tokens, 10_000_000):
            end = min(start + 10_000_000, target_sweep_tokens)
            sweep[start:end] = train[start:end]
        sweep.flush()
        del sweep, train
        pending.replace(sweep_path)
    metadata = {**requested, "complete": True, "source_revision": pinned, "source_shards": sources,
                "validation_documents_disjoint": True,
                "seconds": time.monotonic()-started,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    pending_meta = metadata_path.with_suffix(".json.tmp")
    pending_meta.write_text(json.dumps(metadata, indent=2) + "\n")
    pending_meta.replace(metadata_path)
    print(f"Complete corpus: {train_path}; metadata: {metadata_path}", flush=True)
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "data")
    p.add_argument("--train-tokens", type=int, default=2_500_000_000)
    p.add_argument("--sweep-tokens", type=int, default=0)
    p.add_argument("--valid-tokens", type=int, default=10_000_000)
    p.add_argument("--revision", default="main")
    p.add_argument("--dataset-path", default="sample/10BT")
    p.add_argument("--no-clean", action="store_true")
    p.add_argument("--no-eot", action="store_true")
    p.add_argument("--threads", type=int, default=32)
    a = p.parse_args()
    build_fineweb_dataset(a.output_dir, a.train_tokens, a.sweep_tokens, a.valid_tokens,
                          not a.no_clean, not a.no_eot, a.threads, a.revision, a.dataset_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
