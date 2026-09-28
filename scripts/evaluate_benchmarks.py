#!/usr/bin/env python3
"""Evaluate one final Pretraining study checkpoint on ARC-Easy, HellaSwag, PIQA, and LAMBADA."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os
if "TPU_CHIPS_PER_HOST_BOUNDS" not in os.environ and "JAX_PLATFORMS" not in os.environ:
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"

import jax
import jax.numpy as jnp
import numpy as np
import tiktoken

from scripts.tuning_experiments import get_125m_algebraic_config, get_125m_baseline_config
from scripts.tuning_records import write_json
from scripts.pretraining_records import environment, source_hashes
from scripts.pretraining_experiments import latest_checkpoint
from scripts.run_hparam_sweep import load_training_checkpoint
from src.attention import build_cayley_rotary_matrix
from src.baseline import StandardTransformerLM, _build_standard_rope
from src.dataset import chunked_argmax_tokens, chunked_target_log_probs
from src.model import AlgebraicTransformerLM


def load_jsonl(path: Path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


class CheckpointScorer:
    def __init__(self, architecture: str, checkpoint: Path):
        saved = load_training_checkpoint(checkpoint)
        internal = "baseline" if architecture == "standard" else "algebraic"
        if saved.get("architecture") != internal:
            raise ValueError("checkpoint architecture does not match --architecture")
        self.architecture = architecture
        self.params = jax.device_put(saved["params"])
        self.encoding = tiktoken.get_encoding("gpt2")
        if architecture == "algebraic":
            self.model = AlgebraicTransformerLM(get_125m_algebraic_config())
            self.position = build_cayley_rotary_matrix(self.model.head_dim, 2048)
        else:
            self.model = StandardTransformerLM(get_125m_baseline_config())
            self.position = _build_standard_rope(self.model.head_dim, 2048)
        self._log_prob_functions = {}
        self._argmax_functions = {}

    @staticmethod
    def _bucket_size(length: int) -> int:
        if not 1 <= length <= 2048:
            raise ValueError(f"evaluation input length must be in [1, 2048], got {length}")
        return min(2048, max(128, 1 << (length - 1).bit_length()))

    def _padded_example(self, context: str, continuation: str):
        context_ids = self.encoding.encode_ordinary(context)
        continuation_ids = self.encoding.encode_ordinary(continuation)
        if not context_ids:
            context_ids = [self.encoding.eot_token]
        if not continuation_ids:
            raise ValueError("empty continuation")
        if len(continuation_ids) > 2048:
            raise ValueError("continuation exceeds the 2048-token evaluation context")
        context_ids = context_ids[-(2049 - len(continuation_ids)):]
        ids = context_ids + continuation_ids
        input_length = len(ids) - 1
        bucket = self._bucket_size(input_length)
        padded = ids + [self.encoding.eot_token] * (bucket + 1 - len(ids))
        inputs = jnp.asarray(padded[:-1], dtype=jnp.int32)[None, :]
        targets = jnp.asarray(padded[1:], dtype=jnp.int32)[None, :]
        return inputs, targets, len(context_ids) - 1, len(continuation_ids), bucket

    def _log_prob_function(self, bucket: int):
        if bucket not in self._log_prob_functions:
            self._log_prob_functions[bucket] = jax.jit(
                lambda params, inputs, targets: chunked_target_log_probs(
                    self.model,
                    params,
                    inputs,
                    targets,
                    is_algebraic=self.architecture == "algebraic",
                    rotary_or_angles=self.position,
                )
            )
        return self._log_prob_functions[bucket]

    def _argmax_function(self, bucket: int):
        if bucket not in self._argmax_functions:
            self._argmax_functions[bucket] = jax.jit(
                lambda params, inputs: chunked_argmax_tokens(
                    self.model,
                    params,
                    inputs,
                    is_algebraic=self.architecture == "algebraic",
                    rotary_or_angles=self.position,
                )
            )
        return self._argmax_functions[bucket]

    def _target_log_probs(self, context: str, continuation: str) -> np.ndarray:
        inputs, targets, start, count, bucket = self._padded_example(context, continuation)
        log_probs = self._log_prob_function(bucket)(self.params, inputs, targets)[0]
        return np.asarray(
            jax.device_get(log_probs[start:start + count]), dtype=np.float64
        )

    def choice_score(self, context: str, continuation: str) -> float:
        values = self._target_log_probs(context, continuation)
        return float(np.mean(values))

    def exact_target(self, context: str, target: str) -> bool:
        target_ids = self.encoding.encode_ordinary(target)
        inputs, _, start, count, bucket = self._padded_example(context, target)
        predictions = np.asarray(
            jax.device_get(
                self._argmax_function(bucket)(self.params, inputs)[0, start:start + count]
            )
        )
        return bool(np.array_equal(predictions, np.asarray(target_ids)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("algebraic", "standard"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=Path)
    checkpoint_group.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "data/evals")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Development-only row limit")
    args = parser.parse_args()

    checkpoint = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    if checkpoint is None:
        raise FileNotFoundError("no checkpoint found")
    metadata = json.loads((args.benchmark_dir / "metadata.json").read_text())
    if args.limit is None and any(
        item.get("limited") for item in metadata.get("datasets", {}).values()
    ):
        raise RuntimeError("limited benchmark data cannot produce Pretraining study evidence")
    scorer = CheckpointScorer(args.architecture, checkpoint)
    results = {}
    for name in ("arc_easy", "hellaswag", "piqa"):
        rows = load_jsonl(args.benchmark_dir / f"{name}.jsonl")
        if args.limit is not None:
            rows = rows[:args.limit]
        correct = 0
        for row in rows:
            scores = [scorer.choice_score(row["context"], choice) for choice in row["choices"]]
            correct += int(int(np.argmax(scores)) == row["answer"])
        results[name] = {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}
        print(f"{name}: {correct}/{len(rows)} = {results[name]['accuracy']:.4f}", flush=True)

    rows = load_jsonl(args.benchmark_dir / "lambada.jsonl")
    if args.limit is not None:
        rows = rows[:args.limit]
    correct = sum(scorer.exact_target(row["context"], row["target"]) for row in rows)
    results["lambada"] = {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}

    record = {
        "study": 9,
        "scope": "zero_shot_benchmarks" if args.limit is None else "smoke_only",
        "architecture": "baseline" if args.architecture == "standard" else "algebraic",
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "environment": environment(),
        "dataset_metadata": metadata,
        "results": results,
    }
    if source_hashes() != record["environment"]["source_sha256"]:
        raise RuntimeError("source changed during benchmark evaluation")
    write_json(args.output, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
