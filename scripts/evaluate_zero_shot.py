#!/usr/bin/env python3
"""Zero-shot ARC-Easy, HellaSwag, PIQA, and LAMBADA for a trained model.

Same protocol as the first study: multiple-choice items are scored by the
mean target-token log-probability of each continuation (highest wins);
LAMBADA counts an exact greedy match of every token of the final word.
Contexts are left-truncated to the model context; examples are padded to
power-of-two buckets and scored in batches. Under an algebraic head the
greedy token is the argmax of the logits (the head's link is monotone).

Data: ``python scripts/prepare_eval_benchmarks.py --output-dir ~/data/evals``.

  python scripts/evaluate_zero_shot.py runs/main/algebraic_s42 --out results/main/algebraic_s42_zero_shot.json
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_model(run_dir: Path):
    from dataclasses import replace
    import jax
    import jax.numpy as jnp
    from src.sota.model import LMConfig
    meta = json.loads((run_dir / "meta.json").read_text())
    cfg = LMConfig(**meta["config"])
    if jax.default_backend() != "tpu":   # Pallas kernels need a TPU; dense path elsewhere
        cfg = replace(cfg, attn_impl="dense")
    flat = np.load(run_dir / "params.npz")
    params = {"layers": {}}
    for key in flat.files:
        parts = [x.strip("[]'\"") for x in key.split("]") if x]
        target = params["layers"] if parts[0] == "layers" else params
        target[parts[-1]] = jnp.asarray(flat[key])
    return cfg, params


class Scorer:
    def __init__(self, cfg, params, batch=32):
        import jax
        import tiktoken
        self.jax = jax
        self.cfg = cfg
        self.params = params
        self.batch = batch
        self.enc = tiktoken.get_encoding("gpt2")
        self.T = cfg.seq_len
        self._fns = {}

    def _bucket(self, n):
        return min(self.T, max(128, 1 << (n - 1).bit_length()))

    def _fn(self, bucket):
        if bucket not in self._fns:
            import jax.numpy as jnp
            from src.sota.model import F32, forward_hidden, position_tables
            from src.sota.evaluation import logprob_from_logits
            cfg = self.cfg
            tables = position_tables(cfg, bucket)

            def run(params, x, y):
                h = forward_hidden(params, x, cfg, tables)
                B, T, d = h.shape
                logits = jnp.matmul(h.reshape(B * T, d), params["embed"].astype(h.dtype).T,
                                    preferred_element_type=F32)
                lp = logprob_from_logits(logits, y.reshape(-1), cfg.head, cfg.head_n, cfg.head_c,
                                         cfg.eps_vocab).reshape(B, T)
                return lp, jnp.argmax(logits, axis=-1).reshape(B, T)
            self._fns[bucket] = self.jax.jit(run)
        return self._fns[bucket]

    def _encode(self, context, continuation):
        ctx = self.enc.encode_ordinary(context) or [self.enc.eot_token]
        cont = self.enc.encode_ordinary(continuation)
        if not cont:
            raise ValueError("empty continuation")
        ctx = ctx[-(self.T + 1 - len(cont)):]
        return ctx + cont, len(ctx) - 1, len(cont), cont

    def score(self, pairs):
        """Mean continuation log-prob and greedy exact match for (context, continuation) pairs."""
        encoded = [self._encode(c, t) for c, t in pairs]
        out_lp = [None] * len(pairs)
        out_hit = [None] * len(pairs)
        by_bucket = {}
        for i, (ids, start, count, cont) in enumerate(encoded):
            by_bucket.setdefault(self._bucket(len(ids) - 1), []).append(i)
        for bucket, idxs in by_bucket.items():
            fn = self._fn(bucket)
            for b0 in range(0, len(idxs), self.batch):
                chunk = idxs[b0:b0 + self.batch]
                rows = []
                for i in chunk:
                    ids = encoded[i][0]
                    rows.append(ids + [self.enc.eot_token] * (bucket + 1 - len(ids)))
                while len(rows) < self.batch:          # fixed batch shape per bucket
                    rows.append(rows[-1])
                arr = np.asarray(rows, np.int32)
                lp, am = fn(self.params, arr[:, :-1], arr[:, 1:])
                lp, am = np.asarray(lp, np.float64), np.asarray(am)
                for j, i in enumerate(chunk):
                    _, start, count, cont = encoded[i]
                    out_lp[i] = float(lp[j, start:start + count].mean())
                    out_hit[i] = bool(np.array_equal(am[j, start:start + count], np.asarray(cont)))
        return out_lp, out_hit


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="run directory with params.npz and meta.json")
    p.add_argument("--benchmarks", type=Path, default=Path.home() / "data" / "evals")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, help="development only: first N rows per task")
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()
    cfg, params = load_model(a.run)
    scorer = Scorer(cfg, params, a.batch)
    meta = json.loads((a.benchmarks / "metadata.json").read_text())
    results = {}
    t0 = time.time()
    for name in ("arc_easy", "hellaswag", "piqa"):
        rows = load_jsonl(a.benchmarks / f"{name}.jsonl")[:a.limit]
        pairs = [(r["context"], ch) for r in rows for ch in r["choices"]]
        lps, _ = scorer.score(pairs)
        k, correct = 0, 0
        for r in rows:
            n = len(r["choices"])
            correct += int(int(np.argmax(lps[k:k + n])) == r["answer"])
            k += n
        results[name] = {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}
        print(f"{name}: {correct}/{len(rows)} = {correct/len(rows):.4f} ({time.time()-t0:.0f}s)", flush=True)
    rows = load_jsonl(a.benchmarks / "lambada.jsonl")[:a.limit]
    _, hits = scorer.score([(r["context"], r["target"]) for r in rows])
    correct = int(sum(hits))
    results["lambada"] = {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}
    print(f"lambada: {correct}/{len(rows)} = {correct/len(rows):.4f}", flush=True)
    record = {"run": str(a.run), "config": json.loads((a.run / "meta.json").read_text())["config"],
              "limited": a.limit is not None, "dataset_metadata": meta, "results": results}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
