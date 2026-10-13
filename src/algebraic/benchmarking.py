"""Shared checkpoint loading, provenance, and bounded-context LM scoring.

JAX is imported only when loading/scoring a model. Dataset preparation and
post-training job planning do not initialize an accelerator.
"""

from collections import defaultdict
from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
ZERO_SHOT_PROTOCOL = "algebraic-zero-shot-mean-continuation-logprob"
RECALL_PROTOCOL = "algebraic-recall-greedy-16-token-budget"
PUBLIC_TASKS = ("arc_easy", "arc_challenge", "hellaswag", "piqa", "openbookqa",
                "winogrande", "boolq", "lambada")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_dataset(directory, name, limit=None):
    """Check the full frozen file before applying an explicitly marked limit."""
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    info = meta["datasets"][name]
    path = directory / f"{name}.jsonl"
    if sha256_file(path) != info["sha256"]:
        raise ValueError(f"dataset hash mismatch: {path}")
    rows = load_jsonl(path)
    if len(rows) != info["rows"] or not rows:
        raise ValueError(f"empty or incomplete dataset: {path}")
    ids = [str(row.get("id", f"{name}/{i}")) for i, row in enumerate(rows)]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate example IDs: {path}")
    return rows[:limit], info


def source_fingerprint():
    paths = sorted((ROOT / "src/algebraic").rglob("*.py"))
    paths += [ROOT / "scripts" / name for name in
              ("evaluate_zero_shot.py", "evaluate_recall.py", "evaluate_main.py")]
    digest = hashlib.sha256()
    for path in paths:
        if path.exists():
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"sha256": digest.hexdigest(), "git_revision": revision}


def checkpoint_fingerprint(run):
    run = Path(run)
    meta = json.loads((run / "meta.json").read_text())
    return {"params_sha256": sha256_file(run / "params.npz"),
            "meta_sha256": sha256_file(run / "meta.json"),
            "result_sha256": sha256_file(run / "result.json") if (run / "result.json").exists() else None,
            "config": meta["config"], "training_args": meta.get("args", {})}


def provenance(run, scorer, protocol):
    import jax
    result_path = Path(run) / "result.json"
    training_result = json.loads(result_path.read_text()) if result_path.exists() else {}
    return {"run": str(Path(run).resolve()), "protocol": protocol,
            "checkpoint": checkpoint_fingerprint(run), "source": source_fingerprint(),
            "execution_config": asdict(scorer.cfg), "backend": jax.default_backend(),
            "jax_version": jax.__version__, "batch": scorer.batch,
            "tokenizer": "gpt2", "training_metrics": {key: training_result[key] for key in
            ("final_eval_nll", "final_eval_ppl", "train_seconds", "mean_tok_per_s") if key in training_result}}


def load_model(run_dir):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from algebraic.model import LMConfig

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    # Checkpoint metadata may include extra diagnostic fields.
    names = {field.name for field in fields(LMConfig)}
    cfg = LMConfig(**{k: v for k, v in meta["config"].items() if k in names})
    if cfg.vocab_size != 50257:
        raise ValueError("this evaluation protocol requires the GPT-2 vocabulary")
    if jax.default_backend() != "tpu":
        cfg = replace(cfg, attn_impl="dense")
    L, d, f = cfg.n_layers, cfg.d_model, cfg.d_ff
    expected = {("embed",): (cfg.vocab_size, d)}
    for name in ("w_q", "w_k", "w_v", "w_o"):
        expected[("layers", name)] = (L, d, d)
    for name in ("w_g", "w_u"):
        expected[("layers", name)] = (L, d, f)
    expected[("layers", "w_d")] = (L, f, d)
    if cfg.norm == "rms":
        expected.update({("embed_gain",): (d,), ("final_gain",): (d,),
                         ("layers", "ln1"): (L, d), ("layers", "ln2"): (L, d)})
    if cfg.qk_norm:
        expected.update({("layers", "q_gain"): (L, cfg.head_dim),
                         ("layers", "k_gain"): (L, cfg.head_dim)})
    if cfg.sink == "learned":
        expected[("layers", "sink")] = (L, cfg.n_heads)
    params = {"layers": {}}
    seen = set()
    with np.load(run_dir / "params.npz", allow_pickle=False) as flat:
        for key in flat.files:
            parts = tuple(re.findall(r"\['([^']+)'\]", key))
            if parts not in expected or parts in seen:
                raise ValueError(f"unexpected checkpoint parameter: {key}")
            value = flat[key]
            if value.shape != expected[parts] or not np.isfinite(value).all():
                raise ValueError(f"invalid checkpoint parameter: {key}")
            target = params["layers"] if len(parts) == 2 else params
            target[parts[-1]] = jnp.asarray(value)
            seen.add(parts)
    if seen != set(expected):
        raise ValueError(f"missing checkpoint parameters: {set(expected) - seen}")
    return cfg, params


class Scorer:
    def __init__(self, cfg, params, batch=8):
        import jax
        import tiktoken
        if batch < 1:
            raise ValueError("batch must be positive")
        self.jax, self.cfg, self.params, self.batch = jax, cfg, params, batch
        self.enc = tiktoken.get_encoding("gpt2")
        self.T = cfg.seq_len
        self._score_fns, self._greedy_fns = {}, {}
        self.truncated_pairs = 0

    def _bucket(self, n):
        if n > self.T:
            raise ValueError(f"{n} tokens exceed trained context {self.T}")
        return min(self.T, max(128, 1 << (n - 1).bit_length()))

    def _score_fn(self, bucket, width):
        key = (bucket, width)
        if key not in self._score_fns:
            import jax.numpy as jnp
            from algebraic.model import F32, forward_hidden, position_tables
            from algebraic.evaluation import logprob_from_logits
            cfg, tables = self.cfg, position_tables(self.cfg, bucket)

            def run(params, x, starts, targets):
                h = forward_hidden(params, x, cfg, tables)
                positions = jnp.minimum(starts[:, None] + jnp.arange(width), bucket - 1)
                selected = h[jnp.arange(x.shape[0])[:, None], positions]
                # Project continuation positions only; avoid B*T*V logits.
                logits = jnp.matmul(selected.reshape(-1, h.shape[-1]),
                                    params["embed"].astype(h.dtype).T,
                                    preferred_element_type=F32)
                lp = logprob_from_logits(logits, targets.reshape(-1), cfg.head,
                                         cfg.head_n, cfg.head_c)
                return lp.reshape(x.shape[0], width), jnp.argmax(logits, -1).reshape(targets.shape)
            self._score_fns[key] = self.jax.jit(run)
        return self._score_fns[key]

    def score(self, pairs):
        encoded = [(self.enc.encode_ordinary(c) or [self.enc.eot_token],
                    self.enc.encode_ordinary(t)) for c, t in pairs]
        return self.score_token_pairs(encoded, allow_truncation=True)

    def score_token_pairs(self, pairs, *, allow_truncation=False):
        """Mean target log-probability and exact greedy target-prefix match.

        For a causal model, all teacher-forced target argmaxes matching is
        equivalent to the free greedy generated prefix matching the target.
        Recall uses allow_truncation=False and also reports free generation.
        """
        import numpy as np
        encoded, groups = [], defaultdict(list)
        for context, continuation in pairs:
            ctx, cont = list(context), list(continuation)
            if not ctx or not cont or len(cont) > self.T:
                raise ValueError("need nonempty context/target and target length <= context")
            capacity = self.T + 1 - len(cont)
            if len(ctx) > capacity:
                if not allow_truncation:
                    raise ValueError("recall scoring would truncate its evidence")
                ctx = ctx[-capacity:]
                self.truncated_pairs += 1
            index = len(encoded)
            encoded.append((ctx + cont[:-1], len(ctx) - 1, cont))
            groups[(self._bucket(len(ctx) + len(cont) - 1),
                    1 << (len(cont) - 1).bit_length())].append(index)
        lps, hits = [None] * len(encoded), [None] * len(encoded)
        for (bucket, width), indices in groups.items():
            fn = self._score_fn(bucket, width)
            for offset in range(0, len(indices), self.batch):
                chunk = indices[offset:offset + self.batch]
                padded = chunk + [chunk[-1]] * (self.batch - len(chunk))
                x = np.full((self.batch, bucket), self.enc.eot_token, np.int32)
                targets = np.full((self.batch, width), self.enc.eot_token, np.int32)
                starts = np.empty(self.batch, np.int32)
                for row, index in enumerate(padded):
                    tokens, starts[row], cont = encoded[index]
                    x[row, :len(tokens)] = tokens
                    targets[row, :len(cont)] = cont
                lp, am = fn(self.params, x, starts, targets)
                lp, am = np.asarray(lp, np.float64), np.asarray(am)
                if not np.isfinite(lp).all():
                    raise ValueError("non-finite evaluation log-probabilities")
                for row, index in enumerate(chunk):
                    cont = encoded[index][2]
                    lps[index] = float(lp[row, :len(cont)].mean())
                    hits[index] = bool(np.array_equal(am[row, :len(cont)], cont))
        return lps, hits

    def _greedy_fn(self, bucket, steps):
        key = (bucket, steps)
        if key not in self._greedy_fns:
            import jax.numpy as jnp
            from algebraic.model import F32, forward_hidden, position_tables
            cfg, tables = self.cfg, position_tables(self.cfg, bucket)

            def run(params, tokens, lengths):
                generated = jnp.zeros((tokens.shape[0], steps), jnp.int32)

                def body(i, state):
                    current, out = state
                    h = forward_hidden(params, current, cfg, tables)
                    last = h[jnp.arange(current.shape[0]), lengths + i - 1]
                    logits = jnp.matmul(last, params["embed"].astype(h.dtype).T,
                                        preferred_element_type=F32)
                    token = jnp.argmax(logits, -1).astype(jnp.int32)
                    current = current.at[jnp.arange(current.shape[0]), lengths + i].set(token)
                    return current, out.at[:, i].set(token)

                return self.jax.lax.fori_loop(0, steps, body, (tokens, generated))[1]
            self._greedy_fns[key] = self.jax.jit(run)
        return self._greedy_fns[key]

    def greedy(self, prompts, *, max_new_tokens=16):
        """Free autoregressive decoding. Never truncate or extend trained context."""
        import numpy as np
        if max_new_tokens < 1:
            raise ValueError("generation budget must be positive")
        groups, result = defaultdict(list), [None] * len(prompts)
        for index, prompt in enumerate(prompts):
            if not prompt:
                raise ValueError("empty prompt")
            groups[self._bucket(len(prompt) + max_new_tokens)].append(index)
        for bucket, indices in groups.items():
            fn = self._greedy_fn(bucket, max_new_tokens)
            for offset in range(0, len(indices), self.batch):
                chunk = indices[offset:offset + self.batch]
                padded = chunk + [chunk[-1]] * (self.batch - len(chunk))
                x = np.full((self.batch, bucket), self.enc.eot_token, np.int32)
                lengths = np.asarray([len(prompts[i]) for i in padded], np.int32)
                for row, index in enumerate(padded):
                    x[row, :lengths[row]] = prompts[index]
                out = np.asarray(fn(self.params, x, lengths))
                for row, index in enumerate(chunk):
                    tokens = out[row].tolist()
                    if self.enc.eot_token in tokens:
                        tokens = tokens[:tokens.index(self.enc.eot_token)]
                    result[index] = {"token_ids": tokens, "text": self.enc.decode(tokens)}
        return result
