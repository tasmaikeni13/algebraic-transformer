#!/usr/bin/env python3
"""Train one language model on one chip, one host, or a multi-host TPU slice.

Data parallel: parameters and AdamW moments are replicated, each device runs
its share of the global batch inside ``shard_map`` (so Pallas kernels need no
SPMD partitioning), and gradients are averaged with ``pmean``.

Example (single chip):
  python -m algebraic.train --preset standard --size proxy --steps 4000 \
      --batch 64 --micro 16 --out runs/std
Component overrides: ``--set attn_n=16 --set head=alg_ce``.
"""

import argparse
from dataclasses import asdict, fields, replace
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np


def parse_overrides(pairs):
    from algebraic.model import LMConfig
    types = {f.name: f.type for f in fields(LMConfig)}
    out = {}
    for item in pairs or []:
        key, _, value = item.partition("=")
        if key not in types:
            raise ValueError(f"unknown config field {key!r}")
        kind = types[key]
        kind = kind if isinstance(kind, type) else eval(kind)  # dataclass string annotations
        if kind is bool:
            out[key] = value.lower() in ("1", "true", "yes")
        else:
            out[key] = kind(value)
    return out


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="standard")
    p.add_argument("--size", default="proxy")
    p.add_argument("--set", action="append", default=[], help="config override key=value")
    p.add_argument("--train", default=os.path.expanduser("~/data/tok/fineweb1b_train.npy"))
    p.add_argument("--valid", default=os.path.expanduser("~/data/tok/fineweb1b_valid.npy"))
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=64, help="global batch in sequences")
    p.add_argument("--micro", type=int, default=16, help="per-device micro-batch in sequences")
    p.add_argument("--steps", type=int, default=0, help="optimizer steps (0: use --tokens)")
    p.add_argument("--tokens", type=float, default=0, help="token budget, rounded up to full steps")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=float, default=0.02, help="fraction (<1) or steps (>=1)")
    p.add_argument("--schedule", default="cosine", choices=("cosine", "ards", "wsd", "linear"))
    p.add_argument("--decay-frac", type=float, default=0.2, help="WSD decay fraction")
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--b1", type=float, default=0.9)
    p.add_argument("--b2", type=float, default=0.95)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=None, help="defaults to --seed")
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-tokens", type=float, default=2e6)
    p.add_argument("--eval-batch", type=int, default=16, help="per-device eval sequences")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", required=True)
    p.add_argument("--save-params", action="store_true")
    p.add_argument("--distributed", action="store_true", help="call jax.distributed.initialize")
    p.add_argument("--profile-steps", type=int, default=0, help="time N steps then exit")
    p.add_argument("--ckpt-every", type=int, default=0, help="save a checkpoint every N steps")
    p.add_argument("--keep-ckpts", type=int, default=2)
    p.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in --out")
    return p.parse_args(argv)


class WindowSampler:
    """Deterministic random contiguous windows; same (seed, step) -> same batch."""

    def __init__(self, path, seq_len, global_batch, seed, process_index, process_count):
        self.tokens = np.load(path, mmap_mode="r")
        if self.tokens.ndim != 1 or len(self.tokens) <= seq_len + 1:
            raise ValueError("training tokens must be a one-dimensional array longer than the context")
        self.T = seq_len
        self.B = global_batch
        self.seed = seed
        if global_batch % process_count:
            raise ValueError("global batch must divide across processes")
        self.local = global_batch // process_count
        self.offset = process_index * self.local

    def batch(self, step):
        rng = np.random.default_rng([self.seed, step])
        starts = rng.integers(0, len(self.tokens) - self.T - 1, size=self.B)
        local = starts[self.offset:self.offset + self.local]
        arr = np.stack([np.asarray(self.tokens[s:s + self.T + 1]) for s in local]).astype(np.int32)
        return arr[:, :-1], arr[:, 1:]


class Prefetcher:
    def __init__(self, sampler, start, stop, depth=4):
        self.q = queue.Queue(maxsize=depth)
        self.thread = threading.Thread(target=self._run, args=(sampler, start, stop), daemon=True)
        self.thread.start()

    def _run(self, sampler, start, stop):
        for step in range(start, stop):
            self.q.put(sampler.batch(step))

    def get(self):
        return self.q.get()


def save_checkpoint(directory: Path, step: int, payload: dict, keep: int) -> None:
    """Atomically write this host's copy of the (replicated) training state."""
    import pickle
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step-{step:07d}.pkl"
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    for old in sorted(directory.glob("step-*.pkl"))[:-keep]:
        old.unlink()


def latest_checkpoint(directory: Path):
    import pickle
    found = sorted(Path(directory).glob("step-*.pkl"))
    if not found:
        return None
    with found[-1].open("rb") as fh:
        return pickle.load(fh)


def lr_schedule(args, total_steps):
    import jax.numpy as jnp
    peak = args.lr
    warm = args.warmup if args.warmup >= 1 else max(1, int(round(args.warmup * total_steps)))
    floor = peak * args.min_lr_ratio
    T = float(total_steps)

    def fn(step):
        t = jnp.asarray(step, jnp.float32) + 1.0
        warm_f = jnp.minimum(1.0, t / warm)
        prog = jnp.clip((t - warm) / jnp.maximum(1.0, T - warm), 0.0, 1.0)
        if args.schedule == "cosine":
            lr = floor + (peak - floor) * 0.5 * (1.0 + jnp.cos(jnp.pi * prog))
        elif args.schedule == "linear":
            lr = floor + (peak - floor) * (1.0 - prog)
        elif args.schedule == "wsd":
            start = 1.0 - args.decay_frac
            frac = jnp.clip((prog - start) / args.decay_frac, 0.0, 1.0)
            lr = peak - (peak - floor) * frac
        else:  # Rational decay with scale 0.8 T.
            d = jnp.maximum(0.0, t - warm) / (0.8 * T)
            lr = peak / jnp.sqrt(1.0 + d * d)
        return jnp.where(t < warm, peak * warm_f, lr)

    return fn, warm


def main(argv=None):
    args = get_args(argv)
    if (args.steps < 0 or args.tokens < 0 or not (args.steps or args.tokens)
            or min(args.batch, args.micro, args.seq_len, args.eval_batch, args.log_every) < 1
            or args.keep_ckpts < 1 or args.ckpt_every < 0):
        raise ValueError("specify a positive training budget, batch/context sizes, and checkpoint retention")
    import jax
    if args.distributed:
        jax.distributed.initialize()
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from jax.experimental.shard_map import shard_map

    from algebraic import config as cfgs
    from algebraic.model import (count_params, decay_mask, eval_logprobs, forward_hidden,
                                head_loss, init_params, position_tables,
                                telemetry_is_probability)

    overrides = parse_overrides(args.set)
    overrides["seq_len"] = args.seq_len
    cfg = cfgs.build(args.preset, args.size, **overrides)

    devices = jax.devices()
    n_dev = len(devices)
    pidx, pcount = jax.process_index(), jax.process_count()
    is_main = pidx == 0
    mesh = Mesh(np.array(devices), ("data",))
    T = args.seq_len
    tokens_per_step = args.batch * T
    total_steps = args.steps or math.ceil(args.tokens / tokens_per_step)
    if args.batch % (n_dev * args.micro):
        raise ValueError(f"batch {args.batch} not divisible by devices*micro {n_dev}*{args.micro}")
    accum = args.batch // (n_dev * args.micro)
    out = Path(args.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)

    def log(rec):
        if is_main:
            with open(out / "log.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    tables = position_tables(cfg, T)
    params = init_params(cfg, jax.random.PRNGKey(args.seed))
    mask = decay_mask(params)
    n_params = count_params(params)
    sched, warm = lr_schedule(args, total_steps)
    is_prob = telemetry_is_probability(cfg)

    opt_state = {
        "count": jnp.zeros([], jnp.int32),
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
    }

    def local_step(params, opt_state, xs, ys):
        # xs, ys: (accum, micro, T) for this device
        def micro_fn(carry, batch):
            gsum, lsum = carry
            x, y = batch

            def loss_fn(p):
                h = forward_hidden(p, x, cfg, tables)
                return head_loss(p, h, y, cfg)

            (_, tel), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
            nll = -jnp.log(jnp.maximum(tel, 1e-30)) if is_prob else tel
            gsum = jax.tree_util.tree_map(lambda a, b: a + b.astype(jnp.float32), gsum, g)
            return (gsum, lsum + jnp.mean(nll)), None

        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        (g, lsum), _ = lax.scan(micro_fn, (zeros, jnp.zeros([], jnp.float32)), (xs, ys))
        g = jax.tree_util.tree_map(lambda a: lax.pmean(a / accum, "data"), g)
        loss = lax.pmean(lsum / accum, "data")
        gnorm = jnp.sqrt(sum(jnp.sum(a * a) for a in jax.tree_util.tree_leaves(g)))
        scale = jnp.minimum(1.0, args.clip / jnp.maximum(gnorm, 1e-12))
        count = opt_state["count"] + 1
        lr = sched(count - 1)
        b1, b2 = args.b1, args.b2
        c1 = 1.0 - b1 ** count.astype(jnp.float32)
        c2 = 1.0 - b2 ** count.astype(jnp.float32)

        def upd(p, gi, m, v, dmask):
            gi = gi * scale
            m = b1 * m + (1 - b1) * gi
            v = b2 * v + (1 - b2) * gi * gi
            u = (m / c1) / (jnp.sqrt(v / c2) + args.adam_eps)
            if dmask:
                u = u + args.wd * p
            return p - lr * u, m, v

        flat_p, tdef = jax.tree_util.tree_flatten(params)
        flat_g = tdef.flatten_up_to(g)
        flat_m = tdef.flatten_up_to(opt_state["m"])
        flat_v = tdef.flatten_up_to(opt_state["v"])
        flat_mask = tdef.flatten_up_to(mask)
        new = [upd(*z) for z in zip(flat_p, flat_g, flat_m, flat_v, flat_mask)]
        params = tdef.unflatten([a for a, _, _ in new])
        opt_state = {"count": count, "m": tdef.unflatten([b for _, b, _ in new]),
                     "v": tdef.unflatten([c for _, _, c in new])}
        return params, opt_state, {"loss": loss, "gnorm": gnorm, "lr": lr}

    rep = P()
    data_spec = P(None, "data", None)
    step_fn = jax.jit(
        shard_map(local_step, mesh=mesh, in_specs=(rep, rep, data_spec, data_spec),
                  out_specs=(rep, rep, rep), check_rep=False),
        donate_argnums=(0, 1),
    )

    def local_eval(params, x, y):
        lp = eval_logprobs(params, x, y, cfg, tables)
        return lax.psum(-jnp.sum(lp), "data"), lax.psum(-jnp.sum(lp, axis=0), "data")

    eval_fn = jax.jit(shard_map(local_eval, mesh=mesh, in_specs=(rep, P("data", None), P("data", None)),
                                out_specs=(rep, rep), check_rep=False))

    valid = np.load(args.valid, mmap_mode="r")
    eval_bs = args.eval_batch * n_dev
    n_eval_seq = int(args.eval_tokens) // T
    n_eval_seq = max(eval_bs, (n_eval_seq // eval_bs) * eval_bs)
    if n_eval_seq * T + 1 > len(valid):
        raise ValueError("validation array too small for --eval-tokens")
    rep_sharding = NamedSharding(mesh, rep)
    batch_sharding = NamedSharding(mesh, data_spec)
    eval_sharding = NamedSharding(mesh, P("data", None))
    start_step, history, elapsed_before = 0, [], 0.0
    ckpt_dir = out / "ckpt"
    if args.resume:
        saved = latest_checkpoint(ckpt_dir)
        if saved is not None:
            if saved["config"] != asdict(cfg) or saved["total_steps"] != total_steps:
                raise ValueError("checkpoint configuration does not match this run")
            recipe = {k: v for k, v in vars(args).items()
                      if k not in ("resume", "log_every", "profile_steps", "ckpt_every", "keep_ckpts", "out")}
            if saved.get("training_args") != recipe:
                raise ValueError("checkpoint seed, data, or training recipe does not match this run")
            params = jax.tree_util.tree_map(jnp.asarray, saved["params"])
            opt_state = jax.tree_util.tree_map(jnp.asarray, saved["opt_state"])
            start_step = int(saved["step"])
            history = saved["history"]
            elapsed_before = float(saved["elapsed"])
            if is_main:
                print(f"resumed from step {start_step}", flush=True)
    params = jax.device_put(params, rep_sharding)
    opt_state = jax.device_put(opt_state, rep_sharding)

    def put_global(local_arr, sharding, global_shape):
        return jax.make_array_from_process_local_data(sharding, local_arr, global_shape)

    def run_eval():
        tot, per_pos, count = 0.0, np.zeros(T), 0
        per_proc = eval_bs // pcount
        for b0 in range(0, n_eval_seq, eval_bs):
            idx = np.arange(b0, b0 + eval_bs)[pidx * per_proc:(pidx + 1) * per_proc]
            arr = np.stack([np.asarray(valid[i * T:i * T + T + 1]) for i in idx]).astype(np.int32)
            x = put_global(arr[:, :-1], eval_sharding, (eval_bs, T))
            y = put_global(arr[:, 1:], eval_sharding, (eval_bs, T))
            s, pp = eval_fn(params, x, y)
            tot += float(s)
            per_pos += np.asarray(pp)
            count += eval_bs * T
        nll = tot / count
        per_pos = per_pos / n_eval_seq
        # NLL by position bucket [0,1), [1,2), [2,4), ... (log2 buckets)
        edges = [0] + [2 ** i for i in range(int(math.log2(T)) + 1)]
        buckets = [float(per_pos[a:b].mean()) for a, b in zip(edges[:-1], edges[1:]) if b <= T]
        return nll, buckets

    meta = {"config": asdict(cfg), "args": vars(args), "n_params": n_params,
            "devices": n_dev, "processes": pcount, "accum": accum,
            "total_steps": total_steps, "tokens_per_step": tokens_per_step,
            "warmup_steps": warm, "eval_tokens": n_eval_seq * T}
    if is_main:
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps({k: meta[k] for k in ("n_params", "devices", "accum", "total_steps")}), flush=True)

    sampler = WindowSampler(args.train, T, args.batch, args.data_seed if args.data_seed is not None else args.seed,
                            pidx, pcount)
    prefetch = Prefetcher(sampler, start_step, total_steps)
    local_rows = args.batch // pcount
    local_accum_shape = (accum, local_rows // accum, T)
    global_shape = (accum, args.batch // accum, T)

    t_start = time.time() - elapsed_before
    t_last = time.time()
    compile_s = None
    for step in range(start_step, total_steps):
        x_np, y_np = prefetch.get()
        # rows are interleaved so every device sees a contiguous micro-batch
        x_np = x_np.reshape(local_rows // accum, accum, T).transpose(1, 0, 2)
        y_np = y_np.reshape(local_rows // accum, accum, T).transpose(1, 0, 2)
        x = put_global(np.ascontiguousarray(x_np), batch_sharding, global_shape)
        y = put_global(np.ascontiguousarray(y_np), batch_sharding, global_shape)
        params, opt_state, metrics = step_fn(params, opt_state, x, y)
        if step == start_step:
            jax.block_until_ready(metrics["loss"])
            compile_s = time.time() - t_last
            t_last = time.time()
        if args.profile_steps and step + 1 == args.profile_steps + 1:
            jax.block_until_ready(params)
            dt = (time.time() - t_last) / args.profile_steps
            res = {"step_time": dt, "tokens_per_s": tokens_per_step / dt, "compile_s": compile_s}
            if is_main:
                print(json.dumps(res), flush=True)
                (out / "profile.json").write_text(json.dumps(res))
            return 0
        if (step + 1) % args.log_every == 0 or step + 1 == total_steps:
            m = {k: float(v) for k, v in metrics.items()}
            now = time.time()
            elapsed_steps = min(args.log_every, step - start_step) if step > start_step else 1
            rate = tokens_per_step * max(1, elapsed_steps) / max(now - t_last, 1e-9)
            t_last = now
            rec = {"step": step + 1, "tokens": (step + 1) * tokens_per_step, **m,
                   "tok_per_s": rate, "elapsed": now - t_start}
            history.append(rec)
            log(rec)
            if is_main:
                print(f"step {step+1}/{total_steps} loss {m['loss']:.4f} gnorm {m['gnorm']:.3f} "
                      f"lr {m['lr']:.2e} {rate/1e3:.0f}K tok/s", flush=True)
            if not all(math.isfinite(v) for v in m.values()):
                log({"error": "non-finite training metrics", "step": step + 1})
                raise FloatingPointError(f"non-finite training at step {step+1}; no completed result written")
        if args.ckpt_every and (step + 1) % args.ckpt_every == 0 and step + 1 < total_steps:
            host = jax.tree_util.tree_map(lambda a: np.asarray(a.addressable_data(0)), (params, opt_state))
            save_checkpoint(ckpt_dir, step + 1, {
                "config": asdict(cfg), "total_steps": total_steps, "step": step + 1,
                "training_args": {k: v for k, v in vars(args).items()
                                  if k not in ("resume", "log_every", "profile_steps", "ckpt_every", "keep_ckpts", "out")},
                "params": host[0], "opt_state": host[1], "history": history,
                "elapsed": time.time() - t_start}, args.keep_ckpts)
        if args.eval_every and (step + 1) % args.eval_every == 0 and step + 1 < total_steps:
            nll, _ = run_eval()
            log({"step": step + 1, "eval_nll": nll, "eval_ppl": math.exp(nll)})
            if is_main:
                print(f"  eval nll {nll:.4f} ppl {math.exp(nll):.2f}", flush=True)

    jax.block_until_ready(params)
    train_s = time.time() - t_start
    nll, buckets = run_eval()
    if not math.isfinite(nll):
        raise FloatingPointError("non-finite final evaluation; no completed result written")
    result = {**meta, "completed_steps": total_steps, "final_eval_nll": nll, "final_eval_ppl": math.exp(nll),
              "eval_nll_by_log2_position": buckets, "train_seconds": train_s,
              "compile_seconds": compile_s,
              "mean_tok_per_s": total_steps * tokens_per_step / max(train_s - (compile_s or 0), 1e-9),
              "final_train_loss": history[-1]["loss"] if history else None}
    log({"step": total_steps, "eval_nll": nll, "eval_ppl": math.exp(nll), "final": True})
    if is_main:
        if args.save_params:
            host = jax.tree_util.tree_map(lambda a: np.asarray(jax.device_get(a)), params)
            flat, _ = jax.tree_util.tree_flatten_with_path(host)
            pending = out / "params.npz.tmp"
            with pending.open("wb") as stream:
                np.savez(stream, **{jax.tree_util.keystr(k): v for k, v in flat})
            pending.replace(out / "params.npz")
        pending = out / "result.json.tmp"
        pending.write_text(json.dumps(result, indent=2))
        pending.replace(out / "result.json")
        print(f"FINAL eval nll {nll:.4f} ppl {math.exp(nll):.3f} ({train_s/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
