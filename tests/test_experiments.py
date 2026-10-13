"""Experiment budgets, real preset shapes, sweep selection, and corpus integrity."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from algebraic import config
from algebraic.model import init_params
from algebraic.train import get_args
from scripts import prepare_fineweb_edu as data
from scripts.hparam_sweep import select, sweep_jobs
from scripts.run_comparison import ROOT, build_plan, parameter_count, read_selection, training_jobs
from scripts.run_slice import validate_completed_job


def load(scale):
    return json.loads((ROOT / f"configs/comparison_{scale}.json").read_text())


@pytest.mark.parametrize("scale", ["350m", "1b"])
def test_real_parameter_shapes_and_plan(scale):
    c = load(scale)
    for arch in c["architectures"]:
        cfg = config.build(arch, scale)
        shapes = jax.eval_shape(lambda key: init_params(cfg, key), jax.random.PRNGKey(42))
        assert sum(x.size for x in jax.tree_util.tree_leaves(shapes)) == parameter_count(c["training"], arch)
        assert {k: getattr(cfg, k) for k in c["training"]["dimensions"]} == c["training"]["dimensions"]
        assert cfg.algebraic_path == (arch == "algebraic")
    plan = build_plan(ROOT / f"configs/comparison_{scale}.json", Path("data"), Path("eval"), Path("recall"), "test", 1)
    assert not plan["training_launched"] and plan["selected_hyperparameters"] is None
    assert plan["optimizer_steps"] * plan["tokens_per_step"] >= c["training"]["tokens"]
    assert len(plan["jobs"]) == 2
    assert len({get_args([*j["args"], "--out", "test"]).data_seed for j in plan["jobs"]}) == 1
    for job in plan["jobs"]:
        assert "attn_impl=flash" in job["args"] and "head_impl=fused" in job["args"]


@pytest.mark.parametrize("scale", ["350m", "1b"])
def test_budgeted_sweep_pairing(scale):
    c = load(scale)
    s = json.loads((ROOT / f"configs/sweep_{scale}.json").read_text())
    jobs = sweep_jobs(c, s, Path("data"))
    assert len(jobs) == 8 and len({j["name"] for j in jobs}) == 8
    args = [get_args([*j["args"], "--out", "test"]) for j in jobs]
    assert {a.seed for a in args} == {42}
    assert {a.tokens for a in args} == {s["tokens_per_candidate"]}
    assert {a.eval_tokens for a in args} == {s["eval_tokens"]}
    assert {a.preset for a in args} == {"standard", "algebraic"}
    assert len({a.lr for a in args}) == 3 and len({a.wd for a in args}) == 2
    assert not any(a.save_params for a in args)


def completed_fixture(root, job, loss):
    run = root / job["name"]
    run.mkdir(parents=True)
    args = get_args([*job["args"], "--distributed", "--out", str(run)])
    t = load(args.size)["training"]
    cfg = asdict(config.build(args.preset, args.size, seq_len=args.seq_len,
                              attn_impl="flash", head_impl="fused", flash_block=t["flash_block"],
                              head_block_n=t["head_block_n"], head_block_v=t["head_block_v"]))
    steps = int(np.ceil(args.tokens / (args.batch * args.seq_len)))
    meta = {"config": cfg, "args": vars(args), "total_steps": steps, "tokens_per_step": args.batch*args.seq_len}
    (run / "meta.json").write_text(json.dumps(meta))
    (run / "result.json").write_text(json.dumps({**meta, "completed_steps": steps, "final_eval_nll": loss}))
    (run / "log.jsonl").write_text(json.dumps({"step": steps, "loss": 4.0}) + "\n")
    return run


def test_selection_requires_every_pilot_and_selects_each_arm(tmp_path):
    config_path = ROOT / "configs/comparison_350m.json"
    sweep_path = ROOT / "configs/sweep_350m.json"
    c, s = load("350m"), json.loads(sweep_path.read_text())
    jobs = sweep_jobs(c, s, tmp_path / "data")
    for index, job in enumerate(jobs):
        completed_fixture(tmp_path, job, 3.0 + (index if index % 2 == 0 else 8-index) / 100)
    out = tmp_path / "selected.json"
    result = select(config_path, sweep_path, tmp_path / "data", tmp_path, out)
    assert result["best_candidates"] == {"standard": "lr_low", "algebraic": "lower_decay"}
    assert read_selection(out, config_path) == result["selected"]
    plan = build_plan(config_path, tmp_path / "data", tmp_path, tmp_path, "test", 1, out)
    assert plan["selected_hyperparameters"] == result["selected"]
    assert "--hparams" in plan["evaluation_command"]
    (tmp_path / jobs[-1]["name"] / "result.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        select(config_path, sweep_path, tmp_path / "data", tmp_path, tmp_path / "invalid.json")


def test_partial_and_mismatched_pilots_are_rejected(tmp_path):
    c = load("350m")
    s = json.loads((ROOT / "configs/sweep_350m.json").read_text())
    job = sweep_jobs(c, s, tmp_path)[0]
    run = completed_fixture(tmp_path, job, 3.0)
    record = json.loads((run / "result.json").read_text())
    record["completed_steps"] -= 1
    (run / "result.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="incomplete"):
        validate_completed_job(run, job)
    record["completed_steps"] += 1
    (run / "result.json").write_text(json.dumps(record))
    meta = json.loads((run / "meta.json").read_text())
    meta["args"]["data_seed"] = 999
    (run / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="recipe"):
        validate_completed_job(run, job)


def prepare_local(monkeypatch, tmp_path, text):
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"text": text}), source)
    monkeypatch.setattr(data, "resolve_shards", lambda *a: ("fixture", ["sample/fixture.parquet"]))
    def copy(url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    monkeypatch.setattr(data, "download_file", copy)
    return tmp_path / "corpus"


def test_corpus_is_complete_and_document_disjoint(monkeypatch, tmp_path):
    out = prepare_local(monkeypatch, tmp_path, ["validation " * 20, "training " * 40])
    meta = data.build_fineweb_dataset(out, 30, 0, 10, threads=1)
    assert meta["complete"] and meta["validation_documents_disjoint"]
    assert np.load(out / meta["train_file"]).shape == (30,)
    assert np.load(out / meta["valid_file"]).shape == (10,)
    assert meta["source_revision"] == "fixture"
    assert not list(out.glob("*.incomplete"))
    assert data.build_fineweb_dataset(out, 30, 0, 10, threads=1) == meta


def test_exhausted_shards_never_publish_zero_filled_data(monkeypatch, tmp_path):
    out = prepare_local(monkeypatch, tmp_path, ["small document"])
    with pytest.raises(RuntimeError, match="exhausted"):
        data.build_fineweb_dataset(out, 100, 0, 10, threads=1)
    assert not (out / "fineweb_metadata.json").exists()
    assert not (out / data.train_filename(100)).exists()
    assert not (out / "fineweb_valid.npy").exists()


def test_published_token_filenames():
    assert data.train_filename(7_000_000_000) == "fineweb_train_7B.npy"
    assert data.train_filename(20_000_000_000) == "fineweb_train_20B.npy"
