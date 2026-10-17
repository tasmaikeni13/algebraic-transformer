"""Experiment budgets, real preset shapes, and corpus integrity."""

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
from scripts.run_comparison import ROOT, build_plan, parameter_count, read_selection, training_jobs


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
