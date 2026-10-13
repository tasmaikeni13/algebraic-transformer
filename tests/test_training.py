"""Real trainer checkpoint/resume checks on a small CPU configuration."""

import json
from pathlib import Path

import numpy as np
import pytest

from algebraic.train import main


@pytest.mark.parametrize("preset", ["standard", "algebraic"])
def test_resume_reproduces_weights_and_rejects_recipe_changes(tmp_path, preset):
    rng = np.random.default_rng(42)
    for name in ("train", "valid"):
        np.save(tmp_path / f"{name}.npy", rng.integers(0, 97, 512, dtype=np.uint16))
    run = tmp_path / "run"
    args = ["--preset", preset, "--size", "350m", "--steps", "2", "--seq-len", "16",
            "--batch", "2", "--micro", "1", "--eval-batch", "1", "--eval-tokens", "32",
            "--train", str(tmp_path / "train.npy"), "--valid", str(tmp_path / "valid.npy"),
            "--set", "d_model=64", "--set", "n_layers=1", "--set", "n_heads=1", "--set", "d_ff=96",
            "--set", "vocab_size=97", "--set", "dtype=float32", "--out", str(run),
            "--ckpt-every", "1", "--log-every", "1", "--save-params"]
    assert main(args) == 0
    with np.load(run / "params.npz") as archive:
        expected = {k: archive[k].copy() for k in archive.files}
    assert main([*args, "--resume"]) == 0
    with np.load(run / "params.npz") as archive:
        for key, value in expected.items():
            np.testing.assert_array_equal(archive[key], value)
    assert json.loads((run / "result.json").read_text())["completed_steps"] == 2
    with pytest.raises(ValueError, match="recipe"):
        main([*args, "--resume", "--lr", "0.007"])
