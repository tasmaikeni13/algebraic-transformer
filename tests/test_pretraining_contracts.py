"""Static and small-data contracts for the Pretraining study pipeline."""

import json

from scripts.tuning_experiments import training_steps_for_budget
from scripts.pretraining_experiments import PRETRAINING_TOKEN_BUDGET, summarize_pretraining
from scripts.pretraining_records import run_evidence
from scripts.prepare_eval_benchmarks import (
    DATASETS,
    normalize_arc,
    normalize_hellaswag,
    normalize_lambada,
)


def _records():
    return [
        {
            "architecture": architecture,
            "seed": seed,
            "total_tokens": 2_500_853_760,
            "validation_loss": 4.0,
            "validation_perplexity": 60.0 if architecture == "algebraic" else 61.0,
            "nan_or_inf_count": 0,
            "loss_spike_count": 0,
            "peak_gradient_norm": 1.0,
            "normalization_second_moment_min": 0.99,
            "normalization_second_moment_max": 1.0,
        }
        for architecture in ("algebraic", "baseline")
        for seed in (42, 43, 44)
    ]


def test_pretraining_budget_rounds_up():
    steps, actual = training_steps_for_budget(PRETRAINING_TOKEN_BUDGET, 512 * 2048)
    assert steps == 2385
    assert actual == 2_500_853_760


def test_pretraining_summary_requires_all_paired_runs():
    complete = summarize_pretraining(_records())
    assert complete["pretraining_safety_passed"]
    assert complete["perplexity_parity_passed"]
    incomplete = summarize_pretraining(_records()[:-1])
    assert not incomplete["pretraining_safety_passed"]


def test_pretraining_run_evidence_rejects_cpu_or_short_run(tmp_path):
    path = tmp_path / "run_metrics.json"
    hashes = {"x": "y"}
    path.write_text(json.dumps({
        "passed": True,
        "environment": {"source_sha256": hashes},
        "hardware": {"platform": "cpu", "device_count": 1},
        "run": {
            "architecture": "algebraic", "seed": 42, "total_tokens": 100,
            "nan_or_inf_count": 0, "loss_spike_count": 0, "peak_gradient_norm": 1.0,
        },
    }))
    assert not run_evidence(path, hashes)["passed"]


def test_benchmark_normalization_contracts():
    arc = normalize_arc({
        "question": "Which?", "choices": {"label": ["A", "B"], "text": ["one", "two"]},
        "answerKey": "B",
    })
    assert arc["answer"] == 1
    lambada = normalize_lambada({"text": "a context target"})
    assert lambada == {"context": "a context", "target": " target"}
    hellaswag = normalize_hellaswag({"ctx": "context", "endings": ["next"], "label": "0"})
    assert hellaswag["choices"] == [" next"]
    assert DATASETS["piqa"][1] == "plain_text"
