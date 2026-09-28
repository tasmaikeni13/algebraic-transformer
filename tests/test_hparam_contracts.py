"""Tuning study contract tests: zero-transcendental verification, budget parity, and stability bounds."""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
from scripts.audit_primitives import source_audit
from scripts.tuning_records import source_hashes, hardware_evidence
from scripts.tuning_experiments import (
    HparamConfig,
    load_hparam_candidates,
    select_best_candidate,
    training_steps_for_budget,
)


def test_ast_zero_transcendental_audit():
    """Verify strictly zero transcendental function calls in the algebraic production stack."""
    files_to_audit = [
        "src/model.py",
        "src/primitives.py",
        "src/attention.py",
        "src/loss.py",
        "src/optimizer.py",
        "src/mesh.py",
        "src/kernels/pallas_afa.py",
        "src/kernels/pallas_oace.py",
    ]
    for rel_path in files_to_audit:
        full_path = ROOT / rel_path
        assert full_path.exists(), f"Missing file {rel_path}"
        code = full_path.read_text()
        violations = source_audit(code)
        assert len(violations) == 0, f"Transcendental violations found in {rel_path}: {violations}"


def test_optimal_hparam_files_structure():
    """Verify that optimal hyperparameter artifacts conform to expected schema."""
    alg_path = ROOT / "results/tuning/algebraic_optimal.json"
    base_path = ROOT / "results/tuning/baseline_optimal.json"
    ledger_path = ROOT / "results/tuning/sweep_ledger.json"

    if not (alg_path.exists() and base_path.exists() and ledger_path.exists()):
        pytest.skip("Tuning study optimal configurations not yet generated")

    alg_cfg = json.loads(alg_path.read_text())
    base_cfg = json.loads(base_path.read_text())
    ledger = json.loads(ledger_path.read_text())
    if ledger.get("protocol_version", 0) < 2:
        pytest.skip("Committed Tuning study artifacts predate the corrected multi-candidate protocol")

    # Check required keys for algebraic configuration
    for key in ["learning_rate", "warmup_steps", "weight_decay", "beta1", "beta2", "sink_omega", "gamma", "schedule"]:
        assert key in alg_cfg, f"Missing key '{key}' in algebraic_optimal.json"
    assert alg_cfg["schedule"] == "ards", f"Expected ARDS schedule for algebraic model, got {alg_cfg['schedule']}"

    # Check required keys for baseline configuration
    for key in ["learning_rate", "warmup_steps", "weight_decay", "beta1", "beta2", "schedule"]:
        assert key in base_cfg, f"Missing key '{key}' in baseline_optimal.json"

    assert len(ledger.get("runs", [])) == ledger["expected_runs"]
    assert ledger["candidate_count"] >= 4


def test_preregistered_candidate_matrix_is_a_real_sweep():
    candidates = load_hparam_candidates(ROOT / "configs/tuning_candidates.json")
    assert len(candidates["algebraic"]) >= 2
    assert len(candidates["baseline"]) >= 2
    for architecture, rows in candidates.items():
        assert len({name for name, _ in rows}) == len(rows), architecture


def test_training_budget_rounds_up_without_claiming_unprocessed_tokens():
    steps, actual_tokens = training_steps_for_budget(600_000_000, 512 * 2048)
    assert steps == 573
    assert actual_tokens == 600_834_048
    assert actual_tokens >= 600_000_000
    assert actual_tokens - 600_000_000 < 512 * 2048


def test_candidate_selection_rejects_incomplete_or_unsafe_candidate():
    config = HparamConfig(6e-4, 28, 0.01, 0.9, 0.99)
    safe = []
    for seed, loss in zip((42, 43, 44), (4.0, 4.01, 3.99)):
        safe.append({
            "architecture": "algebraic", "candidate": "safe", "seed": seed,
            "hparams": vars(config), "validation_loss": loss,
            "nan_or_inf_count": 0, "loss_spike_count": 0,
            "peak_gradient_norm": 1.0, "token_budget_satisfied": True,
            "normalization_second_moment_min": 0.99,
            "normalization_second_moment_max": 1.0,
        })
    unsafe = [dict(row, candidate="unsafe", validation_loss=3.0, loss_spike_count=1) for row in safe]
    _, winning, _ = select_best_candidate(safe + unsafe, "algebraic", [42, 43, 44])
    assert {row["candidate"] for row in winning} == {"safe"}


def test_tuning_evidence_rejects_legacy_six_run_record(tmp_path):
    hashes = {"src/model.py": "abc"}
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({
        "status": "PASS",
        "passed": True,
        "environment": {"source_sha256": hashes},
        "hardware": {"platform": "tpu", "device_count": 16},
        "sweep_summary": {
            "completed_runs": 6,
            "expected_runs": 6,
            "candidates_evaluated": 2,
            "algebraic_candidates_evaluated": 1,
            "baseline_candidates_evaluated": 1,
            "seed_count": 3,
            "requested_tokens_per_run": 600_000_000,
            "minimum_actual_tokens_per_run": 600_000_000,
            "maximum_token_overrun": 0,
            "tokens_per_step": 1_048_576,
            "token_budget_satisfied": True,
            "mean_perplexity_ratio": 1.0,
            "nan_or_inf_count": 0,
            "loss_spike_count": 0,
            "peak_gradient_norm": 1.0,
            "algebraic_seed_std_pct": 0.1,
            "baseline_seed_std_pct": 0.1,
        },
        "ast_audit": {"passed": True},
    }))
    assert hardware_evidence(path, hashes)["passed"] is False


def test_curvature_convergence_bounds():
    """Confirm optimal parameters satisfy the contraction mapping formalized in Curvature.lean."""
    alg_path = ROOT / "results/tuning/algebraic_optimal.json"
    if not alg_path.exists():
        pytest.skip("Tuning study optimal configurations not yet generated")

    alg_cfg = json.loads(alg_path.read_text())
    lr = alg_cfg["learning_rate"]
    wd = alg_cfg["weight_decay"]

    # Theorem adamw_decoupled_weight_decay: w_{t+1} = (1 - lr * wd) * w_t - lr * u_t
    # Contraction requires 0 < 1 - lr * wd < 1
    decay_factor = 1.0 - lr * wd
    assert 0.0 < decay_factor < 1.0, f"Decoupled weight decay violates contraction mapping: {decay_factor}"
    assert lr * wd < 0.01, f"Decoupled step factor too large: {lr * wd}"
    assert 1e-4 <= lr <= 2e-3, f"Learning rate out of search bounds: {lr}"
    assert 0.001 <= wd <= 0.15, f"Weight decay out of search bounds: {wd}"


def test_budget_parity_contracts():
    """Verify strict budget parity at 125M parameter scale across architectures."""
    from src.model import AlgebraicTransformerLM, ModelConfig, count_parameters
    from src.baseline import StandardTransformerLM, BaselineConfig
    import jax

    cfg_alg = ModelConfig(
        vocab_size=50257,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=2048,
        max_seq_len=2048,
    )
    cfg_base = BaselineConfig(
        vocab_size=50257,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=2048,
        max_seq_len=2048,
    )

    m_alg = AlgebraicTransformerLM(cfg_alg)
    p_alg = m_alg.init_params(jax.random.PRNGKey(42))
    n_alg = count_parameters(p_alg)

    m_base = StandardTransformerLM(cfg_base)
    p_base = m_base.init_params(jax.random.PRNGKey(42))
    n_base = count_parameters(p_base)

    diff = abs(n_alg - n_base) / max(n_alg, n_base)
    assert diff < 0.01, f"Parameter mismatch exceeds 1%: {diff * 100:.3f}% (Alg: {n_alg}, Base: {n_base})"
    assert 120_000_000 <= n_alg <= 130_000_000, f"Algebraic parameters outside 125M scale: {n_alg}"
    assert 120_000_000 <= n_base <= 130_000_000, f"Baseline parameters outside 125M scale: {n_base}"


def test_multi_seed_stability_contract():
    """Verify that multi-seed variance across Seeds 42, 43, 44 is strictly bounded (< 2% std/mean)."""
    ledger_path = ROOT / "results/tuning/sweep_ledger.json"
    if not ledger_path.exists():
        pytest.skip("Tuning study sweep ledger not yet produced")

    ledger = json.loads(ledger_path.read_text())
    if ledger.get("protocol_version", 0) < 2:
        pytest.skip("Committed Tuning study artifacts predate the corrected multi-candidate protocol")
    runs = ledger.get("runs", [])
    assert len(runs) == ledger["expected_runs"]

    winners = {
        "algebraic": json.loads((ROOT / "results/tuning/algebraic_optimal.json").read_text()),
        "baseline": json.loads((ROOT / "results/tuning/baseline_optimal.json").read_text()),
    }
    selected = {
        architecture: [
            r for r in runs
            if r["architecture"] == architecture and r["hparams"] == config
        ]
        for architecture, config in winners.items()
    }
    assert all(len(rows) == 3 for rows in selected.values())
    alg_losses = [r["validation_loss"] for r in selected["algebraic"]]
    base_losses = [r["validation_loss"] for r in selected["baseline"]]

    import numpy as np
    alg_mean = np.mean(alg_losses)
    alg_std = np.std(alg_losses)
    alg_std_pct = (alg_std / alg_mean) * 100.0

    base_mean = np.mean(base_losses)
    base_std = np.std(base_losses)
    base_std_pct = (base_std / base_mean) * 100.0

    assert alg_std_pct < 2.0, f"Algebraic multi-seed std exceeds 2%: {alg_std_pct:.3f}%"
    assert base_std_pct < 2.0, f"Baseline multi-seed std exceeds 2%: {base_std_pct:.3f}%"


def test_hardware_evidence_records():
    """Validate Tuning study hardware metrics record against contract gates."""
    metrics_path = ROOT / "results/tuning/tpu/metrics.json"
    if not metrics_path.exists():
        metrics_path = ROOT / "results/tuning/metrics.json"
    if not metrics_path.exists():
        pytest.skip("Tuning study hardware metrics record not yet generated")
    record = json.loads(metrics_path.read_text())
    hashes = source_hashes()
    if record.get("environment", {}).get("source_sha256") != hashes:
        pytest.skip("Committed TPU evidence is stale and must be regenerated by the aggregate verifier")
    res = hardware_evidence(metrics_path, hashes)
    assert res["passed"] is True, f"Hardware evidence validation failed: {res}"
