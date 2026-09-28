"""Tuning study evidence fingerprints, including the inherited dependency closure."""
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts.pilot_records import source_hashes as pilot_hashes, environment as pilot_environment

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    """Preserve NumPy scalar values as JSON scalars; reject NaN and infinity."""
    def scalar(item):
        if isinstance(item, (np.generic, np.ndarray)):
            if isinstance(item, np.ndarray) and item.ndim == 0:
                return item.item()
            elif isinstance(item, np.ndarray):
                return item.tolist()
            return item.item()
        raise TypeError(f"Unsupported evidence value: {type(item).__name__}")

    encoded = json.dumps(value, indent=2, allow_nan=False, default=scalar) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def source_hashes():
    result = pilot_hashes()
    names = [
        'src/model.py',
        'src/baseline.py',
        'src/mesh.py',
        'src/dataset.py',
        'src/optimizer.py',
        'formal/AlgebraicTheory/Curvature.lean',
        'configs/tuning_candidates.json',
        'tests/test_hparam_contracts.py',
    ]
    names += [
        str(p.relative_to(ROOT))
        for pattern in ('tuning_*.py', '*tuning*.py', 'run_hparam_sweep.py', 'prepare_fineweb_edu.py')
        for p in (ROOT / 'scripts').glob(pattern)
    ]
    for name in sorted(set(names)):
        p = ROOT / name
        if p.exists():
            result[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def environment():
    env = pilot_environment()
    env['source_sha256'] = source_hashes()
    return env


def hardware_evidence(path, expected_hashes):
    """Validate that Tuning study evidence exercised the real hyperparameter sweep on TPU."""
    path = Path(path)
    if not path.exists():
        return {"passed": False, "reason": "Missing TPU evidence"}
    record = json.loads(path.read_text())
    matched = record.get("environment", {}).get("source_sha256") == expected_hashes

    sweep = record.get("sweep_summary", {})
    completed_runs = sweep.get("completed_runs", 0)
    expected_runs = sweep.get("expected_runs", 0)
    candidate_count = sweep.get("candidates_evaluated", 0)
    algebraic_candidates = sweep.get("algebraic_candidates_evaluated", 0)
    baseline_candidates = sweep.get("baseline_candidates_evaluated", 0)
    seed_count = sweep.get("seed_count", 0)
    requested_tokens = sweep.get("requested_tokens_per_run", 0)
    minimum_actual_tokens = sweep.get("minimum_actual_tokens_per_run", 0)
    maximum_token_overrun = sweep.get("maximum_token_overrun", -1)
    tokens_per_step = sweep.get("tokens_per_step", 0)
    ppl_ratio = sweep.get("mean_perplexity_ratio", 999.0)
    nan_count = sweep.get("nan_or_inf_count", 999)
    spike_count = sweep.get("loss_spike_count", 999)
    peak_grad_norm = sweep.get("peak_gradient_norm", 999.0)
    alg_std_pct = sweep.get("algebraic_seed_std_pct", 999.0)
    base_std_pct = sweep.get("baseline_seed_std_pct", 999.0)
    normalization_passed = sweep.get("normalization_second_moment_passed", False)
    hardware = record.get("hardware", {})
    candidate_matrix = json.loads(
        (ROOT / "configs/tuning_candidates.json").read_text()
    )
    expected_candidate_counts = {
        architecture: len(rows)
        for architecture, rows in candidate_matrix.items()
    }
    expected_run_keys = {
        (architecture, candidate["name"], seed)
        for architecture, candidates in candidate_matrix.items()
        for candidate in candidates
        for seed in (42, 43, 44)
    }
    artifacts = record.get("artifacts", {})
    artifact_names = (
        "sweep_ledger.json",
        "algebraic_optimal.json",
        "baseline_optimal.json",
    )
    artifact_files_match = all(
        isinstance(artifacts.get(name), str)
        and (path.parent / name).exists()
        and hashlib.sha256((path.parent / name).read_bytes()).hexdigest()
        == artifacts[name]
        for name in artifact_names
    )
    ledger_consistent = False
    if artifact_files_match:
        ledger = json.loads((path.parent / "sweep_ledger.json").read_text())
        runs = ledger.get("runs", [])
        selected = {
            "algebraic": (
                sweep.get("selected_algebraic_candidate"),
                json.loads((path.parent / "algebraic_optimal.json").read_text()),
            ),
            "baseline": (
                sweep.get("selected_baseline_candidate"),
                json.loads((path.parent / "baseline_optimal.json").read_text()),
            ),
        }
        winner_rows = {
            architecture: [
                run for run in runs
                if run.get("architecture") == architecture
                and run.get("candidate") == candidate
                and run.get("hparams") == config
            ]
            for architecture, (candidate, config) in selected.items()
        }
        winners_match = all(
            len(rows) == 3 and {run.get("seed") for run in rows} == {42, 43, 44}
            for rows in winner_rows.values()
        )
        selected_safe = winners_match and all(
            np.isfinite(run.get("validation_loss", np.nan))
            and np.isfinite(run.get("validation_perplexity", np.nan))
            and run.get("nan_or_inf_count") == 0
            and run.get("loss_spike_count") == 0
            and run.get("peak_gradient_norm", 999.0) <= 5.0
            and (
                architecture != "algebraic"
                or (
                    run.get("normalization_second_moment_min", 0.0) >= 0.8
                    and run.get("normalization_second_moment_max", 999.0) <= 1.3
                )
            )
            for architecture, rows in winner_rows.items()
            for run in rows
        )
        unique_runs = {
            (run.get("architecture"), run.get("candidate"), run.get("seed"))
            for run in runs
        }
        candidate_configs_match = all(
            run.get("hparams")
            == {key: value for key, value in candidate.items() if key != "name"}
            for architecture, candidates in candidate_matrix.items()
            for candidate in candidates
            for run in runs
            if run.get("architecture") == architecture
            and run.get("candidate") == candidate["name"]
        )
        ledger_consistent = bool(
            ledger.get("protocol_version") == 2
            and ledger.get("expected_runs") == expected_runs
            and len(runs) == expected_runs
            and len(unique_runs) == expected_runs
            and unique_runs == expected_run_keys
            and ledger.get("candidate_count") == candidate_count
            and candidate_configs_match
            and winners_match
            and selected_safe
            and all(
                run.get("token_budget_satisfied") is True
                and run.get("total_tokens", 0) >= requested_tokens
                and run.get("requested_tokens", 0) >= 600_000_000
                and 0 <= run.get("token_overrun", -1) < tokens_per_step
                for run in runs
            )
        )

    passed = (
        matched
        and record.get("passed") is True
        and record.get("status") == "PASS"
        and artifact_files_match
        and ledger_consistent
        and completed_runs == expected_runs
        and expected_runs == candidate_count * seed_count
        and algebraic_candidates == expected_candidate_counts["algebraic"]
        and baseline_candidates == expected_candidate_counts["baseline"]
        and candidate_count == algebraic_candidates + baseline_candidates
        and seed_count == 3
        and requested_tokens >= 600_000_000
        and minimum_actual_tokens >= requested_tokens
        and 0 <= maximum_token_overrun < tokens_per_step
        and sweep.get("token_budget_satisfied") is True
        and hardware.get("platform") == "tpu"
        and hardware.get("device_count") == 16
        and hardware.get("process_count") == 4
        and ppl_ratio <= 1.08
        and nan_count == 0
        and spike_count == 0
        and peak_grad_norm <= 5.0
        and alg_std_pct < 2.0
        and base_std_pct < 2.0
        and normalization_passed is True
        and record.get("ast_audit", {}).get("passed", False)
    )

    return {
        "passed": passed,
        "matched_hashes": matched,
        "completed_runs": completed_runs,
        "expected_runs": expected_runs,
        "candidate_count": candidate_count,
        "algebraic_candidates": algebraic_candidates,
        "baseline_candidates": baseline_candidates,
        "seed_count": seed_count,
        "requested_tokens_per_run": requested_tokens,
        "minimum_actual_tokens_per_run": minimum_actual_tokens,
        "maximum_token_overrun": maximum_token_overrun,
        "platform": hardware.get("platform"),
        "device_count": hardware.get("device_count"),
        "process_count": hardware.get("process_count"),
        "perplexity_ratio": ppl_ratio,
        "nan_count": nan_count,
        "spike_count": spike_count,
        "peak_gradient_norm": peak_grad_norm,
        "algebraic_seed_std_pct": alg_std_pct,
        "baseline_seed_std_pct": base_std_pct,
        "normalization_second_moment_passed": normalization_passed,
        "artifact_files_match": artifact_files_match,
        "ledger_consistent": ledger_consistent,
    }
