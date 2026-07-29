"""Tests for the unified merge / provenance-audit step (Task 11, redesigned)."""
from __future__ import annotations

import json

from smco.experiment_manifests import (
    build_algorithm_config,
    build_baseline_task,
    build_manifest,
    build_task,
    derive_seed,
    freeze_manifest,
    validate_result_against_task,
)
from smco.merge_results import (
    baseline_row_from_outcome,
    build_task_index,
    classify_task,
    smco_row_from_outcome,
)
from smco.paper_contract import (
    NONE_TOKEN,
    RESULT_COLUMNS,
    validate_result_row,
)


def _evo_task():
    cfg = build_algorithm_config(
        "python", "smco", True, "state_preserving",
        evolution_strategy="rand1bin", evolution_points=(0.5, 0.75),
        elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=8,
    )
    return build_task(
        "e1_development", "synthetic_highdim", "Zakharov", 200, 0, 0,
        config=cfg, fe_budget=20000, checkpoints=(5000, 10000), seed=12345,
        instance_hash="ihash", start_points_hash="shash",
    )


def _baseline_task():
    seed = derive_seed("e3_baselines_highdim", "synthetic_highdim", "Zakharov", 200, 0, 0, "DE")
    return build_baseline_task(
        "e3_baselines_highdim", "synthetic_highdim", "Zakharov",
        200, 0, algorithm="DE", fe_budget=20000, checkpoints=(5000, 10000), seed=seed,
        instance_hash="ihash", start_points_hash="shash",
    )


def _smco_outcome(task):
    return {
        "run_id": task["run_id"], "status": "success", "failure_reason": "none",
        "fe_used": 19998, "fe_budget": 20000, "best_value": 1e-6, "known_optimum": 0.0,
        "normalized_gap": 0.001,
        "target_hit_fe": {"1e-1": 500, "1e-2": 5000, "1e-3": None, "1e-5": None},
        "anytime": [], "best_so_far_trace": [[500, 0.1]],
        "termination_reason": "evaluation_budget",
        "fe_counts_by_event": {"initialization": 1}, "wall_time_sec": 1.0,
        "peak_memory_mb": 10.0, "machine_id": "h", "git_commit": "abc",
        "environment_hash": "env", "task": task,
        "algorithm_id": task["algorithm_id"], "supersedes_run_id": "none",
    }


def test_classify_task():
    assert classify_task(_evo_task()) == "smco"
    assert classify_task(_baseline_task()) == "baseline"


def test_build_task_index_loads_all_manifests(tmp_path):
    m1 = freeze_manifest(build_manifest("e1_development", "synthetic_highdim", [_evo_task()]))
    m2 = freeze_manifest(build_manifest("e3_baselines_highdim", "synthetic_highdim", [_baseline_task()]))
    p1 = tmp_path / "m1.json"; p1.write_text(json.dumps(m1))
    p2 = tmp_path / "m2.json"; p2.write_text(json.dumps(m2))
    idx = build_task_index([p1, p2])
    assert set(idx) == {_evo_task()["run_id"], _baseline_task()["run_id"]}


def test_smco_row_from_outcome_is_contract_valid_and_consistent():
    task = _evo_task()
    row = smco_row_from_outcome(_smco_outcome(task), task, manifest_id="m")
    assert set(row) == set(RESULT_COLUMNS)
    assert validate_result_row(row) == []
    assert validate_result_against_task(row, task) == []
    assert row["target_hit_fe_1e-3"] == NONE_TOKEN  # null -> NONE_TOKEN
    assert row["target_hit_fe_1e-1"] == 500


def test_smco_row_tolerates_null_best_value():
    task = _evo_task()
    oc = _smco_outcome(task); oc["best_value"] = None; oc["status"] = "infra_failure"
    row = smco_row_from_outcome(oc, task)
    assert row["status"] == "infra_failure"
    import math
    assert math.isnan(row["best_value"])


def test_baseline_row_from_outcome_has_columns_and_algorithm():
    task = _baseline_task()
    oc = {
        "run_id": task["run_id"], "status": "success", "failure_reason": "none",
        "fe_used": 20000, "best_value": 0.5, "known_optimum": 0.0, "normalized_gap": 0.5,
        "target_hit_fe": {"1e-1": 100, "1e-2": None, "1e-3": None, "1e-5": None},
        "anytime": [], "best_so_far_trace": [], "termination_reason": "evaluation_budget",
        "fe_counts_by_event": {}, "wall_time_sec": 2.0, "peak_memory_mb": None,
        "machine_id": "h", "git_commit": "", "environment_hash": "env",
        "supersedes_run_id": "none",
    }
    row = baseline_row_from_outcome(oc, task, manifest_id="m")
    assert set(row) == set(RESULT_COLUMNS)
    assert row["algorithm_id"] == "DE"
    assert row["family"] == NONE_TOKEN
    assert row["configuration_hash"] == NONE_TOKEN
    assert row["is_confirmatory"] is True  # e3 is confirmatory
