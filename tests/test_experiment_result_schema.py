"""Tests for result-row schema and manifest/result consistency (Task 7).

``paper_contract.validate_result_row`` is the schema authority; here we add a
helper that derives a well-formed result row from a manifest task and a
consistency check that the row's identity fields match the task that produced
it (run_id / configuration_hash / algorithm_id / budget).
"""

from __future__ import annotations

import pytest

from smco.experiment_manifests import (
    build_algorithm_config,
    build_task,
    result_row_from_task,
    validate_result_against_task,
)
from smco.paper_contract import NONE_TOKEN, RESULT_COLUMNS, validate_result_row


def _evo_task():
    cfg = build_algorithm_config(
        language="python", family="smco", evolutionary=True,
        state_semantics="state_preserving", evolution_strategy="rand1bin",
        evolution_points=(0.5, 0.75), elimination_rate=0.25, de_factor=0.8,
        de_crossover=0.7, n_starts=8,
    )
    return build_task(
        "e1_development", "synthetic_highdim", "Rastrigin", 200, 0, 0,
        config=cfg, fe_budget=200000, checkpoints=(20000, 50000, 100000, 200000),
        seed=12345, instance_hash="ihash", start_points_hash="shash",
    )


def _base_task():
    cfg = build_algorithm_config(
        language="r", family="smco_boost_refine", evolutionary=False,
        state_semantics="none", evolution_strategy="none", evolution_points=(),
        elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=8,
    )
    return build_task(
        "e2_factorial_highdim", "synthetic_highdim", "Rosenbrock", 1000, 2, 0,
        config=cfg, fe_budget=2000000, checkpoints=(100000, 500000), seed=77,
    )


# --------------------------- schema validation ------------------------------
def test_valid_evo_result_row_passes_validation():
    row = result_row_from_task(_evo_task(), best_value=1e-6, fe_used=199998, status="success")
    assert validate_result_row(row) == []


def test_valid_base_result_row_passes_validation():
    row = result_row_from_task(_base_task(), best_value=0.3, fe_used=2000000, status="success")
    assert validate_result_row(row) == []


def test_result_row_has_every_contract_column():
    row = result_row_from_task(_evo_task(), best_value=1e-6, fe_used=199998, status="success")
    assert set(row.keys()) == set(RESULT_COLUMNS)


def test_missing_column_detected():
    row = result_row_from_task(_evo_task(), best_value=1e-6, fe_used=199998, status="success")
    del row["run_id"]
    errors = validate_result_row(row)
    assert errors and any("run_id" in e for e in errors)


def test_bad_language_detected():
    row = result_row_from_task(_evo_task(), best_value=1e-6, fe_used=199998, status="success")
    row["language"] = "julia"
    assert validate_result_row(row) != []


def test_bad_status_detected():
    row = result_row_from_task(_evo_task(), best_value=1e-6, fe_used=199998, status="success")
    row["status"] = "great"
    assert validate_result_row(row) != []


def test_algorithm_id_mismatch_detected():
    row = result_row_from_task(_evo_task(), best_value=1e-6, fe_used=199998, status="success")
    row["algorithm_id"] = "PY-RS-SMCO-EVO"  # semantically inconsistent with the row's semantics
    assert validate_result_row(row) != []


def test_base_row_with_nonnone_semantics_detected():
    row = result_row_from_task(_base_task(), best_value=0.3, fe_used=2000000, status="success")
    row["state_semantics"] = "state_preserving"
    assert validate_result_row(row) != []


# ----------------------- manifest/result consistency ------------------------
def test_result_row_consistent_with_task():
    task = _evo_task()
    row = result_row_from_task(task, best_value=1e-6, fe_used=199998, status="success")
    assert validate_result_against_task(row, task) == []


def test_inconsistent_run_id_detected():
    task = _evo_task()
    row = result_row_from_task(task, best_value=1e-6, fe_used=199998, status="success")
    row["run_id"] = "rdeadbeefdeadbeef"
    assert validate_result_against_task(row, task) != []


def test_inconsistent_fe_budget_detected():
    task = _evo_task()
    row = result_row_from_task(task, best_value=1e-6, fe_used=199998, status="success")
    row["fe_budget"] = 1
    assert validate_result_against_task(row, task) != []


def test_infra_failure_carries_failure_reason():
    task = _evo_task()
    row = result_row_from_task(
        task, best_value=1.0, fe_used=1200, status="infra_failure",
        failure_reason="node_lost",
    )
    # schema still valid; consistency with task unaffected by outcome.
    assert validate_result_row(row) == []
    assert validate_result_against_task(row, task) == []
    assert row["failure_reason"] == "node_lost"
