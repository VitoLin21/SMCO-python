"""Tests for the unified merge / provenance-audit step (Task 11, redesigned)."""
from __future__ import annotations

import csv
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
    _identity_key,
    audit_payloads,
    baseline_row_from_outcome,
    build_task_index,
    classify_task,
    merge,
    resolve_supersedes,
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


def _e2_task():
    cfg = build_algorithm_config(
        "python", "smco", True, "state_preserving",
        evolution_strategy="rand1bin", evolution_points=(0.5, 0.75),
        elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=8,
    )
    seed = derive_seed("e2_factorial_highdim", "synthetic_highdim", "Zakharov", 200, 0, 0, cfg["algorithm_id"])
    return build_task(
        "e2_factorial_highdim", "synthetic_highdim", "Zakharov", 200, 0, 0,
        config=cfg, fe_budget=20000, checkpoints=(5000, 10000), seed=seed,
        instance_hash="ihash", start_points_hash="shash",
    )


def _row(run_id, **kw):
    base = {"function": "Zakharov", "dimension": 200, "instance": 0,
            "algorithm_id": "PY-SP-SMCO-EVO", "language": "python",
            "state_semantics": "state_preserving", "evolution_strategy": "rand1bin",
            "seed": 1, "n_starts": 8, "run_id": run_id, "stage": "e2_factorial_highdim",
            "suite": "synthetic_highdim", "fe_budget": 1000, "fe_used": 999,
            "objective_sense": "minimize", "best_value": 1e-6, "known_optimum": 0.0,
            "normalized_gap": 0.01, "family": "smco", "evolutionary": "true",
            "configuration_hash": "cfg", "start_points_hash": "sh",
            "instance_hash": "ih", "supersedes_run_id": "none", "status": "success"}
    base.update(kw)
    return base


def test_resolve_supersedes_excludes_superseded():
    rows = [_row("r1"), _row("r2", supersedes_run_id="r1")]
    valid, superseded = resolve_supersedes(rows)
    assert [r["run_id"] for r in valid] == ["r2"]
    assert superseded == {"r1"}


def test_identity_key_detects_duplicate_identity():
    a = _row("r1"); b = _row("r2")  # same identity, different run_id
    assert _identity_key(a) == _identity_key(b)


def test_audit_passes_clean_rows():
    task = _evo_task()
    row = smco_row_from_outcome(_smco_outcome(task), task)
    audit = audit_payloads([row], {task["run_id"]: task})
    assert audit["passed"] is True, audit


def test_audit_flags_fe_over_budget():
    task = _evo_task()
    row = smco_row_from_outcome(_smco_outcome(task), task)
    row["fe_used"] = task["fe_budget"] + 1
    audit = audit_payloads([row], {task["run_id"]: task})
    assert audit["passed"] is False
    assert any("budget" in c["name"] for c in audit["checks"])


def test_audit_flags_wrong_seed():
    task = _e2_task()  # confirmatory stage -> seed is audited
    row = smco_row_from_outcome(_smco_outcome(task), task)
    row["seed"] = task["seed"] + 1
    audit = audit_payloads([row], {task["run_id"]: task})
    assert audit["passed"] is False
    assert any("seed" in c["name"] for c in audit["checks"])


def test_audit_flags_duplicate_identity():
    task = _evo_task()
    row = smco_row_from_outcome(_smco_outcome(task), task)
    dup = dict(row); dup["run_id"] = "r_other"
    audit = audit_payloads([row, dup], {task["run_id"]: task, "r_other": task})
    assert audit["passed"] is False
    assert any("duplicate" in c["name"] for c in audit["checks"])


def _write(raw_dir, run_id, payload):
    (raw_dir / f"{run_id}.json").write_text(json.dumps(payload))


def test_merge_end_to_end_writes_all_artefacts(tmp_path):
    task = _evo_task()
    btask = _baseline_task()
    manifest = freeze_manifest(build_manifest("e1_development", "synthetic_highdim", [task]))
    bmanifest = freeze_manifest(build_manifest("e3_baselines_highdim", "synthetic_highdim", [btask]))
    mp = tmp_path / "m.json"; mp.write_text(json.dumps(manifest))
    bp = tmp_path / "bm.json"; bp.write_text(json.dumps(bmanifest))
    raw = tmp_path / "raw"; raw.mkdir()
    _write(raw, task["run_id"], _smco_outcome(task))
    boc = {"run_id": btask["run_id"], "status": "success", "failure_reason": "none",
        "fe_used": 20000, "fe_budget": 20000, "best_value": 0.4, "known_optimum": 0.0,
        "normalized_gap": 0.4, "target_hit_fe": {"1e-1": 100, "1e-2": None, "1e-3": None, "1e-5": None},
        "anytime": [], "best_so_far_trace": [], "termination_reason": "evaluation_budget",
        "fe_counts_by_event": {}, "wall_time_sec": 1.0, "peak_memory_mb": None,
        "machine_id": "h", "git_commit": "", "environment_hash": "env", "task": btask,
        "algorithm_id": "DE", "supersedes_run_id": "none"}
    _write(raw, btask["run_id"], boc)

    merged = tmp_path / "merged"
    summary = merge([mp, bp], [raw], merged)

    all_rows = list(csv.DictReader(open(merged / "all_attempts.csv")))
    valid = list(csv.DictReader(open(merged / "valid_runs.csv")))
    missing = list(csv.DictReader(open(merged / "missing_runs.csv")))
    assert len(all_rows) == 2
    assert len(valid) == 2
    assert {r["algorithm_id"] for r in valid} == {task["algorithm_id"], "DE"}
    assert summary["audit"]["passed"] is True
    assert merged.joinpath("provenance_audit.json").exists()
    assert merged.joinpath("provenance_audit.md").exists()
    assert merged.joinpath("anytime.csv").exists()


def test_merge_reports_missing_runs(tmp_path):
    task = _evo_task()
    manifest = freeze_manifest(build_manifest("e1_development", "synthetic_highdim", [task]))
    mp = tmp_path / "m.json"; mp.write_text(json.dumps(manifest))
    raw = tmp_path / "raw"; raw.mkdir()
    merge([mp], [raw], tmp_path / "merged")
    missing = list(csv.DictReader(open(tmp_path / "merged" / "missing_runs.csv")))
    assert len(missing) == 1
    assert missing[0]["run_id"] == task["run_id"]


def test_identity_key_distinguishes_n_starts():
    a = _row("r1"); b = _row("r2", n_starts=16)
    assert _identity_key(a) != _identity_key(b)
