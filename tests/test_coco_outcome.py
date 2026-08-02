"""Tests for the COCO external-validation task-level outcome (review §9 P3)."""
from __future__ import annotations

import json

import numpy as np

import pytest

from smco.coco_outcome import (
    COCO_SUITES,
    build_coco_outcome,
    coco_benchmark_provenance,
    coco_outcome_errors,
    derive_gap_and_targets,
)


def _task(**kw):
    base = {"run_id": "rE4_001", "function": "f1", "dimension": 160, "instance": 0}
    base.update(kw)
    return base


def test_derive_gap_and_targets_relative_convention():
    # f_opt=0, initial_ref=1.0 -> gap_span=1.0; targets f_opt+tau*1.0 = tau.
    # trace reaches 0.1 at FE 100, 0.01 at FE 1000, never 1e-3.
    trace = [(1, 1.0), (100, 0.1), (1000, 0.01), (10000, 0.01)]
    gap, hits = derive_gap_and_targets(trace, f_opt=0.0, initial_ref=1.0, fe_budget=10000)
    assert abs(gap - 0.01) < 1e-9            # final best 0.01 / span 1.0
    assert hits["1e-1"] == 100               # reached at FE 100
    assert hits["1e-2"] == 1000
    assert hits["1e-3"] is None              # never reached -> censored
    assert hits["1e-5"] is None


def test_build_coco_outcome_preserves_native_and_derives_synthetic():
    trace = [(1, 100.0), (500, 12.0), (5000, 1.5)]  # f_opt=0, initial_ref=100
    task = _task()
    payload = build_coco_outcome(
        task, best_observed_fvalue1=1.5, evaluations=5000, final_target_hit=False,
        best_trace=trace, f_opt=0.0, initial_ref=100.0, fe_budget=160000,
        suite="bbob-largescale", problem_id="bbob-largescale_f001_i1_d160",
        cocoex_version="3.4.0", cocopp_version="2.7.1",
        machine_id="node213", git_commit="c" * 40, environment_hash="eh")
    # COCO-native fields preserved verbatim
    bench = payload["benchmark"]
    assert bench["best_observed_fvalue1"] == 1.5
    assert bench["evaluations"] == 5000
    assert bench["f_opt"] == 0.0
    assert bench["problem_id"] == "bbob-largescale_f001_i1_d160"
    assert bench["cocoex_version"] == "3.4.0" and bench["cocopp_version"] == "2.7.1"
    assert payload["final_target_hit"] is False
    # synthetic-style derived fields present (flow into merge/audit/analysis)
    assert payload["normalized_gap"] is not None
    assert "1e-2" in payload["target_hit_fe"]
    assert isinstance(payload["anytime"], list) and len(payload["anytime"]) == 3
    # NO synthetic instance fields
    assert "instance_artifact_dir" not in payload
    assert payload["task"]["run_id"] == "rE4_001"


def test_coco_outcome_errors_rejects_missing_provenance():
    task = _task()
    good = build_coco_outcome(
        task, best_observed_fvalue1=1.5, evaluations=5000, final_target_hit=True,
        best_trace=[(1, 100.0), (5000, 1.5)], f_opt=0.0, initial_ref=100.0,
        fe_budget=160000, suite="bbob-largescale", problem_id="p1",
        cocoex_version="3.4.0", cocopp_version=None,
        machine_id="n", git_commit="c" * 40, environment_hash="eh")
    assert coco_outcome_errors(good) == []
    # drop the benchmark block -> rejected
    nobench = dict(good); nobench.pop("benchmark", None)
    assert coco_outcome_errors(nobench)
    # wrong suite -> rejected
    badsuite = dict(good); badsuite["benchmark"] = dict(good["benchmark"], suite="synthetic_highdim")
    assert any("not in" in e for e in coco_outcome_errors(badsuite))
    # empty problem_id -> rejected
    badpid = dict(good); badpid["benchmark"] = dict(good["benchmark"], problem_id="")
    assert any("problem_id" in e for e in coco_outcome_errors(badpid))
    # missing code provenance -> rejected
    noprov = dict(good); noprov["git_commit"] = ""
    assert any("provenance" in e for e in coco_outcome_errors(noprov))


def test_coco_benchmark_provenance_recognises_block():
    task = _task()
    payload = build_coco_outcome(
        task, best_observed_fvalue1=1.0, evaluations=10, final_target_hit=True,
        best_trace=[(1, 1.0)], f_opt=0.0, initial_ref=1.0, fe_budget=100,
        suite="bbob", problem_id="p", cocoex_version="x", cocopp_version=None,
        machine_id="n", git_commit="c" * 40, environment_hash="eh")
    assert coco_benchmark_provenance(payload) is payload["benchmark"]
    assert coco_benchmark_provenance({"benchmark": {"kind": "synthetic"}}) is None
    assert "bbob-largescale" in COCO_SUITES and "bbob" in COCO_SUITES


# --- audit integration (review P3): COCO rows validated via benchmark_provenance ---

def _e4_task(run_id="rE4_001"):
    from smco.experiment_manifests import derive_seed
    return {
        "schema_version": "1", "manifest_id": "e4_bbob_largescale__bbob-largescale",
        "stage": "e4_bbob_largescale", "suite": "bbob-largescale",
        "function": "f1", "dimension": 160, "instance": 0, "replication": 0,
        "seed": derive_seed("e4_bbob_largescale", "bbob-largescale", "f1", 160, 0, 0,
                            "PY-SP-SMCO-EVO"),
        "language": "python", "state_semantics": "state_preserving",
        "family": "smco", "evolutionary": "true", "evolution_strategy": "rand1bin",
        "algorithm_id": "PY-SP-SMCO-EVO", "n_starts": 8, "fe_budget": 160000,
        "configuration_hash": "cfg_e4", "run_id": run_id,
        "start_points_hash": None, "instance_hash": None,
    }


def _e4_outcome(task):
    return build_coco_outcome(
        task, best_observed_fvalue1=1.5, evaluations=160000, final_target_hit=False,
        best_trace=[(1, 100.0), (80000, 12.0), (160000, 1.5)], f_opt=0.0,
        initial_ref=100.0, fe_budget=160000, suite="bbob-largescale",
        problem_id="bbob-largescale_f001_i1_d160", cocoex_version="3.4.0",
        cocopp_version="2.7.1", machine_id="node213",
        git_commit="c" * 40, environment_hash="eh")


def test_audit_recognises_coco_and_validates_benchmark_provenance():
    from smco.merge_results import audit_payloads, smco_row_from_outcome
    task = _e4_task()
    outcome = _e4_outcome(task)
    row = smco_row_from_outcome(outcome, task, manifest_id=task["manifest_id"])
    audit = audit_payloads([row], {task["run_id"]: task},
                           outcome_index={task["run_id"]: outcome})
    names = [c["name"] for c in audit["checks"]]
    assert "benchmark_provenance" in names          # COCO check present
    bp = next(c for c in audit["checks"] if c["name"] == "benchmark_provenance")
    assert bp["passed"] is True                      # valid COCO provenance
    assert bp["n"] == 1                              # only COCO rows
    # start_points_hash check did NOT inspect the COCO row (no synthetic instance)
    assert all(c["passed"] for c in audit["checks"])
    assert audit["passed"] is True


def test_audit_rejects_coco_with_broken_benchmark_provenance():
    from smco.merge_results import audit_payloads, smco_row_from_outcome
    task = _e4_task()
    outcome = _e4_outcome(task)
    outcome["benchmark"]["problem_id"] = ""          # break provenance
    row = smco_row_from_outcome(outcome, task, manifest_id=task["manifest_id"])
    audit = audit_payloads([row], {task["run_id"]: task},
                           outcome_index={task["run_id"]: outcome})
    bp = next(c for c in audit["checks"] if c["name"] == "benchmark_provenance")
    assert bp["passed"] is False
    assert audit["passed"] is False


def test_synthetic_merge_keeps_exactly_12_checks():
    # no COCO rows -> no benchmark_provenance check; synthetic count unchanged
    from smco.merge_results import audit_payloads, smco_row_from_outcome
    task = dict(_e4_task(), suite="synthetic_highdim", stage="e2_factorial_highdim",
                start_points_hash="sph", instance_hash="ih")
    outcome = {"run_id": task["run_id"], "best_value": 1.0, "known_optimum": 0.0,
               "normalized_gap": 0.1, "fe_used": 100, "status": "success",
               "machine_id": "n", "git_commit": "c" * 40, "environment_hash": "eh",
               "target_hit_fe": {}}
    row = smco_row_from_outcome(outcome, task, manifest_id=task["manifest_id"])
    audit = audit_payloads([row], {task["run_id"]: task})
    assert len(audit["checks"]) == 12
    assert all(c["name"] != "benchmark_provenance" for c in audit["checks"])


# --- run_e4_coco_task end-to-end on a fake cocoex problem (review P3) ---

class _FakeProblem:
    """Minimal cocoex-problem stand-in: a d-sphere with optimum 0 at x=0."""
    id_function = 1
    id_instance = 1

    def __init__(self, dim):
        self.dimension = dim
        self.lower_bounds = np.full(dim, -5.0)
        self.upper_bounds = np.full(dim, 5.0)
        self._best = float("inf")
        self._fe = 0
        self.final_target_hit = False
        self.id = f"fake_f{self.id_function:03d}_i{self.id_instance}_d{dim}"
        self.f_opt = 0.0  # sphere optimum; exposed so coco_problem_f_opt() resolves it

    def __call__(self, x):
        self._fe += 1
        v = float(np.sum(np.asarray(x, dtype=float) ** 2))
        if v < self._best:
            self._best = v
        return v

    @property
    def best_observed_fvalue1(self):
        return self._best

    @property
    def evaluations(self):
        return self._fe


def _e4_smco_task(dim=4):
    from smco.experiment_manifests import derive_seed
    t = _e4_task()
    t["dimension"] = dim
    t["fe_budget"] = 200
    t["seed"] = derive_seed("e4_bbob_largescale", "bbob-largescale", "f1", dim, 0, 0, "PY-SP-SMCO-EVO")
    return t


def test_run_on_problem_records_best_trace():
    from smco.coco_runner import run_on_problem
    res = run_on_problem(_FakeProblem(4), algorithm_id="PY-SP-SMCO-EVO",
                         fe_budget=80, n_starts=4)
    assert "best_trace" in res
    assert all(b >= 0.0 for _, b in res["best_trace"])
    assert res["evaluations"] > 0


def test_run_e4_coco_task_writes_outcome_with_coco_provenance(tmp_path):
    from smco.coco_runner import coco_problem_id, run_e4_coco_task
    task = _e4_smco_task(dim=4)
    problem = _FakeProblem(4)
    payload = run_e4_coco_task(
        task, problem, f_opt=0.0, result_dir=str(tmp_path / "raw"),
        machine_id="node213", git_commit="c" * 40, environment_hash="eh",
        suite="bbob-largescale", n_starts=4)
    # JSON written
    written = json.loads((tmp_path / "raw" / f"{task['run_id']}.json").read_text())
    assert written["run_id"] == task["run_id"]
    # COCO-native + benchmark provenance
    assert written["benchmark"]["suite"] == "bbob-largescale"
    assert written["benchmark"]["problem_id"] == coco_problem_id("bbob-largescale", problem)
    assert written["benchmark"]["f_opt"] == 0.0
    assert written["benchmark"]["best_observed_fvalue1"] == payload["best_value"]
    assert written["final_target_hit"] is False
    # derived synthetic-style fields present
    assert written["normalized_gap"] is not None
    assert isinstance(written["anytime"], list) and len(written["anytime"]) >= 1
    # no synthetic instance fields
    assert "instance_artifact_dir" not in written


def test_run_e4_coco_task_baseline_path(tmp_path):
    from smco.coco_runner import run_e4_coco_task
    task = _e4_smco_task(dim=4)
    task.pop("configuration_hash")           # -> baseline classification
    task["algorithm_id"] = "DE"
    run_e4_coco_task(task, _FakeProblem(4), f_opt=0.0, result_dir=str(tmp_path / "raw"),
                     machine_id="n", git_commit="c" * 40, environment_hash="eh", n_starts=4)
    written = json.loads((tmp_path / "raw" / f"{task['run_id']}.json").read_text())
    assert written["benchmark"]["best_observed_fvalue1"] >= 0.0
    assert written["status"] == "success"


# --- sharding dispatch (review P3): manifest task -> cocoex problem resolution ---

class _FakeSuite:
    """Iterates fake problems covering (function, dim, instance) combos."""
    def __init__(self, problems):
        self._problems = problems
    def __iter__(self):
        return iter(self._problems)


def _fake_problems_for(tasks, dim=4):
    # build one fake problem per (function N, dim, cocoex_instance=manifest_inst+1)
    probs = []
    seen = set()
    for t in tasks:
        n = int(str(t["function"]).lstrip("fF"))
        ci = int(t["instance"]) + 1
        if (n, dim, ci) in seen:
            continue
        seen.add((n, dim, ci))
        p = _FakeProblem(dim)
        p.id_function = n
        p.id_instance = ci
        p.id = f"fake_f{n:03d}_i{ci}_d{dim}"
        probs.append(p)
    return probs


def test_match_manifest_to_coco_problems_instance_offset():
    from smco.coco_runner import match_manifest_to_coco_problems
    tasks = [_e4_smco_task(dim=4), dict(_e4_smco_task(dim=4), run_id="r2", instance=3)]
    suite = _FakeSuite(_fake_problems_for(tasks))
    matched = match_manifest_to_coco_problems(tasks, suite)
    assert set(matched) == {t["run_id"] for t in tasks}
    # manifest instance 0 -> cocoex instance 1; instance 3 -> cocoex 4
    assert matched[tasks[0]["run_id"]].id_instance == 1
    assert matched["r2"].id_instance == 4


def test_match_rejects_incomplete_shard():
    from smco.coco_runner import match_manifest_to_coco_problems
    task = _e4_smco_task(dim=4)  # instance 0 -> cocoex 1
    suite = _FakeSuite([])       # no problems at all
    with pytest.raises(ValueError, match="no cocoex problem"):
        match_manifest_to_coco_problems([task], suite)


def test_dispatch_e4_tasks_runs_shard_and_writes_outcomes(tmp_path):
    from smco.coco_runner import dispatch_e4_tasks
    tasks = [_e4_smco_task(dim=4)]
    suite = _FakeSuite(_fake_problems_for(tasks))
    statuses = dispatch_e4_tasks(
        tasks, suite, result_dir=str(tmp_path / "raw"),
        machine_id="n", git_commit="c" * 40, environment_hash="eh",
        suite="bbob-largescale")
    assert statuses[tasks[0]["run_id"]] == "success"
    assert (tmp_path / "raw" / f"{tasks[0]['run_id']}.json").exists()


def test_coco_outcome_carries_r01_frozen_winner_marker():
    # R-01: the language that ran + frozen-winner-validation status are explicit.
    task = _task()
    payload = build_coco_outcome(
        task, best_observed_fvalue1=1.0, evaluations=10, final_target_hit=True,
        best_trace=[(1, 1.0)], f_opt=0.0, initial_ref=1.0, fe_budget=100,
        suite="bbob-largescale", problem_id="p", cocoex_version="x", cocopp_version=None,
        machine_id="n", git_commit="c" * 40, environment_hash="eh",
        ran_language="python", is_frozen_winner_validation=True,
        external_check_kind="frozen_winner")
    b = payload["benchmark"]
    assert b["ran_language"] == "python"
    assert b["is_frozen_winner_validation"] is True
    assert b["external_check_kind"] == "frozen_winner"
    assert coco_outcome_errors(payload) == []


def test_coco_outcome_rejects_invalid_r01_marker():
    task = _task()
    payload = build_coco_outcome(
        task, best_observed_fvalue1=1.0, evaluations=10, final_target_hit=True,
        best_trace=[(1, 1.0)], f_opt=0.0, initial_ref=1.0, fe_budget=100,
        suite="bbob-largescale", problem_id="p", cocoex_version="x", cocopp_version=None,
        machine_id="n", git_commit="c" * 40, environment_hash="eh",
        external_check_kind="bogus")            # invalid kind
    assert any("external_check_kind" in e for e in coco_outcome_errors(payload))
    # missing marker fields -> rejected
    payload["benchmark"].pop("ran_language", None)
    assert any("ran_language" in e for e in coco_outcome_errors(payload))
