"""Tests for the COCO external-validation task-level outcome (review §9 P3)."""
from __future__ import annotations

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
