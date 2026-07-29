"""Tests for the E1 global implementation selection (Task 9).

Selection picks ONE SMCO-EVO implementation globally across all functions,
dimensions and instances (no per-function cherry-picking). The ranking cascade
is: ECDF-AUC of relative targets over log10(FE/dim) -> (within 1% AUC) median
normalized log-gap -> failure rate -> median wall time. Failed/timeout runs
stay in the denominator as right-censored (never-solved) target pairs.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from smco.selection import (
    SELECTION_RULES,
    build_selection,
    rank_configs,
    score_config,
    selection_candidates,
)


def _run(target_hit, gap, wall, *, status="success"):
    return {
        "status": status,
        "target_hit_fe": dict(target_hit),
        "normalized_gap": gap,
        "wall_time_sec": wall,
    }


def test_selection_candidates_has_18():
    candidates = selection_candidates()
    assert len(candidates) == 18
    assert len({c["algorithm_id"] for c in candidates}) == 18


def test_selection_rules_documented_and_ordered():
    assert len(SELECTION_RULES) == 4
    # Primary must reference ECDF-AUC; cascade order is fixed.
    assert "ECDF-AUC" in SELECTION_RULES[0]
    assert "log-gap" in SELECTION_RULES[1]
    assert "failure" in SELECTION_RULES[2].lower()
    assert "wall time" in SELECTION_RULES[3].lower()


def test_score_config_aggregates_target_hit_rate_and_gap():
    runs = [
        _run({"1e-1": 10, "1e-2": 20, "1e-3": None, "1e-5": None}, 1e-2, 5.0),
        _run({"1e-1": 8, "1e-2": None, "1e-3": None, "1e-5": None}, 5e-2, 7.0),
    ]
    s = score_config(runs)
    assert s["n_runs"] == 2
    # hits: run0 hits 1e-1,1e-2 (2/4); run1 hits 1e-1 (1/4) -> total 3/8
    assert pytest.approx(s["target_hit_rate"], rel=1e-9) == 3 / 8
    assert s["failure_rate"] == 0.0
    assert s["median_wall_time"] == 6.0
    assert s["median_log_gap"] is not None


def test_score_config_counts_failures_in_denominator():
    runs = [
        _run({"1e-1": 10}, 1e-2, 5.0),
        {"status": "algorithm_failure", "target_hit_fe": {}, "normalized_gap": None, "wall_time_sec": 1.0},
    ]
    s = score_config(runs)
    assert s["n_runs"] == 2
    assert s["failure_rate"] == 0.5
    # A-02: failed runs keep their 4 target slots in the denominator (right
    # censored) -> 1 hit / 8 slots, not 1/4.
    assert pytest.approx(s["target_hit_rate"], rel=1e-9) == 1 / 8


def test_rank_configs_orders_by_ecdf_auc_then_gap():
    scored = {
        "A": {"ecdf_auc": 0.8, "median_log_gap": -3.0, "failure_rate": 0.0, "median_wall_time": 5.0},
        "B": {"ecdf_auc": 0.5, "median_log_gap": -2.0, "failure_rate": 0.0, "median_wall_time": 4.0},
        "C": {"ecdf_auc": 0.8, "median_log_gap": -4.0, "failure_rate": 0.0, "median_wall_time": 6.0},
    }
    ranked = rank_configs(scored)
    order = [aid for aid, _ in ranked]
    # A and C tie on ecdf_auc (0.8, within 1%); C has lower log-gap -> C first; B last.
    assert order == ["C", "A", "B"]


def test_ecdf_auc_perfect_when_all_targets_hit_early():
    from smco.selection import ecdf_auc
    run = {"status": "success", "dimension": 10,
           "target_hit_fe": {"1e-1": 10, "1e-2": 10, "1e-3": 10, "1e-5": 10}}
    assert pytest.approx(ecdf_auc([run]), abs=1e-9) == 1.0


def test_ecdf_auc_zero_when_all_fail():
    from smco.selection import ecdf_auc
    run = {"status": "algorithm_failure", "dimension": 10, "target_hit_fe": {}}
    assert ecdf_auc([run]) == 0.0


def test_ecdf_auc_failure_lowers_score_below_all_success():
    from smco.selection import ecdf_auc
    good = {"status": "success", "dimension": 10,
            "target_hit_fe": {"1e-1": 10, "1e-2": 10, "1e-3": 10, "1e-5": 10}}
    mixed = [good, {"status": "algorithm_failure", "dimension": 10, "target_hit_fe": {}}]
    assert 0.0 < ecdf_auc(mixed) < ecdf_auc([good])


def test_rank_configs_ecdf_auc_one_percent_threshold():
    scored = {
        "A": {"ecdf_auc": 0.80, "median_log_gap": -3.0, "failure_rate": 0.0, "median_wall_time": 5.0},
        "B": {"ecdf_auc": 0.50, "median_log_gap": -2.0, "failure_rate": 0.0, "median_wall_time": 4.0},
        "C": {"ecdf_auc": 0.805, "median_log_gap": -4.0, "failure_rate": 0.0, "median_wall_time": 6.0},
    }
    ranked = rank_configs(scored)
    order = [aid for aid, _ in ranked]
    # A(0.80) & C(0.805) within 1% -> tiebreak log-gap: C(-4)<A(-3) -> C,A; B far below -> last.
    assert order == ["C", "A", "B"]


def test_build_selection_dry_run_writes_report_and_json(tmp_path):
    summary = build_selection(result_dir=None, out_dir=tmp_path, dry_run=True)
    assert summary["dry_run"] is True
    assert summary["n_candidates"] == 18
    assert (tmp_path / "selection.json").exists()
    assert (tmp_path / "selection_report.md").exists()
    loaded = json.loads((tmp_path / "selection.json").read_text())
    assert loaded["n_candidates"] == 18
    report = (tmp_path / "selection_report.md").read_text()
    for rule in SELECTION_RULES:
        assert rule in report
    assert "dry-run" in report.lower() or "no results" in report.lower()


def test_build_selection_real_picks_global_winner(tmp_path):
    # Two candidate configs with fabricated per-config runs via a loader hook.
    runs = {
        "PY-SP-SMCO-EVO": [_run({"1e-1": 10, "1e-2": 20}, 1e-3, 5.0)],
        "PY-RS-SMCO-EVO": [_run({"1e-1": 12}, 1e-1, 4.0)],
    }
    candidates = [
        {"algorithm_id": "PY-SP-SMCO-EVO"},
        {"algorithm_id": "PY-RS-SMCO-EVO"},
    ]
    loader = lambda result_dir, cands: runs
    summary = build_selection(
        result_dir=tmp_path, out_dir=tmp_path, dry_run=False,
        candidates=candidates, loader=loader,
    )
    assert summary["winner"] == "PY-SP-SMCO-EVO"
    assert (tmp_path / "selection_candidates.csv").exists()
    assert (tmp_path / "selection_score_components.csv").exists()


_ANALYZE = (
    Path(__file__).resolve().parent.parent / "scripts" / "analyze_smco_evo_highdim_paper.py"
)


def _load_analyze_cli():
    spec = importlib.util.spec_from_file_location("smco_evo_analyze_cli", _ANALYZE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyze_cli_selection_dry_run(tmp_path):
    cli = _load_analyze_cli()
    rc = cli.main([
        "--stage", "e1-development", "--selection-only", "--dry-run",
        "--out-dir", str(tmp_path),
    ])
    assert rc == 0
    assert (tmp_path / "selection.json").exists()
