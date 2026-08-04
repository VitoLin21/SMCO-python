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
    assert summary["winner_language"] == "python"  # A-04: record winner language
    assert (tmp_path / "selection_candidates.csv").exists()
    assert (tmp_path / "selection_score_components.csv").exists()


def test_build_selection_records_selection_and_config_hash(tmp_path):
    # A-03 / A-02 part 2: selection emits selection_hash + winner_config_hash +
    # per-candidate coverage + a result-set fingerprint.
    runs = {
        "PY-SP-SMCO-EVO": [_run({"1e-1": 10, "1e-2": 20}, 1e-3, 5.0)],
        "PY-RS-SMCO-EVO": [_run({"1e-1": 12}, 1e-1, 4.0)],
    }
    runs["PY-SP-SMCO-EVO"][0]["run_id"] = "r1"
    runs["PY-SP-SMCO-EVO"][0]["task"] = {"configuration_hash": "cfg_winner"}
    runs["PY-RS-SMCO-EVO"][0]["run_id"] = "r2"
    candidates = [
        {"algorithm_id": "PY-SP-SMCO-EVO"},
        {"algorithm_id": "PY-RS-SMCO-EVO"},
    ]
    loader = lambda result_dir, cands: runs
    summary = build_selection(
        result_dir=tmp_path, out_dir=tmp_path, dry_run=False,
        candidates=candidates, loader=loader,
    )
    assert summary["winner_config_hash"] == "cfg_winner"
    assert summary.get("selection_hash")
    assert summary["coverage"] == {"PY-SP-SMCO-EVO": 1, "PY-RS-SMCO-EVO": 1}
    assert summary["n_results"] == 2
    assert summary["results_hash"]
    import json as _json
    loaded = _json.loads((tmp_path / "selection.json").read_text())
    assert loaded["selection_hash"] == summary["selection_hash"]


def test_build_selection_raw_requires_development(tmp_path):
    # R-05: reading raw result_dir JSON is development-only.
    with pytest.raises(ValueError, match="development-only"):
        build_selection(result_dir=tmp_path, out_dir=tmp_path, dry_run=False,
                        candidates=[{"algorithm_id": "PY-SP-SMCO-EVO"}])


def test_build_selection_merged_canonical(tmp_path):
    # R-05/R5c: canonical selection reads merged/ (audit + valid_runs.csv) and
    # validates against the full frozen E1 contract. Outcomes are produced for a
    # single candidate; selection is told to consider only that candidate.
    import json as _json
    from smco.merge_results import merge
    from smco.experiment_manifests import (
        E1_DIMENSIONS, E1_FUNCTIONS, E1_N_INSTANCES,
        build_manifest, e1_algorithm_configs, expand_tasks, freeze_manifest,
    )
    tasks = expand_tasks("e1_development", "synthetic_highdim", list(E1_FUNCTIONS),
                         list(E1_DIMENSIONS), E1_N_INSTANCES, e1_algorithm_configs(),
                         fe_budget_per_d=1000, checkpoints_per_d=(1000,))
    manifest = freeze_manifest(build_manifest("e1_development", "synthetic_highdim", tasks))
    (tmp_path / "m.json").write_text(_json.dumps(manifest))
    aid = "PY-SP-SMCO-EVO"
    raw = tmp_path / "raw"
    raw.mkdir()
    for task in tasks:
        if task["algorithm_id"] != aid:
            continue
        outcome = {"run_id": task["run_id"], "status": "success", "failure_reason": "none",
            "fe_used": int(task["fe_budget"]) - 2, "fe_budget": task["fe_budget"],
            "best_value": 1e-6, "known_optimum": 0.0, "normalized_gap": 0.001,
            "target_hit_fe": {"1e-1": 100, "1e-2": 1000, "1e-3": None, "1e-5": None},
            "anytime": [], "best_so_far_trace": [], "termination_reason": "evaluation_budget",
            "fe_counts_by_event": {}, "wall_time_sec": 1.0, "peak_memory_mb": 1.0,
            "machine_id": "h", "git_commit": "abc", "environment_hash": "env", "task": task,
            "algorithm_id": aid, "supersedes_run_id": "none"}
        (raw / f"{task['run_id']}.json").write_text(_json.dumps(outcome))
    merged = tmp_path / "merged"
    merge([tmp_path / "m.json"], [raw], merged)
    summary = build_selection(out_dir=tmp_path / "sel", merged_dir=merged,
                              e1_manifest_paths=[tmp_path / "m.json"],
                              candidates=[{"algorithm_id": aid}])
    assert summary["winner"] == aid


def test_enforce_merged_completeness_rejects_incomplete():
    # R-05: a candidate with no runs, or uneven coverage, must be rejected.
    from smco.selection import _enforce_merged_completeness
    with pytest.raises(ValueError, match="no runs"):
        _enforce_merged_completeness(
            {"A": [{"run_id": "r1"}], "B": []},
            [{"algorithm_id": "A"}, {"algorithm_id": "B"}])
    with pytest.raises(ValueError, match="incomplete coverage"):
        _enforce_merged_completeness(
            {"A": [{"run_id": "r1"}, {"run_id": "r2"}], "B": [{"run_id": "r3"}]},
            [{"algorithm_id": "A"}, {"algorithm_id": "B"}])


# --- R5b: canonical selection must validate the E1 manifest, stage and exact task set ---

def _e1_merged_csv(tmp_path, rows):
    import csv
    import json as _json
    fields = ["algorithm_id", "stage", "run_id", "configuration_hash", "dimension",
              "status", "normalized_gap", "wall_time_sec", "fe_budget",
              "target_hit_fe_1e-1", "target_hit_fe_1e-2", "target_hit_fe_1e-3", "target_hit_fe_1e-5"]
    merged = tmp_path / "merged"
    merged.mkdir(exist_ok=True)
    with open(merged / "valid_runs.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (merged / "provenance_audit.json").write_text(
        _json.dumps({"passed": True, "failed_checks": []}))
    return merged


def _e1_manifest_file(tmp_path, name, tasks):
    import json as _json
    from smco.experiment_manifests import build_manifest, freeze_manifest
    m = freeze_manifest(build_manifest("e1_development", "synthetic_highdim", tasks))
    path = tmp_path / name
    path.write_text(_json.dumps(m))
    return path


def _e1_task(aid, cfg_hash, run_id, *, instance=0):
    return {"algorithm_id": aid, "configuration_hash": cfg_hash, "run_id": run_id,
            "stage": "e1_development", "suite": "synthetic_highdim", "function": "Rastrigin",
            "dimension": 200, "instance": instance, "replication": 0, "seed": 1,
            "language": "python", "family": "smco", "evolutionary": "true",
            "state_semantics": "state_preserving", "evolution_strategy": "rand1bin",
            "n_starts": 8, "fe_budget": 200000, "checkpoints": [200000]}


def _e1_row(aid, run_id, cfg_hash, *, stage="e1_development"):
    return {"algorithm_id": aid, "stage": stage, "run_id": run_id,
            "configuration_hash": cfg_hash, "dimension": 200, "status": "success",
            "normalized_gap": 0.01, "wall_time_sec": 1.0, "fe_budget": 200000,
            "target_hit_fe_1e-1": "100", "target_hit_fe_1e-2": "",
            "target_hit_fe_1e-3": "", "target_hit_fe_1e-5": ""}


def _full_e1_manifest(tmp_path, name="e1.json"):
    """The full frozen 18x60 E1 contract manifest (plan E1)."""
    import json as _json
    from smco.experiment_manifests import (
        E1_DIMENSIONS, E1_FUNCTIONS, E1_N_INSTANCES,
        build_manifest, e1_algorithm_configs, expand_tasks, freeze_manifest,
    )
    tasks = expand_tasks("e1_development", "synthetic_highdim", list(E1_FUNCTIONS),
                         list(E1_DIMENSIONS), E1_N_INSTANCES, e1_algorithm_configs(),
                         fe_budget_per_d=1000, checkpoints_per_d=(1000,))
    manifest = freeze_manifest(build_manifest("e1_development", "synthetic_highdim", tasks))
    path = tmp_path / name
    path.write_text(_json.dumps(manifest))
    return path, manifest


def _e1_rows_for(manifest, aid):
    """One success merged row per manifest task of `aid` (configuration_hash carried)."""
    rows = []
    for t in manifest["tasks"]:
        if t["algorithm_id"] != aid:
            continue
        rows.append({"algorithm_id": aid, "stage": "e1_development", "run_id": t["run_id"],
                     "configuration_hash": t["configuration_hash"], "dimension": t["dimension"],
                     "status": "success", "normalized_gap": 0.01, "wall_time_sec": 1.0,
                     "fe_budget": t["fe_budget"], "target_hit_fe_1e-1": "100",
                     "target_hit_fe_1e-2": "", "target_hit_fe_1e-3": "", "target_hit_fe_1e-5": ""})
    return rows


# --- R5c: the E1 manifest itself must be frozen, e1_development, and the full 18x60 contract ---

def test_e1_manifest_contract_rejects_wrong_stage(tmp_path):
    # Reviewer repro: a wrong-stage single-task manifest must not pass.
    import json as _json
    from smco.experiment_manifests import build_manifest, freeze_manifest
    from smco.selection import _load_and_validate_e1_manifests
    m = freeze_manifest(build_manifest(
        "e2_factorial_highdim", "synthetic_highdim", [_e1_task("PY-SP-SMCO-EVO", "cfg", "r1")]))
    (tmp_path / "m.json").write_text(_json.dumps(m))
    with pytest.raises(ValueError, match="e1_development"):
        _load_and_validate_e1_manifests([tmp_path / "m.json"])


def test_e1_manifest_contract_rejects_unfrozen(tmp_path):
    import json as _json
    from smco.experiment_manifests import build_manifest
    from smco.selection import _load_and_validate_e1_manifests
    m = build_manifest("e1_development", "synthetic_highdim", [_e1_task("PY-SP-SMCO-EVO", "cfg", "r1")])
    (tmp_path / "m.json").write_text(_json.dumps(m))
    with pytest.raises(ValueError, match="not frozen"):
        _load_and_validate_e1_manifests([tmp_path / "m.json"])


def test_e1_manifest_contract_rejects_wrong_suite(tmp_path):
    # R7c-P2: E1 manifests must be suite synthetic_highdim.
    import json as _json
    from smco.experiment_manifests import build_manifest, freeze_manifest
    from smco.selection import _load_and_validate_e1_manifests
    m = freeze_manifest(build_manifest("e1_development", "bbob", [_e1_task("PY-SP-SMCO-EVO", "cfg", "r1")]))
    (tmp_path / "m.json").write_text(_json.dumps(m))
    with pytest.raises(ValueError, match="synthetic_highdim"):
        _load_and_validate_e1_manifests([tmp_path / "m.json"])


def test_e1_manifest_contract_rejects_incomplete_candidates(tmp_path):
    from smco.selection import _load_and_validate_e1_manifests
    p = _e1_manifest_file(tmp_path, "m.json", [_e1_task("PY-SP-SMCO-EVO", "cfg", "r1")])
    with pytest.raises(ValueError, match="18 E1 candidates"):
        _load_and_validate_e1_manifests([p])


def test_e1_manifest_contract_rejects_short_candidate(tmp_path):
    import json as _json
    from smco.experiment_manifests import manifest_sha256
    from smco.selection import _load_and_validate_e1_manifests
    p, manifest = _full_e1_manifest(tmp_path)
    first_aid = manifest["tasks"][0]["algorithm_id"]
    # drop the last task belonging to that candidate -> 59 tasks
    last_idx = max(i for i, t in enumerate(manifest["tasks"]) if t["algorithm_id"] == first_aid)
    del manifest["tasks"][last_idx]
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    (tmp_path / "e1.json").write_text(_json.dumps(manifest))
    with pytest.raises(ValueError, match="60 E1 tasks"):
        _load_and_validate_e1_manifests([tmp_path / "e1.json"])


def test_e1_manifest_contract_passes_full(tmp_path):
    from smco.selection import _load_and_validate_e1_manifests
    p, _ = _full_e1_manifest(tmp_path)
    index = _load_and_validate_e1_manifests([p])
    assert len(index) == 18
    assert all(len(tasks) == 60 for tasks in index.values())


# --- R5b (on the full contract): merged input must match the manifest exactly ---

def test_build_selection_canonical_requires_e1_manifest(tmp_path):
    # canonical selection over merged/ requires the E1 manifest to validate.
    aid = "PY-SP-SMCO-EVO"
    mpath, manifest = _full_e1_manifest(tmp_path)
    merged = _e1_merged_csv(tmp_path, _e1_rows_for(manifest, aid))
    with pytest.raises(ValueError, match="e1_manifest"):
        build_selection(out_dir=tmp_path / "sel", merged_dir=merged,
                        candidates=[{"algorithm_id": aid}])


def test_build_selection_canonical_with_manifest_validates_and_picks(tmp_path):
    aid = "PY-SP-SMCO-EVO"
    mpath, manifest = _full_e1_manifest(tmp_path)
    merged = _e1_merged_csv(tmp_path, _e1_rows_for(manifest, aid))
    summary = build_selection(out_dir=tmp_path / "sel", merged_dir=merged,
                              e1_manifest_paths=[mpath],
                              candidates=[{"algorithm_id": aid}])
    assert summary["winner"] == aid
    assert summary.get("e1_manifest_validated") is True


def test_build_selection_rejects_mixed_stage_contamination(tmp_path):
    # an e1 row + an e2 row for the same candidate must be rejected.
    aid = "PY-SP-SMCO-EVO"
    mpath, manifest = _full_e1_manifest(tmp_path)
    rows = _e1_rows_for(manifest, aid)
    rows.append(_e1_row(aid, "stale-e2", "cfg", stage="e2_factorial_highdim"))
    merged = _e1_merged_csv(tmp_path, rows)
    with pytest.raises(ValueError, match="non-E1"):
        build_selection(out_dir=tmp_path / "sel", merged_dir=merged,
                        e1_manifest_paths=[mpath], candidates=[{"algorithm_id": aid}])


def test_build_selection_rejects_wrong_task_count(tmp_path):
    # merged missing one of the candidate's 60 tasks -> reject.
    aid = "PY-SP-SMCO-EVO"
    mpath, manifest = _full_e1_manifest(tmp_path)
    rows = _e1_rows_for(manifest, aid)
    rows.pop()
    merged = _e1_merged_csv(tmp_path, rows)
    with pytest.raises(ValueError, match="missing"):
        build_selection(out_dir=tmp_path / "sel", merged_dir=merged,
                        e1_manifest_paths=[mpath], candidates=[{"algorithm_id": aid}])


def test_build_selection_rejects_rows_not_in_manifest(tmp_path):
    # a merged row whose run_id is not in the E1 manifest -> extra -> reject.
    aid = "PY-SP-SMCO-EVO"
    mpath, manifest = _full_e1_manifest(tmp_path)
    rows = _e1_rows_for(manifest, aid)
    rows.append(_e1_row(aid, "rX-not-in-manifest", "cfg"))
    merged = _e1_merged_csv(tmp_path, rows)
    with pytest.raises(ValueError, match="not in the E1 manifest"):
        build_selection(out_dir=tmp_path / "sel", merged_dir=merged,
                        e1_manifest_paths=[mpath], candidates=[{"algorithm_id": aid}])


def test_build_selection_rejects_configuration_hash_mismatch(tmp_path):
    # a run_id present in the manifest but a different configuration_hash.
    aid = "PY-SP-SMCO-EVO"
    mpath, manifest = _full_e1_manifest(tmp_path)
    rows = _e1_rows_for(manifest, aid)
    rows[0]["configuration_hash"] = "cfgDIFFERENT"
    merged = _e1_merged_csv(tmp_path, rows)
    with pytest.raises(ValueError, match="configuration_hash mismatch"):
        build_selection(out_dir=tmp_path / "sel", merged_dir=merged,
                        e1_manifest_paths=[mpath], candidates=[{"algorithm_id": aid}])


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
