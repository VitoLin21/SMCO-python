"""Tests for the Task-12 statistics pipeline (R-03)."""
from __future__ import annotations

import csv
import json

import pytest


def _merged_dir(tmp_path, rows, audit_passed=True):
    fields = ["algorithm_id", "dimension", "status", "normalized_gap", "wall_time_sec", "fe_budget",
              "target_hit_fe_1e-1", "target_hit_fe_1e-2", "target_hit_fe_1e-3", "target_hit_fe_1e-5"]
    with open(tmp_path / "valid_runs.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (tmp_path / "provenance_audit.json").write_text(
        json.dumps({"passed": audit_passed, "failed_checks": [] if audit_passed else ["x"]}))
    return tmp_path


def test_load_merged_rows_refuses_failed_audit(tmp_path):
    from smco.paper_analysis import load_merged_rows
    _merged_dir(tmp_path, [], audit_passed=False)
    with pytest.raises(ValueError, match="audit failed"):
        load_merged_rows(tmp_path)


def test_primary_table_computes_ecdf_auc_and_ert(tmp_path):
    from smco.paper_analysis import primary_table
    rows = [
        {"algorithm_id": "A", "dimension": 200, "status": "success", "normalized_gap": 0.01,
         "wall_time_sec": 1.0, "fe_budget": 200000,
         "target_hit_fe_1e-1": "100", "target_hit_fe_1e-2": "1000",
         "target_hit_fe_1e-3": "", "target_hit_fe_1e-5": ""},
        {"algorithm_id": "B", "dimension": 200, "status": "success", "normalized_gap": 0.1,
         "wall_time_sec": 1.0, "fe_budget": 200000,
         "target_hit_fe_1e-1": "200", "target_hit_fe_1e-2": "",
         "target_hit_fe_1e-3": "", "target_hit_fe_1e-5": ""},
    ]
    table = primary_table(rows, ["A", "B"])
    a = next(r for r in table if r["algorithm_id"] == "A")
    b = next(r for r in table if r["algorithm_id"] == "B")
    assert a["n_runs"] == 1 and b["n_runs"] == 1
    assert a["ecdf_auc"] >= b["ecdf_auc"]  # A hits tighter targets
    assert a["ert_1e-1"] == 100.0  # single run reached at FE 100
    assert isinstance(a["median_log_gap"], float)


def test_primary_table_ert_uses_per_run_budget_not_max():
    # R3b: a mixed d/budget algorithm must use EACH run's own fe_budget for ERT,
    # not max(fe_budget). Locks the inflation bug across dimensions.
    from smco.paper_analysis import primary_table
    rows = [
        {"algorithm_id": "A", "dimension": 1000, "status": "success", "normalized_gap": 0.01,
         "wall_time_sec": 1.0, "fe_budget": 1_000_000,
         "target_hit_fe_1e-1": "500", "target_hit_fe_1e-2": "",
         "target_hit_fe_1e-3": "", "target_hit_fe_1e-5": ""},
        {"algorithm_id": "A", "dimension": 200, "status": "success", "normalized_gap": 0.5,
         "wall_time_sec": 1.0, "fe_budget": 200_000,
         "target_hit_fe_1e-1": "", "target_hit_fe_1e-2": "",
         "target_hit_fe_1e-3": "", "target_hit_fe_1e-5": ""},
    ]
    table = primary_table(rows, ["A"], n_boot=200)
    a = table[0]
    # reached [500], unreached contributes its own 200_000 -> (500 + 200_000)/1
    # the old max-budget path would give (500 + 1_000_000)/1 = 1_000_500
    assert a["ert_1e-1"] == 200_500.0


def test_primary_table_bootstrap_is_hierarchical_by_function():
    # R3b: the median log-gap CI uses a function->instance hierarchical bootstrap
    # (plan 6.3) with the pooled median as the point estimate. Choose values where
    # the pooled median differs from the median of per-function medians.
    import math
    from smco.paper_analysis import primary_table

    def row(func, gap):
        return {"algorithm_id": "A", "function": func, "dimension": 200,
                "status": "success", "normalized_gap": gap, "wall_time_sec": 1.0,
                "fe_budget": 200000, "target_hit_fe_1e-1": "10", "target_hit_fe_1e-2": "",
                "target_hit_fe_1e-3": "", "target_hit_fe_1e-5": ""}

    rows = [row("F1", 1e-3), row("F1", 1e-1),
            row("F2", 1.0), row("F2", 1.0), row("F2", 1.0)]
    table = primary_table(rows, ["A"], n_boot=300)
    a = table[0]
    # pooled log-gaps sorted = [log(1e-3), log(1e-1), 0, 0, 0] -> median 0.0
    # median of per-function medians = median([log(1e-2)~ -4.6, 0.0]) = -2.3 (would differ)
    assert a["median_log_gap"] == pytest.approx(0.0, abs=1e-9)
    assert a["median_log_gap_ci_lo"] is not None and a["median_log_gap_ci_hi"] is not None
    assert a["median_log_gap_ci_lo"] <= a["median_log_gap"] <= a["median_log_gap_ci_hi"]


def test_write_primary_table_csv(tmp_path):
    from smco.paper_analysis import write_primary_table
    rows = [{"algorithm_id": "A", "dimension": 200, "status": "success", "normalized_gap": 0.01,
             "wall_time_sec": 1.0, "fe_budget": 200000,
             "target_hit_fe_1e-1": "100", "target_hit_fe_1e-2": "",
             "target_hit_fe_1e-3": "", "target_hit_fe_1e-5": ""}]
    _merged_dir(tmp_path, rows)
    out = tmp_path / "out"
    write_primary_table(tmp_path, out, ["A"])
    rows2 = list(csv.DictReader(open(out / "primary_table.csv")))
    assert len(rows2) == 1
    assert rows2[0]["algorithm_id"] == "A"
    assert "ecdf_auc" in rows2[0] and "ert_1e-1" in rows2[0]


def test_analyze_cli_statistics(tmp_path):
    # R-03: the analyze CLI computes the primary table from merged/.
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "analyze_cli", Path("scripts/analyze_smco_evo_highdim_paper.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    rows = [{"algorithm_id": "PY-SP-SMCO-EVO", "dimension": 200, "status": "success",
             "normalized_gap": 0.01, "wall_time_sec": 1.0, "fe_budget": 200000,
             "target_hit_fe_1e-1": "100", "target_hit_fe_1e-2": "",
             "target_hit_fe_1e-3": "", "target_hit_fe_1e-5": ""}]
    _merged_dir(tmp_path, rows)
    out_dir = tmp_path / "analysis"
    # This fixture is intentionally an ad-hoc merged directory, so it exercises
    # the explicitly labelled development escape hatch.  Formal Task 12 uses
    # the canonical index and artifact key (covered in test_composite_cli).
    rc = cli.main(["--statistics", "--development", "--merged-dir", str(tmp_path),
                   "--out-dir", str(out_dir)])
    assert rc == 0
    assert (out_dir / "primary_table.csv").exists()


def test_package_report_traces_numbers(tmp_path):
    # R-03 Task-13: report.md numbers trace back to selection.json + primary_table.csv.
    import importlib.util
    from pathlib import Path
    (tmp_path / "selection.json").write_text(json.dumps(
        {"winner": "PY-SP-SMCO-EVO", "winner_language": "python",
         "selection_hash": "s1", "winner_config_hash": "cfg", "n_results": 60,
         "results_hash": "rh"}))
    with open(tmp_path / "primary_table.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["algorithm_id", "n_runs", "ecdf_auc", "median_log_gap", "failure_rate",
                    "ert_1e-1", "ert_1e-5"])
        w.writerow(["PY-SP-SMCO-EVO", 60, 0.42, -3.1, 0.0, 1234, 50000])
    spec = importlib.util.spec_from_file_location(
        "pkg_cli", Path("scripts/package_smco_evo_highdim_paper.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    out = cli.build_report(tmp_path, tmp_path / "report.md")
    text = out.read_text()
    assert "PY-SP-SMCO-EVO" in text
    assert "0.42" in text  # ecdf_auc traced
    assert "s1" in text     # selection_hash traced


# --- Task-12 pairwise Holm/superiority (review §9) ---

def _prow(algo, func, dim, inst, gap):
    return {"algorithm_id": algo, "function": func, "dimension": dim, "instance": inst,
            "status": "success", "normalized_gap": gap, "wall_time_sec": 1.0,
            "fe_budget": dim * 1000, "target_hit_fe_1e-1": "", "target_hit_fe_1e-2": "",
            "target_hit_fe_1e-3": "", "target_hit_fe_1e-5": ""}


def test_pairwise_table_detects_a_beats_b():
    # A consistently reaches a lower (better) normalized_gap than B on shared
    # problems -> median_log_gap_diff(A-B) < 0, prob_a_better ~1, small p.
    from smco.paper_analysis import pairwise_table
    rows = []
    for i, g in enumerate([0.01, 0.02, 0.015, 0.03, 0.005]):
        rows += [_prow("A", "f", 200, i, g), _prow("B", "f", 200, i, g * 10)]
    table = pairwise_table(rows, ["A", "B"], n_boot=500, seed=0)
    assert len(table) == 1
    r = table[0]
    assert r["algorithm_a"] == "A" and r["algorithm_b"] == "B"
    assert r["n_pairs"] == 5
    assert r["median_log_gap_diff"] < 0          # A better (lower gap)
    assert r["prob_a_better"] == 1.0             # A beats B on every pair
    assert r["p_value"] < 0.05                   # significant
    assert r["p_holm"] >= r["p_value"]           # Holm never decreases p


def test_pairwise_table_no_signal_is_not_significant():
    # A and B exchange wins ~symmetrically -> large p, prob_a_better ~0.5.
    from smco.paper_analysis import pairwise_table
    gaps_a = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    gaps_b = [0.02, 0.01, 0.04, 0.03, 0.06, 0.05, 0.08, 0.07]  # swap pairs
    rows = []
    for i in range(len(gaps_a)):
        rows += [_prow("A", "f", 200, i, gaps_a[i]), _prow("B", "f", 200, i, gaps_b[i])]
    table = pairwise_table(rows, ["A", "B"], n_boot=500, seed=0)
    r = table[0]
    assert abs(r["median_log_gap_diff"]) < 0.5
    assert 0.3 <= r["prob_a_better"] <= 0.7
    assert r["p_value"] > 0.05                   # not significant


def test_pairwise_only_pairs_shared_problems():
    # problems present for A but not B are not counted in n_pairs
    from smco.paper_analysis import pairwise_table
    rows = [_prow("A", "f", 200, 0, 0.01), _prow("B", "f", 200, 0, 0.5),
            _prow("A", "f", 200, 1, 0.01)]  # instance 1 only for A
    table = pairwise_table(rows, ["A", "B"], n_boot=200, seed=0)
    assert table[0]["n_pairs"] == 1


def test_pairwise_holm_across_many_pairs():
    # 3 algos => 3 pairs; Holm adjusted p is monotone >= raw p
    from smco.paper_analysis import pairwise_table
    rows = []
    for i, g in enumerate([0.01, 0.02, 0.03]):
        rows += [_prow("A", "f", 200, i, g), _prow("B", "f", 200, i, g * 5),
                 _prow("C", "f", 200, i, g * 20)]
    table = pairwise_table(rows, ["A", "B", "C"], n_boot=300, seed=0)
    assert len(table) == 3
    assert all(r["p_holm"] is not None for r in table)
    assert all(r["p_holm"] >= r["p_value"] - 1e-9 for r in table)


def test_write_pairwise_table_csv(tmp_path):
    from smco.paper_analysis import write_pairwise_table
    import csv as _csv
    d = tmp_path / "merged"; d.mkdir()
    rows = []
    for i, g in enumerate([0.01, 0.02]):
        rows += [_prow("A", "f", 200, i, g), _prow("B", "f", 200, i, g * 8)]
    with open(d / "valid_runs.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (d / "provenance_audit.json").write_text(json.dumps({"passed": True}))
    table = write_pairwise_table(str(d), str(tmp_path / "out"), ["A", "B"], n_boot=200)
    written = list(_csv.DictReader(open(tmp_path / "out" / "pairwise_table.csv")))
    assert len(written) == 1
    assert {written[0]["algorithm_a"], written[0]["algorithm_b"]} == {"A", "B"}
    assert written[0]["p_holm"] != ""


def test_package_report_traces_pairwise_and_figures(tmp_path):
    # §9 Task-13: report.md also traces pairwise_table.csv (Holm) + ECDF figures.
    import importlib.util
    from pathlib import Path
    with open(tmp_path / "primary_table.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["algorithm_id", "n_runs", "ecdf_auc", "median_log_gap",
                    "failure_rate", "ert_1e-1", "ert_1e-5"])
        w.writerow(["PY-SP-SMCO-EVO", 60, 0.42, -3.1, 0.0, 1234, 50000])
    with open(tmp_path / "pairwise_table.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["algorithm_a", "algorithm_b", "n_pairs", "median_log_gap_diff",
                    "diff_ci_lo", "diff_ci_hi", "prob_a_better", "p_value", "p_holm"])
        w.writerow(["PY-SP-SMCO-EVO", "DE", 60, "-1.14", "-1.3", "-1.0", "0.98", "0.0", "0.0"])
    (tmp_path / "ecdf_target_1e-2.png").write_bytes(b"\x89PNG fake")
    spec = importlib.util.spec_from_file_location(
        "pkg_cli", Path("scripts/package_smco_evo_highdim_paper.py"))
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    text = cli.build_report(tmp_path, tmp_path / "report.md").read_text()
    assert "Pairwise comparison" in text
    assert "-1.14" in text and "0.98" in text      # median diff + prob traced
    assert "ecdf_target_1e-2.png" in text           # figure referenced
