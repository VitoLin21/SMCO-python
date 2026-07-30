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
    rc = cli.main(["--statistics", "--merged-dir", str(tmp_path), "--out-dir", str(out_dir)])
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
