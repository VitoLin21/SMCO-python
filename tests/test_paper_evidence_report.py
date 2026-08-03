"""Smoke test for the Task-13 consolidated evidence report assembler."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load():
    spec = importlib.util.spec_from_file_location(
        "report_cli", Path("scripts/build_paper_evidence_report.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_consolidated_report_traces_all_evidence(tmp_path):
    cli = _load()
    (tmp_path / "selection.json").write_text(json.dumps(
        {"winner": "PY-SP-SMCO-EVO", "winner_language": "python",
         "selection_hash": "s1"}))
    e3 = tmp_path / "e3"
    e3.mkdir()
    _write_csv(e3 / "primary_table.csv", ["algorithm_id", "n_runs"],
               [{"algorithm_id": "PY-SP-SMCO-EVO", "n_runs": 60}])
    _write_csv(e3 / "pairwise_table.csv",
               ["algorithm_a", "algorithm_b", "n_pairs", "median_log_gap_diff",
                "prob_a_better", "prob_b_better", "tie_rate", "p_holm"],
               [{"algorithm_a": "PY-SP-SMCO-EVO", "algorithm_b": "DE", "n_pairs": 60,
                 "median_log_gap_diff": "-1.1", "prob_a_better": "0.83",
                 "prob_b_better": "0.02", "tie_rate": "0.15", "p_holm": "0.0"},
                {"algorithm_a": "GenSA", "algorithm_b": "PY-SP-SMCO-EVO", "n_pairs": 60,
                 "median_log_gap_diff": "0.05", "prob_a_better": "0.35",
                 "prob_b_better": "0.50", "tie_rate": "0.15", "p_holm": "1.0"}])
    for name in ("e4", "e5"):
        _write_csv(tmp_path / f"{name}.csv",
                   ["algorithm_id", "n_runs", "final_target_hit_rate", "median_fe_used"],
                   [{"algorithm_id": "PY-SP-SMCO-EVO", "n_runs": 360,
                     "final_target_hit_rate": "0.0", "median_fe_used": "320000"},
                    {"algorithm_id": "GenSA", "n_runs": 360,
                     "final_target_hit_rate": "0.083", "median_fe_used": "320000"}])
    out = cli.build(str(tmp_path / "selection.json"), str(e3),
                    str(tmp_path / "e4.csv"), str(tmp_path / "e5.csv"),
                    str(tmp_path / "report.md"))
    text = out.read_text()
    assert "PY-SP-SMCO-EVO" in text and "s1" in text           # selection
    assert "| DE |" in text and "0.0" in text                   # pairwise winner-opponent
    # GenSA row: winner is algorithm_b -> diff flipped to -0.05
    assert "| GenSA |" in text and "-0.05" in text
    assert "| GenSA | 60 | -0.0500 | 0.50 | 1.0 |" in text
    assert "E4 bbob-largescale" in text and "E5 bbob low-dim" in text  # COCO sections
    assert "Honest boundaries" in text                          # framing
