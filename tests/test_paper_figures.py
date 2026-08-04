"""Smoke tests for Task-12 ECDF figures (review §9)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from smco.paper_figures import ecdf_curves, plot_ecdf_target, write_ecdf_figures


def _write_merged(tmp, rows):
    d = tmp / "merged"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "valid_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (d / "provenance_audit.json").write_text(json.dumps({"passed": True}))
    return d


def _row(algo, dim, inst, hit_fe):
    """hit_fe: int reached FE, or None (censored)."""
    return {"algorithm_id": algo, "function": "f", "dimension": dim, "instance": inst,
            "status": "success", "normalized_gap": 0.1, "fe_budget": dim * 1000,
            "target_hit_fe_1e-1": "" if hit_fe is None else str(hit_fe),
            "target_hit_fe_1e-2": "", "target_hit_fe_1e-3": "", "target_hit_fe_1e-5": ""}


def test_ecdf_curves_right_censoring():
    # A reaches target at FE 100 (1/2 runs); B never reaches (0/2).
    hit = {"A": [100, None], "B": [None, None]}
    curves = ecdf_curves(hit, max_budget=1000)
    ax, ay = curves["A"]
    # at budget >= 100, fraction = 1/2
    assert max(ay) == 0.5
    bx, by = curves["B"]
    assert max(by) == 0.0  # censored -> never solves


def test_plot_ecdf_target_renders_png(tmp_path):
    rows = [_row("A", 200, 0, 100), _row("A", 200, 1, None),
            _row("B", 200, 0, None), _row("B", 200, 1, None)]
    merged = _write_merged(tmp_path, rows)
    path = plot_ecdf_target(merged, tmp_path / "out", ["A", "B"], "1e-1",
                            dimensions=(200,))
    assert path.exists() and path.stat().st_size > 0


def test_write_ecdf_figures_all_targets(tmp_path):
    rows = [_row("A", 200, 0, 100), _row("B", 200, 0, 500),
            _row("A", 500, 0, 2000), _row("B", 500, 0, None)]
    merged = _write_merged(tmp_path, rows)
    paths = write_ecdf_figures(merged, tmp_path / "out", ["A", "B"],
                               dimensions=(200, 500))
    assert len(paths) == 4  # one per target in selection.TARGETS
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
