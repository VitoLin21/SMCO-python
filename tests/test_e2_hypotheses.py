from __future__ import annotations

import pytest

from smco.e2_hypotheses import (
    h1_summary, h2_dimension_trend, h3_instance_provenance, paired_e2_cells,
)


def _rows():
    rows = []
    for function in ("F1", "F2"):
        for dimension in (10, 100):
            for instance in (0, 1):
                rows.extend([
                    {"algorithm_id": "W", "function": function, "dimension": dimension,
                     "instance": instance, "normalized_gap": "1"},
                    {"algorithm_id": "B", "function": function, "dimension": dimension,
                     "instance": instance, "normalized_gap": "10"},
                ])
    return rows


def test_paired_cells_are_one_per_problem_and_positive_favours_winner():
    cells = paired_e2_cells(_rows(), "W", "B")
    assert len(cells) == 8
    assert all(c["log_gap_gain"] > 0 for c in cells)
    h1 = h1_summary(cells, n_boot=100, seed=1)
    assert h1["direction"] == "positive_favours_winner"
    assert h1["probability_winner_better"] == 1.0


def test_paired_cells_reject_duplicate_or_unpaired_problem():
    rows = _rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        paired_e2_cells(rows, "W", "B")
    with pytest.raises(ValueError, match="unpaired"):
        paired_e2_cells(_rows()[:-1], "W", "B")


def test_h2_uses_log_dimension_and_hierarchical_bootstrap():
    cells = paired_e2_cells(_rows(), "W", "B")
    summary, by_dim = h2_dimension_trend(cells, n_boot=100, seed=2)
    assert summary["model"] == "ordinary_least_squares_on_paired_problem_cells"
    assert [row["dimension"] for row in by_dim] == [10, 100]


def test_h3_reports_not_testable_without_transform_levels():
    cells = paired_e2_cells(_rows(), "W", "B")
    instances = {"instances": [
        {"function": c["function"], "dimension": c["dimension"], "instance_id": c["instance"],
         "transform_sha256": "x", "file_hashes": {"shift": "x", "permutation": "x", "rotation_blocks": "x"}}
        for c in cells
    ]}
    summary, detail = h3_instance_provenance(cells, instances)
    assert summary["status"] == "not_testable"
    assert len(detail) == len(cells)
