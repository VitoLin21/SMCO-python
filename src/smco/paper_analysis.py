"""Task-12 statistics pipeline over merged/ E1 results (R-03).

Consumes ``merged/valid_runs.csv`` (only after a passing provenance audit) and
emits the paper primary table: per-algorithm ECDF-AUC, COCO ERT per target,
bootstrap CI on the median log-gap, and failure rate. Figure rendering and the
final report package (Task 13) consume this module.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .paper_stats import bootstrap_ci, expected_running_time
from .selection import TARGETS, ecdf_auc

_NAN_TOKENS = ("", None, "none", "nan", "None", "NaN")


def load_merged_rows(merged_dir) -> list[dict]:
    """Load merged/valid_runs.csv; refuse if provenance_audit.json failed."""
    merged_dir = Path(merged_dir)
    audit = json.loads((merged_dir / "provenance_audit.json").read_text())
    if not audit.get("passed"):
        raise ValueError(
            f"provenance audit failed ({audit.get('failed_checks')}); "
            f"refusing to compute paper statistics over unaudited results"
        )
    return list(csv.DictReader(open(merged_dir / "valid_runs.csv")))


def _row_to_payload(r: dict) -> dict:
    th: dict[str, int] = {}
    for t in TARGETS:
        v = r.get(f"target_hit_fe_{t}")
        if v not in _NAN_TOKENS:
            try:
                th[t] = int(float(v))
            except (TypeError, ValueError):
                pass
    gap = r.get("normalized_gap")
    wall = r.get("wall_time_sec")
    dim = r.get("dimension")
    return {
        "status": r.get("status", "success"),
        "dimension": int(float(dim)) if dim not in _NAN_TOKENS else 1,
        "normalized_gap": float(gap) if gap not in _NAN_TOKENS else None,
        "wall_time_sec": float(wall) if wall not in _NAN_TOKENS else None,
        "target_hit_fe": th,
        "fe_budget": int(float(r["fe_budget"])) if r.get("fe_budget") not in _NAN_TOKENS else 0,
    }


def primary_table(rows: list[dict], algorithms) -> list[dict]:
    """Per-algorithm primary statistics: ECDF-AUC, ERT per target, bootstrap CI."""
    by_algo: dict[str, list] = {a: [] for a in algorithms}
    for r in rows:
        aid = r.get("algorithm_id")
        if aid in by_algo:
            by_algo[aid].append(_row_to_payload(r))
    out: list[dict] = []
    for aid in algorithms:
        runs = by_algo.get(aid, [])
        n = len(runs)
        if n == 0:
            out.append({"algorithm_id": aid, "n_runs": 0})
            continue
        gaps = [r["normalized_gap"] for r in runs if r["normalized_gap"] is not None]
        log_gaps = [float(np.log(max(g, 1e-12))) for g in gaps]
        point, lo, hi = (
            bootstrap_ci(log_gaps, stat=np.median, n_boot=2000, seed=0)
            if log_gaps else (None, None, None)
        )
        budget = max((r["fe_budget"] for r in runs), default=0)
        row: dict = {
            "algorithm_id": aid,
            "n_runs": n,
            "ecdf_auc": ecdf_auc(runs),
            "median_log_gap": point,
            "median_log_gap_ci_lo": lo,
            "median_log_gap_ci_hi": hi,
            "failure_rate": 1.0 - sum(1 for r in runs if r["status"] == "success") / n,
        }
        for t in TARGETS:
            row[f"ert_{t}"] = expected_running_time(
                [r["target_hit_fe"].get(t) for r in runs], budget)
        out.append(row)
    return out


_PRIMARY_FIELDS = [
    "algorithm_id", "n_runs", "ecdf_auc", "median_log_gap",
    "median_log_gap_ci_lo", "median_log_gap_ci_hi", "failure_rate",
] + [f"ert_{t}" for t in TARGETS]


def write_primary_table(merged_dir, out_dir, algorithms) -> list[dict]:
    """Write primary_table.csv from merged/ (audit must pass). Returns the rows."""
    rows = load_merged_rows(merged_dir)
    table = primary_table(rows, algorithms)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "primary_table.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=_PRIMARY_FIELDS)
        w.writeheader()
        for r in table:
            w.writerow({c: r.get(c, "") for c in _PRIMARY_FIELDS})
    return table


__all__ = ["load_merged_rows", "primary_table", "write_primary_table"]
