"""Analysis for formal COCO-native external evidence.

This deliberately does *not* compute synthetic normalized gaps, ECDF-AUC or
relative-target ERT.  COCO versions that expose no auditable f_opt produce
``coco_native`` outcomes, for which those quantities are undefined.  The
external report therefore limits itself to native completion, official final
target-hit rate, and FE consumption, stratified by algorithm.
"""
from __future__ import annotations

import csv
from pathlib import Path


_FIELDS = (
    "algorithm_id", "n_runs", "n_success", "failure_rate", "n_final_target_hit",
    "final_target_hit_rate", "median_fe_used", "median_fe_budget",
)


def load_coco_native_runs(path) -> list[dict]:
    """Load a validated COCO-native sidecar; reject mixed/relative metrics."""
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("COCO-native sidecar has no rows")
    modes = {row.get("metric_mode") for row in rows}
    if modes != {"coco_native"}:
        raise ValueError(
            "COCO external analysis requires metric_mode='coco_native'; "
            "use neither normalized-gap primary_table nor derived-relative rows")
    return rows


def native_summary(rows: list[dict]) -> list[dict]:
    """Aggregate only COCO-native quantities, never cross-function objective values."""
    by_algorithm: dict[str, list[dict]] = {}
    for row in rows:
        by_algorithm.setdefault(row["algorithm_id"], []).append(row)
    out = []
    for algorithm_id in sorted(by_algorithm):
        group = by_algorithm[algorithm_id]
        n = len(group)
        success = sum(row.get("status") == "success" for row in group)
        hits = sum(str(row.get("final_target_hit")).lower() in {"true", "1"}
                   for row in group)
        fe_used = sorted(float(row["fe_used"]) for row in group)
        budgets = sorted(float(row["fe_budget"]) for row in group)
        median = lambda values: values[(len(values) - 1) // 2] if len(values) % 2 else (values[len(values)//2-1] + values[len(values)//2]) / 2
        out.append({
            "algorithm_id": algorithm_id, "n_runs": n, "n_success": success,
            "failure_rate": 1.0 - success / n, "n_final_target_hit": hits,
            "final_target_hit_rate": hits / n,
            "median_fe_used": median(fe_used), "median_fe_budget": median(budgets),
        })
    return out


def write_coco_native_report(index_path, artifact_key: str, out_dir, *, root=".") -> list[dict]:
    """Resolve a formal external index and write its native-only summary CSV."""
    import json
    from .external_canonical_artifacts import resolve_external_analysis_target

    index = json.loads(Path(index_path).read_text())
    target = resolve_external_analysis_target(index, artifact_key, root=root)
    rows = load_coco_native_runs(target["native_runs_path"])
    summary = native_summary(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "coco_native_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    (out_dir / "README.md").write_text(
        "# COCO-native external analysis\n\n"
        "This is supporting external evidence. It uses only COCO-native final-target "
        "hits, FE use and completion. It must not be interpreted as the synthetic "
        "normalized-gap Task-12 primary analysis.\n"
    )
    return summary


__all__ = ["load_coco_native_runs", "native_summary", "write_coco_native_report"]
