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

from .paper_stats import (
    expected_running_time,
    hierarchical_bootstrap_ci,
    holm_correction,
)
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
    function = r.get("function")
    return {
        "status": r.get("status", "success"),
        "function": function if function not in _NAN_TOKENS else "_",
        "dimension": int(float(dim)) if dim not in _NAN_TOKENS else 1,
        "normalized_gap": float(gap) if gap not in _NAN_TOKENS else None,
        "wall_time_sec": float(wall) if wall not in _NAN_TOKENS else None,
        "target_hit_fe": th,
        "fe_budget": int(float(r["fe_budget"])) if r.get("fe_budget") not in _NAN_TOKENS else 0,
    }


# Pre-registered bootstrap size for the primary table (plan 6.3: >= 10,000).
PRIMARY_BOOTSTRAPS = 10000


def primary_table(rows: list[dict], algorithms, *, n_boot: int = PRIMARY_BOOTSTRAPS) -> list[dict]:
    """Per-algorithm primary statistics: ECDF-AUC, ERT per target, bootstrap CI.

    R3b (plan 6.3): the median log-gap CI is a function->instance hierarchical
    bootstrap (resample functions, then instances within) with the pooled median
    as the point estimate — not a flat bootstrap. ERT uses each run's OWN
    ``fe_budget`` (not ``max(fe_budget)``), since E1 budget scales with dimension.
    """
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
        # hierarchical (function -> instance) bootstrap of the pooled median
        by_func: dict[str, list[float]] = {}
        for r in runs:
            g = r["normalized_gap"]
            if g is None:
                continue
            by_func.setdefault(r["function"], []).append(float(np.log(max(g, 1e-12))))
        groups = list(by_func.values())
        point, lo, hi = (
            hierarchical_bootstrap_ci(groups, stat=np.median, n_boot=n_boot, seed=0, pool=True)
            if groups else (None, None, None)
        )
        budgets = [r["fe_budget"] for r in runs]
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
                [r["target_hit_fe"].get(t) for r in runs], budgets)
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


# --- Task-12 pairwise comparison (Holm-corrected, paired by problem) ---

# Pre-registered bootstrap size for pairwise p-values (smaller than the primary
# table because there are O(n_algo^2) pairs; deterministic via seed).
PAIRWISE_BOOTSTRAPS = 2000

_PAIRWISE_FIELDS = [
    "algorithm_a", "algorithm_b", "n_pairs", "median_log_gap_diff",
    "diff_ci_lo", "diff_ci_hi", "prob_a_better", "prob_b_better",
    "tie_rate", "p_value", "p_holm",
]


def _paired_log_gaps(rows, algorithms) -> dict:
    """{algo: {(function, dim, instance): log(normalized_gap)}} for paired tests.

    Lower is better; log-gap is the same metric the primary table bootstraps."""
    out = {a: {} for a in algorithms}
    for r in rows:
        aid = r.get("algorithm_id")
        if aid not in out:
            continue
        gap = r.get("normalized_gap")
        if gap in _NAN_TOKENS:
            continue
        key = (r.get("function"), int(float(r["dimension"])), int(float(r["instance"])))
        out[aid][key] = float(np.log(max(float(gap), 1e-12)))
    return out


def _paired_median_test(diffs, *, rng, n_boot):
    """Median paired diff + a bootstrap two-sided p (H0: median diff == 0) + CI.

    p = 2 * min(P(boot median <= 0), P(boot median >= 0)), capped at 1. With the
    diffs resampled with replacement, this is a bootstrap sign test on the median.
    Returns (median_diff, p_value, (ci_lo, ci_hi)) or (None,None,None) if empty.
    """
    if not diffs:
        return None, None, (None, None)
    arr = np.asarray(diffs, dtype=float)
    obs = float(np.median(arr))
    n = arr.size
    boots = np.array([float(np.median(rng.choice(arr, size=n, replace=True)))
                      for _ in range(n_boot)])
    p = min(float(np.mean(boots <= 0)), float(np.mean(boots >= 0))) * 2.0
    p = min(p, 1.0)
    return obs, p, (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def pairwise_table(rows, algorithms, *, n_boot: int = PAIRWISE_BOOTSTRAPS,
                   seed: int = 0) -> list[dict]:
    """All pairwise comparisons over shared (function, dim, instance) problems.

    For each unordered pair (A, B): n matched pairs, the median of
    ``log_gap_A - log_gap_B`` (negative => A better), a paired-bootstrap
    two-sided p, its bootstrap 95% CI, ``prob_a_better`` = P(diff < 0) on the
    matched pairs, and the Holm step-down adjusted p across all pairs.

    Lower normalized_gap is better (review fopt-direction-metric). Honesty: the
    table reports whatever the data shows — including a baseline beating the
    proposed method — without altering the protocol.
    """
    paired = _paired_log_gaps(rows, algorithms)
    algos = [a for a in algorithms if a in paired]
    pair_list = [(algos[i], algos[j]) for i in range(len(algos))
                 for j in range(i + 1, len(algos))]
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    pvals: list[float] = []
    for a, b in pair_list:
        shared = set(paired[a]) & set(paired[b])
        diffs = [paired[a][k] - paired[b][k] for k in shared]
        median_diff, p, (lo, hi) = _paired_median_test(diffs, rng=rng, n_boot=n_boot)
        prob_a_better = float(np.mean([d < 0 for d in diffs])) if diffs else None
        prob_b_better = float(np.mean([d > 0 for d in diffs])) if diffs else None
        tie_rate = float(np.mean([d == 0 for d in diffs])) if diffs else None
        out.append({
            "algorithm_a": a, "algorithm_b": b, "n_pairs": len(diffs),
            "median_log_gap_diff": median_diff, "diff_ci_lo": lo, "diff_ci_hi": hi,
            "prob_a_better": prob_a_better, "prob_b_better": prob_b_better,
            "tie_rate": tie_rate, "p_value": p,
        })
        if p is not None:
            pvals.append(p)
    adjusted = holm_correction(pvals) if pvals else []
    pi = 0
    for row in out:
        if row["p_value"] is not None:
            row["p_holm"] = adjusted[pi]
            pi += 1
        else:
            row["p_holm"] = None
    return out


def write_pairwise_table(merged_dir, out_dir, algorithms, **kw) -> list[dict]:
    """Write pairwise_table.csv from merged/ (audit must pass). Returns the rows."""
    rows = load_merged_rows(merged_dir)
    table = pairwise_table(rows, algorithms, **kw)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pairwise_table.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=_PAIRWISE_FIELDS)
        w.writeheader()
        for r in table:
            w.writerow({c: r.get(c, "") for c in _PAIRWISE_FIELDS})
    return table


__all__ = [
    "load_merged_rows", "primary_table", "write_primary_table",
    "pairwise_table", "write_pairwise_table",
]
