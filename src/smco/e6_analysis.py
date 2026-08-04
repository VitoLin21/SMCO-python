"""Dedicated, frozen-protocol Task-12 analyses for the E6 ablations.

E6 deliberately reuses ``PY-SP-SMCO-EVO`` as its algorithm id while varying
configuration.  It therefore must not be passed through the generic primary
table, which groups only by algorithm id.  This module binds the two analyses
to their canonical artifacts and to the cohorts fixed in the high-dimensional
plan.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .canonical_artifacts import resolve_analysis_target, validate_canonical_index
from .paper_analysis import _row_to_payload, pairwise_table, primary_table
from .selection import ecdf_auc


# E6.2: exactly one 60-problem configuration per strategy.  The other three
# rand1bin configurations are schedule candidates and must never be pooled.
STRATEGY_PRIMARY = {
    "rand1bin": "0ca95d61e9e2b475",
    "current-to-best1bin": "9fe8239f3a19ae93",
    "best1bin": "d871ee17b09db4a8",
    "sobol": "abf0cf7c515e1300",
}
STRATEGY_EXCLUDED_RAND1BIN = {
    "e23c98957fb38635", "103c21385e3de97f", "06b3463f5d507e0f",
}

# E6.1: H4 is strictly 8 starts versus ceil(sqrt(d)).  Sixteen starts remains
# a secondary sensitivity cohort and is never included in the non-inferiority
# result.
START_COUNT_EIGHT = "0ca95d61e9e2b475"
START_COUNT_SQRT = {
    200: (15, "4a6a9cf11817c7d3"),
    500: (23, "c15a07c377310507"),
    1000: (32, "51c3a441984f948b"),
}
START_COUNT_SIXTEEN = "32e695f7a5fcfb44"
H4_MARGIN = math.log(1.10)
H4_BOOTSTRAPS = 10_000


def _read_rows(merged_dir) -> list[dict]:
    return list(csv.DictReader(open(Path(merged_dir) / "valid_runs.csv", newline="")))


def _problem_key(row: dict) -> tuple:
    return (row.get("function"), int(float(row["dimension"])), int(float(row["instance"])))


def _check_exact_cells(label: str, rows: list[dict], *, expected_n: int = 60) -> None:
    cells = [_problem_key(r) for r in rows]
    if len(rows) != expected_n or len(set(cells)) != expected_n:
        raise ValueError(
            f"{label}: expected exactly {expected_n} unique problem cells; "
            f"got rows={len(rows)}, cells={len(set(cells))}")


def _label_rows(rows: list[dict], label: str) -> list[dict]:
    """Copy rows, replacing only the grouping id used by generic statistics."""
    return [{**r, "algorithm_id": label} for r in rows]


def resolve_e6_targets(canonical_index) -> tuple[dict, dict]:
    """Validate an index and resolve the only permitted E6 inputs from it."""
    if isinstance(canonical_index, (str, Path)):
        canonical_index = json.loads(Path(canonical_index).read_text())
    errors = validate_canonical_index(canonical_index)
    if errors:
        raise ValueError("canonical index invalid:\n  " + "\n  ".join(errors))
    strategy = resolve_analysis_target(canonical_index, "e6_strategy_merged")
    starts = resolve_analysis_target(canonical_index, "e6_start_count_merged")
    if strategy.get("analysis_kind") != "strategy_ablation":
        raise ValueError("e6_strategy_merged has unexpected analysis_kind")
    if starts.get("analysis_kind") != "start_count_ablation":
        raise ValueError("e6_start_count_merged has unexpected analysis_kind")
    return strategy, starts


def strategy_cohorts(rows: list[dict]) -> dict[str, list[dict]]:
    """Select and validate the pre-frozen four-by-60 E6 strategy cohort."""
    by_hash: dict[str, list[dict]] = {}
    for row in rows:
        by_hash.setdefault(row.get("configuration_hash", ""), []).append(row)
    unexpected_excluded = STRATEGY_EXCLUDED_RAND1BIN & set(STRATEGY_PRIMARY.values())
    if unexpected_excluded:
        raise AssertionError("strategy exclusion overlaps primary cohort")
    cohorts = {}
    for label, config_hash in STRATEGY_PRIMARY.items():
        selected = by_hash.get(config_hash, [])
        _check_exact_cells(f"strategy {label}", selected)
        if any(r.get("evolution_strategy") != label for r in selected):
            raise ValueError(f"strategy {label}: configuration hash has mismatched strategy")
        cohorts[label] = selected
    if any(by_hash.get(h) for h in STRATEGY_EXCLUDED_RAND1BIN):
        # Presence is expected, but make its exclusion explicit and testable.
        primary_rows = {id(r) for group in cohorts.values() for r in group}
        if any(id(r) in primary_rows for h in STRATEGY_EXCLUDED_RAND1BIN for r in by_hash[h]):
            raise ValueError("schedule-candidate rand1bin rows leaked into primary cohort")
    cell_sets = [set(map(_problem_key, rows)) for rows in cohorts.values()]
    if any(cells != cell_sets[0] for cells in cell_sets[1:]):
        raise ValueError("strategy primary cohorts are not problem matched")
    return cohorts


def start_count_cohorts(rows: list[dict]) -> dict[str, list[dict]]:
    """Select exact H4 primary cohorts plus the n=16 secondary sensitivity set."""
    eight = [r for r in rows if r.get("configuration_hash") == START_COUNT_EIGHT]
    sixteen = [r for r in rows if r.get("configuration_hash") == START_COUNT_SIXTEEN]
    sqrt_rows = []
    for dim, (n_starts, config_hash) in START_COUNT_SQRT.items():
        selected = [r for r in rows if r.get("configuration_hash") == config_hash]
        if any(int(float(r["dimension"])) != dim or int(float(r["n_starts"])) != n_starts
               for r in selected):
            raise ValueError(f"sqrt({dim}) cohort does not match frozen n_starts/configuration")
        _check_exact_cells(f"sqrt({dim})", selected, expected_n=20)
        sqrt_rows.extend(selected)
    _check_exact_cells("n_starts=8", eight)
    _check_exact_cells("n_starts=16 secondary", sixteen)
    _check_exact_cells("sqrt(d)", sqrt_rows)
    all_cells = set(map(_problem_key, eight))
    if set(map(_problem_key, sqrt_rows)) != all_cells:
        raise ValueError("H4 n=8 and sqrt(d) cohorts are not problem matched")
    if set(map(_problem_key, sixteen)) != all_cells:
        raise ValueError("secondary n=16 cohort is not problem matched")
    return {"n_starts=8": eight, "sqrt(d)": sqrt_rows, "n_starts=16 (secondary)": sixteen}


def _efficiency_summary(rows: list[dict]) -> dict:
    payloads = [_row_to_payload(r) for r in rows]
    out = {"n_runs": len(rows), "ecdf_auc": ecdf_auc(payloads)}
    fe_used = np.asarray([float(r["fe_used"]) for r in rows], dtype=float)
    budgets = np.asarray([float(r["fe_budget"]) for r in rows], dtype=float)
    out["median_fe_used"] = float(np.median(fe_used))
    out["median_fe_fraction"] = float(np.median(fe_used / budgets))
    for target in ("1e-1", "1e-2", "1e-3", "1e-5"):
        hits = [r.get(f"target_hit_fe_{target}") not in (None, "", "none", "None") for r in rows]
        out[f"target_hit_rate_{target}"] = float(np.mean(hits))
    return out


def strategy_analysis(rows: list[dict], *, n_boot: int = H4_BOOTSTRAPS) -> tuple[list[dict], list[dict]]:
    """Four-way E6.2 primary cohort summary and all-six paired Holm tests."""
    cohorts = strategy_cohorts(rows)
    labels = list(STRATEGY_PRIMARY)
    labelled = [r for label in labels for r in _label_rows(cohorts[label], label)]
    summary = primary_table(labelled, labels, n_boot=n_boot)
    for row in summary:
        label = row["algorithm_id"]
        row["cohort"] = label
        row["configuration_hash"] = STRATEGY_PRIMARY[label]
        row.update(_efficiency_summary(cohorts[label]))
    pairs = pairwise_table(labelled, labels)
    return summary, pairs


def _hierarchical_upper(groups: list[list[float]], *, n_boot: int, seed: int = 0) -> tuple[float, float]:
    """Function -> instance bootstrap for a one-sided upper 95% bound."""
    grouped = [np.asarray(g, dtype=float) for g in groups if g]
    if not grouped:
        raise ValueError("no paired differences for H4")
    point = float(np.median(np.concatenate(grouped)))
    rng = np.random.default_rng(seed)
    n_groups = len(grouped)
    values = np.empty(n_boot)
    for i in range(n_boot):
        choice = rng.integers(0, n_groups, size=n_groups)
        sample = [grouped[j][rng.integers(0, grouped[j].size, size=grouped[j].size)]
                  for j in choice]
        values[i] = float(np.median(np.concatenate(sample)))
    return point, float(np.percentile(values, 95.0))


def start_count_analysis(rows: list[dict], *, n_boot: int = H4_BOOTSTRAPS) -> tuple[list[dict], dict]:
    """H4 non-inferiority (n=8 vs sqrt(d)) plus n=16 secondary sensitivity."""
    cohorts = start_count_cohorts(rows)
    summary = []
    for label, cohort in cohorts.items():
        item = {"cohort": label, "configuration_hashes": ""}
        if label == "n_starts=8":
            item["configuration_hashes"] = START_COUNT_EIGHT
            item["analysis_role"] = "H4_control"
        elif label == "sqrt(d)":
            item["configuration_hashes"] = ";".join(h for _n, h in START_COUNT_SQRT.values())
            item["analysis_role"] = "H4_comparator"
        else:
            item["configuration_hashes"] = START_COUNT_SIXTEEN
            item["analysis_role"] = "secondary_sensitivity"
        item.update(_efficiency_summary(cohort))
        summary.append(item)

    a = {_problem_key(r): float(np.log(max(float(r["normalized_gap"]), 1e-12)))
         for r in cohorts["n_starts=8"]}
    b = {_problem_key(r): float(np.log(max(float(r["normalized_gap"]), 1e-12)))
         for r in cohorts["sqrt(d)"]}
    if set(a) != set(b):
        raise ValueError("H4 pairing unexpectedly incomplete")
    by_function: dict[str, list[float]] = {}
    for key in sorted(a):
        by_function.setdefault(key[0], []).append(a[key] - b[key])
    point, upper = _hierarchical_upper(list(by_function.values()), n_boot=n_boot)
    h4 = {
        "comparison": "n_starts=8 minus sqrt(d)",
        "metric": "paired_log_normalized_gap_difference",
        "lower_is_better": True,
        "n_pairs": len(a),
        "n_functions": len(by_function),
        "median_diff": point,
        "one_sided_95_upper": upper,
        "noninferiority_margin": H4_MARGIN,
        "noninferior": bool(upper <= H4_MARGIN),
        "bootstrap": "function_to_instance_hierarchical",
        "n_bootstrap": n_boot,
        "secondary_n16_included_in_h4": False,
    }
    return summary, h4


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else []
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(out_dir: Path, strategy_rows: list[dict], start_rows: list[dict]) -> list[Path]:
    """Small, deterministic ECDF figures; wall-clock time is intentionally absent."""
    import matplotlib.pyplot as plt

    paths = []
    for name, grouped in (("strategy_ecdf", strategy_rows), ("start_count_ecdf", start_rows)):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        by_label: dict[str, list[float]] = {}
        for row in grouped:
            label = row["_cohort"]
            by_label.setdefault(label, []).append(float(np.log10(max(float(row["normalized_gap"]), 1e-12))))
        for label, values in by_label.items():
            values = np.sort(values)
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=label)
        ax.set_xlabel("log10(normalized gap), lower is better")
        ax.set_ylabel("ECDF")
        ax.set_title(name.replace("_", " "))
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def write_e6_analysis(canonical_index, out_dir, *, n_boot: int = H4_BOOTSTRAPS,
                      figures: bool = True) -> dict:
    """Write the full dedicated E6 analysis using only canonical artifacts."""
    strategy_target, start_target = resolve_e6_targets(canonical_index)
    strategy_rows = _read_rows(strategy_target["merged_dir"])
    start_rows = _read_rows(start_target["merged_dir"])
    strategy_summary, strategy_pairs = strategy_analysis(strategy_rows, n_boot=n_boot)
    start_summary, h4 = start_count_analysis(start_rows, n_boot=n_boot)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "strategy_summary.csv", strategy_summary)
    _write_csv(out_dir / "strategy_pairwise.csv", strategy_pairs)
    _write_csv(out_dir / "start_count_summary.csv", start_summary)
    (out_dir / "h4_noninferiority.json").write_text(json.dumps(h4, indent=2) + "\n")
    fig_paths = _write_figures(
        out_dir,
        [{**r, "_cohort": label} for label, cohort in strategy_cohorts(strategy_rows).items() for r in cohort],
        [{**r, "_cohort": label} for label, cohort in start_count_cohorts(start_rows).items() for r in cohort],
    ) if figures else []
    report = [
        "# E6 dedicated ablation analysis",
        "",
        "Inputs were resolved from the validated canonical artifact index. Strategy is a frozen "
        "four-cohort comparison (60 problem-matched cells per strategy); the three extra rand1bin "
        "configurations are schedule candidates and are excluded.",
        "",
        f"H4 compares n=8 to ceil(sqrt(d)) on 60 matched cells. Margin: log(1.10) = {H4_MARGIN:.8f}. "
        f"One-sided hierarchical-bootstrap upper bound: {h4['one_sided_95_upper']:.8f}; "
        f"non-inferior: {h4['noninferior']}.",
        "",
        "n=16 is secondary sensitivity only; it is not included in the H4 decision. FE usage and target-hit "
        "rates are reported, while wall time is not a primary conclusion. E6 schedule and state-reset remain deferred.",
    ]
    (out_dir / "report.md").write_text("\n".join(report) + "\n")
    return {"strategy_summary": strategy_summary, "strategy_pairwise": strategy_pairs,
            "start_count_summary": start_summary, "h4": h4,
            "figures": [str(p) for p in fig_paths]}


__all__ = [
    "STRATEGY_PRIMARY", "STRATEGY_EXCLUDED_RAND1BIN", "START_COUNT_EIGHT",
    "START_COUNT_SQRT", "START_COUNT_SIXTEEN", "H4_MARGIN", "strategy_cohorts",
    "start_count_cohorts", "strategy_analysis", "start_count_analysis", "write_e6_analysis",
    "resolve_e6_targets",
]
