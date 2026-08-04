"""Confirmatory E2 winner-versus-base analyses for Task 12.

This module intentionally consumes the canonical E2 artifact rather than an
arbitrary CSV.  A paired problem is one ``(function, dimension, instance)``
cell; checkpoints are never observations in these analyses.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .canonical_artifacts import (
    file_sha256, index_sha256, resolve_analysis_target, validate_canonical_index,
)

ANALYSIS_VERSION = "e2-hypotheses-v1"
DEFAULT_BOOTSTRAPS = 10_000
DEFAULT_SEED = 20260802


def _as_log_gap(row: dict) -> float:
    value = float(row["normalized_gap"])
    if not np.isfinite(value):
        raise ValueError("normalized_gap must be finite")
    return float(np.log(max(value, 1e-12)))


def paired_e2_cells(rows: list[dict], winner: str, base: str) -> list[dict]:
    """Return one paired E2 row per problem, with positive gain favouring winner."""
    by_key: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        algorithm = row.get("algorithm_id")
        if algorithm not in {winner, base}:
            continue
        try:
            key = (row["function"], int(float(row["dimension"])),
                   int(float(row["instance"])))
            _as_log_gap(row)
        except (KeyError, ValueError):
            continue
        values = by_key.setdefault(key, {})
        if algorithm in values:
            raise ValueError(f"duplicate E2 algorithm observation for {key}: {algorithm}")
        values[algorithm] = row
    missing = [key for key, values in by_key.items() if set(values) != {winner, base}]
    if missing:
        raise ValueError(f"E2 has unpaired winner/base cells (first: {missing[0]!r})")
    cells = []
    for (function, dimension, instance), values in sorted(by_key.items()):
        winner_log = _as_log_gap(values[winner])
        base_log = _as_log_gap(values[base])
        cells.append({
            "function": function, "dimension": dimension, "instance": instance,
            "winner_log_gap": winner_log, "base_log_gap": base_log,
            # Positive means the winner has a smaller log gap (is better).
            "log_gap_gain": base_log - winner_log,
            "winner_status": values[winner].get("status", ""),
            "base_status": values[base].get("status", ""),
        })
    if not cells:
        raise ValueError("no paired E2 winner/base cells")
    return cells


def _hierarchical_resample(cells: list[dict], rng: np.random.Generator) -> list[dict]:
    """Resample functions, then problem cells within each selected function."""
    by_function: dict[str, list[dict]] = {}
    for cell in cells:
        by_function.setdefault(cell["function"], []).append(cell)
    names = sorted(by_function)
    chosen = rng.integers(0, len(names), size=len(names))
    out = []
    for index in chosen:
        group = by_function[names[int(index)]]
        sampled = rng.integers(0, len(group), size=len(group))
        out.extend(group[int(i)] for i in sampled)
    return out


def _ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def h1_summary(cells: list[dict], *, n_boot: int = DEFAULT_BOOTSTRAPS,
               seed: int = DEFAULT_SEED) -> dict:
    """H1 paired effect: winner minus base expressed as positive log-gap gain."""
    gains = np.asarray([c["log_gap_gain"] for c in cells], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        float(np.median([c["log_gap_gain"] for c in _hierarchical_resample(cells, rng)]))
        for _ in range(n_boot)
    ])
    lo, hi = _ci(boot)
    return {
        "hypothesis": "H1_winner_vs_matched_base",
        "metric": "base_log_gap_minus_winner_log_gap",
        "direction": "positive_favours_winner",
        "n_paired_cells": len(cells),
        "median_log_gap_gain": float(np.median(gains)),
        "ci95_lo": lo, "ci95_hi": hi,
        "probability_winner_better": float(np.mean(gains > 0)),
        "ties": int(np.sum(gains == 0)),
        "bootstrap": "function_then_problem_cell_hierarchical",
        "n_boot": n_boot, "seed": seed,
    }


def _slope(cells: list[dict]) -> float:
    x = np.log(np.asarray([c["dimension"] for c in cells], dtype=float))
    y = np.asarray([c["log_gap_gain"] for c in cells], dtype=float)
    if np.unique(x).size < 2:
        raise ValueError("H2 needs at least two distinct dimensions")
    return float(np.polyfit(x, y, 1)[0])


def h2_dimension_trend(cells: list[dict], *, n_boot: int = DEFAULT_BOOTSTRAPS,
                       seed: int = DEFAULT_SEED + 1) -> tuple[dict, list[dict]]:
    """H2 fixed OLS slope of paired gain versus log dimension, with hierarchical CI."""
    rng = np.random.default_rng(seed)
    boot = np.asarray([_slope(_hierarchical_resample(cells, rng)) for _ in range(n_boot)])
    lo, hi = _ci(boot)
    by_dimension = []
    for dimension in sorted({c["dimension"] for c in cells}):
        values = [c["log_gap_gain"] for c in cells if c["dimension"] == dimension]
        by_dimension.append({
            "dimension": dimension, "log_dimension": float(np.log(dimension)),
            "n_paired_cells": len(values), "median_log_gap_gain": float(np.median(values)),
            "mean_log_gap_gain": float(np.mean(values)),
            "probability_winner_better": float(np.mean(np.asarray(values) > 0)),
        })
    return ({
        "hypothesis": "H2_gain_vs_log_dimension",
        "outcome": "base_log_gap_minus_winner_log_gap",
        "direction": "positive_slope_means_winner_advantage_grows_with_dimension",
        "model": "ordinary_least_squares_on_paired_problem_cells",
        "n_paired_cells": len(cells),
        "slope_per_log_dimension": _slope(cells), "ci95_lo": lo, "ci95_hi": hi,
        "bootstrap": "function_then_problem_cell_hierarchical",
        "n_boot": n_boot, "seed": seed,
    }, by_dimension)


def h3_instance_provenance(cells: list[dict], instance_index: dict) -> tuple[dict, list[dict]]:
    """Check whether instance metadata identifies separable transform factors.

    The confirmatory generator records hashes for shift/permutation/block
    rotation, but does not label a no-shift/no-permutation/no-rotation control.
    Consequently it supports a provenance audit, not a causal factor contrast.
    """
    entries = instance_index.get("instances")
    if not isinstance(entries, list):
        raise ValueError("instance index has no instances list")
    lookup = {(e.get("function"), int(e.get("dimension", -1)), int(e.get("instance_id", -1))): e
              for e in entries}
    rows = []
    for cell in cells:
        key = (cell["function"], cell["dimension"], cell["instance"])
        entry = lookup.get(key)
        if entry is None:
            raise ValueError(f"instance provenance missing for E2 cell {key!r}")
        hashes = entry.get("file_hashes", {})
        rows.append({
            "function": key[0], "dimension": key[1], "instance": key[2],
            "transform_sha256": entry.get("transform_sha256", ""),
            "has_shift": bool(hashes.get("shift")),
            "has_permutation": bool(hashes.get("permutation")),
            "has_block_rotation": bool(hashes.get("rotation_blocks")),
            "log_gap_gain": cell["log_gap_gain"],
        })
    factors = ("has_shift", "has_permutation", "has_block_rotation")
    separable = all(len({row[factor] for row in rows}) > 1 for factor in factors)
    return ({
        "hypothesis": "H3_transform_robustness",
        "status": "tested" if separable else "not_testable",
        "n_paired_cells": len(rows),
        "required_factors": list(factors),
        "reason": ("all transform factors have both levels" if separable else
                   "confirmatory instances all carry shift, permutation, and block-rotation artifacts; "
                   "there is no frozen factor-level control, so a transform-factor effect would be fabricated"),
    }, rows)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_e2_hypotheses(*, canonical_index_path, out_dir, instance_index_path,
                      n_boot: int = DEFAULT_BOOTSTRAPS, seed: int = DEFAULT_SEED) -> dict:
    """Run the frozen E2 H1/H2/H3 analysis and write reproducible artifacts."""
    index_path = Path(canonical_index_path)
    index = json.loads(index_path.read_text())
    errors = validate_canonical_index(index)
    if errors:
        raise ValueError("canonical index invalid:\n  " + "\n  ".join(errors))
    target = resolve_analysis_target(index, "e2_merged")
    manifest_path = Path(target["source_manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    winner, base = manifest.get("winner_algorithm"), manifest.get("matched_base_algorithm")
    if not isinstance(winner, str) or not isinstance(base, str) or winner == base:
        raise ValueError("frozen E2 manifest must name distinct winner_algorithm and matched_base_algorithm")
    merged_dir = Path(target["merged_dir"])
    rows = list(csv.DictReader(open(merged_dir / "valid_runs.csv", newline="")))
    observed = {r.get("algorithm_id") for r in rows}
    if observed != {winner, base}:
        raise ValueError("E2 merged algorithms do not exactly match frozen manifest")
    cells = paired_e2_cells(rows, winner, base)
    index_data = json.loads(Path(instance_index_path).read_text())
    h1 = h1_summary(cells, n_boot=n_boot, seed=seed)
    h2, dimension_rows = h2_dimension_trend(cells, n_boot=n_boot, seed=seed + 1)
    h3, provenance_rows = h3_instance_provenance(cells, index_data)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "paired_cells.csv", cells)
    _write_csv(out / "h2_dimension_trend.csv", dimension_rows)
    _write_csv(out / "h3_instance_provenance.csv", provenance_rows)
    summary = {"analysis_version": ANALYSIS_VERSION, "winner_algorithm": winner,
               "matched_base_algorithm": base, "h1": h1, "h2": h2, "h3": h3}
    (out / "hypotheses.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    provenance = {
        "analysis_version": ANALYSIS_VERSION, "canonical_index_path": str(index_path),
        "canonical_index_sha256": index_sha256(index), "canonical_index_file_sha256": file_sha256(index_path),
        "artifact_key": "e2_merged", "merged_dir": str(merged_dir),
        "valid_runs_sha256": file_sha256(merged_dir / "valid_runs.csv"),
        "manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
        "instance_index_path": str(instance_index_path), "instance_index_sha256": file_sha256(instance_index_path),
        "bootstrap_seed": seed, "n_boot": n_boot, "n_paired_cells": len(cells),
        "failure_handling": "Rows with non-finite normalized_gap are rejected; unpaired or duplicate cells fail analysis.",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    lines = ["# E2 confirmatory hypotheses (Task 12)", "",
             f"Frozen comparison: `{winner}` vs `{base}`; {len(cells)} paired function×dimension×instance cells.", "",
             "## H1: winner versus matched base", "",
             "Metric is `base_log_gap − winner_log_gap`; positive values favour SMCO-EVO.", "",
             f"Median gain: {h1['median_log_gap_gain']:.6g} (hierarchical 95% CI {h1['ci95_lo']:.6g}, {h1['ci95_hi']:.6g}); "+
             f"P(winner better): {h1['probability_winner_better']:.3f}.", "",
             "## H2: gain versus dimension", "",
             f"OLS slope per log(dimension): {h2['slope_per_log_dimension']:.6g} (hierarchical 95% CI {h2['ci95_lo']:.6g}, {h2['ci95_hi']:.6g}).", "",
             "## H3: transformed-instance robustness", "",
             f"Status: **{h3['status']}**. {h3['reason']}", "",
             "Input hashes, bootstrap seed, sample count, and failure handling are in `provenance.json`. "
             "Cell-level inputs and all dimension/provenance summaries are in the CSV files."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return summary


__all__ = ["paired_e2_cells", "h1_summary", "h2_dimension_trend", "h3_instance_provenance", "run_e2_hypotheses"]
