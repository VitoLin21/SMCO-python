"""Task-12 figures for the SMCO-EVO high-dim paper (review §9).

Renders the standard COCO-style empirical-CDF-of-target-hit figures from the
audited merged valid_runs.csv: for a precision target tau and each dimension,
the fraction of (function x instance) runs that have reached tau by FE budget,
plotted per algorithm (right-censored runs never reach fraction 1). ERT per
target is already in primary_table.csv; this module adds the visual ECDF.

Uses the Agg backend (no display). Pure read over merged/ — no new compute.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (before pyplot import)
import matplotlib.pyplot as plt
import numpy as np

from .paper_analysis import load_merged_rows
from .selection import TARGETS

_NAN_TOKENS = ("", None, "none", "nan", "None", "NaN")


def _hit_fe_for_target(rows, algorithms, target, dimension) -> dict:
    """{algo: [hit_fe or None]} for runs at ``dimension`` on target ``target``.

    None == right-censored (target never reached within budget)."""
    col = f"target_hit_fe_{target}"
    out: dict[str, list] = {a: [] for a in algorithms}
    for r in rows:
        aid = r.get("algorithm_id")
        if aid not in out:
            continue
        dim = r.get("dimension")
        if dim in _NAN_TOKENS:
            continue
        if int(float(dim)) != int(dimension):
            continue
        v = r.get(col)
        if v in _NAN_TOKENS:
            out[aid].append(None)
        else:
            try:
                out[aid].append(int(float(v)))
            except (TypeError, ValueError):
                out[aid].append(None)
    return out


def _max_budget_at_dim(rows, dimension) -> int:
    """Largest fe_budget among runs at ``dimension`` (the x-axis upper bound)."""
    budgets = []
    for r in rows:
        dim = r.get("dimension")
        if dim in _NAN_TOKENS or int(float(dim)) != int(dimension):
            continue
        b = r.get("fe_budget")
        if b not in _NAN_TOKENS:
            try:
                budgets.append(int(float(b)))
            except (TypeError, ValueError):
                pass
    return max(budgets) if budgets else int(dimension) * 1000


def ecdf_curves(hit_fe, *, max_budget) -> dict:
    """Per-algorithm ECDF step points: {algo: (budgets, fractions)}.

    fraction(b) = #{runs with hit_fe <= b} / n_runs (censored runs never solve).
    Budgets span [0, max_budget]; the curve is a right-continuous step function.
    """
    curves: dict[str, tuple[list, list]] = {}
    for algo, hits in hit_fe.items():
        n = len(hits)
        if n == 0:
            curves[algo] = ([], [])
            continue
        reached = sorted(h for h in hits if h is not None)
        budgets = [0] + reached + [max_budget]
        fractions = []
        for b in budgets:
            fractions.append(sum(1 for h in reached if h <= b) / n)
        curves[algo] = (budgets, fractions)
    return curves


def plot_ecdf_target(merged_dir, out_dir, algorithms, target, *,
                     dimensions=(200, 500, 1000)) -> Path:
    """Render ecdf_target_{target}.png: one subplot per dimension, ECDF per algo.

    The x-axis upper bound per dimension is derived from the runs' own
    ``fe_budget`` at that dimension. Returns the PNG path.
    """
    rows = load_merged_rows(merged_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(dimensions), figsize=(5 * len(dimensions), 4),
                             sharey=True, squeeze=False)
    for ax, dim in zip(axes[0], dimensions):
        max_budget = _max_budget_at_dim(rows, dim)
        hit_fe = _hit_fe_for_target(rows, algorithms, target, dim)
        curves = ecdf_curves(hit_fe, max_budget=max_budget)
        for algo in algorithms:
            xs, ys = curves.get(algo, ([], []))
            if xs:
                ax.step(xs, ys, where="post", label=algo)
        ax.set_title(f"d = {dim}")
        ax.set_xlabel("FE budget")
        ax.set_xlim(0, max_budget)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel(f"fraction solved (target {target})")
    axes[0, -1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path = out_dir / f"ecdf_target_{target}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_ecdf_figures(merged_dir, out_dir, algorithms, *,
                       targets=TARGETS, dimensions=(200, 500, 1000)) -> list[Path]:
    """Render an ECDF figure for every target in ``targets``. Returns the PNG paths."""
    paths = []
    for t in targets:
        paths.append(plot_ecdf_target(
            merged_dir, out_dir, algorithms, t, dimensions=dimensions))
    return paths


__all__ = [
    "ecdf_curves",
    "plot_ecdf_target",
    "write_ecdf_figures",
]
