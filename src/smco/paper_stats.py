"""Paper-level statistics for the SMCO-EVO high-dim analysis (Task 12).

Pure numpy functions over merged/ result rows. Figure rendering and the final
report packaging (Task 13) consume these after the provenance audit passes.

Currently implemented: COCO-style ERT, (hierarchical) bootstrap confidence
intervals, Holm step-down multiple-comparison correction, and probability of
superiority. The ECDF-AUC primary score lives in :mod:`smco.selection`.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

TARGETS = ("1e-1", "1e-2", "1e-3", "1e-5")


def expected_running_time(hit_fes: Sequence, budget: float) -> float:
    """COCO-style Expected Running Time to reach one target.

    ``hit_fes`` are per-run FE values (a reached run's hit FE) or ``None``
    (not reached / failure). Unreached runs contribute their full ``budget`` to
    the total and are excluded from the reached denominator, so ERT grows toward
    infinity as fewer runs succeed. Returns ``inf`` when no run reached.
    """
    reached = [f for f in hit_fes if f is not None]
    n_reached = len(reached)
    if n_reached == 0:
        return math.inf
    total = sum(reached) + sum(budget for f in hit_fes if f is None)
    return float(total) / n_reached


def bootstrap_ci(
    values: Sequence,
    *,
    stat: Callable = np.median,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
):
    """Bootstrap confidence interval for ``stat`` over ``values``.

    Returns ``(point_estimate, lo, hi)``; ``(None, None, None)`` for empty input.
    For a hierarchical bootstrap (resample functions, then instances within),
    pass pre-grouped values via ``stat`` that does the two-level resample.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return (None, None, None)
    point = float(stat(arr))
    boots = np.array(
        [float(stat(rng.choice(arr, size=n, replace=True))) for _ in range(n_boot)]
    )
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boots, alpha * 100))
    hi = float(np.percentile(boots, (1.0 - alpha) * 100))
    return (point, lo, hi)


def hierarchical_bootstrap_ci(
    groups: Sequence[Sequence],
    *,
    stat: Callable = np.mean,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
):
    """Two-level (hierarchical) bootstrap: resample groups, then values within.

    ``groups`` is a sequence of per-group value lists (e.g. one list of instance
    gaps per function). This respects the nested design (function > instance)
    that a flat bootstrap would ignore. Returns ``(point, lo, hi)``.
    """
    rng = np.random.default_rng(seed)
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    if not groups:
        return (None, None, None)
    n_groups = len(groups)
    group_stats = [float(stat(g)) for g in groups]
    point = float(stat(group_stats))

    def _one():
        idx = rng.integers(0, n_groups, size=n_groups)
        sampled = []
        for i in idx:
            g = groups[i]
            sub = g[rng.integers(0, g.size, size=g.size)]
            sampled.append(float(stat(sub)))
        return float(stat(sampled))

    boots = np.array([_one() for _ in range(n_boot)])
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boots, alpha * 100))
    hi = float(np.percentile(boots, (1.0 - alpha) * 100))
    return (point, lo, hi)


def holm_correction(pvalues: Sequence) -> list[float]:
    """Holm step-down adjusted p-values, aligned to the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        prev = max(prev, (m - rank) * pvalues[idx])
        adj[idx] = min(prev, 1.0)
    return adj


def probability_of_superiority(a_values: Sequence, b_values: Sequence):
    """P(A < B) for paired lower-is-better samples (A "solves faster").

    Returns ``None`` for empty input. 0.5 = no effect, 1.0 = A always beats B.
    """
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    n = min(a.size, b.size)
    if n == 0:
        return None
    return float(np.mean(a[:n] < b[:n]))


__all__ = [
    "TARGETS",
    "expected_running_time",
    "bootstrap_ci",
    "hierarchical_bootstrap_ci",
    "holm_correction",
    "probability_of_superiority",
]
