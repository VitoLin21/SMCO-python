"""Python single-task worker for the SMCO-EVO high-dim paper (Task 8).

:func:`run_task` takes a manifest task plus its loaded instance and shared
starts, runs the SMCO variant named by the task, and collects FE / quality /
anytime / status into a result payload.

Direction handling: ``HighDimInstance.objective`` is minimisation (optimum 0);
SMCO maximises, so the worker feeds ``g(x) = -instance.objective(x)``. The
returned ``best_value`` is the *minimisation* value (same direction as
``known_optimum``), so ``normalized_gap`` and ``target_hit_fe`` follow the
experiment-plan section 6.1 convention directly.

An objective observer records a best-so-far trace (improvement points only, so
memory stays bounded for multi-million-FE runs). target-hit FE and anytime
checkpoints are derived from that trace, which makes them uniform across every
family — including BR, whose merged context summary does not expose them.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any

import numpy as np

from .experiment_manifests import result_row_from_task
from .highdim_instances import HighDimInstance
from .optimizer import smco, smco_br, smco_br_evo, smco_evo, smco_r, smco_r_evo
from .paper_contract import NONE_TOKEN, parse_algorithm_id

# Minimisation gap targets -> canonical CSV suffixes (paper_contract.RESULT_COLUMNS).
_GAP_TARGETS: dict[str, float] = {"1e-1": 1e-1, "1e-2": 1e-2, "1e-3": 1e-3, "1e-5": 1e-5}
_EPS = 1e-12

_BASE_DISPATCH = {
    ("python", "smco"): smco,
    ("python", "smco_refine"): smco_r,
    ("python", "smco_boost_refine"): smco_br,
}
_EVO_DISPATCH = {
    ("python", "smco"): smco_evo,
    ("python", "smco_refine"): smco_r_evo,
    ("python", "smco_boost_refine"): smco_br_evo,
}


class _AnytimeObserver:
    """Maximisation objective that records a minimisation best-so-far trace.

    Only improvement points are appended, so the trace length is bounded by the
    number of distinct improvements (not the FE budget).
    """

    def __init__(self, instance: HighDimInstance) -> None:
        self.instance = instance
        self.fe = 0
        self.best_min = math.inf
        self.trace: list[tuple[int, float]] = []

    def __call__(self, x: np.ndarray) -> float:
        value = self.instance.objective(x)  # minimisation value
        self.fe += 1
        if value < self.best_min:
            self.best_min = value
            self.trace.append((self.fe, self.best_min))
        return -value  # maximisation for SMCO


def _select_algorithm(task: dict):
    parsed = parse_algorithm_id(task["algorithm_id"])
    if parsed["language"] != "python":
        raise ValueError(
            f"Python worker cannot run language={parsed['language']!r} "
            f"task {task.get('run_id')!r}; route R tasks to the R worker"
        )
    table = _EVO_DISPATCH if parsed["evolutionary"] else _BASE_DISPATCH
    key = ("python", parsed["family"])
    if key not in table:
        raise ValueError(f"no Python dispatch for family={parsed['family']!r}")
    return table[key], parsed


def _first_fe_below_threshold(trace, threshold: float):
    for fe, best_min in trace:
        if best_min <= threshold:
            return fe
    return None


def _best_at_fe(trace, checkpoint_fe: int):
    best = None
    for fe, best_min in trace:
        if fe <= checkpoint_fe:
            best = best_min
        else:
            break
    return best


def _peak_rss_mb() -> float:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is KiB on Linux, bytes on macOS.
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    except Exception:
        return 0.0


def _gap(best_min: float, known_optimum: float, initial_reference: float) -> float:
    return max(best_min - known_optimum, _EPS) / max(initial_reference - known_optimum, _EPS)


def run_task(
    task: dict,
    instance: HighDimInstance,
    starts: np.ndarray,
    *,
    machine_id: str = "",
    git_commit: str = "",
    environment_hash: str = "",
) -> dict:
    """Run one manifest task end-to-end and return the collected result payload."""
    if not isinstance(instance, HighDimInstance):
        raise TypeError("instance must be a HighDimInstance")
    algorithm, parsed = _select_algorithm(task)

    cfg = task["algorithm_config"]
    dim = instance.dimension
    fe_budget = int(task["fe_budget"])
    checkpoints = [int(c) for c in task["checkpoints"]]
    known_optimum = float(instance.known_optimum_value)

    observer = _AnytimeObserver(instance)
    starts = np.asarray(starts, dtype=float)
    initial_reference = float(np.median([instance.objective(s) for s in starts]))

    # SMCO triggers evolution off iter_max (optimizer._evolution_boundaries), so
    # size iter_max to the FE budget: one center-difference iteration costs about
    # 2d+1 evaluations. max_evals remains the hard stop (fe_used <= fe_budget).
    iter_max = max(1, fe_budget // (2 * dim + 1))
    control: dict[str, Any] = {
        "max_evals": fe_budget,
        "objective_sense": "maximize",
        "known_optimum": -known_optimum,  # maximisation-side optimum of g=-f
        "iter_max": iter_max,
        "seed": int(task["seed"]),
    }
    if "refine_ratio" in cfg:
        control["refine_ratio"] = float(cfg["refine_ratio"])
    if parsed["evolutionary"]:
        control["evolution_points"] = tuple(float(p) for p in cfg["evolution_points"])
        control["elimination_rate"] = float(cfg["elimination_rate"])
        control["evolution_strategy"] = cfg["evolution_strategy"]
        control["de_factor"] = float(cfg["de_factor"])
        control["de_crossover"] = float(cfg["de_crossover"])
        control["state_semantics"] = cfg["state_semantics"]

    status = "success"
    failure_reason = NONE_TOKEN
    termination_reason = "evaluation_budget"
    evaluation_counts: dict = {}
    t0 = time.perf_counter()
    try:
        result = algorithm(observer, instance.bounds_lower, instance.bounds_upper, starts, **control)
        fe_summary = result.summary.get("fe", {}) if isinstance(result.summary, dict) else {}
        termination_reason = fe_summary.get("termination_reason") or termination_reason
        evaluation_counts = fe_summary.get("evaluation_counts_by_event", {}) or {}
    except Exception as exc:  # noqa: BLE001 - worker must report, not crash, on algo failure
        status = "algorithm_failure"
        failure_reason = f"{type(exc).__name__}: {exc}"
        termination_reason = "error"

    wall_time = time.perf_counter() - t0
    peak_memory_mb = _peak_rss_mb()

    fe_used = observer.fe
    best_min = observer.best_min if observer.trace else initial_reference
    normalized_gap = _gap(best_min, known_optimum, initial_reference)

    # Targets are RELATIVE to the normalized gap (contract 6 / plan 6.1):
    # target_hit when (best - f*) / (initial_reference - f*) <= target, i.e.
    # best <= f* + target * (initial_reference - f*).
    span = initial_reference - known_optimum
    target_hit: dict[str, int | None] = {}
    for label, target in _GAP_TARGETS.items():
        threshold = known_optimum + target * span
        target_hit[label] = _first_fe_below_threshold(observer.trace, threshold)

    anytime: list[dict] = []
    for cp in checkpoints:
        best_at = _best_at_fe(observer.trace, cp)
        if best_at is None:
            best_at = best_min
        anytime.append(
            {
                "checkpoint_fe": cp,
                "fe_used": min(cp, fe_used),
                "best_value": float(best_at),
                "normalized_gap": _gap(best_at, known_optimum, initial_reference),
            }
        )

    result_row = result_row_from_task(
        task,
        best_value=float(best_min),
        fe_used=fe_used,
        status=status,
        known_optimum=known_optimum,
        normalized_gap=normalized_gap,
        checkpoint_fe=fe_budget,
        target_hit_fe=target_hit,
        wall_time_sec=wall_time,
        peak_memory_mb=peak_memory_mb,
        failure_reason=failure_reason,
        termination_reason=termination_reason,
        fe_counts_by_event=str(evaluation_counts),
        machine_id=machine_id,
        git_commit=git_commit,
        environment_hash=environment_hash,
        objective_sense="minimize",
    )

    return {
        "run_id": task["run_id"],
        "status": status,
        "failure_reason": failure_reason,
        "fe_used": fe_used,
        "fe_budget": fe_budget,
        "best_value": float(best_min),
        "known_optimum": known_optimum,
        "normalized_gap": normalized_gap,
        "target_hit_fe": target_hit,
        "anytime": anytime,
        "termination_reason": termination_reason,
        "evaluation_counts_by_event": evaluation_counts,
        "wall_time_sec": wall_time,
        "peak_memory_mb": peak_memory_mb,
        "result_row": result_row,
    }


__all__ = ["run_task"]
