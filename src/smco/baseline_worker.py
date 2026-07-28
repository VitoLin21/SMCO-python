"""Comparison-baseline single-task worker for the high-dim paper (Task 9 / E3).

Runs one strong black-box baseline (DE / GA / PSO / SA / GenSA from
``src/comparison/methods``) on a loaded high-dim instance under the SAME
function-evaluation budget as SMCO. FE parity is enforced by a minimisation
observer that raises :class:`EvaluationBudgetExceeded` at ``fe_budget``, which
stops the underlying optimiser (scipy or custom loop); the worker then reports
the best-so-far seen up to that point.

The output payload mirrors :mod:`smco.highdim_worker` (best_value in
minimisation sense, relative target_hit_fe, anytime checkpoints) so the merge
step (Task 11) treats SMCO and baseline rows uniformly. ``algorithm_id`` here is
the baseline name (DE / GenSA / ...), not a SMCO algorithm_id, so the SMCO
result-row contract does not apply to baseline rows.

Note: full E3 batch dispatch + frozen-baseline manifest integration is Task 10;
this module is the per-task engine plus a thin CLI.
"""

from __future__ import annotations

import math
import time

import numpy as np

from comparison.methods.de import differential_evo
from comparison.methods.ga import genetic_algorithm
from comparison.methods.gensa import gensa
from comparison.methods.pso import particle_swarm
from comparison.methods.sa import simulated_annealing
from .evaluation import EvaluationBudgetExceeded
from .highdim_instances import HighDimInstance
from .paper_contract import NONE_TOKEN

_BASELINE_DISPATCH = {
    "DE": differential_evo,
    "GA": genetic_algorithm,
    "PSO": particle_swarm,
    "SA": simulated_annealing,
    "GenSA": gensa,
}

_GAP_TARGETS = {"1e-1": 1e-1, "1e-2": 1e-2, "1e-3": 1e-3, "1e-5": 1e-5}
_EPS = 1e-12


class _MinObserver:
    """Minimisation objective that counts FE and stops hard at ``max_evals``."""

    def __init__(self, instance: HighDimInstance, max_evals: int) -> None:
        self.instance = instance
        self.max_evals = int(max_evals)
        self.fe = 0
        self.best_min = math.inf
        self.trace: list[tuple[int, float]] = []

    def __call__(self, x: np.ndarray) -> float:
        if self.fe >= self.max_evals:
            raise EvaluationBudgetExceeded(
                f"baseline FE budget {self.max_evals} reached"
            )
        value = self.instance.objective(x)
        self.fe += 1
        if value < self.best_min:
            self.best_min = value
            self.trace.append((self.fe, self.best_min))
        return value


def _first_fe_below_threshold(trace, threshold):
    for fe, best_min in trace:
        if best_min <= threshold:
            return fe
    return None


def _best_at_fe(trace, checkpoint_fe):
    best = None
    for fe, best_min in trace:
        if fe <= checkpoint_fe:
            best = best_min
        else:
            break
    return best


def _gap(best_min, known_optimum, initial_reference):
    return max(best_min - known_optimum, _EPS) / max(initial_reference - known_optimum, _EPS)


def run_baseline_task(
    algorithm_name: str,
    instance: HighDimInstance,
    starts: np.ndarray,
    *,
    fe_budget: int,
    seed: int,
    checkpoints,
    stage: str = "e3_baselines_highdim",
    machine_id: str = "",
    git_commit: str = "",
    environment_hash: str = "",
) -> dict:
    """Run one baseline under the SMCO FE budget and return the result payload."""
    if not isinstance(instance, HighDimInstance):
        raise TypeError("instance must be a HighDimInstance")
    if algorithm_name not in _BASELINE_DISPATCH:
        raise ValueError(f"unknown baseline: {algorithm_name!r}")
    algorithm = _BASELINE_DISPATCH[algorithm_name]

    observer = _MinObserver(instance, fe_budget)
    starts = np.asarray(starts, dtype=float)
    initial_reference = float(np.median([instance.objective(s) for s in starts]))
    known_optimum = float(instance.known_optimum_value)

    status = "success"
    failure_reason = NONE_TOKEN
    t0 = time.perf_counter()
    try:
        algorithm(
            observer,
            instance.bounds_lower,
            instance.bounds_upper,
            start_points=starts,
            maximize=False,
            max_iter=max(int(fe_budget), 1000),
            seed=int(seed),
        )
    except EvaluationBudgetExceeded:
        pass  # expected hard stop at the FE budget
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        status = "algorithm_failure"
        failure_reason = f"{type(exc).__name__}: {exc}"

    wall_time = time.perf_counter() - t0
    fe_used = observer.fe
    best_min = observer.best_min if observer.trace else initial_reference
    normalized_gap = _gap(best_min, known_optimum, initial_reference)

    span = initial_reference - known_optimum
    target_hit = {
        label: _first_fe_below_threshold(observer.trace, known_optimum + target * span)
        for label, target in _GAP_TARGETS.items()
    }

    anytime = []
    for cp in checkpoints:
        cp = int(cp)
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

    return {
        "algorithm_id": algorithm_name,
        "stage": stage,
        "function": instance.function_name,
        "dimension": instance.dimension,
        "known_optimum": known_optimum,
        "status": status,
        "failure_reason": failure_reason,
        "fe_budget": int(fe_budget),
        "fe_used": fe_used,
        "best_value": float(best_min),
        "normalized_gap": normalized_gap,
        "target_hit_fe": target_hit,
        "anytime": anytime,
        "wall_time_sec": wall_time,
        "machine_id": machine_id,
        "git_commit": git_commit,
        "environment_hash": environment_hash,
    }


__all__ = ["run_baseline_task", "BASELINE_NAMES"]

BASELINE_NAMES = tuple(_BASELINE_DISPATCH.keys())
