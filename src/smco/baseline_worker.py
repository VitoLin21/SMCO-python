"""Comparison-baseline single-task worker for the high-dim paper (E3 / E7).

Runs one frozen comparator on a loaded high-dim instance under the SAME
function-evaluation budget as SMCO. E7 adds R-DEoptim, STOGO, L-BFGS, SPSA and
SignGD without changing the identity of the older Python ``DE``. FE parity is
enforced by one minimisation observer that sees initialization, scipy/R package
callbacks, numerical gradients and perturbation calls, and raises
:class:`EvaluationBudgetExceeded` at ``fe_budget``.

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

from comparison.methods.cmaes import cma_es
from comparison.methods.de import differential_evo
from comparison.methods.ga import genetic_algorithm
from comparison.methods.gensa import gensa
from comparison.methods.pso import particle_swarm
from comparison.methods.sa import simulated_annealing
from .e7_algorithm_adapters import (
    E7_ALGORITHM_IDS,
    UnsupportedAlgorithmError,
    algorithm_metadata,
    prepare_e7_adapter,
)
from .evaluation import EvaluationBudgetExceeded
from .highdim_instances import HighDimInstance
from .paper_contract import NONE_TOKEN
from .provenance import default_environment_hash, default_git_commit, default_machine_id
from .ultrahighdim_extension import WorkerProgressSink

_BASELINE_DISPATCH = {
    "DE": differential_evo,
    "GA": genetic_algorithm,
    "PSO": particle_swarm,
    "SA": simulated_annealing,
    "GenSA": gensa,
    "CMA-ES": cma_es,
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
        self.counts_by_event: dict[str, int] = {}
        self.progress_callback = None

    def __call__(self, x: np.ndarray) -> float:
        return self.evaluate(x, event="iterate")

    def evaluate(self, x: np.ndarray, *, event: str) -> float:
        if self.fe >= self.max_evals:
            raise EvaluationBudgetExceeded(
                f"baseline FE budget {self.max_evals} reached"
            )
        value = self.instance.objective(x)
        self.fe += 1
        self.counts_by_event[event] = self.counts_by_event.get(event, 0) + 1
        if value < self.best_min:
            self.best_min = value
            self.trace.append((self.fe, self.best_min))
        if self.progress_callback is not None:
            self.progress_callback()
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
    e7_r_backend=None,
) -> dict:
    """Run one baseline under the SMCO FE budget and return the result payload."""
    # P1a: default provenance so every baseline outcome is auditable (the merge
    # provenance_complete audit requires non-empty git/env/machine). Applies even
    # when called directly as a task entry point (run_baseline_file), not only
    # via the batch dispatcher.
    machine_id = machine_id or default_machine_id()
    git_commit = git_commit or default_git_commit()
    environment_hash = environment_hash or default_environment_hash()
    if not isinstance(instance, HighDimInstance):
        raise TypeError("instance must be a HighDimInstance")
    if algorithm_name not in _BASELINE_DISPATCH and algorithm_name not in E7_ALGORITHM_IDS:
        raise ValueError(f"unknown baseline: {algorithm_name!r}")
    metadata = None
    try:
        if algorithm_name in E7_ALGORITHM_IDS:
            algorithm, metadata = prepare_e7_adapter(
                algorithm_name, r_backend=e7_r_backend,
            )
        else:
            algorithm = _BASELINE_DISPATCH[algorithm_name]
    except UnsupportedAlgorithmError as exc:
        # A missing native package/version is a scientific unsupported case,
        # not permission to swap in a similarly named Python optimizer.
        frozen = algorithm_metadata(algorithm_name)
        return {
            "algorithm_id": algorithm_name,
            "algorithm_metadata": frozen,
            "stage": stage,
            "function": instance.function_name,
            "dimension": instance.dimension,
            "n_starts": int(np.asarray(starts).shape[0]),
            "known_optimum": float(instance.known_optimum_value),
            "status": "algorithm_failure",
            "failure_reason": f"unsupported_dependency: {exc}",
            "fe_budget": int(fe_budget),
            "fe_used": 0,
            "best_value": None,
            "normalized_gap": None,
            "objective_sense": "minimize",
            "target_hit_fe": {label: None for label in _GAP_TARGETS},
            "anytime": [
                {
                    "checkpoint_fe": int(cp), "fe_used": 0,
                    "best_value": None, "normalized_gap": None,
                }
                for cp in checkpoints
            ],
            "best_so_far_trace": [],
            "termination_reason": "error",
            "fe_counts_by_event": {},
            "wall_time_sec": 0.0,
            "peak_memory_mb": None,
            "machine_id": machine_id,
            "git_commit": git_commit,
            "environment_hash": environment_hash,
            "supersedes_run_id": NONE_TOKEN,
        }

    observer = _MinObserver(instance, fe_budget)
    starts = np.asarray(starts, dtype=float)
    if starts.ndim != 2 or starts.shape[1] != instance.dimension or starts.shape[0] == 0:
        raise ValueError(
            f"starts must have shape (n, {instance.dimension}) with n > 0"
        )
    if int(fe_budget) <= 0:
        raise ValueError("fe_budget must be positive")
    if np.any(starts < instance.bounds_lower) or np.any(starts > instance.bounds_upper):
        raise ValueError("all frozen starts must lie inside the instance bounds")
    known_optimum = float(instance.known_optimum_value)

    status = "success"
    failure_reason = NONE_TOKEN
    t0 = time.perf_counter()
    initial_values: list[float] = []
    progress_sink = WorkerProgressSink()
    initial_reference = known_optimum

    def publish_progress(*, force=False):
        best = observer.best_min if observer.trace else initial_reference
        span = initial_reference - known_optimum
        target_hit = {
            label: _first_fe_below_threshold(
                observer.trace, known_optimum + target * span,
            )
            for label, target in _GAP_TARGETS.items()
        }
        progress_sink.emit(
            fe_used=observer.fe, best_value=float(best),
            normalized_gap=_gap(best, known_optimum, initial_reference),
            target_hit_fe=target_hit, force=force,
        )

    try:
        # E7's stricter contract counts initial-reference evaluations as real
        # objective calls in the same FE pool. Preserve the already-frozen E3
        # semantics for legacy comparators, whose reference values were
        # reporting-only and intentionally excluded from FE.
        for start in starts:
            if algorithm_name in E7_ALGORITHM_IDS:
                value = observer.evaluate(start, event="initialization")
            else:
                value = instance.objective(start)
            initial_values.append(value)
        initial_reference = float(np.median(initial_values))
        observer.progress_callback = publish_progress
        publish_progress(force=True)

        def bounded_objective(x):
            # Native box handling is frozen per adapter. Defensive clipping
            # also keeps numerical-gradient/SPSA probes within the contract.
            point = np.clip(
                np.asarray(x, dtype=float),
                instance.bounds_lower,
                instance.bounds_upper,
            )
            return observer.evaluate(point, event="iterate")

        algorithm(
            bounded_objective,
            instance.bounds_lower,
            instance.bounds_upper,
            start_points=starts,
            maximize=False,
            max_iter=(int(fe_budget) if algorithm_name in ("R-DEoptim", "STOGO") else
                      max(int(fe_budget), 1000)),
            seed=int(seed),
        )
    except EvaluationBudgetExceeded:
        pass  # expected hard stop at the FE budget
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        # Exceptions crossing the rpy2 callback boundary may be rewrapped as an
        # R error. If the shared observer is exactly at cap this is still the
        # expected evaluation-budget stop, not an algorithm failure.
        if observer.fe < int(fe_budget):
            status = "algorithm_failure"
            failure_reason = f"{type(exc).__name__}: {exc}"

    wall_time = time.perf_counter() - t0
    fe_used = observer.fe
    initial_reference = (
        float(np.median(initial_values)) if initial_values
        else (observer.best_min if observer.trace else known_optimum)
    )
    best_min = observer.best_min if observer.trace else initial_reference
    normalized_gap = _gap(best_min, known_optimum, initial_reference)

    span = initial_reference - known_optimum
    target_hit = {
        label: _first_fe_below_threshold(observer.trace, known_optimum + target * span)
        for label, target in _GAP_TARGETS.items()
    }
    publish_progress(force=True)

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

    result = {
        "algorithm_id": algorithm_name,
        "stage": stage,
        "function": instance.function_name,
        "dimension": instance.dimension,
        "n_starts": int(starts.shape[0]),  # A-09 #1: actual start count used
        "known_optimum": known_optimum,
        "status": status,
        "failure_reason": failure_reason,
        "fe_budget": int(fe_budget),
        "fe_used": fe_used,
        "best_value": float(best_min),
        "normalized_gap": normalized_gap,
        "objective_sense": "minimize",
        "target_hit_fe": target_hit,
        "anytime": anytime,
        "best_so_far_trace": [[int(fe), float(val)] for fe, val in observer.trace],
        "termination_reason": (
            "error" if status == "algorithm_failure" else
            ("evaluation_budget" if fe_used >= int(fe_budget) else "convergence")
        ),
        "fe_counts_by_event": (
            dict(observer.counts_by_event)
            if algorithm_name in E7_ALGORITHM_IDS else {}
        ),
        "wall_time_sec": wall_time,
        "peak_memory_mb": None,
        "machine_id": machine_id,
        "git_commit": git_commit,
        "environment_hash": environment_hash,
        "supersedes_run_id": NONE_TOKEN,
    }
    if metadata is not None:
        result["algorithm_metadata"] = metadata
    return result


__all__ = ["run_baseline_task", "BASELINE_NAMES"]

BASELINE_NAMES = tuple(_BASELINE_DISPATCH.keys()) + E7_ALGORITHM_IDS
