"""COCO bbob runner for the E5 low-dim non-degradation check.

Wraps a cocoex Problem as an SMCO objective (``g = -problem(x)``; cocoex is
minimisation, SMCO maximises) and reuses the existing optimizer API — the SMCO
core is not modified. cocoex records every evaluation via its observer; the
runner returns the cocoex-accumulated metrics (best_observed_fvalue1,
final_target_hit, evaluations). See
``docs/superpowers/specs/2026-07-29-e5-lowdim-check-design.md``.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .optimizer import smco, smco_br, smco_br_evo, smco_evo, smco_r, smco_r_evo
from .paper_contract import parse_algorithm_id
from comparison.methods.de import differential_evo
from comparison.methods.ga import genetic_algorithm
from comparison.methods.gensa import gensa
from comparison.methods.pso import particle_swarm
from comparison.methods.sa import simulated_annealing
from .evaluation import EvaluationBudgetExceeded

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

_DEFAULT_EVO_POINTS = (0.5, 0.75)
_DEFAULT_ELIMINATION_RATE = 0.25
_DEFAULT_DE_FACTOR = 0.8
_DEFAULT_DE_CROSSOVER = 0.7
_DEFAULT_STRATEGY = "rand1bin"
_DEFAULT_REFINE_RATIO = 0.5


def problem_seed(problem, n_starts: int = 8) -> int:
    """Stable 32-bit seed derived from the cocoex problem id (order-independent)."""
    key = f"{problem.id}:n{n_starts}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def _select_algorithm(algorithm_id: str):
    parsed = parse_algorithm_id(algorithm_id)
    if parsed["language"] != "python":
        raise ValueError(
            f"coco_runner is Python-only; algorithm_id {algorithm_id!r} is "
            f"{parsed['language']!r}. Convert R winners to their Py equivalent."
        )
    table = _EVO_DISPATCH if parsed["evolutionary"] else _BASE_DISPATCH
    key = ("python", parsed["family"])
    if key not in table:
        raise ValueError(f"no Python dispatch for family={parsed['family']!r}")
    return table[key], parsed


_BASELINE_DISPATCH = {
    "DE": differential_evo,
    "GA": genetic_algorithm,
    "PSO": particle_swarm,
    "SA": simulated_annealing,
    "GenSA": gensa,
}


class _CocoMinObserver:
    """Minimisation objective over a cocoex problem with a FE hard stop.

    Clips probe points to the cocoex bounds and penalises non-finite values
    (mirroring the SMCO path in :func:`run_on_problem`). Raises
    :class:`EvaluationBudgetExceeded` at ``max_evals`` so the baseline loop stops.
    """

    def __init__(self, problem, max_evals: int) -> None:
        self.problem = problem
        self.max_evals = int(max_evals)
        self.fe = 0

    def __call__(self, x):
        if self.fe >= self.max_evals:
            raise EvaluationBudgetExceeded(
                f"cocoex FE budget {self.max_evals} reached"
            )
        self.fe += 1
        x = np.clip(np.asarray(x, dtype=float), self.problem.lower_bounds, self.problem.upper_bounds)
        if not np.all(np.isfinite(x)):
            return 1e10
        v = float(self.problem(x))
        if not np.isfinite(v):
            return 1e10
        return v


def run_baseline_on_problem(
    problem,
    *,
    algorithm_name: str,
    fe_budget: int,
    n_starts: int = 8,
    seed: int | None = None,
    observer: Any = None,
) -> dict:
    """Run one comparison baseline on a cocoex problem; return cocoex metrics.

    Minimisation (``maximize=False``); FE is hard-stopped by ``_CocoMinObserver``.
    """
    if algorithm_name not in _BASELINE_DISPATCH:
        raise ValueError(f"unknown baseline: {algorithm_name!r}")
    if observer is not None:
        problem.observe_with(observer)
    dim = int(problem.dimension)
    algorithm = _BASELINE_DISPATCH[algorithm_name]
    if seed is None:
        seed = problem_seed(problem, n_starts)
    rng = np.random.default_rng(seed)
    span = problem.upper_bounds - problem.lower_bounds
    starts = problem.lower_bounds + rng.uniform(size=(n_starts, dim)) * span

    observer_obj = _CocoMinObserver(problem, fe_budget)
    try:
        algorithm(
            observer_obj, problem.lower_bounds, problem.upper_bounds,
            start_points=starts, maximize=False, max_iter=int(fe_budget), seed=int(seed),
        )
    except EvaluationBudgetExceeded:
        pass  # expected hard stop at the FE budget

    return {
        "algorithm_id": algorithm_name,
        "function": int(problem.id_function),
        "dimension": dim,
        "instance": int(problem.id_instance),
        "best_observed_fvalue1": float(problem.best_observed_fvalue1),
        "final_target_hit": bool(problem.final_target_hit),
        "evaluations": int(problem.evaluations),
    }


def run_on_problem(
    problem,
    *,
    algorithm_id: str,
    fe_budget: int,
    n_starts: int = 8,
    seed: int | None = None,
    observer: Any = None,
) -> dict:
    """Run one SMCO variant on a cocoex problem; return cocoex-accumulated metrics.

    ``problem(x)`` is minimisation; SMCO maximises ``g = -problem(x)``. Each
    evaluation is recorded by cocoex when an observer is attached. The returned
    ``best_observed_fvalue1`` is the minimisation best found during this run.
    """
    if observer is not None:
        problem.observe_with(observer)
    dim = int(problem.dimension)
    algorithm, parsed = _select_algorithm(algorithm_id)
    if seed is None:
        seed = problem_seed(problem, n_starts)
    rng = np.random.default_rng(seed)
    span = problem.upper_bounds - problem.lower_bounds
    starts = problem.lower_bounds + rng.uniform(size=(n_starts, dim)) * span

    iter_max = max(1, int(fe_budget) // (2 * dim + 1))
    control: dict = {
        "max_evals": int(fe_budget),
        "objective_sense": "maximize",
        "known_optimum": 0.0,  # SMCO convergence target; cocoex final_target_hit is authoritative
        "iter_max": iter_max,
        "seed": int(seed),
    }
    if parsed["family"] in ("smco_refine", "smco_boost_refine"):
        control["refine_ratio"] = _DEFAULT_REFINE_RATIO
    if parsed["evolutionary"]:
        control["evolution_points"] = _DEFAULT_EVO_POINTS
        control["elimination_rate"] = _DEFAULT_ELIMINATION_RATE
        control["evolution_strategy"] = _DEFAULT_STRATEGY
        control["de_factor"] = _DEFAULT_DE_FACTOR
        control["de_crossover"] = _DEFAULT_DE_CROSSOVER
        control["state_semantics"] = parsed["state_semantics"]

    lower = problem.lower_bounds
    upper = problem.upper_bounds

    def objective(x):
        # Clip probe points to the cocoex bounds (cocoex extrapolates outside)
        # and penalise non-finite probe values so a diverging trajectory cannot
        # register a misleading "best" via cocoex's NaN handling.
        x = np.clip(np.asarray(x, dtype=float), lower, upper)
        if not np.all(np.isfinite(x)):
            return -1e10
        v = float(problem(x))
        if not np.isfinite(v):
            return -1e10
        return -v

    algorithm(objective, lower, upper, starts, **control)

    return {
        "algorithm_id": algorithm_id,
        "function": int(problem.id_function),
        "dimension": dim,
        "instance": int(problem.id_instance),
        "best_observed_fvalue1": float(problem.best_observed_fvalue1),
        "final_target_hit": bool(problem.final_target_hit),
        "evaluations": int(problem.evaluations),
    }


__all__ = ["problem_seed", "run_on_problem", "run_baseline_on_problem"]
