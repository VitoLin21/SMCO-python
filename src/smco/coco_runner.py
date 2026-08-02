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
import json
from pathlib import Path
from typing import Any

import numpy as np

from .optimizer import global_stage_iter_max, smco, smco_br, smco_br_evo, smco_evo, smco_r, smco_r_evo
from .paper_contract import parse_algorithm_id
from comparison.methods.cmaes import cma_es
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
    "CMA-ES": cma_es,
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
        self.best_min = float("inf")
        self.trace: list[tuple[int, float]] = []  # (fe, best_min_so_far) for E4 P3

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
        if v < self.best_min:
            self.best_min = v
            self.trace.append((self.fe, self.best_min))
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
        "best_trace": observer_obj.trace,  # E4 P3: minimisation best-so-f trace
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

    # Split the FE budget across n_starts (A-01) so the evolution boundaries
    # reflect global, not single-trajectory, progress.
    iter_max = global_stage_iter_max(fe_budget, n_starts, dim)
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
    best_min = {"v": float("inf")}  # closure for the minimisation best-so-far
    trace: list[tuple[int, float]] = []
    fe_counter = {"n": 0}

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
        fe_counter["n"] += 1
        if v < best_min["v"]:
            best_min["v"] = v
            trace.append((fe_counter["n"], v))
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
        "best_trace": trace,  # E4 P3: minimisation best-so-f for derived gap/targets
    }


def _is_true(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def aggregate_instance_summary(rows, algorithms):
    """Aggregate per-(function, dim) rows over instances (A-06).

    Each (function, dimension, algorithm) collapses its instances into a
    target-hit rate, mean best and instance count, instead of keeping only the
    last instance (which silently dropped 4 of 5 instances). Returns
    ``(out_rows, field_order)``. The instance-level data stays in the per-run
    CSV; final ERT/ECDF aggregates are produced by the Task-12 analysis layer.
    """
    by_key: dict[tuple, dict[str, list]] = {}
    for r in rows:
        key = (int(r["function"]), int(r["dimension"]))
        by_key.setdefault(key, {}).setdefault(r["algorithm_id"], []).append(r)
    out = []
    for (func, dim), algos in sorted(by_key.items()):
        row = {"function": func, "dimension": dim}
        for algo in algorithms:
            recs = algos.get(algo, [])
            n = len(recs)
            row[f"{algo}_target_hit_rate"] = (
                sum(1 for r in recs if _is_true(r.get("final_target_hit"))) / n if n else ""
            )
            bests = [float(r["best_observed_fvalue1"]) for r in recs
                     if r.get("best_observed_fvalue1") not in ("", None)]
            row[f"{algo}_mean_best"] = sum(bests) / len(bests) if bests else ""
            row[f"{algo}_n_instances"] = n
        out.append(row)
    fields = (["function", "dimension"]
              + [f"{a}_target_hit_rate" for a in algorithms]
              + [f"{a}_mean_best" for a in algorithms]
              + [f"{a}_n_instances" for a in algorithms])
    return out, fields


def write_run_provenance(result_dir, *, kind, algorithms, winner=None, base=None,
                         suite=None, dims=None, instances=None, fe_budget_per_d=None,
                         original_winner=None, original_language=None, language_note=None,
                         external_check_kind=None, is_frozen_winner_validation=None):
    """Write ``provenance.json`` capturing the E4/E5 run conditions (A-09 #3).

    Records the git commit, Python/platform environment, frozen algorithm set
    and budget so a result directory is self-describing for reproduction and
    cross-language audit.
    """
    import json as _json
    import platform as _platform
    import subprocess as _sp
    from pathlib import Path

    try:
        commit = _sp.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(result_dir),
            stderr=_sp.DEVNULL, timeout=5).decode().strip()
    except Exception:
        commit = ""
    info = {
        "kind": kind,
        "git_commit": commit,
        "python": _platform.python_version(),
        "platform": _platform.platform(),
        "suite": suite,
        "dimensions": list(dims) if dims is not None else None,
        "instances": list(instances) if instances is not None else None,
        "fe_budget_per_d": fe_budget_per_d,
        "algorithms": list(algorithms),
    }
    if winner is not None:
        info["winner"] = winner
    if base is not None:
        info["matched_base"] = base
    if original_winner is not None:
        info["original_winner"] = original_winner
    if original_language is not None:
        info["original_language"] = original_language
    if language_note is not None:
        info["language_note"] = language_note
    if external_check_kind is not None:
        info["external_check_kind"] = external_check_kind
    if is_frozen_winner_validation is not None:
        info["is_frozen_winner_validation"] = is_frozen_winner_validation
    out = Path(result_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").write_text(_json.dumps(info, indent=2))


# --- E4 P3: task-level COCO outcome (one JSON per run_id) ---

def coco_problem_id(suite, problem) -> str:
    """Stable problem id, e.g. ``bbob-largescale_f001_i1_d160``."""
    return (f"{suite}_f{int(problem.id_function):03d}"
            f"_i{int(problem.id_instance)}_d{int(problem.dimension)}")


def coco_problem_f_opt(problem):
    """Best-effort extraction of a cocoex problem's known optimum value.

    cocoex does not expose the per-instance optimum scalar uniformly across
    versions; try the common attributes. Returns None if unknown (the caller
    must then refuse to emit a faked gap — review P3 honesty rule).
    """
    for attr in ("optimality_table", "optimal_fvalue", "f_opt"):
        value = getattr(problem, attr, None)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def coco_versions():
    """Return (cocoex_version, cocopp_version_or_None) for benchmark provenance."""
    try:
        import cocoex  # noqa: F401
        cocoex_v = getattr(cocoex, "__version__", "unknown")
    except Exception:
        cocoex_v = "unknown"
    try:
        import cocopp  # noqa: F401
        cocopp_v = getattr(cocopp, "__version__", None)
    except Exception:
        cocopp_v = None
    return cocoex_v, cocopp_v


def match_manifest_to_coco_problems(tasks, suite_obj, *, instance_offset: int = 1):
    """Match each manifest task to its cocoex problem.

    Manifest ``function`` ``"fN"`` -> cocoex ``id_function`` N; manifest
    ``instance`` i -> cocoex ``id_instance`` i + ``instance_offset`` (the E4
    manifest stores 0-based instance indices while cocoex uses 1-based, so the
    default offset is 1 — P4 verifies against ``problem.id_instance``).
    Returns ``{run_id: problem}``. Raises if ANY requested task has no matching
    problem (no silent skip / no shard can be incomplete).
    """
    index: dict[tuple, object] = {}
    for problem in suite_obj:
        index[(int(problem.id_function), int(problem.dimension),
               int(problem.id_instance))] = problem
    matched: dict[str, object] = {}
    for task in tasks:
        n = int(str(task["function"]).lstrip("fF"))
        coco_instance = int(task["instance"]) + int(instance_offset)
        key = (n, int(task["dimension"]), coco_instance)
        if key not in index:
            raise ValueError(
                f"task {task['run_id']}: no cocoex problem for "
                f"(function={n}, dimension={task['dimension']}, instance={coco_instance})")
        matched[task["run_id"]] = index[key]
    return matched


def dispatch_e4_tasks(
    tasks,
    suite_obj,
    *,
    result_dir,
    machine_id: str = "",
    git_commit: str = "",
    environment_hash: str = "",
    suite: str = "bbob-largescale",
    instance_offset: int = 1,
    require_f_opt: bool = True,
    winner_language: str = "python",
):
    """Run a shard of E4 manifest ``tasks`` on their cocoex problems, writing one
    COCO outcome JSON per run_id under ``result_dir`` (review P3 sharding).

    Sharding is by the frozen manifest's run_id set only. ``f_opt`` comes from
    :func:`coco_problem_f_opt`; if it is unavailable and ``require_f_opt`` is
    set, the task FAILS rather than emit a faked gap (review P3 honesty rule).
    ``winner_language`` sets the R-01 marker: a Python winner is the frozen
    winner's own validation; any other language is only a python_port_external
    check (its main claim must not rest on E4/E5). Returns a per-run_id status
    dict. Requires cocoex at runtime (P4).
    """
    is_frozen = winner_language == "python"
    external_check_kind = "frozen_winner" if is_frozen else "python_port_external"
    problems = match_manifest_to_coco_problems(
        tasks, suite_obj, instance_offset=instance_offset)
    statuses: dict[str, str] = {}
    for task in tasks:
        run_id = task["run_id"]
        problem = problems[run_id]
        try:
            f_opt = coco_problem_f_opt(problem)
            if require_f_opt and f_opt is None:
                raise ValueError("cocoex problem exposes no known f_opt; refusing to fake")
            run_e4_coco_task(
                task, problem, f_opt=(0.0 if f_opt is None else f_opt),
                result_dir=result_dir, machine_id=machine_id, git_commit=git_commit,
                environment_hash=environment_hash, suite=suite,
                n_starts=task.get("n_starts"),
                ran_language="python", is_frozen_winner_validation=is_frozen,
                external_check_kind=external_check_kind)
            statuses[run_id] = "success"
        except Exception as exc:  # noqa: BLE001 — record, don't abort the shard
            statuses[run_id] = f"failed: {type(exc).__name__}: {exc}"
    return statuses


def run_e4_coco_task(
    task: dict,
    problem,
    *,
    f_opt: float,
    result_dir,
    machine_id: str = "",
    git_commit: str = "",
    environment_hash: str = "",
    suite: str = "bbob-largescale",
    n_starts: int | None = None,
    ran_language: str = "python",
    is_frozen_winner_validation: bool = True,
    external_check_kind: str = "frozen_winner",
):
    """Run one E4 manifest task on a cocoex problem and atomically write its
    task-level COCO outcome JSON (``<result_dir>/<run_id>.json``).

    The COCO-native fields (best_observed_fvalue1, evaluations,
    final_target_hit, problem id, f_opt) are preserved verbatim and the
    synthetic-style normalized_gap / target_hit_fe / anytime are DERIVED from
    the recorded best-so-far trace (same relative convention as the synthetic
    contract). ``ran_language`` / ``is_frozen_winner_validation`` /
    ``external_check_kind`` carry the R-01 frozen-winner-validation marker.
    Requires cocoex at runtime; validated on a cocoex node at P4.
    """
    from .coco_outcome import build_coco_outcome

    fe_budget = int(task["fe_budget"])
    starts = int(n_starts if n_starts is not None else task.get("n_starts", 8))
    seed = int(task.get("seed")) if task.get("seed") is not None else None
    algorithm_id = task.get("algorithm_id") or task.get("algorithm")
    is_smco = "configuration_hash" in task
    if is_smco:
        result = run_on_problem(problem, algorithm_id=algorithm_id,
                                fe_budget=fe_budget, n_starts=starts, seed=seed)
    else:
        result = run_baseline_on_problem(problem, algorithm_name=algorithm_id,
                                         fe_budget=fe_budget, n_starts=starts, seed=seed)
    trace = result.get("best_trace") or []
    # initial reference = best-so-far at the first evaluation (the starting point)
    initial_ref = trace[0][1] if trace else float(result["best_observed_fvalue1"])
    cocoex_v, cocopp_v = coco_versions()
    payload = build_coco_outcome(
        task,
        best_observed_fvalue1=result["best_observed_fvalue1"],
        evaluations=result["evaluations"],
        final_target_hit=result["final_target_hit"],
        best_trace=trace,
        f_opt=f_opt,
        initial_ref=initial_ref,
        fe_budget=fe_budget,
        suite=suite,
        problem_id=coco_problem_id(suite, problem),
        cocoex_version=cocoex_v,
        cocopp_version=cocopp_v,
        machine_id=machine_id,
        git_commit=git_commit,
        environment_hash=environment_hash,
        ran_language=ran_language,
        is_frozen_winner_validation=is_frozen_winner_validation,
        external_check_kind=external_check_kind,
    )
    out_dir = Path(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{task['run_id']}.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out_dir / f"{task['run_id']}.json")
    return payload


__all__ = ["problem_seed", "run_on_problem", "run_baseline_on_problem",
           "aggregate_instance_summary", "write_run_provenance",
           "coco_problem_id", "coco_problem_f_opt", "coco_versions",
           "match_manifest_to_coco_problems", "dispatch_e4_tasks",
           "run_e4_coco_task"]
