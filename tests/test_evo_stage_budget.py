"""A-01 budget/scheduling regression tests (review 2026-07-30, P0).

The pre-fix ``iter_max = fe_budget // (2*dim+1)`` sized the iteration budget to a
*single trajectory*: with ``n_starts`` trajectories sharing one global FE pool,
the first evolution boundary alone consumed ~``n_starts`` x budget, so the hard
FE cap starved most initial states before they advanced (minimal repro at
d=20, n=8, B=20000 left ~6 of 8 initial states at 0 iterations).

The fix splits the global FE budget across starts
(``iter_max = fe_budget // (n_starts * (2*dim+1))``) so that every initial state
advances before the first boundary and the two evolution boundaries land near
50% / 75% of the global FE budget. These tests pin that contract.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from smco import smco_evo
from smco.optimizer import global_stage_iter_max


def _r_has(pkgs):
    expr = "cat(" + " && ".join(
        f"requireNamespace('{p}', quietly=TRUE)" for p in pkgs) + ")"
    res = subprocess.run(["Rscript", "-e", expr], capture_output=True, text=True)
    return "TRUE" in res.stdout


def _quad(x):
    return -float(np.sum((x - 0.1) ** 2))


_D, _N, _B = 20, 8, 20000


def test_global_stage_iter_max_formula():
    # A-01: iter_max = fe_budget // (n_starts * (2*dim + 1)), floored to >= 1.
    assert global_stage_iter_max(_B, _N, _D) == _B // (_N * (2 * _D + 1))  # 60
    assert global_stage_iter_max(_B, 1, _D) == _B // (2 * _D + 1)
    # tiny budget floors to 1 rather than 0
    assert global_stage_iter_max(10, _N, _D) == 1
    # n_starts < 1 must not divide by zero
    assert global_stage_iter_max(_B, 0, _D) == _B // (2 * _D + 1)


def test_state_preserving_splits_budget_across_starts():
    iter_max = global_stage_iter_max(_B, _N, _D)  # 60
    result = smco_evo(
        _quad, [-5.0] * _D, [5.0] * _D,
        n_starts=_N, iter_max=iter_max,
        evolution_points=(0.5, 0.75), elimination_rate=0.25,
        seed=1, tol_conv=1e-12, state_semantics="state_preserving",
        max_evals=_B, objective_sense="maximize",
    )
    hist = result.summary["evolution_history"]
    assert len(hist) == 2
    # 1. all 8 initial states advanced (>0) and ~equal before the first elimination
    its0 = hist[0]["state_iterations"]
    assert len(its0) == _N
    assert all(i > 0 for i in its0), f"starved initial states: {its0}"
    assert max(its0) - min(its0) <= 1, f"uneven advance: {its0}"
    # 2. boundaries land near 50% / 75% of the global FE budget
    assert 0.30 * _B <= hist[0]["cumulative_fe"] <= 0.70 * _B, hist[0]["cumulative_fe"]
    assert 0.60 * _B <= hist[1]["cumulative_fe"] <= 0.90 * _B, hist[1]["cumulative_fe"]
    # 3. hard cap holds
    assert result.summary["fe"]["fe_used"] <= _B


def test_restart_splits_budget_across_starts():
    iter_max = global_stage_iter_max(_B, _N, _D)
    result = smco_evo(
        _quad, [-5.0] * _D, [5.0] * _D,
        n_starts=_N, iter_max=iter_max,
        evolution_points=(0.5, 0.75), elimination_rate=0.25,
        seed=1, tol_conv=1e-12, state_semantics="restart",
        max_evals=_B, objective_sense="maximize",
    )
    hist = result.summary["evolution_history"]
    its0 = hist[0]["state_iterations"]
    assert len(its0) == _N
    assert all(i > 0 for i in its0), f"starved initial states: {its0}"
    assert max(its0) - min(its0) <= 1, f"uneven advance: {its0}"
    assert 0.30 * _B <= hist[0]["cumulative_fe"] <= 0.75 * _B, hist[0]["cumulative_fe"]
    assert result.summary["fe"]["fe_used"] <= _B


def test_legacy_iter_max_starves_initial_states():
    """Negative: the pre-A-01 formula (not split by n_starts) leaves most
    initial states at 0 iterations once the hard FE cap fires. Documents the
    bug the fix removes and guards against regressions."""
    legacy_iter_max = _B // (2 * _D + 1)  # 487, the old single-trajectory formula
    result = smco_evo(
        _quad, [-5.0] * _D, [5.0] * _D,
        n_starts=_N, iter_max=legacy_iter_max,
        evolution_points=(0.5, 0.75), elimination_rate=0.25,
        seed=1, tol_conv=1e-12, state_semantics="state_preserving",
        max_evals=_B, objective_sense="maximize",
    )
    hist = result.summary["evolution_history"]
    its0 = hist[0]["state_iterations"]
    starved = sum(1 for i in its0 if i == 0)
    assert starved >= _N // 2, f"legacy formula should starve >= half: {its0}"


def test_highdim_worker_uses_stage_iter_max():
    """The Python entry point must size iter_max off n_starts (not a single
    trajectory), so a multi-start EVO task completes within budget."""
    from smco.experiment_manifests import build_algorithm_config, build_task
    from smco.highdim_instances import generate_instance
    from smco.highdim_worker import run_task

    inst = generate_instance("Rastrigin", _D, 0, seed=1)
    rng = np.random.default_rng(5)
    span = inst.bounds_upper - inst.bounds_lower
    starts = inst.bounds_lower + rng.uniform(size=(_N, _D)) * span
    cfg = build_algorithm_config(
        "python", "smco", True, "state_preserving",
        evolution_strategy="rand1bin", evolution_points=(0.5, 0.75),
        elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=_N,
    )
    task = build_task(
        "e0_contract", "contract", "Rastrigin", _D, 0, 0,
        config=cfg, fe_budget=_B, checkpoints=(_B // 2, _B), seed=42,
    )
    res = run_task(task, inst, starts)
    assert res["status"] == "success"
    assert res["fe_used"] <= _B
    # with the split budget every start advances, so fe_used should be a large
    # fraction of B (not a tiny stub from budget exhaustion at initialization)
    assert res["fe_used"] >= 0.5 * _B, res["fe_used"]


def test_coco_runner_uses_stage_iter_max():
    """The COCO entry point shares the same n_starts-split iter_max."""
    pytest.importorskip("cocoex")
    import cocoex

    from smco.coco_runner import run_on_problem

    suite = cocoex.Suite("bbob", "instances:1", f"dimensions:{_D}")
    problem = next(iter(suite))
    res = run_on_problem(problem, algorithm_id="PY-SP-SMCO-EVO", fe_budget=_B, n_starts=_N)
    assert res["evaluations"] <= _B
    assert res["evaluations"] >= 0.5 * _B, res["evaluations"]


def test_py_r_stage_fe_trajectory_aligned(tmp_path):
    """A-01 cross-language: Python and R must produce the same stage/FE
    trajectory (per-boundary cumulative FE) for the d=20/n=8/B=20000 case.
    The cumulative FE at each boundary is set by the scheduling structure
    (n_starts trajectories x boundary iters x step cost), not the RNG stream,
    so the two ports must agree."""
    if shutil.which("Rscript") is None or not _r_has(["jsonlite"]):
        pytest.skip("Rscript/jsonlite not available")

    repo = Path(__file__).resolve().parent.parent
    vendor = repo / "vendor" / "SMCO_R" / "main"

    rng = np.random.default_rng(5)
    lo, hi = np.full(_D, -5.0), np.full(_D, 5.0)
    starts = lo + rng.uniform(size=(_N, _D)) * (hi - lo)
    starts_csv = tmp_path / "starts.csv"
    np.savetxt(starts_csv, starts, delimiter=",")

    iter_max = global_stage_iter_max(_B, _N, _D)  # 60

    def quad(x):
        return -float(np.sum((x - 0.1) ** 2))

    py = smco_evo(
        quad, lo, hi, starts, n_starts=_N, iter_max=iter_max,
        evolution_points=(0.5, 0.75), elimination_rate=0.25,
        evolution_strategy="rand1bin", de_factor=0.8, de_crossover=0.7,
        seed=123, tol_conv=1e-12, state_semantics="state_preserving",
        max_evals=_B, objective_sense="maximize",
    )
    py_fe = [int(h["cumulative_fe"]) for h in py.summary["evolution_history"]]

    r_script = f'''
    vendor <- "{vendor}"
    source(file.path(vendor, "evaluation_budget.R"))
    source(file.path(vendor, "SMCO.R"))
    source(file.path(vendor, "SMCO_evo.R"))
    starts <- as.matrix(read.csv("{starts_csv}", header = FALSE))
    lo <- rep(-5, {_D}); hi <- rep(5, {_D})
    f <- function(x) -sum((x - 0.1)^2)
    r <- SMCO_EVO(f, lo, hi, start_points = starts,
        evolution_points = c(0.5, 0.75), elimination_rate = 0.25,
        evolution_strategy = "rand1bin", de_factor = 0.8, de_crossover = 0.7,
        state_semantics = "state_preserving",
        iter_max = {iter_max}L, max_evals = {_B}L, objective_sense = "maximize",
        known_optimum = 0, seed = 123L, bounds_buffer = 0.05)
    fe <- sapply(r$evolution_history, function(h) h$cumulative_fe)
    cat(jsonlite::toJSON(as.integer(fe)))
    '''
    res = subprocess.run(["Rscript", "-e", r_script], capture_output=True,
                         text=True, timeout=240)
    assert res.returncode == 0, f"Rscript failed:\n{res.stderr}"
    r_fe = json.loads(res.stdout)

    assert len(py_fe) == len(r_fe) == 2, (py_fe, r_fe)
    for a, b in zip(py_fe, r_fe):
        assert abs(a - b) <= 0.10 * _B, f"stage FE diverged: py={a} r={b}"
