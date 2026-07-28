"""Exact FE-budget regression tests (Task 1 / Gate A, Python side)."""

from __future__ import annotations

import numpy as np
import pytest

from smco import smco, smco_br_evo, smco_evo, smco_multi, smco_r_evo
from smco.evaluation import EvaluationBudgetExceeded, EvaluationContext
from smco.optimizer import (
    _clip_result_to_bounds,
    _initialize_smco_state,
    _run_smco_state_until,
)
from smco.results import SingleResult


def _quad(x):
    return -float(np.sum(x**2))


# --------------------------------------------------------------------------
# Scenario 1: initialization budget insufficient -> fewer starts, no crash.
# --------------------------------------------------------------------------
def test_smco_multi_init_budget_insufficient_reduces_start_count():
    result = smco(
        _quad,
        [-1.0],
        [1.0],
        n_starts=4,
        iter_max=80,
        seed=1,
        max_evals=3,
    )
    fe = result.summary["fe"]
    assert len(result.all_results) < 4
    assert fe["fe_used"] <= 3
    assert fe["termination_reason"] == "evaluation_budget"


# --------------------------------------------------------------------------
# Scenario 2/3: exact iteration accounting via _run_smco_state_until.
# d=2, center -> step_cost = 2*2 + 1 = 5; init = 1.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "max_evals,expected_fe",
    [
        (6, 6),   # init(1) + exactly 1 iteration(5); next pre-check fails
        (11, 11), # init(1) + 2 iterations(10); exactly fills budget
        (7, 6),   # init(1) + 1 iteration(5)=6; remaining 1 < step 5 -> stop
    ],
)
def test_run_smco_state_until_exact_step_accounting(max_evals, expected_fe):
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    ctx = EvaluationContext(_quad, max_evals=max_evals, objective_sense="maximize")
    state = _initialize_smco_state(
        _quad, np.array([0.5, 0.5]), iter_nstart=1, iter_boost=0,
        use_runmax=True, ctx=ctx,
    )
    assert ctx.evaluations == 1  # the single initialization evaluation
    _run_smco_state_until(
        state, _quad, lower, upper, 0.05, False, 10, 1e-12, "center", True,
        np.random.default_rng(0), ctx=ctx,
    )
    assert ctx.evaluations == expected_fe
    assert ctx.evaluations <= max_evals
    assert ctx.termination_reason == "evaluation_budget"


# --------------------------------------------------------------------------
# Scenario: hard guard raises if a stray evaluation is attempted past the cap.
# --------------------------------------------------------------------------
def test_evaluate_raises_hard_guard_at_cap():
    ctx = EvaluationContext(_quad, max_evals=2)
    ctx.evaluate(np.array([0.0]), event="iterate")
    ctx.evaluate(np.array([0.0]), event="iterate")
    with pytest.raises(EvaluationBudgetExceeded):
        ctx.evaluate(np.array([0.0]), event="iterate")
    assert ctx.termination_reason == "evaluation_budget"


# --------------------------------------------------------------------------
# Scenario 4: replacement initialization is counted under its own event.
# --------------------------------------------------------------------------
def test_smco_evo_counts_replacement_initialization_event():
    result = smco_evo(
        _quad, [-1.0, -1.0], [1.0, 1.0], n_starts=8, iter_max=40,
        evolution_points=(0.5,), elimination_rate=0.25, seed=123, tol_conv=1e-12,
        max_evals=200000,
    )
    counts = result.summary["fe"]["evaluation_counts_by_event"]
    # 8 initial starts + 2 replacements at the single boundary.
    assert counts["initialization"] == 8
    assert counts["replacement_initialization"] == 2
    assert result.summary["fe"]["fe_used"] <= 200000


# --------------------------------------------------------------------------
# Scenario 5: refine phase is counted under the 'refine' event.
# --------------------------------------------------------------------------
def test_smco_r_evo_counts_refine_event():
    result = smco_r_evo(
        lambda x: -float(np.sum((x - 0.25) ** 2)),
        [-1.0, -1.0], [1.0, 1.0], n_starts=8, iter_max=60,
        refine_ratio=0.5, seed=456, tol_conv=1e-12, max_evals=400000,
    )
    counts = result.summary["fe"]["evaluation_counts_by_event"]
    assert counts["refine"] > 0
    # Refine restart points are inits re-tagged 'refine'; plain init still counted.
    assert counts["initialization"] == 8


# --------------------------------------------------------------------------
# Scenario 6: BR regular/boosted branches share the global cap (each <= half).
# --------------------------------------------------------------------------
def test_smco_br_evo_branch_total_within_global_budget():
    budget = 20000
    result = smco_br_evo(
        lambda x: -float(np.sum((x - 0.3) ** 2)),
        [-1.0, -1.0], [1.0, 1.0], n_starts=6, iter_max=40, iter_boost=20,
        seed=789, max_evals=budget,
    )
    fe = result.summary["fe"]
    branch = fe["branch_fe"]
    assert branch["regular"] <= budget // 2 + 1
    assert branch["boosted"] <= budget // 2 + 1
    assert fe["fe_used"] == branch["regular"] + branch["boosted"]
    assert fe["fe_used"] <= budget


# --------------------------------------------------------------------------
# Scenario 7: clip re-evaluation is counted, and skipped when budget is out.
# --------------------------------------------------------------------------
def test_clip_result_to_bounds_counts_recheck_and_skips_when_broke():
    f = lambda x: float(x[0])  # noqa: E731

    # Affordable -> clipped and counted.
    ctx = EvaluationContext(f, max_evals=10)
    res = SingleResult(x_optimal=np.array([5.0]), f_optimal=5.0, iterations=0)
    _clip_result_to_bounds(res, f, np.array([-1.0]), np.array([1.0]), ctx=ctx)
    assert res.x_optimal[0] == 1.0
    assert res.f_optimal == pytest.approx(1.0)
    assert ctx.evaluation_counts()["clip_recheck"] == 1

    # Budget exhausted -> leave the (consistent) original point untouched.
    ctx2 = EvaluationContext(f, max_evals=1)
    ctx2.evaluate(np.array([0.0]), event="iterate")
    res2 = SingleResult(x_optimal=np.array([5.0]), f_optimal=5.0, iterations=0)
    _clip_result_to_bounds(res2, f, np.array([-1.0]), np.array([1.0]), ctx=ctx2)
    assert res2.x_optimal[0] == 5.0
    assert res2.f_optimal == 5.0
    assert ctx2.evaluation_counts()["clip_recheck"] == 0


# --------------------------------------------------------------------------
# Scenario 8: max_evals=None keeps the legacy path (no 'fe' in summary).
# --------------------------------------------------------------------------
def test_max_evals_none_is_legacy_path():
    result = smco_evo(
        _quad, [-1.0, -1.0], [1.0, 1.0], n_starts=8, iter_max=40,
        evolution_points=(0.5,), elimination_rate=0.25, seed=123, tol_conv=1e-12,
        max_evals=None,
    )
    assert "fe" not in result.summary


def test_max_evals_none_and_default_are_identical():
    kw = dict(n_starts=8, iter_max=40, seed=321, tol_conv=1e-12)
    explicit_none = smco_evo(_quad, [-1, -1], [1, 1], max_evals=None, **kw)
    default = smco_evo(_quad, [-1, -1], [1, 1], **kw)
    assert explicit_none.best_result.f_optimal == pytest.approx(default.best_result.f_optimal)
    assert np.allclose(explicit_none.best_result.x_optimal, default.best_result.x_optimal)


# --------------------------------------------------------------------------
# Scenario 9: same seed + same budget is fully reproducible.
# --------------------------------------------------------------------------
def test_budget_run_is_reproducible_with_same_seed():
    kw = dict(
        n_starts=8, iter_max=60, evolution_points=(0.5, 0.75),
        elimination_rate=0.25, seed=11, tol_conv=1e-12, max_evals=5000,
    )
    first = smco_evo(_quad, [-2, -2], [2, 2], **kw)
    second = smco_evo(_quad, [-2, -2], [2, 2], **kw)
    assert first.best_result.f_optimal == pytest.approx(second.best_result.f_optimal)
    assert np.allclose(first.best_result.x_optimal, second.best_result.x_optimal)
    assert first.summary["fe"]["fe_used"] == second.summary["fe"]["fe_used"]


# --------------------------------------------------------------------------
# Scenario 10: fe_used never exceeds fe_budget across variants.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("variant", [smco_evo, smco_r_evo, smco_br_evo])
def test_fe_used_never_exceeds_budget(variant):
    budget = 1234  # deliberately not a multiple of any step_cost
    result = variant(
        _quad, [-1.0] * 3, [1.0] * 3, n_starts=8, iter_max=200,
        seed=5, tol_conv=1e-12, max_evals=budget,
    )
    fe = result.summary["fe"]
    assert fe["fe_used"] <= budget
    assert fe["termination_reason"] == "evaluation_budget"


# --------------------------------------------------------------------------
# Sanity: best_value recorded by the context matches the run's best result.
# --------------------------------------------------------------------------
def test_context_best_value_matches_run_best_result():
    result = smco_evo(
        _quad, [-1.0, -1.0], [1.0, 1.0], n_starts=8, iter_max=60,
        evolution_points=(0.5,), elimination_rate=0.25, seed=123, tol_conv=1e-12,
        max_evals=200000, objective_sense="maximize", known_optimum=0.0,
    )
    fe = result.summary["fe"]
    assert fe["best_value"] == pytest.approx(result.best_result.f_optimal)
    # Concave quadratic solved well -> the tightest gap target is hit.
    assert fe["target_hit_evaluations"]["target_hit_fe_0.1"] is not None
