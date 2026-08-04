"""Evolution-semantics contract tests (Task 3: Python restart semantics)."""

from __future__ import annotations

import numpy as np
import pytest

from smco import smco_br_evo, smco_evo, smco_r_evo


def _quad(x):
    return -float(np.sum((x - 0.1) ** 2))


@pytest.mark.parametrize("variant", [smco_evo, smco_r_evo, smco_br_evo])
def test_invalid_state_semantics_raises(variant):
    with pytest.raises(ValueError, match="state_semantics"):
        variant(_quad, [-1, -1], [1, 1], n_starts=6, iter_max=20, seed=1,
                state_semantics="bogus")


@pytest.mark.parametrize("variant", [smco_evo, smco_r_evo, smco_br_evo])
def test_all_evo_variants_support_restart(variant):
    # Every family must run end-to-end under restart semantics.
    result = variant(
        _quad, [-1, -1], [1, 1], n_starts=8, iter_max=40, seed=123, tol_conv=1e-12,
        state_semantics="restart",
    )
    assert np.isfinite(result.best_result.f_optimal)
    # history records the restart semantics tag.
    for event in result.summary["evolution_history"]:
        assert event["state_semantics"] == "restart"


def test_sp_and_restart_agree_without_boundaries():
    # No evolution boundary => SP and RS both just run each start to iter_max,
    # so they must be numerically identical (same seed).
    kw = dict(n_starts=6, iter_max=3, seed=321, tol_conv=1e-12,
              evolution_points=(0.999,))
    sp = smco_evo(_quad, [-1, -1], [1, 1], state_semantics="state_preserving", **kw)
    rs = smco_evo(_quad, [-1, -1], [1, 1], state_semantics="restart", **kw)
    assert sp.best_result.f_optimal == pytest.approx(rs.best_result.f_optimal)
    assert np.allclose(sp.best_result.x_optimal, rs.best_result.x_optimal)


def test_restart_counts_restart_initialization_event():
    result = smco_evo(
        _quad, [-1, -1], [1, 1], n_starts=8, iter_max=60,
        evolution_points=(0.5, 0.75), elimination_rate=0.25, seed=123, tol_conv=1e-12,
        state_semantics="restart", max_evals=200000, objective_sense="maximize",
    )
    counts = result.summary["fe"]["evaluation_counts_by_event"]
    # Two boundaries => survivor continuations produce restart_initialization FE.
    assert counts["restart_initialization"] > 0
    assert counts["replacement_initialization"] > 0
    assert result.summary["fe"]["fe_used"] <= 200000


def test_restart_preserves_global_best_in_archive():
    # The restart archive must not lose the best-ever running-best; the final
    # best must be at least the best value any start reached initially.
    result = smco_evo(
        lambda x: float(np.sin(9.0 * x[0]) - 0.15 * x[0] ** 2),
        [-2.0], [2.0], n_starts=8, iter_max=40, evolution_points=(0.5, 0.75),
        elimination_rate=0.25, seed=7, tol_conv=1e-12, state_semantics="restart",
        use_runmax=True,
    )
    assert np.isfinite(result.best_result.f_optimal)
    # Running-best is monotone per trajectory; the global best is retained.
    assert result.best_result.f_runmax is not None
    assert result.best_result.f_optimal == pytest.approx(result.best_result.f_runmax)


def test_restart_is_reproducible_with_same_seed():
    kw = dict(n_starts=8, iter_max=50, evolution_points=(0.5, 0.75),
              elimination_rate=0.25, seed=11, tol_conv=1e-12,
              state_semantics="restart")
    first = smco_evo(_quad, [-2, -2], [2, 2], **kw)
    second = smco_evo(_quad, [-2, -2], [2, 2], **kw)
    assert first.best_result.f_optimal == pytest.approx(second.best_result.f_optimal)
    assert np.allclose(first.best_result.x_optimal, second.best_result.x_optimal)


def test_restart_replacements_come_from_survivors_only():
    # generated_count must equal eliminated_count, i.e. every eliminated slot is
    # refilled from survivor material (no leakage from eliminated trajectories).
    result = smco_evo(
        _quad, [-1, -1], [1, 1], n_starts=8, iter_max=40,
        evolution_points=(0.5, 0.75), elimination_rate=0.25, seed=123, tol_conv=1e-12,
        state_semantics="restart",
    )
    for event in result.summary["evolution_history"]:
        assert event["eliminated_count"] == event["generated_count"]


def test_restart_br_respects_branch_budget():
    budget = 4000
    result = smco_br_evo(
        _quad, [-1, -1], [1, 1], n_starts=6, iter_max=40, iter_boost=20, seed=789,
        state_semantics="restart", max_evals=budget,
    )
    fe = result.summary["fe"]
    assert fe["fe_used"] <= budget
    assert fe["branch_fe"]["regular"] <= budget // 2 + 1
    assert fe["branch_fe"]["boosted"] <= budget // 2 + 1


def test_restart_runs_under_tight_budget_without_raising():
    result = smco_evo(
        _quad, [-1, -1], [1, 1], n_starts=8, iter_max=200,
        evolution_points=(0.5, 0.75), seed=1, state_semantics="restart", max_evals=80,
    )
    assert result.summary["fe"]["fe_used"] <= 80
    assert result.summary["fe"]["termination_reason"] == "evaluation_budget"
