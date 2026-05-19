import numpy as np
import pytest
from smco import SMCOResult, SingleResult, smco, smco_br, smco_hb, smco_multi, smco_r
from smco.optimizer import _split_refine_iterations


def test_public_api_exports_expected_symbols():
    assert callable(smco)
    assert callable(smco_r)
    assert callable(smco_br)
    assert callable(smco_multi)
    assert callable(smco_hb)


def test_single_result_fields_are_numpy_friendly():
    result = SingleResult(
        x_optimal=np.array([1.0, 2.0]),
        f_optimal=3.0,
        iterations=7,
        x_runmax=np.array([1.5, 2.5]),
        f_runmax=4.0,
    )

    assert np.allclose(result.x_optimal, [1.0, 2.0])
    assert result.f_optimal == 3.0
    assert result.iterations == 7
    assert np.allclose(result.x_runmax, [1.5, 2.5])
    assert result.f_runmax == 4.0


def test_smco_result_groups_best_all_control_and_summary():
    single = SingleResult(x_optimal=np.array([0.0]), f_optimal=0.0, iterations=1)
    result = SMCOResult(
        best_result=single,
        all_results=[single],
        opt_control={"iter_max": 1},
        summary={"n_starts": 1},
    )

    assert result.best_result is single
    assert result.all_results == [single]
    assert result.opt_control["iter_max"] == 1
    assert result.summary["n_starts"] == 1


def test_smco_r_optimizes_concave_quadratic_near_origin():
    def objective(x):
        return -float(np.sum(x**2))

    result = smco_r(
        objective,
        [-5.0, -5.0],
        [5.0, 5.0],
        n_starts=6,
        iter_max=180,
        seed=123,
        tol_conv=1e-12,
    )

    assert result.best_result.f_optimal > -0.05
    assert np.linalg.norm(result.best_result.x_optimal) < 0.25
    assert result.summary["n_starts"] == 6
    assert len(result.all_results) == 6


def test_smco_basic_returns_points_inside_bounds():
    def objective(x):
        return -float((x[0] - 0.2) ** 2)

    result = smco(
        objective,
        [-1.0],
        [1.0],
        start_points=[[0.9]],
        iter_max=80,
        seed=1,
    )

    x = result.best_result.x_optimal
    assert x.shape == (1,)
    assert -1.0 <= x[0] <= 1.0


def test_smco_br_uses_boosted_search_and_returns_valid_structure():
    def objective(x):
        return -float(np.sum((x - 0.3) ** 2))

    result = smco_br(
        objective,
        [-1.0, -1.0],
        [1.0, 1.0],
        n_starts=5,
        iter_max=120,
        iter_boost=100,
        seed=456,
    )

    assert result.best_result.f_optimal > -0.1
    assert isinstance(result.opt_control, dict)
    assert result.opt_control["iter_boost"] == 100
    assert np.all(result.summary["endpoints"] >= -1.0)
    assert np.all(result.summary["endpoints"] <= 1.0)


def test_smco_multi_is_reproducible_with_same_seed():
    def objective(x):
        return -float(np.sum((x + 0.1) ** 2))

    first = smco_multi(
        objective,
        [-2.0, -2.0],
        [2.0, 2.0],
        opt_control={"iter_max": 80, "seed": 99},
    )
    second = smco_multi(
        objective,
        [-2.0, -2.0],
        [2.0, 2.0],
        opt_control={"iter_max": 80, "seed": 99},
    )

    assert np.allclose(first.best_result.x_optimal, second.best_result.x_optimal)
    assert first.best_result.f_optimal == second.best_result.f_optimal


def test_split_refine_iterations_preserves_upstream_rounding_boundaries():
    assert _split_refine_iterations(4, 0.0, True) == (4, 0)
    assert _split_refine_iterations(4, 1.0, True) == (0, 4)
    assert _split_refine_iterations(5, 0.5, True) == (2, 2)
    assert _split_refine_iterations(5, 0.5, False) == (5, 0)


def test_smco_multi_rejects_none_iter_max_with_clean_value_error():
    with pytest.raises(ValueError, match="iter_max"):
        smco_multi(
            lambda x: -float(x[0] ** 2),
            [-1.0],
            [1.0],
            opt_control={"iter_max": None},
        )


def test_smco_accepts_none_seed_for_nondeterministic_rng():
    result = smco(
        lambda x: -float((x[0] - 0.2) ** 2),
        [-1.0],
        [1.0],
        start_points=[[0.0]],
        iter_max=2,
        seed=None,
    )

    assert result.opt_control["seed"] is None
    assert result.best_result.x_optimal.shape == (1,)


@pytest.mark.parametrize("refine_ratio", [0.0, 1.0])
def test_smco_r_accepts_boundary_refine_ratios(refine_ratio):
    result = smco_r(
        lambda x: -float((x[0] - 0.2) ** 2),
        [-1.0],
        [1.0],
        start_points=[[0.0]],
        iter_max=4,
        seed=123,
        refine_ratio=refine_ratio,
    )

    x = result.best_result.x_optimal
    assert np.isfinite(result.best_result.f_optimal)
    assert x.shape == (1,)
    assert -1.0 <= x[0] <= 1.0
    assert result.opt_control["refine_ratio"] == refine_ratio
