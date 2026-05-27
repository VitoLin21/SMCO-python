import numpy as np
import pytest
from smco import SMCOResult, SingleResult, smco, smco_br, smco_multi, smco_r
from smco import smco_br_evo, smco_evo, smco_r_evo
from smco.optimizer import (
    SMCOState,
    _generate_evolution_points,
    _initialize_smco_state,
    _run_evolutionary_states,
    _run_smco_state_until,
    _split_refine_iterations,
    generate_sobol_points,
)


def test_public_api_exports_expected_symbols():
    assert callable(smco)
    assert callable(smco_r)
    assert callable(smco_br)
    assert callable(smco_multi)


def test_public_api_exports_evolutionary_smco_variants():
    assert callable(smco_evo)
    assert callable(smco_r_evo)
    assert callable(smco_br_evo)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"evolution_points": [0.0]}, "evolution_points"),
        ({"evolution_points": [1.0]}, "evolution_points"),
        ({"evolution_points": [0.75, 0.5]}, "evolution_points"),
        ({"elimination_rate": 0.0}, "elimination_rate"),
        ({"elimination_rate": 1.0}, "elimination_rate"),
        ({"evolution_strategy": "rand2bin"}, "evolution_strategy"),
        ({"de_factor": 0.0}, "de_factor"),
        ({"de_crossover": 1.5}, "de_crossover"),
    ],
)
def test_smco_evo_rejects_invalid_evolution_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        smco_evo(
            lambda x: -float(x[0] ** 2),
            [-1.0],
            [1.0],
            n_starts=4,
            iter_max=8,
            seed=123,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"evolution_points": None}, "evolution_points"),
        ({"elimination_rate": None}, "elimination_rate"),
        ({"de_factor": "bad"}, "de_factor"),
        ({"de_crossover": None}, "de_crossover"),
    ],
)
def test_smco_evo_rejects_bad_typed_evolution_controls_cleanly(kwargs, message):
    with pytest.raises(ValueError, match=message):
        smco_evo(
            lambda x: -float(x[0] ** 2),
            [-1.0],
            [1.0],
            n_starts=4,
            iter_max=8,
            seed=123,
            **kwargs,
        )


def test_smco_r_evo_uses_shared_evolution_control_validation():
    with pytest.raises(ValueError, match="elimination_rate"):
        smco_r_evo(
            lambda x: -float(x[0] ** 2),
            [-1.0],
            [1.0],
            n_starts=4,
            iter_max=8,
            seed=123,
            elimination_rate=None,
        )


def test_smco_br_evo_uses_shared_evolution_control_validation():
    with pytest.raises(ValueError, match="de_crossover"):
        smco_br_evo(
            lambda x: -float(x[0] ** 2),
            [-1.0],
            [1.0],
            n_starts=4,
            iter_max=8,
            seed=123,
            de_crossover=None,
        )


def test_smco_evo_records_evolution_history_and_uses_rand1bin_by_default():
    result = smco_evo(
        lambda x: -float(np.sum((x - 0.1) ** 2)),
        [-1.0, -1.0],
        [1.0, 1.0],
        n_starts=8,
        iter_max=40,
        evolution_points=(0.5, 0.75),
        elimination_rate=0.25,
        seed=123,
        tol_conv=1e-12,
    )

    history = result.summary["evolution_history"]
    assert result.summary["evolution_strategy"] == "rand1bin"
    assert result.summary["evolution_points"] == (0.5, 0.75)
    assert len(history) == 2
    assert [event["iteration"] for event in history] == [20, 30]
    assert all(event["eliminated_count"] == 2 for event in history)
    assert all(event["generated_count"] == 2 for event in history)
    assert len(result.all_results) == 8
    assert result.opt_control["evolution_strategy"] == "rand1bin"
    assert result.opt_control["evolution_points"] == (0.5, 0.75)
    assert result.opt_control["elimination_rate"] == pytest.approx(0.25)
    assert result.opt_control["de_factor"] == pytest.approx(0.8)
    assert result.opt_control["de_crossover"] == pytest.approx(0.7)


def test_smco_evo_is_reproducible_with_same_seed():
    def objective(x):
        return -float(np.sum((x + 0.15) ** 2))

    first = smco_evo(objective, [-2.0, -2.0], [2.0, 2.0], n_starts=8, iter_max=35, seed=321)
    second = smco_evo(objective, [-2.0, -2.0], [2.0, 2.0], n_starts=8, iter_max=35, seed=321)

    assert np.allclose(first.best_result.x_optimal, second.best_result.x_optimal)
    assert first.best_result.f_optimal == pytest.approx(second.best_result.f_optimal)
    assert np.allclose(first.summary["endpoints"], second.summary["endpoints"])


def test_smco_evo_optimizes_concave_quadratic_near_origin():
    result = smco_evo(
        lambda x: -float(np.sum(x**2)),
        [-5.0, -5.0],
        [5.0, 5.0],
        n_starts=10,
        iter_max=100,
        evolution_points=(0.5, 0.75),
        elimination_rate=0.25,
        seed=123,
        tol_conv=1e-12,
    )

    assert result.best_result.f_optimal > -0.2
    assert np.linalg.norm(result.best_result.x_optimal) < 0.6
    assert len(result.summary["evolution_history"]) == 2


def test_smco_r_evo_runs_refine_search_and_records_evolution_metadata():
    result = smco_r_evo(
        lambda x: -float(np.sum((x - 0.25) ** 2)),
        [-1.0, -1.0],
        [1.0, 1.0],
        n_starts=8,
        iter_max=60,
        refine_ratio=0.5,
        seed=456,
        tol_conv=1e-12,
    )

    assert result.opt_control["refine_search"] is True
    assert result.summary["evolution_strategy"] == "rand1bin"
    assert len(result.summary["evolution_history"]) >= 1
    assert np.isfinite(result.best_result.f_optimal)


def test_smco_r_evo_runs_zero_buffer_refine_even_when_refine_ratio_is_zero(monkeypatch):
    refine_calls: list[tuple[float, int, int]] = []

    monkeypatch.setattr(
        "smco.optimizer._run_evolutionary_states",
        lambda *args, **kwargs: (
            [SingleResult(x_optimal=np.array([0.1]), f_optimal=0.1, iterations=7)],
            [{"iteration": 10, "generated_count": 0}],
        ),
    )

    def fake_single(
        f,
        bounds_lower,
        bounds_upper,
        start_point,
        *,
        bounds_buffer,
        buffer_rand,
        iter_max,
        iter_nstart,
        iter_boost,
        tol_conv,
        partial_option,
        use_runmax,
        rng,
    ):
        refine_calls.append((float(bounds_buffer), int(iter_max), int(iter_boost)))
        return SingleResult(x_optimal=np.array([0.9]), f_optimal=0.9, iterations=int(iter_nstart))

    monkeypatch.setattr("smco.optimizer._single", fake_single)

    result = smco_r_evo(
        lambda x: float(x[0]),
        [-1.0],
        [1.0],
        start_points=np.array([[0.0]]),
        iter_max=10,
        refine_ratio=0.0,
        seed=123,
    )

    assert refine_calls == [(0.0, 0, 1000)]
    assert result.best_result.f_optimal == pytest.approx(0.9)


def test_smco_r_evo_refine_keeps_full_budget_iteration_accounting(monkeypatch):
    monkeypatch.setattr(
        "smco.optimizer._run_evolutionary_states",
        lambda *args, **kwargs: (
            [SingleResult(x_optimal=np.array([0.2]), f_optimal=0.2, iterations=17)],
            [{"iteration": 8, "generated_count": 1}],
        ),
    )

    monkeypatch.setattr(
        "smco.optimizer._single",
        lambda *args, **kwargs: SingleResult(
            x_optimal=np.array([0.8]),
            f_optimal=0.8,
            iterations=6,
        ),
    )

    result = smco_r_evo(
        lambda x: float(x[0]),
        [-1.0],
        [1.0],
        start_points=np.array([[0.0]]),
        iter_max=20,
        refine_ratio=0.5,
        seed=123,
    )

    assert result.all_results[0].iterations == 22
    assert result.summary["mean_iterations"] == pytest.approx(22.0)
    assert result.summary["evolution_history"] == [{"iteration": 8, "generated_count": 1}]


def test_smco_br_evo_preserves_iter_boost_control():
    result = smco_br_evo(
        lambda x: -float(np.sum((x - 0.3) ** 2)),
        [-1.0, -1.0],
        [1.0, 1.0],
        n_starts=6,
        iter_max=40,
        iter_boost=20,
        seed=789,
    )

    assert result.opt_control["iter_boost"] == 20
    assert result.summary["evolution_strategy"] == "rand1bin"
    assert np.isfinite(result.best_result.f_optimal)


def test_smco_br_evo_records_regular_vs_boosted_competition():
    objective = lambda x: float(np.sin(9.0 * x[0]) - 0.15 * x[0] ** 2)

    regular = smco_br_evo(
        objective,
        [-2.0],
        [2.0],
        n_starts=6,
        iter_max=24,
        iter_boost=0,
        seed=321,
        tol_conv=1e-12,
    )
    boosted = smco_br_evo(
        objective,
        [-2.0],
        [2.0],
        n_starts=6,
        iter_max=24,
        iter_boost=20,
        seed=321,
        tol_conv=1e-12,
    )

    assert regular.summary["evolution_strategy"] == "rand1bin"
    assert boosted.summary["evolution_strategy"] == "rand1bin"
    assert "boost_selection" not in regular.summary
    assert boosted.opt_control["iter_boost"] == 20
    assert np.isfinite(boosted.best_result.f_optimal)

    selection = boosted.summary["boost_selection"]
    assert selection["iter_boost"] == 20
    assert selection["selected_branch"] in {"regular", "boosted"}
    assert np.isfinite(selection["regular_best"])
    assert np.isfinite(selection["boosted_best"])
    assert selection["regular_best"] == pytest.approx(regular.best_result.f_optimal)


def test_run_evolutionary_states_caps_replacement_budget_to_remaining_global_boundary(monkeypatch):
    calls: list[tuple[int, int]] = []

    def fake_run_state_until(
        state,
        f,
        bounds_lower,
        bounds_upper,
        bounds_buffer,
        buffer_rand,
        iter_target,
        tol_conv,
        partial_option,
        use_runmax,
        rng,
    ):
        calls.append((int(state.iter_boost), int(iter_target)))
        state.current_n = int(state.initial_n) + int(iter_target) + 1
        state.iterations = state.current_n - 1 - int(state.iter_boost)
        state.stopped_target_n = int(state.initial_n) + int(iter_target)

    monkeypatch.setattr("smco.optimizer._run_smco_state_until", fake_run_state_until)
    monkeypatch.setattr(
        "smco.optimizer._generate_evolution_points",
        lambda *args, **kwargs: np.array([[9.0], [8.0]], dtype=float),
    )

    results, history = _run_evolutionary_states(
        lambda x: float(x[0]),
        np.array([0.0]),
        np.array([10.0]),
        np.array([[4.0], [3.0], [2.0], [1.0]], dtype=float),
        {
            "iter_nstart": 1,
            "bounds_buffer": 0.0,
            "buffer_rand": False,
            "tol_conv": 1e-12,
            "partial_option": "center",
            "use_runmax": False,
        },
        evolution_points=(0.5,),
        elimination_rate=0.5,
        evolution_strategy="sobol",
        de_factor=0.8,
        de_crossover=0.7,
        iter_max=8,
        iter_boost=0,
        rng=np.random.default_rng(123),
    )

    assert len(history) == 1
    assert len(results) == 4
    assert calls[:4] == [(0, 4), (0, 4), (0, 4), (0, 4)]
    assert calls[4:] == [(0, 8), (0, 8), (4, 4), (4, 4)]


def test_run_evolutionary_states_preserves_survivor_internal_state_across_boundary(monkeypatch):
    boundary_calls: list[tuple[float, float, int]] = []

    def fake_run_state_until(
        state,
        f,
        bounds_lower,
        bounds_upper,
        bounds_buffer,
        buffer_rand,
        iter_target,
        tol_conv,
        partial_option,
        use_runmax,
        rng,
    ):
        boundary_calls.append((float(state.x_current[0]), float(state.s_value[0]), int(iter_target)))
        if state.birth_iteration == 0 and int(iter_target) == 4:
            if state.x_current[0] > 0.5:
                state.x_current = np.array([1.5], dtype=float)
                state.f_current = 1.5
                state.s_value = np.array([123.0], dtype=float)
            else:
                state.x_current = np.array([0.25], dtype=float)
                state.f_current = 0.25
                state.s_value = np.array([7.0], dtype=float)
            state.current_n = int(state.initial_n) + int(iter_target) + 1
            state.iterations = state.current_n - 1 - int(state.iter_boost)
            state.stopped_target_n = int(state.initial_n) + int(iter_target)
            return

        state.current_n = int(state.initial_n) + int(iter_target) + 1
        state.iterations = state.current_n - 1 - int(state.iter_boost)
        state.stopped_target_n = int(state.initial_n) + int(iter_target)

    monkeypatch.setattr("smco.optimizer._run_smco_state_until", fake_run_state_until)
    monkeypatch.setattr(
        "smco.optimizer._generate_evolution_points",
        lambda *args, **kwargs: np.array([[0.1]], dtype=float),
    )

    _run_evolutionary_states(
        lambda x: float(x[0]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([[1.0], [0.0]], dtype=float),
        {
            "iter_nstart": 1,
            "bounds_buffer": 0.5,
            "buffer_rand": False,
            "tol_conv": 1e-12,
            "partial_option": "center",
            "use_runmax": False,
        },
        evolution_points=(0.5,),
        elimination_rate=0.5,
        evolution_strategy="sobol",
        de_factor=0.8,
        de_crossover=0.7,
        iter_max=8,
        iter_boost=0,
        rng=np.random.default_rng(123),
    )

    # The survivor's second-stage call should see the same accumulated state
    # from the first stage, not a clipped-and-rebuilt `s_value`.
    assert boundary_calls[2] == (1.5, 123.0, 8)


def test_smco_evo_use_runmax_promotes_public_best_result():
    objective = lambda x: float(np.sin(10.0 * x[0]) - 0.1 * x[0] ** 2)

    promoted = smco_evo(
        objective,
        [-2.0],
        [2.0],
        n_starts=4,
        iter_max=6,
        evolution_points=(0.5,),
        elimination_rate=0.34,
        seed=1,
        use_runmax=True,
        tol_conv=1e-12,
    )
    plain = smco_evo(
        objective,
        [-2.0],
        [2.0],
        n_starts=4,
        iter_max=6,
        evolution_points=(0.5,),
        elimination_rate=0.34,
        seed=1,
        use_runmax=False,
        tol_conv=1e-12,
    )

    assert promoted.best_result.x_runmax is not None
    assert promoted.best_result.f_runmax is not None
    assert promoted.best_result.f_optimal == pytest.approx(promoted.best_result.f_runmax)
    assert np.allclose(promoted.best_result.x_optimal, promoted.best_result.x_runmax)
    assert promoted.best_result.f_optimal > plain.best_result.f_optimal


def test_smco_evo_reports_public_iterations_on_common_global_boundary_basis():
    result = smco_evo(
        lambda x: -float(np.sum((x - 0.1) ** 2)),
        [-1.0, -1.0],
        [1.0, 1.0],
        n_starts=4,
        iter_nstart=1,
        iter_max=8,
        evolution_points=(0.5,),
        elimination_rate=0.5,
        seed=123,
        tol_conv=1e-12,
        use_runmax=False,
    )

    assert result.summary["evolution_history"][0]["generated_count"] == 2
    iterations = [single.iterations for single in result.all_results]
    expected_iterations = result.opt_control["iter_nstart"] + result.opt_control["iter_max"]

    assert iterations == [expected_iterations] * len(iterations)
    assert result.summary["mean_iterations"] == pytest.approx(expected_iterations)


@pytest.mark.parametrize("strategy", ["rand1bin", "current-to-best1bin", "best1bin", "sobol"])
def test_generate_evolution_points_supports_configured_strategies(strategy):
    parents = np.array(
        [
            [-0.8, -0.4],
            [-0.2, 0.1],
            [0.3, 0.5],
            [0.7, -0.6],
        ],
        dtype=float,
    )
    scores = np.array([0.1, 0.5, 0.9, 0.2], dtype=float)
    generated = _generate_evolution_points(
        parents,
        scores,
        n_new=3,
        strategy=strategy,
        bounds_lower=np.array([-1.0, -1.0]),
        bounds_upper=np.array([1.0, 1.0]),
        de_factor=0.8,
        de_crossover=0.7,
        rng=np.random.default_rng(123),
    )

    assert generated.shape == (3, 2)
    assert np.all(generated >= -1.0)
    assert np.all(generated <= 1.0)


def test_generate_evolution_points_falls_back_to_sobol_when_parent_pool_is_too_small():
    bounds_lower = np.array([-1.0, -1.0])
    bounds_upper = np.array([1.0, 1.0])
    rng = np.random.default_rng(123)
    expected_seed = int(np.random.default_rng(123).integers(0, 2**31))

    generated = _generate_evolution_points(
        np.array([[0.0, 0.0], [0.5, 0.5]], dtype=float),
        np.array([1.0, 0.5], dtype=float),
        n_new=2,
        strategy="rand1bin",
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        de_factor=0.8,
        de_crossover=0.7,
        rng=rng,
    )

    assert generated.shape == (2, 2)
    assert np.all(generated >= -1.0)
    assert np.all(generated <= 1.0)
    assert np.allclose(generated, generate_sobol_points(2, bounds_lower, bounds_upper, seed=expected_seed))


@pytest.mark.parametrize("strategy", ["rand1bin", "current-to-best1bin"])
def test_generate_evolution_points_uses_sobol_when_strategy_lacks_distinct_indices(strategy):
    bounds_lower = np.array([-1.0, -1.0])
    bounds_upper = np.array([1.0, 1.0])
    parents = np.array([[-0.8, -0.4], [-0.2, 0.1], [0.3, 0.5]], dtype=float)
    scores = np.array([0.1, 0.9, 0.2], dtype=float)
    expected_seed = int(np.random.default_rng(456).integers(0, 2**31))

    generated = _generate_evolution_points(
        parents,
        scores,
        n_new=2,
        strategy=strategy,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        de_factor=0.8,
        de_crossover=0.7,
        rng=np.random.default_rng(456),
    )

    assert np.allclose(generated, generate_sobol_points(2, bounds_lower, bounds_upper, seed=expected_seed))


def test_generate_evolution_points_is_reproducible_with_same_seed():
    parents = np.array(
        [
            [-0.8, -0.4],
            [-0.2, 0.1],
            [0.3, 0.5],
            [0.7, -0.6],
        ],
        dtype=float,
    )
    scores = np.array([0.1, 0.5, 0.9, 0.2], dtype=float)

    first = _generate_evolution_points(
        parents,
        scores,
        n_new=5,
        strategy="current-to-best1bin",
        bounds_lower=np.array([-1.0, -1.0]),
        bounds_upper=np.array([1.0, 1.0]),
        de_factor=0.8,
        de_crossover=0.7,
        rng=np.random.default_rng(789),
    )
    second = _generate_evolution_points(
        parents,
        scores,
        n_new=5,
        strategy="current-to-best1bin",
        bounds_lower=np.array([-1.0, -1.0]),
        bounds_upper=np.array([1.0, 1.0]),
        de_factor=0.8,
        de_crossover=0.7,
        rng=np.random.default_rng(789),
    )

    assert np.allclose(first, second)


def test_generate_evolution_points_forced_crossover_dimension_changes_base_parent():
    parents = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    scores = np.array([0.0, 0.1, 0.2, 0.3], dtype=float)
    base_index_for_seed = int(np.random.default_rng(5).integers(0, parents.shape[0]))

    generated = _generate_evolution_points(
        parents,
        scores,
        n_new=1,
        strategy="rand1bin",
        bounds_lower=np.array([-2.0, -2.0]),
        bounds_upper=np.array([2.0, 2.0]),
        de_factor=1.0,
        de_crossover=0.0,
        rng=np.random.default_rng(5),
    )

    assert not np.allclose(generated[0], parents[base_index_for_seed])


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


def test_stateful_smco_runner_matches_single_stage_result_without_evolution():
    def objective(x):
        return -float((x[0] - 0.2) ** 2)

    lower = np.array([-1.0])
    upper = np.array([1.0])
    rng = np.random.default_rng(123)
    state = _initialize_smco_state(
        objective,
        np.array([0.8]),
        iter_nstart=1,
        iter_boost=0,
        use_runmax=True,
    )
    _run_smco_state_until(
        state,
        objective,
        lower,
        upper,
        bounds_buffer=0.05,
        buffer_rand=False,
        iter_target=40,
        tol_conv=1e-12,
        partial_option="center",
        use_runmax=True,
        rng=rng,
    )
    state_result = state.to_result()

    full = smco(
        objective,
        lower,
        upper,
        start_points=[[0.8]],
        iter_max=40,
        n_starts=1,
        iter_nstart=1,
        seed=123,
        tol_conv=1e-12,
    ).best_result

    assert np.allclose(state_result.x_runmax, full.x_optimal)
    assert state_result.f_runmax == pytest.approx(full.f_optimal)
    assert state_result.f_runmax == pytest.approx(full.f_runmax)


def test_stateful_smco_runner_staged_run_matches_one_shot_state_run():
    def objective(x):
        return -float((x[0] - 0.2) ** 2)

    lower = np.array([-1.0])
    upper = np.array([1.0])
    staged = _initialize_smco_state(
        objective,
        np.array([0.8]),
        iter_nstart=1,
        iter_boost=0,
        use_runmax=True,
    )
    one_shot = _initialize_smco_state(
        objective,
        np.array([0.8]),
        iter_nstart=1,
        iter_boost=0,
        use_runmax=True,
    )

    _run_smco_state_until(
        staged,
        objective,
        lower,
        upper,
        0.05,
        False,
        20,
        1e-12,
        "center",
        True,
        np.random.default_rng(123),
    )
    _run_smco_state_until(
        staged,
        objective,
        lower,
        upper,
        0.05,
        False,
        40,
        1e-12,
        "center",
        True,
        np.random.default_rng(123),
    )
    _run_smco_state_until(
        one_shot,
        objective,
        lower,
        upper,
        0.05,
        False,
        40,
        1e-12,
        "center",
        True,
        np.random.default_rng(123),
    )

    assert np.allclose(staged.x_current, one_shot.x_current)
    assert staged.f_current == pytest.approx(one_shot.f_current)
    assert np.allclose(staged.s_value, one_shot.s_value)
    assert staged.current_n == one_shot.current_n
    assert staged.iterations == one_shot.iterations
    assert np.allclose(staged.x_runmax, one_shot.x_runmax)
    assert staged.f_runmax == pytest.approx(one_shot.f_runmax)


def test_stateful_smco_runner_repeated_call_after_early_stop_is_noop():
    def objective(x):
        return 1.0

    lower = np.array([-1.0])
    upper = np.array([1.0])
    state = _initialize_smco_state(
        objective,
        np.array([0.8]),
        iter_nstart=1,
        iter_boost=0,
        use_runmax=True,
    )
    _run_smco_state_until(
        state,
        objective,
        lower,
        upper,
        0.05,
        False,
        40,
        1e-12,
        "center",
        True,
        np.random.default_rng(123),
    )
    current_n_after_stop = state.current_n
    s_value_after_stop = np.array(state.s_value, copy=True)
    iterations_after_stop = state.iterations

    _run_smco_state_until(
        state,
        objective,
        lower,
        upper,
        0.05,
        False,
        40,
        1e-12,
        "center",
        True,
        np.random.default_rng(123),
    )

    assert state.current_n == current_n_after_stop
    assert np.allclose(state.s_value, s_value_after_stop)
    assert state.iterations == iterations_after_stop


def test_smco_state_to_result_keeps_current_point_separate_from_runmax():
    state = SMCOState(
        x_current=np.array([0.4]),
        f_current=-0.04,
        s_value=np.array([1.2]),
        current_n=4,
        iter_boost=0,
        x_runmax=np.array([0.2]),
        f_runmax=0.0,
        iterations=3,
        initial_n=1,
    )

    result = state.to_result()

    assert np.allclose(result.x_optimal, [0.4])
    assert result.f_optimal == pytest.approx(-0.04)
    assert np.allclose(result.x_runmax, [0.2])
    assert result.f_runmax == pytest.approx(0.0)


def test_stateful_smco_runner_resumes_without_reinitializing_state():
    def objective(x):
        return -float((x[0] - 0.2) ** 2)

    lower = np.array([-1.0])
    upper = np.array([1.0])
    rng = np.random.default_rng(123)
    state = _initialize_smco_state(
        objective,
        np.array([0.8]),
        iter_nstart=1,
        iter_boost=0,
        use_runmax=True,
    )
    _run_smco_state_until(state, objective, lower, upper, 0.05, False, 20, 1e-12, "center", True, rng)
    s_value_after_20 = np.array(state.s_value, copy=True)
    _run_smco_state_until(state, objective, lower, upper, 0.05, False, 40, 1e-12, "center", True, rng)

    assert state.current_n >= 40
    assert not np.allclose(state.s_value, s_value_after_20)


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
