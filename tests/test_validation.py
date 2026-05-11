import numpy as np
import pytest

from smco.optimizer import (
    check_bounds,
    compute_partial_signs,
    generate_sobol_points,
    validate_smco_inputs,
)


def test_validate_rejects_non_callable_objective():
    with pytest.raises(TypeError, match="f must be callable"):
        validate_smco_inputs(12, [0.0], [1.0], None, {})


def test_validate_rejects_mismatched_bounds():
    with pytest.raises(ValueError, match="same length"):
        validate_smco_inputs(lambda x: 0.0, [0.0], [1.0, 2.0], None, {})


def test_validate_rejects_non_increasing_bounds():
    with pytest.raises(ValueError, match="strictly less"):
        validate_smco_inputs(lambda x: 0.0, [1.0], [1.0], None, {})


def test_validate_rejects_bad_start_point_shape():
    with pytest.raises(ValueError, match="2 columns"):
        validate_smco_inputs(lambda x: 0.0, [0.0, 0.0], [1.0, 1.0], [[0.5]], {})


def test_validate_rejects_start_points_outside_bounds():
    with pytest.raises(ValueError, match="within bounds"):
        validate_smco_inputs(lambda x: 0.0, [0.0, 0.0], [1.0, 1.0], [[1.5, 0.5]], {})


def test_validate_returns_defensive_copies():
    original_lower = np.array([0.0, 0.0])
    original_upper = np.array([1.0, 1.0])
    original_starts = np.array([[0.25, 0.75]])

    lower, upper, starts = validate_smco_inputs(
        lambda x: 0.0,
        original_lower,
        original_upper,
        original_starts,
        {},
    )
    lower[0] = -10.0
    upper[0] = 10.0
    starts[0, 0] = 0.5

    assert np.allclose(original_lower, [0.0, 0.0])
    assert np.allclose(original_upper, [1.0, 1.0])
    assert np.allclose(original_starts, [[0.25, 0.75]])


def test_check_bounds_clips_out_of_range_values():
    checked = check_bounds(
        np.array([-1.0, 0.5, 3.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
    )

    assert checked.is_out is True
    assert np.allclose(checked.x_in, [0.0, 0.5, 1.0])


def test_generate_sobol_points_is_reproducible_and_inside_bounds():
    lower = np.array([-2.0, 10.0])
    upper = np.array([2.0, 20.0])

    first = generate_sobol_points(5, lower, upper, seed=123)
    second = generate_sobol_points(5, lower, upper, seed=123)

    assert first.shape == (5, 2)
    assert np.all(first >= lower)
    assert np.all(first <= upper)
    assert np.allclose(first, second)


def test_generate_sobol_points_rejects_non_positive_n_starts():
    with pytest.raises(ValueError, match="n_starts"):
        generate_sobol_points(0, np.array([0.0]), np.array([1.0]))


@pytest.mark.parametrize(
    ("opt_control", "message"),
    [
        ({"iter_boost": -1}, "iter_boost"),
        ({"iter_boost": False}, "iter_boost"),
        ({"tol_conv": 0}, "tol_conv"),
        ({"seed": 1.5}, "seed"),
        ({"seed": True}, "seed"),
        ({"buffer_rand": "yes"}, "buffer_rand"),
    ],
)
def test_validate_rejects_invalid_task3_controls(opt_control, message):
    with pytest.raises(ValueError, match=message):
        validate_smco_inputs(lambda x: 0.0, [0.0], [1.0], None, opt_control)


def test_compute_partial_signs_center_tracks_best_perturbation():
    def objective(x):
        return -float((x[0] - 0.75) ** 2 + (x[1] + 0.25) ** 2)

    x = np.array([0.0, 0.0])
    result = compute_partial_signs(
        objective,
        x,
        objective(x),
        h_step=np.array([0.1, 0.1]),
        bounds_lower=np.array([-1.0, -1.0]),
        bounds_upper=np.array([1.0, 1.0]),
        partial_option="center",
        use_runmax=True,
    )

    assert np.allclose(result.signs, [1.0, 0.0])
    assert result.f_partial_best > objective(x)
    assert np.allclose(x, [0.0, 0.0])


def test_compute_partial_signs_forward_handles_upper_boundary():
    def objective(x):
        return -float((x[0] - 0.5) ** 2)

    x = np.array([1.0])
    result = compute_partial_signs(
        objective,
        x,
        objective(x),
        h_step=np.array([0.1]),
        bounds_lower=np.array([0.0]),
        bounds_upper=np.array([1.0]),
        partial_option="forward",
        use_runmax=False,
    )

    # Matches the upstream R upper-bound forward branch: sign = fx > f_perturb.
    assert np.allclose(result.signs, [0.0])
