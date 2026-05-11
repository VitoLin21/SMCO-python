import math

import numpy as np
import pytest

from smco import BenchmarkConfig, assign_config
from smco.test_functions import (
    ackley,
    griewank,
    negative_squared_norm,
    rastrigin,
    rosenbrock,
    squared_norm,
)


@pytest.mark.parametrize(
    ("func", "zero_value", "one_value"),
    [
        (squared_norm, 0.0, 3.0),
        (negative_squared_norm, 0.0, -3.0),
        (rastrigin, 0.0, 3.0),
        (ackley, 0.0, 20.0 * (1.0 - math.exp(-0.2))),
        (
            griewank,
            0.0,
            1.0
            + 3.0 / 4000.0
            - math.cos(1.0) * math.cos(1.0 / math.sqrt(2.0)) * math.cos(1.0 / math.sqrt(3.0)),
        ),
        (rosenbrock, 2.0, 0.0),
    ],
)
def test_benchmark_functions_have_known_values_at_zeros_and_ones(func, zero_value, one_value):
    zeros = np.zeros(3)
    ones = np.ones(3)

    assert func(zeros) == pytest.approx(zero_value, abs=1e-12)
    assert func(ones) == pytest.approx(one_value, abs=1e-12)


def test_assign_config_builds_rastrigin_minimization_config():
    config = assign_config("Rastrigin", 2)

    assert isinstance(config, BenchmarkConfig)
    assert config.name == "Rastrigin2d"
    assert config.dim == 2
    assert np.allclose(config.bounds_lower, [-5.12, -5.12])
    assert np.allclose(config.bounds_upper, [5.12, 5.12])
    assert config.f(np.zeros(2)) == -0.0
    assert config.known_best_value == 0.0
    assert config.sense == "min"


def test_assign_config_supports_norm_alias_for_squared_norm():
    config = assign_config("norm", 2)

    assert config.name == "SquaredNorm2d"
    assert config.dim == 2
    assert np.allclose(config.bounds_lower, [-1.0, -1.0])
    assert np.allclose(config.bounds_upper, [1.0, 1.0])
    assert config.f(np.ones(2)) == pytest.approx(2.0)
    assert config.known_best_value == 2.0
    assert np.allclose(config.known_best_x, [1.0, 1.0])
    assert config.sense == "max"


def test_assign_config_supports_upstream_sqnorm_aliases():
    sqnorm = assign_config("SqNorm", 2)
    minus_sqnorm = assign_config("-SqNorm", 2)

    assert sqnorm.name == "SquaredNorm2d"
    assert minus_sqnorm.name == "NegativeSquaredNorm2d"
    assert sqnorm.f(np.ones(2)) == pytest.approx(2.0)
    assert minus_sqnorm.f(np.zeros(2)) == pytest.approx(0.0)


@pytest.mark.parametrize("name", ["minus norm", "minusnorm"])
def test_assign_config_supports_minus_norm_aliases_for_negative_squared_norm(name):
    config = assign_config(name, 2)

    assert config.name == "NegativeSquaredNorm2d"
    assert config.dim == 2
    assert np.allclose(config.bounds_lower, [-1.0, -1.0])
    assert np.allclose(config.bounds_upper, [1.0, 1.0])
    assert config.f(np.zeros(2)) == pytest.approx(0.0)
    assert config.known_best_value == 0.0
    assert np.allclose(config.known_best_x, [0.0, 0.0])
    assert config.sense == "max"


@pytest.mark.parametrize(
    "name",
    [
        "Ackley",
        "Griewank",
        "Rosenbrock",
        "DixonPrice",
        "Zakharov",
        "Qing",
        "Michalewicz",
    ],
)
def test_assign_config_supports_common_minimization_benchmarks_case_insensitively(name):
    config = assign_config(name.swapcase(), 3)

    assert config.dim == 3
    assert config.bounds_lower.shape == (3,)
    assert config.bounds_upper.shape == (3,)
    assert np.isfinite(config.f(np.zeros(3)))
    assert config.sense == "min"


def test_assign_config_rejects_unknown_benchmark_name():
    with pytest.raises(ValueError, match="Unknown benchmark"):
        assign_config("not-a-benchmark", 2)


@pytest.mark.parametrize("name", ["Rastrigin", "DropWave"])
def test_config_objective_rejects_wrong_dimension(name):
    config = assign_config(name, 2)

    with pytest.raises(ValueError, match="dimension"):
        config.f(np.zeros(3))


def test_fixed_2d_config_rejects_requested_non_2d_dimension():
    with pytest.raises(ValueError, match="2-dimensional|dim"):
        assign_config("DropWave", 7)


def test_mccormick_known_best_objective_is_maximization_scale():
    config = assign_config("McCormick", 2)

    assert config.known_best_x is not None
    assert config.known_best_objective == pytest.approx(-config.known_best_value)
    assert config.f(config.known_best_x) == pytest.approx(config.known_best_objective)


def test_mishra6_config_does_not_advertise_untrusted_optimum_metadata():
    config = assign_config("Mishra6", 2)

    assert config.known_best_value is None
    assert config.known_best_x is None
    assert config.known_best_objective is None


def test_upstream_mishra06_alias_and_2d_bounds_match_r_reference():
    mishra = assign_config("Mishra06", 2)
    shubert = assign_config("Shubert", 2)
    bukin = assign_config("Bukin6", 2)

    assert mishra.name == "Mishra062d"
    assert mishra.known_best_value is None
    assert np.allclose(shubert.bounds_lower, [-5.12, -5.12])
    assert np.allclose(shubert.bounds_upper, [5.12, 5.12])
    assert np.allclose(bukin.bounds_lower, [-15.0, -5.0])
    assert np.allclose(bukin.bounds_upper, [-3.0, 3.0])


def test_config_arrays_are_read_only():
    config = assign_config("Rastrigin", 2)

    with pytest.raises(ValueError):
        config.bounds_lower[0] = -10.0


def test_assign_config_rejects_non_string_name_cleanly():
    with pytest.raises((TypeError, ValueError), match="name"):
        assign_config(123, 2)


@pytest.mark.parametrize("dim", [2.5, "2", True, 0, -1])
def test_assign_config_rejects_invalid_dim_cleanly(dim):
    with pytest.raises((TypeError, ValueError), match="dim"):
        assign_config("Rastrigin", dim)
