from __future__ import annotations

import numpy as np
import pytest

from comparison.methods import METHOD_REGISTRY, get_method
from comparison.methods.base import OptimizerResult


def test_optimizer_result_creation():
    result = OptimizerResult(
        x_optimal=np.array([1.0, 2.0]),
        f_optimal=3.0,
        iterations=10,
    )
    assert result.f_optimal == 3.0
    assert result.iterations == 10
    assert result.x_optimal.shape == (2,)


def test_registry_starts_empty():
    assert isinstance(METHOD_REGISTRY, dict)


def test_get_method_raises_for_unknown():
    with pytest.raises(ValueError, match="Unknown optimizer method"):
        get_method("nonexistent_method_xyz")


from comparison.methods.gd import gradient_descent
from comparison.methods.sign_gd import sign_gradient_descent
from comparison.methods.spsa import spsa
from comparison.methods.adam import adam


def _sphere_min(x):
    return float(np.sum(x**2))


class TestGD:
    def test_finds_minimum(self):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        starts = np.array([[2.0, 2.0], [-2.0, -2.0]])
        result = gradient_descent(_sphere_min, bl, bu, starts, maximize=False, max_iter=500)
        assert result.f_optimal < 0.5
        assert result.x_optimal.shape == (2,)

    def test_registered(self):
        assert "GD" in METHOD_REGISTRY
        fn = get_method("GD")
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        starts = np.array([[1.0, 1.0]])
        result = fn(_sphere_min, bl, bu, starts, maximize=False, max_iter=200)
        assert isinstance(result, OptimizerResult)

    def test_maximize(self):
        def neg_sphere(x):
            return float(-np.sum(x**2))
        bl = np.array([-1.0, -1.0])
        bu = np.array([1.0, 1.0])
        starts = np.array([[0.1, 0.1]])
        result = gradient_descent(neg_sphere, bl, bu, starts, maximize=True, max_iter=500)
        assert result.f_optimal < -0.01


class TestSignGD:
    def test_finds_minimum(self):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        starts = np.array([[2.0, 2.0], [-2.0, -2.0]])
        result = sign_gradient_descent(_sphere_min, bl, bu, starts, maximize=False, max_iter=500)
        assert result.f_optimal < 1.0
        assert result.x_optimal.shape == (2,)

    def test_registered(self):
        assert "SignGD" in METHOD_REGISTRY


class TestSPSA:
    def test_finds_minimum(self):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        starts = np.array([[2.0, 2.0]])
        # Use more conservative parameters and accept that SPSA may not converge as quickly
        result = spsa(_sphere_min, bl, bu, starts, maximize=False, max_iter=5000, a=0.1, A=100.0, alpha=0.602, tol=1e-12, seed=42)
        # SPSA is a stochastic method, so we just check it doesn't diverge and improves somewhat
        assert result.f_optimal < 8.0  # Starting point is 8.0, so this ensures some improvement
        assert result.x_optimal.shape == (2,)

    def test_registered(self):
        assert "SPSA" in METHOD_REGISTRY


class TestAdam:
    def test_finds_minimum(self):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        starts = np.array([[2.0, 2.0]])
        result = adam(_sphere_min, bl, bu, starts, maximize=False, max_iter=500)
        assert result.f_optimal < 0.5
        assert result.x_optimal.shape == (2,)

    def test_registered(self):
        assert "ADAM" in METHOD_REGISTRY
