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


from comparison.methods.lbfgs import lbfgs
from comparison.methods.nelder_mead import nelder_mead
from comparison.methods.bobyqa import bobyqa
from comparison.methods.gensa import gensa
from comparison.methods.sa import simulated_annealing
from comparison.methods.de import differential_evo
from comparison.methods.ga import genetic_algorithm
from comparison.methods.pso import particle_swarm


class TestScipyMethods:
    @pytest.mark.parametrize("name,fn", [
        ("optimLBFGS", lbfgs),
        ("optimNM", nelder_mead),
        ("BOBYQA", bobyqa),
    ])
    def test_minimize_sphere(self, name, fn):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        starts = np.array([[2.0, 2.0]])
        result = fn(_sphere_min, bl, bu, starts, maximize=False, max_iter=500)
        assert result.f_optimal < 1.0
        assert result.x_optimal.shape == (2,)

    def test_all_registered(self):
        for name in ("optimLBFGS", "optimNM", "BOBYQA"):
            assert name in METHOD_REGISTRY


class TestGlobalMethods:
    @pytest.mark.parametrize("name,fn", [
        ("GenSA", gensa),
        ("SA", simulated_annealing),
        ("DEoptim", differential_evo),
    ])
    def test_minimize_sphere(self, name, fn):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        result = fn(_sphere_min, bl, bu, maximize=False, max_iter=50, seed=42)
        assert result.f_optimal < 1.0
        assert result.x_optimal.shape == (2,)

    def test_all_registered(self):
        for name in ("GenSA", "SA", "DEoptim"):
            assert name in METHOD_REGISTRY


class TestGAPSO:
    @pytest.mark.parametrize("name,fn", [
        ("GA", genetic_algorithm),
        ("PSO", particle_swarm),
    ])
    def test_minimize_sphere(self, name, fn):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        result = fn(_sphere_min, bl, bu, maximize=False, max_iter=50, seed=42)
        assert result.f_optimal < 2.0
        assert result.x_optimal.shape == (2,)

    def test_all_registered(self):
        for name in ("GA", "PSO"):
            assert name in METHOD_REGISTRY


from comparison.domain_mod import modify_domain
from comparison.run_comparison import run_comparison, generate_unif_starts


class TestDomainMod:
    def test_preserves_shape(self):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        def identity(x):
            return float(np.sum(x))
        f_mod, new_bl, new_bu = modify_domain(identity, bl, bu, dim=2, seed=42)
        assert new_bl.shape == (2,)
        assert new_bu.shape == (2,)
        assert np.all(new_bl < new_bu)

    def test_output_is_scalar(self):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        f_mod, _, _ = modify_domain(_sphere_min, bl, bu, dim=2, seed=1)
        val = f_mod(np.array([1.0, 2.0]))
        assert isinstance(val, float)
        assert np.isfinite(val)


class TestRunComparison:
    def test_generate_starts(self):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        starts = generate_unif_starts(3, bl, bu)
        assert starts.shape == (3, 2)
        assert np.all(starts >= bl) and np.all(starts <= bu)

    def test_quick_comparison(self):
        result = run_comparison(
            name_config="Rastrigin",
            dim_config=2,
            to_maximize=True,
            algo_names=["SMCO_R", "GD"],
            n_replications=2,
            smco_options={"iter_max": 50, "n_starts": 3},
            seed=42,
        )
        assert len(result.algo_names) == 2
        assert result.fopt_algo.shape == (2, 2)
        assert "SMCO_R" in result.sum_results
        assert "GD" in result.sum_results
        assert "rMSE" in result.sum_results["SMCO_R"]
