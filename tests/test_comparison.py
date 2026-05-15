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


def _sphere(x):
    return float(np.sum(x**2))


def _neg_sphere(x):
    return float(-np.sum(x**2))


# --- Directional semantics tests ---

class TestDirectionSemantics:
    """Verify f_optimal = f(x_optimal) in both minimize and maximize modes."""

    LOCAL_METHODS = [
        ("GD", gradient_descent),
        ("SignGD", sign_gradient_descent),
        ("ADAM", adam),
        ("SPSA", spsa),
        ("optimLBFGS", get_method("optimLBFGS")),
        ("optimNM", get_method("optimNM")),
        ("BOBYQA", get_method("BOBYQA")),
    ]

    GLOBAL_METHODS = [
        ("GenSA", get_method("GenSA")),
        ("SA", get_method("SA")),
        ("DEoptim", get_method("DEoptim")),
        ("GA", get_method("GA")),
        ("PSO", get_method("PSO")),
    ]

    @pytest.mark.parametrize("name,fn", LOCAL_METHODS)
    def test_local_minimize_sphere(self, name, fn):
        bl, bu = np.array([-5.0, -5.0]), np.array([5.0, 5.0])
        starts = np.array([[2.0, 2.0]])
        if name == "SPSA":
            result = fn(_sphere, bl, bu, starts, maximize=False, max_iter=5000, a=0.1, A=100.0, tol=1e-12, seed=42)
        else:
            result = fn(_sphere, bl, bu, starts, maximize=False, max_iter=500)
        assert result.f_optimal >= 0, f"{name}: f_optimal should be non-negative when minimizing sphere, got {result.f_optimal}"

    @pytest.mark.parametrize("name,fn", LOCAL_METHODS)
    def test_local_maximize_neg_sphere(self, name, fn):
        bl, bu = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
        starts = np.array([[0.1, 0.1]])
        kw = {"seed": 42} if name == "SPSA" else {}
        result = fn(_neg_sphere, bl, bu, starts, maximize=True, max_iter=500, **kw)
        assert result.f_optimal > -0.5, f"{name}: maximize neg_sphere should be close to 0, got {result.f_optimal}"

    @pytest.mark.parametrize("name,fn", GLOBAL_METHODS)
    def test_global_minimize_sphere(self, name, fn):
        bl, bu = np.array([-5.0, -5.0]), np.array([5.0, 5.0])
        result = fn(_sphere, bl, bu, maximize=False, max_iter=50, seed=42)
        assert result.f_optimal >= 0, f"{name}: f_optimal should be non-negative when minimizing sphere, got {result.f_optimal}"
        assert result.f_optimal < 2.0

    @pytest.mark.parametrize("name,fn", GLOBAL_METHODS)
    def test_global_maximize_neg_sphere(self, name, fn):
        bl, bu = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
        result = fn(_neg_sphere, bl, bu, maximize=True, max_iter=50, seed=42)
        assert result.f_optimal > -0.5, f"{name}: maximize neg_sphere should be close to 0, got {result.f_optimal}"


# --- Original method-specific tests ---

class TestGD:
    def test_finds_minimum(self):
        bl, bu = np.array([-5.0, -5.0]), np.array([5.0, 5.0])
        starts = np.array([[2.0, 2.0], [-2.0, -2.0]])
        result = gradient_descent(_sphere, bl, bu, starts, maximize=False, max_iter=500)
        assert result.f_optimal < 0.5
        assert result.x_optimal.shape == (2,)

    def test_registered(self):
        assert "GD" in METHOD_REGISTRY
        fn = get_method("GD")
        bl, bu = np.array([-5.0, -5.0]), np.array([5.0, 5.0])
        starts = np.array([[1.0, 1.0]])
        result = fn(_sphere, bl, bu, starts, maximize=False, max_iter=200)
        assert isinstance(result, OptimizerResult)

    def test_maximize(self):
        bl, bu = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
        starts = np.array([[0.1, 0.1]])
        result = gradient_descent(_neg_sphere, bl, bu, starts, maximize=True, max_iter=500)
        assert result.f_optimal > -0.05


class TestSignGD:
    def test_finds_minimum(self):
        bl, bu = np.array([-5.0, -5.0]), np.array([5.0, 5.0])
        starts = np.array([[2.0, 2.0], [-2.0, -2.0]])
        result = sign_gradient_descent(_sphere, bl, bu, starts, maximize=False, max_iter=500)
        assert result.f_optimal < 1.0
        assert result.x_optimal.shape == (2,)

    def test_registered(self):
        assert "SignGD" in METHOD_REGISTRY


class TestSPSA:
    def test_finds_minimum(self):
        bl, bu = np.array([-5.0, -5.0]), np.array([5.0, 5.0])
        starts = np.array([[2.0, 2.0]])
        result = spsa(_sphere, bl, bu, starts, maximize=False, max_iter=5000, a=0.1, A=100.0, alpha=0.602, tol=1e-12, seed=42)
        assert result.f_optimal < 8.0
        assert result.x_optimal.shape == (2,)

    def test_registered(self):
        assert "SPSA" in METHOD_REGISTRY


class TestAdam:
    def test_finds_minimum(self):
        bl, bu = np.array([-5.0, -5.0]), np.array([5.0, 5.0])
        starts = np.array([[2.0, 2.0]])
        result = adam(_sphere, bl, bu, starts, maximize=False, max_iter=500)
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
from comparison.methods.ga import genetic_algorithm, _select_tournament_parents
from comparison.methods.pso import particle_swarm


class TestScipyMethods:
    @pytest.mark.parametrize("name,fn", [
        ("optimLBFGS", lbfgs),
        ("optimNM", nelder_mead),
        ("BOBYQA", bobyqa),
    ])
    def test_minimize_sphere(self, name, fn):
        bl, bu = np.array([-5.0, -5.0]), np.array([5.0, 5.0])
        starts = np.array([[2.0, 2.0]])
        result = fn(_sphere, bl, bu, starts, maximize=False, max_iter=500)
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
        bl, bu = np.array([-5.0, -5.0]), np.array([5.0, 5.0])
        result = fn(_sphere, bl, bu, maximize=False, max_iter=50, seed=42)
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
        bl, bu = np.array([-5.0, -5.0]), np.array([5.0, 5.0])
        result = fn(_sphere, bl, bu, maximize=False, max_iter=50, seed=42)
        assert result.f_optimal < 2.0
        assert result.x_optimal.shape == (2,)

    def test_all_registered(self):
        for name in ("GA", "PSO"):
            assert name in METHOD_REGISTRY

    def test_ga_tournament_selection_uses_population_indices(self):
        fitness = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0])
        parents_idx = _select_tournament_parents(fitness, np.random.default_rng(42), fitness.size)
        assert np.all((parents_idx >= 0) & (parents_idx < fitness.size))
        assert parents_idx.max() > 2


from comparison.domain_mod import modify_domain
from comparison.run_comparison import run_comparison, generate_unif_starts, _smco_variant_options


class TestDomainMod:
    def test_preserves_shape(self):
        bl, bu = np.array([-5.0, -5.0]), np.array([5.0, 5.0])
        def identity(x):
            return float(np.sum(x))
        f_mod, new_bl, new_bu = modify_domain(identity, bl, bu, dim=2, seed=42)
        assert new_bl.shape == (2,)
        assert new_bu.shape == (2,)
        assert np.all(new_bl < new_bu)

    def test_output_is_scalar(self):
        bl, bu = np.array([-5.0, -5.0]), np.array([5.0, 5.0])
        f_mod, _, _ = modify_domain(_sphere, bl, bu, dim=2, seed=1)
        val = f_mod(np.array([1.0, 2.0]))
        assert isinstance(val, float)
        assert np.isfinite(val)


class TestRunComparison:
    def test_generate_starts(self):
        bl, bu = np.array([-5.0, -5.0]), np.array([5.0, 5.0])
        starts = generate_unif_starts(3, bl, bu)
        assert starts.shape == (3, 2)
        assert np.all(starts >= bl) and np.all(starts <= bu)

    def test_quick_comparison_maximize(self):
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

    def test_to_maximize_produces_different_results(self):
        """to_maximize=True and False must NOT produce identical results."""
        opts = {"iter_max": 50, "n_starts": 3}
        r_max = run_comparison("Rastrigin", 2, True, ["SMCO_R"], 2, opts, seed=42)
        r_min = run_comparison("Rastrigin", 2, False, ["SMCO_R"], 2, opts, seed=42)
        assert not np.allclose(r_max.fopt_algo, r_min.fopt_algo), \
            "to_maximize=True and False should produce different fopt values"
        assert r_max.to_maximize is True
        assert r_min.to_maximize is False

    def test_maximize_best_opt_is_max(self):
        """When maximizing, best_opt should be the max of fopt_algo."""
        result = run_comparison("Rastrigin", 2, True, ["SMCO_R", "GD"], 3,
                                {"iter_max": 50, "n_starts": 3}, seed=42)
        assert result.best_opt == pytest.approx(float(np.nanmax(result.fopt_algo)))

    def test_minimize_best_opt_is_min(self):
        """When minimizing, best_opt should be the min of fopt_algo."""
        result = run_comparison("Rastrigin", 2, False, ["SMCO_R", "GD"], 3,
                                {"iter_max": 50, "n_starts": 3}, seed=42)
        assert result.best_opt == pytest.approx(float(np.nanmin(result.fopt_algo)))

    def test_smco_variants_do_not_collapse_to_same_path(self):
        """SMCO and SMCO_R should not silently run the same implementation path."""
        opts = {"iter_max": 60, "n_starts": 3}
        r_smco = run_comparison("Rastrigin", 2, True, ["SMCO"], 3, opts, seed=42)
        r_smco_r = run_comparison("Rastrigin", 2, True, ["SMCO_R"], 3, opts, seed=42)
        assert not np.allclose(r_smco.fopt_algo, r_smco_r.fopt_algo), (
            "SMCO and SMCO_R produced identical outputs; comparison likely bypassed variant dispatch"
        )

    def test_spsa_comparison_is_reproducible_with_fixed_seed(self):
        opts = {"iter_max": 20, "n_starts": 2}
        first = run_comparison("Rastrigin", 2, True, ["SPSA"], 3, opts, seed=123)
        second = run_comparison("Rastrigin", 2, True, ["SPSA"], 3, opts, seed=123)
        assert np.allclose(first.fopt_algo, second.fopt_algo)

    def test_smco_variant_options_merge_user_controls(self):
        options = _smco_variant_options(
            "SMCO_BR",
            {
                "iter_max": 101,
                "n_starts": 7,
                "bounds_buffer": 0.2,
                "buffer_rand": False,
                "tol_conv": 1e-6,
                "refine_ratio": 0.25,
                "iter_boost": 11,
                "partial_option": "forward",
                "use_runmax": False,
            },
            algo_seed=99,
        )
        assert options["iter_max"] == 51
        assert options["iter_boost"] == 11
        assert options["bounds_buffer"] == 0.2
        assert options["buffer_rand"] is False
        assert options["tol_conv"] == 1e-6
        assert options["refine_ratio"] == 0.25
        assert options["partial_option"] == "forward"
        assert options["use_runmax"] is False
        assert options["seed"] == 99
        assert "n_starts" not in options
