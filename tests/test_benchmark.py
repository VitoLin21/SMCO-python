import numpy as np
import pytest

from smco import BenchmarkRun, run_benchmark, run_paper_smoke


def test_run_benchmark_executes_tiny_configuration():
    result = run_benchmark(
        name="Rastrigin",
        dim=2,
        variant="smco_r",
        repetitions=2,
        iter_max=40,
        n_starts=5,
        seed=10,
    )

    assert isinstance(result, BenchmarkRun)
    assert result.config.name == "Rastrigin2d"
    assert len(result.results) == 2
    assert result.best_value == max(run.best_result.f_optimal for run in result.results)
    assert np.isfinite(result.best_value)


def test_run_benchmark_supports_all_variants():
    for variant in ["smco", "smco_r", "smco_br", "smco_multi"]:
        result = run_benchmark(
            name="Minus Norm Squared",
            dim=2,
            variant=variant,
            repetitions=1,
            iter_max=30,
            n_starts=5,
            seed=20,
        )

        assert len(result.results) == 1
        assert result.best_x.shape == (2,)


def test_run_benchmark_normalizes_smco_multi_variant():
    result = run_benchmark(
        name="Minus Norm Squared",
        dim=2,
        variant="SMCO_MULTI",
        repetitions=1,
        iter_max=30,
        n_starts=5,
        seed=21,
    )

    assert len(result.results) == 1
    assert result.best_x.shape == (2,)


def test_run_benchmark_smco_multi_uses_explicit_start_points():
    result = run_benchmark(
        name="Minus Norm Squared",
        dim=2,
        variant="smco_multi",
        repetitions=1,
        iter_max=10,
        start_points=[[0.0, 0.0]],
        seed=1,
    )

    assert len(result.results) == 1
    assert result.results[0].summary["n_starts"] == 1
    assert result.best_x.shape == (2,)


@pytest.mark.parametrize("seed", [1.2, True])
def test_run_benchmark_rejects_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        run_benchmark(
            name="Minus Norm Squared",
            dim=2,
            variant="smco_r",
            repetitions=1,
            iter_max=10,
            n_starts=5,
            seed=seed,
        )


def test_run_benchmark_rejects_non_string_variant_cleanly():
    with pytest.raises((TypeError, ValueError), match="variant"):
        run_benchmark(
            name="Minus Norm Squared",
            dim=2,
            variant=123,
            repetitions=1,
            iter_max=10,
            n_starts=5,
            seed=1,
        )


def test_run_paper_smoke_accepts_none_seed():
    runs = run_paper_smoke(seed=None, iter_max=10, n_starts=5)

    assert set(runs) == {"Rastrigin", "Ackley", "Griewank"}
    assert all(len(run.results) == 1 for run in runs.values())


def test_run_paper_smoke_returns_named_runs():
    runs = run_paper_smoke(iter_max=30, n_starts=5, seed=30)

    assert set(runs) == {"Rastrigin", "Ackley", "Griewank"}
    assert all(len(run.results) == 1 for run in runs.values())
