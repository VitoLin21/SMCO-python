"""Contract tests for the five prospective E7 comparator adapters."""

from __future__ import annotations

import numpy as np
import pytest

from smco.baseline_worker import run_baseline_task
from smco.e7_algorithm_adapters import (
    E7_ALGORITHM_IDS,
    E7_ALGORITHM_METADATA,
    UnsupportedAlgorithmError,
)
from smco.highdim_instances import HighDimInstance, generate_instance


def _starts(instance, n_starts=2, seed=19):
    rng = np.random.default_rng(seed)
    span = instance.bounds_upper - instance.bounds_lower
    return instance.bounds_lower + rng.uniform(
        size=(n_starts, instance.dimension)
    ) * span


class _RStub:
    def __init__(self, package_versions):
        self.package_versions = package_versions
        self.calls = []

    def preflight(self, *, algorithm_id, metadata):
        assert self.package_versions[metadata["package"]] == metadata["package_version"]

    def run(self, *, algorithm_id, objective, bounds_lower, bounds_upper,
            start_points, seed, max_iter, metadata):
        self.calls.append((algorithm_id, seed, metadata, max_iter))
        # Simulate native package callbacks, including a repeated objective
        # call. Every callback must pass through the Python FE observer.
        objective(start_points[0])
        objective((bounds_lower + bounds_upper) / 2.0)
        objective(start_points[0])


def test_e7_algorithm_ids_and_metadata_are_fully_frozen():
    assert E7_ALGORITHM_IDS == (
        "R-DEoptim", "STOGO", "L-BFGS", "SPSA", "SignGD",
    )
    required = {
        "language", "package", "package_version", "hyperparameters",
        "bounds_handling", "rng", "starts_semantics", "fe_counting",
    }
    assert set(E7_ALGORITHM_METADATA) == set(E7_ALGORITHM_IDS)
    for algorithm_id, metadata in E7_ALGORITHM_METADATA.items():
        assert set(metadata) == required, algorithm_id
        assert metadata["language"] in {"python", "r"}
        assert metadata["package"]
        assert metadata["package_version"]
        assert isinstance(metadata["hyperparameters"], dict)
        assert metadata["bounds_handling"]
        assert metadata["rng"]
        assert metadata["starts_semantics"]
        assert "objective" in metadata["fe_counting"].lower()


@pytest.mark.parametrize("algorithm_id", ["L-BFGS", "SPSA", "SignGD"])
def test_python_e7_adapters_count_every_objective_call_and_respect_bounds(
    algorithm_id, monkeypatch,
):
    instance = generate_instance("Rastrigin", 3, 0, seed=7)
    starts = _starts(instance, n_starts=1)
    calls = []
    original = HighDimInstance.objective

    def counted_objective(self, x):
        x = np.asarray(x, dtype=float)
        assert np.all(x >= self.bounds_lower)
        assert np.all(x <= self.bounds_upper)
        calls.append(np.array(x, copy=True))
        return original(self, x)

    monkeypatch.setattr(HighDimInstance, "objective", counted_objective)
    result = run_baseline_task(
        algorithm_id,
        instance,
        starts,
        fe_budget=24,
        seed=42,
        checkpoints=(8, 16, 24),
    )

    assert result["status"] == "success"
    assert result["fe_used"] == len(calls)
    assert result["fe_used"] <= 24
    assert sum(result["fe_counts_by_event"].values()) == result["fe_used"]
    assert result["fe_counts_by_event"]["initialization"] == 1
    assert result["algorithm_metadata"] == E7_ALGORITHM_METADATA[algorithm_id]


@pytest.mark.parametrize("algorithm_id", ["R-DEoptim", "STOGO"])
def test_r_adapters_use_distinct_ids_and_controllable_stub(algorithm_id):
    instance = generate_instance("Ackley", 3, 0, seed=3)
    starts = _starts(instance, n_starts=2)
    metadata = E7_ALGORITHM_METADATA[algorithm_id]
    stub = _RStub({metadata["package"]: metadata["package_version"]})

    result = run_baseline_task(
        algorithm_id,
        instance,
        starts,
        fe_budget=10,
        seed=13,
        checkpoints=(5, 10),
        e7_r_backend=stub,
    )

    assert result["status"] == "success"
    assert result["algorithm_id"] == algorithm_id
    # two frozen-start initial-reference calls + three native callbacks
    assert result["fe_used"] == 5
    assert stub.calls[0][0] == algorithm_id
    assert result["algorithm_metadata"]["language"] == "r"
    # Initial-reference evaluations are charged first; the native adapter gets
    # only the still-available FE budget.
    assert stub.calls[0][3] == 8


def test_e7_rejects_budget_smaller_than_frozen_start_set():
    instance = generate_instance("Ackley", 3, 0, seed=3)
    starts = _starts(instance, n_starts=3)
    with pytest.raises(ValueError, match="at least n_starts"):
        run_baseline_task(
            "SPSA", instance, starts, fe_budget=2, seed=13,
            checkpoints=(2,),
        )


def test_r_deoptim_population_is_capped_for_ultrahigh_dimensions():
    metadata = E7_ALGORITHM_METADATA["R-DEoptim"]
    assert metadata["hyperparameters"]["NP"] == (
        "max(n_starts, min(512, max(50, 10*d)))"
    )


def test_stogo_metadata_freezes_balanced_total_budget_split():
    metadata = E7_ALGORITHM_METADATA["STOGO"]
    assert metadata["hyperparameters"]["maxeval"] == (
        "balanced_split_of_remaining_fe_budget_across_starts"
    )


@pytest.mark.parametrize("algorithm_id", ["R-DEoptim", "STOGO"])
def test_missing_r_runtime_or_package_is_explicitly_unsupported(
    algorithm_id, monkeypatch,
):
    instance = generate_instance("Ackley", 2, 0, seed=2)
    starts = _starts(instance, n_starts=1)

    def unsupported_backend():
        raise UnsupportedAlgorithmError("controlled missing R package")

    monkeypatch.setattr(
        "smco.e7_algorithm_adapters.default_r_backend", unsupported_backend,
    )
    result = run_baseline_task(
        algorithm_id,
        instance,
        starts,
        fe_budget=10,
        seed=1,
        checkpoints=(10,),
    )

    assert result["status"] == "algorithm_failure"
    assert result["failure_reason"].startswith("unsupported_dependency:")
    assert "controlled missing R package" in result["failure_reason"]
    assert result["fe_used"] == 0
    assert result["termination_reason"] == "error"


def test_r_package_version_mismatch_is_not_silently_accepted():
    instance = generate_instance("Ackley", 2, 0, seed=2)
    starts = _starts(instance, n_starts=1)

    class WrongVersion(_RStub):
        def preflight(self, *, algorithm_id, metadata):
            raise UnsupportedAlgorithmError(
                f"{metadata['package']} version mismatch: installed=0 expected="
                f"{metadata['package_version']}"
            )

    result = run_baseline_task(
        "R-DEoptim",
        instance,
        starts,
        fe_budget=10,
        seed=1,
        checkpoints=(10,),
        e7_r_backend=WrongVersion({"DEoptim": "0"}),
    )
    assert result["status"] == "algorithm_failure"
    assert "version mismatch" in result["failure_reason"]
    assert result["fe_used"] == 0
