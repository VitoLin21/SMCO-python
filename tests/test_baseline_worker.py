"""Tests for the comparison-baseline worker (Task 9 / E3)."""

from __future__ import annotations

import numpy as np
import pytest

from smco.baseline_worker import BASELINE_NAMES, run_baseline_task
from smco.highdim_instances import generate_instance


def _starts(instance, n_starts=4, seed=0):
    rng = np.random.default_rng(seed)
    span = instance.bounds_upper - instance.bounds_lower
    return instance.bounds_lower + rng.uniform(size=(n_starts, instance.dimension)) * span


@pytest.mark.parametrize("algorithm_name", ["GenSA", "DE", "PSO", "GA", "SA"])
def test_run_baseline_task_smoke(algorithm_name):
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    starts = _starts(inst)
    res = run_baseline_task(
        algorithm_name, inst, starts,
        fe_budget=200, seed=42, checkpoints=(50, 100, 200),
    )
    assert res["status"] in ("success", "algorithm_failure")
    assert res["fe_used"] <= 200  # FE hard stop honoured
    assert res["best_value"] >= -1e-9  # Rastrigin minimisation is >= 0
    assert set(res["target_hit_fe"]) == {"1e-1", "1e-2", "1e-3", "1e-5"}
    assert [a["checkpoint_fe"] for a in res["anytime"]] == [50, 100, 200]


def test_run_baseline_task_fe_hard_stop_on_tiny_budget():
    inst = generate_instance("Ackley", 4, 0, seed=2)
    starts = _starts(inst)
    res = run_baseline_task("GenSA", inst, starts, fe_budget=15, seed=1, checkpoints=(15,))
    assert res["fe_used"] <= 15


def test_run_baseline_task_rejects_unknown_algorithm():
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    starts = _starts(inst)
    with pytest.raises(ValueError):
        run_baseline_task("CMAES", inst, starts, fe_budget=100, seed=1, checkpoints=(100,))


def test_baseline_names_are_the_strong_set():
    assert set(BASELINE_NAMES) == {"DE", "GA", "PSO", "SA", "GenSA"}
