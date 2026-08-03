"""Tests for the comparison-baseline worker (Task 9 / E3)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from smco.baseline_worker import BASELINE_NAMES, run_baseline_task
from smco.experiment_manifests import (
    build_manifest,
    expand_baseline_tasks,
    freeze_manifest,
)
from smco.highdim_instances import generate_instance, write_instance_artifacts
from smco.paper_contract import validate_outcome

_BASELINES = (
    Path(__file__).resolve().parent.parent / "scripts" / "run_smco_evo_highdim_baselines.py"
)


def _load_baselines_cli():
    spec = importlib.util.spec_from_file_location("smco_evo_baselines_cli", _BASELINES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert isinstance(res["best_so_far_trace"], list)
    assert res["supersedes_run_id"] == "none"
    assert res["termination_reason"] == "evaluation_budget"
    assert res["fe_counts_by_event"] == {}
    assert res["peak_memory_mb"] is None


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


def test_baseline_names_include_legacy_and_e7_comparators():
    assert set(BASELINE_NAMES) == {
        "DE", "GA", "PSO", "SA", "GenSA", "CMA-ES",
        "R-DEoptim", "STOGO", "L-BFGS", "SPSA", "SignGD",
    }


def test_run_baseline_batch_end_to_end(tmp_path):
    cli = _load_baselines_cli()
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    rng = np.random.default_rng(5)
    span = inst.bounds_upper - inst.bounds_lower
    starts = inst.bounds_lower + rng.uniform(size=(4, 4)) * span
    art = tmp_path / "inst" / "instances" / "dev_i0"
    meta = write_instance_artifacts(inst, starts, art)
    index = {
        ("Rastrigin", 4, 0): {
            "artifact_dir": "instances/dev_i0",
            "transform_sha256": meta["transform_sha256"],
            "start_points_hash": meta["file_hashes"]["starts"],
        }
    }
    tasks = expand_baseline_tasks(
        "e3_baselines_highdim", "synthetic_highdim", ["Rastrigin"], [4], 1,
        ["GenSA", "DE"], fe_budget_per_d=50, checkpoints_per_d=(50,), instance_index=index,
    )
    manifest = freeze_manifest(build_manifest("e3_baselines_highdim", "synthetic_highdim", tasks))
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))

    summary = cli.run_baseline_batch(mp, tmp_path / "raw", tmp_path / "inst", workers=2)
    assert summary["n_tasks"] == 2
    assert summary["dispatched"] == 2
    for task in tasks:
        out = tmp_path / "raw" / f"{task['run_id']}.json"
        assert out.exists()
        payload = json.loads(out.read_text())
        assert payload["status"] in ("success", "algorithm_failure", "infra_failure")
        if payload["status"] == "success":
            assert validate_outcome(payload) == []
            assert payload["task"]["algorithm"] in {"GenSA", "DE"}

    # resume: completed runs are skipped on the second pass
    summary2 = cli.run_baseline_batch(mp, tmp_path / "raw", tmp_path / "inst", workers=2)
    successes = sum(1 for t in tasks if cli.is_run_complete(tmp_path / "raw", t["run_id"]))
    assert summary2["dispatched"] == 2 - successes
