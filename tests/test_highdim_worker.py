"""Tests for the Python single-task high-dim worker (Task 8).

``run_task`` takes a manifest task plus its loaded instance and shared starts,
runs the SMCO variant named by the task (maximising ``-instance.objective``),
and collects FE / quality / anytime / status into a result payload. The worker
uses an objective observer to record a best-so-far trace, so target-hit FE and
anytime checkpoints are uniform across all families (including BR, whose merged
context summary does not expose them).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from smco.experiment_manifests import build_algorithm_config, build_task
from smco.highdim_instances import generate_instance, write_instance_artifacts
from smco.highdim_worker import run_task
from smco.paper_contract import validate_result_row


def _starts(instance, n_starts=4, seed=0):
    rng = np.random.default_rng(seed)
    span = instance.bounds_upper - instance.bounds_lower
    return instance.bounds_lower + rng.uniform(size=(n_starts, instance.dimension)) * span


def _base_task(dim=4, fe_budget=200, family="smco"):
    cfg = build_algorithm_config(
        "python", family, False, "none",
        evolution_strategy="none", evolution_points=(),
        elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=4,
    )
    return build_task(
        "e0_contract", "contract", "Rastrigin", dim, 0, 0,
        config=cfg, fe_budget=fe_budget, checkpoints=(50, 100, 200), seed=42,
    )


def _evo_task(dim=4, fe_budget=300, family="smco", semantics="state_preserving"):
    cfg = build_algorithm_config(
        "python", family, True, semantics,
        evolution_strategy="rand1bin", evolution_points=(0.5, 0.75),
        elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=4,
    )
    return build_task(
        "e0_contract", "contract", "Rastrigin", dim, 0, 0,
        config=cfg, fe_budget=fe_budget, checkpoints=(75, 150, 300), seed=42,
    )


def test_run_task_base_smco_smoke():
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    starts = _starts(inst)
    res = run_task(_base_task(), inst, starts)
    assert res["status"] == "success"
    assert 0 < res["fe_used"] <= 200
    # minimization Rastrigin value is >= 0
    assert res["best_value"] >= -1e-9
    initial = float(np.median([inst.objective(s) for s in starts]))
    assert res["best_value"] <= initial + 1e-9
    assert set(res["target_hit_fe"]) == {"1e-1", "1e-2", "1e-3", "1e-5"}
    assert [a["checkpoint_fe"] for a in res["anytime"]] == [50, 100, 200]
    assert validate_result_row(res["result_row"]) == []
    assert res["result_row"]["run_id"] == _base_task()["run_id"]


def test_run_task_evo_sp_smoke():
    inst = generate_instance("Ackley", 4, 0, seed=2)
    starts = _starts(inst)
    res = run_task(_evo_task(), inst, starts)
    assert res["status"] == "success"
    assert res["fe_used"] <= 300
    assert res["best_value"] >= -1e-9
    assert set(res["target_hit_fe"]) == {"1e-1", "1e-2", "1e-3", "1e-5"}
    assert validate_result_row(res["result_row"]) == []


def test_run_task_evo_restart_smoke():
    inst = generate_instance("Rastrigin", 4, 0, seed=3)
    starts = _starts(inst)
    res = run_task(_evo_task(semantics="restart"), inst, starts)
    assert res["status"] == "success"
    assert res["fe_used"] <= 300
    assert validate_result_row(res["result_row"]) == []


def test_run_task_br_smoke():
    # BR family exercises the split-budget path; observer must still collect metrics.
    inst = generate_instance("Rastrigin", 4, 0, seed=4)
    starts = _starts(inst)
    task = _evo_task(family="smco_boost_refine", fe_budget=400)
    res = run_task(task, inst, starts)
    assert res["status"] == "success"
    assert res["fe_used"] <= 400
    assert set(res["target_hit_fe"]) == {"1e-1", "1e-2", "1e-3", "1e-5"}
    assert validate_result_row(res["result_row"]) == []


def test_run_task_rejects_r_language():
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    starts = _starts(inst)
    cfg = build_algorithm_config(
        "r", "smco", False, "none",
        evolution_strategy="none", evolution_points=(),
        elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=4,
    )
    task = build_task(
        "e0_contract", "contract", "Rastrigin", 4, 0, 0,
        config=cfg, fe_budget=200, checkpoints=(200,), seed=42,
    )
    with pytest.raises(ValueError):
        run_task(task, inst, starts)


def test_run_task_normalized_gap_in_unit_interval():
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    starts = _starts(inst)
    res = run_task(_base_task(), inst, starts)
    gap = float(res["result_row"]["normalized_gap"])
    assert not np.isnan(gap)
    assert -1e-9 <= gap <= 1.0 + 1e-9


def test_run_task_target_hit_fe_within_budget_and_ordered():
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    starts = _starts(inst)
    res = run_task(_base_task(fe_budget=600), inst, starts)
    hit = [(t, fe) for t, fe in res["target_hit_fe"].items() if fe is not None]
    for _, fe in hit:
        assert 0 < fe <= res["fe_used"]
    # tighter targets cannot be hit earlier than looser ones.
    order = ["1e-1", "1e-2", "1e-3", "1e-5"]
    present = [(t, res["target_hit_fe"][t]) for t in order if res["target_hit_fe"][t] is not None]
    fes = [fe for _, fe in present]
    assert fes == sorted(fes)


def test_run_task_fe_used_observed_equals_budget_cap_path():
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    starts = _starts(inst)
    # Tiny budget so the hard cap triggers; fe_used must not exceed it.
    task = _base_task(fe_budget=20)
    res = run_task(task, inst, starts)
    assert res["fe_used"] <= 20
    assert res["result_row"]["fe_used"] == res["fe_used"]


_WORKER_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "run_smco_evo_highdim_factorial.py"
)


def _load_worker_cli():
    spec = importlib.util.spec_from_file_location("smco_evo_factorial_cli", _WORKER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_task_file_end_to_end(tmp_path):
    cli = _load_worker_cli()
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    rng = np.random.default_rng(5)
    span = inst.bounds_upper - inst.bounds_lower
    starts = inst.bounds_lower + rng.uniform(size=(4, 4)) * span
    art_dir = tmp_path / "instances" / "dev_Rastrigin_d4_i0"
    meta = write_instance_artifacts(inst, starts, art_dir)

    cfg = build_algorithm_config(
        "python", "smco", False, "none",
        evolution_strategy="none", evolution_points=(),
        elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=4,
    )
    task = build_task(
        "e0_contract", "contract", "Rastrigin", 4, 0, 0,
        config=cfg, fe_budget=200, checkpoints=(50, 100, 200), seed=42,
        instance_artifact_dir="instances/dev_Rastrigin_d4_i0",
        instance_hash=meta["transform_sha256"],
        start_points_hash=meta["file_hashes"]["starts"],
    )
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task))

    result_dir = tmp_path / "raw"
    log_dir = tmp_path / "logs"
    rc = cli.run_task_file(
        str(task_path), instance_root=str(tmp_path),
        result_dir=str(result_dir), log_dir=str(log_dir),
    )
    assert rc == 0
    out = result_dir / f"{task['run_id']}.json"
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["status"] == "success"
    assert payload["fe_used"] <= 200
    assert payload["result_row"]["run_id"] == task["run_id"]
    assert (log_dir / f"{task['run_id']}.log").exists()


def test_run_task_file_rejects_instance_hash_mismatch(tmp_path):
    cli = _load_worker_cli()
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    rng = np.random.default_rng(5)
    span = inst.bounds_upper - inst.bounds_lower
    starts = inst.bounds_lower + rng.uniform(size=(4, 4)) * span
    art_dir = tmp_path / "instances" / "dev_Rastrigin_d4_i0"
    write_instance_artifacts(inst, starts, art_dir)
    cfg = build_algorithm_config(
        "python", "smco", False, "none",
        evolution_strategy="none", evolution_points=(),
        elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=4,
    )
    task = build_task(
        "e0_contract", "contract", "Rastrigin", 4, 0, 0,
        config=cfg, fe_budget=200, checkpoints=(200,), seed=42,
        instance_artifact_dir="instances/dev_Rastrigin_d4_i0",
        instance_hash="0" * 64,  # wrong on purpose
        start_points_hash="x",
    )
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task))
    rc = cli.run_task_file(
        str(task_path), instance_root=str(tmp_path),
        result_dir=str(tmp_path / "raw"), log_dir=str(tmp_path / "logs"),
    )
    assert rc == 1
    payload = json.loads((tmp_path / "raw" / f"{task['run_id']}.json").read_text())
    assert payload["status"] == "infra_failure"
