"""Tests for the E1 batch runner (Task 9): manifest -> worker dispatch.

The batch runner lives in scripts/run_smco_evo_highdim_factorial.py alongside
the single-task worker. It loads a frozen manifest, filters tasks, plans
(resume-aware), and dispatches each task to the Python or R worker subprocess.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from smco.experiment_manifests import (
    build_manifest,
    e1_algorithm_configs,
    expand_tasks,
    freeze_manifest,
)
from smco.highdim_instances import generate_instance, write_instance_artifacts

_FACTORIAL = (
    Path(__file__).resolve().parent.parent / "scripts" / "run_smco_evo_highdim_factorial.py"
)


def _cli():
    spec = importlib.util.spec_from_file_location("smco_evo_factorial_cli", _FACTORIAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pilot_manifest(tmp_path, *, configs=None, dim=4, fe_per_d=100, func="Rastrigin"):
    """Build a tiny frozen manifest + its instance artifact under tmp_path."""
    inst_root = tmp_path / "inst"
    inst = generate_instance(func, dim, 0, seed=1)
    rng = np.random.default_rng(5)
    span = inst.bounds_upper - inst.bounds_lower
    starts = inst.bounds_lower + rng.uniform(size=(4, dim)) * span
    art = inst_root / "instances" / "dev_i0"
    meta = write_instance_artifacts(inst, starts, art)
    index = {
        (func, dim, 0): {
            "artifact_dir": "instances/dev_i0",
            "transform_sha256": meta["transform_sha256"],
            "start_points_hash": meta["file_hashes"]["starts"],
        }
    }
    if configs is None:
        configs = [c for c in e1_algorithm_configs() if c["language"] == "python"]
    tasks = expand_tasks(
        "e0_contract", "contract", [func], [dim], 1, configs,
        fe_budget_per_d=fe_per_d, checkpoints_per_d=(50, 100), instance_index=index,
    )
    manifest = freeze_manifest(build_manifest("e0_contract", "contract", tasks))
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))
    return mp, inst_root, manifest


def test_load_manifest_tasks_respects_filters(tmp_path):
    cli = _cli()
    mp, inst_root, manifest = _pilot_manifest(tmp_path, configs=e1_algorithm_configs())  # 18 incl R
    all_tasks = cli.load_manifest_tasks(mp)
    assert len(all_tasks) == 18
    py_only = cli.load_manifest_tasks(mp, only_language="python")
    assert len(py_only) == 9 and all(t["language"] == "python" for t in py_only)
    r_only = cli.load_manifest_tasks(mp, only_language="r")
    assert len(r_only) == 9
    by_id = cli.load_manifest_tasks(mp, only_run_ids=[all_tasks[0]["run_id"]])
    assert len(by_id) == 1


def test_is_run_complete_detects_success_and_missing(tmp_path):
    cli = _cli()
    result_dir = tmp_path / "raw"
    result_dir.mkdir()
    assert cli.is_run_complete(result_dir, "rabc") is False
    cli._atomic_write_json(result_dir / "rabc.json", {"run_id": "rabc", "status": "success"})
    assert cli.is_run_complete(result_dir, "rabc") is True
    # infra_failure is NOT complete -> the run must be retried on resume.
    cli._atomic_write_json(result_dir / "rdef.json", {"run_id": "rdef", "status": "infra_failure"})
    assert cli.is_run_complete(result_dir, "rdef") is False


def test_plan_batch_reports_counts(tmp_path):
    cli = _cli()
    mp, inst_root, manifest = _pilot_manifest(tmp_path)  # 9 python configs
    result_dir = tmp_path / "raw"
    tasks = cli.load_manifest_tasks(mp)
    plan = cli.plan_batch(tasks, result_dir)
    assert plan["n_tasks"] == 9
    assert plan["completed"] == 0
    assert plan["missing"] == 9
    assert plan["total_fe_budget"] == sum(t["fe_budget"] for t in tasks)
    cli._atomic_write_json(
        result_dir / f"{tasks[0]['run_id']}.json",
        {"run_id": tasks[0]["run_id"], "status": "success"},
    )
    plan2 = cli.plan_batch(tasks, result_dir)
    assert plan2["completed"] == 1 and plan2["missing"] == 8


def test_worker_command_routes_language(tmp_path):
    cli = _cli()
    cmd_py = cli._worker_command(
        {"language": "python"}, tmp_path / "t.json", tmp_path / "inst", tmp_path / "raw", tmp_path / "log"
    )
    assert cmd_py[0] == sys.executable
    assert cmd_py[1].endswith("run_smco_evo_highdim_factorial.py")
    cmd_r = cli._worker_command(
        {"language": "r"}, tmp_path / "t.json", tmp_path / "inst", tmp_path / "raw", tmp_path / "log"
    )
    assert cmd_r[0] == "Rscript"
    assert cmd_r[1].endswith("run_smco_evo_highdim_r.R")


def test_run_batch_dry_run_does_not_dispatch(tmp_path):
    cli = _cli()
    mp, inst_root, manifest = _pilot_manifest(tmp_path)
    summary = cli.run_batch(
        mp, tmp_path / "raw", inst_root, workers=1, only_language="python",
        log_dir=tmp_path / "logs", dry_run=True,
    )
    assert summary["dry_run"] is True
    assert summary["n_tasks"] == 9
    assert summary["dispatched"] == 0
    assert not (tmp_path / "raw").exists() or not list((tmp_path / "raw").iterdir())


def test_run_batch_pilot_python_only_e2e(tmp_path):
    cli = _cli()
    mp, inst_root, manifest = _pilot_manifest(tmp_path)  # 9 python configs, d=4, 100 FE/d
    result_dir = tmp_path / "raw"
    summary = cli.run_batch(
        mp, result_dir, inst_root, workers=2, only_language="python", log_dir=tmp_path / "logs"
    )
    assert summary["n_tasks"] == 9
    assert summary["dispatched"] == 9
    for task in cli.load_manifest_tasks(mp):
        out = result_dir / f"{task['run_id']}.json"
        assert out.exists()
        payload = json.loads(out.read_text())
        assert payload["status"] in ("success", "algorithm_failure", "infra_failure")
    # resume: every successful run is skipped on the second pass.
    summary2 = cli.run_batch(
        mp, result_dir, inst_root, workers=2, only_language="python", log_dir=tmp_path / "logs"
    )
    successes = sum(
        1
        for task in cli.load_manifest_tasks(mp)
        if cli.is_run_complete(result_dir, task["run_id"])
    )
    assert summary2["dispatched"] == 9 - successes
