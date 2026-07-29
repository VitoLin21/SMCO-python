"""Tests for confirmatory-run enforcement (Task 10 / Gate F 强制检查)."""

from __future__ import annotations

import pytest

from smco.confirmatory import (
    confirmatory_errors,
    enforce_confirmatory,
    is_run_complete,
    plan_batch,
)
from smco.experiment_manifests import (
    build_manifest,
    build_task,
    e1_algorithm_configs,
    expand_tasks,
    freeze_manifest,
)


def _frozen_manifest(tasks=None):
    return freeze_manifest(
        build_manifest("e2_factorial_highdim", "synthetic_highdim", tasks or [])
    )


def test_confirmatory_passes_frozen_clean_manifest():
    manifest = _frozen_manifest()
    assert confirmatory_errors(manifest) == []
    assert enforce_confirmatory(manifest) is True


def test_confirmatory_flags_unfrozen_manifest():
    manifest = build_manifest("e2_factorial_highdim", "synthetic_highdim", [])
    errors = confirmatory_errors(manifest)
    assert any("not frozen" in e for e in errors)


def test_confirmatory_flags_tampered_hash():
    manifest = freeze_manifest(build_manifest("e2_factorial_highdim", "synthetic_highdim", []))
    manifest["frozen"] = False  # mutate after freeze
    errors = confirmatory_errors(manifest)
    assert any("mismatch" in e for e in errors)


def test_confirmatory_selection_winner_must_be_in_manifest():
    cfg = e1_algorithm_configs()[0]
    task = build_task(
        "e2_factorial_highdim", "synthetic_highdim", "Rastrigin", 200, 0, 0,
        config=cfg, fe_budget=200000, checkpoints=(20000,), seed=1,
    )
    manifest = _frozen_manifest([task])
    bad = confirmatory_errors(manifest, selection={"winner": "DOES-NOT-EXIST"})
    assert any("not present" in e for e in bad)
    good = confirmatory_errors(manifest, selection={"winner": cfg["algorithm_id"]})
    assert good == []


def test_confirmatory_selection_without_winner_flagged():
    manifest = _frozen_manifest()
    errors = confirmatory_errors(manifest, selection={"winner": None})
    assert any("no winner" in e for e in errors)


def test_enforce_raises_on_violations():
    with pytest.raises(ValueError):
        enforce_confirmatory(build_manifest("e2", "syn", []))  # unfrozen


def test_plan_batch_and_is_run_complete(tmp_path):
    tasks = [
        {"run_id": "r1", "fe_budget": 100},
        {"run_id": "r2", "fe_budget": 200},
    ]
    plan = plan_batch(tasks, tmp_path)
    assert plan["n_tasks"] == 2
    assert plan["completed"] == 0
    assert plan["missing"] == 2
    assert plan["total_fe_budget"] == 300
    assert is_run_complete(tmp_path, "r1") is False
    (tmp_path / "r1.json").write_text('{"status": "success"}')
    assert is_run_complete(tmp_path, "r1") is True
    plan2 = plan_batch(tasks, tmp_path)
    assert plan2["completed"] == 1 and plan2["missing"] == 1
