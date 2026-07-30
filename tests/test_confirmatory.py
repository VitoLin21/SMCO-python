"""Tests for confirmatory-run enforcement (Task 10 / Gate F 强制检查)."""

from __future__ import annotations

import pytest

from smco.confirmatory import (
    build_confirmatory_manifest,
    confirmatory_errors,
    enforce_confirmatory,
    is_run_complete,
    plan_batch,
)
from smco.experiment_manifests import (
    build_algorithm_config,
    build_manifest,
    build_task,
    e1_algorithm_configs,
    expand_tasks,
    freeze_manifest,
    manifest_sha256,
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


# --- A-03: selection-driven confirmatory manifest + Gate-F closure ---

def _selection(winner="PY-SP-SMCO-EVO", language="python", sel_hash="s1"):
    return {"winner": winner, "winner_language": language, "selection_hash": sel_hash}


def _build_e2(selection=None, baselines=()):
    sel = selection or _selection()
    return build_confirmatory_manifest(
        sel, stage="e2_factorial_highdim", suite="synthetic_highdim",
        functions=["Rastrigin"], dims=[200], n_instances=1,
        fe_budget_per_d=1000, checkpoints_per_d=(1000,), baselines=baselines,
    )


def test_build_confirmatory_manifest_carries_closure_fields():
    manifest = _build_e2()
    assert manifest["frozen"] is True
    assert manifest["selection_hash"] == "s1"
    assert "winner_config_hash" in manifest
    assert "matched_base_config_hash" in manifest
    algos = {t["algorithm_id"] for t in manifest["tasks"]}
    assert algos == {"PY-SP-SMCO-EVO", "PY-BASE-SMCO"}  # winner + matched base
    assert set(manifest["allowed_algorithms"]) == algos
    assert confirmatory_errors(manifest, selection=_selection()) == []


def test_build_confirmatory_manifest_includes_baselines():
    manifest = _build_e2(baselines=("DE", "CMA-ES"))
    algos = {t.get("algorithm_id") or t.get("algorithm") for t in manifest["tasks"]}
    assert {"PY-SP-SMCO-EVO", "PY-BASE-SMCO", "DE", "CMA-ES"} <= algos
    assert set(manifest["allowed_algorithms"]) == algos


def test_confirmatory_rejects_algorithm_outside_allowed():
    manifest = _build_e2()
    r_cfg = build_algorithm_config(
        "r", "smco", True, "state_preserving", evolution_strategy="rand1bin",
        evolution_points=(0.5, 0.75), elimination_rate=0.25, de_factor=0.8,
        de_crossover=0.7, n_starts=8,
    )
    extra = build_task("e2_factorial_highdim", "synthetic_highdim", "Rastrigin",
                       200, 0, 0, config=r_cfg, fe_budget=200000, checkpoints=(1000,), seed=1)
    manifest["tasks"].append(extra)
    manifest["manifest_sha256"] = manifest_sha256(manifest)  # keep hash consistent
    errors = confirmatory_errors(manifest)
    assert any("outside the allowed set" in e for e in errors)


def test_confirmatory_rejects_wrong_selection_hash():
    manifest = _build_e2(_selection(sel_hash="s1"))
    errors = confirmatory_errors(manifest, selection=_selection(sel_hash="different"))
    assert any("selection_hash mismatch" in e for e in errors)
