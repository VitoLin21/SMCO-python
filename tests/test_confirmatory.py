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
    cfg = build_algorithm_config("python", "smco", True, "state_preserving",
        evolution_strategy="rand1bin", evolution_points=(0.5, 0.75),
        elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=8)
    sel = {"winner": cfg["algorithm_id"], "winner_language": "python",
           "selection_hash": "s1", "winner_config_hash": cfg["configuration_hash"]}
    manifest = build_confirmatory_manifest(
        sel, stage="e2_factorial_highdim", suite="synthetic_highdim",
        functions=["Rastrigin"], dims=[200], n_instances=1,
        fe_budget_per_d=1000, checkpoints_per_d=(1000,))
    bad = confirmatory_errors(manifest, selection={"winner": "DOES-NOT-EXIST",
                                                    "selection_hash": "s1"})
    assert any("not present" in e for e in bad)
    good = confirmatory_errors(manifest, selection=sel)
    assert good == []


def test_confirmatory_requires_closure_hashes_when_selection_given():
    # R-02: a plain frozen manifest (no closure hashes) must be rejected when a
    # selection is supplied — not silently accepted.
    manifest = _frozen_manifest()
    errors = confirmatory_errors(manifest, selection={"winner": "PY-SP-SMCO-EVO",
                                                       "selection_hash": "s1"})
    assert any("missing selection_hash" in e for e in errors)
    assert any("missing winner_config_hash" in e for e in errors)


def test_generate_manifest_cli_is_selection_driven(tmp_path):
    # R-02: the manifest CLI can build a confirmatory manifest from selection.json.
    import importlib.util
    import json as _json
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "gen_cli", Path("scripts/generate_smco_evo_manifests.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    sel = {"winner": "PY-SP-SMCO-EVO", "winner_language": "python",
           "selection_hash": "s1", "winner_config_hash": "cfg_x"}
    sel_path = tmp_path / "selection.json"
    sel_path.write_text(_json.dumps(sel))
    out_dir = tmp_path / "out"
    rc = cli.main(["--stage", "manifest", "--selection", str(sel_path),
                   "--manifest-stage", "e2_factorial_highdim", "--suite", "synthetic_highdim",
                   "--functions", "Rastrigin", "--dims", "200", "--n-instances", "1",
                   "--fe-budget-per-d", "1000", "--checkpoints-per-d", "1000",
                   "--out-dir", str(out_dir)])
    assert rc == 0
    manifests = list(out_dir.glob("*.json"))
    assert len(manifests) == 1
    manifest = _json.loads(manifests[0].read_text())
    assert manifest["frozen"] is True
    assert manifest["selection_hash"] == "s1"
    assert "winner_config_hash" in manifest
    assert "PY-SP-SMCO-EVO" in {t["algorithm_id"] for t in manifest["tasks"]}


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


# --- R2b: confirmatory manifest must lock stage + suite + run matrix ---

def _build_e4_manifest(dims=(160, 320), n_instances=2, fe_budget_per_d=1000,
                       baselines=("DE", "GA")):
    sel = {"winner": "PY-SP-SMCO-EVO", "winner_language": "python",
           "selection_hash": "s1", "winner_config_hash": "cfg"}
    from smco.confirmatory import build_confirmatory_manifest
    return build_confirmatory_manifest(
        sel, stage="e4_bbob_largescale", suite="bbob-largescale",
        functions=["Rastrigin"], dims=list(dims), n_instances=n_instances,
        fe_budget_per_d=fe_budget_per_d, checkpoints_per_d=(1000,), baselines=baselines,
    )


def test_confirmatory_run_matrix_extracts_locked_matrix():
    from smco.confirmatory import confirmatory_run_matrix
    manifest = _build_e4_manifest(dims=(160, 320), n_instances=2, fe_budget_per_d=1000)
    matrix = confirmatory_run_matrix(
        manifest, expected_stage="e4_bbob_largescale", expected_suite="bbob-largescale")
    assert matrix["suite"] == "bbob-largescale"
    assert matrix["dims"] == [160, 320]
    assert matrix["n_instances"] == 2
    assert matrix["fe_budget_per_d"] == 1000


def test_confirmatory_run_matrix_rejects_wrong_stage():
    from smco.confirmatory import confirmatory_run_matrix
    manifest = _build_e4_manifest()
    # an E4 manifest must not drive an E5 runner (and vice versa)
    with pytest.raises(ValueError, match="stage"):
        confirmatory_run_matrix(manifest, expected_stage="e5_lowdim_check")


def test_confirmatory_run_matrix_rejects_wrong_suite():
    from smco.confirmatory import confirmatory_run_matrix
    manifest = _build_e4_manifest()
    with pytest.raises(ValueError, match="suite"):
        confirmatory_run_matrix(
            manifest, expected_stage="e4_bbob_largescale", expected_suite="bbob")


def test_confirmatory_run_matrix_rejects_mixed_budget():
    from smco.confirmatory import confirmatory_run_matrix
    from smco.experiment_manifests import build_manifest, freeze_manifest, manifest_sha256
    # two tasks with different fe_budget_per_d (1000 vs 2000) inside one stage
    t1 = {"dimension": 160, "instance": 0, "fe_budget": 160000, "suite": "bbob-largescale"}
    t2 = {"dimension": 320, "instance": 0, "fe_budget": 640000, "suite": "bbob-largescale"}
    manifest = freeze_manifest(
        build_manifest("e4_bbob_largescale", "bbob-largescale", [t1, t2]))
    manifest["manifest_sha256"] = manifest_sha256(manifest)  # keep consistent
    with pytest.raises(ValueError, match="mixes fe_budget_per_d"):
        confirmatory_run_matrix(
            manifest, expected_stage="e4_bbob_largescale", expected_suite="bbob-largescale")


def test_confirmatory_run_matrix_rejects_empty_tasks():
    from smco.confirmatory import confirmatory_run_matrix
    from smco.experiment_manifests import build_manifest, freeze_manifest
    manifest = freeze_manifest(build_manifest("e4_bbob_largescale", "bbob-largescale", []))
    with pytest.raises(ValueError, match="no tasks"):
        confirmatory_run_matrix(
            manifest, expected_stage="e4_bbob_largescale", expected_suite="bbob-largescale")


# --- R6c: the E4 manifest must be the full 7-config x 24-function matrix ---

def _coco_tasks(algos, functions, dims, n_instances, *, fe_budget_per_d=1000):
    tasks = []
    for fn in functions:
        for d in dims:
            for i in range(n_instances):
                for a in algos:
                    tasks.append({"algorithm_id": a, "function": fn, "dimension": d,
                                  "instance": i, "fe_budget": fe_budget_per_d * d})
    return tasks


def test_confirmatory_coco_contract_passes_full_e4():
    from smco.confirmatory import E4_BASELINES, E4_DIMENSIONS, confirmatory_coco_contract
    from smco.experiment_manifests import build_manifest, freeze_manifest
    algos = ["PY-SP-SMCO-EVO", "PY-BASE-SMCO"] + list(E4_BASELINES)
    funcs = [f"f{i}" for i in range(1, 25)]
    manifest = freeze_manifest(build_manifest(
        "e4_bbob_largescale", "bbob-largescale",
        _coco_tasks(algos, funcs, E4_DIMENSIONS, 5)))
    assert confirmatory_coco_contract(
        manifest, expected_algos=algos, expected_dims=E4_DIMENSIONS) == sorted(algos)


def test_confirmatory_coco_contract_rejects_partial_algos():
    from smco.confirmatory import E4_BASELINES, E4_DIMENSIONS, confirmatory_coco_contract
    from smco.experiment_manifests import build_manifest, freeze_manifest
    # only DE, not the full 5 baselines
    algos = ["PY-SP-SMCO-EVO", "PY-BASE-SMCO", "DE"]
    funcs = [f"f{i}" for i in range(1, 25)]
    manifest = freeze_manifest(build_manifest(
        "e4_bbob_largescale", "bbob-largescale",
        _coco_tasks(algos, funcs, E4_DIMENSIONS, 5)))
    expected = set(algos) | set(E4_BASELINES)
    with pytest.raises(ValueError, match="algorithm set"):
        confirmatory_coco_contract(
            manifest, expected_algos=expected, expected_dims=E4_DIMENSIONS)


def test_confirmatory_coco_contract_rejects_partial_functions():
    from smco.confirmatory import E4_BASELINES, E4_DIMENSIONS, confirmatory_coco_contract
    from smco.experiment_manifests import build_manifest, freeze_manifest
    algos = ["PY-SP-SMCO-EVO", "PY-BASE-SMCO"] + list(E4_BASELINES)
    manifest = freeze_manifest(build_manifest(
        "e4_bbob_largescale", "bbob-largescale",
        _coco_tasks(algos, ["f1"], E4_DIMENSIONS, 5)))  # 1 function, not 24
    with pytest.raises(ValueError, match="functions"):
        confirmatory_coco_contract(
            manifest, expected_algos=algos, expected_dims=E4_DIMENSIONS)
