"""Tests for confirmatory-run enforcement (Task 10 / Gate F 强制检查)."""

from __future__ import annotations

import json

import pytest

from smco.confirmatory import (
    baseline_component_errors,
    build_baseline_component_manifest,
    build_comparative_composite,
    build_confirmatory_manifest,
    confirmatory_errors,
    enforce_confirmatory,
    is_run_complete,
    plan_batch,
    validate_composite,
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


def test_build_confirmatory_manifest_rejects_development_instances():
    # A confirmatory (E2) manifest must link confirmatory-stage instances, never the
    # development suite — dev/confirmatory transforms are disjoint (plan §6). This
    # catches the pre-E2 bug where --instances-index pointed at the development index.
    sel = _selection()
    dev_index = {("Rastrigin", 200, 0): {
        "function": "Rastrigin", "dimension": 200, "instance_id": 0,
        "stage": "development",
        "artifact_dir": "instances/development_Rastrigin_d200_i0",
        "transform_sha256": "x", "start_points_hash": "y",
    }}
    with pytest.raises(ValueError, match="confirmatory-stage"):
        build_confirmatory_manifest(
            sel, stage="e2_factorial_highdim", suite="synthetic_highdim",
            functions=["Rastrigin"], dims=[200], n_instances=1,
            fe_budget_per_d=1000, checkpoints_per_d=(1000,),
            instance_index=dev_index,
        )
    # confirmatory-stage index is accepted and its artifact dirs are used verbatim
    conf_entry = dict(dev_index[("Rastrigin", 200, 0)], stage="confirmatory",
                      artifact_dir="instances/confirmatory_Rastrigin_d200_i0")
    manifest = build_confirmatory_manifest(
        sel, stage="e2_factorial_highdim", suite="synthetic_highdim",
        functions=["Rastrigin"], dims=[200], n_instances=1,
        fe_budget_per_d=1000, checkpoints_per_d=(1000,),
        instance_index={("Rastrigin", 200, 0): conf_entry},
    )
    assert manifest["frozen"] is True
    assert all("confirmatory_" in t["instance_artifact_dir"] for t in manifest["tasks"])


def test_build_confirmatory_manifest_rejects_missing_stage():
    # P2: a confirmatory manifest must reject instance indexes whose entries
    # lack a stage (or have an empty stage) — not just "development". A missing
    # stage is ambiguous and must not pass the guard.
    sel = _selection()
    missing_index = {("Rastrigin", 200, 0): {
        "function": "Rastrigin", "dimension": 200, "instance_id": 0,
        "artifact_dir": "instances/confirmatory_Rastrigin_d200_i0",
        "transform_sha256": "x", "start_points_hash": "y",
    }}  # no "stage" key at all
    with pytest.raises(ValueError, match="confirmatory-stage"):
        build_confirmatory_manifest(
            sel, stage="e2_factorial_highdim", suite="synthetic_highdim",
            functions=["Rastrigin"], dims=[200], n_instances=1,
            fe_budget_per_d=1000, checkpoints_per_d=(1000,),
            instance_index=missing_index,
        )
    # an empty-string stage is also rejected
    empty_index = {("Rastrigin", 200, 0): dict(missing_index[("Rastrigin", 200, 0)], stage="")}
    with pytest.raises(ValueError, match="confirmatory-stage"):
        build_confirmatory_manifest(
            sel, stage="e2_factorial_highdim", suite="synthetic_highdim",
            functions=["Rastrigin"], dims=[200], n_instances=1,
            fe_budget_per_d=1000, checkpoints_per_d=(1000,),
            instance_index=empty_index,
        )


# --- P1c: baseline component manifest ---

def test_build_baseline_component_no_winner_base():
    sel = _selection()
    manifest = build_baseline_component_manifest(
        sel, functions=["Rastrigin"], dims=[200], n_instances=1,
        fe_budget_per_d=1000, checkpoints_per_d=(1000,),
    )
    assert manifest["frozen"] is True
    assert manifest["component_role"] == "baseline_extension"
    assert manifest["baseline_algorithms"] == ["DE", "GA", "PSO", "SA", "GenSA"]
    assert "winner_algorithm" not in manifest
    assert "winner_config_hash" not in manifest
    algos = {t.get("algorithm") for t in manifest["tasks"]}
    assert algos == {"DE", "GA", "PSO", "SA", "GenSA"}
    assert len(manifest["tasks"]) == 5  # 1 func × 1 dim × 1 inst × 5 baselines


def test_build_baseline_component_rejects_wrong_stage():
    sel = _selection()
    with pytest.raises(ValueError, match="stage"):
        build_baseline_component_manifest(
            sel, stage="e2_factorial_highdim",
            functions=["Rastrigin"], dims=[200], n_instances=1,
            fe_budget_per_d=1000, checkpoints_per_d=(1000,),
        )


def test_build_baseline_component_rejects_wrong_baselines():
    sel = _selection()
    with pytest.raises(ValueError, match="baselines"):
        build_baseline_component_manifest(
            sel, baselines=("DE", "GA"),
            functions=["Rastrigin"], dims=[200], n_instances=1,
            fe_budget_per_d=1000, checkpoints_per_d=(1000,),
        )


def test_component_gate_f_conditional_skip():
    """P1c constraint 3: a valid baseline_extension component passes Gate-F
    without winner_config_hash; one missing a structural constraint does not."""
    sel = _selection()
    # valid component — no winner_config_hash, no winner in tasks → must pass
    manifest = _build_full_baseline_component(sel_hash=sel["selection_hash"])
    assert confirmatory_errors(manifest, selection=sel) == []
    # tamper: remove selection_hash → no longer a valid component → must fail
    # (treated as ordinary manifest missing winner checks)
    manifest["selection_hash"] = None
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    errors = confirmatory_errors(manifest, selection=sel)
    assert any("selection_hash" in e or "winner" in e for e in errors)


# --- P1c strict: baseline_component_errors (review §4.3) ---
# The canonical E3 baseline component is exactly 5 baselines x 4 functions x
# 3 dims x 5 confirmatory instances = 300 tasks. A component claiming
# component_role="baseline_extension" must satisfy the full 14-check contract;
# a partial/tampered matrix must NOT bypass Gate-F.

_E3_FUNCTIONS = ("Rastrigin", "Ackley", "Griewank", "Zakharov")
_E3_DIMS = (200, 500, 1000)
_E3_BASELINES = ("DE", "GA", "PSO", "SA", "GenSA")


def _confirmatory_instance_index(functions=_E3_FUNCTIONS, dims=_E3_DIMS,
                                 n_instances=5, stage="confirmatory"):
    """A confirmatory-stage instance index mirroring the production artifact."""
    idx = {}
    for fn in functions:
        for d in dims:
            for i in range(n_instances):
                idx[(fn, int(d), i)] = {
                    "function": fn, "dimension": int(d), "instance_id": i,
                    "stage": stage,
                    "artifact_dir": f"instances/{stage}_{fn}_d{int(d)}_i{i}",
                    "transform_sha256": f"th_{fn}_{d}_{i}",
                    "start_points_hash": f"sph_{fn}_{d}_{i}",
                }
    return idx


def _build_full_baseline_component(sel_hash="bcf87965006220a0", stage="confirmatory"):
    return build_baseline_component_manifest(
        _selection(sel_hash=sel_hash),
        functions=list(_E3_FUNCTIONS), dims=list(_E3_DIMS), n_instances=5,
        fe_budget_per_d=1000, checkpoints_per_d=(1000,),
        instance_index=_confirmatory_instance_index(stage=stage),
    )


def test_baseline_component_full_300_passes():
    manifest = _build_full_baseline_component()
    assert len(manifest["tasks"]) == 300
    assert baseline_component_errors(manifest) == []
    # also reachable via confirmatory_errors (Gate-F), with the selection
    assert confirmatory_errors(
        manifest, selection=_selection(sel_hash="bcf87965006220a0")) == []


def test_baseline_component_rejects_wrong_selection_hash():
    manifest = _build_full_baseline_component()
    errors = baseline_component_errors(manifest, selection={"selection_hash": "WRONG"})
    assert any("selection_hash" in e for e in errors)


def test_baseline_component_rejects_missing_task():
    manifest = _build_full_baseline_component()
    manifest["tasks"].pop()  # 299 tasks
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    errors = baseline_component_errors(manifest)
    assert errors and any("300" in e for e in errors)


def test_baseline_component_rejects_duplicated_combo_keeps_300():
    # review §4.3: copy one task, keep total at 300 → duplicate combo must fail.
    import copy
    manifest = _build_full_baseline_component()
    tasks = manifest["tasks"]
    tasks.pop(0)                       # drop one combo (299)
    tasks.append(copy.deepcopy(tasks[0]))  # duplicate another (300 again)
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    errors = baseline_component_errors(manifest)
    assert errors  # duplicate combo / distinct run_id count != 300


def test_baseline_component_rejects_small_grid():
    # a 5-task (1 func x 1 dim x 1 inst x 5 baselines) component is NOT canonical
    manifest = build_baseline_component_manifest(
        _selection(), functions=["Rastrigin"], dims=[200], n_instances=1,
        fe_budget_per_d=1000, checkpoints_per_d=(1000,),
        instance_index=_confirmatory_instance_index(
            functions=["Rastrigin"], dims=[200], n_instances=1),
    )
    errors = baseline_component_errors(manifest)
    assert errors and any("300" in e for e in errors)


def test_baseline_component_rejects_wrong_function():
    manifest = _build_full_baseline_component()
    manifest["tasks"][0]["function"] = "Sphere"  # not in the 4-function set
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    errors = baseline_component_errors(manifest)
    assert any("function" in e.lower() for e in errors)


def test_baseline_component_rejects_wrong_dimension():
    manifest = _build_full_baseline_component()
    manifest["tasks"][0]["dimension"] = 777  # not in {200,500,1000}
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    errors = baseline_component_errors(manifest)
    assert any("dimension" in e.lower() for e in errors)


def test_baseline_component_rejects_wrong_instance():
    manifest = _build_full_baseline_component()
    manifest["tasks"][0]["instance"] = 9  # not in {0,1,2,3,4}
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    errors = baseline_component_errors(manifest)
    assert any("instance" in e.lower() for e in errors)


def test_baseline_component_rejects_wrong_baseline():
    manifest = _build_full_baseline_component()
    manifest["tasks"][0]["algorithm"] = "CMA-ES"  # not one of the 5 baselines
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    errors = baseline_component_errors(manifest)
    assert any("algorithm" in e.lower() for e in errors)


def test_baseline_component_rejects_development_instances():
    # a component built from a development instance index → artifact dirs are
    # development_*, not confirmatory_* → must fail (review §4.2 check 12).
    manifest = _build_full_baseline_component(stage="development")
    errors = baseline_component_errors(manifest)
    assert any("confirmatory" in e for e in errors)


def test_baseline_component_rejects_development_index_when_passed():
    # review §4.2 check 13: when an instance_index is supplied, every entry's
    # stage must be 'confirmatory'.
    manifest = _build_full_baseline_component(stage="confirmatory")
    dev_index = _confirmatory_instance_index(stage="development")
    errors = baseline_component_errors(manifest, instance_index=dev_index)
    assert any("confirmatory" in e for e in errors)


def test_baseline_component_rejects_winner_base_algorithm():
    # review §4.2 check 14: a winner/base SMCO algorithm must not appear.
    manifest = _build_full_baseline_component()
    manifest["tasks"][0]["algorithm"] = "PY-SP-SMCO-EVO"
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    errors = baseline_component_errors(manifest)
    assert any("winner" in e or "base" in e.lower() for e in errors)


def test_baseline_component_role_alone_cannot_bypass_gate_f():
    # review §3.2 / §4.3: component_role + metadata present, but the task matrix
    # is empty/bogus → confirmatory_errors must still fail (cannot bypass Gate-F).
    manifest = freeze_manifest(
        build_manifest("e3_companion_baselines", "synthetic_highdim", []))
    manifest["component_role"] = "baseline_extension"
    manifest["baseline_algorithms"] = list(_E3_BASELINES)
    manifest["selection_hash"] = "s1"
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    errors = confirmatory_errors(manifest, selection={"selection_hash": "s1"})
    assert errors  # empty task matrix rejected — Gate-F not bypassed


# --- P1c strict: comparative composite (review §5) ---
# The formal composite is exactly 120 (E2 winner+base) + 300 (baselines) = 420.
# build_comparative_composite verifies both sources before freezing; the formal
# validator only accepts 120+300 and recomputes every hash from disk.

_E3_EXPECTED_ALGOS = {"PY-SP-SMCO-EVO", "PY-BASE-SMCO",
                      "DE", "GA", "PSO", "SA", "GenSA"}


def _build_full_e2(sel_hash="bcf87965006220a0"):
    sel = {"winner": "PY-SP-SMCO-EVO", "winner_language": "python",
           "selection_hash": sel_hash}
    return build_confirmatory_manifest(
        sel, stage="e2_factorial_highdim", suite="synthetic_highdim",
        functions=list(_E3_FUNCTIONS), dims=list(_E3_DIMS), n_instances=5,
        fe_budget_per_d=1000, checkpoints_per_d=(1000,),
        instance_index=_confirmatory_instance_index(),
    )


def _write_component_merged(tmp, manifest, *, n_checks=12, provenance_passed=True,
                            drop_run_id=None, tamper_algo=None, extra_run_id=None):
    """Write a minimal merged/ dir (valid_runs.csv + provenance_audit.json) whose
    run-ids match ``manifest`` tasks. Only the columns the composite validator
    inspects (run_id/algorithm_id/stage) are written."""
    import csv as _csv
    d = tmp / "merged"; d.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in manifest["tasks"]:
        rid = t["run_id"]
        if drop_run_id is not None and rid == drop_run_id:
            continue
        algo = t.get("algorithm_id") or t.get("algorithm")
        if tamper_algo is not None and rid == tamper_algo[0]:
            algo = tamper_algo[1]
        rows.append({"run_id": rid, "algorithm_id": algo, "stage": t.get("stage")})
    if extra_run_id is not None:
        rows.append({"run_id": extra_run_id, "algorithm_id": "DE",
                     "stage": "e3_companion_baselines"})
    with open(d / "valid_runs.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["run_id", "algorithm_id", "stage"])
        w.writeheader(); w.writerows(rows)
    checks = [{"name": f"check_{i}", "passed": True, "n": len(rows), "errors": []}
              for i in range(max(0, n_checks))]
    if n_checks >= 12:
        # the 12th check is provenance_complete (the audit key the composite gate
        # requires); without it the audit is the old 11-check version.
        checks[-1] = {"name": "provenance_complete", "passed": provenance_passed,
                      "n": len(rows), "errors": []}
    audit = {"passed": all(c["passed"] for c in checks), "failed_checks": [],
             "checks": checks, "n_rows": len(rows)}
    (d / "provenance_audit.json").write_text(json.dumps(audit))
    return d


def _component_dir(tmp_path, name):
    return tmp_path / name / "merged"


def _build_valid_composite(tmp_path, *, e2_sel="bcf87965006220a0",
                           bc_sel="bcf87965006220a0"):
    e2 = _build_full_e2(sel_hash=e2_sel)
    bc = _build_full_baseline_component(sel_hash=bc_sel)
    e2_mp = tmp_path / "e2m.json"; e2_mp.write_text(json.dumps(e2))
    bc_mp = tmp_path / "bcm.json"; bc_mp.write_text(json.dumps(bc))
    e2_dir = _write_component_merged(tmp_path / "e2", e2)
    bc_dir = _write_component_merged(tmp_path / "bc", bc)
    comp = build_comparative_composite(
        e2_manifest_path=str(e2_mp), e2_merged_dir=str(e2_dir),
        baseline_component_path=str(bc_mp), baseline_merged_dir=str(bc_dir))
    return comp, e2_mp, bc_mp, e2_dir, bc_dir, e2, bc


def test_composite_full_420_passes_and_revalidates(tmp_path):
    comp, *_ = _build_valid_composite(tmp_path)
    assert comp["schema_version"] == "1"
    assert comp["composite_type"] == "comparative_composite"
    assert comp["stage"] == "e3_comparative_analysis"
    assert comp["suite"] == "synthetic_highdim"
    assert comp["frozen"] is True
    assert comp["total_runs"] == 420
    assert set(comp["algorithms"]) == _E3_EXPECTED_ALGOS
    assert comp["components"]["winner_base"]["n_runs"] == 120
    assert comp["components"]["baseline_extension"]["n_runs"] == 300
    # validator recomputes every hash from the recorded paths and passes
    assert validate_composite(comp) == []


def test_composite_rejects_tampered_composite_hash(tmp_path):
    comp, *_ = _build_valid_composite(tmp_path)
    comp["composite_sha256"] = "0" * 64  # do NOT recompute -> mismatch
    errors = validate_composite(comp)
    assert any("composite_sha256" in e for e in errors)


def test_composite_rejects_frozen_false(tmp_path):
    from smco.confirmatory import composite_sha256
    comp, *_ = _build_valid_composite(tmp_path)
    comp["frozen"] = False
    comp["composite_sha256"] = composite_sha256(comp)  # keep hash consistent
    errors = validate_composite(comp)
    assert any("frozen" in e.lower() for e in errors)


def test_composite_rejects_wrong_schema_type_stage_suite(tmp_path):
    from smco.confirmatory import composite_sha256
    comp, *_ = _build_valid_composite(tmp_path)
    for bad in ({"schema_version": "9"}, {"composite_type": "other"},
                {"stage": "e9"}, {"suite": "other"}):
        c = json.loads(json.dumps(comp))
        c.update(bad)
        c["composite_sha256"] = composite_sha256(c)
        assert validate_composite(c), f"expected failure for {bad}"


def test_composite_rejects_modified_e2_manifest(tmp_path):
    # regenerate the E2 manifest on disk (new hash) after the composite froze
    comp, e2_mp, *_ = _build_valid_composite(tmp_path)
    e2 = json.loads(e2_mp.read_text())
    e2["tasks"][0]["dimension"] = 777
    e2["manifest_sha256"] = manifest_sha256(e2)
    e2_mp.write_text(json.dumps(e2))
    errors = validate_composite(comp)
    assert any("manifest_sha256" in e or "E2" in e for e in errors)


def test_composite_rejects_wrong_component_n_runs(tmp_path):
    from smco.confirmatory import composite_sha256
    for bad in (1, 119, 299, 301):
        comp, *_ = _build_valid_composite(tmp_path)
        comp["components"]["baseline_extension"]["n_runs"] = bad
        comp["composite_sha256"] = composite_sha256(comp)
        errors = validate_composite(comp)
        assert any("300" in e or "n_runs" in e for e in errors), bad


def test_composite_rejects_wrong_total_runs(tmp_path):
    from smco.confirmatory import composite_sha256
    for bad in (419, 421):
        comp, *_ = _build_valid_composite(tmp_path)
        comp["total_runs"] = bad
        comp["composite_sha256"] = composite_sha256(comp)
        errors = validate_composite(comp)
        assert any("420" in e or "total" in e.lower() for e in errors), bad


def test_composite_rejects_csv_with_non_manifest_run_id(tmp_path):
    # same row count, but swap one row's algorithm to one not matching manifest
    comp, _, _, _, bc_dir, _, bc = _build_valid_composite(tmp_path)
    _write_component_merged(bc_dir.parent, bc,
                            tamper_algo=(bc["tasks"][0]["run_id"], "PSO"))
    errors = validate_composite(comp)
    assert errors  # valid_runs hash + run_id-set hash + identity mismatch


def test_composite_rejects_csv_extra_run_id(tmp_path):
    comp, _, _, _, bc_dir, _, bc = _build_valid_composite(tmp_path)
    _write_component_merged(bc_dir.parent, bc, extra_run_id="bBOGUS00000000000")
    errors = validate_composite(comp)
    assert any("run_id" in e.lower() or "hash mismatch" in e for e in errors)


def test_composite_builder_rejects_old_11check_audit(tmp_path):
    e2 = _build_full_e2(); bc = _build_full_baseline_component()
    e2_mp = tmp_path / "e2m.json"; e2_mp.write_text(json.dumps(e2))
    bc_mp = tmp_path / "bcm.json"; bc_mp.write_text(json.dumps(bc))
    _write_component_merged(tmp_path / "e2", e2)
    _write_component_merged(tmp_path / "bc", bc, n_checks=11)  # old audit
    with pytest.raises(ValueError, match="provenance|audit|11"):
        build_comparative_composite(
            e2_manifest_path=str(e2_mp), e2_merged_dir=str(_component_dir(tmp_path, "e2")),
            baseline_component_path=str(bc_mp), baseline_merged_dir=str(_component_dir(tmp_path, "bc")))


def test_composite_builder_rejects_failed_provenance(tmp_path):
    e2 = _build_full_e2(); bc = _build_full_baseline_component()
    e2_mp = tmp_path / "e2m.json"; e2_mp.write_text(json.dumps(e2))
    bc_mp = tmp_path / "bcm.json"; bc_mp.write_text(json.dumps(bc))
    _write_component_merged(tmp_path / "e2", e2)
    _write_component_merged(tmp_path / "bc", bc, provenance_passed=False)
    with pytest.raises(ValueError, match="provenance|audit"):
        build_comparative_composite(
            e2_manifest_path=str(e2_mp), e2_merged_dir=str(_component_dir(tmp_path, "e2")),
            baseline_component_path=str(bc_mp), baseline_merged_dir=str(_component_dir(tmp_path, "bc")))


def test_composite_builder_rejects_run_id_overlap(tmp_path):
    e2 = _build_full_e2(); bc = _build_full_baseline_component()
    e2_mp = tmp_path / "e2m.json"; e2_mp.write_text(json.dumps(e2))
    bc_mp = tmp_path / "bcm.json"; bc_mp.write_text(json.dumps(bc))
    _write_component_merged(tmp_path / "e2", e2)
    # inject an E2 run_id into the baseline CSV -> overlap
    _write_component_merged(tmp_path / "bc", bc, extra_run_id=e2["tasks"][0]["run_id"])
    with pytest.raises(ValueError, match="overlap"):
        build_comparative_composite(
            e2_manifest_path=str(e2_mp), e2_merged_dir=str(_component_dir(tmp_path, "e2")),
            baseline_component_path=str(bc_mp), baseline_merged_dir=str(_component_dir(tmp_path, "bc")))


def test_composite_builder_rejects_selection_hash_mismatch(tmp_path):
    # E2 and baseline carry different selection_hash -> builder rejects
    e2 = _build_full_e2(sel_hash="HASH_A")
    bc = _build_full_baseline_component(sel_hash="HASH_B")
    e2_mp = tmp_path / "e2m.json"; e2_mp.write_text(json.dumps(e2))
    bc_mp = tmp_path / "bcm.json"; bc_mp.write_text(json.dumps(bc))
    _write_component_merged(tmp_path / "e2", e2)
    _write_component_merged(tmp_path / "bc", bc)
    with pytest.raises(ValueError, match="selection_hash"):
        build_comparative_composite(
            e2_manifest_path=str(e2_mp), e2_merged_dir=str(_component_dir(tmp_path, "e2")),
            baseline_component_path=str(bc_mp), baseline_merged_dir=str(_component_dir(tmp_path, "bc")))


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


def test_confirmatory_run_matrix_rejects_wrong_budget():
    # R7c: the FE-budget-per-d is locked to the plan (E4 = 1000*d, E5 = 2000*d).
    from smco.confirmatory import confirmatory_run_matrix
    manifest = _build_e4_manifest()  # fe_budget_per_d = 1000
    with pytest.raises(ValueError, match="fe_budget_per_d"):
        confirmatory_run_matrix(
            manifest, expected_stage="e4_bbob_largescale", expected_suite="bbob-largescale",
            expected_fe_budget_per_d=999)


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


def test_confirmatory_coco_contract_rejects_duplicate_combo():
    # R8c reviewer repro: 2520 rows but only 2519 unique (algorithm,function,dim,
    # instance) combos (drop one, duplicate another) must not pass the validator.
    import copy
    from smco.confirmatory import E4_BASELINES, E4_DIMENSIONS, confirmatory_coco_contract
    from smco.experiment_manifests import build_manifest, freeze_manifest, manifest_sha256
    algos = ["PY-SP-SMCO-EVO", "PY-BASE-SMCO"] + list(E4_BASELINES)
    funcs = [f"f{i}" for i in range(1, 25)]
    tasks = _coco_tasks(algos, funcs, E4_DIMENSIONS, 5)
    assert len(tasks) == 2520
    # drop tasks[0] (its combo goes missing) and append a copy of tasks[1] (dup)
    tasks = tasks[1:]
    tasks.append(copy.deepcopy(tasks[0]))
    assert len(tasks) == 2520  # total still matches, but one combo is duplicated
    manifest = build_manifest("e4_bbob_largescale", "bbob-largescale", tasks)
    manifest["manifest_sha256"] = manifest_sha256(manifest)  # keep hash consistent
    with pytest.raises(ValueError, match="duplicate"):
        confirmatory_coco_contract(
            manifest, expected_algos=algos, expected_dims=E4_DIMENSIONS)
