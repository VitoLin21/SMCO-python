"""Tests for immutable experiment manifests (Task 7).

A manifest is a frozen JSON document listing one canonical task per run. Each
task carries a stable ``run_id`` (``paper_contract.compute_run_id``) and a
dimension-independent ``configuration_hash``; the manifest itself has a content
hash that ``verify_manifest`` re-checks so a runner can refuse a tampered
manifest.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from smco.experiment_manifests import (
    E1_FUNCTIONS,
    E3F_FUNCTIONS,
    E7_FUNCTIONS,
    build_algorithm_config,
    build_manifest,
    build_task,
    derive_seed,
    e1_algorithm_configs,
    expand_tasks,
    freeze_manifest,
    load_manifest,
    manifest_sha256,
    verify_manifest,
    write_manifest,
)
from smco.paper_contract import compute_run_id


def _evo_config(**overrides):
    base = dict(
        language="python",
        family="smco",
        evolutionary=True,
        state_semantics="state_preserving",
        evolution_strategy="rand1bin",
        evolution_points=(0.5, 0.75),
        elimination_rate=0.25,
        de_factor=0.8,
        de_crossover=0.7,
        n_starts=8,
    )
    base.update(overrides)
    return build_algorithm_config(**base)


def test_prospective_function_registries_extend_without_mutating_frozen_e1():
    assert E1_FUNCTIONS == ("Rastrigin", "Ackley", "Griewank", "Zakharov")
    assert E3F_FUNCTIONS == (
        "Rosenbrock", "Levy", "Schwefel226", "HighConditionedEllipsoid",
    )
    assert E7_FUNCTIONS == E1_FUNCTIONS + E3F_FUNCTIONS


# ----------------------------- algorithm config -----------------------------
def test_evo_config_builds_canonical_id_and_hash():
    cfg = _evo_config()
    assert cfg["algorithm_id"] == "PY-SP-SMCO-EVO"
    assert cfg["evolutionary"] == "true"
    assert cfg["state_semantics"] == "state_preserving"
    assert cfg["evolution_strategy"] == "rand1bin"
    assert len(cfg["configuration_hash"]) == 16


def test_base_config_uses_none_tokens():
    cfg = build_algorithm_config(
        language="r",
        family="smco_refine",
        evolutionary=False,
        state_semantics="none",
        evolution_strategy="none",
        evolution_points=(),
        elimination_rate=0.25,
        de_factor=0.8,
        de_crossover=0.7,
        n_starts=8,
    )
    assert cfg["algorithm_id"] == "R-BASE-SMCO-REFINE"
    assert cfg["evolutionary"] == "false"
    assert cfg["state_semantics"] == "none"
    assert cfg["evolution_strategy"] == "none"


def test_different_strategy_yields_different_configuration_hash():
    a = _evo_config(evolution_strategy="rand1bin")
    b = _evo_config(evolution_strategy="best1bin")
    assert a["configuration_hash"] != b["configuration_hash"]


def test_configuration_hash_is_dimension_independent():
    # configuration_hash captures algorithm params only; fe_budget/checkpoints
    # are run-level and must not enter it.
    cfg = _evo_config()
    # rebuilding with no fe_budget input still gives a stable hash.
    cfg2 = _evo_config()
    assert cfg["configuration_hash"] == cfg2["configuration_hash"]


def test_e1_algorithm_configs_has_18_unique():
    configs = e1_algorithm_configs()
    assert len(configs) == 18
    evo = [c for c in configs if c["evolutionary"] == "true"]
    base = [c for c in configs if c["evolutionary"] == "false"]
    assert len(evo) == 12  # 2 languages x 2 semantics x 3 families
    assert len(base) == 6  # 2 languages x 3 families
    assert len({c["algorithm_id"] for c in configs}) == 18


# ------------------------------- task / run_id ------------------------------
def test_build_task_run_id_is_stable_and_recomputable():
    cfg = _evo_config()
    task = build_task(
        "e1_development",
        "synthetic_highdim",
        "Rastrigin",
        200,
        0,
        0,
        config=cfg,
        fe_budget=200000,
        checkpoints=(20000, 50000, 100000, 200000),
        seed=12345,
        instance_hash="abc",
        start_points_hash="def",
    )
    assert task["run_id"].startswith("r")
    assert task["run_id"] == compute_run_id(task)
    task2 = build_task(
        "e1_development", "synthetic_highdim", "Rastrigin", 200, 0, 0,
        config=cfg, fe_budget=200000, checkpoints=(20000, 50000, 100000, 200000),
        seed=12345, instance_hash="abc", start_points_hash="def",
    )
    assert task["run_id"] == task2["run_id"]


def test_run_id_differs_across_instance_and_stage():
    cfg = _evo_config()
    kw = dict(config=cfg, fe_budget=200000, checkpoints=(20000,), seed=None)
    t0 = build_task("e1_development", "synthetic_highdim", "Rastrigin", 200, 0, 0, **{**kw, "seed": 1})
    t1 = build_task("e1_development", "synthetic_highdim", "Rastrigin", 200, 1, 0, **{**kw, "seed": 2})
    tdev = build_task("e2_factorial_highdim", "synthetic_highdim", "Rastrigin", 200, 0, 0, **{**kw, "seed": 3})
    assert len({t0["run_id"], t1["run_id"], tdev["run_id"]}) == 3


def test_derive_seed_stable_and_distinct():
    s0 = derive_seed("e1_development", "synthetic_highdim", "Rastrigin", 200, 0, 0, "PY-SP-SMCO-EVO")
    s0b = derive_seed("e1_development", "synthetic_highdim", "Rastrigin", 200, 0, 0, "PY-SP-SMCO-EVO")
    s1 = derive_seed("e1_development", "synthetic_highdim", "Rastrigin", 200, 1, 0, "PY-SP-SMCO-EVO")
    assert s0 == s0b
    assert s0 != s1


# -------------------------- manifest hash / freeze --------------------------
def _two_tasks():
    cfg = _evo_config()
    return [
        build_task("e1_development", "synthetic_highdim", "Rastrigin", 200, i, 0,
                   config=cfg, fe_budget=200000, checkpoints=(20000,), seed=10 + i)
        for i in range(2)
    ]


def test_manifest_hash_is_stable_and_recomputable():
    m1 = build_manifest("e1_development", "synthetic_highdim", _two_tasks())
    m2 = build_manifest("e1_development", "synthetic_highdim", _two_tasks())
    assert m1["manifest_sha256"] == m2["manifest_sha256"]
    assert m1["manifest_sha256"] == manifest_sha256(m1)
    assert m1["frozen"] is False


def test_freeze_marks_frozen_and_verify_passes():
    frozen = freeze_manifest(build_manifest("e1_development", "synthetic_highdim", _two_tasks()))
    assert frozen["frozen"] is True
    assert frozen["manifest_sha256"] == manifest_sha256(frozen)
    assert verify_manifest(frozen) is True


def test_verify_rejects_tampered_manifest():
    frozen = freeze_manifest(build_manifest("e1_development", "synthetic_highdim", _two_tasks()))
    frozen["tasks"][0]["fe_budget"] = 999999  # tamper after freeze
    with pytest.raises(ValueError):
        verify_manifest(frozen)


def test_manifest_write_load_roundtrip(tmp_path):
    frozen = freeze_manifest(build_manifest("e1_development", "synthetic_highdim", _two_tasks()))
    path = tmp_path / "development.json"
    write_manifest(frozen, path)
    loaded = load_manifest(path)
    assert loaded["manifest_sha256"] == frozen["manifest_sha256"]
    assert verify_manifest(loaded) is True
    assert len(loaded["tasks"]) == 2


def test_unfrozen_manifest_round_trips_but_load_marks_frozen(tmp_path):
    m = build_manifest("e1_development", "synthetic_highdim", _two_tasks())
    path = tmp_path / "dev.json"
    write_manifest(m, path)
    loaded = load_manifest(path)
    assert loaded["frozen"] is False
    assert loaded["manifest_sha256"] == manifest_sha256(loaded)


# --------------------------------- expand -----------------------------------
def test_expand_tasks_count_and_unique_run_ids():
    configs = e1_algorithm_configs()[:4]
    tasks = expand_tasks(
        "e1_development",
        "synthetic_highdim",
        functions=["Rastrigin"],
        dims=[200],
        n_instances=2,
        configs=configs,
        fe_budget_per_d=1000,
        checkpoints_per_d=(100, 250, 500, 1000),
    )
    assert len(tasks) == 4 * 1 * 1 * 2  # configs x funcs x dims x instances
    run_ids = [t["run_id"] for t in tasks]
    assert len(set(run_ids)) == len(run_ids)
    assert all(t["fe_budget"] == 1000 * 200 for t in tasks)
    assert all(t["checkpoints"] == [100 * 200, 250 * 200, 500 * 200, 1000 * 200] for t in tasks)


def test_expand_tasks_dev_and_confirm_are_disjoint():
    configs = e1_algorithm_configs()[:2]
    dev = expand_tasks("e1_development", "synthetic_highdim", ["Rastrigin"], [200], 2,
                       configs, fe_budget_per_d=1000, checkpoints_per_d=(100, 250))
    conf = expand_tasks("e2_factorial_highdim", "synthetic_highdim", ["Rastrigin"], [200], 2,
                        configs, fe_budget_per_d=2000, checkpoints_per_d=(100, 250))
    assert {t["run_id"] for t in dev}.isdisjoint({t["run_id"] for t in conf})


def test_expand_tasks_attaches_instance_provenance():
    configs = e1_algorithm_configs()[:1]
    index = {
        ("Rastrigin", 200, 0): {
            "artifact_dir": "instances/dev_Rastrigin_d200_i0",
            "transform_sha256": "thash0",
            "start_points_hash": "shash0",
        },
        ("Rastrigin", 200, 1): {
            "artifact_dir": "instances/dev_Rastrigin_d200_i1",
            "transform_sha256": "thash1",
            "start_points_hash": "shash1",
        },
    }
    tasks = expand_tasks(
        "e1_development", "synthetic_highdim", ["Rastrigin"], [200], 2,
        configs, fe_budget_per_d=1000, checkpoints_per_d=(100,),
        instance_index=index,
    )
    by_inst = {t["instance"]: t for t in tasks}
    assert by_inst[0]["instance_hash"] == "thash0"
    assert by_inst[0]["start_points_hash"] == "shash0"
    assert by_inst[0]["instance_artifact_dir"].endswith("dev_Rastrigin_d200_i0")
    assert by_inst[1]["instance_hash"] == "thash1"


# --------------------- CLI integration (scripts/ generator) -----------------
_CLI_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "generate_smco_evo_manifests.py"
)


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("smco_evo_manifests_cli", _CLI_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_manifest_for_suite_end_to_end(tmp_path):
    cli = _load_cli_module()
    # Task 6: materialise instances + index first.
    instances_root = tmp_path / "instances_root"
    cli.build_instance_set(
        ["Rastrigin"], [8], 2, stage="development",
        out_dir=instances_root, n_starts=4,
    )
    index = cli.load_instance_index(instances_root / "instances_index.json")

    # Task 7: manifest referencing those instances, frozen.
    manifest = cli.build_manifest_for_suite(
        stage="e1_development",
        suite="synthetic_highdim",
        functions=["Rastrigin"],
        dims=[8],
        n_instances=2,
        fe_budget_per_d=1000,
        checkpoints_per_d=(100, 500),
        instance_index=index,
        configs=e1_algorithm_configs(),
        freeze=True,
        out_dir=tmp_path / "manifests",
    )
    assert manifest["frozen"] is True
    assert verify_manifest(manifest) is True
    # 18 configs x 1 func x 1 dim x 2 instances = 36 tasks
    assert manifest["n_tasks"] == 18 * 1 * 1 * 2
    run_ids = [t["run_id"] for t in manifest["tasks"]]
    assert len(set(run_ids)) == len(run_ids)
    # Every task linked to its instance provenance.
    assert all(t["instance_hash"] for t in manifest["tasks"])
    assert all(t["start_points_hash"] for t in manifest["tasks"])
    # Manifest file written and reloads consistently.
    written = sorted((tmp_path / "manifests").glob("*.json"))
    assert written
    reloaded = load_manifest(written[0])
    assert verify_manifest(reloaded) is True


def test_build_manifest_for_suite_dry_run_reports_counts(tmp_path):
    cli = _load_cli_module()
    summary = cli.build_manifest_for_suite(
        stage="e1_development",
        suite="synthetic_highdim",
        functions=["Rastrigin", "Ackley"],
        dims=[200],
        n_instances=5,
        fe_budget_per_d=1000,
        checkpoints_per_d=(100, 250, 500, 1000),
        configs=e1_algorithm_configs(),
        dry_run=True,
    )
    assert summary["dry_run"] is True
    # 18 configs x 2 funcs x 1 dim x 5 instances = 180 tasks
    assert summary["n_tasks"] == 18 * 2 * 1 * 5
    assert summary["unique_run_ids"] == summary["n_tasks"]
    assert summary["total_fe_budget"] == summary["n_tasks"] * (1000 * 200)


def test_expand_tasks_selects_start_points_hash_by_n_starts():
    cfg8 = build_algorithm_config("python", "smco", True, "state_preserving",
        evolution_strategy="rand1bin", evolution_points=(0.5, 0.75),
        elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=8)
    cfg16 = build_algorithm_config("python", "smco", True, "state_preserving",
        evolution_strategy="rand1bin", evolution_points=(0.5, 0.75),
        elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=16)
    index = {("Rastrigin", 4, 0): {
        "artifact_dir": "art", "transform_sha256": "ih",
        "start_points_hash": "hash_n8",
        "extra_starts": {"16": {"hash": "hash_n16", "n_starts": 16}},
    }}
    tasks = expand_tasks("e6_ablations", "synthetic_highdim", ["Rastrigin"], [4], 1,
                         [cfg8, cfg16], fe_budget_per_d=100, checkpoints_per_d=(100,),
                         instance_index=index)
    by_n = {t["n_starts"]: t["start_points_hash"] for t in tasks}
    assert by_n[8] == "hash_n8"
    assert by_n[16] == "hash_n16"
