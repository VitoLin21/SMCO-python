"""Confirmatory-run enforcement (Task 10 / Gate F 强制检查).

A confirmatory runner (E2/E3/E4/E5) must refuse to start unless its manifest is
frozen and content-hash-consistent, and (when a selection is provided) the
selected winner is actually one of the manifest's algorithm_ids. The runner must
only execute tasks listed in the manifest, and report completed/missing counts.

``is_run_complete`` / ``plan_batch`` live here as the single source of truth
(batch runners import them).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .experiment_manifests import (
    DEFAULT_DE_CROSSOVER,
    DEFAULT_DE_FACTOR,
    DEFAULT_ELIMINATION_RATE,
    DEFAULT_EVO_POINTS,
    DEFAULT_EVO_STRATEGY,
    DEFAULT_N_STARTS,
    build_algorithm_config,
    build_manifest,
    expand_baseline_tasks,
    expand_tasks,
    freeze_manifest,
    manifest_sha256,
)
from .paper_contract import NONE_TOKEN, parse_algorithm_id

# COCO confirmatory matrix (plan E4/E5). E4 = winner + matched base + 5 strong
# baselines over 24 bbob-largescale functions x {160,320,640} x 5 instances =
# 7 x 24 x 3 x 5 = 2520 runs. E5 = winner + matched base over 24 bbob functions
# x {5,20} x 5 instances = 2 x 24 x 2 x 5 = 480 runs.
E4_BASELINES = ("DE", "GA", "PSO", "SA", "GenSA")
E4_DIMENSIONS = (160, 320, 640)
E5_DIMENSIONS = (5, 20)
COCO_N_FUNCTIONS = 24
COCO_N_INSTANCES = 5


def is_run_complete(result_dir, run_id) -> bool:
    """A run is complete only on status=success; infra/timeout must retry."""
    path = Path(result_dir) / f"{run_id}.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    return payload.get("status") == "success"


def plan_batch(tasks, result_dir) -> dict:
    result_dir = Path(result_dir)
    completed = sum(1 for t in tasks if is_run_complete(result_dir, t["run_id"]))
    return {
        "dry_run": False,
        "n_tasks": len(tasks),
        "completed": completed,
        "missing": len(tasks) - completed,
        "total_fe_budget": sum(int(t["fe_budget"]) for t in tasks),
    }


def build_confirmatory_manifest(
    selection: dict,
    *,
    stage: str,
    suite: str,
    functions,
    dims,
    n_instances: int,
    fe_budget_per_d: int,
    checkpoints_per_d,
    baselines=(),
    instance_index: dict | None = None,
    manifest_id: str | None = None,
) -> dict:
    """Build a frozen confirmatory manifest (E2/E3) driven by a selection.

    Tasks: the frozen winner + its matched non-EVO base (same language/family),
    plus optional comparison baselines (E3). The manifest carries selection_hash,
    winner_config_hash, matched_base_config_hash and the allowed algorithm set so
    :func:`confirmatory_errors` can reject any task/result outside this closure.
    """
    if instance_index:
        # A confirmatory manifest (E2/E3/E6 synthetic) must link confirmatory-stage
        # instances — never the development suite. dev/confirmatory transforms are
        # disjoint by design (plan §6); silently reusing development instances would
        # invalidate confirmatory inference. load_instance_index keys by
        # (function, dim, iid) and carries each entry's ``stage``.
        # P2: require stage == "confirmatory" explicitly — reject development,
        # missing/empty stage, and any other namespace (not just "development").
        inst_stages = {e.get("stage") for e in instance_index.values()}
        if inst_stages != {"confirmatory"}:
            raise ValueError(
                f"confirmatory manifest (stage {stage!r}) requires confirmatory-stage "
                f"instances (every entry stage=='confirmatory'), but the index has "
                f"stages {sorted(repr(s) for s in inst_stages)}; development and "
                f"confirmatory suites must use disjoint instances (generate with "
                f"--suite-stage confirmatory and link that index)"
            )
    winner = selection["winner"]
    language = selection.get("winner_language") or "python"
    parsed = parse_algorithm_id(winner)
    family = parsed["family"]
    evo = parsed["evolutionary"]
    sem = parsed["state_semantics"] if evo else NONE_TOKEN
    evo_args = dict(
        evolution_strategy=DEFAULT_EVO_STRATEGY, evolution_points=DEFAULT_EVO_POINTS,
        elimination_rate=DEFAULT_ELIMINATION_RATE, de_factor=DEFAULT_DE_FACTOR,
        de_crossover=DEFAULT_DE_CROSSOVER, n_starts=DEFAULT_N_STARTS,
    )
    winner_cfg = build_algorithm_config(language, family, evo, sem, **evo_args)
    base_cfg = build_algorithm_config(language, family, False, NONE_TOKEN, **evo_args)
    tasks = list(expand_tasks(
        stage, suite, functions, dims, n_instances, [winner_cfg, base_cfg],
        fe_budget_per_d=fe_budget_per_d, checkpoints_per_d=checkpoints_per_d,
        instance_index=instance_index,
    ))
    allowed = [winner_cfg["algorithm_id"], base_cfg["algorithm_id"]]
    if baselines:
        tasks.extend(expand_baseline_tasks(
            stage, suite, functions, dims, n_instances, baselines,
            fe_budget_per_d=fe_budget_per_d, checkpoints_per_d=checkpoints_per_d,
            instance_index=instance_index,
        ))
        allowed.extend(baselines)
    manifest = build_manifest(stage, suite, tasks, manifest_id=manifest_id)
    manifest["selection_hash"] = selection.get("selection_hash")
    manifest["winner_config_hash"] = winner_cfg["configuration_hash"]
    manifest["matched_base_config_hash"] = base_cfg["configuration_hash"]
    manifest["allowed_algorithms"] = allowed
    manifest["winner_algorithm"] = winner_cfg["algorithm_id"]
    manifest["matched_base_algorithm"] = base_cfg["algorithm_id"]
    manifest["baseline_algorithms"] = list(baselines)
    manifest["winner_language"] = language
    return freeze_manifest(manifest)


def _is_baseline_extension(manifest: dict) -> bool:
    """P1c constraint 3: True only when ALL structural conditions for a
    baseline_extension component are met. Missing any → treated as ordinary
    confirmatory manifest (no Gate-F bypass)."""
    return (
        manifest.get("component_role") == "baseline_extension"
        and manifest.get("stage") == _BASELINE_EXTENSION_STAGE
        and manifest.get("suite") == _BASELINE_EXTENSION_SUITE
        and tuple(manifest.get("baseline_algorithms", ())) == _BASELINE_EXTENSION_BASELINES
        and manifest.get("selection_hash") is not None
    )


def baseline_component_errors(
    manifest: dict, *, selection: dict | None = None, instance_index: dict | None = None,
) -> list[str]:
    """Strict E3 baseline-component contract (review §4.2).

    Returns Gate-F violations for a frozen ``baseline_extension`` component.
    An empty list means the manifest is the exact canonical E3 baseline
    component: the 5 baselines x {Rastrigin, Ackley, Griewank, Zakharov} x
    {200, 500, 1000} x 5 confirmatory instances = 300 tasks, each
    (algorithm, function, dimension, instance) exactly once, no winner/base
    algorithm, every instance confirmatory-stage.

    Called by :func:`confirmatory_errors` for any manifest that declares
    ``component_role="baseline_extension"`` (so ``component_role`` alone cannot
    bypass Gate-F — the full task matrix must match). Also reusable directly by
    the composite builder (review §5.3) and the generation CLI (review §6.2).
    """
    errors: list[str] = []
    expected_baselines = set(_BASELINE_EXTENSION_BASELINES)
    expected_functions = {"Rastrigin", "Ackley", "Griewank", "Zakharov"}
    expected_dims = {200, 500, 1000}
    expected_instances = {0, 1, 2, 3, 4}

    # 1. frozen + recomputed content hash
    if not manifest.get("frozen"):
        errors.append("baseline component is not frozen")
    stored = manifest.get("manifest_sha256")
    if stored is None:
        errors.append("baseline component missing manifest_sha256")
    elif manifest_sha256(manifest) != stored:
        errors.append("baseline component manifest_sha256 mismatch (modified after freeze)")

    # 2. component_role
    if manifest.get("component_role") != "baseline_extension":
        errors.append(
            f"component_role {manifest.get('component_role')!r} != 'baseline_extension'")

    # 3. stage
    if manifest.get("stage") != _BASELINE_EXTENSION_STAGE:
        errors.append(f"stage {manifest.get('stage')!r} != {_BASELINE_EXTENSION_STAGE!r}")

    # 4. suite
    if manifest.get("suite") != _BASELINE_EXTENSION_SUITE:
        errors.append(f"suite {manifest.get('suite')!r} != {_BASELINE_EXTENSION_SUITE!r}")

    # 5. selection_hash (non-empty; must match a passed selection's hash)
    sel_hash = manifest.get("selection_hash")
    if not sel_hash:
        errors.append("baseline component missing selection_hash")
    if selection is not None:
        passed_hash = selection.get("selection_hash")
        if passed_hash and sel_hash and passed_hash != sel_hash:
            errors.append(
                f"selection_hash mismatch: manifest={sel_hash!r} selection={passed_hash!r}")

    tasks = manifest.get("tasks") or []
    task_algos = {t.get("algorithm_id") or t.get("algorithm") for t in tasks}

    # 6. baseline set — metadata AND the actual task algorithm set
    meta_baselines = set(manifest.get("baseline_algorithms", []))
    if meta_baselines != expected_baselines:
        errors.append(
            f"baseline_algorithms {sorted(meta_baselines)} != expected "
            f"{sorted(expected_baselines)}")
    if tasks and task_algos != expected_baselines:
        errors.append(
            f"task algorithm set {sorted(task_algos)} != expected "
            f"{sorted(expected_baselines)}")

    # 7. function set
    functions = {t.get("function") for t in tasks}
    if functions != expected_functions:
        errors.append(
            f"function set {sorted(functions)} != expected {sorted(expected_functions)}")

    # 8. dimension set
    dims = {int(t["dimension"]) for t in tasks if t.get("dimension") is not None}
    if dims != expected_dims:
        errors.append(f"dimension set {sorted(dims)} != expected {sorted(expected_dims)}")

    # 9. instance set
    instances = {int(t["instance"]) for t in tasks if t.get("instance") is not None}
    if instances != expected_instances:
        errors.append(f"instance set {sorted(instances)} != expected {sorted(expected_instances)}")

    # 10. each (algorithm, function, dimension, instance) exactly once
    seen: dict[tuple, int] = {}
    duplicates: list[tuple] = []
    for t in tasks:
        key = (
            t.get("algorithm_id") or t.get("algorithm"),
            t.get("function"),
            int(t["dimension"]) if t.get("dimension") is not None else None,
            int(t["instance"]) if t.get("instance") is not None else None,
        )
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2:
            duplicates.append(key)
    if duplicates:
        errors.append(
            f"{len(duplicates)} duplicate (algorithm,function,dimension,instance) task(s); "
            f"each combination must appear exactly once (e.g. {duplicates[:3]})")

    # 11. exactly 300 tasks and 300 distinct run_ids
    run_ids = [t.get("run_id") for t in tasks]
    distinct_run_ids = len({r for r in run_ids if r is not None})
    if len(tasks) != 300:
        errors.append(f"baseline component has {len(tasks)} tasks, expected exactly 300")
    if distinct_run_ids != 300:
        errors.append(
            f"baseline component has {distinct_run_ids} distinct run_ids, expected exactly 300")

    # 12. every instance_artifact_dir must point at a confirmatory instance
    non_conf = sorted({
        t.get("instance_artifact_dir") for t in tasks
        if not (t.get("instance_artifact_dir")
                and "confirmatory_" in t.get("instance_artifact_dir"))
    })
    if non_conf:
        errors.append(
            f"{len(non_conf)} task(s) with non-confirmatory instance_artifact_dir "
            f"(expected 'instances/confirmatory_*'); e.g. {non_conf[:3]}")

    # 13. if an instance index is supplied, every entry must be confirmatory-stage
    if instance_index is not None:
        bad_idx = [
            (k, e.get("stage")) for k, e in instance_index.items()
            if e.get("stage") != "confirmatory"
        ]
        if bad_idx:
            errors.append(
                f"instance_index has {len(bad_idx)} non-confirmatory entry/entries "
                f"(stage != 'confirmatory'); e.g. {bad_idx[:3]}")

    # 14. no winner/base (SMCO-family) algorithm may appear
    forbidden = {a for a in task_algos if a and ("SMCO" in a or "-BASE-" in a)}
    if selection is not None:
        winner = selection.get("winner")
        if winner:
            forbidden |= {a for a in task_algos if a == winner}
    if forbidden:
        errors.append(
            f"baseline component contains winner/base algorithm(s) {sorted(forbidden)}; "
            f"only the 5 baselines are allowed")

    return errors


def confirmatory_errors(manifest: dict, *, selection: dict | None = None) -> list[str]:
    """Return Gate-F violations for a confirmatory manifest (empty == ok)."""
    # P1c (review §4): a baseline_extension component is validated by the exact
    # 300-task contract, NOT by the winner-present checks below. Only a component
    # that passes baseline_component_errors() skips the winner closure — any
    # structural gap (wrong selection, missing/duplicated tasks, development
    # instances, winner/base leak) fails Gate-F here. component_role alone never
    # bypasses Gate-F, because the full task matrix is checked.
    if _is_baseline_extension(manifest):
        return baseline_component_errors(manifest, selection=selection)
    errors: list[str] = []
    if not manifest.get("frozen"):
        errors.append("manifest is not frozen")
    stored = manifest.get("manifest_sha256")
    if stored is None:
        errors.append("manifest missing manifest_sha256")
    elif manifest_sha256(manifest) != stored:
        errors.append("manifest_sha256 mismatch (manifest modified after freeze)")
    tasks = manifest.get("tasks", [])
    # A-03: selection-driven closure — every task algorithm must be in the
    # frozen allowed set, and the winner's configuration_hash must be present.
    allowed = manifest.get("allowed_algorithms")
    if allowed is not None:
        task_algos = {t.get("algorithm_id") or t.get("algorithm") for t in tasks}
        extra = sorted(task_algos - set(allowed))
        if extra:
            errors.append(f"manifest has algorithms outside the allowed set: {extra}")
    winner_hash = manifest.get("winner_config_hash")
    if winner_hash and winner_hash != NONE_TOKEN:
        if not any(t.get("configuration_hash") == winner_hash for t in tasks):
            errors.append("winner_config_hash not present in any manifest task")
    if selection is not None:
        winner = selection.get("winner")
        if not winner:
            errors.append("selection has no winner")
        else:
            ids = {t.get("algorithm_id") for t in tasks}
            if winner not in ids:
                errors.append(
                    f"selection winner {winner!r} not present in manifest tasks"
                )
        # R-02: a confirmatory manifest driven by a selection MUST carry the
        # selection closure hashes, and they must match — not merely be compared
        # when the manifest happens to have them.
        sel_hash = selection.get("selection_hash")
        man_hash = manifest.get("selection_hash")
        if man_hash is None:
            errors.append("confirmatory manifest missing selection_hash")
        elif sel_hash and sel_hash != man_hash:
            errors.append("selection_hash mismatch (manifest not built from this selection)")
        sel_wch = selection.get("winner_config_hash")
        man_wch = manifest.get("winner_config_hash")
        if man_wch is None:
            errors.append("confirmatory manifest missing winner_config_hash")
        elif sel_wch and sel_wch != man_wch:
            errors.append("winner_config_hash mismatch (manifest winner != selection winner)")
    return errors


def enforce_confirmatory(manifest: dict, *, selection: dict | None = None) -> bool:
    """Raise ValueError if any confirmatory check fails; else return True."""
    errors = confirmatory_errors(manifest, selection=selection)
    if errors:
        raise ValueError("confirmatory checks failed: " + "; ".join(errors))
    return True


def confirmatory_run_matrix(manifest, *, expected_stage, expected_suite=None,
                            expected_fe_budget_per_d=None) -> dict:
    """R2b: derive the locked run matrix from a frozen confirmatory manifest's
    tasks and verify its stage/suite/budget match the calling runner.

    A confirmatory runner (E4/E5) must read ``suite``, ``dims``, ``instances``,
    ``FE budget`` and the algorithm set ONLY from the manifest, so a frozen
    manifest of the wrong stage (e.g. an E2 manifest driving E4) or a CLI matrix
    override cannot change what actually runs. ``enforce_confirmatory`` must have
    already validated the frozen/hash/selection closure; this function enforces
    stage/suite (+ optional FE-budget-per-d) and extracts the matrix.

    Returns ``{suite, dims, n_instances, fe_budget_per_d}``; the runner maps
    ``n_instances`` to COCO instance ids ``1..n_instances``.
    """
    stage = manifest.get("stage")
    if stage != expected_stage:
        raise ValueError(
            f"manifest stage {stage!r} != expected {expected_stage!r}; "
            f"a {expected_stage} runner cannot be driven by this manifest"
        )
    suite = manifest.get("suite")
    if expected_suite is not None and suite != expected_suite:
        raise ValueError(
            f"manifest suite {suite!r} != expected {expected_suite!r} "
            f"for stage {expected_stage!r}"
        )
    tasks = manifest.get("tasks") or []
    if not tasks:
        raise ValueError("confirmatory manifest has no tasks; cannot derive run matrix")
    dims = sorted({int(t["dimension"]) for t in tasks})
    n_instances = len({int(t["instance"]) for t in tasks})
    per_d: set[int] = set()
    for t in tasks:
        d = int(t["dimension"]) or 1
        per_d.add(int(t["fe_budget"]) // d)
    if len(per_d) != 1:
        raise ValueError(
            f"manifest mixes fe_budget_per_d within {expected_stage}: {sorted(per_d)}"
        )
    budget_per_d = per_d.pop()
    if expected_fe_budget_per_d is not None and budget_per_d != int(expected_fe_budget_per_d):
        raise ValueError(
            f"manifest fe_budget_per_d {budget_per_d} != expected "
            f"{expected_fe_budget_per_d} for stage {expected_stage!r}"
        )
    return {
        "suite": suite,
        "dims": dims,
        "n_instances": n_instances,
        "fe_budget_per_d": budget_per_d,
    }


def confirmatory_coco_contract(
    manifest, *, expected_algos, expected_dims,
    n_instances: int = COCO_N_INSTANCES, n_functions: int = COCO_N_FUNCTIONS,
) -> list:
    """R6c: validate a COCO confirmatory manifest is the FULL plan matrix.

    The manifest tasks must cover exactly ``expected_algos`` algorithms over
    ``n_functions`` functions x ``expected_dims`` dimensions x ``n_instances``
    instances — so a manifest with a baseline subset (e.g. only DE) or a partial
    function/dim/instance grid cannot run as canonical E4/E5. Returns the
    verified algorithm set (sorted).
    """
    tasks = manifest.get("tasks") or []
    if not tasks:
        raise ValueError("confirmatory manifest has no tasks")
    algos = {t.get("algorithm_id") or t.get("algorithm") for t in tasks}
    expected = set(expected_algos)
    if algos != expected:
        missing = sorted(expected - algos)
        extra = sorted(algos - expected)
        raise ValueError(
            f"manifest algorithm set {sorted(algos)} != expected {sorted(expected)} "
            f"(missing {missing}, extra {extra})")
    dims = {int(t["dimension"]) for t in tasks}
    expected_dim_set = {int(d) for d in expected_dims}
    if dims != expected_dim_set:
        raise ValueError(
            f"manifest dims {sorted(dims)} != expected {sorted(expected_dim_set)}")
    instances = {int(t["instance"]) for t in tasks}
    if len(instances) != n_instances:
        raise ValueError(
            f"manifest has {len(instances)} instances, expected {n_instances} "
            f"(matrix: {n_functions} functions x {sorted(expected_dim_set)} dims "
            f"x {n_instances} instances)")
    functions = {t.get("function") for t in tasks}
    if len(functions) != n_functions:
        raise ValueError(
            f"manifest has {len(functions)} functions, expected {n_functions}")
    # R8c: each (algorithm, function, dimension, instance) combination must appear
    # exactly once. A manifest with the right set sizes and total count but a
    # duplicated/missing combination (e.g. 2520 rows, 2519 unique) must not pass.
    combos: set[tuple] = set()
    duplicates: list[tuple] = []
    for t in tasks:
        key = (t.get("algorithm_id") or t.get("algorithm"), t.get("function"),
               int(t["dimension"]), int(t["instance"]))
        if key in combos:
            duplicates.append(key)
        else:
            combos.add(key)
    if duplicates:
        raise ValueError(
            f"manifest has {len(duplicates)} duplicate (algorithm,function,dimension,"
            f"instance) task(s); each combination must appear exactly once "
            f"(e.g. {duplicates[:3]})")
    expected_tasks = n_functions * len(expected_dim_set) * n_instances * len(expected)
    if len(tasks) != expected_tasks:
        raise ValueError(
            f"manifest has {len(tasks)} tasks, expected "
            f"{n_functions}x{len(expected_dim_set)}x{n_instances}x{len(expected)}="
            f"{expected_tasks}")
    return sorted(expected)


# --- P1c: E3 composite manifest (baseline component + comparative composite) ---

_BASELINE_EXTENSION_BASELINES = ("DE", "GA", "PSO", "SA", "GenSA")
_BASELINE_EXTENSION_STAGE = "e3_companion_baselines"
_BASELINE_EXTENSION_SUITE = "synthetic_highdim"


def build_baseline_component_manifest(
    selection, *,
    stage=_BASELINE_EXTENSION_STAGE,
    suite=_BASELINE_EXTENSION_SUITE,
    functions, dims, n_instances,
    fe_budget_per_d, checkpoints_per_d,
    baselines=_BASELINE_EXTENSION_BASELINES,
    instance_index=None, manifest_id=None,
) -> dict:
    """P1c: build a frozen baseline-only component manifest (no winner/base).

    The E3 comparative analysis reuses E2's audited winner/base (120 rows,
    stage=e2) and only runs the 5 baselines fresh. This component carries
    exactly ``baselines × functions × dims × n_instances`` tasks.
    ``component_role="baseline_extension"`` lets Gate-F skip winner-present
    checks, but only when ALL structural constraints are met (stage, suite,
    baselines, selection_hash) — see :func:`confirmatory_errors`.
    """
    if stage != _BASELINE_EXTENSION_STAGE:
        raise ValueError(
            f"baseline component stage must be {_BASELINE_EXTENSION_STAGE!r}, "
            f"got {stage!r}")
    if suite != _BASELINE_EXTENSION_SUITE:
        raise ValueError(
            f"baseline component suite must be {_BASELINE_EXTENSION_SUITE!r}, "
            f"got {suite!r}")
    if tuple(baselines) != _BASELINE_EXTENSION_BASELINES:
        raise ValueError(
            f"baseline component baselines must be exactly "
            f"{_BASELINE_EXTENSION_BASELINES}, got {tuple(baselines)}")
    tasks = list(expand_baseline_tasks(
        stage, suite, functions, dims, n_instances, baselines,
        fe_budget_per_d=fe_budget_per_d, checkpoints_per_d=checkpoints_per_d,
        instance_index=instance_index,
    ))
    expected = len(functions) * len(dims) * n_instances * len(baselines)
    if len(tasks) != expected:
        raise ValueError(
            f"baseline component has {len(tasks)} tasks, expected {expected}")
    manifest = build_manifest(stage, suite, tasks, manifest_id=manifest_id)
    manifest["component_role"] = "baseline_extension"
    manifest["selection_hash"] = selection.get("selection_hash")
    manifest["baseline_algorithms"] = list(baselines)
    return freeze_manifest(manifest)


# --- P1c composite: build + validate (constraints 1-3) ---

_EXPECTED_E3_ALGORITHMS = frozenset({
    "PY-SP-SMCO-EVO", "PY-BASE-SMCO",
    "DE", "GA", "PSO", "SA", "GenSA",
})


def _file_sha256(path) -> str:
    """SHA-256 of a file's raw bytes (for content-hash binding)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_ids_from_csv(path) -> set:
    """Read the run_id column from a merged valid_runs.csv."""
    with open(path, newline="") as f:
        return {row["run_id"] for row in csv.DictReader(f)}


def _run_id_set_sha256(run_ids) -> str:
    """SHA-256 of the sorted run_id set (order-independent)."""
    return hashlib.sha256(json.dumps(sorted(run_ids)).encode()).hexdigest()


def build_comparative_composite(*, e2_manifest_path, e2_merged_dir,
                                baseline_component_path, baseline_merged_dir) -> dict:
    """P1c: build a frozen comparative composite referencing E2 winner/base +
    E3 baseline component. Auto-derives algorithm set (constraint 2), checks
    selection_hash consistency (constraint 3), and binds content hashes
    (constraint 1). Does NOT modify E2 fields.
    """
    e2m = json.loads(Path(e2_manifest_path).read_text())
    bcm = json.loads(Path(baseline_component_path).read_text())
    # constraint 2: auto-derive algorithms from manifests (no CLI param)
    e2_algos = {e2m["winner_algorithm"], e2m["matched_base_algorithm"]}
    bl_algos = set(bcm["baseline_algorithms"])
    algos = e2_algos | bl_algos
    if algos != _EXPECTED_E3_ALGORITHMS:
        raise ValueError(
            f"composite algorithms {sorted(algos)} != expected "
            f"{sorted(_EXPECTED_E3_ALGORITHMS)}")
    # constraint 3: selection_hash must be consistent
    e2_sel = e2m.get("selection_hash")
    bc_sel = bcm.get("selection_hash")
    if not e2_sel or e2_sel != bc_sel:
        raise ValueError(
            f"selection_hash mismatch: E2={e2_sel!r} baseline={bc_sel!r}")
    # constraint 1: content hash binding (valid_runs + audit + run_id set)
    e2_vr = Path(e2_merged_dir) / "valid_runs.csv"
    e2_au = Path(e2_merged_dir) / "provenance_audit.json"
    bc_vr = Path(baseline_merged_dir) / "valid_runs.csv"
    bc_au = Path(baseline_merged_dir) / "provenance_audit.json"
    e2_rids = _run_ids_from_csv(e2_vr)
    bc_rids = _run_ids_from_csv(bc_vr)
    if e2_rids & bc_rids:
        raise ValueError(
            f"run_id overlap between E2 and baseline: {sorted(e2_rids & bc_rids)[:5]}")
    e2_audit = json.loads(e2_au.read_text())
    bc_audit = json.loads(bc_au.read_text())
    composite = {
        "composite_type": "comparative_composite",
        "components": {
            "winner_base": {
                "manifest_path": str(e2_manifest_path),
                "manifest_sha256": e2m["manifest_sha256"],
                "merged_dir": str(e2_merged_dir),
                "valid_runs_sha256": _file_sha256(e2_vr),
                "audit_sha256": _file_sha256(e2_au),
                "run_id_set_sha256": _run_id_set_sha256(e2_rids),
                "audit_passed": e2_audit.get("passed") is True,
                "n_runs": len(e2_rids),
                "stage": e2m["stage"],
                "selection_hash": e2_sel,
                "algorithms": sorted(e2_algos),
            },
            "baseline_extension": {
                "manifest_path": str(baseline_component_path),
                "manifest_sha256": bcm["manifest_sha256"],
                "merged_dir": str(baseline_merged_dir),
                "valid_runs_sha256": _file_sha256(bc_vr),
                "audit_sha256": _file_sha256(bc_au),
                "run_id_set_sha256": _run_id_set_sha256(bc_rids),
                "audit_passed": bc_audit.get("passed") is True,
                "n_runs": len(bc_rids),
                "stage": bcm["stage"],
                "component_role": bcm.get("component_role"),
                "selection_hash": bc_sel,
                "algorithms": sorted(bl_algos),
            },
        },
        "algorithms": sorted(algos),
        "total_runs": len(e2_rids) + len(bc_rids),
    }
    composite["frozen"] = True
    composite["composite_sha256"] = hashlib.sha256(
        json.dumps(composite, sort_keys=True).encode()).hexdigest()
    return composite


def validate_composite(composite, *, e2_merged_dir, baseline_merged_dir) -> list:
    """P1c: validate a comparative composite by recomputing content hashes
    (constraint 1) and checking all structural constraints. Returns [] on pass.
    """
    errors: list[str] = []
    for key, mdir in [("winner_base", e2_merged_dir),
                      ("baseline_extension", baseline_merged_dir)]:
        comp = composite["components"][key]
        vr = Path(mdir) / "valid_runs.csv"
        au = Path(mdir) / "provenance_audit.json"
        if _file_sha256(vr) != comp["valid_runs_sha256"]:
            errors.append(f"{key} valid_runs.csv hash mismatch (source modified after freeze)")
        if _file_sha256(au) != comp["audit_sha256"]:
            errors.append(f"{key} audit.json hash mismatch")
        rids = _run_ids_from_csv(vr)
        if _run_id_set_sha256(rids) != comp["run_id_set_sha256"]:
            errors.append(f"{key} run_id set hash mismatch")
        if not comp["audit_passed"]:
            errors.append(f"{key} audit not passed")
    e2_rids = _run_ids_from_csv(Path(e2_merged_dir) / "valid_runs.csv")
    bc_rids = _run_ids_from_csv(Path(baseline_merged_dir) / "valid_runs.csv")
    if e2_rids & bc_rids:
        errors.append(f"run_id overlap: {sorted(e2_rids & bc_rids)[:5]}")
    if set(composite["algorithms"]) != _EXPECTED_E3_ALGORITHMS:
        errors.append(f"algorithms {composite['algorithms']} != expected")
    e2_sel = composite["components"]["winner_base"]["selection_hash"]
    bc_sel = composite["components"]["baseline_extension"]["selection_hash"]
    if e2_sel != bc_sel:
        errors.append("selection_hash mismatch between components")
    if composite["total_runs"] != len(e2_rids) + len(bc_rids):
        errors.append("total_runs mismatch")
    return errors


__all__ = [
    "is_run_complete",
    "baseline_component_errors",
    "build_baseline_component_manifest",
    "build_comparative_composite",
    "validate_composite",
    "plan_batch",
    "build_confirmatory_manifest",
    "confirmatory_errors",
    "enforce_confirmatory",
    "confirmatory_run_matrix",
    "confirmatory_coco_contract",
    "E4_BASELINES",
    "E4_DIMENSIONS",
    "E5_DIMENSIONS",
    "COCO_N_FUNCTIONS",
    "COCO_N_INSTANCES",
]
