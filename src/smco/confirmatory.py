"""Confirmatory-run enforcement (Task 10 / Gate F 强制检查).

A confirmatory runner (E2/E3/E4/E5) must refuse to start unless its manifest is
frozen and content-hash-consistent, and (when a selection is provided) the
selected winner is actually one of the manifest's algorithm_ids. The runner must
only execute tasks listed in the manifest, and report completed/missing counts.

``is_run_complete`` / ``plan_batch`` live here as the single source of truth
(batch runners import them).
"""

from __future__ import annotations

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


def confirmatory_errors(manifest: dict, *, selection: dict | None = None) -> list[str]:
    """Return Gate-F violations for a confirmatory manifest (empty == ok)."""
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


__all__ = [
    "is_run_complete",
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
