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


__all__ = [
    "is_run_complete",
    "plan_batch",
    "build_confirmatory_manifest",
    "confirmatory_errors",
    "enforce_confirmatory",
]
