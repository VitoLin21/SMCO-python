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

from .experiment_manifests import manifest_sha256


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
    if selection is not None:
        winner = selection.get("winner")
        if not winner:
            errors.append("selection has no winner")
        else:
            ids = {t.get("algorithm_id") for t in manifest.get("tasks", [])}
            if winner not in ids:
                errors.append(
                    f"selection winner {winner!r} not present in manifest tasks"
                )
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
    "confirmatory_errors",
    "enforce_confirmatory",
]
