"""Merge + provenance audit for the SMCO-EVO high-dim paper (Task 11, redesigned).

All three workers (Py SMCO / R SMCO / baseline) emit one unified outcome payload.
This module is the single place that builds ``RESULT_COLUMNS`` rows from an
outcome plus its frozen manifest task, resolves supersedes, runs the provenance
audit and writes the ``merged/`` artefacts. See
``docs/superpowers/specs/2026-07-29-smco-evo-unified-output-contract-design.md``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .experiment_manifests import (
    derive_seed,
    load_manifest,
    result_row_from_task,
    verify_manifest,
)
from .paper_contract import NONE_TOKEN, RESULT_COLUMNS, SCHEMA_VERSION, STATUSES

_CONFIRMATORY_STAGES = {
    "e2_factorial_highdim",
    "e3_baselines_highdim",
    "e4_bbob_largescale",
    "e5_lowdim_check",
}
_NAN = float("nan")


def classify_task(task: dict) -> str:
    """'smco' if the task carries configuration_hash, else 'baseline'."""
    return "smco" if "configuration_hash" in task else "baseline"


def build_task_index(manifest_paths: Iterable[str]) -> dict[str, dict]:
    """Load + verify all manifests; return {run_id: task}."""
    index: dict[str, dict] = {}
    for path in manifest_paths:
        manifest = load_manifest(path)
        verify_manifest(manifest)
        for task in manifest.get("tasks", []):
            index[task["run_id"]] = task
    return index


def _num(value, default=_NAN):
    return default if value is None else value


def smco_row_from_outcome(outcome: dict, task: dict, manifest_id: str = "") -> dict:
    """Build a contract-valid SMCO RESULT_COLUMNS row from outcome + task."""
    th = {k: v for k, v in (outcome.get("target_hit_fe") or {}).items() if v is not None}
    gap = outcome.get("normalized_gap")
    return result_row_from_task(
        task,
        best_value=_num(outcome.get("best_value")),
        fe_used=int(outcome.get("fe_used") or 0),
        status=outcome.get("status", "infra_failure"),
        known_optimum=_num(outcome.get("known_optimum"), 0.0),
        normalized_gap=NONE_TOKEN if gap is None else gap,
        checkpoint_fe=task["fe_budget"],
        target_hit_fe=th,
        wall_time_sec=float(outcome.get("wall_time_sec") or 0.0),
        peak_memory_mb=float(outcome.get("peak_memory_mb") or 0.0),
        failure_reason=outcome.get("failure_reason", NONE_TOKEN),
        termination_reason=outcome.get("termination_reason", "evaluation_budget"),
        fe_counts_by_event=str(outcome.get("fe_counts_by_event") or {}),
        machine_id=outcome.get("machine_id", ""),
        git_commit=outcome.get("git_commit", ""),
        environment_hash=outcome.get("environment_hash", ""),
        objective_sense="minimize",
        manifest_id=manifest_id,
        supersedes_run_id=outcome.get("supersedes_run_id", NONE_TOKEN),
    )


def _th_cell(th: dict, label: str):
    v = (th or {}).get(label)
    return NONE_TOKEN if v is None else v


def baseline_row_from_outcome(outcome: dict, task: dict, manifest_id: str = "") -> dict:
    """Build a RESULT_COLUMNS row for a baseline run (algorithm_id = DE/GenSA/...).

    Baseline rows bypass ``validate_result_row``'s SMCO ``algorithm_id`` rebuild
    check; only field presence + numeric sanity is enforced (audit step).
    """
    th = outcome.get("target_hit_fe") or {}
    gap = outcome.get("normalized_gap")
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "stage": task["stage"],
        "suite": task.get("suite", "synthetic_highdim"),
        "function": task["function"],
        "dimension": int(task["dimension"]),
        "instance": int(task["instance"]),
        "replication": 0,
        "seed": int(task["seed"]),
        "language": "python",
        "state_semantics": NONE_TOKEN,
        "family": NONE_TOKEN,
        "evolutionary": "false",
        "evolution_strategy": NONE_TOKEN,
        "algorithm_id": task["algorithm"],
        "n_starts": 0,
        "fe_budget": int(task["fe_budget"]),
        "fe_used": int(outcome.get("fe_used") or 0),
        "checkpoint_fe": int(task["fe_budget"]),
        "best_value": _num(outcome.get("best_value")),
        "known_optimum": _num(outcome.get("known_optimum"), 0.0),
        "normalized_gap": NONE_TOKEN if gap is None else gap,
        "objective_sense": "minimize",
        "target_hit_fe_1e-1": _th_cell(th, "1e-1"),
        "target_hit_fe_1e-2": _th_cell(th, "1e-2"),
        "target_hit_fe_1e-3": _th_cell(th, "1e-3"),
        "target_hit_fe_1e-5": _th_cell(th, "1e-5"),
        "wall_time_sec": float(outcome.get("wall_time_sec") or 0.0),
        "peak_memory_mb": float(outcome.get("peak_memory_mb") or 0.0),
        "status": outcome.get("status", "infra_failure"),
        "failure_reason": outcome.get("failure_reason", NONE_TOKEN),
        "is_confirmatory": task["stage"] in _CONFIRMATORY_STAGES,
        "supersedes_run_id": outcome.get("supersedes_run_id", NONE_TOKEN),
        "machine_id": outcome.get("machine_id", ""),
        "git_commit": outcome.get("git_commit", ""),
        "environment_hash": outcome.get("environment_hash", ""),
        "start_points_hash": task.get("start_points_hash") or NONE_TOKEN,
        "instance_hash": task.get("instance_hash") or NONE_TOKEN,
        "configuration_hash": NONE_TOKEN,
        "run_id": task["run_id"],
        "termination_reason": outcome.get("termination_reason", "evaluation_budget"),
        "fe_counts_by_event": str(outcome.get("fe_counts_by_event") or {}),
    }


__all__ = [
    "classify_task",
    "build_task_index",
    "smco_row_from_outcome",
    "baseline_row_from_outcome",
]
