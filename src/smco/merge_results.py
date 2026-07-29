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


def resolve_supersedes(rows: list[dict]) -> tuple[list[dict], set[str]]:
    """Split rows into (valid, superseded_run_ids).

    A row whose ``supersedes_run_id`` is a real run_id removes that run_id from
    the valid set (it stays in all_attempts).
    """
    superseded: set[str] = set()
    for row in rows:
        sup = row.get("supersedes_run_id")
        if sup and sup != NONE_TOKEN:
            superseded.add(sup)
    valid = [r for r in rows if r["run_id"] not in superseded]
    return valid, superseded


def _identity_key(row: dict) -> tuple:
    """Identity (excluding run_id) — same key => duplicate unless supersedes."""
    return (
        row["function"], int(row["dimension"]), int(row["instance"]),
        row["algorithm_id"], row["language"], row["state_semantics"],
        row["evolution_strategy"], int(row["seed"]),
    )


def _check(name: str, rows: list[dict], ok: bool, errors: list[str]) -> dict:
    return {"name": name, "passed": ok, "n": len(rows), "errors": errors}


def audit_payloads(rows: list[dict], task_index: dict[str, dict]) -> dict:
    """Run the 11 provenance checks; return {passed, failed_checks, checks, n_rows}.

    ``passed=False`` does not crash the merge — the analysis layer (Task 12)
    refuses to build primary tables when the audit fails.
    """
    checks: list[dict] = []

    # 1. run_id uniqueness
    ids = [r["run_id"] for r in rows]
    dup_ids = sorted({i for i in ids if ids.count(i) > 1})
    checks.append(_check("run_id_uniqueness", rows, not dup_ids,
                         [f"duplicate run_id: {i}" for i in dup_ids]))

    # 2. manifest coverage (orphans: run_id not in task_index)
    orphans = [r["run_id"] for r in rows if r["run_id"] not in task_index]
    checks.append(_check("manifest_coverage", rows, not orphans,
                         [f"run_id not in any manifest: {o}" for o in orphans]))

    # 3. supersedes target exists
    known = set(ids) | set(task_index)
    dangling = [r["supersedes_run_id"] for r in rows
                if r.get("supersedes_run_id") not in (NONE_TOKEN, None)
                and r["supersedes_run_id"] not in known]
    checks.append(_check("supersedes_resolvable", rows, not dangling,
                         [f"supersedes unknown run_id: {d}" for d in dangling]))

    # 4. configuration_hash consistent with task (SMCO only)
    bad_cfg = []
    for r in rows:
        t = task_index.get(r["run_id"])
        if t and "configuration_hash" in t and r.get("configuration_hash") != t["configuration_hash"]:
            bad_cfg.append(r["run_id"])
    checks.append(_check("configuration_hash_consistent", rows, not bad_cfg,
                         [f"hash mismatch: {b}" for b in bad_cfg]))

    # 5. FE <= budget
    over = [r["run_id"] for r in rows if int(r["fe_used"]) > int(r["fe_budget"])]
    checks.append(_check("fe_within_budget", rows, not over,
                         [f"fe_over_budget: {o}" for o in over]))

    # 6. objective direction
    wrong_dir = [r["run_id"] for r in rows if r.get("objective_sense") != "minimize"]
    checks.append(_check("objective_direction", rows, not wrong_dir,
                         [f"non-minimize: {w}" for w in wrong_dir]))

    # 7. known_optimum / gap sanity (best >= optimum - tol in minimisation)
    bad_gap = []
    for r in rows:
        try:
            if r["best_value"] < r["known_optimum"] - 1e-6:
                bad_gap.append(r["run_id"])
        except TypeError:
            pass  # NaN best (infra/timeout) stays in the denominator, not a gap error
    checks.append(_check("gap_sanity", rows, not bad_gap,
                         [f"best<optimum: {b}" for b in bad_gap]))

    # 8. start_points_hash consistent within (function,dim,instance)
    by_inst: dict[tuple, set] = {}
    for r in rows:
        key = (r["function"], int(r["dimension"]), int(r["instance"]))
        by_inst.setdefault(key, set()).add(r.get("start_points_hash"))
    clash = [f"{k}" for k, v in by_inst.items() if len(v) > 1]
    checks.append(_check("start_points_hash_consistent", rows, not clash,
                         [f"instance has multiple starts hashes: {c}" for c in clash]))

    # 9. non-EVO rows not duplicated by strategy + identity duplicates
    bad_strategy = [r["run_id"] for r in rows
                    if r["evolutionary"] == "false" and r["evolution_strategy"] != NONE_TOKEN]
    seen: dict[tuple, list[str]] = {}
    for r in rows:
        seen.setdefault(_identity_key(r), []).append(r["run_id"])
    dups = [rids for rids in seen.values() if len(rids) > 1]
    checks.append(_check("no_pseudo_duplicates", rows, not bad_strategy and not dups,
                         [f"base row has strategy: {b}" for b in bad_strategy]
                         + [f"identity duplicated: {rids}" for rids in dups]))

    # 10. confirmatory seed equals derive_seed(stage,...,algorithm)
    bad_seed = []
    for r in rows:
        t = task_index.get(r["run_id"])
        if not t or t.get("stage") not in _CONFIRMATORY_STAGES:
            continue
        algo = t.get("algorithm_id") or t.get("algorithm")
        expected = derive_seed(t["stage"], t.get("suite", "synthetic_highdim"),
                               t["function"], int(t["dimension"]), int(t["instance"]),
                               int(t.get("replication", 0)), algo)
        if int(r["seed"]) != int(expected):
            bad_seed.append(r["run_id"])
    checks.append(_check("seed_matches_derive_seed", rows, not bad_seed,
                         [f"seed mismatch (possible dev seed): {b}" for b in bad_seed]))

    # 11. statuses are all in the contract vocabulary (kept in the denominator)
    bad_status = [r["run_id"] for r in rows if r["status"] not in STATUSES]
    checks.append(_check("status_vocabulary", rows, not bad_status,
                         [f"unknown status: {b}" for b in bad_status]))

    failed = [c["name"] for c in checks if not c["passed"]]
    return {
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "n_rows": len(rows),
    }


__all__ = [
    "classify_task",
    "build_task_index",
    "smco_row_from_outcome",
    "baseline_row_from_outcome",
    "resolve_supersedes",
    "audit_payloads",
]
