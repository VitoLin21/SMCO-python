"""Merge + provenance audit for the SMCO-EVO high-dim paper (Task 11, redesigned).

All three workers (Py SMCO / R SMCO / baseline) emit one unified outcome payload.
This module is the single place that builds ``RESULT_COLUMNS`` rows from an
outcome plus its frozen manifest task, resolves supersedes, runs the provenance
audit and writes the ``merged/`` artefacts. See
``docs/superpowers/specs/2026-07-29-smco-evo-unified-output-contract-design.md``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .experiment_manifests import (
    derive_seed,
    load_manifest,
    result_row_from_task,
    verify_manifest,
)
from .coco_outcome import COCO_SUITES, coco_outcome_errors
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
    """Load + verify all manifests; return {run_id: task}.

    Each task is annotated with its source ``manifest_id`` so :func:`merge`
    can stamp the real manifest id onto result rows (A-08 #3).
    """
    index: dict[str, dict] = {}
    for path in manifest_paths:
        manifest = load_manifest(path)
        verify_manifest(manifest)
        mid = manifest.get("manifest_id") or Path(path).stem
        for task in manifest.get("tasks", []):
            task["manifest_id"] = mid
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
        # COCO-native outcomes deliberately have no f_opt on cocoex versions
        # that do not expose one.  Preserve that as NaN in the generic row;
        # do not silently substitute the synthetic optimum 0.
        known_optimum=_num(outcome.get("known_optimum")),
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
        "n_starts": int(outcome.get("n_starts") or 0),  # A-09 #1: from outcome, not hard 0
        "fe_budget": int(task["fe_budget"]),
        "fe_used": int(outcome.get("fe_used") or 0),
        "checkpoint_fe": int(task["fe_budget"]),
        "best_value": _num(outcome.get("best_value")),
        "known_optimum": _num(outcome.get("known_optimum")),
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
    """Identity (excluding run_id) — same key => duplicate unless supersedes.

    n_starts is included so the E6.1 start-count tiers (same algorithm_id/seed,
    different n_starts) are distinct, not flagged as pseudo-duplicates.
    configuration_hash is included so E6.2/E6.3 ablation configs (same
    algorithm_id/strategy/seed but different evolution_points/elimination_rate)
    are distinct, not flagged as pseudo-duplicates.
    """
    return (
        row["function"], int(row["dimension"]), int(row["instance"]),
        row["algorithm_id"], row["language"], row["state_semantics"],
        row["evolution_strategy"], int(row["seed"]), int(row["n_starts"]),
        row.get("configuration_hash", ""),
    )


def _check(name: str, rows: list[dict], ok: bool, errors: list[str]) -> dict:
    return {"name": name, "passed": ok, "n": len(rows), "errors": errors}


def audit_payloads(rows: list[dict], task_index: dict[str, dict],
                   *, outcome_index: dict[str, dict] | None = None) -> dict:
    """Run the provenance checks; return {passed, failed_checks, checks, n_rows}.

    Every homogeneous suite receives exactly 12 checks. COCO-suite rows
    (E4/E5 external validation) replace synthetic ``start_points_hash`` check 8
    with ``benchmark_provenance`` — never passing through empty synthetic
    fields. ``outcome_index`` (run_id -> raw
    outcome) supplies the benchmark block; it is required when COCO rows exist.
    ``passed=False`` does not crash the merge — the analysis layer refuses to
    build tables when the audit fails.
    """
    checks: list[dict] = []
    coco_rows = [r for r in rows if r.get("suite") in COCO_SUITES]
    non_coco_rows = [r for r in rows if r.get("suite") not in COCO_SUITES]

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

    # 8. Synthetic starts provenance OR COCO benchmark provenance.  This is a
    # replacement, not an additive 13th check, so canonical merged artifacts
    # retain one common 12-check contract.
    if coco_rows and not non_coco_rows:
        bad_bench: list[str] = []
        for r in coco_rows:
            outcome = (outcome_index or {}).get(r["run_id"])
            if outcome is None:
                bad_bench.append(f"{r['run_id']}: no raw outcome for benchmark provenance")
                continue
            for err in coco_outcome_errors(outcome):
                bad_bench.append(f"{r['run_id']}: {err}")
        checks.append(_check("benchmark_provenance", coco_rows, not bad_bench, bad_bench))
    else:
        by_inst: dict[tuple, set] = {}
        for r in non_coco_rows:
            # n_starts in the key so legitimate E6.1 tiers (same instance, more
            # starts -> a different starts artifact) are not flagged as a clash.
            key = (r["function"], int(r["dimension"]), int(r["instance"]), int(r["n_starts"]))
            by_inst.setdefault(key, set()).add(r.get("start_points_hash"))
        clash = [f"{k}" for k, v in by_inst.items() if len(v) > 1]
        checks.append(_check("start_points_hash_consistent", rows, not clash,
                             [f"instance has multiple starts hashes: {c}" for c in clash]))
        if coco_rows:
            # Mixed-suite merges are not canonical, but still cannot evade COCO
            # provenance validation. Kept separate for diagnostic use only.
            bad_bench = []
            for r in coco_rows:
                outcome = (outcome_index or {}).get(r["run_id"])
                if outcome is None:
                    bad_bench.append(f"{r['run_id']}: no raw outcome for benchmark provenance")
                else:
                    bad_bench.extend(f"{r['run_id']}: {e}"
                                     for e in coco_outcome_errors(outcome))
            checks.append(_check("benchmark_provenance", coco_rows,
                                 not bad_bench, bad_bench))

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

    # 12. P1a: provenance complete — every row must carry a non-empty
    # git_commit, environment_hash and machine_id so the result is auditable
    # (reproducible source). Applies to ALL merged input (E1 winner freezing +
    # E2-E6 confirmatory); a missing field means the result cannot stand as
    # formal evidence.
    missing_prov = [r["run_id"] for r in rows
                    if not r.get("git_commit") or not r.get("environment_hash")
                    or not r.get("machine_id")]
    checks.append(_check("provenance_complete", rows, not missing_prov,
                         [f"missing provenance (git_commit/environment_hash/machine_id): {m}"
                          for m in missing_prov]))

    failed = [c["name"] for c in checks if not c["passed"]]
    return {
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "n_rows": len(rows),
    }


def load_raw_outcomes(raw_dirs: Iterable[str]):
    """Yield (path, payload) for every <run_id>.json across raw_dirs."""
    for raw_dir in raw_dirs:
        for path in sorted(Path(raw_dir).glob("*.json")):
            if path.name.startswith(".") or ".tmp" in path.name:
                continue
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            if isinstance(payload, dict) and "run_id" in payload:
                yield path, payload


def _write_csv(path: Path, columns, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _audit_md(audit: dict) -> str:
    lines = ["# Provenance Audit", "",
             f"**Passed:** {audit['passed']}", f"**Rows:** {audit['n_rows']}", ""]
    for c in audit["checks"]:
        flag = "PASS" if c["passed"] else "FAIL"
        lines.append(f"- [{flag}] {c['name']}")
        for e in c["errors"]:
            lines.append(f"    - {e}")
    return "\n".join(lines) + "\n"


def merge(manifest_paths, raw_dirs, merged_dir) -> dict:
    """Load outcomes, build rows, resolve supersedes, audit, write merged/."""
    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    task_index = build_task_index(manifest_paths)

    attempts: list[dict] = []
    outcome_index: dict[str, dict] = {}
    for _path, outcome in load_raw_outcomes(raw_dirs):
        run_id = outcome["run_id"]
        outcome_index[run_id] = outcome
        task = task_index.get(run_id)
        if task is None:
            # A-08 #2: orphan (run_id in no manifest). Build a row from the
            # embedded task so the manifest_coverage audit flags it instead of
            # silently dropping it. manifest_id stays empty (no source manifest).
            task = outcome.get("task") or {}
            if not task.get("run_id"):
                continue
        mid = task.get("manifest_id") or ""
        if classify_task(task) == "smco":
            attempts.append(smco_row_from_outcome(outcome, task, manifest_id=mid))
        else:
            attempts.append(baseline_row_from_outcome(outcome, task, manifest_id=mid))

    valid, superseded = resolve_supersedes(attempts)
    audit = audit_payloads(attempts, task_index, outcome_index=outcome_index)

    # missing = manifest tasks with no raw outcome
    have = {r["run_id"] for r in attempts}
    missing = [{"run_id": t["run_id"], "stage": t["stage"], "function": t["function"],
                "dimension": t["dimension"], "instance": t["instance"],
                "algorithm_id": t.get("algorithm_id") or t.get("algorithm")}
               for t in task_index.values() if t["run_id"] not in have]

    # anytime long table from raw outcomes
    anytime_rows = []
    for _path, outcome in load_raw_outcomes(raw_dirs):
        for a in outcome.get("anytime") or []:
            anytime_rows.append({
                "run_id": outcome["run_id"],
                "checkpoint_fe": a.get("checkpoint_fe"),
                "fe_used": a.get("fe_used"),
                "best_value": a.get("best_value"),
                "normalized_gap": a.get("normalized_gap"),
            })

    _write_csv(merged_dir / "all_attempts.csv", RESULT_COLUMNS, attempts)
    _write_csv(merged_dir / "valid_runs.csv", RESULT_COLUMNS, valid)
    _write_csv(merged_dir / "missing_runs.csv",
               ("run_id", "stage", "function", "dimension", "instance", "algorithm_id"), missing)
    _write_csv(merged_dir / "duplicate_runs.csv", RESULT_COLUMNS,
               [r for r in attempts if r["run_id"] in superseded])
    _write_csv(merged_dir / "anytime.csv",
               ("run_id", "checkpoint_fe", "fe_used", "best_value", "normalized_gap"), anytime_rows)
    (merged_dir / "provenance_audit.json").write_text(json.dumps(audit, indent=2))
    (merged_dir / "provenance_audit.md").write_text(_audit_md(audit))

    return {"n_attempts": len(attempts), "n_valid": len(valid),
            "n_missing": len(missing), "audit": audit}


__all__ = [
    "classify_task",
    "build_task_index",
    "smco_row_from_outcome",
    "baseline_row_from_outcome",
    "resolve_supersedes",
    "audit_payloads",
    "load_raw_outcomes",
    "merge",
]
