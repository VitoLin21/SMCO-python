"""Canonical artifact index for the SMCO-EVO high-dim paper (review §8 + P1 2026-08-02).

Old ``merged/`` and new ``merged_v2/`` (and ``selection/`` vs ``selection_v2/``)
coexist on disk; without a single frozen index, Task 12/13 can read the wrong
version. This module freezes, for every result that feeds the paper, the single
formal path + content hashes + audit status, and validates them by re-reading
from disk.

The contract is CODE (``CANONICAL_CONTRACT``), not data in the index, so a
tampered index cannot downgrade the requirements: ``validate_canonical_index``
always checks the index carries EXACTLY the required keys, each with the
required kind/status/row_count, a matching ``index_sha256`` (frozen), and — for
canonical merged artifacts — that BOTH ``valid_runs.csv`` and
``provenance_audit.json`` still hash-match and the audit is the current 12-check
version that passed. Development (``development_only``) and deferred artifacts
only need to exist.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

CANONICAL_SCHEMA_VERSION = "1"
REQUIRED_AUDIT_CHECKS = 12
_PROVENANCE_CHECK_NAME = "provenance_complete"

# Fixed contract: every canonical index MUST carry exactly these artifacts.
# The value is {kind, status, row_count?}. kind is "merged" (valid_runs.csv +
# provenance_audit.json), "file" (single JSON/CSV) or "dir" (existence-only).
# This is the single source of truth — Task 12/13 resolve paths ONLY through it.
CANONICAL_CONTRACT = {
    "e1_merged": {"kind": "merged", "status": "canonical", "row_count": 1080},
    "e1_selection": {"kind": "file", "status": "canonical"},
    "e2_manifest": {"kind": "file", "status": "canonical"},
    "e2_merged": {"kind": "merged", "status": "canonical", "row_count": 120},
    "e3_baseline_component_manifest": {"kind": "file", "status": "canonical"},
    "e3_baseline_merged": {"kind": "merged", "status": "canonical", "row_count": 300},
    "e3_composite": {"kind": "file", "status": "canonical"},
    "e3_composite_merged": {"kind": "merged", "status": "canonical", "row_count": 420},
    "e6_strategy_merged": {"kind": "merged", "status": "canonical", "row_count": 420},
    "e6_start_count_manifest": {"kind": "file", "status": "canonical"},
    "e6_start_count_merged": {"kind": "merged", "status": "canonical", "row_count": 180},
    "e4_dev": {"kind": "file", "status": "development_only"},
    "e5_dev": {"kind": "file", "status": "development_only"},
    "e6_schedule": {"kind": "dir", "status": "deferred"},
}

# Which merged artifact is the E3 comparative analysis input, and which key
# carries its frozen composite (so Task 12 can force the composite gate + read
# the algorithm set from the composite, not from a user stage string).
E3_MERGED_KEY = "e3_composite_merged"
E3_COMPOSITE_KEY = "e3_composite"


def file_sha256(path) -> str:
    """SHA-256 of a file's raw bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merged_audit_summary(merged_dir) -> dict:
    """Read a merged dir's ``provenance_audit.json``: passed / n_rows / n_checks
    / provenance_passed."""
    audit = json.loads((Path(merged_dir) / "provenance_audit.json").read_text())
    checks = audit.get("checks", [])
    return {
        "passed": bool(audit.get("passed")),
        "n_rows": int(audit.get("n_rows", 0)),
        "n_checks": len(checks),
        "provenance_passed": bool(next(
            (c.get("passed") for c in checks
             if c.get("name") == _PROVENANCE_CHECK_NAME), False)),
    }


def distinct_git_commits(merged_dir) -> list[str]:
    """Distinct non-empty ``git_commit`` values in a merged dir's valid_runs.csv
    (ties the artifact to the source code SHA(s))."""
    with open(Path(merged_dir) / "valid_runs.csv", newline="") as handle:
        return sorted({row["git_commit"] for row in csv.DictReader(handle)
                       if row.get("git_commit")})


def index_sha256(index: dict) -> str:
    """Unified SHA-256 over the index content, excluding ``index_sha256`` itself
    (used by BOTH build and validate so they can never disagree)."""
    payload = {k: v for k, v in index.items() if k != "index_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _resolve(path, root):
    p = Path(path)
    return p if (p.is_absolute() or root is None) else Path(root) / p


def _record_artifact(key, path, contract_entry, resolved) -> dict:
    record = {
        "key": key,
        "path": path,
        "kind": contract_entry["kind"],
        "status": contract_entry["status"],
    }
    if "row_count" in contract_entry:
        record["row_count"] = contract_entry["row_count"]
    if not resolved.exists():
        record["missing"] = True
        return record
    if contract_entry["kind"] == "merged":
        record["valid_runs_sha256"] = file_sha256(resolved / "valid_runs.csv")
        record["audit_sha256"] = file_sha256(resolved / "provenance_audit.json")
        record.update(merged_audit_summary(resolved))
        record["git_commits"] = distinct_git_commits(resolved)
    elif contract_entry["kind"] == "file":
        record["sha256"] = file_sha256(resolved)
    # "dir": existence-only (no content hash)
    return record


def build_canonical_index(paths, *, root=".", git_commit=None,
                          contract=CANONICAL_CONTRACT) -> dict:
    """Build a frozen index from a ``{key: path}`` map over a fixed contract.

    ``paths`` must supply every contract key (else the entry is marked missing
    and ``validate_canonical_index`` rejects it). The kind/status/row_count come
    from the contract, NOT from the caller, so a path map cannot downgrade an
    artifact's status.
    """
    artifacts = []
    for key, contract_entry in contract.items():
        if key not in paths:
            artifacts.append({"key": key, "path": None, "kind": contract_entry["kind"],
                              "status": contract_entry["status"], "missing": True})
            continue
        path = paths[key]
        record = _record_artifact(key, path, contract_entry, _resolve(path, root))
        artifacts.append(record)
    index = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "generated_from_git_commit": git_commit,
        "frozen": True,
        "artifacts": artifacts,
    }
    index["index_sha256"] = index_sha256(index)
    return index


def validate_canonical_index(index, *, root=".", contract=CANONICAL_CONTRACT) -> list[str]:
    """Validate a canonical index against the fixed contract. Returns violations
    (empty == intact). Checks: schema version, frozen + index_sha256, every
    contract key present with the right kind/status/row_count, no duplicate or
    unknown keys, and (for canonical merged) both valid_runs.csv and
    provenance_audit.json hash-match with a current 12-check passing audit.
    """
    errors: list[str] = []
    if index.get("schema_version") != CANONICAL_SCHEMA_VERSION:
        errors.append(f"schema_version {index.get('schema_version')!r} != {CANONICAL_SCHEMA_VERSION!r}")
    if index.get("frozen") is not True:
        errors.append("index is not frozen")
    if index.get("index_sha256") != index_sha256(index):
        errors.append("index_sha256 mismatch (index modified after freeze)")

    by_key: dict[str, dict] = {}
    for entry in index.get("artifacts", []):
        key = entry.get("key")
        if key in by_key:
            errors.append(f"duplicate artifact key {key!r}")
        by_key[key] = entry

    # unknown keys (in index but not in the contract)
    for key in by_key:
        if key not in contract:
            errors.append(f"unknown artifact key {key!r} (not in contract)")

    # required keys: present + correct kind/status/row_count + on-disk intact
    for key, want in contract.items():
        entry = by_key.get(key)
        if entry is None:
            errors.append(f"missing required artifact {key!r}")
            continue
        for field in ("kind", "status"):
            if entry.get(field) != want[field]:
                errors.append(
                    f"{key}: {field} {entry.get(field)!r} != contract {want[field]!r}")
        if "row_count" in want and entry.get("row_count") != want["row_count"]:
            errors.append(
                f"{key}: row_count {entry.get('row_count')} != contract {want['row_count']}")
        resolved = _resolve(entry.get("path"), root) if entry.get("path") else None
        if resolved is None or not resolved.exists():
            errors.append(f"{key}: missing path {entry.get('path')!r}")
            continue
        if want["kind"] == "merged":
            errors += _validate_merged_on_disk(key, entry, resolved, want)
        elif want["kind"] == "file":
            if entry.get("sha256") and file_sha256(resolved) != entry["sha256"]:
                errors.append(f"{key}: file hash mismatch (changed after freeze)")
    return errors


def _validate_merged_on_disk(key, entry, resolved, want) -> list[str]:
    """On-disk checks for a merged artifact (used for both the per-source merged
    entries and the final composite merged)."""
    errors: list[str] = []
    valid_runs = resolved / "valid_runs.csv"
    audit_path = resolved / "provenance_audit.json"
    if not valid_runs.exists():
        errors.append(f"{key}: missing valid_runs.csv")
        return errors
    if not audit_path.exists():
        errors.append(f"{key}: missing provenance_audit.json")
        return errors
    if entry.get("valid_runs_sha256") and file_sha256(valid_runs) != entry["valid_runs_sha256"]:
        errors.append(f"{key}: valid_runs.csv hash mismatch (changed after freeze)")
    if entry.get("audit_sha256") and file_sha256(audit_path) != entry["audit_sha256"]:
        errors.append(f"{key}: provenance_audit.json hash mismatch (changed after freeze)")
    if entry.get("status") == "canonical":
        summary = merged_audit_summary(resolved)
        if not summary["passed"]:
            errors.append(f"{key}: audit not passed")
        if summary["n_checks"] != REQUIRED_AUDIT_CHECKS:
            errors.append(
                f"{key}: audit has {summary['n_checks']} checks, expected {REQUIRED_AUDIT_CHECKS}")
        if not summary["provenance_passed"]:
            errors.append(f"{key}: provenance_complete not passed")
        if "row_count" in want and summary["n_rows"] != want["row_count"]:
            errors.append(f"{key}: rows {summary['n_rows']} != contract {want['row_count']}")
    return errors


def resolve_analysis_target(index, key, *, root=".") -> dict:
    """Resolve a Task-12 analysis target by artifact key (review P0): Task 12/13
    must take inputs from the canonical index, never an arbitrary merged path.

    Returns ``{key, merged_dir, is_e3, composite_path}``. For the E3 composite
    merged key, ``is_e3=True`` and ``composite_path`` points at the frozen
    composite so the caller can force the composite gate and read the algorithm
    set from the composite (not from a user stage string).
    """
    if key not in CANONICAL_CONTRACT:
        raise ValueError(f"unknown artifact key {key!r} (not in canonical contract)")
    entry = next((a for a in index.get("artifacts", []) if a.get("key") == key), None)
    if entry is None:
        raise ValueError(f"artifact {key!r} not in index")
    if CANONICAL_CONTRACT[key]["kind"] != "merged":
        raise ValueError(f"artifact {key!r} is not a merged analysis target")
    merged_dir = _resolve(entry["path"], root)
    result = {"key": key, "merged_dir": str(merged_dir), "is_e3": key == E3_MERGED_KEY,
              "composite_path": None}
    if key == E3_MERGED_KEY:
        comp = next((a for a in index.get("artifacts", []) if a.get("key") == E3_COMPOSITE_KEY), None)
        if comp is None:
            raise ValueError(f"E3 composite key {E3_COMPOSITE_KEY!r} not in index")
        result["composite_path"] = str(_resolve(comp["path"], root))
    return result


__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "REQUIRED_AUDIT_CHECKS",
    "CANONICAL_CONTRACT",
    "E3_MERGED_KEY",
    "E3_COMPOSITE_KEY",
    "file_sha256",
    "merged_audit_summary",
    "distinct_git_commits",
    "index_sha256",
    "build_canonical_index",
    "validate_canonical_index",
    "resolve_analysis_target",
]
