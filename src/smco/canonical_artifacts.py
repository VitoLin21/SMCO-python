"""Canonical artifact index for the SMCO-EVO high-dim paper (review §8).

Old ``merged/`` and new ``merged_v2/`` (and ``selection/`` vs ``selection_v2/``)
coexist on disk; without a single frozen index, Task 12/13 analyses can read
the wrong version. This module freezes, for every result that feeds the paper,
the single formal path + a content hash + the audit status, and validates them
by re-reading from disk.

``validate_canonical_index`` rejects: a missing path, a hash mismatch, or a
``canonical`` merged artifact whose audit is not the current 12-check version
or did not pass (or whose ``provenance_complete`` check failed). Development
(``development_only``) and deferred artifacts only need to exist.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

CANONICAL_SCHEMA_VERSION = "1"
REQUIRED_AUDIT_CHECKS = 12
_PROVENANCE_CHECK_NAME = "provenance_complete"


def file_sha256(path) -> str:
    """SHA-256 of a file's raw bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merged_valid_runs_sha256(merged_dir) -> str:
    """SHA-256 of a merged dir's ``valid_runs.csv`` (the canonical data file)."""
    return file_sha256(Path(merged_dir) / "valid_runs.csv")


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


def _resolve(path, root):
    p = Path(path)
    return p if (p.is_absolute() or root is None) else Path(root) / p


def build_canonical_index(spec, *, root=".", git_commit=None) -> dict:
    """Build a frozen index from a spec list.

    Each spec entry: ``{key, path, kind, row_count?, status?}`` where kind is
    ``"merged"`` (has valid_runs.csv + provenance_audit.json), ``"file"``
    (a single JSON/CSV) or ``"dir"`` (existence-only, no content hash), and
    status is ``"canonical"`` (the validator enforces a current 12-check passing
    audit), ``"development_only"`` or ``"deferred"``. Computes the content hash +
    audit summary for each present artifact; records ``missing=true`` for absent
    ones.
    """
    artifacts = []
    for entry in spec:
        path = entry["path"]
        resolved = _resolve(path, root)
        record = {
            "key": entry["key"],
            "path": path,
            "kind": entry["kind"],
            "status": entry.get("status", "canonical"),
        }
        if "row_count" in entry:
            record["row_count"] = entry["row_count"]
        if not resolved.exists():
            record["missing"] = True
            artifacts.append(record)
            continue
        if entry["kind"] == "merged":
            record["sha256"] = merged_valid_runs_sha256(resolved)
            record.update(merged_audit_summary(resolved))
            record["git_commits"] = distinct_git_commits(resolved)
        elif entry["kind"] == "dir":
            pass  # existence-only (e.g. a deferred/un-formalized result tree)
        else:
            record["sha256"] = file_sha256(resolved)
        artifacts.append(record)
    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "generated_from_git_commit": git_commit,
        "artifacts": artifacts,
    }


def validate_canonical_index(index, *, root=".") -> list[str]:
    """Re-read every artifact from disk and recompute hashes; return violations.

    Empty list == the index is intact. Canonical merged artifacts must still
    pass the current 12-check audit with provenance_complete; development_only
    and deferred artifacts only need to exist (and match their recorded hash).
    """
    errors: list[str] = []
    for entry in index.get("artifacts", []):
        key = entry["key"]
        resolved = _resolve(entry["path"], root)
        if not resolved.exists():
            errors.append(f"{key}: missing path {entry['path']}")
            continue
        if entry["kind"] == "merged":
            valid_runs = resolved / "valid_runs.csv"
            if not valid_runs.exists():
                errors.append(f"{key}: missing valid_runs.csv")
                continue
            if entry.get("sha256") and file_sha256(valid_runs) != entry["sha256"]:
                errors.append(
                    f"{key}: valid_runs.csv hash mismatch (changed after freeze)")
            if entry.get("status") == "canonical":
                summary = merged_audit_summary(resolved)
                if not summary["passed"]:
                    errors.append(f"{key}: audit not passed")
                if summary["n_checks"] != REQUIRED_AUDIT_CHECKS:
                    errors.append(
                        f"{key}: audit has {summary['n_checks']} checks, "
                        f"expected {REQUIRED_AUDIT_CHECKS}")
                if not summary["provenance_passed"]:
                    errors.append(f"{key}: provenance_complete not passed")
                if entry.get("row_count") is not None and summary["n_rows"] != entry["row_count"]:
                    errors.append(
                        f"{key}: rows {summary['n_rows']} != {entry['row_count']}")
        else:
            if entry.get("sha256") and file_sha256(resolved) != entry["sha256"]:
                errors.append(f"{key}: hash mismatch (changed after freeze)")
    return errors


__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "REQUIRED_AUDIT_CHECKS",
    "file_sha256",
    "merged_valid_runs_sha256",
    "merged_audit_summary",
    "distinct_git_commits",
    "build_canonical_index",
    "validate_canonical_index",
]
