"""Tests for the canonical artifact index + validator (review §8)."""
from __future__ import annotations

import json

import pytest

from smco.canonical_artifacts import (
    build_canonical_index,
    file_sha256,
    merged_audit_summary,
    validate_canonical_index,
)


def _write_merged(tmp, *, n_rows=3, passed=True, n_checks=12, provenance=True,
                  tamper_row=None):
    import csv
    d = tmp / "merged"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"run_id": f"r{i}", "git_commit": "c0" * 20, "best_value": i}
            for i in range(n_rows)]
    if tamper_row is not None:
        rows[0]["best_value"] = tamper_row
    with open(d / "valid_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "git_commit", "best_value"])
        w.writeheader(); w.writerows(rows)
    checks = [{"name": f"c{i}", "passed": True, "n": n_rows, "errors": []}
              for i in range(max(0, n_checks))]
    if n_checks >= 12:
        checks[-1] = {"name": "provenance_complete", "passed": provenance,
                      "n": n_rows, "errors": []}
    (d / "provenance_audit.json").write_text(json.dumps(
        {"passed": passed and all(c["passed"] for c in checks),
         "failed_checks": [], "checks": checks, "n_rows": n_rows}))
    return d


def test_index_passes_valid_canonical_merged(tmp_path):
    merged = _write_merged(tmp_path / "e1", n_rows=5)
    (tmp_path / "manifest.json").write_text(json.dumps({"stage": "e1"}))
    spec = [
        {"key": "e1_merged", "path": str(merged), "kind": "merged",
         "row_count": 5, "status": "canonical"},
        {"key": "e1_manifest", "path": str(tmp_path / "manifest.json"),
         "kind": "file"},
    ]
    index = build_canonical_index(spec)
    assert index["schema_version"] == "1"
    assert validate_canonical_index(index) == []


def test_index_rejects_missing_path(tmp_path):
    spec = [{"key": "gone", "path": str(tmp_path / "nope"), "kind": "merged",
             "status": "canonical"}]
    index = build_canonical_index(spec)
    errs = validate_canonical_index(index)
    assert any("missing path" in e for e in errs)


def test_index_rejects_hash_mismatch(tmp_path):
    merged = _write_merged(tmp_path / "e2", n_rows=4)
    spec = [{"key": "e2_merged", "path": str(merged), "kind": "merged",
             "row_count": 4, "status": "canonical"}]
    index = build_canonical_index(spec)
    # tamper valid_runs.csv after freeze -> hash mismatch
    with open(merged / "valid_runs.csv", "a") as f:
        f.write("tampered\n")
    errs = validate_canonical_index(index)
    assert any("hash mismatch" in e for e in errs)


def test_index_rejects_old_11check_audit(tmp_path):
    merged = _write_merged(tmp_path / "e6", n_rows=4, n_checks=11)
    spec = [{"key": "e6_merged", "path": str(merged), "kind": "merged",
             "row_count": 4, "status": "canonical"}]
    index = build_canonical_index(spec)
    errs = validate_canonical_index(index)
    assert any("11 checks" in e or "expected 12" in e for e in errs)


def test_index_rejects_failed_provenance(tmp_path):
    merged = _write_merged(tmp_path / "e6", n_rows=4, provenance=False, passed=False)
    spec = [{"key": "e6_merged", "path": str(merged), "kind": "merged",
             "row_count": 4, "status": "canonical"}]
    index = build_canonical_index(spec)
    errs = validate_canonical_index(index)
    assert any("provenance" in e.lower() or "not passed" in e for e in errs)


def test_index_rejects_wrong_row_count(tmp_path):
    merged = _write_merged(tmp_path / "e2", n_rows=4)
    spec = [{"key": "e2_merged", "path": str(merged), "kind": "merged",
             "row_count": 120, "status": "canonical"}]  # claims 120, actually 4
    index = build_canonical_index(spec)
    errs = validate_canonical_index(index)
    assert any("rows" in e for e in errs)


def test_index_development_only_does_not_require_audit(tmp_path):
    # E4/E5 dev CSVs are development_only: only need to exist (+hash match)
    (tmp_path / "e4.csv").write_text("a,b\n1,2\n")
    spec = [{"key": "e4_dev", "path": str(tmp_path / "e4.csv"), "kind": "file",
             "status": "development_only"}]
    index = build_canonical_index(spec)
    assert validate_canonical_index(index) == []


def test_merged_audit_summary_reads_provenance(tmp_path):
    merged = _write_merged(tmp_path / "x", n_rows=7)
    s = merged_audit_summary(merged)
    assert s == {"passed": True, "n_rows": 7, "n_checks": 12, "provenance_passed": True}
