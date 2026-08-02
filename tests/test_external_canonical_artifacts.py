"""Formal COCO external indexes are isolated from the primary paper index."""
from __future__ import annotations

import csv
import json

import pytest

from smco.coco_external_analysis import load_coco_native_runs, native_summary, write_coco_native_report
from smco.experiment_manifests import build_manifest, freeze_manifest, manifest_sha256
from smco.external_canonical_artifacts import (
    build_formal_external_index,
    resolve_external_analysis_target,
    validate_formal_external_index,
)


def _write_e5_external(tmp_path):
    n = 480
    tasks = [{"run_id": f"r{i:04d}"} for i in range(n)]
    manifest = freeze_manifest(build_manifest("e5_lowdim_check", "bbob", tasks))
    manifest_path = tmp_path / "e5_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    merged = tmp_path / "merged"
    merged.mkdir()
    runs = [{"run_id": task["run_id"], "git_commit": "a" * 40}
            for task in tasks]
    with open(merged / "valid_runs.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "git_commit"])
        writer.writeheader(); writer.writerows(runs)
    checks = [{"name": f"check_{i}", "passed": True, "n": n, "errors": []}
              for i in range(11)]
    checks.append({"name": "provenance_complete", "passed": True, "n": n, "errors": []})
    (merged / "provenance_audit.json").write_text(json.dumps({
        "passed": True, "failed_checks": [], "checks": checks, "n_rows": n,
        "manifest_sha256": manifest_sha256(manifest),
        "manifest_sha256s": [manifest_sha256(manifest)],
    }))
    fields = ["run_id", "algorithm_id", "function", "dimension", "instance", "status",
              "fe_used", "fe_budget", "final_target_hit", "best_observed_fvalue1",
              "evaluations", "suite", "problem_id", "metric_mode", "cocoex_version",
              "cocopp_version", "ran_language", "is_frozen_winner_validation",
              "external_check_kind"]
    native = []
    for i, task in enumerate(tasks):
        native.append({"run_id": task["run_id"], "algorithm_id": "PY-SP-SMCO-EVO" if i % 2 else "PY-BASE-SMCO",
                       "function": "f1", "dimension": "5", "instance": str(i % 5),
                       "status": "success", "fe_used": "10", "fe_budget": "10000",
                       "final_target_hit": "false", "best_observed_fvalue1": "1.0",
                       "evaluations": "10", "suite": "bbob", "problem_id": f"p{i}",
                       "metric_mode": "coco_native", "cocoex_version": "x", "cocopp_version": "",
                       "ran_language": "python", "is_frozen_winner_validation": "true",
                       "external_check_kind": "frozen_winner"})
    with open(merged / "coco_native_runs.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(native)
    index = build_formal_external_index(
        {"e5_formal_manifest": str(manifest_path), "e5_formal_merged": str(merged)},
        campaign="e5")
    return index, manifest_path, merged


def test_external_e5_index_binds_manifest_audit_and_native_result(tmp_path):
    index, _manifest, merged = _write_e5_external(tmp_path)
    assert validate_formal_external_index(index) == []
    target = resolve_external_analysis_target(index, "e5_formal_merged")
    assert target["analysis_kind"] == "coco_native_external"
    assert target["native_runs_path"] == str(merged / "coco_native_runs.csv")


def test_external_index_rejects_audit_from_another_manifest(tmp_path):
    index, _manifest, merged = _write_e5_external(tmp_path)
    audit_path = merged / "provenance_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["manifest_sha256"] = "0" * 64
    audit_path.write_text(json.dumps(audit))
    errors = validate_formal_external_index(index)
    assert any("audit manifest_sha256" in error for error in errors)


def test_external_index_rejects_non_native_sidecar_after_refreeze(tmp_path):
    index, _manifest, merged = _write_e5_external(tmp_path)
    path = merged / "coco_native_runs.csv"
    rows = list(csv.DictReader(open(path)))
    rows[0]["metric_mode"] = "derived_relative"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    errors = validate_formal_external_index(index)
    assert any("coco_native_runs.csv hash mismatch" in error for error in errors)


def test_native_analysis_writes_only_native_summary(tmp_path):
    index, _manifest, _merged = _write_e5_external(tmp_path)
    index_path = tmp_path / "external_index.json"
    index_path.write_text(json.dumps(index))
    result = write_coco_native_report(index_path, "e5_formal_merged", tmp_path / "analysis")
    assert {row["algorithm_id"] for row in result} == {"PY-BASE-SMCO", "PY-SP-SMCO-EVO"}
    assert (tmp_path / "analysis" / "coco_native_summary.csv").exists()


def test_native_analysis_rejects_derived_relative_rows(tmp_path):
    path = tmp_path / "native.csv"
    path.write_text("algorithm_id,metric_mode\nA,derived_relative\n")
    with pytest.raises(ValueError, match="metric_mode='coco_native'"):
        load_coco_native_runs(path)
