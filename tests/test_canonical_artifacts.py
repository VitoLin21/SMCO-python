"""Tests for the canonical artifact index + validator (review §8 + P1 2026-08-02).

The contract is CODE (CANONICAL_CONTRACT), so a tampered index cannot drop or
downgrade artifacts: validate_canonical_index always checks the index carries
exactly the required keys with the right kind/status/row_count, a matching
frozen index_sha256, and canonical merged artifacts whose valid_runs.csv +
provenance_audit.json both still hash-match with a current 12-check passing
audit. Task 12 resolves inputs by artifact key (review P0).
"""
from __future__ import annotations

import csv
import json

import pytest

from smco.canonical_artifacts import (
    CANONICAL_CONTRACT,
    E3_COMPOSITE_KEY,
    E3_MERGED_KEY,
    build_canonical_index,
    index_sha256,
    merged_audit_summary,
    resolve_analysis_target,
    validate_canonical_index,
)


# A small test contract (the production contract is fixed in the module).
def _test_contract():
    return {
        "e2_manifest": {"kind": "file", "status": "canonical"},
        "e2_merged": {"kind": "merged", "status": "canonical", "row_count": 4},
        "e3_composite": {"kind": "file", "status": "canonical"},
        "e3_composite_merged": {"kind": "merged", "status": "canonical", "row_count": 6},
        "e4_dev": {"kind": "file", "status": "development_only"},
        "e6_schedule": {"kind": "dir", "status": "deferred"},
    }


def _write_merged(tmp, *, n_rows=3, passed=True, n_checks=12, provenance=True):
    d = tmp / "merged"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"run_id": f"r{i}", "git_commit": "c0" * 20, "best_value": i}
            for i in range(n_rows)]
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


def _build_valid_index(tmp_path, contract=None):
    contract = contract or _test_contract()
    e2m = _write_merged(tmp_path / "e2", n_rows=4)
    e3m = _write_merged(tmp_path / "e3", n_rows=6)
    (tmp_path / "e2_manifest.json").write_text(json.dumps({"stage": "e2"}))
    (tmp_path / "e3_composite.json").write_text(json.dumps({"stage": "e3"}))
    (tmp_path / "e4.csv").write_text("a,b\n1,2\n")
    (tmp_path / "schedule").mkdir()
    paths = {
        "e2_manifest": str(tmp_path / "e2_manifest.json"),
        "e2_merged": str(e2m),
        "e3_composite": str(tmp_path / "e3_composite.json"),
        "e3_composite_merged": str(e3m),
        "e4_dev": str(tmp_path / "e4.csv"),
        "e6_schedule": str(tmp_path / "schedule"),
    }
    return build_canonical_index(paths, contract=contract), paths


def test_index_passes_valid(tmp_path):
    index, _ = _build_valid_index(tmp_path)
    assert index["frozen"] is True
    assert index["index_sha256"] == index_sha256(index)
    assert validate_canonical_index(index, contract=_test_contract()) == []


# --- review P1 negative cases (all must now be rejected) ---

def test_index_rejects_empty_artifacts(tmp_path):
    index, _ = _build_valid_index(tmp_path)
    index["artifacts"] = []
    index["index_sha256"] = index_sha256(index)
    errs = validate_canonical_index(index, contract=_test_contract())
    assert any("missing required" in e for e in errs)
    assert len([e for e in errs if "missing required" in e]) == len(_test_contract())


def test_index_rejects_bad_schema(tmp_path):
    index, _ = _build_valid_index(tmp_path)
    index["schema_version"] = "9"
    index["index_sha256"] = index_sha256(index)
    errs = validate_canonical_index(index, contract=_test_contract())
    assert any("schema_version" in e for e in errs)


def test_index_rejects_status_downgrade(tmp_path):
    # E2 canonical -> development_only (and drop its hashes): must be rejected
    index, _ = _build_valid_index(tmp_path)
    e2 = next(a for a in index["artifacts"] if a["key"] == "e2_merged")
    e2["status"] = "development_only"
    e2.pop("valid_runs_sha256", None)
    e2.pop("audit_sha256", None)
    index["index_sha256"] = index_sha256(index)
    errs = validate_canonical_index(index, contract=_test_contract())
    assert any("status" in e and "contract" in e for e in errs)


def test_index_rejects_dropped_required_entry(tmp_path):
    index, _ = _build_valid_index(tmp_path)
    index["artifacts"] = [a for a in index["artifacts"] if a["key"] != "e6_schedule"]
    index["index_sha256"] = index_sha256(index)
    errs = validate_canonical_index(index, contract=_test_contract())
    assert any("missing required" in e and "e6_schedule" in e for e in errs)


def test_index_rejects_unknown_key(tmp_path):
    index, _ = _build_valid_index(tmp_path)
    index["artifacts"].append({"key": "bogus", "path": "/x", "kind": "file"})
    index["index_sha256"] = index_sha256(index)
    errs = validate_canonical_index(index, contract=_test_contract())
    assert any("unknown artifact key" in e for e in errs)


def test_index_rejects_duplicate_key(tmp_path):
    index, _ = _build_valid_index(tmp_path)
    dup = dict(index["artifacts"][0])
    index["artifacts"].append(dup)
    index["index_sha256"] = index_sha256(index)
    errs = validate_canonical_index(index, contract=_test_contract())
    assert any("duplicate artifact key" in e for e in errs)


def test_index_rejects_tampered_index_hash(tmp_path):
    index, _ = _build_valid_index(tmp_path)
    index["index_sha256"] = "0" * 64
    errs = validate_canonical_index(index, contract=_test_contract())
    assert any("index_sha256 mismatch" in e for e in errs)


def test_index_rejects_tampered_valid_runs(tmp_path):
    index, paths = _build_valid_index(tmp_path)
    with open(Path(paths["e2_merged"]) / "valid_runs.csv", "a") as f:
        f.write("tampered\n")
    errs = validate_canonical_index(index, contract=_test_contract())
    assert any("valid_runs.csv hash mismatch" in e for e in errs)


def test_index_rejects_tampered_audit(tmp_path):
    index, paths = _build_valid_index(tmp_path)
    # rewrite the audit (same content shape, different bytes via re-serialize)
    ap = Path(paths["e2_merged"]) / "provenance_audit.json"
    audit = json.loads(ap.read_text())
    audit["remarks"] = "tampered"
    ap.write_text(json.dumps(audit))
    errs = validate_canonical_index(index, contract=_test_contract())
    assert any("provenance_audit.json hash mismatch" in e for e in errs)


def test_index_rejects_old_11check_audit(tmp_path):
    contract = _test_contract()
    e2m = _write_merged(tmp_path / "e2", n_rows=4, n_checks=11)
    e3m = _write_merged(tmp_path / "e3", n_rows=6)
    paths = {"e2_manifest": str(tmp_path / "m.json"), "e2_merged": str(e2m),
             "e3_composite": str(tmp_path / "c.json"), "e3_composite_merged": str(e3m),
             "e4_dev": str(tmp_path / "e4.csv"), "e6_schedule": str(tmp_path / "sched")}
    (tmp_path / "m.json").write_text("{}"); (tmp_path / "c.json").write_text("{}")
    (tmp_path / "e4.csv").write_text("x"); (tmp_path / "sched").mkdir()
    index = build_canonical_index(paths, contract=contract)
    errs = validate_canonical_index(index, contract=contract)
    assert any("11 checks" in e or "expected 12" in e for e in errs)


def test_index_development_only_does_not_require_audit(tmp_path):
    # e4_dev is development_only: only needs to exist (+hash match)
    index, _ = _build_valid_index(tmp_path)
    assert validate_canonical_index(index, contract=_test_contract()) == []


def test_index_rejects_missing_canonical_file_hash_after_refreeze(tmp_path):
    # A new index_sha256 cannot legitimize an unhashed canonical file.
    index, _ = _build_valid_index(tmp_path)
    entry = next(a for a in index["artifacts"] if a["key"] == "e2_manifest")
    entry.pop("sha256")
    index["index_sha256"] = index_sha256(index)
    errs = validate_canonical_index(index, contract=_test_contract())
    assert any("e2_manifest" in e and "sha256" in e for e in errs)


def test_index_rejects_missing_canonical_merged_hashes_after_refreeze(tmp_path):
    # Both physical files must remain bound; merely re-freezing the index is not
    # enough when either hash field is removed.
    index, _ = _build_valid_index(tmp_path)
    entry = next(a for a in index["artifacts"] if a["key"] == "e2_merged")
    entry.pop("valid_runs_sha256")
    entry.pop("audit_sha256")
    index["index_sha256"] = index_sha256(index)
    errs = validate_canonical_index(index, contract=_test_contract())
    assert any("e2_merged" in e and "valid_runs_sha256" in e for e in errs)
    assert any("e2_merged" in e and "audit_sha256" in e for e in errs)


def test_index_rejects_invalid_canonical_file_hash_after_refreeze(tmp_path):
    index, _ = _build_valid_index(tmp_path)
    entry = next(a for a in index["artifacts"] if a["key"] == "e3_composite")
    entry["sha256"] = "not-a-sha256"
    index["index_sha256"] = index_sha256(index)
    errs = validate_canonical_index(index, contract=_test_contract())
    assert any("e3_composite" in e and "sha256" in e for e in errs)


def test_resolve_analysis_target_e3_returns_composite_path(tmp_path):
    index, _ = _build_valid_index(tmp_path)
    target = resolve_analysis_target(index, E3_MERGED_KEY)
    assert target["is_e3"] is True
    assert target["composite_path"].endswith("e3_composite.json")
    assert target["merged_dir"].endswith("merged")


def test_resolve_analysis_target_non_e3(tmp_path):
    index, _ = _build_valid_index(tmp_path)
    target = resolve_analysis_target(index, "e2_merged")
    assert target["is_e3"] is False
    assert target["composite_path"] is None


def test_production_analysis_kinds_are_contract_controlled():
    assert CANONICAL_CONTRACT["e1_merged"]["analysis_kind"] == "selection_matrix"
    assert CANONICAL_CONTRACT["e2_merged"]["analysis_kind"] == "winner_vs_base"
    assert CANONICAL_CONTRACT[E3_MERGED_KEY]["analysis_kind"] == "comparative"
    assert CANONICAL_CONTRACT["e6_strategy_merged"]["analysis_kind"] == "strategy_ablation"
    assert CANONICAL_CONTRACT["e6_start_count_merged"]["analysis_kind"] == "start_count_ablation"


def test_resolve_analysis_target_rejects_unknown_key(tmp_path):
    index, _ = _build_valid_index(tmp_path)
    with pytest.raises(ValueError, match="unknown artifact key"):
        resolve_analysis_target(index, "bogus")


def test_production_contract_has_all_required_keys():
    # guard: the fixed production contract is exactly the paper's artifact set
    expected = {
        "e1_merged", "e1_selection", "e2_manifest", "e2_merged",
        "e3_baseline_component_manifest", "e3_baseline_merged", "e3_composite",
        "e3_composite_merged", "e6_strategy_merged", "e6_start_count_manifest",
        "e6_start_count_merged", "e4_dev", "e5_dev", "e6_schedule",
    }
    assert set(CANONICAL_CONTRACT) == expected
    # E3 keys present
    assert CANONICAL_CONTRACT[E3_MERGED_KEY]["row_count"] == 420
    assert CANONICAL_CONTRACT[E3_COMPOSITE_KEY]["kind"] == "file"


from pathlib import Path  # noqa: E402  (used by tamper tests above)
