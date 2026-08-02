"""Frozen, isolated evidence indexes for formal COCO external checks.

The high-dimensional paper's primary canonical index is intentionally closed:
adding future E4/E5 entries to it would invalidate an already frozen result.
This module instead defines one *separate*, fixed contract for each external
campaign.  A formal external index is only valid after its own manifest,
merged audit and COCO-native sidecar all exist and are mutually hash-bound.

These artifacts are supporting external evidence.  Their ``coco_native_external``
analysis kind must never enter the synthetic normalized-gap Task-12 pipeline.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .canonical_artifacts import (
    CANONICAL_SCHEMA_VERSION,
    REQUIRED_AUDIT_CHECKS,
    _is_sha256,
    _resolve,
    build_canonical_index,
    file_sha256,
    index_sha256,
    validate_canonical_index,
)
from .experiment_manifests import load_manifest, verify_manifest


E4_FORMAL_CONTRACT = {
    "e4_formal_manifest": {"kind": "file", "status": "canonical"},
    "e4_formal_merged": {
        "kind": "merged", "status": "canonical", "row_count": 2520,
        "analysis_kind": "coco_native_external",
        "source_manifest_key": "e4_formal_manifest",
        "expected_stage": "e4_bbob_largescale",
        "expected_suite": "bbob-largescale",
    },
}

E5_FORMAL_CONTRACT = {
    "e5_formal_manifest": {"kind": "file", "status": "canonical"},
    "e5_formal_merged": {
        "kind": "merged", "status": "canonical", "row_count": 480,
        "analysis_kind": "coco_native_external",
        "source_manifest_key": "e5_formal_manifest",
        "expected_stage": "e5_lowdim_check",
        "expected_suite": "bbob",
    },
}

FORMAL_EXTERNAL_CONTRACTS = {"e4": E4_FORMAL_CONTRACT, "e5": E5_FORMAL_CONTRACT}


def external_contract(campaign: str) -> dict:
    """Return the code-owned contract for one external campaign."""
    try:
        return FORMAL_EXTERNAL_CONTRACTS[campaign]
    except KeyError as exc:
        raise ValueError(f"unknown formal external campaign {campaign!r}; expected e4 or e5") from exc


def build_formal_external_index(paths: dict, *, campaign: str, root=".", git_commit=None) -> dict:
    """Freeze a campaign-local external index after all formal artifacts exist.

    The generic index records byte hashes for manifest, ``valid_runs.csv`` and
    audit.  We add a hash for ``coco_native_runs.csv`` and duplicate the source
    manifest hash on the merged entry to make the relationship explicit.
    """
    contract = external_contract(campaign)
    index = build_canonical_index(paths, root=root, git_commit=git_commit, contract=contract)
    index["external_campaign"] = campaign
    index["evidence_scope"] = "external_supporting"
    by_key = {entry["key"]: entry for entry in index["artifacts"]}
    for key, spec in contract.items():
        if spec["kind"] != "merged":
            continue
        entry = by_key[key]
        source = by_key[spec["source_manifest_key"]]
        entry["source_manifest_sha256"] = source.get("sha256")
        merged_dir = _resolve(entry.get("path"), root) if entry.get("path") else None
        sidecar = merged_dir / "coco_native_runs.csv" if merged_dir else None
        if sidecar and sidecar.exists():
            entry["coco_native_runs_sha256"] = file_sha256(sidecar)
    index["index_sha256"] = index_sha256(index)
    return index


def _artifact(index: dict, key: str) -> dict | None:
    return next((entry for entry in index.get("artifacts", []) if entry.get("key") == key), None)


def _native_sidecar_errors(merged_dir: Path, entry: dict, *, expected_rows: int,
                           expected_run_ids: set[str]) -> list[str]:
    errors: list[str] = []
    sidecar = merged_dir / "coco_native_runs.csv"
    if not sidecar.exists():
        return ["missing coco_native_runs.csv"]
    expected_hash = entry.get("coco_native_runs_sha256")
    if not _is_sha256(expected_hash):
        errors.append("missing or invalid coco_native_runs_sha256")
    elif file_sha256(sidecar) != expected_hash:
        errors.append("coco_native_runs.csv hash mismatch (changed after freeze)")
    with open(sidecar, newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row.get("run_id") for row in rows]
    if len(rows) != expected_rows:
        errors.append(f"coco_native_runs.csv rows {len(rows)} != contract {expected_rows}")
    if len(ids) != len(set(ids)) or any(not run_id for run_id in ids):
        errors.append("coco_native_runs.csv has duplicate or empty run_id")
    if set(ids) != expected_run_ids:
        errors.append("coco_native_runs.csv run_ids do not exactly match valid_runs.csv")
    modes = {row.get("metric_mode") for row in rows}
    if modes != {"coco_native"}:
        errors.append(f"coco_native_runs.csv metric_mode {sorted(modes)} != ['coco_native']")
    return errors


def validate_formal_external_index(index: dict, *, root=".") -> list[str]:
    """Validate an external campaign index and its manifest/audit/sidecar binding."""
    errors: list[str] = []
    campaign = index.get("external_campaign")
    try:
        contract = external_contract(campaign)
    except ValueError as exc:
        return [str(exc)]
    if index.get("evidence_scope") != "external_supporting":
        errors.append("evidence_scope must be 'external_supporting'")
    errors.extend(validate_canonical_index(index, root=root, contract=contract))
    for key, spec in contract.items():
        if spec["kind"] != "merged":
            continue
        entry = _artifact(index, key)
        source = _artifact(index, spec["source_manifest_key"])
        if entry is None or source is None:
            continue
        manifest_path = _resolve(source.get("path"), root) if source.get("path") else None
        merged_dir = _resolve(entry.get("path"), root) if entry.get("path") else None
        if not manifest_path or not manifest_path.exists() or not merged_dir or not merged_dir.exists():
            continue
        try:
            manifest = load_manifest(manifest_path)
            verify_manifest(manifest)
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{spec['source_manifest_key']}: invalid frozen manifest: {exc}")
            continue
        if manifest.get("frozen") is not True:
            errors.append(f"{spec['source_manifest_key']}: manifest is not frozen")
        if manifest.get("stage") != spec["expected_stage"]:
            errors.append(f"{spec['source_manifest_key']}: stage {manifest.get('stage')!r} != {spec['expected_stage']!r}")
        if manifest.get("suite") != spec["expected_suite"]:
            errors.append(f"{spec['source_manifest_key']}: suite {manifest.get('suite')!r} != {spec['expected_suite']!r}")
        if entry.get("source_manifest_sha256") != source.get("sha256"):
            errors.append(f"{key}: source_manifest_sha256 does not bind indexed manifest")
        audit_path = merged_dir / "provenance_audit.json"
        try:
            audit = json.loads(audit_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            errors.append(f"{key}: cannot read provenance audit for manifest binding: {exc}")
            continue
        if audit.get("manifest_sha256") != manifest.get("manifest_sha256"):
            errors.append(f"{key}: audit manifest_sha256 does not match frozen manifest")
        if len(audit.get("checks", [])) != REQUIRED_AUDIT_CHECKS:
            errors.append(f"{key}: audit has noncanonical check count")
        with open(merged_dir / "valid_runs.csv", newline="") as handle:
            valid_run_ids = {row.get("run_id") for row in csv.DictReader(handle)}
        manifest_run_ids = {task.get("run_id") for task in manifest.get("tasks", [])}
        if valid_run_ids != manifest_run_ids:
            errors.append(f"{key}: valid_runs.csv run_ids do not exactly match frozen manifest")
        errors.extend(f"{key}: {err}" for err in _native_sidecar_errors(
            merged_dir, entry, expected_rows=spec["row_count"],
            expected_run_ids=valid_run_ids))
    return errors


def resolve_external_analysis_target(index: dict, artifact_key: str, *, root=".") -> dict:
    """Resolve only a validated COCO-native external analysis target."""
    errors = validate_formal_external_index(index, root=root)
    if errors:
        raise ValueError("formal external index invalid:\n  " + "\n  ".join(errors))
    contract = external_contract(index["external_campaign"])
    spec = contract.get(artifact_key)
    if spec is None:
        raise ValueError(f"unknown external artifact key {artifact_key!r}")
    if spec.get("analysis_kind") != "coco_native_external":
        raise ValueError(f"artifact {artifact_key!r} is not a COCO-native analysis target")
    entry = _artifact(index, artifact_key)
    merged_dir = _resolve(entry["path"], root)
    return {
        "key": artifact_key,
        "analysis_kind": "coco_native_external",
        "merged_dir": str(merged_dir),
        "native_runs_path": str(merged_dir / "coco_native_runs.csv"),
        "campaign": index["external_campaign"],
    }


__all__ = [
    "E4_FORMAL_CONTRACT", "E5_FORMAL_CONTRACT", "FORMAL_EXTERNAL_CONTRACTS",
    "external_contract", "build_formal_external_index", "validate_formal_external_index",
    "resolve_external_analysis_target",
]
