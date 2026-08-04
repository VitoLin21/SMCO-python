#!/usr/bin/env python
"""Compose the exact E3-F/E7 instance index from audited source indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from smco.ultrahighdim_extension import (
    E3F_FUNCTIONS,
    E7_FUNCTIONS,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected(campaign: str) -> set[tuple[str, int, int]]:
    if campaign == "e3f":
        return {
            (function, dimension, instance)
            for function in E3F_FUNCTIONS for dimension in (200, 500, 1000)
            for instance in range(5)
        }
    return {
        (function, 1000, instance)
        for function in E7_FUNCTIONS for instance in range(5)
    } | {
        (function, dimension, instance)
        for function in E7_FUNCTIONS for dimension in (2000, 3000, 5000, 10000)
        for instance in range(4)
    }


def compose_instance_index(campaign: str, sources, *, repo_root) -> dict:
    repo_root = Path(repo_root).resolve()
    expected = _expected(campaign)
    candidates: dict[tuple[str, int, int], dict] = {}
    source_records = []
    for source_value in sources:
        source = Path(source_value).resolve()
        document = json.loads(source.read_text())
        source_records.append({"path": str(source), "sha256": _sha256(source)})
        for raw in document.get("instances") or []:
            key = (
                raw["function"], int(raw["dimension"]), int(raw["instance_id"]),
            )
            if key not in expected:
                continue
            entry = dict(raw)
            if entry.get("stage") not in {"confirmatory", "extension_confirmatory"}:
                raise ValueError(f"{key}: non-confirmatory stage {entry.get('stage')!r}")
            artifact = (source.parent / entry["artifact_dir"]).resolve()
            try:
                relative_artifact = artifact.relative_to(repo_root)
            except ValueError as exc:
                raise ValueError(f"{key}: artifact lies outside repo root: {artifact}") from exc
            metadata_path = artifact / "metadata.json"
            if not metadata_path.is_file():
                raise ValueError(f"{key}: missing {metadata_path}")
            metadata = json.loads(metadata_path.read_text())
            hashes = entry.get("file_hashes") or {}
            if entry.get("transform_sha256") != metadata.get("transform_sha256"):
                raise ValueError(f"{key}: transform hash differs from metadata")
            if hashes.get("starts") != (metadata.get("file_hashes") or {}).get("starts"):
                raise ValueError(f"{key}: starts hash differs from metadata")
            for label, filename in (
                ("shift", "shift.csv.gz"),
                ("permutation", "permutation.csv.gz"),
                ("rotation_blocks", "rotation_blocks.csv.gz"),
                ("starts", "starts.csv.gz"),
            ):
                path = artifact / filename
                if not path.is_file() or _sha256(path) != hashes.get(label):
                    raise ValueError(f"{key}: artifact hash mismatch for {filename}")
            entry["artifact_dir"] = str(relative_artifact)
            prior = candidates.get(key)
            identity = (
                entry.get("transform_sha256"), hashes.get("starts"),
                entry.get("artifact_dir"),
            )
            if prior is not None:
                prior_hashes = prior.get("file_hashes") or {}
                prior_identity = (
                    prior.get("transform_sha256"), prior_hashes.get("starts"),
                    prior.get("artifact_dir"),
                )
                if identity != prior_identity:
                    raise ValueError(f"{key}: conflicting source index entries")
                continue
            candidates[key] = entry
    missing = sorted(expected - set(candidates))
    if missing:
        raise ValueError(f"missing {len(missing)} required instance cells; e.g. {missing[:3]}")
    entries = [candidates[key] for key in sorted(candidates)]
    return {
        "schema_version": "1",
        "campaign": campaign,
        "stage": "extension_confirmatory",
        "repo_root": str(repo_root),
        "sources": source_records,
        "n_instances": len(entries),
        "instances": entries,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=("e3f", "e7"), required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    document = compose_instance_index(
        args.campaign, args.source, repo_root=args.repo_root,
    )
    print(json.dumps({
        "campaign": args.campaign,
        "n_instances": document["n_instances"],
        "sources": len(document["sources"]),
        "dry_run": args.dry_run,
    }, indent=2))
    if not args.dry_run:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
