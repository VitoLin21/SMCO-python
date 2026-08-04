#!/usr/bin/env python
"""Build, validate, shard, dispatch and canonicalize E3-F/E7 extensions.

This command never treats the frozen 72-hour operational deadline as a process
timeout.  ``dispatch`` supervises every selected task until its worker reaches
the FE budget, terminates normally, or reports an infrastructure failure.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from smco.provenance import (
    default_environment_hash,
    default_git_commit,
    require_confirmatory_provenance,
)
from smco.ultrahighdim_extension import (
    E3F_FUNCTIONS,
    E7_FUNCTIONS,
    build_e3f_manifest,
    build_e7_manifest,
    build_extension_composite,
    build_extension_index,
    build_shards,
    merge_extension_results,
    plan_extension_dispatch,
    recover_stalled_attempt,
    supervise_command,
    validate_e3f_manifest,
    validate_e7_manifest,
    validate_extension_composite,
    validate_extension_index,
    validate_shards,
)

_THIS_SCRIPT = Path(__file__).resolve()
if str(_THIS_SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_SCRIPT.parent))


def _write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _load_instance_index(path) -> dict:
    data = json.loads(Path(path).read_text())
    index = {}
    for source in data.get("instances") or []:
        entry = dict(source)
        hashes = entry.get("file_hashes") or {}
        entry.setdefault("transform_sha256", entry.get("instance_hash"))
        entry.setdefault("start_points_hash", hashes.get("starts"))
        index[(entry["function"], int(entry["dimension"]), int(entry["instance_id"]))] = entry
    return index


def _load_manifest(path):
    return json.loads(Path(path).read_text())


def _manifest_errors(manifest):
    return (validate_e3f_manifest(manifest) if manifest.get("campaign") == "e3f"
            else validate_e7_manifest(manifest))


def _selected_tasks(manifest, shard_path=None, shard_id=None):
    if shard_path is None:
        return list(manifest["tasks"])
    shards = json.loads(Path(shard_path).read_text())
    errors = validate_shards(shards, manifest)
    if errors:
        raise ValueError("invalid shard document: " + "; ".join(errors))
    if not shard_id:
        raise ValueError("--shard-id is required with --shards")
    shard = next((item for item in shards["shards"] if item["shard_id"] == shard_id), None)
    if shard is None:
        raise ValueError(f"unknown shard_id {shard_id!r}")
    return list(shard["tasks"])


def _worker_command(
    task_path: Path, instance_root: str, *, machine_id: str,
    git_commit: str, environment_hash: str,
) -> list[str]:
    return [
        sys.executable, str(_THIS_SCRIPT), "worker",
        "--task", str(task_path), "--instance-root", str(instance_root),
        "--machine-id", machine_id, "--git-commit", git_commit,
        "--environment-hash", environment_hash,
    ]


def _dispatch_task(task, *, instance_root, evidence_root, machine_id,
                   git_commit, environment_hash):
    cache = Path(evidence_root) / "_task_cache"
    cache.mkdir(parents=True, exist_ok=True)
    task_path = cache / f"{task['run_id']}.task.json"
    task_path.write_text(json.dumps(task, indent=2, sort_keys=True))
    return supervise_command(
        _worker_command(
            task_path, instance_root, machine_id=machine_id,
            git_commit=git_commit, environment_hash=environment_hash,
        ), run_id=task["run_id"],
        evidence_root=evidence_root, machine_id=machine_id,
        git_commit=git_commit, environment_hash=environment_hash,
    )


def _run_worker(args) -> int:
    """Adapter from the existing workers to the supervisor env-file contract."""
    task = json.loads(Path(args.task).read_text())
    result_path = Path(os.environ["SMCO_RESULT_PATH"])
    result_dir = result_path.parent
    result_dir.mkdir(parents=True, exist_ok=True)
    if "algorithm" in task:
        from run_smco_evo_highdim_baselines import run_baseline_file

        code = run_baseline_file(
            args.task, instance_root=args.instance_root, result_dir=result_dir,
            machine_id=args.machine_id or socket.gethostname(),
            git_commit=args.git_commit or default_git_commit(),
            environment_hash=args.environment_hash or default_environment_hash(),
        )
    else:
        from run_smco_evo_highdim_factorial import run_task_file

        code = run_task_file(
            args.task, instance_root=args.instance_root, result_dir=result_dir,
            machine_id=args.machine_id or socket.gethostname(),
            git_commit=args.git_commit or default_git_commit(),
            environment_hash=args.environment_hash or default_environment_hash(),
        )
    produced = result_dir / f"{task['run_id']}.json"
    if produced.exists():
        result_path.write_bytes(produced.read_bytes())
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    manifest_parser = sub.add_parser("manifest", help="Build exact E3-F/E7 physical manifest.")
    manifest_parser.add_argument("--campaign", choices=("e3f", "e7"), required=True)
    manifest_parser.add_argument("--selection", required=True)
    manifest_parser.add_argument("--instances-index", required=True)
    manifest_parser.add_argument("--out", required=True)
    manifest_parser.add_argument("--dry-run", action="store_true")
    manifest_parser.add_argument("--validate-only", action="store_true")

    shard_parser = sub.add_parser("shard", help="Deterministic greedy problem-bundle shards.")
    shard_parser.add_argument("--manifest", required=True)
    shard_parser.add_argument("--n-shards", type=int, required=True)
    shard_parser.add_argument("--cost-estimates", default=None)
    shard_parser.add_argument("--out", required=True)
    shard_parser.add_argument("--dry-run", action="store_true")
    shard_parser.add_argument("--validate-only", action="store_true")

    dispatch_parser = sub.add_parser("dispatch", help="Run/resume a manifest or one shard.")
    dispatch_parser.add_argument("--manifest", required=True)
    dispatch_parser.add_argument("--instance-root", required=True)
    dispatch_parser.add_argument("--evidence-root", required=True)
    dispatch_parser.add_argument("--shards", default=None)
    dispatch_parser.add_argument("--shard-id", default=None)
    dispatch_parser.add_argument("--workers", type=int, default=1)
    dispatch_parser.add_argument("--no-resume", action="store_true")
    dispatch_parser.add_argument("--dry-run", action="store_true")
    dispatch_parser.add_argument("--validate-only", action="store_true")
    dispatch_parser.add_argument("--machine-id", default=None)
    dispatch_parser.add_argument("--git-commit", default=None)
    dispatch_parser.add_argument("--environment-hash", default=None)

    worker_parser = sub.add_parser("worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--task", required=True)
    worker_parser.add_argument("--instance-root", required=True)
    worker_parser.add_argument("--machine-id", default=None)
    worker_parser.add_argument("--git-commit", default=None)
    worker_parser.add_argument("--environment-hash", default=None)

    merge_parser = sub.add_parser("merge", help="Merge attempts and run extension audits.")
    merge_parser.add_argument("--manifest", required=True)
    merge_parser.add_argument("--evidence-root", required=True)
    merge_parser.add_argument("--out-dir", required=True)
    merge_parser.add_argument("--validate-only", action="store_true")

    composite_parser = sub.add_parser("composite", help="Build 840/2016-row logical composite.")
    composite_parser.add_argument("--campaign", choices=("e3f", "e7"), required=True)
    composite_parser.add_argument("--selection-hash", required=True)
    composite_parser.add_argument("--original-e3-valid", default=None)
    composite_parser.add_argument("--e3f-valid", default=None)
    composite_parser.add_argument("--e3f-manifest", default=None)
    composite_parser.add_argument("--e3f-audit", default=None)
    composite_parser.add_argument("--e3-combined-valid", default=None)
    composite_parser.add_argument("--e7-new-valid", default=None)
    composite_parser.add_argument("--e7-manifest", default=None)
    composite_parser.add_argument("--e7-audit", default=None)
    composite_parser.add_argument("--source-document", action="append", default=[])
    composite_parser.add_argument("--materialized-out", required=True)
    composite_parser.add_argument("--out", required=True)
    composite_parser.add_argument("--validate-only", action="store_true")

    index_parser = sub.add_parser("index", help="Build isolated canonical extension index.")
    index_parser.add_argument("--campaign", choices=("e3f", "e7"), required=True)
    index_parser.add_argument("--manifest", required=True)
    index_parser.add_argument("--merged-dir", required=True)
    index_parser.add_argument("--composite", required=True)
    index_parser.add_argument("--root", default=".")
    index_parser.add_argument("--out", required=True)
    index_parser.add_argument("--validate-only", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "worker":
        return _run_worker(args)

    if args.command == "manifest":
        selection = json.loads(Path(args.selection).read_text())
        instance_index = _load_instance_index(args.instances_index)
        manifest = (build_e3f_manifest(selection, instance_index=instance_index)
                    if args.campaign == "e3f"
                    else build_e7_manifest(selection, instance_index=instance_index))
        errors = _manifest_errors(manifest)
        summary = {
            "campaign": args.campaign, "n_tasks": manifest["n_tasks"],
            "manifest_sha256": manifest["manifest_sha256"], "errors": errors,
            "dry_run": bool(args.dry_run),
        }
        if not args.dry_run and not args.validate_only:
            _write_json(args.out, manifest)
        print(json.dumps(summary, indent=2))
        return 0 if not errors else 2

    if args.command == "shard":
        manifest = _load_manifest(args.manifest)
        estimates = json.loads(Path(args.cost_estimates).read_text()) if args.cost_estimates else None
        shards = build_shards(manifest, n_shards=args.n_shards, cost_estimates=estimates)
        errors = validate_shards(shards, manifest)
        summary = {
            "n_shards": len(shards["shards"]),
            "n_tasks": sum(item["n_tasks"] for item in shards["shards"]),
            "shard_sha256": shards["shard_sha256"], "errors": errors,
            "dry_run": bool(args.dry_run),
        }
        if not args.dry_run and not args.validate_only:
            _write_json(args.out, shards)
        print(json.dumps(summary, indent=2))
        return 0 if not errors else 2

    if args.command == "dispatch":
        manifest = _load_manifest(args.manifest)
        errors = _manifest_errors(manifest)
        if errors:
            raise ValueError("manifest invalid: " + "; ".join(errors))
        tasks = _selected_tasks(manifest, args.shards, args.shard_id)
        plan = plan_extension_dispatch(tasks, args.evidence_root)
        plan["operational_deadline_hours"] = 72
        plan["deadline_is_kill_threshold"] = False
        if args.dry_run or args.validate_only:
            print(json.dumps(plan, indent=2))
            return 0
        if args.no_resume:
            todo = tasks
        else:
            for run_id in plan["run_ids"]["stalled"]:
                recover_stalled_attempt(args.evidence_root, run_id)
            wanted = set(
                plan["run_ids"]["pending"] + plan["run_ids"]["retryable"]
                + plan["run_ids"]["stalled"]
            )
            todo = [task for task in tasks if task["run_id"] in wanted]
        metadata = {
            "machine_id": args.machine_id or socket.gethostname(),
            "git_commit": args.git_commit or default_git_commit(),
            "environment_hash": args.environment_hash or default_environment_hash(),
        }
        require_confirmatory_provenance(**metadata)
        outcomes = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    _dispatch_task, task, instance_root=args.instance_root,
                    evidence_root=args.evidence_root, **metadata,
                ): task for task in todo
            }
            for future in as_completed(futures):
                outcomes.append(future.result())
        print(json.dumps({
            **plan, "dispatched": len(todo),
            "terminal_statuses": {status: sum(o.get("status") == status for o in outcomes)
                                  for status in {o.get("status") for o in outcomes}},
        }, indent=2))
        return 0

    if args.command == "merge":
        if args.validate_only:
            manifest = _load_manifest(args.manifest)
            print(json.dumps(plan_extension_dispatch(manifest["tasks"], args.evidence_root), indent=2))
            return 0
        summary = merge_extension_results(args.manifest, args.evidence_root, args.out_dir)
        print(json.dumps(summary, indent=2))
        return 0 if summary["audit"]["passed"] else 2

    if args.command == "composite":
        if not args.source_document:
            parser.error(
                "formal composite requires --source-document for the frozen upstream "
                "canonical/composite index"
            )
        upstream_values = [json.loads(Path(path).read_text()) for path in args.source_document]
        if args.campaign == "e3f":
            from smco.confirmatory import (
                validate_composite as validate_legacy_e3_composite,
                validate_e3_merged_against_composite,
            )

            matching = [value for value in upstream_values
                        if value.get("composite_type") == "comparative_composite"]
            if len(matching) != 1:
                parser.error("e3f requires exactly one legacy frozen E3 comparative composite")
            upstream_errors = validate_legacy_e3_composite(matching[0])
            if upstream_errors:
                raise ValueError("legacy E3 composite invalid: " + "; ".join(upstream_errors))
            if matching[0].get("selection_hash") != args.selection_hash:
                raise ValueError("legacy E3 composite selection_hash mismatch")
            merged_errors = validate_e3_merged_against_composite(
                matching[0], Path(args.original_e3_valid).parent,
            )
            if merged_errors:
                raise ValueError(
                    "--original-e3-valid is not the validated legacy composite merge: "
                    + "; ".join(merged_errors)
                )
        else:
            matching = [value for value in upstream_values
                        if value.get("composite_type") == "extension_logical_composite"
                        and value.get("campaign") == "e3f"]
            if len(matching) != 1:
                parser.error("e7 requires exactly one frozen E3+E3-F extension composite")
            upstream_errors = validate_extension_composite(matching[0])
            if upstream_errors:
                raise ValueError("E3-F extension composite invalid: " + "; ".join(upstream_errors))
            if matching[0].get("selection_hash") != args.selection_hash:
                raise ValueError("E3-F composite selection_hash mismatch")
            if Path(matching[0].get("materialized_valid_runs_path", "")).resolve() != Path(
                args.e3_combined_valid
            ).resolve():
                raise ValueError(
                    "--e3-combined-valid is not the materialized CSV bound by E3-F composite"
                )
        if args.campaign == "e3f":
            if not all((args.original_e3_valid, args.e3f_valid,
                        args.e3f_manifest, args.e3f_audit)):
                parser.error(
                    "e3f composite requires --original-e3-valid, --e3f-valid, "
                    "--e3f-manifest and --e3f-audit"
                )
            sources = [
                {"role": "original_e3", "valid_runs_path": args.original_e3_valid},
                {"role": "e3f", "valid_runs_path": args.e3f_valid,
                 "manifest_path": args.e3f_manifest, "audit_path": args.e3f_audit},
            ]
        else:
            if not all((args.e3_combined_valid, args.e7_new_valid,
                        args.e7_manifest, args.e7_audit)):
                parser.error(
                    "e7 composite requires --e3-combined-valid, --e7-new-valid, "
                    "--e7-manifest and --e7-audit"
                )
            sources = [
                {"role": "reused_d1000", "valid_runs_path": args.e3_combined_valid,
                 "filter": {"dimension": 1000}},
                {"role": "physically_new", "valid_runs_path": args.e7_new_valid,
                 "manifest_path": args.e7_manifest, "audit_path": args.e7_audit},
            ]
        composite = build_extension_composite(
            args.campaign, sources=sources, selection_hash=args.selection_hash,
            source_documents=args.source_document, output_csv=args.materialized_out,
        )
        errors = validate_extension_composite(composite)
        if not args.validate_only:
            _write_json(args.out, composite)
        print(json.dumps({"total_rows": composite["total_rows"], "errors": errors}, indent=2))
        return 0 if not errors else 2

    if args.command == "index":
        index = build_extension_index(
            args.campaign, manifest_path=args.manifest, merged_dir=args.merged_dir,
            composite_path=args.composite, root=args.root,
            git_commit=default_git_commit(),
        )
        errors = validate_extension_index(index, root=args.root)
        if not args.validate_only:
            _write_json(args.out, index)
        print(json.dumps({"campaign": args.campaign, "errors": errors}, indent=2))
        return 0 if not errors else 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
