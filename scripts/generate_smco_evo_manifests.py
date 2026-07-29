#!/usr/bin/env python
"""Generate reproducible high-dimensional instances and manifest scaffolding.

Task 6 stage (``--stage instances``): materialise per-instance artifacts
(shift / permutation / block-rotation / shared starts + ``metadata.json`` with
SHA-256 of each payload) and an ``instances_index.json`` summarising them.

Task 7 will extend this same script with frozen experiment manifests (run
lists, ``run_id``, ``configuration_hash``) layered on top of these instances;
the instance artifacts produced here are the language-neutral inputs that both
the Python and R workers (Task 8) consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from smco.highdim_instances import (
    GENERATOR_VERSION,
    DEFAULT_BLOCK_SIZE,
    generate_instance,
    write_instance_artifacts,
)
from smco.experiment_manifests import (
    build_manifest,
    e1_algorithm_configs,
    expand_tasks,
    freeze_manifest,
    write_manifest,
)

# E1 development function set (Michalewicz replaced by Zakharov, 2026-07-29).
E1_FUNCTIONS = ("Rastrigin", "Ackley", "Griewank", "Zakharov")


def _starts_seed(function: str, dim: int, instance_id: int, stage: str, n_starts: int = 8) -> int:
    """Stable seed for a start matrix, decorrelated from the transform seed and across n_starts tiers."""
    key = f"{stage}:starts:{function}:{dim}:{instance_id}:{n_starts}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def _shared_starts(instance, n_starts: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    span = instance.bounds_upper - instance.bounds_lower
    return instance.bounds_lower + rng.uniform(size=(n_starts, instance.dimension)) * span


def _artifact_dir(out_dir: Path, function: str, dim: int, instance_id: int, stage: str) -> Path:
    return out_dir / "instances" / f"{stage}_{function}_d{dim}_i{instance_id}"


def _resolve_n_starts(spec, dim: int) -> int:
    if str(spec) == "sqrt":
        return int(math.ceil(math.sqrt(dim)))
    return int(spec)


def build_instance_set(
    functions,
    dims,
    n_instances: int,
    *,
    stage: str,
    out_dir,
    n_starts: int = 8,
    extra_n_starts=(),
    block_size: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Materialise instance artifacts and return the instances index.

    ``stage`` selects the instance-id/seed namespace (``development`` vs
    ``confirmatory``) so the two suites never share transforms. ``extra_n_starts``
    adds per-tier starts artifacts (entries like ``"16"`` or ``"sqrt"`` →
    ``ceil(sqrt(dim))``) for the E6.1 start-count ablation.
    """
    out_dir = Path(out_dir)
    functions = list(functions)
    dims = [int(d) for d in dims]
    total = len(functions) * len(dims) * n_instances

    if dry_run:
        return {
            "dry_run": True,
            "stage": stage,
            "generator_version": GENERATOR_VERSION,
            "functions": functions,
            "dims": dims,
            "n_instances": n_instances,
            "n_starts": n_starts,
            "extra_n_starts": list(extra_n_starts),
            "instances_planned": total,
        }

    gen_kwargs: dict = {"stage": stage}
    if block_size is not None:
        gen_kwargs["block_size"] = block_size

    entries: list[dict] = []
    for function in functions:
        for dim in dims:
            for instance_id in range(n_instances):
                instance = generate_instance(function, dim, instance_id, **gen_kwargs)
                starts = _shared_starts(
                    instance, n_starts,
                    _starts_seed(function, dim, instance_id, stage, n_starts),
                )
                extra: dict = {}
                for spec in extra_n_starts:
                    n_tier = _resolve_n_starts(spec, dim)
                    if n_tier == n_starts:
                        continue
                    extra[n_tier] = _shared_starts(
                        instance, n_tier,
                        _starts_seed(function, dim, instance_id, stage, n_tier),
                    )
                art_dir = _artifact_dir(out_dir, function, dim, instance_id, stage)
                meta = write_instance_artifacts(instance, starts, art_dir, extra_starts=extra)
                entries.append(
                    {
                        "function": function,
                        "dimension": dim,
                        "instance_id": instance_id,
                        "stage": stage,
                        "artifact_dir": str(art_dir.relative_to(out_dir)),
                        "n_starts": n_starts,
                        "known_optimum_value": meta["known_optimum_value"],
                        "transform_sha256": meta["transform_sha256"],
                        "file_hashes": meta["file_hashes"],
                        "extra_starts": meta.get("extra_starts", {}),
                    }
                )

    index = {
        "generator_version": GENERATOR_VERSION,
        "default_block_size": DEFAULT_BLOCK_SIZE,
        "stage": stage,
        "n_starts": n_starts,
        "extra_n_starts": list(extra_n_starts),
        "functions": functions,
        "dims": dims,
        "n_instances": n_instances,
        "instances": entries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "instances_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False)
    )
    return index


def load_instance_index(path):
    """Load a Task 6 instances_index.json into a lookup keyed by (function, dim, iid).

    Normalises provenance fields so :func:`expand_tasks` can attach
    ``instance_hash`` / ``start_points_hash`` from the instance artifacts.
    """
    data = json.loads(Path(path).read_text())
    index = {}
    for entry in data.get("instances", []):
        entry = dict(entry)
        hashes = entry.get("file_hashes", {})
        entry.setdefault("instance_hash", entry.get("transform_sha256"))
        entry.setdefault("start_points_hash", hashes.get("starts"))
        index[(entry["function"], entry["dimension"], entry["instance_id"])] = entry
    return index


def build_manifest_for_suite(
    *,
    stage,
    suite,
    functions,
    dims,
    n_instances,
    fe_budget_per_d,
    checkpoints_per_d,
    configs=None,
    instance_index=None,
    manifest_id=None,
    freeze=True,
    out_dir=None,
    dry_run=False,
):
    """Expand the E1-style task grid, freeze, and write a manifest.

    Links instance provenance from ``instance_index`` (a Task 6 index) when
    provided so each task carries ``instance_hash`` / ``start_points_hash``.
    """
    configs = configs if configs is not None else e1_algorithm_configs()
    tasks = expand_tasks(
        stage,
        suite,
        functions,
        dims,
        n_instances,
        configs,
        fe_budget_per_d=fe_budget_per_d,
        checkpoints_per_d=checkpoints_per_d,
        instance_index=instance_index,
    )
    if dry_run:
        return {
            "dry_run": True,
            "stage": stage,
            "suite": suite,
            "n_tasks": len(tasks),
            "unique_run_ids": len({t["run_id"] for t in tasks}),
            "total_fe_budget": sum(t["fe_budget"] for t in tasks),
            "frozen": freeze,
        }
    manifest = build_manifest(stage, suite, tasks, manifest_id=manifest_id)
    if freeze:
        manifest = freeze_manifest(manifest)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(manifest, out_dir / f"{manifest['manifest_id']}.json")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["instances", "manifest"],
        default="instances",
        help="instances: Task 6 instance artifacts; manifest: Task 7 frozen run manifest.",
    )
    parser.add_argument(
        "--suite-stage",
        default="development",
        choices=["development", "confirmatory"],
        help="Instance-id / seed namespace (dev vs confirmatory must not overlap).",
    )
    parser.add_argument(
        "--out-dir",
        default="result/smco-evo-paper-highdim-2026",
        help="Root directory for instances/ and instances_index.json.",
    )
    parser.add_argument(
        "--functions", nargs="+", default=list(E1_FUNCTIONS), help="Base function names."
    )
    parser.add_argument(
        "--dims", nargs="+", type=int, default=[200, 500, 1000], help="Dimensions."
    )
    parser.add_argument("--n-instances", type=int, default=5, help="Instances per (function, dim).")
    parser.add_argument("--n-starts", type=int, default=8, help="Shared starts per instance.")
    parser.add_argument(
        "--extra-n-starts", nargs="*", default=[],
        help="Extra start-count tiers (e.g. 16 sqrt). sqrt → ceil(sqrt(dim)).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help=f"Block-rotation size (default {DEFAULT_BLOCK_SIZE}).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report plan only; write nothing.")
    parser.add_argument(
        "--manifest-stage",
        default="e1_development",
        help="paper_contract stage for the manifest (e.g. e1_development, e2_factorial_highdim).",
    )
    parser.add_argument("--suite", default="synthetic_highdim", help="paper_contract suite.")
    parser.add_argument("--fe-budget-per-d", type=int, default=1000, help="FE budget as multiple of dimension.")
    parser.add_argument(
        "--checkpoints-per-d",
        nargs="+",
        type=int,
        default=[100, 250, 500, 1000],
        help="Checkpoints as multiples of dimension.",
    )
    parser.add_argument("--instances-index", default=None, help="Path to instances_index.json to link provenance.")
    parser.add_argument("--no-freeze", action="store_true", help="Write an unfrozen manifest.")
    args = parser.parse_args(argv)

    if args.stage == "instances":
        index = build_instance_set(
            args.functions,
            args.dims,
            args.n_instances,
            stage=args.suite_stage,
            out_dir=args.out_dir,
            n_starts=args.n_starts,
            extra_n_starts=args.extra_n_starts,
            block_size=args.block_size,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(
                f"[dry-run] would build {index['instances_planned']} instances "
                f"({len(args.functions)} funcs x {len(args.dims)} dims x {args.n_instances}) "
                f"at {args.out_dir}"
            )
        else:
            print(
                f"built {len(index['instances'])} instances at {args.out_dir} "
                f"(index: {Path(args.out_dir) / 'instances_index.json'})"
            )
        return 0

    if args.stage == "manifest":
        instance_index = (
            load_instance_index(args.instances_index) if args.instances_index else None
        )
        manifest = build_manifest_for_suite(
            stage=args.manifest_stage,
            suite=args.suite,
            functions=args.functions,
            dims=args.dims,
            n_instances=args.n_instances,
            fe_budget_per_d=args.fe_budget_per_d,
            checkpoints_per_d=args.checkpoints_per_d,
            instance_index=instance_index,
            freeze=not args.no_freeze,
            out_dir=args.out_dir,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(
                f"[dry-run] {manifest['n_tasks']} tasks, "
                f"{manifest['unique_run_ids']} unique run_ids, "
                f"total FE budget {manifest['total_fe_budget']}"
            )
        else:
            print(
                f"wrote {'frozen' if manifest['frozen'] else 'unfrozen'} manifest "
                f"{manifest['manifest_id']} ({manifest['n_tasks']} tasks) to {args.out_dir}"
            )
        return 0

    parser.error(f"stage {args.stage!r} not implemented yet")
    return 1


if __name__ == "__main__":
    sys.exit(main())
