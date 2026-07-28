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
import sys
from pathlib import Path

import numpy as np

from smco.highdim_instances import (
    GENERATOR_VERSION,
    DEFAULT_BLOCK_SIZE,
    generate_instance,
    instance_seed,
    write_instance_artifacts,
)

# E1 development function set (Michalewicz replaced by Zakharov, 2026-07-29).
E1_FUNCTIONS = ("Rastrigin", "Ackley", "Griewank", "Zakharov")


def _starts_seed(function: str, dim: int, instance_id: int, stage: str) -> int:
    """Stable seed for the shared start matrix, decorrelated from the transform seed."""
    key = f"{stage}:starts:{function}:{dim}:{instance_id}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def _shared_starts(instance, n_starts: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    span = instance.bounds_upper - instance.bounds_lower
    return instance.bounds_lower + rng.uniform(size=(n_starts, instance.dimension)) * span


def _artifact_dir(out_dir: Path, function: str, dim: int, instance_id: int, stage: str) -> Path:
    return out_dir / "instances" / f"{stage}_{function}_d{dim}_i{instance_id}"


def build_instance_set(
    functions,
    dims,
    n_instances: int,
    *,
    stage: str,
    out_dir,
    n_starts: int = 8,
    block_size: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Materialise instance artifacts and return the instances index.

    ``stage`` selects the instance-id/seed namespace (``development`` vs
    ``confirmatory``) so the two suites never share transforms.
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
                    instance, n_starts, _starts_seed(function, dim, instance_id, stage)
                )
                art_dir = _artifact_dir(out_dir, function, dim, instance_id, stage)
                meta = write_instance_artifacts(instance, starts, art_dir)
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
                    }
                )

    index = {
        "generator_version": GENERATOR_VERSION,
        "default_block_size": DEFAULT_BLOCK_SIZE,
        "stage": stage,
        "n_starts": n_starts,
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["instances"],
        default="instances",
        help="Task 6 only implements instance generation; manifests arrive in Task 7.",
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
        "--block-size",
        type=int,
        default=None,
        help=f"Block-rotation size (default {DEFAULT_BLOCK_SIZE}).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report plan only; write nothing.")
    args = parser.parse_args(argv)

    if args.stage == "instances":
        index = build_instance_set(
            args.functions,
            args.dims,
            args.n_instances,
            stage=args.suite_stage,
            out_dir=args.out_dir,
            n_starts=args.n_starts,
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

    parser.error(f"stage {args.stage!r} not implemented yet")
    return 1


if __name__ == "__main__":
    sys.exit(main())
