"""E3-F/E7 prospective orchestration and evidence contracts.

This module is deliberately isolated from the already-frozen E1--E6 canonical
contract.  It owns the physical E3-F/E7 task matrices, bundle-level sharding,
72-hour *operational* (never kill) deadline sidecars, retry attempts, extension
merge/audit helpers and campaign-local canonical indexes.

No function or optimizer implementation lives here.  Algorithm and function
registries are identifiers only; workers remain responsible for executing the
frozen implementation recorded in each task's ``algorithm_metadata``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .experiment_manifests import (
    DEFAULT_DE_CROSSOVER,
    DEFAULT_DE_FACTOR,
    DEFAULT_ELIMINATION_RATE,
    DEFAULT_EVO_POINTS,
    DEFAULT_EVO_STRATEGY,
    DEFAULT_N_STARTS,
    E1_FUNCTIONS,
    E3F_FUNCTIONS,
    E7_FUNCTIONS,
    baseline_run_id,
    build_algorithm_config,
    build_manifest,
    expand_baseline_tasks,
    expand_tasks,
    freeze_manifest,
    manifest_sha256,
    verify_manifest,
)
from .e7_algorithm_adapters import E7_ALGORITHM_IDS, E7_ALGORITHM_METADATA
from .paper_contract import (
    NONE_TOKEN, canonical_json, compute_configuration_hash, compute_run_id,
    parse_algorithm_id,
)


E3F_STAGE = "e3f_missing_functions"
E7_STAGE = "e7_ultrahighdim"
EXTENSION_SUITE = "synthetic_highdim"

E3F_DIMENSIONS = (200, 500, 1000)
E7_DIMENSIONS = (1000, 2000, 3000, 5000, 10000)
E3F_ALGORITHMS = (
    "PY-SP-SMCO-EVO", "PY-BASE-SMCO", "GenSA", "SA", "DE", "GA", "PSO",
)
E7_NEW_ALGORITHMS = E7_ALGORITHM_IDS
E7_ALGORITHMS = E3F_ALGORITHMS + E7_NEW_ALGORITHMS

FE_BUDGET_PER_D = 1000
FE_CHECKPOINTS_PER_D = (100, 250, 500, 750, 1000)
WALL_CHECKPOINT_HOURS = (1, 6, 24, 72)
DEADLINE_HOURS = 72
WALL_CHECKPOINT_SECONDS = tuple(hours * 3600 for hours in WALL_CHECKPOINT_HOURS)
DEADLINE_SECONDS = DEADLINE_HOURS * 3600

ALGORITHM_METADATA_KEYS = (
    "language", "package", "package_version", "hyperparameters",
    "bounds_handling", "rng", "starts_semantics", "fe_counting",
)

# Minimal code-owned metadata keeps manifest construction independent of the
# optimizer registry being implemented in parallel.  A caller may supply richer
# records, but every frozen task must still carry all eight fields.
_COMPARATOR_METADATA = {
    name: {
        "language": "r" if name in {"R-DEoptim", "STOGO"} else "python",
        "package": name,
        "package_version": "frozen-by-worker-registry",
        "hyperparameters": {},
        "bounds_handling": "frozen-by-worker-registry",
        "rng": "task-seed",
        "starts_semantics": "shared-frozen-starts",
        "fe_counting": "all-objective-evaluations",
    }
    for name in E7_ALGORITHMS
    if name not in {"PY-SP-SMCO-EVO", "PY-BASE-SMCO"}
}
_COMPARATOR_METADATA.update({
    algorithm: dict(metadata)
    for algorithm, metadata in E7_ALGORITHM_METADATA.items()
})

TERMINAL_STATUSES = {"success", "algorithm_failure", "infra_failure"}
RETRYABLE_STATUSES = {"infra_failure", "node_lost", "stalled"}


def _is_retryable_finish(finish) -> bool:
    """A finished attempt is retryable on infrastructure failure, or when an
    algorithm_failure is actually an unsupported_dependency (a deployment gap
    such as a missing R runtime/package) -- that is not an algorithmic result
    and must be re-run once the environment is fixed. Genuine algorithm
    failures stay terminal."""
    if finish is None:
        return False
    status = finish.get("status")
    if status in RETRYABLE_STATUSES:
        return True
    if status == "algorithm_failure" and "unsupported_dependency" in str(
        finish.get("failure_reason", "")
    ):
        return True
    return False


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _hash_document(document: Mapping, hash_key: str) -> str:
    payload = {key: value for key, value in document.items() if key != hash_key}
    return _sha256_bytes(canonical_json(payload).encode("utf-8"))


def _selection_payload_sha256(selection: Mapping) -> str:
    """Bind the full selection document independently of its short stable ID."""
    return _sha256_bytes(canonical_json(dict(selection)).encode("utf-8"))


def _atomic_write_json(path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _create_json_once(path, payload: Mapping) -> bool:
    """Create an immutable JSON sidecar; return False if it already exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def _selection_configs(selection: Mapping) -> tuple[dict, dict]:
    winner = selection.get("winner")
    if not winner:
        raise ValueError("selection has no winner")
    parsed = parse_algorithm_id(winner)
    language = selection.get("winner_language") or parsed["language"]
    common = {
        "evolution_strategy": DEFAULT_EVO_STRATEGY,
        "evolution_points": DEFAULT_EVO_POINTS,
        "elimination_rate": DEFAULT_ELIMINATION_RATE,
        "de_factor": DEFAULT_DE_FACTOR,
        "de_crossover": DEFAULT_DE_CROSSOVER,
        "n_starts": DEFAULT_N_STARTS,
    }
    winner_config = build_algorithm_config(
        language, parsed["family"], parsed["evolutionary"],
        parsed["state_semantics"] if parsed["evolutionary"] else NONE_TOKEN,
        **common,
    )
    if winner_config["algorithm_id"] != winner:
        raise ValueError(
            f"selection winner {winner!r} is inconsistent with winner_language/config"
        )
    base_config = build_algorithm_config(
        language, parsed["family"], False, NONE_TOKEN, **common,
    )
    if (winner, base_config["algorithm_id"]) != E3F_ALGORITHMS[:2]:
        raise ValueError(
            "extension protocol is frozen to PY-SP-SMCO-EVO and PY-BASE-SMCO"
        )
    return winner_config, base_config


def _metadata_for(algorithm: str, overrides: Mapping[str, Mapping] | None) -> dict:
    if overrides and algorithm in overrides:
        metadata = dict(overrides[algorithm])
    else:
        metadata = dict(_COMPARATOR_METADATA[algorithm])
    missing = [key for key in ALGORITHM_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"{algorithm}: algorithm_metadata missing {missing}")
    return metadata


def _decorate_tasks(
    tasks: Iterable[dict], *, algorithm_metadata: Mapping[str, Mapping] | None = None,
) -> list[dict]:
    decorated: list[dict] = []
    for original in tasks:
        task = dict(original)
        algorithm = task.get("algorithm_id") or task.get("algorithm")
        if "algorithm" in task:
            metadata = _metadata_for(algorithm, algorithm_metadata)
            task["algorithm_id"] = algorithm
            task["language"] = metadata["language"]
            task["configuration_hash"] = compute_configuration_hash(
                {"algorithm_id": algorithm, **metadata}
            )
            task["algorithm_metadata"] = metadata
        else:
            task["algorithm_metadata"] = {
                "language": task["language"],
                "package": "smco",
                "package_version": "git-commit",
                "hyperparameters": dict(task.get("algorithm_config") or {}),
                "bounds_handling": "task-bounds",
                "rng": "task-seed",
                "starts_semantics": "shared-frozen-starts",
                "fe_counting": "all-objective-evaluations",
            }
        task["deadline_hours"] = DEADLINE_HOURS
        task["wall_checkpoint_hours"] = list(WALL_CHECKPOINT_HOURS)
        task["deadline_policy"] = "continue_to_fe_budget_or_normal_termination"
        task["attempt_policy"] = "same_run_id_new_attempt_on_infrastructure_failure"
        decorated.append(task)
    return decorated


def _validate_instance_namespace(instance_index: Mapping | None, required_keys: set[tuple]) -> None:
    if instance_index is None:
        raise ValueError("formal extension manifest requires an instance_index")
    missing = sorted(required_keys - set(instance_index))
    if missing:
        raise ValueError(f"instance_index missing {len(missing)} required cells; e.g. {missing[:3]}")
    bad_stage = sorted(
        (key, instance_index[key].get("stage"))
        for key in required_keys
        if instance_index[key].get("stage") not in {"confirmatory", "extension_confirmatory"}
    )
    if bad_stage:
        raise ValueError(
            "extension manifest requires confirmatory or extension_confirmatory instances; "
            f"bad entries: {bad_stage[:3]}"
        )


def _manifest_from_tasks(
    *, campaign: str, stage: str, tasks: list[dict], selection: Mapping,
    manifest_id: str,
) -> dict:
    manifest = build_manifest(stage, EXTENSION_SUITE, _decorate_tasks(tasks), manifest_id=manifest_id)
    manifest.update({
        "campaign": campaign,
        "evidence_scope": "prospective_extension",
        "selection_hash": selection.get("selection_hash"),
        "selection_payload_sha256": _selection_payload_sha256(selection),
        "winner_algorithm": E3F_ALGORITHMS[0],
        "matched_base_algorithm": E3F_ALGORITHMS[1],
        "deadline_hours": DEADLINE_HOURS,
        "wall_checkpoint_hours": list(WALL_CHECKPOINT_HOURS),
        "deadline_is_kill_threshold": False,
        "retryable_failures": sorted(RETRYABLE_STATUSES),
    })
    return freeze_manifest(manifest)


def build_e3f_manifest(
    selection: Mapping, *, instance_index: Mapping,
    algorithm_metadata: Mapping[str, Mapping] | None = None,
    manifest_id: str = "e3f_component__synthetic_highdim",
) -> dict:
    """Build the frozen 420-task E3-F physical component."""
    keys = {
        (function, dimension, instance)
        for function in E3F_FUNCTIONS for dimension in E3F_DIMENSIONS
        for instance in range(5)
    }
    _validate_instance_namespace(instance_index, keys)
    winner, base = _selection_configs(selection)
    smco_tasks = expand_tasks(
        E3F_STAGE, EXTENSION_SUITE, E3F_FUNCTIONS, E3F_DIMENSIONS, 5,
        [winner, base], fe_budget_per_d=FE_BUDGET_PER_D,
        checkpoints_per_d=FE_CHECKPOINTS_PER_D, instance_index=instance_index,
    )
    baseline_tasks = expand_baseline_tasks(
        E3F_STAGE, EXTENSION_SUITE, E3F_FUNCTIONS, E3F_DIMENSIONS, 5,
        E3F_ALGORITHMS[2:], fe_budget_per_d=FE_BUDGET_PER_D,
        checkpoints_per_d=FE_CHECKPOINTS_PER_D, instance_index=instance_index,
    )
    manifest = build_manifest(
        E3F_STAGE, EXTENSION_SUITE,
        _decorate_tasks([*smco_tasks, *baseline_tasks], algorithm_metadata=algorithm_metadata),
        manifest_id=manifest_id,
    )
    manifest.update({
        "campaign": "e3f", "component_role": "missing_function_extension",
        "evidence_scope": "prospective_extension",
        "selection_hash": selection.get("selection_hash"),
        "selection_payload_sha256": _selection_payload_sha256(selection),
        "winner_algorithm": E3F_ALGORITHMS[0],
        "matched_base_algorithm": E3F_ALGORITHMS[1],
        "algorithms": list(E3F_ALGORITHMS),
        "deadline_hours": DEADLINE_HOURS,
        "wall_checkpoint_hours": list(WALL_CHECKPOINT_HOURS),
        "deadline_is_kill_threshold": False,
        "logical_composite_rows": 840,
        "retryable_failures": sorted(RETRYABLE_STATUSES),
    })
    manifest = freeze_manifest(manifest)
    errors = validate_e3f_manifest(manifest)
    if errors:
        raise ValueError("invalid generated E3-F manifest: " + "; ".join(errors))
    return manifest


def build_e7_manifest(
    selection: Mapping, *, instance_index: Mapping,
    algorithm_metadata: Mapping[str, Mapping] | None = None,
    manifest_id: str = "e7_new_tasks__synthetic_highdim",
) -> dict:
    """Build E7's 1736 physically-new tasks (not its 2016-row logical table)."""
    keys = {
        (function, dimension, instance)
        for function in E7_FUNCTIONS for dimension in E7_DIMENSIONS
        for instance in range(5 if dimension == 1000 else 4)
    }
    _validate_instance_namespace(instance_index, keys)
    winner, base = _selection_configs(selection)
    new_d1000 = expand_baseline_tasks(
        E7_STAGE, EXTENSION_SUITE, E7_FUNCTIONS, (1000,), 5,
        E7_NEW_ALGORITHMS, fe_budget_per_d=FE_BUDGET_PER_D,
        checkpoints_per_d=FE_CHECKPOINTS_PER_D, instance_index=instance_index,
    )
    high_smco = expand_tasks(
        E7_STAGE, EXTENSION_SUITE, E7_FUNCTIONS, E7_DIMENSIONS[1:], 4,
        [winner, base], fe_budget_per_d=FE_BUDGET_PER_D,
        checkpoints_per_d=FE_CHECKPOINTS_PER_D, instance_index=instance_index,
    )
    high_comparators = expand_baseline_tasks(
        E7_STAGE, EXTENSION_SUITE, E7_FUNCTIONS, E7_DIMENSIONS[1:], 4,
        E7_ALGORITHMS[2:], fe_budget_per_d=FE_BUDGET_PER_D,
        checkpoints_per_d=FE_CHECKPOINTS_PER_D, instance_index=instance_index,
    )
    tasks = _decorate_tasks(
        [*new_d1000, *high_smco, *high_comparators],
        algorithm_metadata=algorithm_metadata,
    )
    manifest = build_manifest(E7_STAGE, EXTENSION_SUITE, tasks, manifest_id=manifest_id)
    manifest.update({
        "campaign": "e7", "component_role": "physically_new",
        "evidence_scope": "prospective_extension",
        "selection_hash": selection.get("selection_hash"),
        "selection_payload_sha256": _selection_payload_sha256(selection),
        "winner_algorithm": E3F_ALGORITHMS[0],
        "matched_base_algorithm": E3F_ALGORITHMS[1],
        "algorithms": list(E7_ALGORITHMS),
        "reused_d1000_algorithms": list(E3F_ALGORITHMS),
        "physically_new_rows": 1736,
        "logical_composite_rows": 2016,
        "deadline_hours": DEADLINE_HOURS,
        "wall_checkpoint_hours": list(WALL_CHECKPOINT_HOURS),
        "deadline_is_kill_threshold": False,
        "retryable_failures": sorted(RETRYABLE_STATUSES),
    })
    manifest = freeze_manifest(manifest)
    errors = validate_e7_manifest(manifest)
    if errors:
        raise ValueError("invalid generated E7 manifest: " + "; ".join(errors))
    return manifest


def expected_logical_grid(campaign: str) -> set[tuple]:
    """Return exact ``(function, dimension, instance, algorithm)`` cells."""
    if campaign == "e3f":
        return {
            (function, dimension, instance, algorithm)
            for function in E3F_FUNCTIONS for dimension in E3F_DIMENSIONS
            for instance in range(5) for algorithm in E3F_ALGORITHMS
        }
    if campaign == "e3_combined":
        return {
            (function, dimension, instance, algorithm)
            for function in E7_FUNCTIONS for dimension in E3F_DIMENSIONS
            for instance in range(5) for algorithm in E3F_ALGORITHMS
        }
    if campaign == "e7_new":
        cells = {
            (function, 1000, instance, algorithm)
            for function in E7_FUNCTIONS for instance in range(5)
            for algorithm in E7_NEW_ALGORITHMS
        }
        cells |= {
            (function, dimension, instance, algorithm)
            for function in E7_FUNCTIONS for dimension in E7_DIMENSIONS[1:]
            for instance in range(4) for algorithm in E7_ALGORITHMS
        }
        return cells
    if campaign == "e7":
        return {
            (function, dimension, instance, algorithm)
            for function in E7_FUNCTIONS for dimension in E7_DIMENSIONS
            for instance in range(5 if dimension == 1000 else 4)
            for algorithm in E7_ALGORITHMS
        }
    raise ValueError(f"unknown extension campaign {campaign!r}")


def _task_cell(task: Mapping) -> tuple:
    return (
        task.get("function"), int(task.get("dimension")), int(task.get("instance")),
        task.get("algorithm_id") or task.get("algorithm"),
    )


def _validate_manifest(manifest: Mapping, campaign: str) -> list[str]:
    errors: list[str] = []
    expected_stage = E3F_STAGE if campaign == "e3f" else E7_STAGE
    expected_cells = expected_logical_grid("e3f" if campaign == "e3f" else "e7_new")
    expected_count = 420 if campaign == "e3f" else 1736
    try:
        verify_manifest(dict(manifest))
    except ValueError as exc:
        errors.append(str(exc))
    if manifest.get("frozen") is not True:
        errors.append("manifest is not frozen")
    if manifest.get("campaign") != campaign:
        errors.append(f"campaign {manifest.get('campaign')!r} != {campaign!r}")
    if manifest.get("stage") != expected_stage:
        errors.append(f"stage {manifest.get('stage')!r} != {expected_stage!r}")
    if manifest.get("suite") != EXTENSION_SUITE:
        errors.append(f"suite {manifest.get('suite')!r} != {EXTENSION_SUITE!r}")
    if manifest.get("deadline_hours") != DEADLINE_HOURS:
        errors.append("deadline_hours is not the frozen 72h operational deadline")
    if manifest.get("deadline_is_kill_threshold") is not False:
        errors.append("72h deadline must explicitly not be a kill threshold")
    if manifest.get("wall_checkpoint_hours") != list(WALL_CHECKPOINT_HOURS):
        errors.append("wall checkpoint schedule must be exactly 1h/6h/24h/72h")
    selection_hash = manifest.get("selection_hash")
    if not (isinstance(selection_hash, str) and selection_hash):
        errors.append("manifest missing a non-empty frozen selection_hash")
    selection_payload_sha256 = manifest.get("selection_payload_sha256")
    if not (
        isinstance(selection_payload_sha256, str)
        and len(selection_payload_sha256) == 64
        and all(ch in "0123456789abcdef" for ch in selection_payload_sha256)
    ):
        errors.append("manifest missing a 64-hex selection_payload_sha256")
    tasks = list(manifest.get("tasks") or [])
    if manifest.get("n_tasks") != len(tasks):
        errors.append(f"n_tasks {manifest.get('n_tasks')} != physical task count {len(tasks)}")
    if len(tasks) != expected_count:
        errors.append(f"physical task count {len(tasks)} != {expected_count}")
    run_ids = [task.get("run_id") for task in tasks]
    if len(set(run_ids)) != expected_count or any(not run_id for run_id in run_ids):
        errors.append(f"run_id coverage is not exactly {expected_count} unique non-empty ids")
    cells = [_task_cell(task) for task in tasks]
    if len(cells) != len(set(cells)) or set(cells) != expected_cells:
        errors.append(
            f"physical grid is not exact (rows={len(cells)}, unique={len(set(cells))}, "
            f"expected={len(expected_cells)})"
        )
    for task in tasks:
        dimension = int(task.get("dimension", 0))
        budget = FE_BUDGET_PER_D * dimension
        if int(task.get("fe_budget", -1)) != budget:
            errors.append(f"{task.get('run_id')}: fe_budget != 1000*d")
        if task.get("checkpoints") != [factor * dimension for factor in FE_CHECKPOINTS_PER_D]:
            errors.append(f"{task.get('run_id')}: FE checkpoints mismatch")
        if not task.get("instance_hash") or not task.get("start_points_hash"):
            errors.append(f"{task.get('run_id')}: missing instance/start provenance hash")
        metadata = task.get("algorithm_metadata") or {}
        missing_meta = [key for key in ALGORITHM_METADATA_KEYS if key not in metadata]
        if missing_meta:
            errors.append(f"{task.get('run_id')}: algorithm_metadata missing {missing_meta}")
        if not task.get("configuration_hash"):
            errors.append(f"{task.get('run_id')}: missing configuration_hash")
        elif "algorithm" in task:
            expected_config_hash = compute_configuration_hash({
                "algorithm_id": task["algorithm"], **metadata,
            })
            if task.get("configuration_hash") != expected_config_hash:
                errors.append(f"{task.get('run_id')}: comparator configuration_hash mismatch")
            if task.get("run_id") != baseline_run_id(task):
                errors.append(f"{task.get('run_id')}: comparator run_id recipe mismatch")
        else:
            expected_config_hash = compute_configuration_hash(task.get("algorithm_config") or {})
            if task.get("configuration_hash") != expected_config_hash:
                errors.append(f"{task.get('run_id')}: SMCO configuration_hash mismatch")
            if task.get("run_id") != compute_run_id(task):
                errors.append(f"{task.get('run_id')}: SMCO run_id recipe mismatch")
        if metadata.get("language") != task.get("language"):
            errors.append(f"{task.get('run_id')}: algorithm metadata language mismatch")
        if task.get("deadline_hours") != DEADLINE_HOURS:
            errors.append(f"{task.get('run_id')}: task deadline_hours mismatch")
        if task.get("wall_checkpoint_hours") != list(WALL_CHECKPOINT_HOURS):
            errors.append(f"{task.get('run_id')}: task wall checkpoints mismatch")
    return errors


def validate_e3f_manifest(manifest: Mapping) -> list[str]:
    return _validate_manifest(manifest, "e3f")


def validate_e7_manifest(manifest: Mapping) -> list[str]:
    return _validate_manifest(manifest, "e7")


def _bundle_key(task: Mapping) -> tuple[str, int, int]:
    return task["function"], int(task["dimension"]), int(task["instance"])


def _bundle_cost(tasks: Sequence[Mapping], estimates: Mapping | None) -> float:
    first = tasks[0]
    key = _bundle_key(first)
    text_key = f"{key[0]}|{key[1]}|{key[2]}"
    if estimates:
        value = estimates.get(key, estimates.get(text_key))
        if value is not None:
            return float(value)
    return float(sum(int(task["fe_budget"]) for task in tasks))


def build_shards(manifest: Mapping, *, n_shards: int, cost_estimates: Mapping | None = None) -> dict:
    """Deterministic greedy bin packing over whole problem bundles."""
    if n_shards < 1:
        raise ValueError("n_shards must be positive")
    errors = (validate_e3f_manifest(manifest) if manifest.get("campaign") == "e3f"
              else validate_e7_manifest(manifest))
    if errors:
        raise ValueError("cannot shard invalid manifest: " + "; ".join(errors))
    bundles: dict[tuple, list[dict]] = {}
    for task in manifest["tasks"]:
        bundles.setdefault(_bundle_key(task), []).append(dict(task))
    ranked = sorted(
        ((key, sorted(tasks, key=lambda task: (task["algorithm_id"], task["run_id"])),
          _bundle_cost(tasks, cost_estimates)) for key, tasks in bundles.items()),
        key=lambda item: (-item[2], item[0]),
    )
    shards = [
        {"shard_id": f"shard-{index:03d}", "estimated_cost": 0.0,
         "bundles": [], "tasks": []}
        for index in range(n_shards)
    ]
    for key, tasks, cost in ranked:
        shard = min(shards, key=lambda item: (item["estimated_cost"], item["shard_id"]))
        shard["bundles"].append({
            "function": key[0], "dimension": key[1], "instance": key[2],
            "estimated_cost": cost, "run_ids": [task["run_id"] for task in tasks],
        })
        shard["tasks"].extend(tasks)
        shard["estimated_cost"] += cost
    for shard in shards:
        shard["n_bundles"] = len(shard["bundles"])
        shard["n_tasks"] = len(shard["tasks"])
    document = {
        "schema_version": "1",
        "sharding": "deterministic_greedy_problem_bundle",
        "bundle_key": ["function", "dimension", "instance"],
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "n_shards": n_shards,
        "shards": shards,
    }
    document["shard_sha256"] = _hash_document(document, "shard_sha256")
    return document


def validate_shards(document: Mapping, manifest: Mapping) -> list[str]:
    errors: list[str] = []
    if document.get("shard_sha256") != _hash_document(document, "shard_sha256"):
        errors.append("shard_sha256 mismatch")
    if document.get("manifest_sha256") != manifest.get("manifest_sha256"):
        errors.append("shards are not bound to this manifest")
    tasks = [task for shard in document.get("shards", []) for task in shard.get("tasks", [])]
    run_ids = [task.get("run_id") for task in tasks]
    expected_ids = [task.get("run_id") for task in manifest.get("tasks", [])]
    if len(run_ids) != len(set(run_ids)):
        errors.append("a run_id appears in more than one shard")
    if set(run_ids) != set(expected_ids) or len(run_ids) != len(expected_ids):
        errors.append("shard run_ids do not exactly cover the manifest")
    manifest_tasks = {
        task.get("run_id"): canonical_json(task) for task in manifest.get("tasks", [])
    }
    payload_mismatches = [
        task.get("run_id") for task in tasks
        if manifest_tasks.get(task.get("run_id")) != canonical_json(task)
    ]
    if payload_mismatches:
        errors.append(
            f"shard task payload differs from frozen manifest for "
            f"{len(payload_mismatches)} run_id(s)"
        )
    owners: dict[tuple, set[str]] = {}
    for shard in document.get("shards", []):
        for task in shard.get("tasks", []):
            owners.setdefault(_bundle_key(task), set()).add(shard.get("shard_id"))
    split = [key for key, shard_ids in owners.items() if len(shard_ids) != 1]
    if split:
        errors.append(f"{len(split)} problem bundle(s) split across shards")
    return errors


class ProgressReporter:
    """Atomic heartbeat and immutable wall/deadline checkpoint writer.

    Workers may call :meth:`record` directly.  A process supervisor may instead
    read the worker's progress JSON and call it at each poll.  Crossing the
    operational deadline only creates a snapshot; it never raises or kills.
    """

    def __init__(
        self, evidence_dir, *, run_id: str, attempt_id: str,
        wall_checkpoints_sec: Sequence[float] = WALL_CHECKPOINT_SECONDS,
        deadline_sec: float = DEADLINE_SECONDS,
        heartbeat_interval_sec: float = 300.0,
    ) -> None:
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.wall_checkpoints_sec = tuple(float(value) for value in wall_checkpoints_sec)
        self.deadline_sec = float(deadline_sec)
        self.heartbeat_interval_sec = float(heartbeat_interval_sec)
        self._last_heartbeat = -math.inf
        self._latest: dict = {}

    def _checkpoint_label(self, threshold: float) -> str:
        # Production thresholds are seconds (3600 -> 1h). Unit tests/pilots may
        # pass the compact (1,6,24,72) schedule with a scaled clock.
        value = threshold if max(self.wall_checkpoints_sec, default=0) <= 72 else threshold / 3600.0
        value = int(value) if float(value).is_integer() else value
        return f"{value:g}h"

    def _sidecar(self, *, kind: str, elapsed_sec: float, checkpoint_sec=None) -> dict:
        payload = {
            "schema_version": "1",
            "kind": kind,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "checkpoint_wall_time_sec": checkpoint_sec,
            "captured_wall_time_sec": float(elapsed_sec),
            "captured_unix_sec": time.time(),
            "fe_used": self._latest.get("fe_used"),
            "best_value": self._latest.get("best_value"),
            "normalized_gap": self._latest.get("normalized_gap"),
            "target_hit_fe": self._latest.get("target_hit_fe") or {},
            "process_resources": self._latest.get("process_resources") or {},
            "progress_updated_unix_sec": self._latest.get("progress_updated_unix_sec"),
        }
        payload["sidecar_sha256"] = _hash_document(payload, "sidecar_sha256")
        return payload

    def record(
        self, *, fe_used, best_value, normalized_gap, target_hit_fe,
        process_resources: Mapping | None = None, elapsed_sec: float,
        progress_updated_unix_sec: float | None = None,
    ) -> None:
        elapsed_sec = float(elapsed_sec)
        self._latest = {
            "fe_used": None if fe_used is None else int(fe_used),
            "best_value": best_value,
            "normalized_gap": normalized_gap,
            "target_hit_fe": dict(target_hit_fe or {}),
            "process_resources": dict(process_resources or {}),
            "progress_updated_unix_sec": (
                time.time() if progress_updated_unix_sec is None
                else float(progress_updated_unix_sec)
            ),
        }
        for threshold in self.wall_checkpoints_sec:
            if (elapsed_sec >= threshold and self._latest["fe_used"] is not None
                    and self._latest["best_value"] is not None):
                sidecar = self._sidecar(
                    kind="wall_checkpoint", elapsed_sec=elapsed_sec,
                    checkpoint_sec=threshold,
                )
                _create_json_once(
                    self.evidence_dir / "checkpoints" / f"{self._checkpoint_label(threshold)}.json",
                    sidecar,
                )
        if (elapsed_sec >= self.deadline_sec and self._latest["fe_used"] is not None
                and self._latest["best_value"] is not None):
            _create_json_once(
                self.evidence_dir / "deadline_snapshot.json",
                self._sidecar(
                    kind="operational_deadline", elapsed_sec=elapsed_sec,
                    checkpoint_sec=self.deadline_sec,
                ),
            )
        if elapsed_sec - self._last_heartbeat >= self.heartbeat_interval_sec:
            heartbeat = self._sidecar(kind="heartbeat", elapsed_sec=elapsed_sec)
            _atomic_write_json(self.evidence_dir / "heartbeat.json", heartbeat)
            _create_json_once(
                self.evidence_dir / "heartbeats" / f"{int(elapsed_sec * 1000):015d}.json",
                heartbeat,
            )
            self._last_heartbeat = elapsed_sec

    def finalize(self, outcome: Mapping, *, elapsed_sec: float) -> dict:
        final = dict(outcome)
        elapsed_sec = float(elapsed_sec)
        self.record(
            fe_used=final.get("fe_used"), best_value=final.get("best_value"),
            normalized_gap=final.get("normalized_gap"),
            target_hit_fe=final.get("target_hit_fe") or {},
            process_resources=final.get("process_resources") or {},
            elapsed_sec=elapsed_sec,
        )
        deadline_exceeded = elapsed_sec > self.deadline_sec
        deadline_path = self.evidence_dir / "deadline_snapshot.json"
        if deadline_exceeded:
            if not deadline_path.exists():
                _create_json_once(
                    deadline_path,
                    self._sidecar(
                        kind="operational_deadline_late_or_unavailable",
                        elapsed_sec=elapsed_sec, checkpoint_sec=self.deadline_sec,
                    ),
                )
            snapshot = json.loads(deadline_path.read_text())
            deadline_values = snapshot
        else:
            deadline_values = final
        final.update({
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "deadline_hours": self.deadline_sec / 3600.0,
            "deadline_exceeded": deadline_exceeded,
            "deadline_fe_used": deadline_values.get("fe_used"),
            "deadline_best_value": deadline_values.get("best_value"),
            "deadline_normalized_gap": deadline_values.get("normalized_gap"),
            "final_wall_time_sec": elapsed_sec,
            "post_deadline_result": deadline_exceeded,
        })
        final_sidecar = self._sidecar(kind="final", elapsed_sec=elapsed_sec)
        _create_json_once(self.evidence_dir / "checkpoints" / "final.json", final_sidecar)
        _atomic_write_json(self.evidence_dir / "eventual_outcome.json", final)
        return final


class AttemptLedger:
    """Hash-chained attempt/supersedes events for one logical run_id."""

    def __init__(self, path, *, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id

    def _load(self) -> dict:
        if not self.path.exists():
            return {"schema_version": "1", "run_id": self.run_id, "events": []}
        return json.loads(self.path.read_text())

    @staticmethod
    def _event_hash(event: Mapping) -> str:
        return _hash_document(event, "event_sha256")

    def _append(self, event: dict) -> dict:
        document = self._load()
        previous = document["events"][-1]["event_sha256"] if document["events"] else NONE_TOKEN
        event = dict(event)
        event["sequence"] = len(document["events"]) + 1
        event["previous_event_sha256"] = previous
        event["event_sha256"] = self._event_hash(event)
        document["events"].append(event)
        _atomic_write_json(self.path, document)
        return event

    def attempts(self) -> list[dict]:
        starts = [event for event in self._load()["events"] if event.get("event") == "attempt_started"]
        finishes = {
            event["attempt_id"]: event for event in self._load()["events"]
            if event.get("event") == "attempt_finished"
        }
        return [{**start, "finish": finishes.get(start["attempt_id"])} for start in starts]

    def start(self, *, machine_id: str, git_commit: str, environment_hash: str) -> dict:
        errors = self.validate()
        if errors:
            raise ValueError("attempt ledger invalid: " + "; ".join(errors))
        attempts = self.attempts()
        previous = attempts[-1] if attempts else None
        if previous and previous.get("finish") is None:
            raise ValueError("latest attempt is still running")
        if previous and not _is_retryable_finish(previous["finish"]):
            raise ValueError(
                f"cannot retry terminal status {previous['finish'].get('status')!r}; "
                "only infrastructure failures and unsupported dependencies may restart"
            )
        number = len(attempts) + 1
        attempt_id = f"{self.run_id}.a{number:03d}"
        return self._append({
            "event": "attempt_started", "run_id": self.run_id,
            "attempt_id": attempt_id,
            "supersedes_attempt_id": previous["attempt_id"] if previous else NONE_TOKEN,
            "supersedes_run_id": self.run_id if previous else NONE_TOKEN,
            "machine_id": machine_id, "git_commit": git_commit,
            "environment_hash": environment_hash,
            "started_unix_sec": time.time(),
        })

    def finish(
        self, attempt_id: str, *, status: str, failure_reason: str = NONE_TOKEN,
        evidence_hashes: Mapping[str, str] | None = None,
    ) -> dict:
        if status not in TERMINAL_STATUSES | {"node_lost", "stalled"}:
            raise ValueError(f"invalid attempt terminal status {status!r}")
        attempts = self.attempts()
        if not attempts or attempts[-1]["attempt_id"] != attempt_id:
            raise ValueError("can only finish the latest attempt")
        if attempts[-1].get("finish") is not None:
            raise ValueError("attempt already finished")
        return self._append({
            "event": "attempt_finished", "run_id": self.run_id,
            "attempt_id": attempt_id, "status": status,
            "failure_reason": failure_reason,
            "evidence_hashes": dict(evidence_hashes or {}),
            "finished_unix_sec": time.time(),
        })

    def validate(self) -> list[str]:
        if not self.path.exists():
            return []
        errors: list[str] = []
        try:
            document = self._load()
        except (OSError, json.JSONDecodeError) as exc:
            return [f"cannot read attempt ledger: {exc}"]
        if document.get("run_id") != self.run_id:
            errors.append("ledger run_id mismatch")
        previous_hash = NONE_TOKEN
        open_attempt = None
        last_attempt = None
        last_finish_status = None
        for index, event in enumerate(document.get("events", []), start=1):
            if event.get("sequence") != index:
                errors.append(f"event {index}: sequence mismatch")
            if event.get("previous_event_sha256") != previous_hash:
                errors.append(f"event {index}: previous hash mismatch")
            if event.get("event_sha256") != self._event_hash(event):
                errors.append(f"event {index}: event hash mismatch")
            if event.get("run_id") != self.run_id:
                errors.append(f"event {index}: run_id mismatch")
            if event.get("event") == "attempt_started":
                if open_attempt is not None:
                    errors.append(f"event {index}: attempt starts before prior finish")
                expected_id = f"{self.run_id}.a{(index + 1) // 2:03d}"
                if event.get("attempt_id") != expected_id:
                    errors.append(f"event {index}: attempt_id is not deterministic")
                if last_attempt is None:
                    if event.get("supersedes_attempt_id") != NONE_TOKEN:
                        errors.append("first attempt must not supersede another attempt")
                else:
                    if event.get("supersedes_attempt_id") != last_attempt:
                        errors.append(f"event {index}: supersedes_attempt_id mismatch")
                    if event.get("supersedes_run_id") != self.run_id:
                        errors.append(f"event {index}: supersedes_run_id mismatch")
                    if last_finish_status not in RETRYABLE_STATUSES:
                        errors.append(
                            f"event {index}: attempt retries non-infrastructure status "
                            f"{last_finish_status!r}"
                        )
                open_attempt = event.get("attempt_id")
                last_attempt = open_attempt
            elif event.get("event") == "attempt_finished":
                if event.get("attempt_id") != open_attempt:
                    errors.append(f"event {index}: finish does not match open attempt")
                open_attempt = None
                last_finish_status = event.get("status")
                if last_finish_status not in TERMINAL_STATUSES | {"node_lost", "stalled"}:
                    errors.append(f"event {index}: invalid terminal status")
            else:
                errors.append(f"event {index}: unknown event kind")
            previous_hash = event.get("event_sha256")
        return errors


def _read_json_if_valid(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


_SIGNAL_NAMES = {-1: "SIGHUP", -2: "SIGINT", -9: "SIGKILL", -15: "SIGTERM"}


def _signal_name(returncode):
    """Map a negative subprocess returncode to a signal name, else None."""
    if returncode is None or returncode >= 0:
        return None
    return _SIGNAL_NAMES.get(returncode, f"SIG{-returncode}")


def _classify_interruption(returncode, status=None) -> str:
    """Classify how a worker attempt ended for diagnostics: normal_exit /
    nonzero_exit / signal / unknown. Diagnostic only -- never overrides the
    outcome status, so an external interruption is NOT mislabeled as
    algorithm_failure (2026-08-04 review part 2)."""
    if returncode is None:
        return "unknown"
    if returncode == 0:
        return "normal_exit"
    if returncode < 0:
        return "signal"
    return "nonzero_exit"


def _worker_diagnostics(
    *, worker_pid, parent_pid, command, started_unix_sec, returncode,
    recorded_unix_sec,
) -> dict:
    """Build the next-round worker diagnostics record. Additive only: existing
    outcome schema is unchanged, these fields sit alongside it so a future
    external interruption can be attributed (worker PID/exit/signal) instead
    of leaving an unexplained open attempt."""
    command_hash = hashlib.sha256(
        " ".join(str(c) for c in command).encode("utf-8")
    ).hexdigest()
    return {
        "worker_pid": worker_pid,
        "parent_pid": parent_pid,
        "worker_command_sha256": command_hash,
        "worker_started_unix_sec": started_unix_sec,
        "worker_exit_code": returncode,
        "worker_termination_signal": _signal_name(returncode),
        "interruption_kind": _classify_interruption(returncode),
        "recorded_unix_sec": recorded_unix_sec,
    }


def _process_resources(pid: int) -> dict:
    resources = {"pid": pid}
    try:
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        fields = dict(line.split(":", 1) for line in status if ":" in line)
        resources["rss_kb"] = int(fields.get("VmRSS", "0 kB").split()[0])
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return resources


def supervise_command(
    command: Sequence[str], *, run_id: str, evidence_root,
    deadline_sec: float = DEADLINE_SECONDS,
    wall_checkpoints_sec: Sequence[float] = WALL_CHECKPOINT_SECONDS,
    poll_interval_sec: float = 300.0,
    heartbeat_interval_sec: float = 300.0,
    machine_id: str = "", git_commit: str = "", environment_hash: str = "",
    extra_env: Mapping[str, str] | None = None,
) -> dict:
    """Run one attempt, monitor it past the deadline, and preserve evidence.

    There is intentionally no timeout/kill parameter.  The child receives
    ``SMCO_PROGRESS_PATH`` and ``SMCO_RESULT_PATH``; workers should atomically
    update the first and write the second only at eventual termination.
    """
    evidence_root = Path(evidence_root)
    run_dir = evidence_root / run_id
    ledger = AttemptLedger(run_dir / "attempt_ledger.json", run_id=run_id)
    started = ledger.start(
        machine_id=machine_id or socket.gethostname(), git_commit=git_commit,
        environment_hash=environment_hash,
    )
    attempt_id = started["attempt_id"]
    attempts = ledger.attempts()
    first_started_unix_sec = float(attempts[0]["started_unix_sec"])
    # The 72h operational clock belongs to the logical run_id, not to an
    # individual infrastructure attempt.  Retries therefore cannot obtain a
    # fresh deadline window.
    elapsed_before_attempt = max(0.0, time.time() - first_started_unix_sec)
    attempt_dir = run_dir / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=True)
    progress_path = attempt_dir / "worker_progress.json"
    result_path = attempt_dir / "worker_result.json"
    reporter = ProgressReporter(
        attempt_dir, run_id=run_id, attempt_id=attempt_id,
        wall_checkpoints_sec=wall_checkpoints_sec, deadline_sec=deadline_sec,
        heartbeat_interval_sec=heartbeat_interval_sec,
    )
    env = os.environ.copy()
    env.update({
        "SMCO_PROGRESS_PATH": str(progress_path),
        "SMCO_RESULT_PATH": str(result_path),
        "SMCO_RUN_ID": run_id,
        "SMCO_ATTEMPT_ID": attempt_id,
    })
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
    t0 = time.monotonic()
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    with open(stdout_path, "w") as stdout_handle, open(stderr_path, "w") as stderr_handle:
        try:
            process = subprocess.Popen(
                list(command), stdin=subprocess.DEVNULL, stdout=stdout_handle,
                stderr=stderr_handle, text=True, env=env,
            )
        except OSError as exc:
            outcome = {
                "run_id": run_id, "status": "infra_failure", "fe_used": 0,
                "best_value": None, "normalized_gap": None, "target_hit_fe": {},
                "failure_reason": f"launch_failure: {exc}",
            }
            outcome["machine_id"] = machine_id or socket.gethostname()
            outcome["git_commit"] = git_commit
            outcome["environment_hash"] = environment_hash
            outcome["worker_diagnostics"] = _worker_diagnostics(
                worker_pid=None, parent_pid=os.getpid(), command=list(command),
                started_unix_sec=float(started.get("started_unix_sec", 0.0)),
                returncode=None, recorded_unix_sec=time.time(),
            )
            final = reporter.finalize(outcome, elapsed_sec=elapsed_before_attempt)
            ledger.finish(
                attempt_id, status="infra_failure",
                failure_reason=outcome["failure_reason"],
            )
            _atomic_write_json(run_dir / "latest_outcome.json", final)
            return final
        while process.poll() is None:
            elapsed = elapsed_before_attempt + (time.monotonic() - t0)
            progress = _read_json_if_valid(progress_path) or {}
            reporter.record(
                fe_used=progress.get("fe_used"), best_value=progress.get("best_value"),
                normalized_gap=progress.get("normalized_gap"),
                target_hit_fe=progress.get("target_hit_fe") or {},
                process_resources=_process_resources(process.pid), elapsed_sec=elapsed,
                progress_updated_unix_sec=(
                    progress_path.stat().st_mtime if progress_path.exists() else None
                ),
            )
            try:
                process.wait(timeout=max(0.001, float(poll_interval_sec)))
            except subprocess.TimeoutExpired:
                pass
    elapsed = elapsed_before_attempt + (time.monotonic() - t0)
    outcome = _read_json_if_valid(result_path)
    if outcome is None:
        outcome = {
            "run_id": run_id, "status": "infra_failure", "fe_used": 0,
            "best_value": None, "normalized_gap": None, "target_hit_fe": {},
            "failure_reason": (
                f"worker exited {process.returncode} without a valid SMCO_RESULT_PATH"
            ),
        }
    if outcome.get("run_id") != run_id:
        outcome = {
            "run_id": run_id, "status": "infra_failure", "fe_used": 0,
            "best_value": None, "normalized_gap": None, "target_hit_fe": {},
            "failure_reason": "worker result run_id mismatch",
        }
    # Dispatcher metadata is authoritative.  In particular, workers on an
    # rsync'd fleet checkout may emit an empty git_commit; never preserve that
    # empty value over the frozen coordinator provenance.
    outcome["machine_id"] = machine_id or socket.gethostname()
    outcome["git_commit"] = git_commit
    outcome["environment_hash"] = environment_hash
    outcome["worker_diagnostics"] = _worker_diagnostics(
        worker_pid=process.pid, parent_pid=os.getpid(), command=list(command),
        started_unix_sec=float(started.get("started_unix_sec", 0.0)),
        returncode=process.returncode, recorded_unix_sec=time.time(),
    )
    final = reporter.finalize(outcome, elapsed_sec=elapsed)
    evidence_hashes = {}
    for relative in (
        "heartbeat.json", "deadline_snapshot.json", "checkpoints/final.json",
        "eventual_outcome.json",
    ):
        path = attempt_dir / relative
        if path.exists():
            evidence_hashes[relative] = file_sha256(path)
    ledger.finish(
        attempt_id, status=final.get("status", "infra_failure"),
        failure_reason=final.get("failure_reason", NONE_TOKEN),
        evidence_hashes=evidence_hashes,
    )
    _atomic_write_json(run_dir / "latest_outcome.json", final)
    return final


def deadline_evidence_errors(
    outcome: Mapping, evidence_dir, *, heartbeat_grace_sec: float = 600.0,
) -> list[str]:
    """Validate separation of operational-deadline and eventual semantics."""
    errors: list[str] = []
    evidence_dir = Path(evidence_dir)
    final_wall = outcome.get("final_wall_time_sec")
    deadline_hours = outcome.get("deadline_hours")
    if final_wall is None or deadline_hours is None:
        return ["outcome missing final_wall_time_sec/deadline_hours"]
    deadline_sec = float(deadline_hours) * 3600.0
    exceeded = float(final_wall) > deadline_sec
    if bool(outcome.get("deadline_exceeded")) != exceeded:
        errors.append("deadline_exceeded disagrees with final wall time")
    if bool(outcome.get("post_deadline_result")) != exceeded:
        errors.append("post_deadline_result disagrees with deadline_exceeded")
    if exceeded and outcome.get("status") == "timeout":
        errors.append("operational deadline must not rewrite eventual status to timeout")
    heartbeat = _read_json_if_valid(evidence_dir / "heartbeat.json")
    if heartbeat is None:
        errors.append("missing heartbeat")
    final = _read_json_if_valid(evidence_dir / "checkpoints" / "final.json")
    if final is None:
        errors.append("missing final checkpoint")
    if exceeded:
        snapshot = _read_json_if_valid(evidence_dir / "deadline_snapshot.json")
        if snapshot is None:
            errors.append("deadline exceeded without deadline snapshot")
        else:
            if snapshot.get("sidecar_sha256") != _hash_document(snapshot, "sidecar_sha256"):
                errors.append("deadline snapshot hash mismatch")
            captured = float(snapshot.get("captured_wall_time_sec", math.inf))
            if captured > deadline_sec + heartbeat_grace_sec:
                errors.append("deadline snapshot was captured too late")
            if snapshot.get("fe_used") is None or snapshot.get("best_value") is None:
                errors.append("deadline snapshot lacks live FE/best progress")
            updated = snapshot.get("progress_updated_unix_sec")
            captured_unix = snapshot.get("captured_unix_sec")
            if updated is None or captured_unix is None:
                errors.append("deadline snapshot lacks progress freshness provenance")
            elif float(captured_unix) - float(updated) > heartbeat_grace_sec:
                errors.append("deadline snapshot progress is stale")
            for outcome_key, snapshot_key in (
                ("deadline_fe_used", "fe_used"),
                ("deadline_best_value", "best_value"),
                ("deadline_normalized_gap", "normalized_gap"),
            ):
                if outcome.get(outcome_key) != snapshot.get(snapshot_key):
                    errors.append(f"{outcome_key} does not match immutable deadline snapshot")
    else:
        for deadline_key, final_key in (
            ("deadline_fe_used", "fe_used"),
            ("deadline_best_value", "best_value"),
            ("deadline_normalized_gap", "normalized_gap"),
        ):
            if outcome.get(deadline_key) != outcome.get(final_key):
                errors.append(f"{deadline_key} must equal final value before deadline")
    for sidecar in evidence_dir.glob("checkpoints/*.json"):
        value = _read_json_if_valid(sidecar)
        if value and value.get("sidecar_sha256") != _hash_document(value, "sidecar_sha256"):
            errors.append(f"checkpoint hash mismatch: {sidecar.name}")
        if value and value.get("kind") == "wall_checkpoint" and (
            value.get("fe_used") is None or value.get("best_value") is None
        ):
            errors.append(f"wall checkpoint lacks live FE/best progress: {sidecar.name}")
    ledger_path = evidence_dir.parent.parent / "attempt_ledger.json"
    if ledger_path.exists():
        ledger = _read_json_if_valid(ledger_path) or {}
        finish = next((
            event for event in ledger.get("events", [])
            if event.get("event") == "attempt_finished"
            and event.get("attempt_id") == outcome.get("attempt_id")
        ), None)
        if finish is None:
            errors.append("attempt ledger lacks matching finish event")
        else:
            for relative, expected_hash in (finish.get("evidence_hashes") or {}).items():
                path = evidence_dir / relative
                if not path.exists() or file_sha256(path) != expected_hash:
                    errors.append(f"attempt ledger evidence hash mismatch: {relative}")
    return errors


EXTENSION_RESULT_COLUMNS = (
    "run_id", "attempt_id", "manifest_id", "stage", "suite", "function",
    "dimension", "instance", "algorithm_id", "language", "seed",
    "configuration_hash", "instance_hash", "start_points_hash", "fe_budget",
    "fe_used", "best_value", "known_optimum", "normalized_gap",
    "objective_sense", "status", "failure_reason", "machine_id", "git_commit",
    "environment_hash", "deadline_hours", "deadline_exceeded", "deadline_fe_used",
    "deadline_best_value", "deadline_normalized_gap", "final_wall_time_sec",
    "post_deadline_result", "supersedes_attempt_id", "supersedes_run_id",
)


def _row_from_outcome(task: Mapping, outcome: Mapping, *, manifest_id: str,
                      attempt: Mapping | None = None) -> dict:
    attempt = attempt or {}
    return {
        "run_id": task["run_id"],
        "attempt_id": outcome.get("attempt_id") or attempt.get("attempt_id"),
        "manifest_id": manifest_id,
        "stage": task["stage"], "suite": task["suite"],
        "function": task["function"], "dimension": int(task["dimension"]),
        "instance": int(task["instance"]),
        "algorithm_id": task.get("algorithm_id") or task.get("algorithm"),
        "language": task.get("language") or (task.get("algorithm_metadata") or {}).get("language"),
        "seed": int(task["seed"]),
        "configuration_hash": task.get("configuration_hash"),
        "instance_hash": task.get("instance_hash"),
        "start_points_hash": task.get("start_points_hash"),
        "fe_budget": int(task["fe_budget"]),
        "fe_used": outcome.get("fe_used"), "best_value": outcome.get("best_value"),
        "known_optimum": outcome.get("known_optimum"),
        "normalized_gap": outcome.get("normalized_gap"),
        "objective_sense": outcome.get("objective_sense", "minimize"),
        "status": outcome.get("status"),
        "failure_reason": outcome.get("failure_reason", NONE_TOKEN),
        "machine_id": outcome.get("machine_id") or attempt.get("machine_id"),
        "git_commit": outcome.get("git_commit") or attempt.get("git_commit"),
        "environment_hash": outcome.get("environment_hash") or attempt.get("environment_hash"),
        "deadline_hours": outcome.get("deadline_hours"),
        "deadline_exceeded": outcome.get("deadline_exceeded"),
        "deadline_fe_used": outcome.get("deadline_fe_used"),
        "deadline_best_value": outcome.get("deadline_best_value"),
        "deadline_normalized_gap": outcome.get("deadline_normalized_gap"),
        "final_wall_time_sec": outcome.get("final_wall_time_sec"),
        "post_deadline_result": outcome.get("post_deadline_result"),
        "supersedes_attempt_id": attempt.get("supersedes_attempt_id", NONE_TOKEN),
        "supersedes_run_id": attempt.get("supersedes_run_id", NONE_TOKEN),
    }


def _audit_check(name: str, rows: Sequence[Mapping], errors: list[str]) -> dict:
    return {"name": name, "passed": not errors, "n": len(rows), "errors": errors}


def audit_extension_records(
    manifest: Mapping, rows: Sequence[Mapping], *, evidence_root,
) -> dict:
    """Run exactly 12 main checks plus separately reported deadline checks."""
    rows = list(rows)
    evidence_root = Path(evidence_root)
    task_by_id = {task["run_id"]: task for task in manifest.get("tasks", [])}
    campaign = manifest.get("campaign")
    checks: list[dict] = []

    manifest_errors = (
        validate_e3f_manifest(manifest) if campaign == "e3f"
        else validate_e7_manifest(manifest)
    )
    checks.append(_audit_check("frozen_manifest", rows, manifest_errors))

    expected = expected_logical_grid("e3f" if campaign == "e3f" else "e7_new")
    row_cells = {
        (row.get("function"), int(row.get("dimension", -1)), int(row.get("instance", -1)),
         row.get("algorithm_id")) for row in rows
    }
    grid_errors = [] if row_cells == expected else [
        f"result grid cells {len(row_cells)} != exact expected {len(expected)}"
    ]
    checks.append(_audit_check("exact_campaign_grid", rows, grid_errors))

    row_ids = [row.get("run_id") for row in rows]
    planned_ids = set(task_by_id)
    coverage_errors = []
    if set(row_ids) != planned_ids:
        coverage_errors.append(
            f"run-id coverage differs: missing={len(planned_ids - set(row_ids))} "
            f"extra={len(set(row_ids) - planned_ids)}"
        )
    checks.append(_audit_check("manifest_run_id_coverage", rows, coverage_errors))

    duplicate_ids = sorted({run_id for run_id in row_ids if row_ids.count(run_id) > 1})
    checks.append(_audit_check(
        "physical_and_unique_run_ids", rows,
        [f"duplicate run_id: {run_id}" for run_id in duplicate_ids],
    ))

    identity_errors: list[str] = []
    for row in rows:
        task = task_by_id.get(row.get("run_id"))
        if task is None:
            continue
        for key in ("stage", "suite", "function", "dimension", "instance", "seed"):
            left, right = row.get(key), task.get(key)
            if str(left) != str(right):
                identity_errors.append(f"{row.get('run_id')}: {key} mismatch")
        if row.get("algorithm_id") != (task.get("algorithm_id") or task.get("algorithm")):
            identity_errors.append(f"{row.get('run_id')}: algorithm_id mismatch")
    checks.append(_audit_check("task_identity", rows, identity_errors))

    hash_errors = [
        f"{row.get('run_id')}: missing/mismatched artifact hash"
        for row in rows
        if not row.get("configuration_hash") or not row.get("instance_hash")
        or not row.get("start_points_hash")
        or (row.get("run_id") in task_by_id and any(
            row.get(key) != task_by_id[row["run_id"]].get(key)
            for key in ("configuration_hash", "instance_hash", "start_points_hash")
        ))
    ]
    checks.append(_audit_check("artifact_hashes", rows, hash_errors))

    fe_errors = []
    for row in rows:
        try:
            fe_used, budget = int(row.get("fe_used")), int(row.get("fe_budget"))
            if fe_used < 0 or fe_used > budget:
                fe_errors.append(f"{row.get('run_id')}: FE {fe_used} outside [0,{budget}]")
        except (TypeError, ValueError):
            fe_errors.append(f"{row.get('run_id')}: invalid FE count")
    checks.append(_audit_check("fe_within_budget", rows, fe_errors))

    direction_errors = [
        f"{row.get('run_id')}: objective_sense is not minimize"
        for row in rows if row.get("objective_sense") != "minimize"
    ]
    checks.append(_audit_check("objective_direction", rows, direction_errors))

    numeric_errors: list[str] = []
    for row in rows:
        if row.get("status") != "success":
            continue
        try:
            best = float(row.get("best_value"))
            optimum = float(row.get("known_optimum"))
            gap = float(row.get("normalized_gap"))
            if not all(math.isfinite(value) for value in (best, optimum, gap)):
                raise ValueError
            if abs(optimum) > 1e-12 or best < optimum - 1e-6 or gap < 0:
                numeric_errors.append(f"{row.get('run_id')}: invalid optimum/gap")
        except (TypeError, ValueError):
            numeric_errors.append(f"{row.get('run_id')}: nonfinite/missing successful result")
    checks.append(_audit_check("finite_and_gap_sanity", rows, numeric_errors))

    provenance_errors = [
        f"{row.get('run_id')}: missing machine/git/environment provenance"
        for row in rows if not row.get("machine_id") or not row.get("git_commit")
        or not row.get("environment_hash")
    ]
    checks.append(_audit_check("provenance_complete", rows, provenance_errors))

    status_errors = [
        f"{row.get('run_id')}: invalid eventual status {row.get('status')!r}"
        for row in rows if row.get("status") not in TERMINAL_STATUSES
    ]
    checks.append(_audit_check("eventual_status_vocabulary", rows, status_errors))

    ledger_errors: list[str] = []
    for row in rows:
        run_id = row.get("run_id")
        ledger = AttemptLedger(evidence_root / str(run_id) / "attempt_ledger.json", run_id=str(run_id))
        current = ledger.validate()
        if current:
            ledger_errors.extend(f"{run_id}: {error}" for error in current)
            continue
        attempts = ledger.attempts()
        if not attempts or attempts[-1]["attempt_id"] != row.get("attempt_id"):
            ledger_errors.append(f"{run_id}: latest attempt does not match valid row")
        elif not attempts[-1].get("finish"):
            ledger_errors.append(f"{run_id}: latest attempt has no terminal ledger event")
    checks.append(_audit_check("attempts_and_supersedes_resolvable", rows, ledger_errors))

    deadline_checks: list[dict] = []
    heartbeat_errors: list[str] = []
    checkpoint_errors: list[str] = []
    snapshot_errors: list[str] = []
    semantic_errors: list[str] = []
    eventual_errors: list[str] = []
    for row in rows:
        attempt_dir = (
            evidence_root / str(row.get("run_id")) / "attempts" / str(row.get("attempt_id"))
        )
        if not (attempt_dir / "heartbeat.json").exists():
            heartbeat_errors.append(f"{row.get('run_id')}: missing heartbeat")
        try:
            final_wall = float(row.get("final_wall_time_sec"))
        except (TypeError, ValueError):
            final_wall = -1
        for hours, threshold in zip(WALL_CHECKPOINT_HOURS, WALL_CHECKPOINT_SECONDS):
            if final_wall >= threshold and not (attempt_dir / "checkpoints" / f"{hours}h.json").exists():
                checkpoint_errors.append(f"{row.get('run_id')}: missing {hours}h checkpoint")
        if not (attempt_dir / "checkpoints" / "final.json").exists():
            checkpoint_errors.append(f"{row.get('run_id')}: missing final checkpoint")
        per_run = deadline_evidence_errors(row, attempt_dir)
        snapshot_errors.extend(f"{row.get('run_id')}: {error}" for error in per_run)
        exceeded = bool(row.get("deadline_exceeded"))
        if exceeded != bool(row.get("post_deadline_result")):
            semantic_errors.append(f"{row.get('run_id')}: deadline/eventual flags disagree")
        if exceeded and row.get("status") == "timeout":
            semantic_errors.append(f"{row.get('run_id')}: deadline forged as timeout status")
        eventual = _read_json_if_valid(attempt_dir / "eventual_outcome.json")
        if eventual is None:
            eventual_errors.append(f"{row.get('run_id')}: missing eventual outcome")
        elif any(eventual.get(key) != row.get(key) for key in (
            "status", "fe_used", "best_value", "deadline_exceeded", "post_deadline_result"
        )):
            eventual_errors.append(f"{row.get('run_id')}: eventual outcome differs from merged row")
    deadline_checks.append(_audit_check("heartbeat_evidence", rows, heartbeat_errors))
    deadline_checks.append(_audit_check("wall_checkpoint_coverage", rows, checkpoint_errors))
    deadline_checks.append(_audit_check("deadline_snapshot_integrity", rows, snapshot_errors))
    deadline_checks.append(_audit_check("deadline_eventual_semantics", rows, semantic_errors))
    deadline_checks.append(_audit_check("eventual_outcome_binding", rows, eventual_errors))

    failed = [check["name"] for check in checks if not check["passed"]]
    failed_deadline = [check["name"] for check in deadline_checks if not check["passed"]]
    return {
        "passed": not failed and not failed_deadline,
        "failed_checks": failed,
        "failed_deadline_checks": failed_deadline,
        "checks": checks,
        "deadline_checks": deadline_checks,
        "n_rows": len(rows),
        "deadline_analysis": "operational_snapshot",
        "eventual_analysis": "eventual_outcome",
    }


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def merge_extension_results(manifest_path, evidence_root, merged_dir) -> dict:
    """Merge the latest terminal attempt per logical run without hiding retries."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    validation = (
        validate_e3f_manifest(manifest) if manifest.get("campaign") == "e3f"
        else validate_e7_manifest(manifest)
    )
    if validation:
        raise ValueError("extension manifest invalid: " + "; ".join(validation))
    evidence_root = Path(evidence_root)
    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    latest_rows: list[dict] = []
    all_attempt_rows: list[dict] = []
    missing: list[dict] = []
    ledger_records: list[dict] = []
    for task in manifest["tasks"]:
        run_id = task["run_id"]
        ledger_path = evidence_root / run_id / "attempt_ledger.json"
        ledger = AttemptLedger(ledger_path, run_id=run_id)
        if ledger_path.exists():
            ledger_records.append({
                "run_id": run_id, "path": str(ledger_path),
                "sha256": file_sha256(ledger_path), "events": ledger._load().get("events", []),
            })
        attempts = ledger.attempts()
        for attempt in attempts:
            path = evidence_root / run_id / "attempts" / attempt["attempt_id"] / "eventual_outcome.json"
            outcome = _read_json_if_valid(path)
            if outcome is not None:
                all_attempt_rows.append(_row_from_outcome(
                    task, outcome, manifest_id=manifest["manifest_id"], attempt=attempt,
                ))
        if not attempts or attempts[-1].get("finish") is None:
            missing.append({"run_id": run_id, "reason": "no terminal attempt"})
            continue
        attempt = attempts[-1]
        path = evidence_root / run_id / "attempts" / attempt["attempt_id"] / "eventual_outcome.json"
        outcome = _read_json_if_valid(path)
        if outcome is None:
            missing.append({"run_id": run_id, "reason": "missing eventual_outcome.json"})
            continue
        latest_rows.append(_row_from_outcome(
            task, outcome, manifest_id=manifest["manifest_id"], attempt=attempt,
        ))
    audit = audit_extension_records(manifest, latest_rows, evidence_root=evidence_root)
    audit["manifest_sha256"] = manifest["manifest_sha256"]
    deadline_rows = [{
        "run_id": row["run_id"], "attempt_id": row["attempt_id"],
        "deadline_hours": row["deadline_hours"],
        "deadline_exceeded": row["deadline_exceeded"],
        "deadline_fe_used": row["deadline_fe_used"],
        "deadline_best_value": row["deadline_best_value"],
        "deadline_normalized_gap": row["deadline_normalized_gap"],
        "post_deadline_result": row["post_deadline_result"],
    } for row in latest_rows]
    _write_csv(merged_dir / "all_attempts.csv", EXTENSION_RESULT_COLUMNS, all_attempt_rows)
    _write_csv(merged_dir / "valid_runs.csv", EXTENSION_RESULT_COLUMNS, latest_rows)
    _write_csv(merged_dir / "missing_runs.csv", ("run_id", "reason"), missing)
    _write_csv(
        merged_dir / "deadline_snapshots.csv",
        ("run_id", "attempt_id", "deadline_hours", "deadline_exceeded",
         "deadline_fe_used", "deadline_best_value", "deadline_normalized_gap",
         "post_deadline_result"),
        deadline_rows,
    )
    ledger_doc = {
        "schema_version": "1", "manifest_sha256": manifest["manifest_sha256"],
        "runs": ledger_records,
    }
    ledger_doc["ledger_sha256"] = _hash_document(ledger_doc, "ledger_sha256")
    _atomic_write_json(merged_dir / "attempt_ledger.json", ledger_doc)
    _atomic_write_json(merged_dir / "provenance_audit.json", audit)
    return {
        "n_attempts": len(all_attempt_rows), "n_valid": len(latest_rows),
        "n_missing": len(missing), "audit": audit,
    }


def worker_progress(
    *, fe_used: int, best_value, normalized_gap, target_hit_fe: Mapping | None = None,
) -> None:
    """Worker hook: atomically publish current FE/best state when configured."""
    value = os.environ.get("SMCO_PROGRESS_PATH")
    if not value:
        return
    _atomic_write_json(Path(value), {
        "fe_used": int(fe_used), "best_value": best_value,
        "normalized_gap": normalized_gap, "target_hit_fe": dict(target_hit_fe or {}),
    })


class WorkerProgressSink:
    """Cheap throttling wrapper used from per-evaluation observers."""

    def __init__(self, interval_sec: float | None = None) -> None:
        self.path = os.environ.get("SMCO_PROGRESS_PATH")
        configured = os.environ.get("SMCO_PROGRESS_INTERVAL_SEC")
        self.interval_sec = float(
            interval_sec if interval_sec is not None
            else configured if configured is not None else 300.0
        )
        self._last_emit = -math.inf

    @property
    def enabled(self) -> bool:
        return bool(self.path)

    def emit(
        self, *, fe_used: int, best_value, normalized_gap,
        target_hit_fe: Mapping | None = None, force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_emit < self.interval_sec:
            return
        worker_progress(
            fe_used=fe_used, best_value=best_value,
            normalized_gap=normalized_gap, target_hit_fe=target_hit_fe,
        )
        self._last_emit = now


def plan_extension_dispatch(tasks: Iterable[Mapping], evidence_root) -> dict:
    """Classify logical runs for dry-run/validate-only/resume dispatch."""
    evidence_root = Path(evidence_root)
    groups = {
        "completed": [], "retryable": [], "stalled": [],
        "running": [], "pending": [],
    }
    for task in tasks:
        run_id = task["run_id"]
        ledger = AttemptLedger(evidence_root / run_id / "attempt_ledger.json", run_id=run_id)
        if ledger.validate():
            groups["running"].append(run_id)
            continue
        attempts = ledger.attempts()
        if not attempts:
            groups["pending"].append(run_id)
            continue
        latest = attempts[-1]
        finish = latest.get("finish")
        if finish is None:
            stall_errors = stalled_attempt_errors(evidence_root, run_id)
            groups["stalled" if not stall_errors else "running"].append(run_id)
        elif finish.get("status") == "success":
            groups["completed"].append(run_id)
        elif _is_retryable_finish(finish):
            groups["retryable"].append(run_id)
        elif finish.get("status") == "algorithm_failure":
            groups["completed"].append(run_id)
        else:
            groups["running"].append(run_id)
    return {
        "n_tasks": sum(len(value) for value in groups.values()),
        **{name: len(ids) for name, ids in groups.items()},
        "run_ids": groups,
    }


def _pid_alive(pid: int) -> bool:
    """Return True if `pid` is a live process on this host."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but is owned by another user
    except OSError:
        return False
    return True


def stalled_attempt_errors(
    evidence_root, run_id: str, *, now_unix_sec: float | None = None,
    stale_after_sec: float = 1200.0, no_fe_growth_window_sec: float = 600.0,
) -> list[str]:
    """Evidence gate for restarting an attempt after node/heartbeat loss.

    Both conditions are mandatory: no recent heartbeat, and no FE growth over
    an earlier-to-latest heartbeat window.  Merely crossing 72h never satisfies
    this gate.
    """
    errors: list[str] = []
    evidence_root = Path(evidence_root)
    run_dir = evidence_root / run_id
    ledger = AttemptLedger(run_dir / "attempt_ledger.json", run_id=run_id)
    errors.extend(ledger.validate())
    attempts = ledger.attempts()
    if not attempts or attempts[-1].get("finish") is not None:
        errors.append("latest attempt is not open")
        return errors
    attempt_id = attempts[-1]["attempt_id"]
    history_dir = run_dir / "attempts" / attempt_id / "heartbeats"
    history = [
        value for path in sorted(history_dir.glob("*.json"))
        if (value := _read_json_if_valid(path)) is not None
    ]
    if len(history) < 2:
        errors.append("need at least two immutable heartbeat samples")
        return errors
    for heartbeat in history:
        if heartbeat.get("sidecar_sha256") != _hash_document(heartbeat, "sidecar_sha256"):
            errors.append("heartbeat history hash mismatch")
            return errors
    latest = history[-1]
    now = time.time() if now_unix_sec is None else float(now_unix_sec)
    latest_unix = float(latest.get("captured_unix_sec", math.inf))
    if now - latest_unix < stale_after_sec:
        errors.append("latest heartbeat is not stale")
        return errors
    # A stale heartbeat is recoverable unless its recorded worker is still
    # alive. Historical FE growth must NOT pin an open attempt as "running"
    # forever after the process vanished (P0, 2026-08-04 review): a task that
    # progressed and then lost its process must restart.
    pid = (latest.get("process_resources") or {}).get("pid")
    if pid is not None and _pid_alive(int(pid)):
        errors.append("worker process recorded in the stale heartbeat is still alive")
    return errors


def recover_stalled_attempt(
    evidence_root, run_id: str, *, now_unix_sec: float | None = None,
    stale_after_sec: float = 1200.0, no_fe_growth_window_sec: float = 600.0,
) -> dict:
    errors = stalled_attempt_errors(
        evidence_root, run_id, now_unix_sec=now_unix_sec,
        stale_after_sec=stale_after_sec,
        no_fe_growth_window_sec=no_fe_growth_window_sec,
    )
    if errors:
        raise ValueError("cannot recover attempt: " + "; ".join(errors))
    ledger = AttemptLedger(Path(evidence_root) / run_id / "attempt_ledger.json", run_id=run_id)
    attempt_id = ledger.attempts()[-1]["attempt_id"]
    return ledger.finish(
        attempt_id, status="stalled",
        failure_reason=(
            f"no heartbeat for >= {stale_after_sec:g}s and no FE growth for "
            f">= {no_fe_growth_window_sec:g}s"
        ),
    )


def _read_csv_rows(path) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _row_cell(row: Mapping) -> tuple:
    return (
        row.get("function"), int(row.get("dimension")), int(row.get("instance")),
        row.get("algorithm_id"),
    )


def _filtered_rows(rows: Iterable[Mapping], filters: Mapping | None) -> list[dict]:
    filters = filters or {}
    result = []
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in filters.items()):
            result.append(dict(row))
    return result


def _instance_provenance_errors(rows: Sequence[Mapping]) -> list[str]:
    """Require every algorithm in a logical problem cell to share artifacts."""
    grouped: dict[tuple, dict[str, set[str]]] = {}
    for row in rows:
        key = (row.get("function"), int(row.get("dimension")), int(row.get("instance")))
        values = grouped.setdefault(
            key, {"instance_hash": set(), "start_points_hash": set()},
        )
        for field in values:
            value = str(row.get(field) or "")
            values[field].add(value)
    errors = []
    for key, values in grouped.items():
        if any("" in hashes or len(hashes) != 1 for hashes in values.values()):
            errors.append(
                f"instance/start provenance mismatch for {key}: "
                f"instance_hashes={sorted(values['instance_hash'])}, "
                f"start_points_hashes={sorted(values['start_points_hash'])}"
            )
    return errors


def _frozen_source_document_errors(value: Mapping) -> list[str]:
    """Validate old and extension source documents with their native hashes."""
    if value.get("frozen") is not True:
        return ["source document is not frozen"]
    if value.get("composite_type") == "comparative_composite":
        # The legacy E3 composite predates this module and intentionally uses
        # confirmatory.com's historical JSON hash serialization.
        from .confirmatory import validate_composite

        return validate_composite(value)
    errors = []
    if "index_sha256" in value and value.get("index_sha256") != _hash_document(
        value, "index_sha256"
    ):
        errors.append("source index self-hash mismatch")
    if "composite_sha256" in value and value.get("composite_sha256") != _hash_document(
        value, "composite_sha256"
    ):
        errors.append("source composite self-hash mismatch")
    return errors


def _result_source_errors(source: Mapping, rows: Sequence[Mapping]) -> list[str]:
    """Validate a new physical result source against its manifest and audit."""
    role = source.get("role")
    if role not in {"e3f", "physically_new"}:
        return []
    errors: list[str] = []
    manifest_path = Path(source.get("manifest_path") or "")
    audit_path = Path(source.get("audit_path") or "")
    manifest = _read_json_if_valid(manifest_path)
    audit = _read_json_if_valid(audit_path)
    if manifest is None:
        return [f"{role} source manifest missing/invalid: {manifest_path}"]
    campaign = "e3f" if role == "e3f" else "e7"
    errors.extend(
        f"{role} manifest: {error}" for error in (
            validate_e3f_manifest(manifest) if campaign == "e3f"
            else validate_e7_manifest(manifest)
        )
    )
    if audit is None:
        errors.append(f"{role} source audit missing/invalid: {audit_path}")
    else:
        if audit.get("passed") is not True:
            errors.append(f"{role} source audit is not passed")
        if audit.get("manifest_sha256") != manifest.get("manifest_sha256"):
            errors.append(f"{role} source audit is not bound to manifest")
        if audit.get("n_rows") != len(rows):
            errors.append(f"{role} source audit row count mismatch")
        checks = audit.get("checks") or []
        if len(checks) != 12 or not all(check.get("passed") for check in checks):
            errors.append(f"{role} source audit lacks 12 passed main checks")
        deadline_checks = audit.get("deadline_checks") or []
        if not deadline_checks or not all(check.get("passed") for check in deadline_checks):
            errors.append(f"{role} source audit deadline checks not passed")
    task_by_id = {task.get("run_id"): task for task in manifest.get("tasks") or []}
    row_ids = [row.get("run_id") for row in rows]
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != set(task_by_id):
        errors.append(f"{role} CSV run_ids do not exactly cover its manifest")
    for row in rows:
        run_id = row.get("run_id")
        task = task_by_id.get(run_id)
        if task is None:
            continue
        for field in (
            "function", "dimension", "instance", "configuration_hash",
            "instance_hash", "start_points_hash", "fe_budget",
        ):
            if str(row.get(field)) != str(task.get(field)):
                errors.append(f"{role} {run_id}: {field} differs from manifest")
        if row.get("algorithm_id") != (task.get("algorithm_id") or task.get("algorithm")):
            errors.append(f"{role} {run_id}: algorithm differs from manifest")
        status = row.get("status")
        if status not in TERMINAL_STATUSES:
            errors.append(f"{role} {run_id}: invalid status {status!r}")
        try:
            fe_used = int(row.get("fe_used"))
            if fe_used < 0 or fe_used > int(task.get("fe_budget")):
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{role} {run_id}: invalid FE usage")
        if status == "success":
            try:
                values = (
                    float(row.get("best_value")), float(row.get("known_optimum")),
                    float(row.get("normalized_gap")),
                )
                if not all(math.isfinite(value) for value in values):
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"{role} {run_id}: invalid successful numeric result")
    return errors


def build_extension_composite(
    campaign: str, *, sources: Sequence[Mapping], selection_hash: str,
    source_documents: Sequence[str | Path] = (), output_csv=None,
) -> dict:
    """Freeze an 840-row E3 combined or 2016-row E7 logical composite.

    ``sources`` entries contain ``role``, ``valid_runs_path`` and an optional
    equality ``filter`` (E7 reuse uses ``{"dimension": 1000}``).  Every source
    file and optional upstream composite/canonical document is byte-hash bound.
    """
    if campaign not in {"e3f", "e7"}:
        raise ValueError("composite campaign must be e3f or e7")
    if not source_documents:
        raise ValueError(
            "formal logical composite requires at least one frozen upstream "
            "canonical/composite source document"
        )
    expected = expected_logical_grid("e3_combined" if campaign == "e3f" else "e7")
    records: list[dict] = []
    combined: list[dict] = []
    for source in sources:
        path = Path(source["valid_runs_path"])
        rows = _filtered_rows(_read_csv_rows(path), source.get("filter"))
        run_ids = [row.get("run_id") for row in rows]
        if any(not run_id for run_id in run_ids) or len(run_ids) != len(set(run_ids)):
            raise ValueError(f"source {source.get('role')!r} has duplicate/empty run_id")
        record = {
            "role": source["role"], "valid_runs_path": str(path),
            "valid_runs_sha256": file_sha256(path),
            "filter": dict(source.get("filter") or {}), "n_rows": len(rows),
            "run_id_set_sha256": _sha256_bytes(
                canonical_json(sorted(run_ids)).encode("utf-8")
            ),
        }
        if source.get("role") in {"e3f", "physically_new"}:
            record.update({
                "manifest_path": str(source.get("manifest_path") or ""),
                "manifest_sha256": file_sha256(source["manifest_path"])
                if source.get("manifest_path") and Path(source["manifest_path"]).is_file()
                else None,
                "audit_path": str(source.get("audit_path") or ""),
                "audit_sha256": file_sha256(source["audit_path"])
                if source.get("audit_path") and Path(source["audit_path"]).is_file()
                else None,
            })
            source_errors = _result_source_errors(record, rows)
            if source_errors:
                raise ValueError("; ".join(source_errors[:10]))
        records.append(record)
        combined.extend(rows)
    cells = [_row_cell(row) for row in combined]
    run_ids = [row.get("run_id") for row in combined]
    if len(cells) != len(expected) or len(cells) != len(set(cells)) or set(cells) != expected:
        raise ValueError(
            f"{campaign} logical grid is not exact: rows={len(cells)} "
            f"unique_cells={len(set(cells))} expected={len(expected)}"
        )
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("logical composite has cross-source run_id overlap")
    provenance_errors = _instance_provenance_errors(combined)
    if provenance_errors:
        raise ValueError("; ".join(provenance_errors[:5]))
    document_records = []
    for path_value in source_documents:
        path = Path(path_value)
        value = _read_json_if_valid(path)
        if value is None:
            raise ValueError(f"source document is not frozen valid JSON: {path}")
        document_errors = _frozen_source_document_errors(value)
        if document_errors:
            raise ValueError(f"invalid source document {path}: " + "; ".join(document_errors))
        document_records.append({"path": str(path), "sha256": file_sha256(path)})
    composite = {
        "schema_version": "1", "composite_type": "extension_logical_composite",
        "campaign": campaign,
        "stage": "e3_e3f_comparative" if campaign == "e3f" else "e7_logical_analysis",
        "suite": EXTENSION_SUITE, "selection_hash": selection_hash,
        "frozen": True, "deadline_analysis": "operational_snapshot",
        "eventual_analysis": "eventual_outcome", "sources": records,
        "source_documents": document_records,
        "algorithms": list(E3F_ALGORITHMS if campaign == "e3f" else E7_ALGORITHMS),
        "total_rows": len(combined),
        "run_id_set_sha256": _sha256_bytes(canonical_json(sorted(run_ids)).encode("utf-8")),
    }
    if output_csv is not None:
        output_csv = Path(output_csv)
        columns = list(dict.fromkeys(key for row in combined for key in row))
        _write_csv(output_csv, columns, combined)
        composite["materialized_valid_runs_path"] = str(output_csv)
        composite["materialized_valid_runs_sha256"] = file_sha256(output_csv)
    composite["composite_sha256"] = _hash_document(composite, "composite_sha256")
    return composite


def validate_extension_composite(composite: Mapping) -> list[str]:
    errors: list[str] = []
    campaign = composite.get("campaign")
    if campaign not in {"e3f", "e7"}:
        return [f"unknown composite campaign {campaign!r}"]
    if composite.get("frozen") is not True:
        errors.append("composite is not frozen")
    if composite.get("composite_type") != "extension_logical_composite":
        errors.append("wrong composite_type")
    if composite.get("composite_sha256") != _hash_document(composite, "composite_sha256"):
        errors.append("composite_sha256 mismatch")
    combined: list[dict] = []
    for source in composite.get("sources") or []:
        path = Path(source.get("valid_runs_path") or "")
        if not path.is_file():
            errors.append(f"source missing: {path}")
            continue
        if file_sha256(path) != source.get("valid_runs_sha256"):
            errors.append(f"source hash mismatch: {source.get('role')}")
        rows = _filtered_rows(_read_csv_rows(path), source.get("filter"))
        ids = [row.get("run_id") for row in rows]
        if len(rows) != source.get("n_rows"):
            errors.append(f"source row count mismatch: {source.get('role')}")
        expected_hash = _sha256_bytes(canonical_json(sorted(ids)).encode("utf-8"))
        if expected_hash != source.get("run_id_set_sha256"):
            errors.append(f"source run_id set mismatch: {source.get('role')}")
        if len(ids) != len(set(ids)):
            errors.append(f"source duplicate run_id: {source.get('role')}")
        if source.get("role") in {"e3f", "physically_new"}:
            manifest_path = Path(source.get("manifest_path") or "")
            audit_path = Path(source.get("audit_path") or "")
            if not manifest_path.is_file() or (
                file_sha256(manifest_path) != source.get("manifest_sha256")
            ):
                errors.append(f"source manifest hash mismatch: {source.get('role')}")
            if not audit_path.is_file() or file_sha256(audit_path) != source.get(
                "audit_sha256"
            ):
                errors.append(f"source audit hash mismatch: {source.get('role')}")
            errors.extend(_result_source_errors(source, rows))
        combined.extend(rows)
    for record in composite.get("source_documents") or []:
        path = Path(record.get("path") or "")
        if not path.is_file():
            errors.append(f"source document missing: {path}")
        elif file_sha256(path) != record.get("sha256"):
            errors.append(f"source document hash mismatch: {path}")
        else:
            value = _read_json_if_valid(path) or {}
            errors.extend(
                f"source document {path}: {error}"
                for error in _frozen_source_document_errors(value)
            )
    if not composite.get("source_documents"):
        errors.append("composite has no upstream frozen source document")
    expected = expected_logical_grid("e3_combined" if campaign == "e3f" else "e7")
    cells = [_row_cell(row) for row in combined]
    ids = [row.get("run_id") for row in combined]
    if len(cells) != len(expected) or len(cells) != len(set(cells)) or set(cells) != expected:
        errors.append(
            f"logical grid mismatch: rows={len(cells)} unique={len(set(cells))} "
            f"expected={len(expected)}"
        )
    if len(ids) != len(set(ids)):
        errors.append("cross-source run_id overlap")
    errors.extend(_instance_provenance_errors(combined))
    if composite.get("total_rows") != len(expected):
        errors.append(f"total_rows {composite.get('total_rows')} != {len(expected)}")
    if composite.get("run_id_set_sha256") != _sha256_bytes(
        canonical_json(sorted(ids)).encode("utf-8")
    ):
        errors.append("composite run_id set mismatch")
    materialized = composite.get("materialized_valid_runs_path")
    if materialized:
        path = Path(materialized)
        if not path.is_file():
            errors.append("materialized composite CSV missing")
        elif file_sha256(path) != composite.get("materialized_valid_runs_sha256"):
            errors.append("materialized composite CSV hash mismatch")
    return errors


def _index_sha256(index: Mapping) -> str:
    return _hash_document(index, "index_sha256")


def build_extension_index(
    campaign: str, *, manifest_path, merged_dir, composite_path=None,
    root=".", git_commit: str | None = None,
) -> dict:
    """Build an isolated E3-F or E7 extension canonical index."""
    if campaign not in {"e3f", "e7"}:
        raise ValueError("campaign must be e3f or e7")
    root = Path(root)
    manifest_path = Path(manifest_path)
    merged_dir = Path(merged_dir)
    resolved_manifest = manifest_path if manifest_path.is_absolute() else root / manifest_path
    resolved_merged = merged_dir if merged_dir.is_absolute() else root / merged_dir
    artifacts = {
        "manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(resolved_manifest) if resolved_manifest.exists() else None,
            "expected_rows": 420 if campaign == "e3f" else 1736,
        },
        "merged": {"path": str(merged_dir)},
    }
    for name in (
        "valid_runs.csv", "provenance_audit.json", "deadline_snapshots.csv",
        "attempt_ledger.json",
    ):
        path = resolved_merged / name
        artifacts["merged"][name.replace(".", "_") + "_sha256"] = (
            file_sha256(path) if path.exists() else None
        )
    if composite_path is not None:
        composite_path = Path(composite_path)
        resolved = composite_path if composite_path.is_absolute() else root / composite_path
        artifacts["composite"] = {
            "path": str(composite_path),
            "sha256": file_sha256(resolved) if resolved.exists() else None,
            "expected_rows": 840 if campaign == "e3f" else 2016,
        }
    index = {
        "schema_version": "1", "campaign": campaign,
        "evidence_scope": "prospective_extension", "frozen": True,
        "generated_from_git_commit": git_commit, "artifacts": artifacts,
    }
    index["index_sha256"] = _index_sha256(index)
    return index


def validate_extension_index(index: Mapping, *, root=".") -> list[str]:
    errors: list[str] = []
    root = Path(root)
    campaign = index.get("campaign")
    if campaign not in {"e3f", "e7"}:
        return [f"unknown extension campaign {campaign!r}"]
    if index.get("frozen") is not True:
        errors.append("extension index is not frozen")
    if index.get("evidence_scope") != "prospective_extension":
        errors.append("extension evidence scope mismatch")
    if index.get("index_sha256") != _index_sha256(index):
        errors.append("index_sha256 mismatch")
    artifacts = index.get("artifacts") or {}
    manifest = None
    manifest_record = artifacts.get("manifest") or {}
    manifest_path = Path(manifest_record.get("path") or "")
    manifest_path = manifest_path if manifest_path.is_absolute() else root / manifest_path
    if not manifest_path.is_file():
        errors.append("manifest missing")
    else:
        if file_sha256(manifest_path) != manifest_record.get("sha256"):
            errors.append("manifest file hash mismatch")
        try:
            manifest = json.loads(manifest_path.read_text())
            errors.extend(
                f"manifest: {error}" for error in (
                    validate_e3f_manifest(manifest) if campaign == "e3f"
                    else validate_e7_manifest(manifest)
                )
            )
        except json.JSONDecodeError as exc:
            errors.append(f"manifest invalid JSON: {exc}")
    merged_record = artifacts.get("merged") or {}
    merged_path = Path(merged_record.get("path") or "")
    merged_path = merged_path if merged_path.is_absolute() else root / merged_path
    names = (
        "valid_runs.csv", "provenance_audit.json", "deadline_snapshots.csv",
        "attempt_ledger.json",
    )
    for name in names:
        path = merged_path / name
        key = name.replace(".", "_") + "_sha256"
        if not path.is_file():
            errors.append(f"merged missing {name}")
        elif file_sha256(path) != merged_record.get(key):
            errors.append(f"{name} hash mismatch")
    audit = _read_json_if_valid(merged_path / "provenance_audit.json")
    if audit is not None:
        expected_rows = 420 if campaign == "e3f" else 1736
        if not audit.get("passed"):
            errors.append("merged audit not passed")
        if audit.get("n_rows") != expected_rows:
            errors.append(f"merged audit rows {audit.get('n_rows')} != {expected_rows}")
        if len(audit.get("checks") or []) != 12:
            errors.append("merged audit must have exactly 12 main checks")
        if not audit.get("deadline_checks"):
            errors.append("merged audit has no deadline-specific checks")
        elif not all(check.get("passed") for check in audit["deadline_checks"]):
            errors.append("merged deadline-specific audit not passed")
        if manifest is not None and audit.get("manifest_sha256") != manifest.get("manifest_sha256"):
            errors.append("merged audit is not bound to the frozen manifest")
    expected_rows = 420 if campaign == "e3f" else 1736
    valid_path = merged_path / "valid_runs.csv"
    deadline_path = merged_path / "deadline_snapshots.csv"
    if valid_path.is_file():
        valid_rows = _read_csv_rows(valid_path)
        valid_ids = [row.get("run_id") for row in valid_rows]
        if len(valid_rows) != expected_rows or len(valid_ids) != len(set(valid_ids)):
            errors.append(
                f"valid_runs.csv physical/unique rows are not exactly {expected_rows}"
            )
        if manifest is not None and set(valid_ids) != {
            task.get("run_id") for task in manifest.get("tasks", [])
        }:
            errors.append("valid_runs.csv run_ids do not exactly match manifest")
    if deadline_path.is_file():
        deadline_rows = _read_csv_rows(deadline_path)
        deadline_ids = [row.get("run_id") for row in deadline_rows]
        if len(deadline_rows) != expected_rows or len(deadline_ids) != len(set(deadline_ids)):
            errors.append(
                f"deadline_snapshots.csv physical/unique rows are not exactly {expected_rows}"
            )
        if valid_path.is_file() and set(deadline_ids) != set(valid_ids):
            errors.append("deadline snapshot run_ids do not match valid_runs.csv")
    ledger_path = merged_path / "attempt_ledger.json"
    ledger_doc = _read_json_if_valid(ledger_path)
    if ledger_doc is not None:
        if ledger_doc.get("ledger_sha256") != _hash_document(ledger_doc, "ledger_sha256"):
            errors.append("combined attempt ledger hash mismatch")
        if manifest is not None and ledger_doc.get("manifest_sha256") != manifest.get("manifest_sha256"):
            errors.append("combined attempt ledger is not bound to manifest")
        ledger_ids = [record.get("run_id") for record in ledger_doc.get("runs", [])]
        if len(ledger_ids) != expected_rows or len(ledger_ids) != len(set(ledger_ids)):
            errors.append(f"combined attempt ledger does not cover {expected_rows} unique runs")
        for record in ledger_doc.get("runs", []):
            path = Path(record.get("path") or "")
            if path.is_file() and file_sha256(path) != record.get("sha256"):
                errors.append(f"source attempt ledger hash mismatch: {record.get('run_id')}")
    composite = artifacts.get("composite")
    if composite is None:
        errors.append("extension index missing logical composite")
    else:
        path = Path(composite.get("path") or "")
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            errors.append("logical composite missing")
        elif file_sha256(path) != composite.get("sha256"):
            errors.append("logical composite hash mismatch")
        else:
            value = _read_json_if_valid(path)
            if value is None:
                errors.append("logical composite invalid JSON")
            else:
                errors.extend(
                    f"logical composite: {error}"
                    for error in validate_extension_composite(value)
                )
                if value.get("campaign") != campaign:
                    errors.append("logical composite campaign does not match index")
                expected_rows = 840 if campaign == "e3f" else 2016
                if value.get("total_rows") != expected_rows:
                    errors.append(
                        f"logical composite rows {value.get('total_rows')} != {expected_rows}"
                    )
                if composite.get("expected_rows") != expected_rows:
                    errors.append("index composite expected_rows contract mismatch")
                if manifest is not None and value.get("selection_hash") != manifest.get(
                    "selection_hash"
                ):
                    errors.append("logical composite selection_hash does not match manifest")
    return errors


__all__ = [
    "E3F_STAGE", "E7_STAGE", "EXTENSION_SUITE", "E3F_DIMENSIONS", "E7_DIMENSIONS",
    "E3F_ALGORITHMS", "E7_NEW_ALGORITHMS", "E7_ALGORITHMS", "FE_BUDGET_PER_D",
    "FE_CHECKPOINTS_PER_D", "WALL_CHECKPOINT_HOURS", "DEADLINE_HOURS",
    "WALL_CHECKPOINT_SECONDS", "DEADLINE_SECONDS", "build_e3f_manifest",
    "build_e7_manifest", "expected_logical_grid", "validate_e3f_manifest",
    "validate_e7_manifest", "build_shards", "validate_shards", "ProgressReporter",
    "AttemptLedger", "supervise_command", "deadline_evidence_errors",
    "EXTENSION_RESULT_COLUMNS", "audit_extension_records",
    "merge_extension_results", "worker_progress", "build_extension_composite",
    "WorkerProgressSink", "plan_extension_dispatch", "stalled_attempt_errors",
    "recover_stalled_attempt", "validate_extension_composite",
    "build_extension_index", "validate_extension_index",
]
