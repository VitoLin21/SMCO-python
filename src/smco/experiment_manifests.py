"""Immutable experiment manifests for the SMCO-EVO high-dim paper (Task 7).

A manifest is a JSON document listing one canonical task per run. Each task
carries a stable ``run_id`` (delegated to :func:`paper_contract.compute_run_id`)
and a dimension-independent ``configuration_hash`` that captures the algorithm
parameters only; ``fe_budget`` and ``checkpoints`` are run-level and scale with
the dimension, so the same algorithm at d=200 and d=5000 shares its
``configuration_hash`` but differs in ``run_id``.

The manifest itself carries a ``manifest_sha256`` computed over its content
(excluding the hash field). ``freeze_manifest`` flips ``frozen=True`` and
recomputes the hash; ``verify_manifest`` recomputes it and raises if the
content has been mutated after freezing, so a runner can refuse a tampered
manifest (Gate C / plan Task 10 enforced checks).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .highdim_instances import GENERATOR_VERSION
from .paper_contract import (
    NONE_TOKEN,
    SCHEMA_VERSION,
    STRATEGIES,
    build_algorithm_id,
    canonical_json,
    compute_configuration_hash,
    compute_run_id,
    format_cfg_float,
)

LANGUAGES = ("python", "r")
FAMILIES = ("smco", "smco_refine", "smco_boost_refine")
SEMANTICS = ("state_preserving", "restart")

# Frozen EVO defaults (experiment plan section 4.2).
DEFAULT_EVO_STRATEGY = "rand1bin"
DEFAULT_EVO_POINTS = (0.5, 0.75)
DEFAULT_ELIMINATION_RATE = 0.25
DEFAULT_DE_FACTOR = 0.8
DEFAULT_DE_CROSSOVER = 0.7
DEFAULT_N_STARTS = 8
DEFAULT_REFINE_RATIO = 0.5


def build_algorithm_config(
    language: str,
    family: str,
    evolutionary: bool,
    state_semantics: str,
    *,
    evolution_strategy: str,
    evolution_points,
    elimination_rate: float,
    de_factor: float,
    de_crossover: float,
    n_starts: int,
    refine_ratio: float = DEFAULT_REFINE_RATIO,
) -> dict:
    """Build an algorithm config dict with ``algorithm_id`` and ``configuration_hash``.

    The hash covers algorithm parameters only (no ``fe_budget``/``checkpoints``),
    so it is invariant to dimension.
    """
    evolutionary = bool(evolutionary)
    if evolutionary:
        if state_semantics not in SEMANTICS:
            raise ValueError(f"state_semantics must be one of {SEMANTICS}")
        if evolution_strategy not in STRATEGIES:
            raise ValueError(f"evolution_strategy must be one of {STRATEGIES}")
        sem = state_semantics
        strategy = evolution_strategy
        evo_points = [format_cfg_float(p) for p in evolution_points]
    else:
        sem = NONE_TOKEN
        strategy = NONE_TOKEN
        evo_points = [NONE_TOKEN]

    algorithm_id = build_algorithm_id(
        language, family, evolutionary, None if not evolutionary else sem
    )
    config = {
        "algorithm_id": algorithm_id,
        "language": language,
        "family": family,
        "evolutionary": "true" if evolutionary else "false",
        "state_semantics": sem,
        "evolution_strategy": strategy,
        "evolution_points": evo_points,
        "elimination_rate": format_cfg_float(elimination_rate),
        "de_factor": format_cfg_float(de_factor),
        "de_crossover": format_cfg_float(de_crossover),
        "n_starts": int(n_starts),
        "refine_ratio": format_cfg_float(refine_ratio),
    }
    # Hash over the config WITHOUT the configuration_hash field (no circularity).
    config["configuration_hash"] = compute_configuration_hash(config)
    return config


def e1_algorithm_configs() -> list[dict]:
    """The 18 E1 candidate configurations: 12 EVO (2 lang x 2 sem x 3 family) + 6 base."""
    configs: list[dict] = []
    for language in LANGUAGES:
        for family in FAMILIES:
            configs.append(
                build_algorithm_config(
                    language, family, False, NONE_TOKEN,
                    evolution_strategy=NONE_TOKEN,
                    evolution_points=(),
                    elimination_rate=DEFAULT_ELIMINATION_RATE,
                    de_factor=DEFAULT_DE_FACTOR,
                    de_crossover=DEFAULT_DE_CROSSOVER,
                    n_starts=DEFAULT_N_STARTS,
                )
            )
            for sem in SEMANTICS:
                configs.append(
                    build_algorithm_config(
                        language, family, True, sem,
                        evolution_strategy=DEFAULT_EVO_STRATEGY,
                        evolution_points=DEFAULT_EVO_POINTS,
                        elimination_rate=DEFAULT_ELIMINATION_RATE,
                        de_factor=DEFAULT_DE_FACTOR,
                        de_crossover=DEFAULT_DE_CROSSOVER,
                        n_starts=DEFAULT_N_STARTS,
                    )
                )
    return configs


def derive_seed(
    stage: str,
    suite: str,
    function: str,
    dimension: int,
    instance: int,
    replication: int,
    algorithm_id: str,
) -> int:
    """Stable 32-bit seed derived from the run key (independent of run order)."""
    key = canonical_json(
        {
            "stage": stage,
            "suite": suite,
            "function": function,
            "dimension": int(dimension),
            "instance": int(instance),
            "replication": int(replication),
            "algorithm_id": algorithm_id,
        }
    )
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def build_task(
    stage: str,
    suite: str,
    function: str,
    dimension: int,
    instance: int,
    replication: int,
    *,
    config: dict,
    fe_budget: int,
    checkpoints,
    seed: int,
    instance_artifact_dir: str | None = None,
    instance_hash: str | None = None,
    start_points_hash: str | None = None,
) -> dict:
    """Assemble one canonical task dict, computing its ``run_id``."""
    task = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "suite": suite,
        "function": function,
        "dimension": int(dimension),
        "instance": int(instance),
        "replication": int(replication),
        "algorithm_id": config["algorithm_id"],
        "language": config["language"],
        "family": config["family"],
        "evolutionary": config["evolutionary"],
        "state_semantics": config["state_semantics"],
        "evolution_strategy": config["evolution_strategy"],
        "n_starts": int(config["n_starts"]),
        "fe_budget": int(fe_budget),
        "checkpoints": [int(c) for c in checkpoints],
        "seed": int(seed),
        "configuration_hash": config["configuration_hash"],
        "instance_artifact_dir": instance_artifact_dir,
        "instance_hash": instance_hash,
        "start_points_hash": start_points_hash,
        # Full algorithm-config snapshot for audit / reproducibility.
        "algorithm_config": {k: v for k, v in config.items() if k != "configuration_hash"},
    }
    task["run_id"] = compute_run_id(task)
    return task


def _select_start_points_hash(entry: dict, n_starts: int):
    """Pick the start_points_hash for the requested n_starts tier.

    n_starts=8 (or an entry without extra_starts) → the default starts hash;
    other tiers → the matching extra_starts hash.
    """
    if n_starts != 8:
        extra = (entry.get("extra_starts") or {}).get(str(int(n_starts)))
        if extra:
            return extra.get("hash")
    return entry.get("start_points_hash")


def expand_tasks(
    stage: str,
    suite: str,
    functions,
    dims,
    n_instances: int,
    configs,
    *,
    fe_budget_per_d: int,
    checkpoints_per_d,
    replication: int = 0,
    instance_index: dict | None = None,
) -> list[dict]:
    """Expand the (function x dim x instance x config) task grid.

    ``instance_index`` maps ``(function, dim, instance_id)`` to an entry with
    ``artifact_dir`` / ``transform_sha256`` / ``start_points_hash`` (produced by
    the Task 6 instance generator); when absent the provenance fields stay None.
    """
    tasks: list[dict] = []
    for function in functions:
        for dim in dims:
            dim = int(dim)
            fe_budget = int(fe_budget_per_d) * dim
            checkpoints = [int(c) * dim for c in checkpoints_per_d]
            for instance in range(n_instances):
                for config in configs:
                    provenance: dict = {}
                    if instance_index is not None:
                        entry = instance_index.get((function, dim, instance))
                        if entry is not None:
                            provenance = {
                                "instance_artifact_dir": entry.get("artifact_dir"),
                                "instance_hash": entry.get("transform_sha256"),
                                "start_points_hash": _select_start_points_hash(
                                    entry, int(config["n_starts"])
                                ),
                            }
                    seed = derive_seed(
                        stage, suite, function, dim, instance, replication,
                        config["algorithm_id"],
                    )
                    tasks.append(
                        build_task(
                            stage, suite, function, dim, instance, replication,
                            config=config, fe_budget=fe_budget, checkpoints=checkpoints,
                            seed=seed, **provenance,
                        )
                    )
    return tasks


# --- comparison baseline tasks (E3 / E1B; algorithm is a baseline name, not a SMCO algorithm_id) ---
def baseline_run_id(task: dict) -> str:
    """``run_id = 'b' + sha256(canonical_json(task_subset))[:16]`` for baselines."""
    digest = hashlib.sha256(canonical_json(
        {
            "stage": task["stage"],
            "suite": task["suite"],
            "function": task["function"],
            "dimension": int(task["dimension"]),
            "instance": int(task["instance"]),
            "algorithm": task["algorithm"],
            "fe_budget": int(task["fe_budget"]),
            "checkpoints": [int(c) for c in task["checkpoints"]],
            "seed": int(task["seed"]),
        }
    ).encode("utf-8")).hexdigest()
    return "b" + digest[:16]


def build_baseline_task(
    stage, suite, function, dim, instance, *, algorithm, fe_budget, checkpoints, seed,
    instance_artifact_dir=None, instance_hash=None, start_points_hash=None,
) -> dict:
    task = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "suite": suite,
        "function": function,
        "dimension": int(dim),
        "instance": int(instance),
        "algorithm": algorithm,
        "fe_budget": int(fe_budget),
        "checkpoints": [int(c) for c in checkpoints],
        "seed": int(seed),
        "instance_artifact_dir": instance_artifact_dir,
        "instance_hash": instance_hash,
        "start_points_hash": start_points_hash,
    }
    task["run_id"] = baseline_run_id(task)
    return task


def expand_baseline_tasks(
    stage, suite, functions, dims, n_instances, baselines, *,
    fe_budget_per_d, checkpoints_per_d, instance_index=None,
) -> list[dict]:
    tasks: list[dict] = []
    for function in functions:
        for dim in dims:
            dim = int(dim)
            fe_budget = int(fe_budget_per_d) * dim
            checkpoints = [int(c) * dim for c in checkpoints_per_d]
            for instance in range(n_instances):
                provenance: dict = {}
                if instance_index is not None:
                    entry = instance_index.get((function, dim, instance))
                    if entry is not None:
                        provenance = {
                            "instance_artifact_dir": entry.get("artifact_dir"),
                            "instance_hash": entry.get("transform_sha256"),
                            "start_points_hash": entry.get("start_points_hash"),
                        }
                for algorithm in baselines:
                    seed = derive_seed(stage, suite, function, dim, instance, 0, algorithm)
                    tasks.append(
                        build_baseline_task(
                            stage, suite, function, dim, instance, algorithm=algorithm,
                            fe_budget=fe_budget, checkpoints=checkpoints, seed=seed, **provenance,
                        )
                    )
    return tasks


def manifest_sha256(doc: dict) -> str:
    """SHA-256 over the manifest content, excluding the hash field itself."""
    payload = {k: v for k, v in doc.items() if k != "manifest_sha256"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_manifest(
    stage: str, suite: str, tasks, *, manifest_id: str | None = None, frozen: bool = False
) -> dict:
    doc = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest_id or f"{stage}__{suite}",
        "stage": stage,
        "suite": suite,
        "generator_version": GENERATOR_VERSION,
        "frozen": bool(frozen),
        "n_tasks": len(tasks),
        "tasks": [dict(t) for t in tasks],
    }
    doc["manifest_sha256"] = manifest_sha256(doc)
    return doc


def freeze_manifest(doc: dict) -> dict:
    """Return a frozen copy with ``frozen=True`` and a recomputed content hash."""
    frozen = dict(doc)
    frozen["frozen"] = True
    frozen["manifest_sha256"] = manifest_sha256(frozen)
    return frozen


def verify_manifest(doc: dict, *, expected_sha256: str | None = None) -> bool:
    """Recompute the content hash and raise if the manifest was mutated."""
    stored = doc.get("manifest_sha256")
    recomputed = manifest_sha256(doc)
    if stored != recomputed:
        raise ValueError(
            "manifest_sha256 mismatch: manifest content changed after hashing"
        )
    if expected_sha256 is not None and stored != expected_sha256:
        raise ValueError(
            f"manifest_sha256 {stored!r} does not match expected {expected_sha256!r}"
        )
    return True


def write_manifest(doc: dict, path) -> None:
    Path(path).write_text(json.dumps(doc, indent=2, ensure_ascii=False))


def load_manifest(path) -> dict:
    return json.loads(Path(path).read_text())


# Stages whose results count as confirmatory (is_confirmatory=True on the row).
_CONFIRMATORY_STAGES = {
    "e2_factorial_highdim",
    "e3_baselines_highdim",
    "e4_bbob_largescale",
    "e5_lowdim_check",
}

# Identity fields a result row must inherit unchanged from its manifest task.
_IDENTITY_KEYS = (
    "run_id",
    "configuration_hash",
    "algorithm_id",
    "stage",
    "suite",
    "function",
    "dimension",
    "instance",
    "replication",
    "seed",
    "language",
    "family",
    "evolutionary",
    "state_semantics",
    "evolution_strategy",
    "n_starts",
    "fe_budget",
)


def result_row_from_task(
    task: dict,
    *,
    best_value: float,
    fe_used: int,
    status: str = "success",
    known_optimum: float = 0.0,
    normalized_gap=None,
    checkpoint_fe=None,
    target_hit_fe=None,
    wall_time_sec: float = 0.0,
    peak_memory_mb: float = 0.0,
    failure_reason: str = NONE_TOKEN,
    supersedes_run_id: str = NONE_TOKEN,
    machine_id: str = "",
    git_commit: str = "",
    environment_hash: str = "",
    termination_reason: str = "evaluation_budget",
    fe_counts_by_event: str = "",
    objective_sense: str = "minimize",
    manifest_id: str = "",
) -> dict:
    """Derive a contract-valid result row from a manifest task plus its outcome.

    Fills every column in ``paper_contract.RESULT_COLUMNS``; identity fields
    come from the task, outcome fields from the keyword arguments.
    """
    if checkpoint_fe is None:
        checkpoint_fe = task["fe_budget"]
    if normalized_gap is None:
        normalized_gap = NONE_TOKEN
    th = target_hit_fe or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "stage": task["stage"],
        "suite": task["suite"],
        "function": task["function"],
        "dimension": task["dimension"],
        "instance": task["instance"],
        "replication": task["replication"],
        "seed": task["seed"],
        "language": task["language"],
        "state_semantics": task["state_semantics"],
        "family": task["family"],
        "evolutionary": task["evolutionary"],
        "evolution_strategy": task["evolution_strategy"],
        "algorithm_id": task["algorithm_id"],
        "n_starts": task["n_starts"],
        "fe_budget": task["fe_budget"],
        "fe_used": int(fe_used),
        "checkpoint_fe": int(checkpoint_fe),
        "best_value": float(best_value),
        "known_optimum": float(known_optimum),
        "normalized_gap": normalized_gap,
        "objective_sense": objective_sense,
        "target_hit_fe_1e-1": th.get("1e-1", NONE_TOKEN),
        "target_hit_fe_1e-2": th.get("1e-2", NONE_TOKEN),
        "target_hit_fe_1e-3": th.get("1e-3", NONE_TOKEN),
        "target_hit_fe_1e-5": th.get("1e-5", NONE_TOKEN),
        "wall_time_sec": float(wall_time_sec),
        "peak_memory_mb": float(peak_memory_mb),
        "status": status,
        "failure_reason": failure_reason,
        "is_confirmatory": task["stage"] in _CONFIRMATORY_STAGES,
        "supersedes_run_id": supersedes_run_id,
        "machine_id": machine_id,
        "git_commit": git_commit,
        "environment_hash": environment_hash,
        "start_points_hash": task["start_points_hash"] or NONE_TOKEN,
        "instance_hash": task["instance_hash"] or NONE_TOKEN,
        "configuration_hash": task["configuration_hash"],
        "run_id": task["run_id"],
        "termination_reason": termination_reason,
        "fe_counts_by_event": fe_counts_by_event,
    }


def validate_result_against_task(row: dict, task: dict) -> list[str]:
    """Return identity-field mismatches between a result row and its task."""
    errors: list[str] = []
    for key in _IDENTITY_KEYS:
        if row.get(key) != task.get(key):
            errors.append(f"{key} mismatch: row={row.get(key)!r} task={task.get(key)!r}")
    return errors


__all__ = [
    "LANGUAGES",
    "FAMILIES",
    "SEMANTICS",
    "DEFAULT_EVO_STRATEGY",
    "DEFAULT_EVO_POINTS",
    "DEFAULT_N_STARTS",
    "build_algorithm_config",
    "e1_algorithm_configs",
    "derive_seed",
    "build_task",
    "expand_tasks",
    "manifest_sha256",
    "build_manifest",
    "freeze_manifest",
    "verify_manifest",
    "write_manifest",
    "load_manifest",
    "result_row_from_task",
    "validate_result_against_task",
]
