#!/usr/bin/env python
"""Python single-task high-dim worker CLI (Task 8).

Reads one canonical manifest task JSON, loads its instance artifact and shared
starts, runs the SMCO variant named by the task, and atomically writes a result
payload to ``<result-dir>/<run_id>.json`` plus a per-run log.

Single-thread BLAS/OpenMP is forced before importing NumPy/SMCO so each run is
process-isolated and reproducible (plan Task 8, process isolation).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

# Force single-threaded BLAS/OpenMP before importing numpy/SMCO.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np

from smco.highdim_instances import load_instance, load_starts
from smco.highdim_worker import run_task


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _resolve_instance_dir(instance_root: Path, task: dict) -> Path:
    rel = task.get("instance_artifact_dir")
    if not rel:
        raise ValueError(
            f"task {task.get('run_id')} has no instance_artifact_dir; "
            "cannot locate its instance artifact"
        )
    return Path(instance_root) / rel


def _verify_provenance(instance, starts: np.ndarray, inst_dir: Path, task: dict) -> None:
    expected_instance_hash = task.get("instance_hash")
    if expected_instance_hash:
        actual = instance.transform_spec.sha256()
        if actual != expected_instance_hash:
            raise ValueError(
                f"instance_hash mismatch: task={expected_instance_hash!r} artifact={actual!r}"
            )
    expected_starts_hash = task.get("start_points_hash")
    if expected_starts_hash:
        actual = _sha256_file(inst_dir / "starts.csv.gz")
        if actual != expected_starts_hash:
            raise ValueError(
                f"start_points_hash mismatch: task={expected_starts_hash!r} artifact={actual!r}"
            )
    if starts.shape[1] != instance.dimension:
        raise ValueError(
            f"starts have {starts.shape[1]} cols but instance dimension is {instance.dimension}"
        )


def _default_git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def _default_environment_hash() -> str:
    import platform

    payload = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    import hashlib

    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def run_task_file(
    task_path,
    *,
    instance_root,
    result_dir,
    log_dir=None,
    machine_id="",
    git_commit="",
    environment_hash="",
) -> int:
    """Run one task file end-to-end. Returns 0 on success, 1 on infra failure."""
    task_path = Path(task_path)
    task = json.loads(task_path.read_text())
    run_id = task["run_id"]

    result_dir = Path(result_dir)
    log_dir = Path(log_dir) if log_dir else result_dir.parent / "logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_dir / f"{run_id}.log", "w")

    def say(message: str) -> None:
        print(message)
        log_handle.write(message + "\n")
        log_handle.flush()

    say(
        f"[worker] start run_id={run_id} algo={task['algorithm_id']} "
        f"func={task['function']} d={task['dimension']} fe_budget={task['fe_budget']}"
    )
    try:
        inst_dir = _resolve_instance_dir(Path(instance_root), task)
        instance = load_instance(inst_dir)
        starts = load_starts(inst_dir)
        _verify_provenance(instance, starts, inst_dir, task)
        payload = run_task(
            task, instance, starts,
            machine_id=machine_id, git_commit=git_commit, environment_hash=environment_hash,
        )
    except Exception as exc:  # noqa: BLE001 - infra/loading failure must be reported, not raised
        say(f"[worker] INFRA_FAILURE {type(exc).__name__}: {exc}")
        log_handle.close()
        _atomic_write_json(
            result_dir / f"{run_id}.json",
            {
                "run_id": run_id,
                "status": "infra_failure",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "result_row": None,
            },
        )
        return 1

    _atomic_write_json(result_dir / f"{run_id}.json", payload)
    say(
        f"[worker] done status={payload['status']} fe_used={payload['fe_used']} "
        f"best={payload['best_value']:.6e} gap={payload['normalized_gap']:.4f}"
    )
    log_handle.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Path to the canonical task JSON.")
    parser.add_argument("--instance-root", required=True, help="Root dir for instance artifacts.")
    parser.add_argument("--result-dir", required=True, help="Directory for raw/<run_id>.json outputs.")
    parser.add_argument("--log-dir", default=None, help="Directory for per-run logs.")
    parser.add_argument("--machine-id", default=None, help="Defaults to hostname.")
    parser.add_argument("--git-commit", default=None, help="Defaults to current HEAD.")
    parser.add_argument("--environment-hash", default=None, help="Defaults to a py/numpy hash.")
    args = parser.parse_args(argv)

    return run_task_file(
        args.task,
        instance_root=args.instance_root,
        result_dir=args.result_dir,
        log_dir=args.log_dir,
        machine_id=args.machine_id if args.machine_id is not None else socket.gethostname(),
        git_commit=args.git_commit if args.git_commit is not None else _default_git_commit(),
        environment_hash=(
            args.environment_hash
            if args.environment_hash is not None
            else _default_environment_hash()
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
