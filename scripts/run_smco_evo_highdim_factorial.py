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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from smco.highdim_instances import load_instance, load_starts, starts_filename
from smco.highdim_worker import run_task
from smco.experiment_manifests import load_manifest, verify_manifest
from smco.confirmatory import enforce_confirmatory, is_run_complete, plan_batch

_THIS_SCRIPT = Path(__file__).resolve()
_R_WORKER = _THIS_SCRIPT.parent / "run_smco_evo_highdim_r.R"


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
        n_starts = int(task.get("n_starts", 8))
        starts_file = starts_filename(inst_dir, n_starts)
        actual = _sha256_file(inst_dir / starts_file)
        if actual != expected_starts_hash:
            raise ValueError(
                f"start_points_hash mismatch (n_starts={n_starts}): "
                f"task={expected_starts_hash!r} artifact={actual!r}"
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
        starts = load_starts(inst_dir, n_starts=int(task.get("n_starts", 8)))
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


def load_manifest_tasks(
    manifest_path,
    *,
    only_language=None,
    only_dims=None,
    only_run_ids=None,
) -> list[dict]:
    """Load + verify a frozen manifest and return (optionally filtered) tasks."""
    manifest = load_manifest(manifest_path)
    verify_manifest(manifest)
    tasks = list(manifest.get("tasks", []))
    if only_language:
        tasks = [t for t in tasks if t.get("language") == only_language]
    if only_dims:
        wanted = {int(d) for d in only_dims}
        tasks = [t for t in tasks if int(t["dimension"]) in wanted]
    if only_run_ids:
        wanted = set(only_run_ids)
        tasks = [t for t in tasks if t["run_id"] in wanted]
    return tasks


def _worker_command(task, task_json, instance_root, result_dir, log_dir) -> list[str]:
    common = [
        "--task", str(task_json),
        "--instance-root", str(instance_root),
        "--result-dir", str(result_dir),
        "--log-dir", str(log_dir),
    ]
    if task.get("language") == "python":
        return [sys.executable, str(_THIS_SCRIPT)] + common
    return ["Rscript", str(_R_WORKER)] + common


def _dispatch_one(task, instance_root, result_dir, log_dir, task_dir, wall_time_cap):
    run_id = task["run_id"]
    task_json = Path(task_dir) / f"{run_id}.task.json"
    task_json.write_text(json.dumps(task))
    cmd = _worker_command(task, task_json, instance_root, result_dir, log_dir)
    try:
        proc = subprocess.run(cmd, timeout=wall_time_cap, capture_output=True, text=True)
        status = "success" if proc.returncode == 0 else "worker_nonzero"
    except subprocess.TimeoutExpired:
        _atomic_write_json(
            Path(result_dir) / f"{run_id}.json",
            {"run_id": run_id, "status": "timeout",
             "failure_reason": f"wall_time_cap={wall_time_cap}s exceeded"},
        )
        status = "timeout"
    except FileNotFoundError as exc:
        _atomic_write_json(
            Path(result_dir) / f"{run_id}.json",
            {"run_id": run_id, "status": "infra_failure",
             "failure_reason": f"worker executable not found: {exc}"},
        )
        status = "infra_failure"
    return run_id, status


def run_batch(
    manifest_path,
    result_dir,
    instance_root,
    *,
    workers=1,
    resume=True,
    dry_run=False,
    wall_time_cap=None,
    only_language=None,
    only_dims=None,
    only_run_ids=None,
    log_dir=None,
    confirmatory=False,
    selection=None,
) -> dict:
    """Dispatch a manifest's tasks to Python/R worker subprocesses.

    Resume: tasks whose ``raw/<run_id>.json`` reports ``status=success`` are
    skipped (plan 9 / contract resume semantics). ``wall_time_cap`` per task is
    enforced by the outer subprocess manager (Task 8 process-isolation rule).
    ``confirmatory=True`` enforces the Gate-F checks (frozen + hash + selection
    winner in manifest) before dispatching.
    """
    if confirmatory:
        enforce_confirmatory(load_manifest(manifest_path), selection=selection)
    tasks = load_manifest_tasks(
        manifest_path, only_language=only_language,
        only_dims=only_dims, only_run_ids=only_run_ids,
    )
    plan = plan_batch(tasks, result_dir)
    if dry_run:
        plan["dry_run"] = True
        plan["dispatched"] = 0
        return plan

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir) if log_dir else result_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    task_dir = result_dir / "_tasks"
    task_dir.mkdir(exist_ok=True)

    todos = [t for t in tasks if not (resume and is_run_complete(result_dir, t["run_id"]))]
    statuses: dict[str, str] = {}
    if todos:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = {
                executor.submit(
                    _dispatch_one, t, str(instance_root), str(result_dir),
                    str(log_dir), task_dir, wall_time_cap,
                ): t
                for t in todos
            }
            for future in as_completed(futures):
                run_id, status = future.result()
                statuses[run_id] = status
    plan["dispatched"] = len(todos)
    plan["statuses"] = statuses
    return plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None, help="Single task JSON (worker mode).")
    parser.add_argument("--manifest", default=None, help="Frozen manifest JSON (batch mode).")
    parser.add_argument("--instance-root", required=True, help="Root dir for instance artifacts.")
    parser.add_argument("--result-dir", required=True, help="Directory for raw/<run_id>.json outputs.")
    parser.add_argument("--log-dir", default=None, help="Directory for per-run logs.")
    # single-task provenance
    parser.add_argument("--machine-id", default=None, help="Defaults to hostname.")
    parser.add_argument("--git-commit", default=None, help="Defaults to current HEAD.")
    parser.add_argument("--environment-hash", default=None, help="Defaults to a py/numpy hash.")
    # batch mode
    parser.add_argument("--workers", type=int, default=1, help="Concurrent worker subprocesses.")
    parser.add_argument("--only-language", default=None, choices=["python", "r"], help="Shard filter.")
    parser.add_argument("--only-dims", nargs="+", type=int, default=None, help="Shard filter.")
    parser.add_argument("--only-run-ids", nargs="+", default=None, help="Shard filter.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Skip completed runs (default).")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Re-run everything.")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; dispatch nothing.")
    parser.add_argument("--validate-only", action="store_true", help="Report completed/missing; run nothing.")
    parser.add_argument("--wall-time-cap", type=int, default=None, help="Per-task wall-time cap (seconds).")
    parser.add_argument("--confirmatory", action="store_true", help="Enforce Gate-F checks (frozen/hash/selection) before dispatch.")
    parser.add_argument("--selection", default=None, help="selection.json path (used with --confirmatory).")
    args = parser.parse_args(argv)

    if args.manifest:
        tasks = load_manifest_tasks(
            args.manifest, only_language=args.only_language,
            only_dims=args.only_dims, only_run_ids=args.only_run_ids,
        )
        if args.validate_only:
            print(json.dumps(plan_batch(tasks, args.result_dir), indent=2))
            return 0
        summary = run_batch(
            args.manifest, args.result_dir, args.instance_root,
            workers=args.workers, resume=args.resume, dry_run=args.dry_run,
            wall_time_cap=args.wall_time_cap, only_language=args.only_language,
            only_dims=args.only_dims, only_run_ids=args.only_run_ids, log_dir=args.log_dir,
            confirmatory=args.confirmatory,
            selection=json.loads(Path(args.selection).read_text()) if args.selection else None,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.task:
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

    parser.error("either --task (worker) or --manifest (batch) is required")
    return 1


if __name__ == "__main__":
    sys.exit(main())
