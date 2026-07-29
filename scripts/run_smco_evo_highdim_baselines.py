#!/usr/bin/env python
"""Comparison-baseline single-task CLI (Task 9 / E3).

Reads a comparison task JSON (algorithm/function/dimension/instance_id/
fe_budget/seed/checkpoints/instance_artifact_dir), loads the instance artifact,
runs the named baseline under the SMCO FE budget, and atomically writes a result
payload to <result-dir>/<run_id>.json. Mirrors run_smco_evo_highdim_factorial.py
but dispatches to :mod:`smco.baseline_worker`.

Task 10 adds E3 batch dispatch + a frozen baselines manifest on top of this.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

from smco.baseline_worker import run_baseline_task
from smco.confirmatory import enforce_confirmatory, is_run_complete, plan_batch
from smco.experiment_manifests import baseline_run_id, load_manifest, verify_manifest
from smco.highdim_instances import load_instance, load_starts

_THIS_SCRIPT = Path(__file__).resolve()


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def run_baseline_file(task_path, *, instance_root, result_dir, log_dir=None,
                      machine_id="", git_commit="", environment_hash="") -> int:
    task = json.loads(Path(task_path).read_text())
    run_id = baseline_run_id(task)
    result_dir = Path(result_dir)
    log_dir = Path(log_dir) if log_dir else result_dir.parent / "logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_dir / f"{run_id}.log", "w")

    def say(message):
        print(message)
        log_handle.write(message + "\n")
        log_handle.flush()

    say(f"[baseline] start {run_id} algo={task['algorithm']} func={task['function']} d={task['dimension']}")
    try:
        inst_dir = Path(instance_root) / task["instance_artifact_dir"]
        instance = load_instance(inst_dir)
        starts = load_starts(inst_dir)
        payload = run_baseline_task(
            task["algorithm"], instance, starts,
            fe_budget=int(task["fe_budget"]), seed=int(task["seed"]),
            checkpoints=task["checkpoints"], stage=task.get("stage", "e3_baselines_highdim"),
            machine_id=machine_id, git_commit=git_commit, environment_hash=environment_hash,
        )
    except Exception as exc:  # noqa: BLE001
        say(f"[baseline] INFRA_FAILURE {type(exc).__name__}: {exc}")
        log_handle.close()
        _atomic_write_json(
            result_dir / f"{run_id}.json",
            {"run_id": run_id, "algorithm_id": task.get("algorithm"),
             "status": "infra_failure", "failure_reason": f"{type(exc).__name__}: {exc}"},
        )
        return 1

    payload["run_id"] = run_id
    payload["task"] = task
    _atomic_write_json(result_dir / f"{run_id}.json", payload)
    say(f"[baseline] done status={payload['status']} fe_used={payload['fe_used']} best={payload['best_value']:.6e}")
    log_handle.close()
    return 0


def load_baseline_manifest_tasks(manifest_path, *, only_dims=None, only_run_ids=None):
    manifest = load_manifest(manifest_path)
    verify_manifest(manifest)
    tasks = list(manifest.get("tasks", []))
    if only_dims:
        wanted = {int(d) for d in only_dims}
        tasks = [t for t in tasks if int(t["dimension"]) in wanted]
    if only_run_ids:
        wanted = set(only_run_ids)
        tasks = [t for t in tasks if t["run_id"] in wanted]
    return tasks


def _dispatch_baseline(task, instance_root, result_dir, log_dir, task_dir, wall_time_cap):
    run_id = task["run_id"]
    task_json = Path(task_dir) / f"{run_id}.task.json"
    task_json.write_text(json.dumps(task))
    cmd = [
        sys.executable, str(_THIS_SCRIPT), "--task", str(task_json),
        "--instance-root", str(instance_root), "--result-dir", str(result_dir),
        "--log-dir", str(log_dir),
    ]
    try:
        proc = subprocess.run(cmd, timeout=wall_time_cap, capture_output=True, text=True)
        status = "success" if proc.returncode == 0 else "worker_nonzero"
    except subprocess.TimeoutExpired:
        _atomic_write_json(
            Path(result_dir) / f"{run_id}.json",
            {"run_id": run_id, "algorithm_id": task.get("algorithm"), "status": "timeout",
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


def run_baseline_batch(
    manifest_path, result_dir, instance_root, *, workers=1, resume=True,
    dry_run=False, wall_time_cap=None, only_dims=None, only_run_ids=None,
    log_dir=None, confirmatory=False,
) -> dict:
    if confirmatory:
        enforce_confirmatory(load_manifest(manifest_path))  # baselines: no SMCO selection
    tasks = load_baseline_manifest_tasks(
        manifest_path, only_dims=only_dims, only_run_ids=only_run_ids,
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
                    _dispatch_baseline, t, str(instance_root), str(result_dir),
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
    parser.add_argument("--task", default=None, help="Comparison task JSON (worker mode).")
    parser.add_argument("--manifest", default=None, help="Frozen baseline manifest (batch mode).")
    parser.add_argument("--instance-root", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only-dims", nargs="+", type=int, default=None)
    parser.add_argument("--only-run-ids", nargs="+", default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--wall-time-cap", type=int, default=None)
    parser.add_argument("--confirmatory", action="store_true", help="Enforce frozen/hash before dispatch.")
    args = parser.parse_args(argv)

    if args.manifest:
        tasks = load_baseline_manifest_tasks(
            args.manifest, only_dims=args.only_dims, only_run_ids=args.only_run_ids,
        )
        if args.validate_only:
            print(json.dumps(plan_batch(tasks, args.result_dir), indent=2))
            return 0
        summary = run_baseline_batch(
            args.manifest, args.result_dir, args.instance_root,
            workers=args.workers, resume=args.resume, dry_run=args.dry_run,
            wall_time_cap=args.wall_time_cap, only_dims=args.only_dims,
            only_run_ids=args.only_run_ids, log_dir=args.log_dir, confirmatory=args.confirmatory,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.task:
        return run_baseline_file(
            args.task, instance_root=args.instance_root,
            result_dir=args.result_dir, log_dir=args.log_dir,
            machine_id=socket.gethostname(),
        )

    parser.error("either --task (worker) or --manifest (batch) is required")
    return 1


if __name__ == "__main__":
    sys.exit(main())
