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
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

from smco.baseline_worker import run_baseline_task
from smco.highdim_instances import load_instance, load_starts


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


def baseline_run_id(task: dict) -> str:
    key = f"{task['stage']}:{task['algorithm']}:{task['function']}:{task['dimension']}:{task['instance_id']}"
    return "b" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


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
    _atomic_write_json(result_dir / f"{run_id}.json", payload)
    say(f"[baseline] done status={payload['status']} fe_used={payload['fe_used']} best={payload['best_value']:.6e}")
    log_handle.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Comparison task JSON.")
    parser.add_argument("--instance-root", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args(argv)
    return run_baseline_file(
        args.task, instance_root=args.instance_root,
        result_dir=args.result_dir, log_dir=args.log_dir,
        machine_id=socket.gethostname(),
    )


if __name__ == "__main__":
    sys.exit(main())
