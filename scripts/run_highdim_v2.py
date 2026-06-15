#!/usr/bin/env python
"""High-dimensional SMCO-EVO vs SMCO comparison — multiprocessing version.

Dimensions: 500, 1000, 2000, 5000
Uses multiprocessing.Pool to distribute work across all CPU cores.

Usage:
  uv run python scripts/run_highdim_v2.py
  uv run python scripts/run_highdim_v2.py --quick
  uv run python scripts/run_highdim_v2.py --workers 32
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from smco import smco, smco_br, smco_br_evo, smco_evo, smco_r, smco_r_evo
from smco.test_functions import assign_config

SMCO_VARIANT_MAP = {
    "SMCO": smco, "SMCO_R": smco_r, "SMCO_BR": smco_br,
    "SMCO_EVO": smco_evo, "SMCO_R_EVO": smco_r_evo, "SMCO_BR_EVO": smco_br_evo,
}

PAIRS = [
    ("SMCO_R", "SMCO_R_EVO"),
    ("SMCO", "SMCO_EVO"),
    ("SMCO_BR", "SMCO_BR_EVO"),
]

DEFAULT_OPTIONS = {
    "iter_max": 300,
    "bounds_buffer": 0.05,
    "buffer_rand": True,
    "tol_conv": 1e-8,
}

FUNCS_BY_DIM = {
    500:  ["Rastrigin", "Ackley", "Rosenbrock", "Zakharov"],
    1000: ["Rastrigin", "Ackley", "Rosenbrock", "Zakharov"],
    2000: ["Rastrigin", "Ackley", "Rosenbrock", "Zakharov"],
    5000: ["Rastrigin", "Ackley", "Rosenbrock"],
}

REPS_BY_DIM = {500: 10, 1000: 5, 2000: 3, 5000: 2}


def _output_dir(base: str | None) -> Path:
    if base:
        p = Path(base)
    else:
        p = Path("result") / f"ultrahighdim-{date.today()}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _n_starts(dim: int) -> int:
    return max(3, int(math.ceil(math.sqrt(dim))))


def _algo_options(algo_name: str, base: dict, seed: int) -> dict:
    opts = dict(base)
    if algo_name in ("SMCO_R", "SMCO_BR", "SMCO_R_EVO", "SMCO_BR_EVO"):
        opts.setdefault("refine_ratio", 0.5)
    if algo_name in ("SMCO_BR", "SMCO_BR_EVO"):
        opts["iter_max"] = math.ceil(opts["iter_max"] / 2)
        opts.setdefault("iter_boost", 1000)
    else:
        opts.pop("iter_boost", None)
    opts["seed"] = seed
    return opts


def _save_csv(rows: list[dict], path: Path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Single task — runs one (algo, func, dim, rep) combination
# ---------------------------------------------------------------------------

def _run_task(task: dict) -> dict:
    """Worker function — runs in a child process."""
    func_name = task["func"]
    dim = task["dim"]
    algo_name = task["algo"]
    algo_seed = task["seed"]
    starts = task["starts"]
    lower = task["lower"]
    upper = task["upper"]
    record_history = task.get("record_history", False)

    config = assign_config(func_name, dim)
    f = config.f
    if config.sense == "min":
        f = lambda x, _f=config.f: -_f(x)

    opts = _algo_options(algo_name, dict(DEFAULT_OPTIONS), algo_seed)
    if record_history:
        opts["record_history"] = True

    t0 = time.perf_counter()
    result = SMCO_VARIANT_MAP[algo_name](f, lower, upper, start_points=starts, **opts)
    elapsed = time.perf_counter() - t0

    out = {
        "task_id": task["task_id"],
        "algo": algo_name,
        "func": func_name,
        "dim": dim,
        "rep": task["rep"],
        "fopt": result.best_result.f_optimal,
        "time": elapsed,
        "iterations": result.best_result.iterations,
        "pair": task.get("pair", ""),
    }

    if record_history and result.best_result.runmax_history is not None:
        out["history"] = result.best_result.runmax_history.tolist()

    return out


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def _generate_tasks(dims, funcs_by_dim, reps_by_dim, seed, algos_per_task=None,
                    record_history=False):
    """Generate all independent tasks. algos_per_task: list of algo names, or None for default pair."""
    rng = np.random.default_rng(seed)
    tasks = []
    task_id = 0

    for dim in dims:
        funcs = funcs_by_dim.get(dim, [])
        n_reps = reps_by_dim[dim]
        n_starts = _n_starts(dim)

        for func_name in funcs:
            config = assign_config(func_name, dim)

            for rep in range(n_reps):
                algo_seed = int(rng.integers(0, 2**31))
                starts = (config.bounds_lower
                          + rng.uniform(size=(n_starts, dim))
                          * (config.bounds_upper - config.bounds_lower))

                if algos_per_task:
                    algo_list = algos_per_task
                else:
                    algo_list = ["SMCO_R", "SMCO_R_EVO"]

                for algo_name in algo_list:
                    tasks.append({
                        "task_id": task_id,
                        "algo": algo_name,
                        "func": func_name,
                        "dim": dim,
                        "rep": rep,
                        "seed": algo_seed,
                        "starts": starts,
                        "lower": config.bounds_lower.copy(),
                        "upper": config.bounds_upper.copy(),
                        "record_history": record_history,
                        "pair": "",
                    })
                    task_id += 1

    return tasks


def _generate_variant_tasks(dims, funcs_by_dim, reps_by_dim, seed):
    """Generate tasks for all 3 variant pairs."""
    rng = np.random.default_rng(seed)
    tasks = []
    task_id = 0

    for dim in dims:
        funcs = funcs_by_dim.get(dim, [])
        n_reps = reps_by_dim[dim]
        n_starts = _n_starts(dim)

        for func_name in funcs:
            config = assign_config(func_name, dim)

            for rep in range(n_reps):
                algo_seed = int(rng.integers(0, 2**31))
                starts = (config.bounds_lower
                          + rng.uniform(size=(n_starts, dim))
                          * (config.bounds_upper - config.bounds_lower))

                for base_algo, evo_algo in PAIRS:
                    for algo_name in (base_algo, evo_algo):
                        tasks.append({
                            "task_id": task_id,
                            "algo": algo_name,
                            "func": func_name,
                            "dim": dim,
                            "rep": rep,
                            "seed": algo_seed,
                            "starts": starts,
                            "lower": config.bounds_lower.copy(),
                            "upper": config.bounds_upper.copy(),
                            "record_history": False,
                            "pair": f"{base_algo}_vs_{evo_algo}",
                        })
                        task_id += 1

    return tasks


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

def _run_parallel(tasks, workers, label):
    results = []
    total = len(tasks)
    done = 0
    print(f"[{label}] {total} tasks, {workers} workers")

    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_task, t): t["task_id"] for t in tasks}
        for future in as_completed(futures):
            out = future.result()
            results.append(out)
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                elapsed = time.perf_counter() - t_start
                print(f"[{label}] {done}/{total} done ({elapsed:.0f}s)")

    return results, time.perf_counter() - t_start


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_scaling(output, dims, quick, seed, workers):
    """Exp A: SMCO_R vs SMCO_R_EVO scaling."""
    reps = {d: 3 for d in dims} if quick else REPS_BY_DIM
    funcs = {d: ["Rastrigin", "Ackley"] for d in dims} if quick else FUNCS_BY_DIM

    tasks = _generate_tasks(dims, funcs, reps, seed)
    results, wall = _run_parallel(tasks, workers, "A")

    rows = []
    conv_rows = []
    for r in results:
        hist = r.pop("history", None)
        rows.append(r)
        if hist:
            for i, val in enumerate(hist):
                conv_rows.append({
                    "algo": r["algo"], "func": r["func"], "dim": r["dim"],
                    "rep": r["rep"], "iteration": i, "runmax": val,
                })

    _save_csv(rows, output / "expA_scaling.csv")
    if conv_rows:
        _save_csv(conv_rows, output / "expA_convergence.csv")
    print(f"[A] Done. {len(rows)} result rows, wall={wall:.0f}s")


def run_variants(output, dims, quick, seed, workers):
    """Exp B: all 3 variant pairs."""
    reps = {d: 3 for d in dims} if quick else {d: REPS_BY_DIM.get(d, 3) for d in dims}
    funcs = {d: ["Rastrigin", "Ackley"] for d in dims} if quick else FUNCS_BY_DIM

    tasks = _generate_variant_tasks(dims, funcs, reps, seed)
    results, wall = _run_parallel(tasks, workers, "B")

    _save_csv(results, output / "expB_variants.csv")
    print(f"[B] Done. {len(results)} rows, wall={wall:.0f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ultra-high-dim SMCO comparison (multiprocessing)")
    parser.add_argument("--dims", type=int, nargs="+", default=[500, 1000, 2000, 5000],
                        help="Dimensions to test")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--experiment", type=str, choices=["A", "B"], default=None)
    args = parser.parse_args()

    workers = args.workers or min(os.cpu_count() or 1, 64)
    output = _output_dir(args.output_dir)
    dims = args.dims
    print(f"Output: {output}")
    print(f"Dims: {dims}, Workers: {workers}")

    if args.experiment is None or args.experiment == "A":
        print(f"\n{'='*60}\nExperiment A: Scaling\n{'='*60}")
        run_scaling(output, dims, args.quick, args.seed, workers)

    if args.experiment is None or args.experiment == "B":
        print(f"\n{'='*60}\nExperiment B: Variants\n{'='*60}")
        run_variants(output, dims, args.quick, args.seed, workers)

    print(f"\nAll done. Results in {output}")


if __name__ == "__main__":
    main()
