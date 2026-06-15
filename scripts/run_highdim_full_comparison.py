#!/usr/bin/env python
"""Full high-dimensional comparison: SMCO variants + baselines across evolution strategies.

Dimensions: 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000
Evolution strategies: rand1bin, current-to-best1bin, best1bin, sobol
elimination_rate: 0.5

Runs tasks in batches of (strategy, dim). Results are saved after each batch,
so progress is preserved even if the process is interrupted. Already-completed
batches are skipped on restart (resume support).

Usage:
  uv run python scripts/run_highdim_full_comparison.py
  uv run python scripts/run_highdim_full_comparison.py --quick
  uv run python scripts/run_highdim_full_comparison.py --dims 50 100 --strategies rand1bin
"""
from __future__ import annotations

import argparse
import csv
import inspect
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

from smco import smco, smco_br, smco_br_evo, smco_evo, smco_r, smco_r_evo
from smco.test_functions import assign_config

import comparison.methods  # noqa: F401
from comparison.methods import get_method

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIMS = [50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]

STRATEGIES = ["rand1bin", "current-to-best1bin", "best1bin", "sobol"]

FUNCS_BY_DIM = {
    50:   ["Rastrigin", "Ackley", "Rosenbrock", "Griewank", "Zakharov"],
    100:  ["Rastrigin", "Ackley", "Rosenbrock", "Griewank", "Zakharov"],
    200:  ["Rastrigin", "Ackley", "Rosenbrock", "Griewank", "Zakharov"],
    500:  ["Rastrigin", "Ackley", "Rosenbrock", "Zakharov"],
    1000: ["Rastrigin", "Ackley", "Rosenbrock"],
    2000: ["Rastrigin", "Ackley", "Rosenbrock"],
    3000: ["Rastrigin", "Ackley", "Rosenbrock"],
    4000: ["Rastrigin", "Ackley", "Rosenbrock"],
    5000: ["Rastrigin", "Ackley", "Rosenbrock"],
}

REPS_BY_DIM = {
    50: 20, 100: 15, 200: 10,
    500: 5, 1000: 5, 2000: 3,
    3000: 2, 4000: 2, 5000: 2,
}

SMCO_VARIANT_MAP = {
    "SMCO": smco,
    "SMCO_R": smco_r,
    "SMCO_BR": smco_br,
    "SMCO_EVO": smco_evo,
    "SMCO_R_EVO": smco_r_evo,
    "SMCO_BR_EVO": smco_br_evo,
}
SMCO_VARIANTS = list(SMCO_VARIANT_MAP)

LOCAL_METHODS = ["GD", "SignGD", "ADAM", "SPSA", "optimLBFGS", "BOBYQA", "optimNM"]
GLOBAL_METHODS = ["GenSA", "SA", "DEoptim", "GA", "PSO"]

DEFAULT_OPTIONS = {
    "iter_max": 300,
    "bounds_buffer": 0.05,
    "buffer_rand": True,
    "tol_conv": 1e-8,
    "elimination_rate": 0.5,
    "evolution_points": (0.5, 0.75),
    "de_factor": 0.8,
    "de_crossover": 0.7,
}

RESULT_FIELDS = ["strategy", "algo", "func", "dim", "rep", "fopt", "time", "iterations", "seed"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _output_dir(base: str | None) -> Path:
    if base:
        p = Path(base)
    else:
        p = Path("result") / f"highdim-full-comparison-{date.today()}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _n_starts(dim: int) -> int:
    return max(3, int(math.ceil(math.sqrt(dim))))


def _methods_for_dim(dim: int) -> list[str]:
    methods = list(SMCO_VARIANTS)
    if dim <= 200:
        methods.extend(LOCAL_METHODS)
    methods.extend(GLOBAL_METHODS)
    return methods


def _algo_options(algo_name: str, base_opts: dict, seed: int, strategy: str) -> dict:
    opts = dict(base_opts)

    if algo_name in ("SMCO_R", "SMCO_BR", "SMCO_R_EVO", "SMCO_BR_EVO"):
        opts.setdefault("refine_ratio", 0.5)

    if algo_name in ("SMCO_BR", "SMCO_BR_EVO"):
        opts["iter_max"] = math.ceil(opts["iter_max"] / 2)
        opts.setdefault("iter_boost", 1000)
    else:
        opts.pop("iter_boost", None)

    if algo_name in ("SMCO_EVO", "SMCO_R_EVO", "SMCO_BR_EVO"):
        opts["evolution_strategy"] = strategy
    else:
        for key in ("evolution_strategy", "evolution_points", "elimination_rate",
                     "de_factor", "de_crossover"):
            opts.pop(key, None)

    opts["seed"] = seed
    return opts


def _call_with_supported_kwargs(fn, *args, **kwargs):
    signature = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        supported = kwargs
    else:
        supported = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return fn(*args, **supported)


# ---------------------------------------------------------------------------
# CSV I/O with append support
# ---------------------------------------------------------------------------


def _append_csv(rows: list[dict], path: Path):
    """Append rows to CSV, creating with header if needed."""
    if not rows:
        return
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def _read_existing_keys(path: Path) -> set[tuple]:
    """Read existing results and return set of (strategy, algo, func, dim, rep) keys."""
    keys = set()
    if not path.exists():
        return keys
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            keys.add((row["strategy"], row["algo"], row["func"], int(row["dim"]), int(row["rep"])))
    return keys


# ---------------------------------------------------------------------------
# Task generation (per batch: strategy × dim)
# ---------------------------------------------------------------------------


def _generate_batch(strategy: str, dim: int, funcs: list[str], n_reps: int,
                    methods: list[str], rng: np.random.Generator):
    """Generate tasks for a single (strategy, dim) batch."""
    n_starts = _n_starts(dim)
    tasks = []

    for func_name in funcs:
        config = assign_config(func_name, dim)

        for rep in range(n_reps):
            algo_seed = int(rng.integers(0, 2**31))
            starts = (config.bounds_lower
                      + rng.uniform(size=(n_starts, dim))
                      * (config.bounds_upper - config.bounds_lower))

            for algo_name in methods:
                tasks.append({
                    "strategy": strategy,
                    "algo": algo_name,
                    "func": func_name,
                    "dim": dim,
                    "rep": rep,
                    "seed": algo_seed,
                    "starts": starts.copy(),
                    "lower": config.bounds_lower.copy(),
                    "upper": config.bounds_upper.copy(),
                    "sense": config.sense,
                })

    return tasks


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _run_task(task: dict) -> dict:
    """Worker function — runs in a child process."""
    try:
        func_name = task["func"]
        dim = task["dim"]
        algo_name = task["algo"]
        strategy = task["strategy"]
        algo_seed = task["seed"]
        starts = task["starts"]
        lower = task["lower"]
        upper = task["upper"]
        sense = task["sense"]

        config = assign_config(func_name, dim)
        f = config.f
        if sense == "min":
            f = lambda x, _f=config.f: -_f(x)

        opts = _algo_options(algo_name, dict(DEFAULT_OPTIONS), algo_seed, strategy)

        t0 = time.perf_counter()

        if algo_name in SMCO_VARIANT_MAP:
            variant_fn = SMCO_VARIANT_MAP[algo_name]
            result = variant_fn(f, lower, upper, start_points=starts, **opts)
            fopt = result.best_result.f_optimal
            iterations = result.best_result.iterations
        else:
            opt_fn = get_method(algo_name)
            raw_result = _call_with_supported_kwargs(
                opt_fn,
                f, lower, upper,
                start_points=starts,
                maximize=True,
                max_iter=opts["iter_max"],
                seed=algo_seed,
            )
            fopt = raw_result.f_optimal
            iterations = raw_result.iterations

        elapsed = time.perf_counter() - t0

        return {
            "strategy": strategy,
            "algo": algo_name,
            "func": func_name,
            "dim": dim,
            "rep": task["rep"],
            "fopt": fopt,
            "time": elapsed,
            "iterations": iterations,
            "seed": algo_seed,
        }
    except Exception as e:
        return {
            "strategy": task["strategy"],
            "algo": task["algo"],
            "func": task["func"],
            "dim": task["dim"],
            "rep": task["rep"],
            "fopt": float("nan"),
            "time": 0.0,
            "iterations": 0,
            "seed": task["seed"],
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------


def _run_parallel(tasks, workers, label):
    results = []
    total = len(tasks)
    done = 0
    print(f"[{label}] {total} tasks, {workers} workers", flush=True)

    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_task, t) for t in tasks]
        for future in as_completed(futures):
            out = future.result()
            results.append(out)
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                elapsed = time.perf_counter() - t_start
                print(f"[{label}] {done}/{total} done ({elapsed:.0f}s)", flush=True)

    return results, time.perf_counter() - t_start


# ---------------------------------------------------------------------------
# Main experiment (incremental save, batch by strategy×dim)
# ---------------------------------------------------------------------------


def run_full_comparison(output, dims, strategies, seed, workers, quick):
    if quick:
        reps = {d: 3 for d in dims}
        funcs = {d: ["Rastrigin", "Ackley"] for d in dims}
    else:
        reps = REPS_BY_DIM
        funcs = FUNCS_BY_DIM

    all_results_path = output / "all_results.csv"
    existing_keys = _read_existing_keys(all_results_path)

    total_batches = len(strategies) * len(dims)
    batch_idx = 0

    for strategy in strategies:
        strat_path = output / f"strategy_{strategy}.csv"
        strat_existing = _read_existing_keys(strat_path)

        for dim in dims:
            batch_idx += 1
            dim_funcs = funcs.get(dim, [])
            n_reps = reps[dim]
            methods = _methods_for_dim(dim)
            label = f"{batch_idx}/{total_batches} {strategy} {dim}D"

            # Skip batch if all results already exist
            expected_keys = set()
            for func_name in dim_funcs:
                for rep in range(n_reps):
                    for algo_name in methods:
                        expected_keys.add((strategy, algo_name, func_name, dim, rep))

            if expected_keys.issubset(existing_keys):
                print(f"[{label}] SKIP (already done)", flush=True)
                continue

            # Generate tasks with deterministic seed per (dim, func, rep)
            rng = np.random.default_rng(seed)
            # Advance RNG to the correct position for this dim
            # We need to consume RNG in the same order as the original _generate_tasks
            # to maintain seed consistency across batches.
            # Pre-consume: iterate dims in order, funcs in order, reps in order.
            batch_rng = _make_batch_rng(seed, dims, funcs, reps, dim)

            tasks = _generate_batch(strategy, dim, dim_funcs, n_reps, methods, batch_rng)

            if not tasks:
                continue

            results, wall = _run_parallel(tasks, workers, label)

            # Save to strategy CSV
            _append_csv(results, strat_path)
            # Save to all_results CSV
            _append_csv(results, all_results_path)
            # Update existing keys
            for r in results:
                existing_keys.add((r["strategy"], r["algo"], r["func"], r["dim"], r["rep"]))

            print(f"[{label}] SAVED {len(results)} rows ({wall:.0f}s)", flush=True)


def _make_batch_rng(seed, dims, funcs, reps, target_dim):
    """Create an RNG at the correct state for generating tasks at target_dim.

    The original task generation consumes RNG in order: dim->func->rep,
    drawing one algo_seed + one starts array per rep. Strategy/algo loops
    don't consume RNG. So we advance past all dims before target_dim.
    """
    rng = np.random.default_rng(seed)

    for dim in dims:
        if dim == target_dim:
            return rng
        dim_funcs = funcs.get(dim, [])
        n_reps = reps[dim]
        n_starts = _n_starts(dim)
        for _func_name in dim_funcs:
            for _rep in range(n_reps):
                _ = int(rng.integers(0, 2**31))
                _ = rng.uniform(size=(n_starts, dim))

    return rng


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Full high-dim SMCO + baselines comparison across evolution strategies",
    )
    parser.add_argument("--dims", type=int, nargs="+", default=None,
                        help="Dimensions to test (default: all 9)")
    parser.add_argument("--strategies", type=str, nargs="+", default=STRATEGIES,
                        choices=STRATEGIES,
                        help="Evolution strategies to test")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--workers", type=int, default=48,
                        help="Number of parallel workers")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Reduced replications and functions for smoke test")
    args = parser.parse_args()

    output = _output_dir(args.output_dir)
    dims = args.dims or DIMS
    workers = args.workers

    print(f"Output: {output}", flush=True)
    print(f"Dims: {dims}", flush=True)
    print(f"Strategies: {args.strategies}", flush=True)
    print(f"Workers: {workers}, Seed: {args.seed}", flush=True)

    run_full_comparison(output, dims, args.strategies, args.seed, workers, args.quick)


if __name__ == "__main__":
    main()
