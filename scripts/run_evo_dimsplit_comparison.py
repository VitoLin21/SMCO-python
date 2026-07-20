#!/usr/bin/env python
"""Experiment 2: per-dimension-group ("dimsplit") evolution.

Question: does splitting the D-dimensional vector into G contiguous groups and
letting each group independently choose its DE parents (shared ranking, group-
level mutation) beat whole-vector evolution on high-dimensional problems?

Design:
  * dim_groups G in {1, 2, 4, 8}. G=1 is the whole-vector path (identical code
    to the baseline) and serves as a correctness anchor -- its numbers must
    match result/highdim-full-comparison-2026-06-04/all_results.csv for the
    same strategy/variant/dim/func/rep.
  * strategies: rand1bin and best1bin (per the plan; each group internally uses
    the chosen strategy but re-draws its own a/b/c).
  * variants: SMCO_EVO / SMCO_R_EVO / SMCO_BR_EVO.
  * dims {1000,3000,5000} x funcs {Rastrigin,Ackley,Rosenbrock} x reps {5,2,2}.

The start matrix is generated with the baseline's _make_batch_rng stream, so
runs are point-for-point comparable to the full high-dim baseline (which also
supplies the whole-vector evo + non-evo reference rows via all_results.csv).

Usage:
  uv run python scripts/run_evo_dimsplit_comparison.py
  uv run python scripts/run_evo_dimsplit_comparison.py --quick
  uv run python scripts/run_evo_dimsplit_comparison.py --dims 1000 --groups 2 4 --strategies rand1bin
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

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

from smco import smco_br_evo, smco_evo, smco_r_evo
from smco.test_functions import assign_config

from run_highdim_full_comparison import (
    DEFAULT_OPTIONS,
    DIMS as BASE_DIMS,
    FUNCS_BY_DIM as BASE_FUNCS_BY_DIM,
    REPS_BY_DIM as BASE_REPS_BY_DIM,
    _make_batch_rng,
    _n_starts,
)

# ---------------------------------------------------------------------------
# Experiment-2 configuration
# ---------------------------------------------------------------------------

DIMS = [1000, 3000, 5000]
FUNCS = ["Rastrigin", "Ackley", "Rosenbrock"]
REPS = {1000: 5, 3000: 2, 5000: 2}
GROUPS = [1, 2, 4, 8]
STRATEGIES = ["rand1bin", "best1bin"]

EVO_VARIANT_MAP = {
    "SMCO_EVO": smco_evo,
    "SMCO_R_EVO": smco_r_evo,
    "SMCO_BR_EVO": smco_br_evo,
}
EVO_VARIANTS = list(EVO_VARIANT_MAP)

RESULT_FIELDS = [
    "strategy", "dim_groups", "algo", "func", "dim", "rep",
    "fopt", "time", "iterations", "seed",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _output_dir(base: str | None) -> Path:
    p = Path(base) if base else Path("result") / f"evo-dimsplit-{date.today()}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _algo_options(algo_name: str, base_opts: dict, seed: int, strategy: str, dim_groups: int) -> dict:
    opts = dict(base_opts)
    if algo_name in ("SMCO_R_EVO", "SMCO_BR_EVO"):
        opts.setdefault("refine_ratio", 0.5)
    if algo_name == "SMCO_BR_EVO":
        opts["iter_max"] = math.ceil(opts["iter_max"] / 2)
        opts.setdefault("iter_boost", 1000)
    else:
        opts.pop("iter_boost", None)
    opts["evolution_strategy"] = strategy
    opts["dim_groups"] = dim_groups
    opts["seed"] = seed
    return opts


# ---------------------------------------------------------------------------
# CSV I/O with append support
# ---------------------------------------------------------------------------


def _append_csv(rows: list[dict], path: Path):
    if not rows:
        return
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def _existing_keys(path: Path) -> set[tuple]:
    keys: set[tuple] = set()
    if not path.exists():
        return keys
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            keys.add((
                row["strategy"], int(row["dim_groups"]), row["algo"],
                row["func"], int(row["dim"]), int(row["rep"]),
            ))
    return keys


# ---------------------------------------------------------------------------
# Task generation + worker
# ---------------------------------------------------------------------------


def _generate_tasks(seed: int, dim: int, funcs: list[str], n_reps: int,
                    groups: list[int], strategies: list[str]):
    """Generate dimsplit tasks for one dim.

    Batch RNG is advanced with the BASELINE ordering so the start matrices
    match the baseline (full budget, n_starts = ceil(sqrt(dim))); dim_groups
    and strategy are inner loops that do not consume the task RNG.
    """
    n_starts = _n_starts(dim)
    batch_rng = _make_batch_rng(seed, BASE_DIMS, BASE_FUNCS_BY_DIM, BASE_REPS_BY_DIM, dim)

    tasks = []
    for func_name in funcs:
        config = assign_config(func_name, dim)
        for rep in range(n_reps):
            algo_seed = int(batch_rng.integers(0, 2**31))
            starts = (
                config.bounds_lower
                + batch_rng.uniform(size=(n_starts, dim))
                * (config.bounds_upper - config.bounds_lower)
            )
            for strategy in strategies:
                for dim_groups in groups:
                    for algo_name in EVO_VARIANTS:
                        tasks.append({
                            "strategy": strategy,
                            "dim_groups": dim_groups,
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


def _run_task(task: dict) -> dict:
    try:
        func_name = task["func"]
        dim = task["dim"]
        algo_name = task["algo"]
        strategy = task["strategy"]
        dim_groups = task["dim_groups"]
        algo_seed = task["seed"]
        starts = task["starts"]
        lower = task["lower"]
        upper = task["upper"]

        config = assign_config(func_name, dim)
        f = config.f  # config.f 已是 -raw；最大化它 = 最小化 raw（正确方向）

        opts = _algo_options(algo_name, dict(DEFAULT_OPTIONS), algo_seed, strategy, dim_groups)
        variant_fn = EVO_VARIANT_MAP[algo_name]
        t0 = time.perf_counter()
        result = variant_fn(f, lower, upper, start_points=starts, **opts)
        elapsed = time.perf_counter() - t0

        return {
            "strategy": strategy,
            "dim_groups": dim_groups,
            "algo": algo_name,
            "func": func_name,
            "dim": dim,
            "rep": task["rep"],
            "fopt": result.best_result.f_optimal,
            "time": elapsed,
            "iterations": result.best_result.iterations,
            "seed": algo_seed,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "strategy": task["strategy"],
            "dim_groups": task["dim_groups"],
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


def _run_parallel(tasks, workers, label, out_csv):
    total = len(tasks)
    done = 0
    print(f"[{label}] {total} tasks, {workers} workers", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_task, t) for t in tasks]
        for fut in as_completed(futures):
            r = fut.result()
            _append_csv([r], out_csv)
            done += 1
            print(f"[{label}] {r['func']}/{r['dim']}D rep{r['rep']} {r['algo']} "
                  f"g{r['dim_groups']}/{r['strategy']} fopt={r['fopt']:.6g} "
                  f"{r['time']:.0f}s ({done}/{total})", flush=True)
    print(f"[{label}] DONE {total} tasks in {time.perf_counter() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(output: Path, dims, groups, strategies, seed, workers, quick, func_filter=None):
    if quick:
        reps = {d: 1 for d in dims}
        base_funcs = ["Rastrigin"]
    else:
        reps = REPS
        base_funcs = FUNCS
    funcs = {d: (func_filter if func_filter else base_funcs) for d in dims}

    out_csv = output / "dimsplit_results.csv"
    done_keys = _existing_keys(out_csv)

    for dim in dims:
        dim_funcs = funcs.get(dim, [])
        n_reps = reps[dim]
        label = f"{dim}D"

        expected = set()
        for strategy in strategies:
            for dim_groups in groups:
                for func_name in dim_funcs:
                    for rep in range(n_reps):
                        for algo_name in EVO_VARIANTS:
                            expected.add((strategy, dim_groups, algo_name, func_name, dim, rep))
        if expected.issubset(done_keys):
            print(f"[{label}] SKIP (already done)", flush=True)
            continue

        tasks = _generate_tasks(seed, dim, dim_funcs, n_reps, groups, strategies)
        # per-task 去重：跳过已落盘的 task，避免重启/分流时重复跑（整dim SKIP 之外的精细化断点续跑）
        tasks = [t for t in tasks if (t["strategy"], t["dim_groups"], t["algo"],
                                      t["func"], t["dim"], t["rep"]) not in done_keys]
        if not tasks:
            print(f"[{label}] SKIP (all tasks done)", flush=True)
            continue
        _run_parallel(tasks, workers, label, out_csv)
        done_keys = _existing_keys(out_csv)
        print(f"[{label}] DONE (flushed per-task)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: SMCO-EVO per-dimension-group evolution")
    parser.add_argument("--dims", type=int, nargs="+", default=DIMS)
    parser.add_argument("--funcs", type=str, nargs="+", default=None)
    parser.add_argument("--groups", type=int, nargs="+", default=GROUPS)
    parser.add_argument("--strategies", type=str, nargs="+", default=STRATEGIES)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    output = _output_dir(args.output_dir)
    print(f"Output: {output}", flush=True)
    print(f"Dims: {args.dims}", flush=True)
    print(f"dim_groups: {args.groups}", flush=True)
    print(f"Strategies: {args.strategies}", flush=True)
    print(f"Variants: {EVO_VARIANTS}", flush=True)
    print(f"Workers: {args.workers}, Seed: {args.seed}", flush=True)

    run(output, args.dims, args.groups, args.strategies, args.seed, args.workers, args.quick, args.funcs)


if __name__ == "__main__":
    main()
