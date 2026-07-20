#!/usr/bin/env python
"""Experiment 1: SMCO-EVO with reduced start-point counts.

Question: can the three evo variants (SMCO_EVO / SMCO_R_EVO / SMCO_BR_EVO)
reach comparable quality when started from a *fraction* of the original
start-point budget (1/2, 2/3, 3/4, 4/5 of the default n_starts)?

Design:
  * Default n_starts follows the high-dim baseline: ceil(sqrt(dim))
      1000D -> 32, 3000D -> 55, 5000D -> 71 starts.
  * For each fraction we keep only the first k = max(1, int(base * frac))
    rows of the SAME start matrix the baseline used, so every reduced run is
    paired point-for-point with the full-budget baseline (same seed=21, same
    _make_batch_rng stream).
  * Only the three evo variants run here, fixed to rand1bin (per the plan).
    The full-budget evo + non-evo baselines are reused from
    result/highdim-full-comparison-2026-06-04/all_results.csv -- not re-run.

Usage:
  uv run python scripts/run_evo_startsweep_comparison.py
  uv run python scripts/run_evo_startsweep_comparison.py --quick
  uv run python scripts/run_evo_startsweep_comparison.py --dims 1000 --fractions 0.5
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

# Reuse the baseline's RNG plumbing so reduced starts line up exactly with the
# baseline's full start matrix (same dim/func/rep ordering, same seed stream).
from run_highdim_full_comparison import (
    DEFAULT_OPTIONS,
    DIMS as BASE_DIMS,
    FUNCS_BY_DIM as BASE_FUNCS_BY_DIM,
    REPS_BY_DIM as BASE_REPS_BY_DIM,
    _make_batch_rng,
    _n_starts,
)

# ---------------------------------------------------------------------------
# Experiment-1 configuration
# ---------------------------------------------------------------------------

DIMS = [1000, 3000, 5000]
FUNCS = ["Rastrigin", "Ackley", "Rosenbrock"]
REPS = {1000: 5, 3000: 2, 5000: 2}
FRACTIONS = [1.0, 0.5, 2.0 / 3.0, 0.75, 0.8]  # full + 1/2, 2/3, 3/4, 4/5
STRATEGY = "rand1bin"

EVO_VARIANT_MAP = {
    "SMCO_EVO": smco_evo,
    "SMCO_R_EVO": smco_r_evo,
    "SMCO_BR_EVO": smco_br_evo,
}
EVO_VARIANTS = list(EVO_VARIANT_MAP)

RESULT_FIELDS = [
    "strategy", "start_fraction", "n_starts", "algo", "func", "dim", "rep",
    "fopt", "time", "iterations", "seed",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _output_dir(base: str | None) -> Path:
    p = Path(base) if base else Path("result") / f"evo-startsweep-{date.today()}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _algo_options(algo_name: str, base_opts: dict, seed: int) -> dict:
    """Same option assembly as the baseline (_algo_options) for the evo variants."""
    opts = dict(base_opts)

    if algo_name in ("SMCO_R_EVO", "SMCO_BR_EVO"):
        opts.setdefault("refine_ratio", 0.5)

    if algo_name == "SMCO_BR_EVO":
        opts["iter_max"] = math.ceil(opts["iter_max"] / 2)
        opts.setdefault("iter_boost", 1000)
    else:
        opts.pop("iter_boost", None)

    opts["evolution_strategy"] = STRATEGY
    opts["seed"] = seed
    return opts


def _fraction_label(frac: float) -> str:
    # Stable human-readable tag: 1.0->full, 0.5->1/2, 0.667->2/3, 0.75->3/4, 0.8->4/5.
    if abs(frac - 1.0) < 1e-9:
        return "full"
    for num, den in ((1, 2), (2, 3), (3, 4), (4, 5)):
        if abs(frac - num / den) < 1e-9:
            return f"{num}/{den}"
    return f"{frac:.4f}"


def _reduced_start_count(base: int, frac: float) -> int:
    # full budget for frac==1.0; otherwise truncate so the reduced count is
    # strictly less than the full budget (frac < 1); never drop below 1.
    if abs(frac - 1.0) < 1e-9:
        return base
    k = int(base * frac)
    return max(1, k)


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
                row["strategy"], row["start_fraction"], row["algo"],
                row["func"], int(row["dim"]), int(row["rep"]),
            ))
    return keys


# ---------------------------------------------------------------------------
# Task generation + worker
# ---------------------------------------------------------------------------


def _generate_tasks(seed: int, dim: int, funcs: list[str], n_reps: int):
    """Generate reduced-start tasks for one dim.

    The batch RNG is advanced with the BASELINE's full dim/func/rep ordering so
    that the generated start matrices match the baseline point-for-point; we
    then keep only the first ``k`` rows for each fraction.
    """
    base_n_starts = _n_starts(dim)
    batch_rng = _make_batch_rng(seed, BASE_DIMS, BASE_FUNCS_BY_DIM, BASE_REPS_BY_DIM, dim)

    tasks = []
    for func_name in funcs:
        config = assign_config(func_name, dim)
        for rep in range(n_reps):
            algo_seed = int(batch_rng.integers(0, 2**31))
            starts_full = (
                config.bounds_lower
                + batch_rng.uniform(size=(base_n_starts, dim))
                * (config.bounds_upper - config.bounds_lower)
            )
            for frac in FRACTIONS:
                k = _reduced_start_count(base_n_starts, frac)
                starts_k = starts_full[:k]
                for algo_name in EVO_VARIANTS:
                    tasks.append({
                        "strategy": STRATEGY,
                        "start_fraction": _fraction_label(frac),
                        "n_starts": k,
                        "algo": algo_name,
                        "func": func_name,
                        "dim": dim,
                        "rep": rep,
                        "seed": algo_seed,
                        "starts": starts_k.copy(),
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
        algo_seed = task["seed"]
        starts = task["starts"]
        lower = task["lower"]
        upper = task["upper"]

        config = assign_config(func_name, dim)
        f = config.f  # config.f 已是 -raw（最大化目标）；最大化它 = 最小化 raw（正确方向）

        opts = _algo_options(algo_name, dict(DEFAULT_OPTIONS), algo_seed)
        variant_fn = EVO_VARIANT_MAP[algo_name]
        t0 = time.perf_counter()
        result = variant_fn(f, lower, upper, start_points=starts, **opts)
        elapsed = time.perf_counter() - t0

        return {
            "strategy": task["strategy"],
            "start_fraction": task["start_fraction"],
            "n_starts": task["n_starts"],
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
            "start_fraction": task["start_fraction"],
            "n_starts": task["n_starts"],
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
            _append_csv([r], out_csv)  # 每 task 立即落盘
            done += 1
            print(f"[{label}] {r['func']}/{r['dim']}D rep{r['rep']} {r['algo']} "
                  f"fopt={r['fopt']:.6g} {r['time']:.0f}s ({done}/{total})", flush=True)
    print(f"[{label}] DONE {total} tasks in {time.perf_counter() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(output: Path, dims, fractions, seed, workers, quick, func_filter=None):
    if quick:
        reps = {d: 1 for d in dims}
        base_funcs = ["Rastrigin"]
    else:
        reps = REPS
        base_funcs = FUNCS
    funcs = {d: (func_filter if func_filter else base_funcs) for d in dims}

    global FRACTIONS
    FRACTIONS = fractions

    out_csv = output / "startsweep_results.csv"
    done_keys = _existing_keys(out_csv)

    for dim in dims:
        dim_funcs = funcs.get(dim, [])
        n_reps = reps[dim]
        label = f"{dim}D"

        # Skip a whole dim if every (frac, algo, func, rep) is already saved.
        expected = set()
        for frac in fractions:
            frac_label = _fraction_label(frac)
            base_n = _n_starts(dim)
            k = _reduced_start_count(base_n, frac)
            for func_name in dim_funcs:
                for rep in range(n_reps):
                    for algo_name in EVO_VARIANTS:
                        expected.add((STRATEGY, frac_label, algo_name, func_name, dim, rep))
        if expected.issubset(done_keys):
            print(f"[{label}] SKIP (already done)", flush=True)
            continue

        tasks = _generate_tasks(seed, dim, dim_funcs, n_reps)
        _run_parallel(tasks, workers, label, out_csv)
        done_keys = _existing_keys(out_csv)  # per-task 已写盘，从磁盘刷新
        print(f"[{label}] DONE (flushed per-task)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: SMCO-EVO reduced start-point sweep")
    parser.add_argument("--dims", type=int, nargs="+", default=DIMS)
    parser.add_argument("--funcs", type=str, nargs="+", default=None)
    parser.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    output = _output_dir(args.output_dir)
    print(f"Output: {output}", flush=True)
    print(f"Dims: {args.dims}", flush=True)
    print(f"Fractions: {args.fractions}", flush=True)
    print(f"Variants: {EVO_VARIANTS} | strategy: {STRATEGY}", flush=True)
    print(f"Workers: {args.workers}, Seed: {args.seed}", flush=True)

    run(output, args.dims, args.fractions, args.seed, args.workers, args.quick, args.funcs)


if __name__ == "__main__":
    main()
