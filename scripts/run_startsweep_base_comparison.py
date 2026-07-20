#!/usr/bin/env python
"""Experiment 1b: SMCO BASE (non-evo) variants with reduced start-point counts.

实验1 (run_evo_startsweep_comparison.py) 只跑了 evo 变体在减少起点下的
结果，对照基线的满起点。但缺一组关键对照：减少起点的 SMCO 原版（非 evo）。
本脚本补齐它，形成完整 2x2 矩阵：

    起点{满, 减少} x 算法{SMCO原版, SMCO_EVO}
      满起点 SMCO原版   -> 基线 all_results.csv (rand1bin)
      满起点 SMCO_EVO   -> 基线 all_results.csv (rand1bin)
      减少起点 SMCO原版 -> 本脚本 (startsweep_base_results.csv)
      减少起点 SMCO_EVO -> 实验1 startsweep_results.csv

Design mirrors run_evo_startsweep_comparison.py exactly (same fractions, same
reduced-start RNG pairing via _make_batch_rng so reduced starts line up
point-for-point with the baseline's full start matrix, same dims/funcs/reps)
but swaps the evo entry points for the non-evo ones (smco / smco_r / smco_br)
and drops the evo-only options. strategy is fixed to rand1bin to align with
experiment 1 and the baseline (the base variants don't use evolution_strategy,
but the column is kept for schema-comparable merging).

Usage:
  uv run python scripts/run_startsweep_base_comparison.py
  uv run python scripts/run_startsweep_base_comparison.py --quick
  uv run python scripts/run_startsweep_base_comparison.py --dims 1000 --fractions 0.5
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

from smco import smco, smco_br, smco_r
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
# Experiment-1b configuration
# ---------------------------------------------------------------------------

DIMS = [1000, 3000, 5000]
FUNCS = ["Rastrigin", "Ackley", "Rosenbrock"]
REPS = {1000: 5, 3000: 2, 5000: 2}
FRACTIONS = [0.5, 2.0 / 3.0, 0.75, 0.8]  # 1/2, 2/3, 3/4, 4/5
STRATEGY = "rand1bin"  # 仅用于 CSV schema 对齐；原版 SMCO 不依赖 evolution_strategy

BASE_VARIANT_MAP = {
    "SMCO": smco,
    "SMCO_R": smco_r,
    "SMCO_BR": smco_br,
}
BASE_VARIANTS = list(BASE_VARIANT_MAP)

BASELINE_CSV = Path("result/highdim-full-comparison-2026-06-04/all_results.csv")

RESULT_FIELDS = [
    "strategy", "start_fraction", "n_starts", "algo", "func", "dim", "rep",
    "fopt", "time", "iterations", "seed",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _output_dir(base: str | None) -> Path:
    p = Path(base) if base else Path("result") / f"startsweep-base-{date.today()}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _algo_options(algo_name: str, base_opts: dict, seed: int) -> dict:
    """Non-evo option assembly (mirrors run_highdim_full_comparison._algo_options
    non-evo branch): refine_ratio for R/BR, halved iter_max + iter_boost for BR,
    and drop ALL evolution_* params (the base variants don't evolve)."""
    opts = dict(base_opts)

    if algo_name in ("SMCO_R", "SMCO_BR"):
        opts.setdefault("refine_ratio", 0.5)

    if algo_name == "SMCO_BR":
        opts["iter_max"] = math.ceil(opts["iter_max"] / 2)
        opts.setdefault("iter_boost", 1000)
    else:
        opts.pop("iter_boost", None)

    for key in ("evolution_strategy", "evolution_points", "elimination_rate",
                "de_factor", "de_crossover"):
        opts.pop(key, None)

    opts["seed"] = seed
    return opts


def _fraction_label(frac: float) -> str:
    # Stable human-readable tag: 0.5->1/2, 0.667->2/3, 0.75->3/4, 0.8->4/5.
    for num, den in ((1, 2), (2, 3), (3, 4), (4, 5)):
        if abs(frac - num / den) < 1e-9:
            return f"{num}/{den}"
    return f"{frac:.4f}"


def _reduced_start_count(base: int, frac: float) -> int:
    # Truncate so the reduced count is strictly less than the full budget
    # (for frac < 1); never drop below 1.
    return max(1, int(base * frac))


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
    """Generate reduced-start tasks (base variants) for one dim.

    The batch RNG is advanced with the BASELINE's full dim/func/rep ordering so
    that the generated start matrices match the baseline point-for-point; we
    then keep only the first ``k`` rows for each fraction. This is IDENTICAL to
    run_evo_startsweep_comparison._generate_tasks except the algo loop iterates
    over the base variants (smco/smco_r/smco_br) instead of the evo ones, so a
    reduced-start base run shares the exact same starts[:k] as the corresponding
    reduced-start evo run -> directly comparable.
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
                for algo_name in BASE_VARIANTS:
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
        f = config.f
        if task["sense"] == "min":
            f = lambda x, _f=config.f: -_f(x)

        opts = _algo_options(algo_name, dict(DEFAULT_OPTIONS), algo_seed)
        variant_fn = BASE_VARIANT_MAP[algo_name]
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


def _run_parallel(tasks, workers, label):
    results = []
    total = len(tasks)
    done = 0
    print(f"[{label}] {total} tasks, {workers} workers", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_task, t) for t in tasks]
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % max(1, total // 10) == 0 or done == total:
                print(f"[{label}] {done}/{total} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(output: Path, dims, fractions, seed, workers, quick):
    if quick:
        reps = {d: 1 for d in dims}
        funcs = {d: ["Rastrigin"] for d in dims}
    else:
        reps = REPS
        funcs = {d: FUNCS for d in dims}

    global FRACTIONS
    FRACTIONS = fractions

    out_csv = output / "startsweep_base_results.csv"
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
            _k = _reduced_start_count(base_n, frac)  # noqa: F841 -- computed for parity with exp1
            for func_name in dim_funcs:
                for rep in range(n_reps):
                    for algo_name in BASE_VARIANTS:
                        expected.add((STRATEGY, frac_label, algo_name, func_name, dim, rep))
        if expected.issubset(done_keys):
            print(f"[{label}] SKIP (already done)", flush=True)
            continue

        tasks = _generate_tasks(seed, dim, dim_funcs, n_reps)
        results = _run_parallel(tasks, workers, label)
        _append_csv(results, out_csv)
        for r in results:
            done_keys.add((r["strategy"], r["start_fraction"], r["algo"], r["func"], r["dim"], r["rep"]))
        print(f"[{label}] SAVED {len(results)} rows", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1b: SMCO base (non-evo) reduced start-point sweep")
    parser.add_argument("--dims", type=int, nargs="+", default=DIMS)
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
    print(f"Variants: {BASE_VARIANTS} (non-evo) | strategy: {STRATEGY}", flush=True)
    print(f"Workers: {args.workers}, Seed: {args.seed}", flush=True)
    print(f"Baseline (reused, not re-run): {BASELINE_CSV}", flush=True)

    run(output, args.dims, args.fractions, args.seed, args.workers, args.quick)


if __name__ == "__main__":
    main()
