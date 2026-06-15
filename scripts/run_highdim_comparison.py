#!/usr/bin/env python
"""High-dimensional (d >= 50) SMCO-EVO vs SMCO comparison experiments.

Experiments:
  A: Scaling benchmark — SMCO_R vs SMCO_R_EVO across dims [50, 100, 200, 500, 1000]
  B: Variant sweep — all 3 base/EVO pairs at [100, 200, 500]
  C: Convergence profiling — record_history at [100, 200, 500]

Usage:
  uv run python scripts/run_highdim_comparison.py --experiment A
  uv run python scripts/run_highdim_comparison.py --experiment B
  uv run python scripts/run_highdim_comparison.py --experiment C
  uv run python scripts/run_highdim_comparison.py              # all
  uv run python scripts/run_highdim_comparison.py --quick      # fast smoke test
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from smco import smco, smco_br, smco_br_evo, smco_evo, smco_r, smco_r_evo
from smco.test_functions import assign_config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SMCO_VARIANT_MAP = {
    "SMCO": smco,
    "SMCO_R": smco_r,
    "SMCO_BR": smco_br,
    "SMCO_EVO": smco_evo,
    "SMCO_R_EVO": smco_r_evo,
    "SMCO_BR_EVO": smco_br_evo,
}

PAIRS = [
    ("SMCO_R", "SMCO_R_EVO"),
    ("SMCO", "SMCO_EVO"),
    ("SMCO_BR", "SMCO_BR_EVO"),
]

SCALABLE_FUNCS = ["Rastrigin", "Ackley", "Rosenbrock", "Griewank",
                  "DixonPrice", "Zakharov", "Qing"]

# Functions per dimension tier (some too slow at 1000D)
FUNCS_BY_MAX_DIM = {
    50: SCALABLE_FUNCS,
    100: SCALABLE_FUNCS,
    200: ["Rastrigin", "Ackley", "Rosenbrock", "Griewank", "Zakharov"],
    500: ["Rastrigin", "Ackley", "Rosenbrock", "Zakharov"],
    1000: ["Rastrigin", "Ackley", "Rosenbrock"],
}

DEFAULT_OPTIONS = {
    "iter_max": 300,
    "bounds_buffer": 0.05,
    "buffer_rand": True,
    "tol_conv": 1e-8,
}


def _output_dir(base: str | None) -> Path:
    if base:
        p = Path(base)
    else:
        p = Path("result") / f"highdim-comparison-{date.today()}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _make_objective(name: str, dim: int):
    config = assign_config(name, dim)
    if config.sense == "min":
        f = lambda x, _f=config.f: -_f(x)
    else:
        f = config.f
    return config, f


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


def _n_starts(dim: int) -> int:
    return max(3, int(math.ceil(math.sqrt(dim))))


def _save_csv(rows: list[dict], path: Path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Experiment A: Scaling benchmark
# ---------------------------------------------------------------------------

def run_experiment_a(output: Path, quick: bool = False, seed: int = 2026):
    """SMCO_R vs SMCO_R_EVO across 50-1000D, all scalable functions."""
    dims = [50, 100] if quick else [50, 100, 200, 500, 1000]
    reps_per_dim = {50: 30, 100: 20, 200: 10, 500: 5, 1000: 3}
    if quick:
        reps_per_dim = {50: 5, 100: 3}

    rows = []
    for dim in dims:
        funcs = FUNCS_BY_MAX_DIM.get(dim, [])
        n_reps = reps_per_dim[dim]
        rng = np.random.default_rng(seed)

        for func_name in funcs:
            config, f = _make_objective(func_name, dim)
            n_starts = _n_starts(dim)
            print(f"[A] {func_name} {dim}D, {n_reps} reps, {n_starts} starts")

            for rep in range(n_reps):
                algo_seed = int(rng.integers(0, 2**31))
                starts = (config.bounds_lower
                          + rng.uniform(size=(n_starts, dim))
                          * (config.bounds_upper - config.bounds_lower))

                for algo_name in ("SMCO_R", "SMCO_R_EVO"):
                    opts = _algo_options(algo_name, dict(DEFAULT_OPTIONS), algo_seed)
                    t0 = time.perf_counter()
                    result = SMCO_VARIANT_MAP[algo_name](
                        f, config.bounds_lower, config.bounds_upper,
                        start_points=starts, **opts)
                    elapsed = time.perf_counter() - t0

                    rows.append({
                        "experiment": "A",
                        "algo": algo_name,
                        "func": func_name,
                        "dim": dim,
                        "rep": rep,
                        "fopt": result.best_result.f_optimal,
                        "time": elapsed,
                        "iterations": result.best_result.iterations,
                        "n_starts": n_starts,
                    })

    _save_csv(rows, output / "expA_scaling.csv")
    print(f"[A] Done. {len(rows)} rows saved.")


# ---------------------------------------------------------------------------
# Experiment B: Variant sweep
# ---------------------------------------------------------------------------

def run_experiment_b(output: Path, quick: bool = False, seed: int = 2026):
    """All 3 base/EVO pairs at 100, 200, 500D."""
    dims = [100] if quick else [100, 200, 500]
    funcs_per_dim = {
        100: ["Rastrigin", "Ackley", "Rosenbrock", "Griewank"],
        200: ["Rastrigin", "Ackley", "Rosenbrock", "Griewank"],
        500: ["Rastrigin", "Ackley", "Rosenbrock"],
    }
    reps_per_dim = {100: 15, 200: 10, 500: 5}
    if quick:
        reps_per_dim = {100: 3}

    rows = []
    for dim in dims:
        funcs = funcs_per_dim.get(dim, [])
        n_reps = reps_per_dim[dim]
        rng = np.random.default_rng(seed)

        for func_name in funcs:
            config, f = _make_objective(func_name, dim)
            n_starts = _n_starts(dim)
            print(f"[B] {func_name} {dim}D, {n_reps} reps, pairs=3")

            for rep in range(n_reps):
                algo_seed = int(rng.integers(0, 2**31))
                starts = (config.bounds_lower
                          + rng.uniform(size=(n_starts, dim))
                          * (config.bounds_upper - config.bounds_lower))

                for base_algo, evo_algo in PAIRS:
                    for algo_name in (base_algo, evo_algo):
                        opts = _algo_options(algo_name, dict(DEFAULT_OPTIONS), algo_seed)
                        t0 = time.perf_counter()
                        result = SMCO_VARIANT_MAP[algo_name](
                            f, config.bounds_lower, config.bounds_upper,
                            start_points=starts, **opts)
                        elapsed = time.perf_counter() - t0

                        rows.append({
                            "experiment": "B",
                            "pair": f"{base_algo}_vs_{evo_algo}",
                            "algo": algo_name,
                            "func": func_name,
                            "dim": dim,
                            "rep": rep,
                            "fopt": result.best_result.f_optimal,
                            "time": elapsed,
                        })

    _save_csv(rows, output / "expB_variants.csv")
    print(f"[B] Done. {len(rows)} rows saved.")


# ---------------------------------------------------------------------------
# Experiment C: Convergence profiling
# ---------------------------------------------------------------------------

def run_experiment_c(output: Path, quick: bool = False, seed: int = 2026):
    """record_history for SMCO_R vs SMCO_R_EVO at 100, 200, 500D."""
    dims_funcs = [
        (100, "Rastrigin"), (100, "Ackley"), (100, "Rosenbrock"),
        (200, "Rastrigin"), (200, "Ackley"), (200, "Rosenbrock"),
        (500, "Rastrigin"), (500, "Ackley"),
    ]
    if quick:
        dims_funcs = [(100, "Rastrigin"), (100, "Ackley")]

    n_reps = 3 if quick else 10
    rows = []

    for dim, func_name in dims_funcs:
        config, f = _make_objective(func_name, dim)
        n_starts = _n_starts(dim)
        rng = np.random.default_rng(seed)
        print(f"[C] {func_name} {dim}D, {n_reps} reps (convergence)")

        for rep in range(n_reps):
            algo_seed = int(rng.integers(0, 2**31))
            starts = (config.bounds_lower
                      + rng.uniform(size=(n_starts, dim))
                      * (config.bounds_upper - config.bounds_lower))

            for algo_name in ("SMCO_R", "SMCO_R_EVO"):
                opts = _algo_options(algo_name, dict(DEFAULT_OPTIONS), algo_seed)
                opts["record_history"] = True
                result = SMCO_VARIANT_MAP[algo_name](
                    f, config.bounds_lower, config.bounds_upper,
                    start_points=starts, **opts)

                history = result.best_result.runmax_history
                if history is not None:
                    for i, val in enumerate(history):
                        rows.append({
                            "experiment": "C",
                            "algo": algo_name,
                            "func": func_name,
                            "dim": dim,
                            "rep": rep,
                            "iteration": i,
                            "runmax": float(val),
                        })

    _save_csv(rows, output / "expC_convergence.csv")
    print(f"[C] Done. {len(rows)} rows saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="High-dimensional SMCO-EVO vs SMCO comparison")
    parser.add_argument("--experiment", type=str, choices=["A", "B", "C"],
                        default=None, help="Run single experiment (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced replications for smoke test")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output = _output_dir(args.output_dir)
    print(f"Output: {output}")

    experiments = {"A": run_experiment_a, "B": run_experiment_b, "C": run_experiment_c}
    to_run = [args.experiment] if args.experiment else ["A", "B", "C"]

    for exp_id in to_run:
        print(f"\n{'='*60}")
        print(f"Experiment {exp_id}")
        print(f"{'='*60}")
        experiments[exp_id](output, quick=args.quick, seed=args.seed)

    print(f"\nAll done. Results in {output}")


if __name__ == "__main__":
    main()
