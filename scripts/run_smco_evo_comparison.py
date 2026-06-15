#!/usr/bin/env python
"""SMCO-EVO vs SMCO comparison experiments.

Experiments:
  1: Core comparison (multi-dim x multi-function x multi-rep)
  2: Convergence curve analysis (using record_history)
  3: Evolution strategy sensitivity
  4: Parameter sensitivity (elimination_rate x evolution_points)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.stats import wilcoxon

from smco import smco, smco_br, smco_br_evo, smco_evo, smco_r, smco_r_evo
from smco.test_functions import assign_config

# ---------------------------------------------------------------------------
# Shared constants / helpers
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
    ("SMCO", "SMCO_EVO"),
    ("SMCO_R", "SMCO_R_EVO"),
    ("SMCO_BR", "SMCO_BR_EVO"),
]

DEFAULT_OPTIONS = {
    "iter_max": 300,
    "bounds_buffer": 0.05,
    "buffer_rand": True,
    "tol_conv": 1e-8,
}

# Functions by dimension tier
FUNC_DIMS = {
    2: ["Rastrigin", "Ackley", "Griewank", "Rosenbrock", "Michalewicz"],
    10: ["Rastrigin", "Ackley", "Griewank", "Rosenbrock", "DixonPrice", "Zakharov", "Qing", "Michalewicz"],
    50: ["Rastrigin", "Ackley", "Griewank", "Rosenbrock", "Michalewicz"],
    200: ["Rastrigin", "Ackley", "Griewank"],
}

REPLICATIONS = {2: 100, 10: 100, 50: 30, 200: 10}


def _output_dir(base: str | None) -> Path:
    if base:
        p = Path(base)
    else:
        p = Path("result") / f"evo-comparison-{date.today()}"
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


def _run_once(algo_name, f, config, start_points, opts):
    variant_fn = SMCO_VARIANT_MAP[algo_name]
    result = variant_fn(f, config.bounds_lower, config.bounds_upper,
                        start_points=start_points, **opts)
    return result


def _n_starts(dim: int) -> int:
    return max(3, int(math.ceil(math.sqrt(dim))))


# ---------------------------------------------------------------------------
# Experiment 1: Core comparison
# ---------------------------------------------------------------------------

def run_experiment1(output: Path, quick: bool = False, seed: int = 2026):
    rows = []
    dims = [2, 10] if quick else [2, 10, 50, 200]

    for dim in dims:
        funcs = FUNC_DIMS.get(dim, [])
        n_reps = 5 if quick else REPLICATIONS.get(dim, 30)
        for func_name in funcs:
            config, f = _make_objective(func_name, dim)
            n_starts = _n_starts(dim)
            rng = np.random.default_rng(seed)

            print(f"[Exp1] {func_name} {dim}D, {n_reps} reps, {n_starts} starts")

            for base_algo, evo_algo in PAIRS:
                for rep in range(n_reps):
                    algo_seed = int(rng.integers(0, 2**31))
                    starts = config.bounds_lower + rng.uniform(size=(n_starts, dim)) * (config.bounds_upper - config.bounds_lower)

                    for algo_name in (base_algo, evo_algo):
                        opts = _algo_options(algo_name, dict(DEFAULT_OPTIONS), algo_seed)
                        t0 = time.perf_counter()
                        result = _run_once(algo_name, f, config, starts, opts)
                        elapsed = time.perf_counter() - t0

                        rows.append({
                            "experiment": 1,
                            "pair": f"{base_algo}_vs_{evo_algo}",
                            "algo": algo_name,
                            "func": func_name,
                            "dim": dim,
                            "rep": rep,
                            "fopt": result.best_result.f_optimal,
                            "time": elapsed,
                            "iterations": result.best_result.iterations,
                        })

    _save_csv(rows, output / "experiment1_core.csv")
    _summarize_experiment1(rows, output)
    print(f"[Exp1] Done. Results in {output / 'experiment1_core.csv'}")


def _summarize_experiment1(rows: list[dict], output: Path):
    import csv
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["pair"], r["func"], r["dim"], r["rep"])].append(r)

    summary_rows = []
    for key, entries in grouped.items():
        pair, func, dim, rep = key
        if len(entries) != 2:
            continue
        base = next(e for e in entries if e["algo"] == pair.split("_vs_")[0])
        evo = next(e for e in entries if e["algo"] == pair.split("_vs_")[1])
        summary_rows.append({
            "pair": pair, "func": func, "dim": dim, "rep": rep,
            "base_fopt": base["fopt"], "evo_fopt": evo["fopt"],
            "evo_wins": int(evo["fopt"] > base["fopt"]),
            "base_wins": int(base["fopt"] > evo["fopt"]),
            "tie": int(evo["fopt"] == base["fopt"]),
        })

    _save_csv(summary_rows, output / "experiment1_pairwise.csv")

    # Wilcoxon tests per (pair, func, dim)
    pair_data = defaultdict(lambda: {"base": [], "evo": []})
    for s in summary_rows:
        pair_data[(s["pair"], s["func"], s["dim"])]["base"].append(s["base_fopt"])
        pair_data[(s["pair"], s["func"], s["dim"])]["evo"].append(s["evo_fopt"])

    stat_rows = []
    for (pair, func, dim), vals in sorted(pair_data.items()):
        base_arr = np.array(vals["base"])
        evo_arr = np.array(vals["evo"])
        diff = evo_arr - base_arr
        n_wins = int(np.sum(diff > 0))
        n_losses = int(np.sum(diff < 0))
        n_ties = int(np.sum(diff == 0))
        if np.any(diff != 0) and len(diff) >= 5:
            try:
                stat, pval = wilcoxon(base_arr, evo_arr, alternative="two-sided")
            except ValueError:
                stat, pval = float("nan"), float("nan")
            effect = abs(stat)
        else:
            stat, pval = float("nan"), float("nan")
        stat_rows.append({
            "pair": pair, "func": func, "dim": dim,
            "n_reps": len(base_arr),
            "evo_wins": n_wins, "base_wins": n_losses, "ties": n_ties,
            "base_mean": float(np.mean(base_arr)),
            "evo_mean": float(np.mean(evo_arr)),
            "base_std": float(np.std(base_arr, ddof=1)) if len(base_arr) > 1 else 0.0,
            "evo_std": float(np.std(evo_arr, ddof=1)) if len(evo_arr) > 1 else 0.0,
            "wilcoxon_stat": stat, "p_value": pval,
            "significant_005": int(pval < 0.05) if not math.isnan(pval) else 0,
        })

    _save_csv(stat_rows, output / "experiment1_wilcoxon.csv")


# ---------------------------------------------------------------------------
# Experiment 2: Convergence curves
# ---------------------------------------------------------------------------

def run_experiment2(output: Path, quick: bool = False, seed: int = 2026):
    dims_funcs = [
        (2, "Rastrigin"), (10, "Rastrigin"), (50, "Rastrigin"),
        (10, "Ackley"), (10, "Rosenbrock"),
    ]
    if quick:
        dims_funcs = [(2, "Rastrigin"), (10, "Rastrigin")]

    n_reps = 5 if quick else 20
    rows = []

    for dim, func_name in dims_funcs:
        config, f = _make_objective(func_name, dim)
        n_starts = _n_starts(dim)
        rng = np.random.default_rng(seed)
        print(f"[Exp2] {func_name} {dim}D, {n_reps} reps")

        for rep in range(n_reps):
            algo_seed = int(rng.integers(0, 2**31))
            starts = config.bounds_lower + rng.uniform(size=(n_starts, dim)) * (config.bounds_upper - config.bounds_lower)

            for algo_name in ("SMCO_R", "SMCO_R_EVO"):
                opts = _algo_options(algo_name, dict(DEFAULT_OPTIONS), algo_seed)
                opts["record_history"] = True
                result = _run_once(algo_name, f, config, starts, opts)
                history = result.best_result.runmax_history
                if history is not None:
                    for i, val in enumerate(history):
                        rows.append({
                            "experiment": 2,
                            "algo": algo_name,
                            "func": func_name,
                            "dim": dim,
                            "rep": rep,
                            "iteration": i,
                            "runmax": float(val),
                        })

    _save_csv(rows, output / "experiment2_convergence.csv")
    print(f"[Exp2] Done. Results in {output / 'experiment2_convergence.csv'}")


# ---------------------------------------------------------------------------
# Experiment 3: Strategy sensitivity
# ---------------------------------------------------------------------------

STRATEGIES = ["rand1bin", "current-to-best1bin", "best1bin", "sobol"]


def run_experiment3(output: Path, quick: bool = False, seed: int = 2026):
    funcs = ["Rastrigin", "Ackley", "Rosenbrock", "Griewank"]
    dim = 10
    n_reps = 10 if quick else 50

    rows = []
    for func_name in funcs:
        config, f = _make_objective(func_name, dim)
        n_starts = _n_starts(dim)
        rng = np.random.default_rng(seed)
        print(f"[Exp3] {func_name} {dim}D, strategies={STRATEGIES}, {n_reps} reps")

        for strategy in STRATEGIES:
            for rep in range(n_reps):
                algo_seed = int(rng.integers(0, 2**31))
                starts = config.bounds_lower + rng.uniform(size=(n_starts, dim)) * (config.bounds_upper - config.bounds_lower)
                opts = _algo_options("SMCO_R_EVO", dict(DEFAULT_OPTIONS), algo_seed)
                opts["evolution_strategy"] = strategy
                t0 = time.perf_counter()
                result = _run_once("SMCO_R_EVO", f, config, starts, opts)
                elapsed = time.perf_counter() - t0

                rows.append({
                    "experiment": 3,
                    "strategy": strategy,
                    "func": func_name,
                    "dim": dim,
                    "rep": rep,
                    "fopt": result.best_result.f_optimal,
                    "time": elapsed,
                })

    _save_csv(rows, output / "experiment3_strategies.csv")
    print(f"[Exp3] Done. Results in {output / 'experiment3_strategies.csv'}")


# ---------------------------------------------------------------------------
# Experiment 4: Parameter sensitivity
# ---------------------------------------------------------------------------

ELIM_RATES = [0.1, 0.25, 0.4, 0.5]
EVO_POINTS_LIST = [
    (0.5,),
    (0.5, 0.75),
    (0.33, 0.66),
    (0.25, 0.5, 0.75),
]


def run_experiment4(output: Path, quick: bool = False, seed: int = 2026):
    funcs = ["Rastrigin", "Ackley", "Rosenbrock"]
    dim = 10
    n_reps = 5 if quick else 30

    rows = []
    for func_name in funcs:
        config, f = _make_objective(func_name, dim)
        n_starts = _n_starts(dim)
        rng = np.random.default_rng(seed)
        total = len(ELIM_RATES) * len(EVO_POINTS_LIST) * n_reps
        count = 0
        print(f"[Exp4] {func_name} {dim}D, {total} total runs")

        for elim_rate in ELIM_RATES:
            for evo_pts in EVO_POINTS_LIST:
                for rep in range(n_reps):
                    algo_seed = int(rng.integers(0, 2**31))
                    starts = config.bounds_lower + rng.uniform(size=(n_starts, dim)) * (config.bounds_upper - config.bounds_lower)
                    opts = _algo_options("SMCO_R_EVO", dict(DEFAULT_OPTIONS), algo_seed)
                    opts["elimination_rate"] = elim_rate
                    opts["evolution_points"] = evo_pts
                    result = _run_once("SMCO_R_EVO", f, config, starts, opts)

                    rows.append({
                        "experiment": 4,
                        "func": func_name,
                        "dim": dim,
                        "elimination_rate": elim_rate,
                        "evolution_points": str(evo_pts),
                        "rep": rep,
                        "fopt": result.best_result.f_optimal,
                    })
                    count += 1
                    if count % 50 == 0:
                        print(f"  ... {count}/{total}")

    _save_csv(rows, output / "experiment4_params.csv")
    print(f"[Exp4] Done. Results in {output / 'experiment4_params.csv'}")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save_csv(rows: list[dict], path: Path):
    if not rows:
        return
    import csv
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SMCO-EVO vs SMCO comparison experiments")
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Run a single experiment (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Use reduced replications for fast verification")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output = _output_dir(args.output_dir)
    print(f"Output directory: {output}")

    experiments = {
        1: run_experiment1,
        2: run_experiment2,
        3: run_experiment3,
        4: run_experiment4,
    }

    to_run = [args.experiment] if args.experiment else [1, 2, 3, 4]
    for exp_id in to_run:
        print(f"\n{'='*60}")
        print(f"Experiment {exp_id}")
        print(f"{'='*60}")
        experiments[exp_id](output, quick=args.quick, seed=args.seed)

    print(f"\nAll experiments complete. Results in {output}")


if __name__ == "__main__":
    main()
