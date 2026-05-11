#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path

import numpy as np

import comparison.methods  # noqa: F401 — trigger registrations
from comparison.run_comparison import run_comparison

FUNC_NAMES = ["Rastrigin", "Ackley", "Griewank", "Michalewicz"]

GROUP_I_LOCAL = ["SMCO", "SMCO_R", "SMCO_BR", "GD", "SignGD", "ADAM", "SPSA", "optimLBFGS", "BOBYQA"]
GROUP_II_GLOBAL = ["SMCO", "SMCO_R", "SMCO_BR", "GenSA", "SA", "DEoptim", "GA", "PSO"]
ALL_ALGOS = GROUP_I_LOCAL + ["GenSA", "SA", "DEoptim", "GA", "PSO"]


def save_results(comp_result, output_dir: Path, label: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for algo_name in comp_result.algo_names:
        metrics = comp_result.sum_results[algo_name]
        rows.append({
            "label": label,
            "name": comp_result.name,
            "dim": comp_result.dim,
            "maximize": comp_result.to_maximize,
            "n_replications": comp_result.n_replications,
            "best_opt": comp_result.best_opt,
            "algo": algo_name,
            **metrics,
        })

    json_path = output_dir / f"{label}.json"
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    csv_path = output_dir / f"{label}.csv"
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writerows(rows)

    print(f"\n--- {label} ---")
    for row in rows:
        print(f"  {row['algo']:15s}  rMSE={row['rMSE']:.6f}  AE99={row['AE99']:.6f}  time={row['time_avg']:.4f}s")
    print(f"  Saved: {json_path}, {csv_path}")


def run_quick(output_dir: Path):
    print("=== Quick Validation ===")
    for func_name in FUNC_NAMES:
        for to_max in [True]:
            label = f"quick_{func_name}_d2_{'max' if to_max else 'min'}"
            comp = run_comparison(
                name_config=func_name,
                dim_config=2,
                to_maximize=to_max,
                algo_names=ALL_ALGOS,
                n_replications=3,
                smco_options={"iter_max": 100, "n_starts": 3},
                seed=2026,
            )
            save_results(comp, output_dir, label)


def run_table1(output_dir: Path):
    print("=== Table 1: d=2, Group I ===")
    for func_name in FUNC_NAMES:
        for to_max in [True, False]:
            label = f"table1_{func_name}_d2_{'max' if to_max else 'min'}"
            comp = run_comparison(
                name_config=func_name,
                dim_config=2,
                to_maximize=to_max,
                algo_names=GROUP_I_LOCAL,
                n_replications=500,
                smco_options={"iter_max": 500, "n_starts": 1},
                seed=2026,
            )
            save_results(comp, output_dir, label)


def run_table2(output_dir: Path):
    print("=== Table 2: d=200, Group II ===")
    for func_name in FUNC_NAMES:
        for to_max in [True, False]:
            label = f"table2_{func_name}_d200_{'max' if to_max else 'min'}"
            comp = run_comparison(
                name_config=func_name,
                dim_config=200,
                to_maximize=to_max,
                algo_names=GROUP_II_GLOBAL,
                n_replications=10,
                smco_options={"iter_max": 500, "n_starts": 14},
                seed=2026,
            )
            save_results(comp, output_dir, label)


def main():
    parser = argparse.ArgumentParser(description="Run comparison experiments")
    parser.add_argument("--quick", action="store_true", help="Quick validation (small scale)")
    parser.add_argument("--table1", action="store_true", help="Reproduce Table 1 (d=2, Group I)")
    parser.add_argument("--table2", action="store_true", help="Reproduce Table 2 (d=200, Group II)")
    parser.add_argument("--output", default="result/comparison", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.quick:
        run_quick(output_dir)
    if args.table1:
        run_table1(output_dir)
    if args.table2:
        run_table2(output_dir)
    if not (args.quick or args.table1 or args.table2):
        print("No experiment selected. Use --quick, --table1, or --table2.")


if __name__ == "__main__":
    main()
