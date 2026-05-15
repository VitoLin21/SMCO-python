#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import comparison.methods  # noqa: F401
from comparison.run_comparison import run_comparison


FUNC_NAMES = ["Rastrigin", "Ackley", "Griewank", "Michalewicz"]
DIRECTIONS = [True, False]


def save_results(comp_result, output_dir: Path, label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for algo_name in comp_result.algo_names:
        metrics = comp_result.sum_results[algo_name]
        rows.append(
            {
                "label": label,
                "name": comp_result.name,
                "dim": comp_result.dim,
                "maximize": comp_result.to_maximize,
                "n_replications": comp_result.n_replications,
                "best_opt": comp_result.best_opt,
                "algo": algo_name,
                **metrics,
            }
        )

    json_path = output_dir / f"{label}.json"
    csv_path = output_dir / f"{label}.csv"
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"  {row['algo']:8s} rMSE={row['rMSE']:.6g} "
            f"AE99={row['AE99']:.6g} time={row['time_avg']:.4f}s",
            flush=True,
        )


def run_task(
    output_dir: Path,
    label: str,
    func_name: str,
    dim: int,
    maximize: bool,
    algos: list[str],
    n_replications: int,
    smco_options: dict,
    seed: int,
    force: bool,
) -> None:
    csv_path = output_dir / f"{label}.csv"
    if csv_path.exists() and not force:
        print(f"[skip] {label} exists", flush=True)
        return

    print(
        f"[run] {label}: dim={dim}, maximize={maximize}, "
        f"algos={','.join(algos)}, reps={n_replications}",
        flush=True,
    )
    t0 = time.perf_counter()
    comp = run_comparison(
        name_config=func_name,
        dim_config=dim,
        to_maximize=maximize,
        algo_names=algos,
        n_replications=n_replications,
        smco_options=smco_options,
        seed=seed,
    )
    save_results(comp, output_dir, label)
    print(f"[done] {label}: elapsed={time.perf_counter() - t0:.1f}s", flush=True)


def run_table1(output_dir: Path, seed: int, force: bool) -> None:
    print("=== Targeted Table 1: SMCO_BR + GenSA, d=2 ===", flush=True)
    for func_name in FUNC_NAMES:
        for maximize in DIRECTIONS:
            label = f"table1_{func_name}_d2_{'max' if maximize else 'min'}"
            run_task(
                output_dir,
                label,
                func_name,
                dim=2,
                maximize=maximize,
                algos=["SMCO_BR", "GenSA"],
                n_replications=500,
                smco_options={"iter_max": 500, "n_starts": 1},
                seed=seed,
                force=force,
            )


def run_table2(output_dir: Path, seed: int, force: bool) -> None:
    print("=== Targeted Table 2: SMCO_BR, d=200 ===", flush=True)
    for func_name in FUNC_NAMES:
        for maximize in DIRECTIONS:
            label = f"table2_{func_name}_d200_{'max' if maximize else 'min'}"
            run_task(
                output_dir,
                label,
                func_name,
                dim=200,
                maximize=maximize,
                algos=["SMCO_BR"],
                n_replications=10,
                smco_options={"iter_max": 500, "n_starts": 14},
                seed=seed,
                force=force,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted SMCO_BR and GenSA benchmark subsets.")
    parser.add_argument("--output", default="result/comparison-targeted-br-gensa-2026-05-14")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--table1", action="store_true")
    parser.add_argument("--table2", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    if not args.table1 and not args.table2:
        args.table1 = True
        args.table2 = True
    if args.table1:
        run_table1(output_dir, args.seed, args.force)
    if args.table2:
        run_table2(output_dir, args.seed, args.force)


if __name__ == "__main__":
    main()
