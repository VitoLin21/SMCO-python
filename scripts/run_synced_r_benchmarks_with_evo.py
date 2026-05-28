#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

# Prevent BLAS/OpenMP oversubscription when running multiple benchmarks in parallel.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import comparison.methods  # noqa: F401
from comparison.methods import METHOD_REGISTRY
from comparison.run_comparison import run_comparison
from smco.test_functions import assign_config


SMCO_ALGOS = ["SMCO", "SMCO_R", "SMCO_BR", "SMCO_EVO", "SMCO_R_EVO", "SMCO_BR_EVO"]
COMPARISON_ALGOS = [
    "GD",
    "SignGD",
    "ADAM",
    "SPSA",
    "optimLBFGS",
    "optimNM",
    "BOBYQA",
    "GenSA",
    "SA",
    "DEoptim",
    "GA",
    "PSO",
]
ALL_ALGOS = SMCO_ALGOS + COMPARISON_ALGOS

# Benchmarks present in the upstream R benchmark bundle and ported to Python here.
SYNCED_R_BENCHMARKS = [
    ("SquaredNorm", 10),
    ("NegativeSquaredNorm", 10),
    ("Rastrigin", 10),
    ("Ackley", 10),
    ("Griewank", 10),
    ("Rosenbrock", 10),
    ("DixonPrice", 10),
    ("Zakharov", 10),
    ("Qing", 10),
    ("Michalewicz", 10),
    ("DropWave", 2),
    ("Shubert", 2),
    ("McCormick", 2),
    ("SixHumpCamel", 2),
    ("Eggholder", 2),
    ("Bukin6", 2),
    ("Easom", 2),
    ("Mishra6", 2),
]


def canonical_job(name: str, dim: int) -> tuple[str, int, bool]:
    config = assign_config(name, dim)
    return name, dim, config.sense == "max"


def build_rows(
    *,
    name: str,
    dim: int,
    maximize: bool,
    repetitions: int,
    iter_max: int,
    n_starts: int,
    seed: int,
    algo_names: list[str],
    evolution_strategy: str,
):
    comp = run_comparison(
        name_config=name,
        dim_config=dim,
        to_maximize=maximize,
        algo_names=algo_names,
        n_replications=repetitions,
        smco_options={
            "iter_max": iter_max,
            "n_starts": n_starts,
            "evolution_strategy": evolution_strategy,
        },
        seed=seed,
    )
    rows = []
    label = f"{name}_{dim}d"
    for algo_name in algo_names:
        rows.append(
            {
                "label": label,
                "name": name,
                "dim": dim,
                "maximize": maximize,
                "repetitions": repetitions,
                "iter_max": iter_max,
                "n_starts": n_starts,
                "seed": seed,
                "evolution_strategy": evolution_strategy,
                "best_opt": float(comp.best_opt),
                "algo": algo_name,
                **comp.sum_results[algo_name],
            }
        )
    return rows


def write_rows(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"

    if not rows:
        return

    merged: dict[tuple[str, str, str], dict] = {}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["label"], row["algo"], row.get("evolution_strategy", "rand1bin"))
                merged[key] = row
    for row in rows:
        key = (str(row["label"]), str(row["algo"]), str(row.get("evolution_strategy", "rand1bin")))
        merged[key] = row

    merged_rows = sorted(
        merged.values(),
        key=lambda row: (
            str(row["label"]),
            str(row.get("evolution_strategy", "rand1bin")),
            str(row["algo"]),
        ),
    )
    fieldnames = list(merged_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    json_path.write_text(json.dumps(merged_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    labels = sorted({row["label"] for row in merged_rows})
    report_lines = [
        "# R-Synced Benchmarks With SMCO-EVO",
        "",
        "## Setup",
        "",
        f"- Algorithms: {', '.join(ALL_ALGOS)}",
        f"- Benchmarks: {len(labels)}",
        "",
        "## Labels",
        "",
    ]
    report_lines.extend(f"- `{label}`" for label in labels)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all R-synced Python benchmarks with SMCO variants and comparison methods")
    parser.add_argument("--out-dir", default="result/r-synced-benchmarks-2026-05-28", help="Output directory")
    parser.add_argument("--repetitions", type=int, default=20, help="Replications per benchmark")
    parser.add_argument("--iter-max", type=int, default=300, help="Iteration budget")
    parser.add_argument("--n-starts", type=int, default=10, help="Shared number of start points")
    parser.add_argument("--seed", type=int, default=20260528, help="Base seed")
    parser.add_argument("--only-benchmarks", nargs="*", default=None, help="Optional benchmark names to run")
    parser.add_argument(
        "--evolution-strategy",
        default="rand1bin",
        help="Evolution strategy used by SMCO_EVO variants: rand1bin/current-to-best1bin/best1bin/sobol",
    )
    parser.add_argument(
        "--algo-set",
        choices=["all", "smco"],
        default="all",
        help="`all` runs SMCO family + comparison methods; `smco` runs only SMCO family",
    )
    args = parser.parse_args()

    unsupported = [name for name in COMPARISON_ALGOS if name not in METHOD_REGISTRY]
    if unsupported:
        raise ValueError(f"Unsupported comparison methods: {unsupported}")
    algo_names = ALL_ALGOS if args.algo_set == "all" else SMCO_ALGOS

    selected = set(args.only_benchmarks) if args.only_benchmarks else None
    rows: list[dict] = []
    for idx, (name, dim) in enumerate(SYNCED_R_BENCHMARKS):
        if selected and name not in selected:
            continue
        job_name, job_dim, maximize = canonical_job(name, dim)
        rows.extend(
            build_rows(
                name=job_name,
                dim=job_dim,
                maximize=maximize,
                repetitions=args.repetitions,
                iter_max=args.iter_max,
                n_starts=args.n_starts,
                seed=args.seed + idx * 100,
                algo_names=algo_names,
                evolution_strategy=args.evolution_strategy,
            )
        )
        write_rows(rows, Path(args.out_dir))
        print(f"WROTE rows for {job_name} dim={job_dim}")


if __name__ == "__main__":
    main()
