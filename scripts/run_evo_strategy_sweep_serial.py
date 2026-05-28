#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "result" / "evo_strategy_sweep"

EXPECTED_TABLE12 = [
    "table1_Rastrigin_d2_max",
    "table1_Rastrigin_d2_min",
    "table1_Ackley_d2_max",
    "table1_Ackley_d2_min",
    "table1_Griewank_d2_max",
    "table1_Griewank_d2_min",
    "table1_Michalewicz_d2_max",
    "table1_Michalewicz_d2_min",
    "table2_Rastrigin_d200_max",
    "table2_Rastrigin_d200_min",
    "table2_Ackley_d200_max",
    "table2_Ackley_d200_min",
    "table2_Griewank_d200_max",
    "table2_Griewank_d200_min",
    "table2_Michalewicz_d200_max",
    "table2_Michalewicz_d200_min",
]

EXPECTED_TABLE3 = [
    "table3_MaximumScore_d5",
    "table3_MaximumScore_d10",
    "table3_MaximumScore_d20",
    "table3_MaximumScore_d50",
    "table3_EmpiricalWelfare_d5",
    "table3_EmpiricalWelfare_d10",
    "table3_EmpiricalWelfare_d20",
    "table3_EmpiricalWelfare_d50",
]

EXPECTED_BENCH = [
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


def run_cmd(cmd: list[str]) -> int:
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    print(f"[RET] {proc.returncode}", flush=True)
    return proc.returncode


def missing_table12(strategy: str) -> list[str]:
    have = {p.stem for p in (RESULT / strategy / "paper_tables_integrated").glob("table*.csv")}
    return [x for x in EXPECTED_TABLE12 if x not in have]


def missing_table3(strategy: str) -> list[str]:
    have = {p.stem for p in (RESULT / strategy / "table3").glob("table3_*.csv")}
    return [x for x in EXPECTED_TABLE3 if x not in have]


def missing_bench(strategy: str) -> list[str]:
    summary = RESULT / strategy / "r_synced" / "summary.csv"
    have: set[tuple[str, int]] = set()
    if summary.exists():
        with summary.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["algo"] in {"SMCO_EVO", "SMCO_R_EVO", "SMCO_BR_EVO"}:
                    have.add((row["name"], int(row["dim"])))
    return [name for name, dim in EXPECTED_BENCH if (name, dim) not in have]


def run_strategy(strategy: str, with_rsynced: bool) -> int:
    code = 0
    miss12 = missing_table12(strategy)
    miss3 = missing_table3(strategy)
    missb = missing_bench(strategy) if with_rsynced else []

    print(f"[STRATEGY] {strategy}", flush=True)
    print(f"  table12 missing: {len(miss12)}", flush=True)
    print(f"  table3 missing: {len(miss3)}", flush=True)
    print(f"  rsynced missing: {len(missb)}", flush=True)

    for label in miss3:
        cmd = [
            ".venv/bin/python",
            "scripts/run_paper_tables_with_evo.py",
            "--table3",
            "--table3-evo-only",
            "--evolution-strategy",
            strategy,
            "--table3-dir",
            f"result/evo_strategy_sweep/{strategy}/table3",
            "--cache-dir",
            "result/table3_cache",
            "--only-labels",
            label,
        ]
        code = run_cmd(cmd)
        if code != 0:
            return code

    if missb:
        cmd = [
            ".venv/bin/python",
            "scripts/run_synced_r_benchmarks_with_evo.py",
            "--algo-set",
            "smco",
            "--evolution-strategy",
            strategy,
            "--out-dir",
            f"result/evo_strategy_sweep/{strategy}/r_synced",
            "--repetitions",
            "20",
            "--iter-max",
            "300",
            "--n-starts",
            "10",
            "--only-benchmarks",
            *missb,
        ]
        code = run_cmd(cmd)
        if code != 0:
            return code

    for label in miss12:
        cmd = [
            ".venv/bin/python",
            "scripts/run_paper_tables_with_evo.py",
            "--table12-evo",
            "--evolution-strategy",
            strategy,
            "--base-result-dir",
            "result/comparison",
            "--evo-result-dir",
            f"result/evo_strategy_sweep/{strategy}/comparison_evo",
            "--integrated-dir",
            f"result/evo_strategy_sweep/{strategy}/paper_tables_integrated",
            "--only-labels",
            label,
        ]
        code = run_cmd(cmd)
        if code != 0:
            return code
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run missing evo sweep tasks serially with explicit progress.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["current-to-best1bin", "best1bin", "sobol"],
        choices=["rand1bin", "current-to-best1bin", "best1bin", "sobol"],
    )
    parser.add_argument("--no-rsynced", action="store_true")
    args = parser.parse_args()

    for strategy in args.strategies:
        code = run_strategy(strategy, with_rsynced=not args.no_rsynced)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
