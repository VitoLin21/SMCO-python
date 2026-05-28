#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import subprocess
from pathlib import Path


STRATEGIES = ["current-to-best1bin", "best1bin", "sobol"]
TABLE12_LABELS = [
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
TABLE3_LABELS = [
    "table3_MaximumScore_d5",
    "table3_MaximumScore_d10",
    "table3_MaximumScore_d20",
    "table3_MaximumScore_d50",
    "table3_EmpiricalWelfare_d5",
    "table3_EmpiricalWelfare_d10",
    "table3_EmpiricalWelfare_d20",
    "table3_EmpiricalWelfare_d50",
]
SYNCED_BENCHES = [
    "SquaredNorm",
    "NegativeSquaredNorm",
    "Rastrigin",
    "Ackley",
    "Griewank",
    "Rosenbrock",
    "DixonPrice",
    "Zakharov",
    "Qing",
    "Michalewicz",
    "DropWave",
    "Shubert",
    "McCormick",
    "SixHumpCamel",
    "Eggholder",
    "Bukin6",
    "Easom",
    "Mishra6",
]


def _has_label_csv(path: Path, label: str) -> bool:
    return (path / f"{label}.csv").exists()


def _synced_summary_labels(path: Path) -> set[str]:
    csv_path = path / "summary.csv"
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row["label"] for row in rows}


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run missing evo strategy sweep labels in parallel")
    parser.add_argument("--strategies", nargs="*", default=STRATEGIES)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--run-table12", action="store_true")
    parser.add_argument("--run-table3", action="store_true")
    parser.add_argument("--run-rsynced", action="store_true")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--iter-max", type=int, default=300)
    parser.add_argument("--n-starts", type=int, default=10)
    args = parser.parse_args()

    tasks_by_strategy: dict[str, list[list[str]]] = {strategy: [] for strategy in args.strategies}
    for strategy in args.strategies:
        root = Path("result/evo_strategy_sweep") / strategy
        if args.run_table12:
            evo_dir = root / "comparison_evo"
            integrated_dir = root / "paper_tables_integrated"
            for label in TABLE12_LABELS:
                if _has_label_csv(evo_dir, label) and _has_label_csv(integrated_dir, label):
                    continue
                tasks_by_strategy[strategy].append(
                    [
                        "./.venv/bin/python",
                        "scripts/run_paper_tables_with_evo.py",
                        "--table12-evo",
                        "--evolution-strategy",
                        strategy,
                        "--base-result-dir",
                        "result/comparison",
                        "--evo-result-dir",
                        str(evo_dir),
                        "--integrated-dir",
                        str(integrated_dir),
                        "--only-labels",
                        label,
                    ]
                )
        if args.run_table3:
            table3_dir = root / "table3"
            for label in TABLE3_LABELS:
                if _has_label_csv(table3_dir, label):
                    continue
                tasks_by_strategy[strategy].append(
                    [
                        "./.venv/bin/python",
                        "scripts/run_paper_tables_with_evo.py",
                        "--table3",
                        "--table3-evo-only",
                        "--evolution-strategy",
                        strategy,
                        "--table3-dir",
                        str(table3_dir),
                        "--cache-dir",
                        "result/table3_cache",
                        "--only-labels",
                        label,
                    ]
                )
        if args.run_rsynced:
            r_dir = root / "r_synced"
            done_labels = _synced_summary_labels(r_dir)
            done_benches = {label.rsplit("_", 1)[0] for label in done_labels}
            for bench in SYNCED_BENCHES:
                if bench in done_benches:
                    continue
                tasks_by_strategy[strategy].append(
                    [
                        "./.venv/bin/python",
                        "scripts/run_synced_r_benchmarks_with_evo.py",
                        "--algo-set",
                        "smco",
                        "--evolution-strategy",
                        strategy,
                        "--out-dir",
                        str(r_dir),
                        "--repetitions",
                        str(args.repetitions),
                        "--iter-max",
                        str(args.iter_max),
                        "--n-starts",
                        str(args.n_starts),
                        "--only-benchmarks",
                        bench,
                    ]
                )

    # Round-robin tasks across strategies so one heavy strategy does not starve others.
    tasks: list[list[str]] = []
    pending = {k: list(v) for k, v in tasks_by_strategy.items() if v}
    while pending:
        finished: list[str] = []
        for strategy in list(pending.keys()):
            queue = pending[strategy]
            if queue:
                tasks.append(queue.pop(0))
            if not queue:
                finished.append(strategy)
        for strategy in finished:
            pending.pop(strategy, None)

    if not tasks:
        print("No missing tasks.")
        return

    print(f"Running {len(tasks)} tasks with workers={args.workers}")
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_run, cmd) for cmd in tasks]
        for fut in cf.as_completed(futures):
            fut.result()
    print("All missing tasks completed.")


if __name__ == "__main__":
    main()
