from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from smco import run_benchmark


def main() -> None:
    # 统一 variant 与重复次数，便于横向对照不同函数/维度。
    jobs = [
        ("Rastrigin", 2),
        ("Ackley", 2),
        ("Griewank", 2),
        ("Rosenbrock", 10),
        ("DixonPrice", 10),
        ("Zakharov", 10),
        ("Qing", 10),
        ("Michalewicz", 10),
    ]

    variant = "smco_r"
    repetitions = 20
    iter_max = 300
    n_starts = 10
    seed = 2026

    rows: list[dict[str, object]] = []
    for name, dim in jobs:
        run = run_benchmark(
            name=name,
            dim=dim,
            variant=variant,
            repetitions=repetitions,
            iter_max=iter_max,
            n_starts=n_starts,
            seed=seed,
        )
        values = np.array([r.best_result.f_optimal for r in run.results], dtype=float)
        rows.append(
            {
                "name": name,
                "dim": dim,
                "variant": variant,
                "repetitions": repetitions,
                "iter_max": iter_max,
                "n_starts": n_starts,
                "seed": seed,
                "best_value": float(np.max(values)),
                "mean_value": float(np.mean(values)),
                "std_value": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "best_x": run.best_x.tolist(),
                "all_values": values.tolist(),
            }
        )

    out_json = Path("result/benchmark-results-paper-comparison.json")
    out_csv = Path("result/benchmark-results-paper-comparison.csv")

    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "dim",
                "variant",
                "repetitions",
                "iter_max",
                "n_starts",
                "seed",
                "best_value",
                "mean_value",
                "std_value",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    for row in rows:
        print(
            f"{row['name']:12s} d={row['dim']:>3} "
            f"best={row['best_value']:.6f} mean={row['mean_value']:.6f} std={row['std_value']:.6f}"
        )
    print(f"\nWROTE: {out_json}")
    print(f"WROTE: {out_csv}")


if __name__ == "__main__":
    main()
