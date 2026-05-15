#!/usr/bin/env python
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT / "result" / "comparison-targeted-br-gensa-2026-05-14"
BASE_TABLE1_DIR = ROOT / "result" / "comparison-rerun-2026-05-13"
BASE_TABLE2_DIR = ROOT / "result" / "comparison-rerun-2026-05-13-d200"
OUT_RESULT_DIR = ROOT / "result" / "targeted-br-gensa-summary-2026-05-14"
OUT_FIG_DIR = ROOT / "figures" / "targeted-br-gensa-2026-05-14"
OUT_DOC = ROOT / "docs" / "targeted-br-gensa-2026-05-14.md"

FUNC_ORDER = ["Rastrigin", "Ackley", "Griewank", "Michalewicz"]
DIR_ORDER = ["max", "min"]
TABLE1_BASE_ALGOS = ["SMCO", "SMCO_R", "GD", "SignGD", "ADAM", "SPSA", "optimLBFGS", "BOBYQA"]
TABLE1_ALGOS = ["SMCO_BR", "GenSA", *TABLE1_BASE_ALGOS]
TABLE2_ALGOS = ["SMCO_BR", "SMCO_BR_old", "SMCO", "SMCO_R", "GenSA", "SA", "DEoptim", "GA", "PSO"]

matplotlib.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def load_rows(base_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for csv_path in sorted(base_dir.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                direction = "max" if row["maximize"] == "True" else "min"
                rows.append(
                    {
                        "label": row["label"],
                        "name": row["name"],
                        "dim": int(row["dim"]),
                        "direction": direction,
                        "algo": row["algo"],
                        "rMSE": float(row["rMSE"]),
                        "AE99": float(row["AE99"]),
                        "time_avg": float(row["time_avg"]),
                        "source": str(base_dir.relative_to(ROOT)),
                    }
                )
    return rows


def task_key(row: dict[str, object]) -> tuple[str, str, int]:
    return str(row["name"]), str(row["direction"]), int(row["dim"])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def geometric_mean(values: list[float]) -> float:
    arr = np.array([v for v in values if math.isfinite(v) and v > 0], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(arr))))


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.4e}"
    return f"{value:.4g}"


def format_md_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = []
        for col in columns:
            val = row[col]
            values.append(format_float(val) if isinstance(val, float) else str(val))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def combine_table1_rows(base_rows: list[dict[str, object]], target_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    combined = [
        row
        for row in base_rows
        if row["dim"] == 2 and str(row["algo"]) in TABLE1_BASE_ALGOS
    ]
    combined.extend(
        row
        for row in target_rows
        if row["dim"] == 2 and str(row["algo"]) in {"SMCO_BR", "GenSA"}
    )
    return combined


def combine_table2_rows(base_rows: list[dict[str, object]], target_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    combined = []
    for row in base_rows:
        if row["dim"] != 200:
            continue
        item = dict(row)
        if item["algo"] == "SMCO_BR":
            item["algo"] = "SMCO_BR_old"
        combined.append(item)
    combined.extend(
        row
        for row in target_rows
        if row["dim"] == 200 and str(row["algo"]) == "SMCO_BR"
    )
    return combined


def summarize_by_algo(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_algo: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_algo.setdefault(str(row["algo"]), []).append(row)
    summary = []
    for algo, items in by_algo.items():
        summary.append(
            {
                "algo": algo,
                "cases": len(items),
                "gm_rMSE": geometric_mean([float(row["rMSE"]) for row in items]),
                "gm_AE99": geometric_mean([float(row["AE99"]) for row in items]),
                "gm_time": geometric_mean([float(row["time_avg"]) for row in items]),
            }
        )
    return sorted(summary, key=lambda row: row["gm_rMSE"])


def table1_task_rank(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for func in FUNC_ORDER:
        for direction in DIR_ORDER:
            task_rows = [
                row
                for row in rows
                if row["name"] == func and row["direction"] == direction and row["algo"] in {"SMCO_BR", "GenSA"}
            ]
            for row in sorted(task_rows, key=lambda item: float(item["rMSE"])):
                out.append(
                    {
                        "task": f"{func}-{direction}",
                        "algo": row["algo"],
                        "rMSE": row["rMSE"],
                        "AE99": row["AE99"],
                        "time_avg": row["time_avg"],
                    }
                )
    return out


def plot_summary(summary: list[dict[str, object]], metric: str, title: str, filename: str) -> None:
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = [str(row["algo"]) for row in summary]
    values = np.array([float(row[metric]) for row in summary], dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.85), 4.8))
    ax.bar(labels, values)
    ax.set_yscale("log")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    for i, value in enumerate(values):
        ax.text(i, value, format_float(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / filename)
    plt.close(fig)


def build_doc(
    table1_summary: list[dict[str, object]],
    table2_summary: list[dict[str, object]],
    table1_rank: list[dict[str, object]],
) -> str:
    return f"""# SMCO_BR 与 GenSA 定向重跑结果

本次实验使用修复后的 comparison 代码：

- `SMCO_BR` 在 comparison 中按 regular + boosted 两轮 refine 的语义，将单次 `iter_max` 从 500 减半为 250。
- 表一补跑 `GenSA`，覆盖 Rastrigin、Ackley、Griewank、Michalewicz 的 max/min 共 8 个 d=2 场景。
- 表二只重跑 `SMCO_BR`，覆盖相同函数的 max/min 共 8 个 d=200 场景。

## 输出位置

- 原始结果：`result/comparison-targeted-br-gensa-2026-05-14/`
- 汇总 CSV：`result/targeted-br-gensa-summary-2026-05-14/`
- 图表：`figures/targeted-br-gensa-2026-05-14/`
- 运行日志：`result/targeted-br-gensa-2026-05-14.log`

## 表一 d=2 汇总

以下几何均值将新跑的 `SMCO_BR`、`GenSA` 与旧表一重跑结果中的其他算法放在一起比较。注意 `SMCO_BR` 是修复预算后的新结果。

{format_md_table(table1_summary, ["algo", "cases", "gm_rMSE", "gm_AE99", "gm_time"])}

## 表一中 SMCO_BR 与 GenSA 的逐场景表现

{format_md_table(table1_rank, ["task", "algo", "rMSE", "AE99", "time_avg"])}

## 表二 d=200 汇总

`SMCO_BR` 是修复预算后的新结果；`SMCO_BR_old` 是旧的表二重跑结果，旧结果使用了未减半的 BR 预算。

{format_md_table(table2_summary, ["algo", "cases", "gm_rMSE", "gm_AE99", "gm_time"])}

## 简要结论

- 表一 d=2 上，`GenSA` 在 Rastrigin、Ackley、Michalewicz 的 max/min 场景都明显优于修复预算后的 `SMCO_BR`，但耗时通常更高。
- Griewank d=2 上，`GenSA` 仍优于 `SMCO_BR`，不过优势没有 Rastrigin/Ackley 那么大。
- 表二 d=200 上，修复预算后的 `SMCO_BR` 相比旧 `SMCO_BR_old` 速度显著下降到更公平的预算水平，但误差整体变大；这是预期现象，因为旧结果实际给了 BR 接近两轮完整预算。
"""


def main() -> None:
    OUT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    target_rows = load_rows(TARGET_DIR)
    base_table1_rows = load_rows(BASE_TABLE1_DIR)
    base_table2_rows = load_rows(BASE_TABLE2_DIR)

    table1_rows = combine_table1_rows(base_table1_rows, target_rows)
    table2_rows = combine_table2_rows(base_table2_rows, target_rows)
    table1_summary = summarize_by_algo(table1_rows)
    table2_summary = summarize_by_algo(table2_rows)
    task_rank = table1_task_rank(table1_rows)

    write_csv(OUT_RESULT_DIR / "table1_combined.csv", table1_rows)
    write_csv(OUT_RESULT_DIR / "table2_combined.csv", table2_rows)
    write_csv(OUT_RESULT_DIR / "table1_summary.csv", table1_summary)
    write_csv(OUT_RESULT_DIR / "table2_summary.csv", table2_summary)
    write_csv(OUT_RESULT_DIR / "table1_smco_br_gensa_by_task.csv", task_rank)

    plot_summary(table1_summary, "gm_rMSE", "Table 1 d=2: Geometric Mean rMSE", "table1_gm_rmse.png")
    plot_summary(table1_summary, "gm_time", "Table 1 d=2: Geometric Mean Time", "table1_gm_time.png")
    plot_summary(table2_summary, "gm_rMSE", "Table 2 d=200: Geometric Mean rMSE", "table2_gm_rmse.png")
    plot_summary(table2_summary, "gm_time", "Table 2 d=200: Geometric Mean Time", "table2_gm_time.png")

    OUT_DOC.write_text(build_doc(table1_summary, table2_summary, task_rank), encoding="utf-8")
    print(f"Wrote {OUT_DOC}")
    print(f"Wrote {OUT_RESULT_DIR}")
    print(f"Wrote {OUT_FIG_DIR}")


if __name__ == "__main__":
    main()
