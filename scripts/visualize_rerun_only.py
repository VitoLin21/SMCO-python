from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

ROOT = Path(__file__).resolve().parent.parent
TABLE1_DIR = ROOT / "result" / "comparison-rerun-2026-05-13"
TABLE2_DIR = ROOT / "result" / "comparison-rerun-2026-05-13-d200"
OUT_DIR = ROOT / "figures" / "rerun-only"
DOC_PATH = ROOT / "docs" / "rerun-only-visualization-2026-05-14.md"

OUT_DIR.mkdir(parents=True, exist_ok=True)

FUNC_ORDER = ["Rastrigin", "Ackley", "Griewank", "Michalewicz"]
DIR_ORDER = ["max", "min"]
TABLE1_ALGOS = ["SMCO", "SMCO_R", "SMCO_BR", "GD", "SignGD", "ADAM", "SPSA", "optimLBFGS", "BOBYQA"]
TABLE2_ALGOS = ["SMCO", "SMCO_R", "SMCO_BR", "GenSA", "SA", "DEoptim", "GA", "PSO"]


def load_rows(base_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for csv_path in sorted(base_dir.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "label": row["label"],
                        "name": row["name"],
                        "direction": "max" if row["maximize"] == "True" else "min",
                        "algo": row["algo"],
                        "rMSE": float(row["rMSE"]),
                        "AE99": float(row["AE99"]),
                        "time_avg": float(row["time_avg"]),
                    }
                )
    return rows


def build_matrix(
    rows: list[dict[str, object]],
    algos: list[str],
    metric: str,
) -> tuple[np.ndarray, list[str]]:
    tasks = [f"{func}-{direction}" for func in FUNC_ORDER for direction in DIR_ORDER]
    matrix = np.full((len(algos), len(tasks)), np.nan)
    for i, algo in enumerate(algos):
        for j, task in enumerate(tasks):
            func, direction = task.split("-")
            match = [
                row
                for row in rows
                if row["algo"] == algo and row["name"] == func and row["direction"] == direction
            ]
            if match:
                matrix[i, j] = float(match[0][metric])
    return matrix, tasks


def geometric_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(values))))


def plot_heatmap(
    matrix: np.ndarray,
    algos: list[str],
    tasks: list[str],
    title: str,
    out_name: str,
) -> None:
    safe = np.where(np.isfinite(matrix) & (matrix > 0), matrix, np.nan)
    log_matrix = np.log10(safe)

    fig, ax = plt.subplots(figsize=(max(10, len(tasks) * 1.1), max(5, len(algos) * 0.45)))
    im = ax.imshow(log_matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels(tasks, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(algos)))
    ax.set_yticklabels(algos)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                text = f"{val:.3g}" if val >= 0.001 else f"{val:.1e}"
                ax.text(j, i, text, ha="center", va="center", color="white", fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(metric)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name)
    plt.close(fig)


def plot_algo_gm_bars(
    rows: list[dict[str, object]],
    algos: list[str],
    metric: str,
    title: str,
    out_name: str,
) -> list[dict[str, object]]:
    values = []
    for algo in algos:
        algo_vals = np.array([float(r[metric]) for r in rows if r["algo"] == algo], dtype=float)
        gm = geometric_mean(algo_vals)
        values.append({"algo": algo, "gm": gm})

    values.sort(key=lambda x: x["gm"])
    labels = [x["algo"] for x in values]
    gms = [x["gm"] for x in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, gms, color="#4c72b0")
    ax.set_yscale("log")
    ax.set_ylabel(f"Geometric mean of {metric}")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name)
    plt.close(fig)
    return values


def format_md_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4g}" if np.isfinite(val) else "NA")
            else:
                vals.append(str(val))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def write_doc(
    table1_rmse_rank: list[dict[str, object]],
    table1_time_rank: list[dict[str, object]],
    table2_rmse_rank: list[dict[str, object]],
    table2_time_rank: list[dict[str, object]],
) -> None:
    text = f"""# 本次重跑结果可视化说明

日期：2026-05-14

## 1. 范围

- `table1`：`result/comparison-rerun-2026-05-13/`
- `table2(d=200)`：`result/comparison-rerun-2026-05-13-d200/`

本说明只基于本次重跑结果，不引入论文原表指标。

## 2. 图表说明

- 热力图：
  - 横轴是任务（函数 + `max/min`）
  - 纵轴是算法
  - 单元格数字是原始指标值
  - 颜色按 `log10(metric)` 编码，方便同时看数量级差异
- 平均柱状图：
  - 使用几何平均汇总算法在全部任务上的指标表现
  - `rMSE` / `AE99` / `time` 都是越低越好

## 3. 算法汇总

### Table 1：rMSE 几何平均排序

{format_md_table(table1_rmse_rank, ["algo", "gm"])}

### Table 1：time 几何平均排序

{format_md_table(table1_time_rank, ["algo", "gm"])}

### Table 2(d=200)：rMSE 几何平均排序

{format_md_table(table2_rmse_rank, ["algo", "gm"])}

### Table 2(d=200)：time 几何平均排序

{format_md_table(table2_time_rank, ["algo", "gm"])}

## 4. 产出图表

- `figures/rerun-only/table1_rmse_heatmap.png`
- `figures/rerun-only/table1_ae99_heatmap.png`
- `figures/rerun-only/table1_time_heatmap.png`
- `figures/rerun-only/table1_rmse_gm.png`
- `figures/rerun-only/table1_time_gm.png`
- `figures/rerun-only/table2_d200_rmse_heatmap.png`
- `figures/rerun-only/table2_d200_ae99_heatmap.png`
- `figures/rerun-only/table2_d200_time_heatmap.png`
- `figures/rerun-only/table2_d200_rmse_gm.png`
- `figures/rerun-only/table2_d200_time_gm.png`
"""
    DOC_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    table1_rows = load_rows(TABLE1_DIR)
    table2_rows = load_rows(TABLE2_DIR)

    t1_rmse, t1_tasks = build_matrix(table1_rows, TABLE1_ALGOS, "rMSE")
    t1_ae99, _ = build_matrix(table1_rows, TABLE1_ALGOS, "AE99")
    t1_time, _ = build_matrix(table1_rows, TABLE1_ALGOS, "time_avg")
    t2_rmse, t2_tasks = build_matrix(table2_rows, TABLE2_ALGOS, "rMSE")
    t2_ae99, _ = build_matrix(table2_rows, TABLE2_ALGOS, "AE99")
    t2_time, _ = build_matrix(table2_rows, TABLE2_ALGOS, "time_avg")

    plot_heatmap(t1_rmse, TABLE1_ALGOS, t1_tasks, "Table 1 Rerun: rMSE Heatmap", "table1_rmse_heatmap.png")
    plot_heatmap(t1_ae99, TABLE1_ALGOS, t1_tasks, "Table 1 Rerun: AE99 Heatmap", "table1_ae99_heatmap.png")
    plot_heatmap(t1_time, TABLE1_ALGOS, t1_tasks, "Table 1 Rerun: Time Heatmap", "table1_time_heatmap.png")
    plot_heatmap(t2_rmse, TABLE2_ALGOS, t2_tasks, "Table 2 d=200 Rerun: rMSE Heatmap", "table2_d200_rmse_heatmap.png")
    plot_heatmap(t2_ae99, TABLE2_ALGOS, t2_tasks, "Table 2 d=200 Rerun: AE99 Heatmap", "table2_d200_ae99_heatmap.png")
    plot_heatmap(t2_time, TABLE2_ALGOS, t2_tasks, "Table 2 d=200 Rerun: Time Heatmap", "table2_d200_time_heatmap.png")

    table1_rmse_rank = plot_algo_gm_bars(table1_rows, TABLE1_ALGOS, "rMSE", "Table 1 Rerun: Geometric Mean rMSE", "table1_rmse_gm.png")
    table1_time_rank = plot_algo_gm_bars(table1_rows, TABLE1_ALGOS, "time_avg", "Table 1 Rerun: Geometric Mean Time", "table1_time_gm.png")
    table2_rmse_rank = plot_algo_gm_bars(table2_rows, TABLE2_ALGOS, "rMSE", "Table 2 d=200 Rerun: Geometric Mean rMSE", "table2_d200_rmse_gm.png")
    table2_time_rank = plot_algo_gm_bars(table2_rows, TABLE2_ALGOS, "time_avg", "Table 2 d=200 Rerun: Geometric Mean Time", "table2_d200_time_gm.png")

    write_doc(table1_rmse_rank, table1_time_rank, table2_rmse_rank, table2_time_rank)
    print(f"Wrote figures to {OUT_DIR}")
    print(f"Wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
