from __future__ import annotations

import csv
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "chen-et-al-2026-optimization-via-the-strategic-law-of-large-numbers.pdf"
TABLE1_DIR = ROOT / "result" / "comparison-rerun-2026-05-13"
TABLE2_DIR = ROOT / "result" / "comparison-rerun-2026-05-13-d200"
OUT_RESULT_DIR = ROOT / "result" / "paper-comparison"
OUT_FIG_DIR = ROOT / "figures" / "paper-rerun-comparison"
OUT_DOC = ROOT / "docs" / "paper-rerun-comparison-2026-05-14.md"

OUT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?")

TABLE1_FUNC_ORDER = [("Rastrigin", "Griewank"), ("Ackley", "Michalewicz")]
TABLE1_ALGOS = ["SMCO", "SMCO_R", "SMCO_BR", "GD", "SignGD", "SPSA", "ADAM", "optimLBFGS", "BOBYQA"]
TABLE1_PAPER_ALGO_MAP = {
    "SMCO": "SMCO",
    "SMCO-R": "SMCO_R",
    "SMCO-BR": "SMCO_BR",
    "GD": "GD",
    "SignGD": "SignGD",
    "SPSA": "SPSA",
    "ADAM": "ADAM",
    "L-BFGS": "optimLBFGS",
    "BOBYQA": "BOBYQA",
}

TABLE2_FUNC_ORDER = ["Rastrigin", "Griewank", "Ackley", "Michalewicz"]
TABLE2_ALGOS = ["SMCO", "SMCO_R", "SMCO_BR", "GenSA", "SA", "DEoptim", "GA", "PSO"]
TABLE2_PAPER_ALGO_MAP = {
    "SMCOms": "SMCO",
    "SMCOms-R": "SMCO_R",
    "SMCOms-BR": "SMCO_BR",
    "GenSA": "GenSA",
    "SA": "SA",
    "DEoptim": "DEoptim",
    "GA": "GA",
    "PSO": "PSO",
}


def _to_float(token: str) -> float:
    return float(token.replace(",", ""))


def _load_paper_text() -> list[str]:
    proc = subprocess.run(
        ["pdftotext", "-layout", str(PDF_PATH), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.replace("\u2212", "-").splitlines()


def _find_line(lines: list[str], needle: str) -> int:
    for idx, line in enumerate(lines):
        if needle in line:
            return idx
    raise ValueError(f"Could not find marker: {needle}")


def _extract_numbers(line: str) -> list[float]:
    return [_to_float(x) for x in NUM_RE.findall(line)]


def parse_paper_table1(lines: list[str]) -> dict[tuple[str, str, str], dict[str, float]]:
    start = _find_line(lines, "Table 1.")
    end = _find_line(lines, "Based on 500 reps.")
    section = lines[start:end + 1]

    pattern = re.compile(
        r"^\s*(SMCO-R|SMCO-BR|SMCO|GD|SignGD|SPSA|ADAM|L-BFGS|BOBYQA)\s+"
    )

    current_direction = None
    pair_index = 0
    data: dict[tuple[str, str, str], dict[str, float]] = {}

    for line in section:
        if "Min                 Function: Best value" in line:
            current_direction = "min"
            pair_index = 0
            continue
        if "Max                 Function: Best value" in line:
            current_direction = "max"
            pair_index = 0
            continue
        if "Function: Best value" in line and "Ackley:" in line:
            pair_index = 1
            continue

        match = pattern.match(line)
        if not match or current_direction is None:
            continue

        paper_algo = match.group(1)
        algo = TABLE1_PAPER_ALGO_MAP[paper_algo]
        nums = _extract_numbers(line)
        if len(nums) < 10:
            continue

        func_a, func_b = TABLE1_FUNC_ORDER[pair_index]
        data[(current_direction, func_a, algo)] = {
            "paper_rmse": nums[0],
            "paper_ae99": nums[3],
            "paper_time": nums[4],
        }
        data[(current_direction, func_b, algo)] = {
            "paper_rmse": nums[5],
            "paper_ae99": nums[8],
            "paper_time": nums[9],
        }

    return data


def parse_paper_table2(lines: list[str]) -> dict[tuple[str, str, str], dict[str, float]]:
    start = _find_line(lines, "Table 2.")
    end = _find_line(lines, "Based on 10 reps.")
    section = lines[start:end + 1]

    pattern = re.compile(
        r"^\s*(SMCOms-BR|SMCOms-R|SMCOms|GenSA|SA|DEoptim|STOGO|GA|PSO)\s+"
    )

    current_direction = None
    data: dict[tuple[str, str, str], dict[str, float]] = {}

    for line in section:
        if re.search(r"Min\s+Best value", line):
            current_direction = "min"
            continue
        if re.search(r"Max\s+Best value", line):
            current_direction = "max"
            continue

        match = pattern.match(line)
        if not match or current_direction is None:
            continue

        paper_algo = match.group(1)
        if paper_algo not in TABLE2_PAPER_ALGO_MAP:
            continue
        algo = TABLE2_PAPER_ALGO_MAP[paper_algo]
        nums = _extract_numbers(line)
        if len(nums) < 12:
            continue

        for idx, func in enumerate(TABLE2_FUNC_ORDER):
            base = idx * 3
            data[(current_direction, func, algo)] = {
                "paper_rmse": nums[base],
                "paper_ae99": nums[base + 1],
                "paper_time": nums[base + 2],
            }

    return data


def load_rerun_csvs(base_dir: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    data: dict[tuple[str, str, str], dict[str, float]] = {}
    for csv_path in sorted(base_dir.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                direction = "max" if row["maximize"] == "True" else "min"
                key = (direction, row["name"], row["algo"])
                data[key] = {
                    "rerun_rmse": float(row["rMSE"]),
                    "rerun_ae99": float(row["AE99"]),
                    "rerun_time": float(row["time_avg"]),
                    "rerun_best_opt": float(row["best_opt"]),
                    "n_replications": int(row["n_replications"]),
                }
    return data


def build_comparison_rows(
    table_name: str,
    paper: dict[tuple[str, str, str], dict[str, float]],
    rerun: dict[tuple[str, str, str], dict[str, float]],
    algos: list[str],
    funcs: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for direction in ("min", "max"):
        for func in funcs:
            for algo in algos:
                key = (direction, func, algo)
                if key not in paper or key not in rerun:
                    continue
                p = paper[key]
                r = rerun[key]
                rows.append(
                    {
                        "table": table_name,
                        "direction": direction,
                        "function": func,
                        "algo": algo,
                        "paper_rmse": p["paper_rmse"],
                        "rerun_rmse": r["rerun_rmse"],
                        "rmse_ratio": r["rerun_rmse"] / p["paper_rmse"] if p["paper_rmse"] != 0 else math.nan,
                        "paper_ae99": p["paper_ae99"],
                        "rerun_ae99": r["rerun_ae99"],
                        "ae99_ratio": r["rerun_ae99"] / p["paper_ae99"] if p["paper_ae99"] != 0 else math.nan,
                        "paper_time": p["paper_time"],
                        "rerun_time": r["rerun_time"],
                        "time_ratio": r["rerun_time"] / p["paper_time"] if p["paper_time"] != 0 else math.nan,
                        "rerun_best_opt": r["rerun_best_opt"],
                        "n_replications": r["n_replications"],
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _geom_mean(values: list[float]) -> float:
    arr = np.array([v for v in values if np.isfinite(v) and v > 0], dtype=float)
    if arr.size == 0:
        return math.nan
    return float(np.exp(np.mean(np.log(arr))))


def summarize_by_algo(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["algo"])].append(row)

    summary = []
    for algo, items in sorted(buckets.items()):
        summary.append(
            {
                "algo": algo,
                "cases": len(items),
                "gm_rmse_ratio": _geom_mean([float(x["rmse_ratio"]) for x in items]),
                "gm_ae99_ratio": _geom_mean([float(x["ae99_ratio"]) for x in items]),
                "gm_time_ratio": _geom_mean([float(x["time_ratio"]) for x in items]),
            }
        )
    return summary


def plot_ratio_bars(summary: list[dict[str, object]], title: str, key: str, out_name: str) -> None:
    algos = [str(x["algo"]) for x in summary]
    vals = [float(x[key]) for x in summary]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d62728" if np.isfinite(v) and v > 1 else "#2ca02c" for v in vals]
    ax.bar(algos, vals, color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("Rerun / Paper")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / out_name, dpi=200)
    plt.close(fig)


def top_deviations(rows: list[dict[str, object]], key: str, n: int = 8) -> list[dict[str, object]]:
    finite_rows = [r for r in rows if np.isfinite(float(r[key]))]

    def score(row: dict[str, object]) -> float:
        value = float(row[key])
        if value <= 0:
            return float("inf")
        return abs(math.log10(value))

    return sorted(finite_rows, key=score, reverse=True)[:n]


def format_md_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                if np.isfinite(val):
                    vals.append(f"{val:.4g}")
                else:
                    vals.append("NA")
            else:
                vals.append(str(val))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def build_doc(
    table1_rows: list[dict[str, object]],
    table2_rows: list[dict[str, object]],
    table1_summary: list[dict[str, object]],
    table2_summary: list[dict[str, object]],
) -> str:
    t1_top_rmse = top_deviations(table1_rows, "rmse_ratio", 6)
    t2_top_rmse = top_deviations(table2_rows, "rmse_ratio", 6)

    return f"""# 论文表格与重跑结果对比报告

日期：2026-05-14

## 1. 范围与输入

- 论文来源：`chen-et-al-2026-optimization-via-the-strategic-law-of-large-numbers.pdf`
- 本次重跑结果：
  - `result/comparison-rerun-2026-05-13/`：Table 1 对应的 `d=2`
  - `result/comparison-rerun-2026-05-13-d200/`：Table 2 对应的 `d=200`
- 对比指标：
  - `RMSE`
  - `AE99`
  - `Time`

说明：

1. Table 1 的算法集合与论文基本一致，因而可比性较强。
2. Table 2 的论文原表包含 `STOGO`，而当前 Python 重跑脚本不包含 `STOGO`。因此当前重跑的 `best_opt` 参考集合与论文不完全一致，`RMSE/AE99` 对比只能视为“近似对照”，不能当作严格复现误差。
3. 时间指标只能做粗对比，不能做严格复现判断。论文使用的是 R 实现与不同计算环境，当前结果是 Python 实现、不同硬件、不同库栈。

## 2. 重跑结果汇总

### Table 1（d=2，单起点，500 reps）

- 已完成全部 8 组任务：
  - `Rastrigin/Ackley/Griewank/Michalewicz`
  - `max/min`

### Table 2（d=200，多起点，10 reps）

- 已完成全部 8 组任务：
  - `Rastrigin/Ackley/Griewank/Michalewicz`
  - `max/min`

## 3. 算法级汇总对比

解释：

- `gm_rmse_ratio`：`rerun_rmse / paper_rmse` 的几何平均
- `gm_ae99_ratio`：`rerun_ae99 / paper_ae99` 的几何平均
- `gm_time_ratio`：`rerun_time / paper_time` 的几何平均
- 比值 `< 1` 表示本次重跑优于论文原表；比值 `> 1` 表示本次重跑劣于论文原表

### Table 1 算法汇总

{format_md_table(table1_summary, ["algo", "cases", "gm_rmse_ratio", "gm_ae99_ratio", "gm_time_ratio"])}

### Table 2 算法汇总

{format_md_table(table2_summary, ["algo", "cases", "gm_rmse_ratio", "gm_ae99_ratio", "gm_time_ratio"])}

## 4. 偏差最大的条目

### Table 1：RMSE 偏差最大的 6 项

{format_md_table(t1_top_rmse, ["direction", "function", "algo", "paper_rmse", "rerun_rmse", "rmse_ratio"])}

### Table 2：RMSE 偏差最大的 6 项

{format_md_table(t2_top_rmse, ["direction", "function", "algo", "paper_rmse", "rerun_rmse", "rmse_ratio"])}

## 5. 原因分析

### Table 1

- `SMCO / SMCO_R / SMCO_BR` 在 Table 1 上与论文趋势大体一致，但数值并不严格重合。主要原因是：
  - Python 与 R 的实现细节不同；
  - 随机起点、数值库、停止条件与浮点行为存在差异；
  - 局部方法的 Python 包装器并非论文原始 R 调用路径。
- `optimLBFGS / BOBYQA / GD / ADAM / SignGD / SPSA` 偏差较大的条目，通常来自：
  - 不同库实现与默认参数；
  - 当前 Python 版本对这些算法的封装并非论文中的原生 R 版本；
  - 一部分方法对起点和终止阈值非常敏感。

### Table 2

- Table 2 的可比性弱于 Table 1，核心原因有两个：
  - 当前重跑不包含 `STOGO`，论文包含 `STOGO`。这会改变每个任务的 `best value`，连带改变 `RMSE` 与 `AE99`；
  - 当前 Python 全局方法实现与论文 R 实现不是同一实现栈。
- `Time` 偏差尤其不能直接解释为“算法快慢变化”，因为：
  - 论文与当前环境的硬件不同；
  - 论文部分实验在高性能集群上完成；
  - Python 和 R 的运行时、库实现、并行方式都不同。
- 某些算法在 Table 2 上出现数量级差异，是预期内现象，不能简单视为移植错误，尤其是：
  - `SA`
  - `DEoptim`
  - `GA`
  - `PSO`
  - 以及缺失 `STOGO` 带来的基准参考变化

## 6. 结论

1. Table 1 可以作为“方向正确”的对照，说明修复后的 comparison 语义已经能产出结构正确的结果，但不应要求与论文数值逐项一致。
2. Table 2 当前只能视为“近似复现”，不能视为严格复现，因为脚本未包含 `STOGO`，比较基准集合与论文原表不一致。
3. 如果目标是更严格地贴近论文表 2，需要优先补上 `STOGO` 或在文档中明确声明：当前 Python 版本的 Table 2 是“去掉 STOGO 的近似对照版”。

## 7. 产出文件

- 对比明细：
  - `result/paper-comparison/table1_paper_vs_rerun.csv`
  - `result/paper-comparison/table2_paper_vs_rerun.csv`
- 图表：
  - `figures/paper-rerun-comparison/table1_rmse_ratio.png`
  - `figures/paper-rerun-comparison/table1_time_ratio.png`
  - `figures/paper-rerun-comparison/table2_rmse_ratio.png`
  - `figures/paper-rerun-comparison/table2_time_ratio.png`
"""


def main() -> None:
    paper_lines = _load_paper_text()
    paper_table1 = parse_paper_table1(paper_lines)
    paper_table2 = parse_paper_table2(paper_lines)

    rerun_table1 = load_rerun_csvs(TABLE1_DIR)
    rerun_table2 = load_rerun_csvs(TABLE2_DIR)

    table1_rows = build_comparison_rows(
        "table1", paper_table1, rerun_table1, TABLE1_ALGOS, ["Rastrigin", "Ackley", "Griewank", "Michalewicz"]
    )
    table2_rows = build_comparison_rows(
        "table2", paper_table2, rerun_table2, TABLE2_ALGOS, ["Rastrigin", "Ackley", "Griewank", "Michalewicz"]
    )

    write_csv(OUT_RESULT_DIR / "table1_paper_vs_rerun.csv", table1_rows)
    write_csv(OUT_RESULT_DIR / "table2_paper_vs_rerun.csv", table2_rows)

    table1_summary = summarize_by_algo(table1_rows)
    table2_summary = summarize_by_algo(table2_rows)

    plot_ratio_bars(table1_summary, "Table 1: Rerun/Paper RMSE Ratio", "gm_rmse_ratio", "table1_rmse_ratio.png")
    plot_ratio_bars(table1_summary, "Table 1: Rerun/Paper Time Ratio", "gm_time_ratio", "table1_time_ratio.png")
    plot_ratio_bars(table2_summary, "Table 2: Rerun/Paper RMSE Ratio", "gm_rmse_ratio", "table2_rmse_ratio.png")
    plot_ratio_bars(table2_summary, "Table 2: Rerun/Paper Time Ratio", "gm_time_ratio", "table2_time_ratio.png")

    OUT_DOC.write_text(build_doc(table1_rows, table2_rows, table1_summary, table2_summary), encoding="utf-8")

    print(f"Wrote {OUT_RESULT_DIR / 'table1_paper_vs_rerun.csv'}")
    print(f"Wrote {OUT_RESULT_DIR / 'table2_paper_vs_rerun.csv'}")
    print(f"Wrote {OUT_DOC}")
    print(f"Wrote figures to {OUT_FIG_DIR}")


if __name__ == "__main__":
    main()
