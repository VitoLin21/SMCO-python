"""Generate publication-style summary tables for Table 1 and Table 2 benchmarks."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "monospace",
})

RESULT_DIR = Path(__file__).resolve().parent.parent / "result" / "comparison"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

FUNC_NAMES = ["Ackley", "Griewank", "Michalewicz", "Rastrigin"]


def load_json(pattern_prefix):
    records = []
    for fpath in sorted(RESULT_DIR.glob(f"{pattern_prefix}_*.json")):
        if "lite" in fpath.name:
            continue
        with open(fpath) as f:
            data = json.load(f)
        for r in data:
            if "label" not in r:
                stem = fpath.stem
                r["label"] = stem
                r["maximize"] = stem.endswith("_max")
            records.append(r)
    return records


def fmt(val, precision=4):
    """Format a number in scientific notation if very small, else fixed point."""
    if val == 0:
        return "0"
    if abs(val) < 0.001 or abs(val) >= 1e4:
        return f"{val:.{precision-1}e}"
    if abs(val) < 1:
        return f"{val:.{precision}f}"
    return f"{val:.{precision - 1}f}"


def build_table_data(records, funcs, direction="max"):
    """Build a dict: {func_name: {algo: {metric: value}}}"""
    table = {}
    for fn in funcs:
        table[fn] = {}
        fn_recs = [r for r in records if r["name"] == fn and r["label"].endswith(f"_{direction}")]
        for r in fn_recs:
            table[fn][r["algo"]] = {
                "rMSE": r["rMSE"],
                "AE50": r["AE50"],
                "AE95": r["AE95"],
                "AE99": r["AE99"],
                "fopt_mean": r["fopt_mean"],
                "fopt_sd": r["fopt_sd"],
                "time_avg": r["time_avg"],
            }
    return table


def find_best_algo(table_data, metric="rMSE"):
    """For each function, find the best (lowest) algorithm for a metric."""
    best = {}
    for fn, algos in table_data.items():
        best_val = float("inf")
        best_algo = None
        for algo, vals in algos.items():
            if vals[metric] < best_val:
                best_val = vals[metric]
                best_algo = algo
        best[fn] = best_algo
    return best


def render_table_png(table_data, algos, title, filename, metrics=("rMSE", "AE95", "time_avg")):
    """Render a table as a matplotlib figure and save to PNG."""

    metric_labels = {
        "rMSE": "rMSE",
        "AE95": "AE95",
        "time_avg": "Time(s)",
    }

    n_func = len(FUNC_NAMES)
    n_algo = len(algos)
    n_metrics = len(metrics)

    # Cell layout: each algo gets 3 sub-columns (one per metric)
    # Columns: Algorithm | Ackley(rMSE, AE95, Time) | Griewank(...) | ...
    total_cols = 1 + n_func * n_metrics
    total_rows = n_algo + 1  # header + data

    # Figure sizing
    col_w = 1.8
    row_h = 0.45
    fig_w = total_cols * col_w + 2.5
    fig_h = total_rows * row_h + 2.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, total_cols)
    ax.set_ylim(0, total_rows + 1)
    ax.axis("off")

    # Colors
    header_bg = "#4472C4"
    header_fg = "white"
    algo_col_bg = "#D9E2F3"
    best_bg = "#C6EFCE"  # green highlight for best
    row_even = "#F2F2F2"
    row_odd = "white"

    def draw_cell(x, y, w, h, text, bg="white", fg="black", bold=False, fontsize=9):
        rect = plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor="white", linewidth=1)
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=fg, fontweight=weight)

    # --- Header row ---
    y = total_rows
    # Algorithm column
    draw_cell(0, y, 1, 1, "Algorithm", header_bg, header_fg, bold=True, fontsize=10)

    for j, fn in enumerate(FUNC_NAMES):
        x_start = 1 + j * n_metrics
        # Function name spanning all metric sub-cols
        draw_cell(x_start, y + 0.5, n_metrics, 0.5, fn, header_bg, header_fg, bold=True, fontsize=10)
        for k, m in enumerate(metrics):
            draw_cell(x_start + k, y, 1, 0.5, metric_labels[m], "#5B9BD5", header_fg, bold=False, fontsize=8)

    # --- Data rows ---
    best_per_func = {}
    for fn in FUNC_NAMES:
        best_algo = min(table_data[fn].keys(), key=lambda a: table_data[fn][a]["rMSE"])
        best_per_func[fn] = best_algo

    for i, algo in enumerate(algos):
        y = total_rows - 1 - i
        bg = row_even if i % 2 == 0 else row_odd
        # Algorithm name
        draw_cell(0, y, 1, 1, algo, algo_col_bg, "black", bold=True, fontsize=9)

        for j, fn in enumerate(FUNC_NAMES):
            x_start = 1 + j * n_metrics
            if algo in table_data[fn]:
                vals = table_data[fn][algo]
                is_best = (algo == best_per_func[fn])
                for k, m in enumerate(metrics):
                    cell_bg = best_bg if is_best else bg
                    fg = "#006100" if is_best else "black"
                    weight = "bold" if is_best else "normal"
                    draw_cell(x_start + k, y, 1, 1, fmt(vals[m]), cell_bg, fg, bold=is_best)
            else:
                for k in range(n_metrics):
                    draw_cell(x_start + k, y, 1, 1, "—", bg, "gray")

    # Title
    ax.text(total_cols / 2, total_rows + 0.7, title,
            ha="center", va="center", fontsize=14, fontweight="bold")

    # Legend
    ax.text(0.5, -0.3,
            "Green cells = best rMSE for that function.   Metrics: rMSE (lower=better), AE95 (lower=better), Time(s)",
            ha="left", va="center", fontsize=8, color="gray")

    fig.savefig(OUT_DIR / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {filename}")


def main():
    # Load data
    t1 = load_json("table1")
    t2 = load_json("table2")

    # Table 1
    t1_data = build_table_data(t1, FUNC_NAMES, direction="max")
    t1_algos = ["SMCO", "SMCO_R", "SMCO_BR", "GD", "SignGD", "ADAM", "SPSA", "optimLBFGS", "BOBYQA"]
    render_table_png(t1_data, t1_algos,
                     "Table 1: Optimization Benchmark (d=2, 500 reps)",
                     "table1_summary.png")

    # Table 2
    t2_data = build_table_data(t2, FUNC_NAMES, direction="max")
    t2_algos = ["SMCO_R", "GenSA", "DEoptim", "GA", "PSO"]
    render_table_png(t2_data, t2_algos,
                     "Table 2: Optimization Benchmark (d=100, 500 reps)",
                     "table2_summary.png")


if __name__ == "__main__":
    main()
