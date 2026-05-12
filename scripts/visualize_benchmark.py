"""Visualize Table 1 (d=2) and Table 2 (d=100) benchmark results."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

RESULT_DIR = Path(__file__).resolve().parent.parent / "result" / "comparison"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

FUNC_NAMES = ["Ackley", "Griewank", "Michalewicz", "Rastrigin"]

ALGO_COLORS = {
    "SMCO":    "#1f77b4",
    "SMCO_R":  "#2ca02c",
    "SMCO_BR": "#98df8a",
    "GD":      "#ff7f0e",
    "SignGD":  "#ffbb78",
    "ADAM":    "#d62728",
    "SPSA":    "#ff9896",
    "optimLBFGS": "#9467bd",
    "BOBYQA":  "#c5b0d5",
    "GenSA":   "#8c564b",
    "DEoptim": "#e377c2",
    "GA":      "#7f7f7f",
    "PSO":     "#bcbd22",
    "NM":      "#17becf",
}


def load_json(pattern_prefix):
    """Load all JSON files matching table_pattern_{func}_d{dim}_{direction}.json"""
    records = []
    for fpath in sorted(RESULT_DIR.glob(f"{pattern_prefix}_*.json")):
        if "lite" in fpath.name:
            continue
        with open(fpath) as f:
            data = json.load(f)
        for r in data:
            if "label" not in r:
                # Infer label from filename: table2_Ackley_d100_max.json
                stem = fpath.stem  # e.g. table2_Ackley_d100_max
                r["label"] = stem
                r["maximize"] = stem.endswith("_max")
            records.append(r)
    return records


def filter_records(records, funcs=None, algos=None, direction="max"):
    """Filter records by function names, algorithm names, and max/min."""
    out = []
    for r in records:
        label = r["label"]
        if not label.endswith(f"_{direction}"):
            continue
        if funcs and r["name"] not in funcs:
            continue
        if algos and r["algo"] not in algos:
            continue
        out.append(r)
    return out


def plot_table1_rmse_bar(records):
    """Grouped bar chart: rMSE by function, grouped by algorithm (Table 1, d=2)."""
    records = filter_records(records, direction="max")
    algos = sorted(set(r["algo"] for r in records),
                   key=lambda a: next(r["rMSE"] for r in records if r["algo"] == a and r["name"] == FUNC_NAMES[0]))

    n_func = len(FUNC_NAMES)
    n_algo = len(algos)
    x = np.arange(n_func)
    width = 0.8 / n_algo

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, algo in enumerate(algos):
        vals = []
        for fn in FUNC_NAMES:
            match = [r for r in records if r["algo"] == algo and r["name"] == fn]
            vals.append(match[0]["rMSE"] if match else 0)
        offset = (i - n_algo / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=algo, color=ALGO_COLORS.get(algo, "#333"))
        for bar, v in zip(bars, vals):
            if v < ax.get_ylim()[1] * 0.3 if ax.get_ylim()[1] > 0 else True:
                pass

    ax.set_xticks(x)
    ax.set_xticklabels(FUNC_NAMES)
    ax.set_ylabel("rMSE (lower is better)")
    ax.set_title("Table 1: rMSE Comparison (d=2, maximize)")
    ax.set_yscale("log")
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "table1_rmse_bar.png")
    plt.close(fig)
    print("Saved table1_rmse_bar.png")


def plot_table1_time_bar(records):
    """Grouped bar chart: time_avg by function (Table 1)."""
    records = filter_records(records, direction="max")
    algos = sorted(set(r["algo"] for r in records))

    n_func = len(FUNC_NAMES)
    n_algo = len(algos)
    x = np.arange(n_func)
    width = 0.8 / n_algo

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, algo in enumerate(algos):
        vals = []
        for fn in FUNC_NAMES:
            match = [r for r in records if r["algo"] == algo and r["name"] == fn]
            vals.append(match[0]["time_avg"] if match else 0)
        offset = (i - n_algo / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=algo, color=ALGO_COLORS.get(algo, "#333"))

    ax.set_xticks(x)
    ax.set_xticklabels(FUNC_NAMES)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Table 1: Average Runtime (d=2, maximize)")
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "table1_time_bar.png")
    plt.close(fig)
    print("Saved table1_time_bar.png")


def plot_table1_radar(records):
    """Radar/spider chart: normalized rMSE per function for each algorithm."""
    records = filter_records(records, direction="max")
    algos = sorted(set(r["algo"] for r in records))

    # Build matrix: algo x func
    matrix = np.zeros((len(algos), len(FUNC_NAMES)))
    for i, algo in enumerate(algos):
        for j, fn in enumerate(FUNC_NAMES):
            match = [r for r in records if r["algo"] == algo and r["name"] == fn]
            if match:
                matrix[i, j] = match[0]["rMSE"]

    # Normalize per function: lower rMSE is better, so invert
    max_per_func = matrix.max(axis=0)
    max_per_func[max_per_func == 0] = 1
    norm = matrix / max_per_func  # 0=best, 1=worst

    angles = np.linspace(0, 2 * np.pi, len(FUNC_NAMES), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, algo in enumerate(algos):
        vals = norm[i].tolist() + norm[i].tolist()[:1]
        ax.plot(angles, vals, "o-", linewidth=1.5, label=algo, color=ALGO_COLORS.get(algo))
        ax.fill(angles, vals, alpha=0.05, color=ALGO_COLORS.get(algo))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(FUNC_NAMES)
    ax.set_yticklabels([])
    ax.set_title("Table 1: Normalized rMSE (d=2)\n(closer to center = better)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "table1_radar.png")
    plt.close(fig)
    print("Saved table1_radar.png")


def plot_table1_ae_quantile(records):
    """Grouped bar chart comparing AE50/AE95/AE99 for each function."""
    records = filter_records(records, direction="max")
    algos = sorted(set(r["algo"] for r in records))

    for fn in FUNC_NAMES:
        fn_records = [r for r in records if r["name"] == fn]
        if not fn_records:
            continue
        algos_fn = sorted(fn_records, key=lambda r: r["rMSE"])
        algos_fn = [r["algo"] for r in algos_fn]

        x = np.arange(len(algos_fn))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(10, len(algos_fn) * 1.2), 5))
        ae50 = [next(r["AE50"] for r in fn_records if r["algo"] == a) for a in algos_fn]
        ae95 = [next(r["AE95"] for r in fn_records if r["algo"] == a) for a in algos_fn]
        ae99 = [next(r["AE99"] for r in fn_records if r["algo"] == a) for a in algos_fn]

        ax.bar(x - width, ae50, width, label="AE50", color="#4c72b0")
        ax.bar(x, ae95, width, label="AE95", color="#dd8452")
        ax.bar(x + width, ae99, width, label="AE99", color="#55a868")

        ax.set_xticks(x)
        ax.set_xticklabels(algos_fn, rotation=30, ha="right")
        ax.set_ylabel("Absolute Error")
        ax.set_title(f"Table 1: AE Quantiles — {fn} (d=2)")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        safe_fn = fn.lower()
        fig.savefig(OUT_DIR / f"table1_ae_{safe_fn}.png")
        plt.close(fig)
        print(f"Saved table1_ae_{safe_fn}.png")


def plot_table2_rmse_bar(records):
    """Grouped bar chart: rMSE by function (Table 2, d=100)."""
    records = filter_records(records, direction="max")
    algos = sorted(set(r["algo"] for r in records))

    n_func = len(FUNC_NAMES)
    n_algo = len(algos)
    x = np.arange(n_func)
    width = 0.8 / n_algo

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, algo in enumerate(algos):
        vals = []
        for fn in FUNC_NAMES:
            match = [r for r in records if r["algo"] == algo and r["name"] == fn]
            vals.append(match[0]["rMSE"] if match else 0)
        offset = (i - n_algo / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=algo, color=ALGO_COLORS.get(algo, "#333"))

    ax.set_xticks(x)
    ax.set_xticklabels(FUNC_NAMES)
    ax.set_ylabel("rMSE (lower is better)")
    ax.set_title("Table 2: rMSE Comparison (d=100, maximize)")
    ax.set_yscale("log")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "table2_rmse_bar.png")
    plt.close(fig)
    print("Saved table2_rmse_bar.png")


def plot_table2_time_bar(records):
    """Grouped bar chart: time_avg (Table 2, d=100)."""
    records = filter_records(records, direction="max")
    algos = sorted(set(r["algo"] for r in records))

    n_func = len(FUNC_NAMES)
    n_algo = len(algos)
    x = np.arange(n_func)
    width = 0.8 / n_algo

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, algo in enumerate(algos):
        vals = []
        for fn in FUNC_NAMES:
            match = [r for r in records if r["algo"] == algo and r["name"] == fn]
            vals.append(match[0]["time_avg"] if match else 0)
        offset = (i - n_algo / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=algo, color=ALGO_COLORS.get(algo, "#333"))

    ax.set_xticks(x)
    ax.set_xticklabels(FUNC_NAMES)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Table 2: Average Runtime (d=100, maximize)")
    ax.set_yscale("log")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "table2_time_bar.png")
    plt.close(fig)
    print("Saved table2_time_bar.png")


def plot_table2_radar(records):
    """Radar chart for Table 2 normalized rMSE."""
    records = filter_records(records, direction="max")
    algos = sorted(set(r["algo"] for r in records))

    matrix = np.zeros((len(algos), len(FUNC_NAMES)))
    for i, algo in enumerate(algos):
        for j, fn in enumerate(FUNC_NAMES):
            match = [r for r in records if r["algo"] == algo and r["name"] == fn]
            if match:
                matrix[i, j] = match[0]["rMSE"]

    max_per_func = matrix.max(axis=0)
    max_per_func[max_per_func == 0] = 1
    norm = matrix / max_per_func

    angles = np.linspace(0, 2 * np.pi, len(FUNC_NAMES), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, algo in enumerate(algos):
        vals = norm[i].tolist() + norm[i].tolist()[:1]
        ax.plot(angles, vals, "o-", linewidth=1.8, label=algo, color=ALGO_COLORS.get(algo))
        ax.fill(angles, vals, alpha=0.08, color=ALGO_COLORS.get(algo))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(FUNC_NAMES)
    ax.set_yticklabels([])
    ax.set_title("Table 2: Normalized rMSE (d=100)\n(closer to center = better)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "table2_radar.png")
    plt.close(fig)
    print("Saved table2_radar.png")


def plot_table2_rmse_vs_time(records):
    """Scatter plot: rMSE vs time_avg for each function (Table 2)."""
    records = filter_records(records, direction="max")

    fig, axes = plt.subplots(1, len(FUNC_NAMES), figsize=(18, 5), sharey=False)
    for idx, fn in enumerate(FUNC_NAMES):
        ax = axes[idx]
        fn_records = [r for r in records if r["name"] == fn]
        for r in fn_records:
            ax.scatter(r["time_avg"], r["rMSE"], s=100,
                       color=ALGO_COLORS.get(r["algo"], "#333"),
                       label=r["algo"], zorder=3, edgecolors="black", linewidths=0.5)
        ax.set_xlabel("Time (s)")
        if idx == 0:
            ax.set_ylabel("rMSE")
        ax.set_title(fn)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        # Annotate points
        for r in fn_records:
            ax.annotate(r["algo"], (r["time_avg"], r["rMSE"]),
                        fontsize=7, textcoords="offset points", xytext=(5, 5))

    fig.suptitle("Table 2: rMSE vs Runtime (d=100)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "table2_rmse_vs_time.png")
    plt.close(fig)
    print("Saved table2_rmse_vs_time.png")


def plot_table1_max_vs_min(records):
    """Compare max vs min direction rMSE side by side."""
    fig, axes = plt.subplots(1, len(FUNC_NAMES), figsize=(18, 5))
    for idx, fn in enumerate(FUNC_NAMES):
        ax = axes[idx]
        max_recs = [r for r in records if r["name"] == fn and r["label"].endswith("_max")]
        min_recs = [r for r in records if r["name"] == fn and r["label"].endswith("_min")]
        algos = sorted(set(r["algo"] for r in max_recs))
        x = np.arange(len(algos))
        width = 0.35
        max_vals = [next((r["rMSE"] for r in max_recs if r["algo"] == a), 0) for a in algos]
        min_vals = [next((r["rMSE"] for r in min_recs if r["algo"] == a), 0) for a in algos]
        ax.bar(x - width / 2, max_vals, width, label="max", color="#4c72b0")
        ax.bar(x + width / 2, min_vals, width, label="min", color="#dd8452")
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("rMSE")
        ax.set_title(f"{fn}")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Table 1: Max vs Min rMSE (d=2)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "table1_max_vs_min.png")
    plt.close(fig)
    print("Saved table1_max_vs_min.png")


def main():
    print("Loading Table 1 data (d=2)...")
    t1 = load_json("table1")
    print(f"  {len(t1)} records loaded")

    print("Loading Table 2 data (d=100)...")
    t2 = load_json("table2")
    print(f"  {len(t2)} records loaded")

    print("\n--- Generating Table 1 figures (d=2) ---")
    plot_table1_rmse_bar(t1)
    plot_table1_time_bar(t1)
    plot_table1_radar(t1)
    plot_table1_ae_quantile(t1)
    plot_table1_max_vs_min(t1)

    print("\n--- Generating Table 2 figures (d=100) ---")
    plot_table2_rmse_bar(t2)
    plot_table2_time_bar(t2)
    plot_table2_radar(t2)
    plot_table2_rmse_vs_time(t2)

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
