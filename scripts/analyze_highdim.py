#!/usr/bin/env python
"""Analyze high-dimensional SMCO-EVO vs SMCO comparison results."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


def _load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Exp A: Scaling analysis
# ---------------------------------------------------------------------------

def analyze_exp_a(input_dir: Path, report: list[str]):
    path = input_dir / "expA_scaling.csv"
    if not path.exists():
        report.append("Experiment A: No data.\n")
        return

    rows = _load_csv(path)
    report.append("## Experiment A: Scaling Benchmark (SMCO_R vs SMCO_R_EVO)\n")

    # Pairwise comparison
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["func"], int(r["dim"]), int(r["rep"]))].append(r)

    pairs = []
    for key, entries in sorted(grouped.items()):
        func, dim, rep = key
        base_entries = [e for e in entries if e["algo"] == "SMCO_R"]
        evo_entries = [e for e in entries if e["algo"] == "SMCO_R_EVO"]
        if len(base_entries) == 1 and len(evo_entries) == 1:
            pairs.append({
                "func": func, "dim": dim, "rep": rep,
                "base_fopt": float(base_entries[0]["fopt"]),
                "evo_fopt": float(evo_entries[0]["fopt"]),
                "base_time": float(base_entries[0]["time"]),
                "evo_time": float(evo_entries[0]["time"]),
            })

    # Win/loss by dimension
    by_dim = defaultdict(lambda: {"w": 0, "l": 0, "t": 0, "n": 0})
    for p in pairs:
        d = by_dim[p["dim"]]
        d["n"] += 1
        if p["evo_fopt"] > p["base_fopt"]:
            d["w"] += 1
        elif p["evo_fopt"] < p["base_fopt"]:
            d["l"] += 1
        else:
            d["t"] += 1

    report.append("### Win/Loss by Dimension\n")
    report.append("| Dim | N | EVO wins | Base wins | Ties | EVO win rate |")
    report.append("|-----|---|----------|-----------|------|-------------|")
    for dim in sorted(by_dim):
        d = by_dim[dim]
        rate = d["w"] / d["n"] * 100 if d["n"] > 0 else 0
        report.append(f"| {dim}D | {d['n']} | {d['w']} | {d['l']} | {d['t']} | {rate:.1f}% |")
    report.append("")

    # Wilcoxon per (func, dim)
    report.append("### Wilcoxon Test by Function and Dimension\n")
    func_dim_data = defaultdict(lambda: {"base": [], "evo": []})
    for p in pairs:
        func_dim_data[(p["func"], p["dim"])]["base"].append(p["base_fopt"])
        func_dim_data[(p["func"], p["dim"])]["evo"].append(p["evo_fopt"])

    report.append("| Function | Dim | N | EVO mean | Base mean | Rel. impr. | p-value |")
    report.append("|----------|-----|---|----------|-----------|------------|---------|")
    stat_rows = []
    for (func, dim) in sorted(func_dim_data):
        vals = func_dim_data[(func, dim)]
        n = len(vals["base"])
        base_arr = np.array(vals["base"])
        evo_arr = np.array(vals["evo"])
        evo_mean = float(np.mean(evo_arr))
        base_mean = float(np.mean(base_arr))
        rel_impr = (evo_mean - base_mean) / abs(base_mean) * 100 if base_mean != 0 else 0

        if n >= 5 and np.any(base_arr != evo_arr):
            try:
                _, pval = wilcoxon(base_arr, evo_arr, alternative="two-sided")
            except ValueError:
                pval = float("nan")
        else:
            pval = float("nan")

        sig = " **" if (not math.isnan(pval) and pval < 0.05) else ""
        pval_str = f"{pval:.4f}" if not math.isnan(pval) else "N/A"
        report.append(
            f"| {func} | {dim}D | {n} | {evo_mean:.4f} | {base_mean:.4f} | "
            f"{rel_impr:+.2f}% | {pval_str}{sig} |"
        )
        stat_rows.append({"func": func, "dim": dim, "evo_mean": evo_mean,
                          "base_mean": base_mean, "rel_impr": rel_impr, "pval": pval})

    report.append("")

    # Scaling plot: EVO win rate vs dimension
    _plot_scaling(pairs, by_dim, input_dir)

    # Relative improvement heatmap
    _plot_improvement_heatmap(stat_rows, input_dir)

    return pairs


def _plot_scaling(pairs, by_dim, output_dir):
    dims = sorted(by_dim)
    win_rates = [by_dim[d]["w"] / by_dim[d]["n"] * 100 if by_dim[d]["n"] > 0 else 0 for d in dims]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dims, win_rates, "o-", linewidth=2, markersize=8, color="#2196F3")
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% (no advantage)")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("EVO win rate (%)")
    ax.set_title("SMCO_R_EVO Win Rate vs Dimension")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    for d, wr in zip(dims, win_rates):
        ax.annotate(f"{wr:.0f}%", (d, wr), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "expA_win_rate_scaling.png", dpi=150)
    plt.close()


def _plot_improvement_heatmap(stat_rows, output_dir):
    if not stat_rows:
        return
    funcs = sorted(set(r["func"] for r in stat_rows))
    dims = sorted(set(r["dim"] for r in stat_rows))

    heat = np.full((len(funcs), len(dims)), np.nan)
    for r in stat_rows:
        i = funcs.index(r["func"])
        j = dims.index(r["dim"])
        heat[i, j] = r["rel_impr"]

    fig, ax = plt.subplots(figsize=(max(6, len(dims) * 1.5), max(4, len(funcs) * 0.6)))
    im = ax.imshow(heat, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([f"{d}D" for d in dims])
    ax.set_yticks(range(len(funcs)))
    ax.set_yticklabels(funcs)
    ax.set_title("Relative Improvement of EVO over Base (%)")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if not np.isnan(heat[i, j]):
                color = "white" if abs(heat[i, j]) > 5 else "black"
                ax.text(j, i, f"{heat[i,j]:+.1f}%", ha="center", va="center",
                        fontsize=8, color=color)
    fig.colorbar(im, ax=ax, label="Relative improvement (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "expA_improvement_heatmap.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Exp B: Variant sweep
# ---------------------------------------------------------------------------

def analyze_exp_b(input_dir: Path, report: list[str]):
    path = input_dir / "expB_variants.csv"
    if not path.exists():
        report.append("Experiment B: No data.\n")
        return

    rows = _load_csv(path)
    report.append("## Experiment B: All Variant Pairs at High Dimensions\n")

    # Per pair analysis
    for base_algo, evo_algo in [("SMCO", "SMCO_EVO"), ("SMCO_R", "SMCO_R_EVO"), ("SMCO_BR", "SMCO_BR_EVO")]:
        pair_label = f"{base_algo} vs {evo_algo}"
        pair_rows = [r for r in rows if r["algo"] in (base_algo, evo_algo)]

        grouped = defaultdict(list)
        for r in pair_rows:
            grouped[(r["func"], int(r["dim"]), int(r["rep"]))].append(r)

        wins, losses, ties = 0, 0, 0
        by_dim = defaultdict(lambda: {"w": 0, "l": 0, "t": 0})
        for key, entries in grouped.items():
            func, dim, rep = key
            base_e = [e for e in entries if e["algo"] == base_algo]
            evo_e = [e for e in entries if e["algo"] == evo_algo]
            if len(base_e) == 1 and len(evo_e) == 1:
                bf = float(base_e[0]["fopt"])
                ef = float(evo_e[0]["fopt"])
                d = by_dim[dim]
                if ef > bf:
                    wins += 1; d["w"] += 1
                elif ef < bf:
                    losses += 1; d["l"] += 1
                else:
                    ties += 1; d["t"] += 1

        total = wins + losses + ties
        report.append(f"### {pair_label}\n")
        report.append(f"Overall: EVO {wins}/{total} wins ({wins/total*100:.1f}%)")
        for dim in sorted(by_dim):
            d = by_dim[dim]
            n = d["w"] + d["l"] + d["t"]
            report.append(f"- {dim}D: {d['w']}/{n} wins")
        report.append("")

    # Bar chart per pair per dim
    _plot_variant_bars(rows, input_dir)


def _plot_variant_bars(rows, output_dir):
    pairs = [("SMCO", "SMCO_EVO"), ("SMCO_R", "SMCO_R_EVO"), ("SMCO_BR", "SMCO_BR_EVO")]
    dims = sorted(set(int(r["dim"]) for r in rows))
    pair_names = {
        ("SMCO", "SMCO_EVO"): "SMCO_vs_SMCO_EVO",
        ("SMCO_R", "SMCO_R_EVO"): "SMCO_R_vs_SMCO_R_EVO",
        ("SMCO_BR", "SMCO_BR_EVO"): "SMCO_BR_vs_SMCO_BR_EVO",
    }

    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4), squeeze=False)
    for idx, dim in enumerate(dims):
        ax = axes[0, idx]
        x = np.arange(len(pairs))
        w_vals = []
        l_vals = []
        labels = []
        for base, evo in pairs:
            grouped = defaultdict(list)
            pair_name = pair_names[(base, evo)]
            for r in rows:
                if r.get("pair", "") == pair_name and int(r["dim"]) == dim:
                    grouped[(r["func"], int(r["dim"]), int(r["rep"]))].append(r)
            bw = 0
            bl = 0
            for entries in grouped.values():
                algos = {r["algo"]: r for r in entries}
                if base not in algos or evo not in algos:
                    continue
                if float(algos[evo]["fopt"]) > float(algos[base]["fopt"]):
                    bw += 1
                elif float(algos[evo]["fopt"]) < float(algos[base]["fopt"]):
                    bl += 1
            w_vals.append(bw)
            l_vals.append(bl)
            labels.append(base)

        ax.bar(x - 0.15, w_vals, 0.3, label="EVO wins", color="#2196F3")
        ax.bar(x + 0.15, l_vals, 0.3, label="Base wins", color="#FF5722")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Count")
        ax.set_title(f"{dim}D")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Variant Comparison by Dimension", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "expB_variant_bars.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Exp C: Convergence
# ---------------------------------------------------------------------------

def analyze_exp_c(input_dir: Path, report: list[str]):
    path = input_dir / "expC_convergence.csv"
    if not path.exists():
        report.append("Experiment C: No data.\n")
        return

    rows = _load_csv(path)
    report.append("## Experiment C: Convergence Profiling\n")

    func_dims = sorted(set((r["func"], int(r["dim"])) for r in rows))
    n_plots = len(func_dims)
    ncols = min(3, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    report.append("| Function | Dim | SMCO_R 95% iter | SMCO_R_EVO 95% iter | Ratio |")
    report.append("|----------|-----|-----------------|---------------------|-------|")

    for plot_idx, (func, dim) in enumerate(func_dims):
        ax = axes[plot_idx // ncols][plot_idx % ncols]
        for algo in ("SMCO_R", "SMCO_R_EVO"):
            entries = [r for r in rows if r["algo"] == algo and r["func"] == func and int(r["dim"]) == dim]
            by_iter = defaultdict(list)
            for e in entries:
                by_iter[int(e["iteration"])].append(float(e["runmax"]))
            iters = sorted(by_iter)
            means = [np.mean(by_iter[i]) for i in iters]
            stds = [np.std(by_iter[i]) for i in iters]
            label = algo.replace("SMCO_R", "SMCO-R").replace("SMCO_EVO", "SMCO-EVO")
            ax.plot(iters, means, label=label, linewidth=1.5)
            ax.fill_between(iters, np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds), alpha=0.15)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Runmax f(x)")
        ax.set_title(f"{func} {dim}D")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Compute 95% convergence
        speeds = {}
        for algo in ("SMCO_R", "SMCO_R_EVO"):
            entries = [r for r in rows if r["algo"] == algo and r["func"] == func and int(r["dim"]) == dim]
            by_iter = defaultdict(list)
            for e in entries:
                by_iter[int(e["iteration"])].append(float(e["runmax"]))
            iters = sorted(by_iter)
            means = [np.mean(by_iter[i]) for i in iters]
            if not means:
                speeds[algo] = 0
                continue
            target = means[-1] * 0.95
            for i, m in enumerate(means):
                if m >= target:
                    speeds[algo] = iters[i]
                    break
            else:
                speeds[algo] = iters[-1]

        if len(speeds) == 2 and speeds["SMCO_R_EVO"] > 0:
            ratio = speeds["SMCO_R"] / speeds["SMCO_R_EVO"]
            report.append(f"| {func} | {dim}D | {speeds['SMCO_R']} | {speeds['SMCO_R_EVO']} | {ratio:.2f}x |")

    # Hide empty subplots
    for idx in range(n_plots, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig(input_dir / "expC_convergence_curves.png", dpi=150)
    plt.close()
    report.append("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze high-dim comparison results")
    parser.add_argument("--input-dir", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    report = [
        "# High-Dimensional SMCO-EVO vs SMCO Comparison Report",
        f"Generated: {date.today()}",
        f"Input: {input_dir}\n",
    ]

    analyze_exp_a(input_dir, report)
    analyze_exp_b(input_dir, report)
    analyze_exp_c(input_dir, report)

    (input_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Report: {input_dir / 'report.md'}")


if __name__ == "__main__":
    main()
