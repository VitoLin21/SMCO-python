#!/usr/bin/env python
"""Analyze SMCO-EVO vs SMCO comparison results and generate report + figures."""
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
from scipy.stats import friedmanchisquare, wilcoxon


def _load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _save_csv(rows: list[dict], path: Path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Experiment 1 analysis
# ---------------------------------------------------------------------------

def analyze_experiment1(input_dir: Path, report_lines: list[str]):
    wilcoxon_path = input_dir / "experiment1_wilcoxon.csv"
    if not wilcoxon_path.exists():
        report_lines.append("Experiment 1: No data found.\n")
        return

    rows = _load_csv(wilcoxon_path)
    report_lines.append("## Experiment 1: Core Comparison\n")

    # Overall wins/losses
    total_wins = sum(int(r["evo_wins"]) for r in rows)
    total_losses = sum(int(r["base_wins"]) for r in rows)
    total_ties = sum(int(r["ties"]) for r in rows)
    n_sig = sum(int(r["significant_005"]) for r in rows)

    report_lines.append(f"Total EVO wins: {total_wins}, Base wins: {total_losses}, Ties: {total_ties}")
    report_lines.append(f"Significant differences (p<0.05): {n_sig}/{len(rows)}\n")

    # Per-dim summary
    by_dim = defaultdict(lambda: {"w": 0, "l": 0, "t": 0})
    for r in rows:
        d = by_dim[int(r["dim"])]
        d["w"] += int(r["evo_wins"])
        d["l"] += int(r["base_wins"])
        d["t"] += int(r["ties"])

    report_lines.append("| Dim | EVO wins | Base wins | Ties |")
    report_lines.append("|-----|----------|-----------|------|")
    for dim in sorted(by_dim):
        d = by_dim[dim]
        report_lines.append(f"| {dim}D | {d['w']} | {d['l']} | {d['t']} |")
    report_lines.append("")

    # Detailed table
    report_lines.append("### Wilcoxon Test Results (p < 0.05 marked with *)\n")
    report_lines.append("| Pair | Function | Dim | EVO mean | Base mean | EVO wins | p-value |")
    report_lines.append("|------|----------|-----|----------|-----------|----------|---------|")
    for r in sorted(rows, key=lambda x: (x["pair"], int(x["dim"]), x["func"])):
        pval = float(r["p_value"])
        sig = " *" if pval < 0.05 else ""
        pval_str = f"{pval:.4f}" if not math.isnan(pval) else "N/A"
        report_lines.append(
            f"| {r['pair']} | {r['func']} | {r['dim']}D | "
            f"{float(r['evo_mean']):.4f} | {float(r['base_mean']):.4f} | "
            f"{r['evo_wins']}/{r['n_reps']} | {pval_str}{sig} |"
        )
    report_lines.append("")

    # Bar chart: wins by function x dim
    _plot_exp1_bars(rows, input_dir)


def _plot_exp1_bars(rows: list[dict], output_dir: Path):
    pairs = sorted(set(r["pair"] for r in rows))
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5), squeeze=False)
    for idx, pair in enumerate(pairs):
        ax = axes[0, idx]
        pair_rows = [r for r in rows if r["pair"] == pair]
        labels = [f"{r['func']}\n{r['dim']}D" for r in pair_rows]
        evo_wins = [int(r["evo_wins"]) for r in pair_rows]
        base_wins = [int(r["base_wins"]) for r in pair_rows]
        ties = [int(r["ties"]) for r in pair_rows]

        x = np.arange(len(labels))
        w = 0.25
        ax.bar(x - w, evo_wins, w, label="EVO wins", color="#2196F3")
        ax.bar(x, base_wins, w, label="Base wins", color="#FF5722")
        ax.bar(x + w, ties, w, label="Ties", color="#9E9E9E")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(pair.replace("_vs_", "\nvs "))
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "exp1_wins_bars.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Experiment 2 analysis
# ---------------------------------------------------------------------------

def analyze_experiment2(input_dir: Path, report_lines: list[str]):
    conv_path = input_dir / "experiment2_convergence.csv"
    if not conv_path.exists():
        report_lines.append("Experiment 2: No data found.\n")
        return

    rows = _load_csv(conv_path)
    report_lines.append("## Experiment 2: Convergence Curves\n")

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["algo"], r["func"], int(r["dim"]))].append(r)

    fig, axes = plt.subplots(1, len(grouped) // 2, figsize=(4 * (len(grouped) // 2), 4), squeeze=False)

    plot_idx = 0
    func_dims = sorted(set((r["func"], int(r["dim"])) for r in rows))

    for func_name, dim in func_dims:
        ax = axes[0, plot_idx] if len(func_dims) > 1 else axes[0, 0]
        for algo in ("SMCO_R", "SMCO_R_EVO"):
            key = (algo, func_name, dim)
            entries = grouped.get(key, [])
            if not entries:
                continue
            by_iter = defaultdict(list)
            for e in entries:
                by_iter[int(e["iteration"])].append(float(e["runmax"]))
            iters = sorted(by_iter)
            means = [np.mean(by_iter[i]) for i in iters]
            stds = [np.std(by_iter[i]) for i in iters]

            label = algo.replace("SMCO_R", "SMCO-R")
            ax.plot(iters, means, label=label, linewidth=1.5)
            ax.fill_between(iters, np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds), alpha=0.2)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Runmax f(x)")
        ax.set_title(f"{func_name} {dim}D")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    plt.tight_layout()
    plt.savefig(input_dir / "exp2_convergence_curves.png", dpi=150)
    plt.close()
    report_lines.append(f"Convergence curves saved to exp2_convergence_curves.png\n")

    # Compute convergence speed metric
    report_lines.append("### Iterations to reach 95% of final value\n")
    report_lines.append("| Function | Dim | SMCO_R iter | SMCO_R_EVO iter | Speedup |")
    report_lines.append("|----------|-----|-------------|-----------------|---------|")
    for func_name, dim in func_dims:
        speeds = {}
        for algo in ("SMCO_R", "SMCO_R_EVO"):
            key = (algo, func_name, dim)
            entries = grouped.get(key, [])
            by_iter = defaultdict(list)
            for e in entries:
                by_iter[int(e["iteration"])].append(float(e["runmax"]))
            iters = sorted(by_iter)
            means = [np.mean(by_iter[i]) for i in iters]
            if not means:
                continue
            target = means[-1] * 0.95
            for i, m in enumerate(means):
                if m >= target:
                    speeds[algo] = iters[i]
                    break
            else:
                speeds[algo] = iters[-1] if iters else 0
        if len(speeds) == 2:
            ratio = speeds["SMCO_R"] / max(speeds["SMCO_R_EVO"], 1)
            report_lines.append(
                f"| {func_name} | {dim}D | {speeds['SMCO_R']} | {speeds['SMCO_R_EVO']} | {ratio:.2f}x |"
            )
    report_lines.append("")


# ---------------------------------------------------------------------------
# Experiment 3 analysis
# ---------------------------------------------------------------------------

def analyze_experiment3(input_dir: Path, report_lines: list[str]):
    strat_path = input_dir / "experiment3_strategies.csv"
    if not strat_path.exists():
        report_lines.append("Experiment 3: No data found.\n")
        return

    rows = _load_csv(strat_path)
    report_lines.append("## Experiment 3: Evolution Strategy Sensitivity\n")

    strategies = sorted(set(r["strategy"] for r in rows))
    funcs = sorted(set(r["func"] for r in rows))

    # Mean fopt per strategy x func
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["strategy"], r["func"])].append(float(r["fopt"]))

    report_lines.append("| Strategy | " + " | ".join(funcs) + " | Mean |")
    report_lines.append("|----------|" + "|".join(["------"] * (len(funcs) + 1)))
    for s in strategies:
        vals = [grouped.get((s, f), []) for f in funcs]
        means = [np.mean(v) if v else float("nan") for v in vals]
        overall = np.mean([float(r["fopt"]) for r in rows if r["strategy"] == s])
        report_lines.append(f"| {s} | " + " | ".join(f"{m:.4f}" for m in means) + f" | {overall:.4f} |")
    report_lines.append("")

    # Friedman test
    for func in funcs:
        groups = []
        for s in strategies:
            vals = [float(r["fopt"]) for r in rows if r["strategy"] == s and r["func"] == func]
            if len(vals) >= 5:
                groups.append(vals[:min(len(v) for v in groups)] if groups else vals)
        if len(groups) == len(strategies) and all(len(g) >= 5 for g in groups):
            min_len = min(len(g) for g in groups)
            groups = [g[:min_len] for g in groups]
            try:
                stat, pval = friedmanchisquare(*groups)
                report_lines.append(f"Friedman test ({func}): stat={stat:.4f}, p={pval:.4f}")
            except Exception:
                report_lines.append(f"Friedman test ({func}): could not compute")
    report_lines.append("")

    # Bar chart
    _plot_exp3_bars(rows, strategies, funcs, input_dir)


def _plot_exp3_bars(rows, strategies, funcs, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["strategy"], r["func"])].append(float(r["fopt"]))

    x = np.arange(len(funcs))
    w = 0.8 / len(strategies)
    for i, s in enumerate(strategies):
        means = [np.mean(grouped.get((s, f), [0])) for f in funcs]
        stds = [np.std(grouped.get((s, f), [0])) for f in funcs]
        ax.bar(x + i * w, means, w, yerr=stds, label=s, capsize=3)

    ax.set_xticks(x + w * len(strategies) / 2 - w / 2)
    ax.set_xticklabels(funcs)
    ax.set_ylabel("Mean f_optimal")
    ax.set_title("Evolution Strategy Comparison (10D)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "exp3_strategy_bars.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Experiment 4 analysis
# ---------------------------------------------------------------------------

def analyze_experiment4(input_dir: Path, report_lines: list[str]):
    params_path = input_dir / "experiment4_params.csv"
    if not params_path.exists():
        report_lines.append("Experiment 4: No data found.\n")
        return

    rows = _load_csv(params_path)
    report_lines.append("## Experiment 4: Parameter Sensitivity\n")

    funcs = sorted(set(r["func"] for r in rows))
    elim_rates = sorted(set(float(r["elimination_rate"]) for r in rows))
    evo_points = sorted(set(r["evolution_points"] for r in rows))

    # Heatmap per function
    for func in funcs:
        grouped = defaultdict(list)
        func_rows = [r for r in rows if r["func"] == func]
        for r in func_rows:
            grouped[(float(r["elimination_rate"]), r["evolution_points"])].append(float(r["fopt"]))

        heat = np.full((len(elim_rates), len(evo_points)), np.nan)
        for i, er in enumerate(elim_rates):
            for j, ep in enumerate(evo_points):
                vals = grouped.get((er, ep), [])
                if vals:
                    heat[i, j] = np.mean(vals)

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(heat, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(evo_points)))
        ax.set_xticklabels([ep.replace(", ", ",\n") for ep in evo_points], fontsize=8)
        ax.set_yticks(range(len(elim_rates)))
        ax.set_yticklabels([f"{er:.2f}" for er in elim_rates])
        ax.set_xlabel("evolution_points")
        ax.set_ylabel("elimination_rate")
        ax.set_title(f"Parameter Sensitivity: {func} (10D, mean f_optimal)")

        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                if not np.isnan(heat[i, j]):
                    ax.text(j, i, f"{heat[i,j]:.1f}", ha="center", va="center", fontsize=7)

        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(input_dir / f"exp4_heatmap_{func}.png", dpi=150)
        plt.close()
        report_lines.append(f"Heatmap saved: exp4_heatmap_{func}.png")

    # Best params
    report_lines.append("\n### Best Parameters\n")
    for func in funcs:
        grouped = defaultdict(list)
        for r in rows:
            if r["func"] == func:
                grouped[(float(r["elimination_rate"]), r["evolution_points"])].append(float(r["fopt"]))
        best_key = max(grouped, key=lambda k: np.mean(grouped[k]))
        best_val = np.mean(grouped[best_key])
        report_lines.append(f"- {func}: elimination_rate={best_key[0]}, evolution_points={best_key[1]}, mean fopt={best_val:.4f}")
    report_lines.append("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze SMCO-EVO comparison results")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing experiment CSV files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    report_lines = [
        f"# SMCO-EVO vs SMCO Comparison Report",
        f"Generated: {date.today()}",
        f"Input: {input_dir}\n",
    ]

    analyze_experiment1(input_dir, report_lines)
    analyze_experiment2(input_dir, report_lines)
    analyze_experiment3(input_dir, report_lines)
    analyze_experiment4(input_dir, report_lines)

    report_path = input_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report saved to {report_path}")
    print(f"Figures saved to {input_dir}/")


if __name__ == "__main__":
    main()
