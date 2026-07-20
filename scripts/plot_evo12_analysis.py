#!/usr/bin/env python
"""Figures for Experiments 1 (start-point reduction) & 2 (dimsplit).

Quality metric = raw = -fopt (SMCO maximises config.f = -raw, so smaller raw
is closer to the optimum raw=0 and therefore BETTER).  All plots use this
orientation explicitly.

Outputs PNGs to result/figures/evo12/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SS = ROOT / "result/evo-startsweep-2026-07-18/startsweep_results.csv"
DS = ROOT / "result/evo-dimsplit-2026-07-18/dimsplit_results.csv"
OUT = ROOT / "result/figures/evo12"
OUT.mkdir(parents=True, exist_ok=True)

FRAC_ORDER = ["full", "1/2", "2/3", "3/4", "4/5"]
ALGO_COLOR = {"SMCO_EVO": "#1f77b4", "SMCO_R_EVO": "#2ca02c", "SMCO_BR_EVO": "#d62728"}
ALGO_MARK = {"SMCO_EVO": "o", "SMCO_R_EVO": "s", "SMCO_BR_EVO": "^"}
FUNC_LABEL = {"Ackley": "Ackley (single basin)", "Rastrigin": "Rastrigin (multimodal)",
              "Rosenbrock": "Rosenbrock (strong coupling)"}


def load():
    ss = pd.read_csv(SS)
    ds = pd.read_csv(DS)
    ss["raw"] = -ss["fopt"]
    ds["raw"] = -ds["fopt"]
    ds["time_h"] = ds["time"] / 3600
    return ss, ds


def fig1_startsweep_quality(ss):
    """raw vs start_fraction, per func x dim, one line per algo. Lower=better."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey="row")
    for i, func in enumerate(["Ackley", "Rastrigin", "Rosenbrock"]):
        for j, dim in enumerate([1000, 3000, 5000]):
            ax = axes[i, j]
            sub = ss[(ss.func == func) & (ss.dim == dim)]
            for algo in ["SMCO_EVO", "SMCO_R_EVO", "SMCO_BR_EVO"]:
                med = [sub[(sub.algo == algo) & (sub.start_fraction == f)]["raw"].median()
                       for f in FRAC_ORDER]
                ax.plot(FRAC_ORDER, med, marker=ALGO_MARK[algo], color=ALGO_COLOR[algo],
                        label=algo, lw=2, ms=7)
            ax.set_title(f"{func}  {dim}D", fontsize=11)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            if j == 0:
                ax.set_ylabel("raw (lower = better)\n[log]", fontsize=10)
            if i == 2:
                ax.set_xlabel("start-point fraction", fontsize=10)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle("Exp.1  start-point reduction: solution quality (raw) vs fraction\n"
                 "raw = -fopt, lower is better; full is the rightmost point",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig1_startsweep_quality.png", dpi=130)
    plt.close(fig)


def fig2_br_evo_catastrophe(ss):
    """SMCO_BR_EVO full-vs-reduced: bar chart on Rosenbrock 5000D (catastrophic full)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (func, dim) in zip(axes, [("Rosenbrock", 5000), ("Ackley", 5000)]):
        sub = ss[(ss.func == func) & (ss.dim == dim)]
        x = np.arange(len(FRAC_ORDER))
        w = 0.26
        for k, algo in enumerate(["SMCO_EVO", "SMCO_R_EVO", "SMCO_BR_EVO"]):
            med = [sub[(sub.algo == algo) & (sub.start_fraction == f)]["raw"].median()
                   for f in FRAC_ORDER]
            ax.bar(x + (k - 1) * w, med, w, label=algo, color=ALGO_COLOR[algo])
        ax.set_xticks(x)
        ax.set_xticklabels(FRAC_ORDER)
        ax.set_yscale("log")
        ax.set_title(f"{func} {dim}D", fontsize=12)
        ax.set_ylabel("raw (lower = better) [log]", fontsize=10)
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Exp.1  SMCO_BR_EVO diverges on full start points (high dim)\n"
                 "reducing the start budget rescues BR_EVO; EVO/R_EVO stay stable",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig2_br_evo_catastrophe.png", dpi=130)
    plt.close(fig)


def fig3_dimsplit_quality(ds):
    """raw vs dim_groups G, per func x strategy at 5000D. Lower=better."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharey="row")
    Gs = [1, 2, 4, 8]
    for i, func in enumerate(["Ackley", "Rastrigin", "Rosenbrock"]):
        for j, strat in enumerate(["rand1bin", "best1bin"]):
            ax = axes[i, j]
            sub = ds[(ds.func == func) & (ds.dim == 5000) & (ds.strategy == strat)]
            # aggregate over the 3 evo variants -> median per G
            med = [sub[sub.dim_groups == G]["raw"].median() for G in Gs]
            best = int(Gs[int(np.argmin(med))])
            ax.plot([str(G) for G in Gs], med, "-o", color="#9467bd", lw=2.5, ms=9)
            ax.scatter([str(best)], [min(med)], s=220, facecolors="none",
                       edgecolors="red", lw=2.5, zorder=5, label=f"best G={best}")
            # also overlay per-algo thin lines for context
            for algo, c in ALGO_COLOR.items():
                m = [sub[(sub.algo == algo) & (sub.dim_groups == G)]["raw"].median() for G in Gs]
                ax.plot([str(G) for G in Gs], m, "--", color=c, lw=1, alpha=0.5)
            ax.set_title(f"{func}  {strat}", fontsize=11)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            if j == 0:
                ax.set_ylabel("raw (lower = better)\n[log]", fontsize=10)
            if i == 2:
                ax.set_xlabel("dim_groups G  (1 = whole vector)", fontsize=10)
            ax.legend(fontsize=8, loc="best")
    fig.suptitle("Exp.2  dimsplit: solution quality (raw) vs number of groups G  @ 5000D\n"
                 "solid purple = median over evo variants; dashed = per variant; red ring = best G",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig3_dimsplit_quality.png", dpi=130)
    plt.close(fig)


def fig4_dimsplit_speedup(ds):
    """wall-clock vs G at high dim: dimsplit gets FASTER as G grows."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    Gs = [1, 2, 4, 8]
    # left: 5000D per func; right: Rosenbrock across dims
    ax = axes[0]
    for func, c in [("Ackley", "#1f77b4"), ("Rastrigin", "#ff7f0e"), ("Rosenbrock", "#2ca02c")]:
        sub = ds[(ds.func == func) & (ds.dim == 5000)]
        med = [sub[sub.dim_groups == G]["time_h"].median() for G in Gs]
        ax.plot([str(G) for G in Gs], med, "-o", color=c, lw=2.5, ms=8, label=func)
    ax.set_title("5000D: wall-clock vs G", fontsize=12)
    ax.set_ylabel("time (h, median)", fontsize=10)
    ax.set_xlabel("dim_groups G", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax = axes[1]
    for dim, c in [(1000, "#9467bd"), (3000, "#17becf"), (5000, "#d62728")]:
        sub = ds[(ds.func == "Rosenbrock") & (ds.dim == dim)]
        med = [sub[sub.dim_groups == G]["time_h"].median() for G in Gs]
        ax.plot([str(G) for G in Gs], med, "-o", color=c, lw=2.5, ms=8, label=f"Rosenbrock {dim}D")
    ax.set_title("Rosenbrock: wall-clock vs G across dims", fontsize=12)
    ax.set_ylabel("time (h, median)", fontsize=10)
    ax.set_xlabel("dim_groups G", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.suptitle("Exp.2  dimsplit speed-up at high dimension\n"
                 "larger G -> smaller per-group search subspace -> faster convergence",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig4_dimsplit_speedup.png", dpi=130)
    plt.close(fig)


def fig5_bestG_distribution(ds):
    """How often each G is the best, across all func x dim x strategy cells."""
    Gs = [1, 2, 4, 8]
    counts = {G: 0 for G in Gs}
    for (func, dim, strat), g in ds.groupby(["func", "dim", "strategy"]):
        med = {G: g[g.dim_groups == G]["raw"].median() for G in Gs}
        best = min(med, key=med.get)
        counts[best] += 1
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([f"G={G}" for G in Gs], [counts[G] for G in Gs],
                  color=["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"])
    for b, G in zip(bars, Gs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                str(counts[G]), ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("# cells where G is best  (of 18)", fontsize=11)
    ax.set_title("Exp.2  which group count G wins most often\n"
                 "(quality median across 3 funcs x 3 dims x 2 strategies)", fontsize=12)
    ax.set_ylim(0, max(counts.values()) + 2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_bestG_distribution.png", dpi=130)
    plt.close(fig)


def main():
    ss, ds = load()
    fig1_startsweep_quality(ss)
    fig2_br_evo_catastrophe(ss)
    fig3_dimsplit_quality(ds)
    fig4_dimsplit_speedup(ds)
    fig5_bestG_distribution(ds)
    print(f"wrote 5 figures to {OUT}")


if __name__ == "__main__":
    main()
