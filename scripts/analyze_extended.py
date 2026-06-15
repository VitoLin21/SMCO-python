#!/usr/bin/env python3
"""Extended deep analysis of SMCO-EVO results.

Additional analyses beyond the base comprehensive report:
  A. 4-strategy full comparison (heatmaps, rankings, interaction)
  B. Best strategy selection per context
  C. Function characteristics × EVO interaction
  D. Convergence quality gradient (low→high dim)
  E. SMCO variant deep-dive (which base benefits most from EVO)
  F. Global ranking across all configs
  G. Statistical significance landscape
  H. Scalability index (performance degradation rate)
  I. Pareto frontier analysis
  J. Strategy robustness (consistency across configs)

Usage:
  python scripts/analyze_extended.py
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from datetime import date

import numpy as np
from scipy.stats import wilcoxon, spearmanr, friedmanchisquare, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_fm.fontManager.addfont(_CJK_FONT_PATH)
_CJK_FP = _fm.FontProperties(fname=_CJK_FONT_PATH)
plt.rcParams["font.sans-serif"] = [_CJK_FP.get_name(), "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PAIRS = [("SMCO", "SMCO_EVO"), ("SMCO_R", "SMCO_R_EVO"), ("SMCO_BR", "SMCO_BR_EVO")]
EVO_VARIANTS = ["SMCO_EVO", "SMCO_R_EVO", "SMCO_BR_EVO"]
BASE_VARIANTS = ["SMCO", "SMCO_R", "SMCO_BR"]
SMCO_ALL = BASE_VARIANTS + EVO_VARIANTS
GLOBAL_BASELINES = ["GenSA", "SA", "DEoptim", "GA", "PSO"]
LOCAL_BASELINES = ["GD", "SignGD", "ADAM", "SPSA", "optimLBFGS", "BOBYQA", "optimNM"]
ALL_METHODS = SMCO_ALL + GLOBAL_BASELINES + LOCAL_BASELINES
STRATEGIES = ["rand1bin", "current-to-best1bin", "best1bin", "sobol"]
FUNCS = ["Rastrigin", "Ackley", "Rosenbrock", "Griewank", "Zakharov"]
ALL_DIMS = [50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]
HD_DIMS = [1000, 2000, 3000, 4000, 5000]

EVO_COLORS = {"SMCO_EVO": "#1f77b4", "SMCO_R_EVO": "#ff7f0e", "SMCO_BR_EVO": "#2ca02c"}
STRAT_COLORS = {
    "rand1bin": "#1f77b4", "current-to-best1bin": "#ff7f0e",
    "best1bin": "#2ca02c", "sobol": "#d62728",
}
STRAT_MARKERS = {
    "rand1bin": "o", "current-to-best1bin": "s",
    "best1bin": "^", "sobol": "D",
}


def _load(p):
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _group(rows, *keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def _save_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {path.name} ({len(rows)} rows)")


def _save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# A. Four-strategy comparison
# ═══════════════════════════════════════════════════════════════════════════════
def analysis_A(rows, out):
    """Compare all 4 strategies head-to-head for each EVO variant."""
    detail = []
    for evo_algo in EVO_VARIANTS:
        for func in FUNCS:
            for dim in ALL_DIMS:
                means = {}
                for strat in STRATEGIES:
                    vals = [float(r["fopt"]) for r in rows
                            if r["strategy"] == strat and r["algo"] == evo_algo
                            and r["func"] == func and int(r["dim"]) == dim]
                    if vals:
                        means[strat] = np.mean(vals)
                if len(means) >= 2:
                    ranked = sorted(means.items(), key=lambda x: -x[1])
                    for rank, (s, v) in enumerate(ranked, 1):
                        detail.append({
                            "evo_algo": evo_algo, "func": func, "dim": dim,
                            "strategy": s, "fopt_mean": round(v, 8),
                            "rank_among_strategies": rank,
                        })
    _save_csv(detail, out / "ext01_strategy_comparison_detail.csv")

    # Win count per strategy
    wins = defaultdict(int)
    for r in detail:
        if r["rank_among_strategies"] == 1:
            wins[(r["evo_algo"], r["strategy"])] += 1
    summary = []
    for evo_algo in EVO_VARIANTS:
        for strat in STRATEGIES:
            w = wins.get((evo_algo, strat), 0)
            summary.append({"evo_algo": evo_algo, "strategy": strat, "wins": w})
    _save_csv(summary, out / "ext01b_strategy_wins.csv")

    # Overall strategy ranking
    overall = defaultdict(int)
    for r in detail:
        overall[r["strategy"]] += r["rank_among_strategies"]
    n_configs = len(set((r["evo_algo"], r["func"], r["dim"]) for r in detail))
    strat_summary = []
    for s in STRATEGIES:
        total_rank_sum = overall[s]
        strat_summary.append({
            "strategy": s,
            "total_rank_sum": total_rank_sum,
            "avg_rank": round(total_rank_sum / n_configs, 3) if n_configs else 0,
            "wins": sum(w for (ea, st), w in wins.items() if st == s),
        })
    _save_csv(strat_summary, out / "ext01c_strategy_overall.csv")
    return detail


# ═══════════════════════════════════════════════════════════════════════════════
# B. Best strategy selection per context
# ═══════════════════════════════════════════════════════════════════════════════
def analysis_B(rows, out):
    """For each (func, dim), which strategy is best for each EVO variant?"""
    result = []
    for evo_algo in EVO_VARIANTS:
        for func in FUNCS:
            for dim in ALL_DIMS:
                strat_vals = {}
                for strat in STRATEGIES:
                    vals = [float(r["fopt"]) for r in rows
                            if r["strategy"] == strat and r["algo"] == evo_algo
                            and r["func"] == func and int(r["dim"]) == dim]
                    if vals:
                        strat_vals[strat] = (np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0)
                if strat_vals:
                    best = max(strat_vals, key=lambda s: strat_vals[s][0])
                    worst = min(strat_vals, key=lambda s: strat_vals[s][0])
                    gap = (strat_vals[best][0] - strat_vals[worst][0]) / abs(strat_vals[worst][0]) * 100 if strat_vals[worst][0] != 0 else 0
                    result.append({
                        "evo_algo": evo_algo, "func": func, "dim": dim,
                        "best_strategy": best,
                        "best_fopt": round(strat_vals[best][0], 6),
                        "worst_strategy": worst,
                        "worst_fopt": round(strat_vals[worst][0], 6),
                        "strategy_gap_pct": round(gap, 4),
                    })
    _save_csv(result, out / "ext02_best_strategy_per_context.csv")

    # Aggregate: per func, per dim range
    agg = []
    for evo_algo in EVO_VARIANTS:
        for func in FUNCS:
            subset = [r for r in result if r["evo_algo"] == evo_algo and r["func"] == func]
            if subset:
                strat_counts = defaultdict(int)
                for r in subset:
                    strat_counts[r["best_strategy"]] += 1
                best_overall = max(strat_counts, key=strat_counts.get)
                avg_gap = np.mean([r["strategy_gap_pct"] for r in subset])
                agg.append({
                    "evo_algo": evo_algo, "func": func,
                    "best_strategy": best_overall,
                    "wins": strat_counts[best_overall],
                    "total_dims": len(subset),
                    "avg_gap_pct": round(avg_gap, 4),
                })
    _save_csv(agg, out / "ext02b_best_strategy_by_func.csv")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# C. Function characteristics × EVO interaction
# ═══════════════════════════════════════════════════════════════════════════════
def analysis_C(rows, out):
    """How does EVO improvement vary by function type?"""
    result = []
    for strat in STRATEGIES:
        srows = [r for r in rows if r["strategy"] == strat]
        for func in FUNCS:
            for base, evo in PAIRS:
                imps, wrs = [], []
                for dim in ALL_DIMS:
                    bv = [float(r["fopt"]) for r in srows if r["algo"] == base and r["func"] == func and int(r["dim"]) == dim]
                    ev = [float(r["fopt"]) for r in srows if r["algo"] == evo and r["func"] == func and int(r["dim"]) == dim]
                    if bv and ev:
                        n = min(len(bv), len(ev))
                        bm, em = np.mean(bv[:n]), np.mean(ev[:n])
                        if bm != 0:
                            imps.append((em - bm) / abs(bm) * 100)
                        wrs.append(sum(1 for i in range(n) if ev[i] > bv[i]) / n * 100)
                if imps:
                    result.append({
                        "strategy": strat, "func": func, "pair": f"{base}_vs_{evo}",
                        "mean_improve": round(np.mean(imps), 4),
                        "max_improve": round(max(imps), 4),
                        "min_improve": round(min(imps), 4),
                        "std_improve": round(float(np.std(imps, ddof=1)), 4) if len(imps) > 1 else 0,
                        "mean_win_rate": round(np.mean(wrs), 1),
                        "positive_rate": round(sum(1 for x in imps if x > 0) / len(imps) * 100, 1),
                    })
    _save_csv(result, out / "ext03_func_interaction.csv")

    # Per-function aggregate (across all strategies)
    func_agg = []
    for func in FUNCS:
        subset = [r for r in result if r["func"] == func]
        if subset:
            func_agg.append({
                "func": func,
                "avg_improve": round(np.mean([r["mean_improve"] for r in subset]), 4),
                "avg_win_rate": round(np.mean([r["mean_win_rate"] for r in subset]), 1),
                "avg_positive_rate": round(np.mean([r["positive_rate"] for r in subset]), 1),
                "best_pair": max(subset, key=lambda r: r["mean_improve"])["pair"],
            })
    _save_csv(func_agg, out / "ext03b_func_summary.csv")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# D. Convergence quality gradient
# ═══════════════════════════════════════════════════════════════════════════════
def analysis_D(rows, out):
    """How does solution quality degrade as dimension increases?"""
    result = []
    for strat in STRATEGIES:
        srows = [r for r in rows if r["strategy"] == strat]
        for func in FUNCS:
            for algo in SMCO_ALL + GLOBAL_BASELINES:
                for dim in ALL_DIMS:
                    vals = [float(r["fopt"]) for r in srows
                            if r["algo"] == algo and r["func"] == func and int(r["dim"]) == dim]
                    if vals:
                        result.append({
                            "strategy": strat, "func": func, "algo": algo,
                            "dim": dim, "fopt_mean": round(np.mean(vals), 8),
                            "fopt_std": round(float(np.std(vals, ddof=1)), 8) if len(vals) > 1 else 0,
                        })
    _save_csv(result, out / "ext04_quality_gradient.csv")

    # Degradation rate: how fast does fopt drop from 50D to 5000D?
    degradation = []
    for strat in STRATEGIES:
        for func in FUNCS:
            for algo in SMCO_ALL + GLOBAL_BASELINES:
                subset = [r for r in result if r["strategy"] == strat and r["func"] == func and r["algo"] == algo]
                if len(subset) >= 3:
                    dims = np.array([r["dim"] for r in subset])
                    fopts = np.array([r["fopt_mean"] for r in subset])
                    if len(dims) >= 3:
                        rho, p = spearmanr(dims, fopts)
                        degradation.append({
                            "strategy": strat, "func": func, "algo": algo,
                            "spearman_rho": round(float(rho), 4),
                            "spearman_p": round(float(p), 6),
                            "correlation": "increasing" if rho > 0 else "decreasing",
                            "significant": int(p < 0.05),
                            "n_dims": len(dims),
                            "fopt_50D": next((r["fopt_mean"] for r in subset if r["dim"] == 50), None),
                            "fopt_maxD": next((r["fopt_mean"] for r in subset if r["dim"] == max(dims)), None),
                        })
    _save_csv(degradation, out / "ext04b_degradation_rate.csv")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# E. SMCO variant deep-dive
# ═══════════════════════════════════════════════════════════════════════════════
def analysis_E(rows, out):
    """Detailed analysis of which SMCO base variant benefits most from EVO."""
    result = []
    for strat in STRATEGIES:
        srows = [r for r in rows if r["strategy"] == strat]
        for base, evo in PAIRS:
            for func in FUNCS:
                for dim in ALL_DIMS:
                    bv = [float(r["fopt"]) for r in srows if r["algo"] == base and r["func"] == func and int(r["dim"]) == dim]
                    ev = [float(r["fopt"]) for r in srows if r["algo"] == evo and r["func"] == func and int(r["dim"]) == dim]
                    bt = [float(r["time"]) for r in srows if r["algo"] == base and r["func"] == func and int(r["dim"]) == dim]
                    et = [float(r["time"]) for r in srows if r["algo"] == evo and r["func"] == func and int(r["dim"]) == dim]
                    if bv and ev:
                        n = min(len(bv), len(ev))
                        bm, em = np.mean(bv[:n]), np.mean(ev[:n])
                        imp = (em - bm) / abs(bm) * 100 if bm != 0 else 0
                        time_overhead = 0
                        if bt and et:
                            time_overhead = (np.mean(et) - np.mean(bt)) / np.mean(bt) * 100 if np.mean(bt) > 0 else 0
                        result.append({
                            "strategy": strat, "pair": f"{base}_vs_{evo}",
                            "func": func, "dim": dim,
                            "improve_pct": round(imp, 4),
                            "time_overhead_pct": round(time_overhead, 2),
                            "cost_benefit": round(imp / time_overhead, 4) if time_overhead > 0.01 else float("inf"),
                        })
    _save_csv(result, out / "ext05_variant_deep_dive.csv")

    # Which variant benefits most: aggregate by base variant
    variant_benefit = []
    for strat in STRATEGIES:
        for base, evo in PAIRS:
            subset = [r for r in result if r["strategy"] == strat and r["pair"] == f"{base}_vs_{evo}"]
            if subset:
                imps = [r["improve_pct"] for r in subset]
                variant_benefit.append({
                    "strategy": strat, "base_variant": base, "evo_variant": evo,
                    "mean_improve": round(np.mean(imps), 4),
                    "median_improve": round(float(np.median(imps)), 4),
                    "std_improve": round(float(np.std(imps, ddof=1)), 4) if len(imps) > 1 else 0,
                    "positive_rate": round(sum(1 for x in imps if x > 0) / len(imps) * 100, 1),
                    "max_improve": round(max(imps), 4),
                    "avg_time_overhead": round(np.mean([r["time_overhead_pct"] for r in subset]), 2),
                })
    _save_csv(variant_benefit, out / "ext05b_variant_benefit_summary.csv")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# F. Global ranking across all configs
# ═══════════════════════════════════════════════════════════════════════════════
def analysis_F(rows, out):
    """Friedman test + global ranking across all (func, dim, strategy) configs."""
    all_configs = []
    for strat in STRATEGIES:
        srows = [r for r in rows if r["strategy"] == strat]
        grouped = _group(srows, "func", "dim")
        for (func, dim), grp in sorted(grouped.items()):
            config_id = f"{strat}_{func}_{dim}"
            algo_means = {}
            for algo in set(r["algo"] for r in grp):
                fv = [float(r["fopt"]) for r in grp if r["algo"] == algo]
                if fv:
                    algo_means[algo] = np.mean(fv)
            ranked = sorted(algo_means.items(), key=lambda x: -x[1])
            for rank, (algo, val) in enumerate(ranked, 1):
                all_configs.append({
                    "config": config_id, "strategy": strat,
                    "func": func, "dim": int(dim),
                    "algo": algo, "fopt_mean": val, "rank": rank,
                })
    _save_csv(all_configs, out / "ext06_global_rankings.csv")

    # Average rank across ALL configs
    rank_data = defaultdict(list)
    for r in all_configs:
        rank_data[r["algo"]].append(r["rank"])

    summary = []
    for algo in sorted(rank_data, key=lambda a: np.mean(rank_data[a])):
        rks = rank_data[algo]
        summary.append({
            "algo": algo,
            "avg_rank": round(np.mean(rks), 3),
            "median_rank": float(np.median(rks)),
            "std_rank": round(float(np.std(rks, ddof=1)), 3) if len(rks) > 1 else 0,
            "best_rank": int(np.min(rks)),
            "worst_rank": int(np.max(rks)),
            "n_configs": len(rks),
            "is_evo": algo in EVO_VARIANTS,
        })
    _save_csv(summary, out / "ext06b_global_ranking_summary.csv")

    # Friedman test per dimension range
    friedman_results = []
    for dim_range_name, dim_list in [("all", ALL_DIMS), ("low", [50, 100, 200]), ("mid", [500, 1000, 2000]), ("high", [3000, 4000, 5000])]:
        subset_configs = [r for r in all_configs if r["dim"] in dim_list]
        # Build matrix: rows=configs, cols=methods
        algo_set = sorted(set(r["algo"] for r in subset_configs))
        config_set = sorted(set(r["config"] for r in subset_configs))
        if len(algo_set) >= 3 and len(config_set) >= 3:
            matrix = []
            for cfg in config_set:
                row = []
                for algo in algo_set:
                    ranks = [r["rank"] for r in subset_configs if r["config"] == cfg and r["algo"] == algo]
                    row.append(ranks[0] if ranks else 0)
                matrix.append(row)
            try:
                stat, p = friedmanchisquare(*[np.array(matrix)[:, i] for i in range(len(algo_set))])
                friedman_results.append({
                    "dim_range": dim_range_name,
                    "n_methods": len(algo_set),
                    "n_configs": len(config_set),
                    "friedman_stat": round(float(stat), 2),
                    "friedman_p": float(p),
                    "significant": int(p < 0.05),
                })
            except Exception:
                pass
    _save_csv(friedman_results, out / "ext06c_friedman_test.csv")
    return all_configs


# ═══════════════════════════════════════════════════════════════════════════════
# G. Statistical significance landscape
# ═══════════════════════════════════════════════════════════════════════════════
def analysis_G(rows, out):
    """Wilcoxon test for EVO vs base: p-value landscape."""
    result = []
    for strat in STRATEGIES:
        srows = [r for r in rows if r["strategy"] == strat]
        for base, evo in PAIRS:
            grouped = _group(srows, "func", "dim")
            for (func, dim), grp in sorted(grouped.items()):
                bv = [float(r["fopt"]) for r in grp if r["algo"] == base]
                ev = [float(r["fopt"]) for r in grp if r["algo"] == evo]
                if bv and ev:
                    n = min(len(bv), len(ev))
                    bv, ev = bv[:n], ev[:n]
                    bm, em = np.mean(bv), np.mean(ev)
                    imp = (em - bm) / abs(bm) * 100 if bm != 0 else 0
                    pval = float("nan")
                    if n >= 3 and np.any(np.array(ev) != np.array(bv)):
                        try:
                            _, pval = wilcoxon(bv, ev, alternative="two-sided")
                        except ValueError:
                            pass
                    # Effect size: Cohen's d
                    pooled_std = math.sqrt((np.var(bv, ddof=1) + np.var(ev, ddof=1)) / 2) if n > 1 else 1
                    cohens_d = (em - bm) / pooled_std if pooled_std > 0 else 0
                    result.append({
                        "strategy": strat, "pair": f"{base}_vs_{evo}",
                        "func": func, "dim": int(dim),
                        "n": n, "improve_pct": round(imp, 4),
                        "wilcoxon_p": round(float(pval), 6) if not math.isnan(pval) else "NaN",
                        "significant_005": int(pval < 0.05) if not math.isnan(pval) else 0,
                        "significant_001": int(pval < 0.01) if not math.isnan(pval) else 0,
                        "cohens_d": round(cohens_d, 4),
                        "effect_size": "large" if abs(cohens_d) >= 0.8 else ("medium" if abs(cohens_d) >= 0.5 else "small"),
                    })
    _save_csv(result, out / "ext07_significance_landscape.csv")

    # Aggregate significance counts
    sig_agg = []
    for strat in STRATEGIES:
        for base, evo in PAIRS:
            subset = [r for r in result if r["strategy"] == strat and r["pair"] == f"{base}_vs_{evo}"]
            if subset:
                sig_agg.append({
                    "strategy": strat, "pair": f"{base}_vs_{evo}",
                    "total_configs": len(subset),
                    "sig_005": sum(r["significant_005"] for r in subset),
                    "sig_001": sum(r["significant_001"] for r in subset),
                    "large_effect": sum(1 for r in subset if r["effect_size"] == "large"),
                    "medium_effect": sum(1 for r in subset if r["effect_size"] == "medium"),
                    "small_effect": sum(1 for r in subset if r["effect_size"] == "small"),
                    "avg_cohens_d": round(np.mean([abs(r["cohens_d"]) for r in subset]), 4),
                })
    _save_csv(sig_agg, out / "ext07b_significance_summary.csv")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# H. Scalability index
# ═══════════════════════════════════════════════════════════════════════════════
def analysis_H(rows, out):
    """How fast does each method's performance degrade with increasing dim?"""
    result = []
    for strat in STRATEGIES:
        srows = [r for r in rows if r["strategy"] == strat]
        for func in FUNCS:
            for algo in SMCO_ALL + GLOBAL_BASELINES:
                dim_fopt = {}
                for dim in ALL_DIMS:
                    vals = [float(r["fopt"]) for r in srows if r["algo"] == algo and r["func"] == func and int(r["dim"]) == dim]
                    if vals:
                        dim_fopt[dim] = np.mean(vals)
                if len(dim_fopt) >= 3:
                    dims_arr = np.array(sorted(dim_fopt.keys()))
                    fopts_arr = np.array([dim_fopt[d] for d in dims_arr])
                    rho, p = spearmanr(dims_arr, fopts_arr)
                    # Degradation rate: % drop from 50D to max dim
                    base_dim = min(dim_fopt.keys())
                    max_dim = max(dim_fopt.keys())
                    degradation_pct = (dim_fopt[base_dim] - dim_fopt[max_dim]) / abs(dim_fopt[base_dim]) * 100 if dim_fopt[base_dim] != 0 else 0
                    result.append({
                        "strategy": strat, "func": func, "algo": algo,
                        "spearman_rho": round(float(rho), 4),
                        "degradation_pct": round(degradation_pct, 2),
                        "fopt_at_base_dim": round(dim_fopt[base_dim], 6),
                        "fopt_at_max_dim": round(dim_fopt[max_dim], 6),
                        "base_dim": base_dim,
                        "max_dim": max_dim,
                        "n_dims": len(dim_fopt),
                        "scalability": "good" if rho >= -0.3 else ("moderate" if rho >= -0.7 else "poor"),
                    })
    _save_csv(result, out / "ext08_scalability_index.csv")

    # Per-algo average scalability
    algo_scalability = []
    for algo in SMCO_ALL + GLOBAL_BASELINES:
        subset = [r for r in result if r["algo"] == algo]
        if subset:
            algo_scalability.append({
                "algo": algo,
                "avg_spearman_rho": round(np.mean([r["spearman_rho"] for r in subset]), 4),
                "avg_degradation_pct": round(np.mean([r["degradation_pct"] for r in subset]), 2),
                "good_scalability_count": sum(1 for r in subset if r["scalability"] == "good"),
                "poor_scalability_count": sum(1 for r in subset if r["scalability"] == "poor"),
                "is_evo": algo in EVO_VARIANTS,
            })
    _save_csv(algo_scalability, out / "ext08b_scalability_by_algo.csv")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# I. Pareto frontier (quality vs time)
# ═══════════════════════════════════════════════════════════════════════════════
def analysis_I(rows, out):
    """Identify Pareto-optimal methods (quality vs time tradeoff)."""
    result = []
    for strat in STRATEGIES:
        srows = [r for r in rows if r["strategy"] == strat]
        grouped = _group(srows, "func", "dim")
        for (func, dim), grp in sorted(grouped.items()):
            algo_stats = {}
            for algo in set(r["algo"] for r in grp):
                fv = [float(r["fopt"]) for r in grp if r["algo"] == algo]
                tv = [float(r["time"]) for r in grp if r["algo"] == algo]
                if fv and tv:
                    algo_stats[algo] = {"fopt": np.mean(fv), "time": np.mean(tv)}

            # Find Pareto frontier (maximize fopt, minimize time)
            algos = list(algo_stats.keys())
            pareto = set()
            for a in algos:
                dominated = False
                for b in algos:
                    if a == b:
                        continue
                    # b dominates a if b has better fopt AND less time
                    if (algo_stats[b]["fopt"] >= algo_stats[a]["fopt"] and
                        algo_stats[b]["time"] <= algo_stats[a]["time"] and
                        (algo_stats[b]["fopt"] > algo_stats[a]["fopt"] or
                         algo_stats[b]["time"] < algo_stats[a]["time"])):
                        dominated = True
                        break
                if not dominated:
                    pareto.add(a)

            for algo, stats in algo_stats.items():
                result.append({
                    "strategy": strat, "func": func, "dim": int(dim),
                    "algo": algo,
                    "fopt_mean": round(stats["fopt"], 8),
                    "time_mean": round(stats["time"], 2),
                    "is_pareto": int(algo in pareto),
                    "is_evo": algo in EVO_VARIANTS,
                })
    _save_csv(result, out / "ext09_pareto_frontier.csv")

    # Pareto count per method
    pareto_count = defaultdict(lambda: {"pareto": 0, "total": 0})
    for r in result:
        pareto_count[r["algo"]]["total"] += 1
        if r["is_pareto"]:
            pareto_count[r["algo"]]["pareto"] += 1
    summary = []
    for algo in sorted(pareto_count, key=lambda a: -pareto_count[a]["pareto"]):
        summary.append({
            "algo": algo,
            "pareto_count": pareto_count[algo]["pareto"],
            "total_configs": pareto_count[algo]["total"],
            "pareto_rate": round(pareto_count[algo]["pareto"] / pareto_count[algo]["total"] * 100, 1),
            "is_evo": algo in EVO_VARIANTS,
        })
    _save_csv(summary, out / "ext09b_pareto_summary.csv")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# J. Strategy robustness
# ═══════════════════════════════════════════════════════════════════════════════
def analysis_J(rows, out):
    """How consistent is each strategy's EVO improvement across configs?"""
    result = []
    for strat in STRATEGIES:
        srows = [r for r in rows if r["strategy"] == strat]
        for base, evo in PAIRS:
            grouped = _group(srows, "func", "dim")
            imps = []
            for (func, dim), grp in grouped.items():
                bv = [float(r["fopt"]) for r in grp if r["algo"] == base]
                ev = [float(r["fopt"]) for r in grp if r["algo"] == evo]
                if bv and ev:
                    n = min(len(bv), len(ev))
                    bm, em = np.mean(bv[:n]), np.mean(ev[:n])
                    if bm != 0:
                        imps.append((em - bm) / abs(bm) * 100)
            if imps:
                result.append({
                    "strategy": strat, "pair": f"{base}_vs_{evo}",
                    "mean_improve": round(np.mean(imps), 4),
                    "std_improve": round(float(np.std(imps, ddof=1)), 4) if len(imps) > 1 else 0,
                    "cv": round(float(np.std(imps, ddof=1) / abs(np.mean(imps))) * 100, 1) if np.mean(imps) != 0 and len(imps) > 1 else float("inf"),
                    "positive_rate": round(sum(1 for x in imps if x > 0) / len(imps) * 100, 1),
                    "min_improve": round(min(imps), 4),
                    "max_improve": round(max(imps), 4),
                    "iqr": round(float(np.percentile(imps, 75) - np.percentile(imps, 25)), 4),
                    "robustness": "high" if sum(1 for x in imps if x > 0) / len(imps) > 0.8 else ("medium" if sum(1 for x in imps if x > 0) / len(imps) > 0.6 else "low"),
                })
    _save_csv(result, out / "ext10_strategy_robustness.csv")

    # Overall strategy robustness score
    strat_robust = []
    for strat in STRATEGIES:
        subset = [r for r in result if r["strategy"] == strat]
        if subset:
            strat_robust.append({
                "strategy": strat,
                "avg_positive_rate": round(np.mean([r["positive_rate"] for r in subset]), 1),
                "avg_mean_improve": round(np.mean([r["mean_improve"] for r in subset]), 4),
                "avg_cv": round(np.mean([r["cv"] for r in subset if r["cv"] != float("inf")]), 1),
                "high_robustness_count": sum(1 for r in subset if r["robustness"] == "high"),
            })
    _save_csv(strat_robust, out / "ext10b_strategy_robustness_summary.csv")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# K. EVO vs all non-SMCO methods
# ═══════════════════════════════════════════════════════════════════════════════
def analysis_K(rows, out):
    """EVO variants vs all baselines (not just within SMCO family)."""
    all_baselines = GLOBAL_BASELINES + LOCAL_BASELINES
    result = []
    for strat in STRATEGIES:
        srows = [r for r in rows if r["strategy"] == strat]
        grouped = _group(srows, "func", "dim")
        for evo_algo in EVO_VARIANTS:
            for baseline in all_baselines:
                ew, bw, total = 0, 0, 0
                for (func, dim), grp in grouped.items():
                    ev = [float(r["fopt"]) for r in grp if r["algo"] == evo_algo]
                    bv = [float(r["fopt"]) for r in grp if r["algo"] == baseline]
                    if ev and bv:
                        total += 1
                        if np.mean(ev) > np.mean(bv):
                            ew += 1
                        elif np.mean(bv) > np.mean(ev):
                            bw += 1
                if total > 0:
                    result.append({
                        "strategy": strat, "evo_algo": evo_algo,
                        "baseline": baseline,
                        "evo_wins": ew, "baseline_wins": bw,
                        "ties": total - ew - bw, "total": total,
                        "evo_win_rate": round(ew / total * 100, 1),
                    })
    _save_csv(result, out / "ext11_evo_vs_all_baselines.csv")

    # Summary per EVO variant (aggregated across strategies)
    evo_summary = []
    for evo_algo in EVO_VARIANTS:
        for baseline in all_baselines:
            subset = [r for r in result if r["evo_algo"] == evo_algo and r["baseline"] == baseline]
            if subset:
                evo_summary.append({
                    "evo_algo": evo_algo, "baseline": baseline,
                    "avg_win_rate": round(np.mean([r["evo_win_rate"] for r in subset]), 1),
                    "min_win_rate": round(min(r["evo_win_rate"] for r in subset), 1),
                    "max_win_rate": round(max(r["evo_win_rate"] for r in subset), 1),
                })
    _save_csv(evo_summary, out / "ext11b_evo_vs_baselines_summary.csv")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# L. Dimension-stratified analysis
# ═══════════════════════════════════════════════════════════════════════════════
def analysis_L(rows, out):
    """How does EVO effectiveness change across dimension ranges?"""
    dim_ranges = {
        "low_50_200": [50, 100, 200],
        "mid_500_1000": [500, 1000],
        "high_2000_5000": [2000, 3000, 4000, 5000],
    }
    result = []
    for strat in STRATEGIES:
        srows = [r for r in rows if r["strategy"] == strat]
        for range_name, dims in dim_ranges.items():
            for base, evo in PAIRS:
                imps, wrs = [], []
                for func in FUNCS:
                    for dim in dims:
                        bv = [float(r["fopt"]) for r in srows if r["algo"] == base and r["func"] == func and int(r["dim"]) == dim]
                        ev = [float(r["fopt"]) for r in srows if r["algo"] == evo and r["func"] == func and int(r["dim"]) == dim]
                        if bv and ev:
                            n = min(len(bv), len(ev))
                            bm, em = np.mean(bv[:n]), np.mean(ev[:n])
                            if bm != 0:
                                imps.append((em - bm) / abs(bm) * 100)
                            wrs.append(sum(1 for i in range(n) if ev[i] > bv[i]) / n * 100)
                if imps:
                    result.append({
                        "strategy": strat, "dim_range": range_name,
                        "pair": f"{base}_vs_{evo}",
                        "n_configs": len(imps),
                        "mean_improve": round(np.mean(imps), 4),
                        "median_improve": round(float(np.median(imps)), 4),
                        "std_improve": round(float(np.std(imps, ddof=1)), 4) if len(imps) > 1 else 0,
                        "mean_win_rate": round(np.mean(wrs), 1),
                        "positive_rate": round(sum(1 for x in imps if x > 0) / len(imps) * 100, 1),
                    })
    _save_csv(result, out / "ext12_dim_stratified.csv")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# M. EVO absolute contribution
# ═══════════════════════════════════════════════════════════════════════════════
def analysis_M(rows, out):
    """What absolute fopt value does EVO add compared to base?"""
    result = []
    for strat in STRATEGIES:
        srows = [r for r in rows if r["strategy"] == strat]
        for base, evo in PAIRS:
            grouped = _group(srows, "func", "dim")
            for (func, dim), grp in sorted(grouped.items()):
                bv = [float(r["fopt"]) for r in grp if r["algo"] == base]
                ev = [float(r["fopt"]) for r in grp if r["algo"] == evo]
                if bv and ev:
                    n = min(len(bv), len(ev))
                    bm, em = np.mean(bv[:n]), np.mean(ev[:n])
                    abs_diff = em - bm
                    result.append({
                        "strategy": strat, "pair": f"{base}_vs_{evo}",
                        "func": func, "dim": int(dim),
                        "base_fopt": round(bm, 8),
                        "evo_fopt": round(em, 8),
                        "absolute_diff": round(abs_diff, 8),
                        "pct_diff": round(abs_diff / abs(bm) * 100, 4) if bm != 0 else 0,
                    })
    _save_csv(result, out / "ext13_absolute_contribution.csv")

    # Aggregate by dim
    dim_agg = []
    for dim in ALL_DIMS:
        for base, evo in PAIRS:
            subset = [r for r in result if r["dim"] == dim and r["pair"] == f"{base}_vs_{evo}"]
            if subset:
                dim_agg.append({
                    "dim": dim, "pair": f"{base}_vs_{evo}",
                    "avg_abs_diff": round(np.mean([r["absolute_diff"] for r in subset]), 6),
                    "avg_pct_diff": round(np.mean([r["pct_diff"] for r in subset]), 4),
                })
    _save_csv(dim_agg, out / "ext13b_contribution_by_dim.csv")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_all(rows, out):
    # ---- P1: Strategy comparison heatmap ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for idx, evo_algo in enumerate(EVO_VARIANTS):
        ax = axes[idx]
        strat_means = {}
        for strat in STRATEGIES:
            srows = [r for r in rows if r["strategy"] == strat and r["algo"] == evo_algo]
            grouped = _group(srows, "func", "dim")
            vals = []
            for (func, dim), grp in grouped.items():
                fv = [float(r["fopt"]) for r in grp]
                if fv:
                    vals.append(np.mean(fv))
            strat_means[strat] = np.mean(vals) if vals else 0

        algos = list(strat_means.keys())
        vals = [strat_means[a] for a in algos]
        colors = [STRAT_COLORS.get(a, "#888") for a in algos]
        bars = ax.bar(range(len(algos)), vals, color=colors)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("平均 fopt")
        ax.set_title(evo_algo)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7)
    fig.suptitle("各策略 EVO 变体平均目标函数值", fontsize=13)
    plt.tight_layout()
    _save_fig(fig, out / "ext_plot01_strategy_comparison.png")

    # ---- P2: Strategy win counts ----
    fig, ax = plt.subplots(figsize=(10, 5))
    win_data = defaultdict(lambda: defaultdict(int))
    for evo_algo in EVO_VARIANTS:
        for func in FUNCS:
            for dim in ALL_DIMS:
                vals = {}
                for strat in STRATEGIES:
                    v = [float(r["fopt"]) for r in rows
                         if r["strategy"] == strat and r["algo"] == evo_algo
                         and r["func"] == func and int(r["dim"]) == dim]
                    if v:
                        vals[strat] = np.mean(v)
                if vals:
                    best = max(vals, key=vals.get)
                    win_data[evo_algo][best] += 1

    x = np.arange(len(STRATEGIES))
    width = 0.25
    for i, evo_algo in enumerate(EVO_VARIANTS):
        counts = [win_data[evo_algo].get(s, 0) for s in STRATEGIES]
        ax.bar(x + i * width, counts, width, label=evo_algo, color=EVO_COLORS[evo_algo], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(STRATEGIES)
    ax.set_ylabel("胜出次数")
    ax.set_title("各策略在不同 EVO 变体中的胜出次数")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(fig, out / "ext_plot02_strategy_wins.png")

    # ---- P3: Function × EVO improvement (grouped bar) ----
    fig, ax = plt.subplots(figsize=(12, 5))
    func_imps = defaultdict(list)
    for strat in STRATEGIES:
        srows = [r for r in rows if r["strategy"] == strat]
        for func in FUNCS:
            for base, evo in PAIRS:
                grouped = _group(srows, "func", "dim")
                for (f2, dim), grp in grouped.items():
                    if f2 != func:
                        continue
                    bv = [float(r2["fopt"]) for r2 in grp if r2["algo"] == base]
                    ev = [float(r2["fopt"]) for r2 in grp if r2["algo"] == evo]
                    if bv and ev:
                        n = min(len(bv), len(ev))
                        bm, em = np.mean(bv[:n]), np.mean(ev[:n])
                        if bm != 0:
                            func_imps[(func, evo)].append((em - bm) / abs(bm) * 100)

    x = np.arange(len(FUNCS))
    width = 0.25
    for i, evo_algo in enumerate(EVO_VARIANTS):
        means = [np.mean(func_imps.get((f, evo_algo), [0])) for f in FUNCS]
        ax.bar(x + i * width, means, width, label=evo_algo, color=EVO_COLORS[evo_algo], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(FUNCS)
    ax.set_ylabel("平均改进百分比 (%)")
    ax.set_title("各函数上 EVO 改进效果 (所有策略平均)")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(fig, out / "ext_plot03_func_improvement.png")

    # ---- P4: Scalability comparison ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in SMCO_ALL + GLOBAL_BASELINES:
        xs, ys = [], []
        for dim in ALL_DIMS:
            vals = [float(r["fopt"]) for r in rows if r["algo"] == algo and int(r["dim"]) == dim]
            if vals:
                xs.append(dim)
                ys.append(np.mean(vals))
        if xs:
            style = "-" if algo in SMCO_ALL else "--"
            lw = 2 if algo in EVO_VARIANTS else 1
            alpha = 1 if algo in EVO_VARIANTS else 0.5
            ax.plot(xs, ys, style, label=algo, linewidth=lw, alpha=alpha, marker="o" if algo in SMCO_ALL else "s", markersize=3)
    ax.set_xlabel("维度")
    ax.set_ylabel("平均 fopt (所有函数)")
    ax.set_title("各方法解质量随维度变化 (实线=SMCO系列, 虚线=基线)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out / "ext_plot04_scalability.png")

    # ---- P5: Pareto frontier visualization (selected dims) ----
    for dim_label, dim_list in [("50-200D", [50, 100, 200]), ("1000-5000D", HD_DIMS)]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for algo in SMCO_ALL + GLOBAL_BASELINES:
            ts, fs = [], []
            for strat in STRATEGIES:
                for dim in dim_list:
                    vals = [float(r["fopt"]) for r in rows if r["strategy"] == strat and r["algo"] == algo and int(r["dim"]) == dim]
                    times = [float(r["time"]) for r in rows if r["strategy"] == strat and r["algo"] == algo and int(r["dim"]) == dim]
                    if vals and times:
                        ts.append(np.mean(times))
                        fs.append(np.mean(vals))
            if ts:
                marker = "o" if algo in EVO_VARIANTS else ("s" if algo in BASE_VARIANTS else "^")
                color = "#d62728" if algo in EVO_VARIANTS else ("#1f77b4" if algo in BASE_VARIANTS else "#aaaaaa")
                ax.scatter(ts, fs, label=algo, marker=marker, color=color, s=40, alpha=0.6)
        ax.set_xlabel("平均运行时间 (秒)")
        ax.set_ylabel("平均 fopt")
        ax.set_title(f"质量-时间 Pareto 分析 ({dim_label})")
        ax.legend(fontsize=5, ncol=2)
        ax.grid(True, alpha=0.3)
        _save_fig(fig, out / f"ext_plot05_pareto_{dim_label.replace('-', '_')}.png")

    # ---- P6: Dimension-stratified EVO improvement ----
    fig, ax = plt.subplots(figsize=(10, 5))
    dim_ranges = {"50-200D": [50, 100, 200], "500-1000D": [500, 1000], "2000-5000D": HD_DIMS}
    x = np.arange(len(PAIRS))
    width = 0.25
    for i, (range_name, dims) in enumerate(dim_ranges.items()):
        means = []
        for base, evo in PAIRS:
            imps = []
            for strat in STRATEGIES:
                for func in FUNCS:
                    for dim in dims:
                        bv = [float(r["fopt"]) for r in rows if r["strategy"] == strat and r["algo"] == base and r["func"] == func and int(r["dim"]) == dim]
                        ev = [float(r["fopt"]) for r in rows if r["strategy"] == strat and r["algo"] == evo and r["func"] == func and int(r["dim"]) == dim]
                        if bv and ev:
                            n = min(len(bv), len(ev))
                            bm, em = np.mean(bv[:n]), np.mean(ev[:n])
                            if bm != 0:
                                imps.append((em - bm) / abs(bm) * 100)
            means.append(np.mean(imps) if imps else 0)
        ax.bar(x + i * width, means, width, label=range_name, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{b}→{e}" for b, e in PAIRS])
    ax.set_ylabel("平均改进百分比 (%)")
    ax.set_title("不同维度范围下 EVO 改进效果")
    ax.legend()
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(fig, out / "ext_plot06_dim_stratified.png")

    # ---- P7: Robustness plot (win rate vs std of improvement) ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for strat in STRATEGIES:
        srows = [r for r in rows if r["strategy"] == strat]
        for base, evo in PAIRS:
            grouped = _group(srows, "func", "dim")
            imps = []
            for (func, dim), grp in grouped.items():
                bv = [float(r["fopt"]) for r in grp if r["algo"] == base]
                ev = [float(r["fopt"]) for r in grp if r["algo"] == evo]
                if bv and ev:
                    n = min(len(bv), len(ev))
                    bm, em = np.mean(bv[:n]), np.mean(ev[:n])
                    if bm != 0:
                        imps.append((em - bm) / abs(bm) * 100)
            if imps:
                win_rate = sum(1 for x in imps if x > 0) / len(imps) * 100
                std_imp = float(np.std(imps, ddof=1))
                ax.scatter(std_imp, win_rate, s=80, color=STRAT_COLORS[strat],
                          marker=STRAT_MARKERS[strat], label=f"{strat} {base}→{evo}", alpha=0.8)
    ax.set_xlabel("改进标准差 (%)")
    ax.set_ylabel("胜率 (%)")
    ax.set_title("策略稳健性: 胜率 vs 改进波动性")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out / "ext_plot07_robustness.png")

    # ---- P8: Global ranking bar chart ----
    all_configs_rows = []
    for strat in STRATEGIES:
        srows = [r for r in rows if r["strategy"] == strat]
        grouped = _group(srows, "func", "dim")
        for (func, dim), grp in grouped.items():
            algo_means = {}
            for algo in set(r["algo"] for r in grp):
                fv = [float(r["fopt"]) for r in grp if r["algo"] == algo]
                if fv:
                    algo_means[algo] = np.mean(fv)
            ranked = sorted(algo_means.items(), key=lambda x: -x[1])
            for rank, (algo, _) in enumerate(ranked, 1):
                all_configs_rows.append(rank)

    # Recompute for proper grouping
    rank_data = defaultdict(list)
    for strat in STRATEGIES:
        srows = [r for r in rows if r["strategy"] == strat]
        grouped = _group(srows, "func", "dim")
        for (func, dim), grp in grouped.items():
            algo_means = {}
            for algo in set(r["algo"] for r in grp):
                fv = [float(r["fopt"]) for r in grp if r["algo"] == algo]
                if fv:
                    algo_means[algo] = np.mean(fv)
            ranked = sorted(algo_means.items(), key=lambda x: -x[1])
            for rank, (algo, _) in enumerate(ranked, 1):
                rank_data[algo].append(rank)

    algos_sorted = sorted(rank_data.keys(), key=lambda a: np.mean(rank_data[a]))
    avg_ranks = [np.mean(rank_data[a]) for a in algos_sorted]
    colors = ["#d62728" if a in EVO_VARIANTS else "#1f77b4" if a in BASE_VARIANTS else "#aaaaaa" for a in algos_sorted]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(algos_sorted)), avg_ranks, color=colors)
    ax.set_yticks(range(len(algos_sorted)))
    ax.set_yticklabels(algos_sorted)
    ax.set_xlabel("全局平均排名 (越低越好)")
    ax.set_title("全局平均排名 (所有策略×函数×维度)")
    ax.invert_yaxis()
    for bar, val in zip(bars, avg_ranks):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=7)
    ax.legend(handles=[
        Patch(color="#d62728", label="EVO 变体"),
        Patch(color="#1f77b4", label="SMCO base"),
        Patch(color="#aaaaaa", label="其他基线"),
    ], loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    _save_fig(fig, out / "ext_plot08_global_ranking.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="result/highdim-full-comparison-2026-06-04/all_results.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    out = Path(args.output) if args.output else input_path.parent / "analysis_extended"
    out.mkdir(parents=True, exist_ok=True)

    rows = _load(input_path)
    print(f"Loaded {len(rows)} rows")
    print(f"Output: {out}\n")

    print("Running extended analyses...")
    analysis_A(rows, out)
    analysis_B(rows, out)
    analysis_C(rows, out)
    analysis_D(rows, out)
    analysis_E(rows, out)
    analysis_F(rows, out)
    analysis_G(rows, out)
    analysis_H(rows, out)
    analysis_I(rows, out)
    analysis_J(rows, out)
    analysis_K(rows, out)
    analysis_L(rows, out)
    analysis_M(rows, out)

    print("\nGenerating extended plots...")
    plot_all(rows, out)

    print(f"\nDone! Results in {out}")


if __name__ == "__main__":
    main()
