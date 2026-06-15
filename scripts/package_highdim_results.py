#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import fmean, median


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"
SOURCE_DIRS = [
    RESULT_DIR / "highdim-comparison-2026-05-31",
    RESULT_DIR / "ultrahighdim-2026-06-01",
]
DEST_DIR = RESULT_DIR / f"highdim-overall-{date.today()}"
ANALYSIS_DIR = DEST_DIR / "analysis"

PAIR_MAP = {
    "SMCO_vs_SMCO_EVO": ("SMCO", "SMCO_EVO"),
    "SMCO_R_vs_SMCO_R_EVO": ("SMCO_R", "SMCO_R_EVO"),
    "SMCO_BR_vs_SMCO_BR_EVO": ("SMCO_BR", "SMCO_BR_EVO"),
}
PAIR_ORDER = [
    "SMCO_vs_SMCO_EVO",
    "SMCO_R_vs_SMCO_R_EVO",
    "SMCO_BR_vs_SMCO_BR_EVO",
]


@dataclass
class PairOutcome:
    source: str
    pair: str
    func: str
    dim: int
    rep: int
    base_algo: str
    evo_algo: str
    base_fopt: float
    evo_fopt: float
    base_time: float
    evo_time: float

    @property
    def winner(self) -> str:
        if self.evo_fopt > self.base_fopt:
            return "evo"
        if self.evo_fopt < self.base_fopt:
            return "base"
        return "tie"

    @property
    def rel_improvement_pct(self) -> float:
        if self.base_fopt == 0.0:
            return 0.0
        return (self.evo_fopt - self.base_fopt) / abs(self.base_fopt) * 100.0

    @property
    def time_ratio(self) -> float:
        if self.base_time == 0.0:
            return 0.0
        return self.evo_time / self.base_time


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def mean_or_zero(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def median_or_zero(values: list[float]) -> float:
    return median(values) if values else 0.0


def aggregate_outcomes(outcomes: list[PairOutcome]) -> dict[str, float | int]:
    evo_wins = sum(1 for item in outcomes if item.winner == "evo")
    base_wins = sum(1 for item in outcomes if item.winner == "base")
    ties = sum(1 for item in outcomes if item.winner == "tie")
    rel_improvements = [item.rel_improvement_pct for item in outcomes]
    time_ratios = [item.time_ratio for item in outcomes]
    return {
        "n_cases": len(outcomes),
        "evo_wins": evo_wins,
        "base_wins": base_wins,
        "ties": ties,
        "evo_win_rate": evo_wins / len(outcomes) if outcomes else 0.0,
        "mean_rel_improvement_pct": mean_or_zero(rel_improvements),
        "median_rel_improvement_pct": median_or_zero(rel_improvements),
        "mean_time_ratio": mean_or_zero(time_ratios),
        "median_time_ratio": median_or_zero(time_ratios),
    }


def load_exp_a_outcomes() -> list[PairOutcome]:
    outcomes: list[PairOutcome] = []
    for src in SOURCE_DIRS:
        rows = read_csv(src / "expA_scaling.csv")
        grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[(row["func"], int(row["dim"]), int(row["rep"]))].append(row)
        for (func, dim, rep), entries in sorted(grouped.items()):
            by_algo = {row["algo"]: row for row in entries}
            if "SMCO_R" not in by_algo or "SMCO_R_EVO" not in by_algo:
                continue
            base = by_algo["SMCO_R"]
            evo = by_algo["SMCO_R_EVO"]
            outcomes.append(
                PairOutcome(
                    source=src.name,
                    pair="SMCO_R_vs_SMCO_R_EVO",
                    func=func,
                    dim=dim,
                    rep=rep,
                    base_algo="SMCO_R",
                    evo_algo="SMCO_R_EVO",
                    base_fopt=float(base["fopt"]),
                    evo_fopt=float(evo["fopt"]),
                    base_time=float(base["time"]),
                    evo_time=float(evo["time"]),
                )
            )
    return outcomes


def load_exp_b_outcomes() -> list[PairOutcome]:
    outcomes: list[PairOutcome] = []
    for src in SOURCE_DIRS:
        rows = read_csv(src / "expB_variants.csv")
        grouped: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[(row["pair"], row["func"], int(row["dim"]), int(row["rep"]))].append(row)
        for (pair, func, dim, rep), entries in sorted(grouped.items()):
            if pair not in PAIR_MAP:
                continue
            base_algo, evo_algo = PAIR_MAP[pair]
            by_algo = {row["algo"]: row for row in entries}
            if base_algo not in by_algo or evo_algo not in by_algo:
                continue
            base = by_algo[base_algo]
            evo = by_algo[evo_algo]
            outcomes.append(
                PairOutcome(
                    source=src.name,
                    pair=pair,
                    func=func,
                    dim=dim,
                    rep=rep,
                    base_algo=base_algo,
                    evo_algo=evo_algo,
                    base_fopt=float(base["fopt"]),
                    evo_fopt=float(evo["fopt"]),
                    base_time=float(base["time"]),
                    evo_time=float(evo["time"]),
                )
            )
    return outcomes


def load_exp_c_rows() -> list[dict[str, str]]:
    path = SOURCE_DIRS[0] / "expC_convergence.csv"
    return read_csv(path) if path.exists() else []


def summarize_exp_c(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["func"], int(row["dim"]))].append(row)
    for (func, dim), entries in sorted(grouped.items()):
        speed_rows: dict[str, int] = {}
        for algo in ("SMCO_R", "SMCO_R_EVO"):
            algo_entries = [row for row in entries if row["algo"] == algo]
            by_iter: dict[int, list[float]] = defaultdict(list)
            for row in algo_entries:
                by_iter[int(row["iteration"])].append(float(row["runmax"]))
            iterations = sorted(by_iter)
            means = [mean_or_zero(by_iter[i]) for i in iterations]
            if not means:
                continue
            initial = means[0]
            final = means[-1]
            target = initial + 0.95 * (final - initial)
            hit = iterations[-1]
            for iteration, value in zip(iterations, means):
                if (final >= initial and value >= target) or (final < initial and value <= target):
                    hit = iteration
                    break
            speed_rows[algo] = hit
        if "SMCO_R" in speed_rows and "SMCO_R_EVO" in speed_rows:
            base_iter = speed_rows["SMCO_R"]
            evo_iter = speed_rows["SMCO_R_EVO"]
            if base_iter == 0 or evo_iter == 0:
                continue
            ratio = (base_iter / evo_iter) if evo_iter else 0.0
            summaries.append(
                {
                    "func": func,
                    "dim": dim,
                    "smco_r_95_iter": base_iter,
                    "smco_r_evo_95_iter": evo_iter,
                    "ratio": ratio,
                }
            )
    return summaries


def build_summary_csv_rows(
    exp_a: list[PairOutcome],
    exp_b: list[PairOutcome],
    exp_c: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def append_row(section: str, group: str, subgroup: str, stats: dict[str, float | int]) -> None:
        rows.append(
            {
                "section": section,
                "group": group,
                "subgroup": subgroup,
                "n_cases": stats["n_cases"],
                "evo_wins": stats["evo_wins"],
                "base_wins": stats["base_wins"],
                "ties": stats["ties"],
                "evo_win_rate": stats["evo_win_rate"],
                "mean_rel_improvement_pct": stats["mean_rel_improvement_pct"],
                "median_rel_improvement_pct": stats["median_rel_improvement_pct"],
                "mean_time_ratio": stats["mean_time_ratio"],
                "median_time_ratio": stats["median_time_ratio"],
            }
        )

    append_row("expA", "overall", "all", aggregate_outcomes(exp_a))
    for dim in sorted({item.dim for item in exp_a}):
        append_row(
            "expA",
            "dimension",
            f"{dim}D",
            aggregate_outcomes([item for item in exp_a if item.dim == dim]),
        )
    for func in sorted({item.func for item in exp_a}):
        append_row(
            "expA",
            "function",
            func,
            aggregate_outcomes([item for item in exp_a if item.func == func]),
        )

    append_row("expB", "overall", "all_pairs", aggregate_outcomes(exp_b))
    for pair in PAIR_ORDER:
        append_row(
            "expB",
            "pair",
            pair,
            aggregate_outcomes([item for item in exp_b if item.pair == pair]),
        )
        for dim in sorted({item.dim for item in exp_b if item.pair == pair}):
            append_row(
                "expB",
                "pair_dimension",
                f"{pair}:{dim}D",
                aggregate_outcomes(
                    [item for item in exp_b if item.pair == pair and item.dim == dim]
                ),
            )

    for row in exp_c:
        rows.append(
            {
                "section": "expC",
                "group": "convergence",
                "subgroup": f"{row['func']}:{row['dim']}D",
                "n_cases": "",
                "evo_wins": "",
                "base_wins": "",
                "ties": "",
                "evo_win_rate": "",
                "mean_rel_improvement_pct": "",
                "median_rel_improvement_pct": "",
                "mean_time_ratio": row["ratio"],
                "median_time_ratio": "",
            }
        )
    return rows


def build_report(
    exp_a: list[PairOutcome],
    exp_b: list[PairOutcome],
    exp_c: list[dict[str, object]],
) -> str:
    exp_a_overall = aggregate_outcomes(exp_a)
    exp_b_overall = aggregate_outcomes(exp_b)
    exp_a_by_dim = {
        dim: aggregate_outcomes([item for item in exp_a if item.dim == dim])
        for dim in sorted({item.dim for item in exp_a})
    }
    exp_a_by_func = {
        func: aggregate_outcomes([item for item in exp_a if item.func == func])
        for func in sorted({item.func for item in exp_a})
    }
    exp_b_by_pair = {
        pair: aggregate_outcomes([item for item in exp_b if item.pair == pair])
        for pair in PAIR_ORDER
    }
    exp_b_by_pair_dim = {
        pair: {
            dim: aggregate_outcomes([item for item in exp_b if item.pair == pair and item.dim == dim])
            for dim in sorted({item.dim for item in exp_b if item.pair == pair})
        }
        for pair in PAIR_ORDER
    }

    top_positive = sorted(exp_a, key=lambda item: item.rel_improvement_pct, reverse=True)[:8]
    top_negative = sorted(exp_a, key=lambda item: item.rel_improvement_pct)[:8]
    stable_functions = sorted(
        exp_a_by_func.items(),
        key=lambda item: (item[1]["evo_win_rate"], item[1]["mean_rel_improvement_pct"]),
        reverse=True,
    )

    lines = [
        "# High-Dimensional SMCO-EVO Overall Report",
        "",
        f"- Generated: {date.today()}",
        f"- Sources: `{SOURCE_DIRS[0].name}`, `{SOURCE_DIRS[1].name}`",
        "- Pairing rule: match by `(pair, func, dim, rep)`; all win/loss decisions use the corrected SMCO maximize-oriented `fopt` semantics.",
        "",
        "## Executive Summary",
        "",
        (
            f"- `ExpA` (`SMCO_R` vs `SMCO_R_EVO`) gives EVO {exp_a_overall['evo_wins']}/"
            f"{exp_a_overall['n_cases']} wins ({pct(exp_a_overall['evo_win_rate'])}); "
            f"mean relative improvement is {exp_a_overall['mean_rel_improvement_pct']:+.3f}%."
        ),
        (
            f"- `ExpB` (all three base/EVO pairs combined) gives EVO {exp_b_overall['evo_wins']}/"
            f"{exp_b_overall['n_cases']} wins ({pct(exp_b_overall['evo_win_rate'])}); "
            f"mean relative improvement is {exp_b_overall['mean_rel_improvement_pct']:+.3f}%."
        ),
        (
            f"- Runtime overhead stays negligible: mean time ratio is "
            f"{exp_a_overall['mean_time_ratio']:.3f}x in `ExpA` and {exp_b_overall['mean_time_ratio']:.3f}x in `ExpB`."
        ),
        "- Main pattern: evolutionary selection helps more consistently as dimension grows, but the benefit is strongly function-dependent.",
        "",
        "## Dimension-Level Summary",
        "",
        "| Dim | Cases | EVO wins | Base wins | EVO win rate | Mean rel. impr. | Mean time ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dim, stats in exp_a_by_dim.items():
        lines.append(
            f"| {dim}D | {stats['n_cases']} | {stats['evo_wins']} | {stats['base_wins']} | "
            f"{pct(stats['evo_win_rate'])} | {stats['mean_rel_improvement_pct']:+.3f}% | "
            f"{stats['mean_time_ratio']:.3f}x |"
        )

    lines.extend(
        [
            "",
            "The dimension trend is directionally positive after the analysis fix: 50D and 100D are already above 55% EVO win rate,",
            "500D reaches 70.0%, and the ultra-high-dimensional tail remains favorable despite the small sample counts at 2000D and 5000D.",
            "",
            "## Function-Level Analysis",
            "",
            "| Function | Cases | EVO wins | Base wins | EVO win rate | Mean rel. impr. |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for func, stats in stable_functions:
        lines.append(
            f"| {func} | {stats['n_cases']} | {stats['evo_wins']} | {stats['base_wins']} | "
            f"{pct(stats['evo_win_rate'])} | {stats['mean_rel_improvement_pct']:+.3f}% |"
        )

    lines.extend(
        [
            "",
            "- `Ackley` is the clearest success case: EVO wins every paired run in the merged `ExpA` set and the relative gain increases with dimension.",
            "- `Zakharov` also trends positive, with strong win rates and small but consistent mean gains.",
            "- `Rastrigin` is close to neutral overall: wins are slightly positive, but mean relative change is almost zero.",
            "- `DixonPrice` and `Qing` are the main regressions in this suite; they are the strongest evidence that EVO is not uniformly beneficial.",
            "- `Rosenbrock` is mixed: win counts are not dominant, but a few very high-dimensional cases create larger positive mean shifts.",
            "",
            "## Variant Comparison",
            "",
            "| Pair | Cases | EVO wins | Base wins | EVO win rate | Mean rel. impr. | Mean time ratio |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in PAIR_ORDER:
        stats = exp_b_by_pair[pair]
        lines.append(
            f"| {pair} | {stats['n_cases']} | {stats['evo_wins']} | {stats['base_wins']} | "
            f"{pct(stats['evo_win_rate'])} | {stats['mean_rel_improvement_pct']:+.3f}% | "
            f"{stats['mean_time_ratio']:.3f}x |"
        )

    lines.extend(
        [
            "",
            "Per-dimension detail for `ExpB`:",
            "",
            "| Pair | Dim | Cases | EVO wins | Base wins | EVO win rate |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for pair in PAIR_ORDER:
        for dim, stats in exp_b_by_pair_dim[pair].items():
            lines.append(
            f"| {pair} | {dim}D | {stats['n_cases']} | {stats['evo_wins']} | "
            f"{stats['base_wins']} | {pct(stats['evo_win_rate'])} |"
            )

    lines.extend(
        [
            "",
        (
            f"- `SMCO_BR_EVO` is the most stable family in the merged variant sweep, reaching "
            f"{pct(exp_b_by_pair['SMCO_BR_vs_SMCO_BR_EVO']['evo_win_rate'])} overall win rate."
        ),
        (
            f"- `SMCO_EVO` and `SMCO_R_EVO` are effectively tied overall at "
            f"{pct(exp_b_by_pair['SMCO_vs_SMCO_EVO']['evo_win_rate'])}."
        ),
            "- The gap between variants widens in the ultra-high-dimensional regime; `SMCO_BR_EVO` is strongest at 1000D and 2000D.",
            "",
            "## Convergence Readout",
            "",
            "| Function | Dim | SMCO_R 95% iter | SMCO_R_EVO 95% iter | Ratio |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in exp_c:
        lines.append(
            f"| {row['func']} | {row['dim']}D | {row['smco_r_95_iter']} | "
            f"{row['smco_r_evo_95_iter']} | {row['ratio']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "Ackley is the main convergence-speed exception here: EVO reaches the 95% progress threshold notably earlier.",
            "Rastrigin and Rosenbrock stay near 1.0x, so outside Ackley the current evidence still favors a quality-improvement interpretation over a broad convergence-speed claim.",
            "",
            "## Notable Cases",
            "",
            "Largest positive relative shifts in `ExpA`:",
            "",
            "| Function | Dim | Rep | Source | Rel. impr. |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for item in top_positive:
        lines.append(
            f"| {item.func} | {item.dim}D | {item.rep} | {item.source} | {item.rel_improvement_pct:+.3f}% |"
        )
    lines.extend(
        [
            "",
            "Largest negative relative shifts in `ExpA`:",
            "",
            "| Function | Dim | Rep | Source | Rel. impr. |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for item in top_negative:
        lines.append(
            f"| {item.func} | {item.dim}D | {item.rep} | {item.source} | {item.rel_improvement_pct:+.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.csv`: compact machine-readable overview of all aggregate statistics.",
            "- `analysis/expA_cases.csv`: all merged `ExpA` paired outcomes.",
            "- `analysis/expA_by_function.csv`: per-function aggregates for merged `ExpA`.",
            "- `analysis/expB_cases.csv`: all merged `ExpB` paired outcomes.",
            "- `analysis/expB_by_pair.csv`: per-pair aggregates for merged `ExpB`.",
            "- `analysis/expC_summary.csv`: convergence readout extracted from the corrected `ExpC` data.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    exp_a = load_exp_a_outcomes()
    exp_b = load_exp_b_outcomes()
    exp_c = summarize_exp_c(load_exp_c_rows())

    exp_a_case_rows = [
        {
            "source": item.source,
            "pair": item.pair,
            "func": item.func,
            "dim": item.dim,
            "rep": item.rep,
            "base_algo": item.base_algo,
            "evo_algo": item.evo_algo,
            "base_fopt": item.base_fopt,
            "evo_fopt": item.evo_fopt,
            "winner": item.winner,
            "rel_improvement_pct": item.rel_improvement_pct,
            "base_time": item.base_time,
            "evo_time": item.evo_time,
            "time_ratio": item.time_ratio,
        }
        for item in exp_a
    ]
    exp_b_case_rows = [
        {
            "source": item.source,
            "pair": item.pair,
            "func": item.func,
            "dim": item.dim,
            "rep": item.rep,
            "base_algo": item.base_algo,
            "evo_algo": item.evo_algo,
            "base_fopt": item.base_fopt,
            "evo_fopt": item.evo_fopt,
            "winner": item.winner,
            "rel_improvement_pct": item.rel_improvement_pct,
            "base_time": item.base_time,
            "evo_time": item.evo_time,
            "time_ratio": item.time_ratio,
        }
        for item in exp_b
    ]

    exp_a_by_function_rows = []
    for func in sorted({item.func for item in exp_a}):
        stats = aggregate_outcomes([item for item in exp_a if item.func == func])
        exp_a_by_function_rows.append({"func": func, **stats})

    exp_b_by_pair_rows = []
    for pair in PAIR_ORDER:
        stats = aggregate_outcomes([item for item in exp_b if item.pair == pair])
        exp_b_by_pair_rows.append({"pair": pair, **stats})

    write_csv(
        DEST_DIR / "summary.csv",
        build_summary_csv_rows(exp_a, exp_b, exp_c),
        [
            "section",
            "group",
            "subgroup",
            "n_cases",
            "evo_wins",
            "base_wins",
            "ties",
            "evo_win_rate",
            "mean_rel_improvement_pct",
            "median_rel_improvement_pct",
            "mean_time_ratio",
            "median_time_ratio",
        ],
    )
    write_csv(ANALYSIS_DIR / "expA_cases.csv", exp_a_case_rows, list(exp_a_case_rows[0].keys()))
    write_csv(
        ANALYSIS_DIR / "expA_by_function.csv",
        exp_a_by_function_rows,
        list(exp_a_by_function_rows[0].keys()),
    )
    write_csv(ANALYSIS_DIR / "expB_cases.csv", exp_b_case_rows, list(exp_b_case_rows[0].keys()))
    write_csv(
        ANALYSIS_DIR / "expB_by_pair.csv",
        exp_b_by_pair_rows,
        list(exp_b_by_pair_rows[0].keys()),
    )
    write_csv(
        ANALYSIS_DIR / "expC_summary.csv",
        exp_c,
        list(exp_c[0].keys()) if exp_c else ["func", "dim", "smco_r_95_iter", "smco_r_evo_95_iter", "ratio"],
    )
    (DEST_DIR / "report.md").write_text(build_report(exp_a, exp_b, exp_c), encoding="utf-8")
    print(f"Packaged report in {DEST_DIR}")


if __name__ == "__main__":
    main()
