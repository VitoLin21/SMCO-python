#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"
DEST_DIR = RESULT_DIR / "smco-evo"
SOURCE_DIR = DEST_DIR / "source"
ANALYSIS_DIR = DEST_DIR / "analysis"

DIRECT_DIR = RESULT_DIR / "smco-evo-vs-smco-2026-05-27"
PAPER_DIR = RESULT_DIR / "paper_tables_integrated_2026-05-27"
TABLE3_DIR = RESULT_DIR / "table3_2026-05-27"
SYNCED_DIR = RESULT_DIR / "r-synced-benchmarks-2026-05-28"
STRATEGY_SWEEP_DIR = RESULT_DIR / "evo_strategy_sweep"
RAND1BIN_SWEEP_DIR = STRATEGY_SWEEP_DIR / "rand1bin"

PAIRS = [
    ("SMCO", "SMCO_EVO"),
    ("SMCO_R", "SMCO_R_EVO"),
    ("SMCO_BR", "SMCO_BR_EVO"),
]
SMCO_FAMILY = ["SMCO", "SMCO_R", "SMCO_BR", "SMCO_EVO", "SMCO_R_EVO", "SMCO_BR_EVO"]
EVO_ALGOS = ["SMCO_EVO", "SMCO_R_EVO", "SMCO_BR_EVO"]
STRATEGIES = ["rand1bin", "current-to-best1bin", "best1bin", "sobol"]


@dataclass
class PairCase:
    source: str
    scenario: str
    name: str
    dim: int
    maximize: bool
    base_algo: str
    evo_algo: str
    base_gap: float
    evo_gap: float
    gap_improvement: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def copy_source_tree(src: Path, dest_parent: Path) -> None:
    dest = dest_parent / src.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def ensure_rand1bin_strategy_snapshot() -> None:
    RAND1BIN_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    for name, src in (
        ("paper_tables_integrated", PAPER_DIR),
        ("table3", TABLE3_DIR),
        ("r_synced", SYNCED_DIR),
    ):
        dest = RAND1BIN_SWEEP_DIR / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)


def copy_source_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def ensure_clean_dest() -> None:
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def direct_summary() -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    rows = read_csv(DIRECT_DIR / "summary.csv")
    case_rows: list[dict[str, object]] = []
    for row in rows:
        gap_improvement = safe_float(row.get("gap_improvement"))
        if gap_improvement is not None:
            decision_metric = gap_improvement
            decision_basis = "gap_improvement"
        else:
            evo_wins = safe_int(row.get("evo_win_reps")) or 0
            smco_wins = safe_int(row.get("smco_win_reps")) or 0
            decision_metric = float(evo_wins - smco_wins)
            decision_basis = "rep_win_delta"
        if decision_metric > 1e-12:
            outcome = "evo_better"
        elif decision_metric < -1e-12:
            outcome = "smco_better"
        else:
            outcome = "tie"
        case_rows.append(
            {
                "function": row["function"],
                "dim": row["dim"],
                "sense": row["sense"],
                "known_best_objective": row["known_best_objective"],
                "smco_mean": row["smco_mean"],
                "evo_mean": row["evo_mean"],
                "smco_mean_time": row["smco_mean_time"],
                "evo_mean_time": row["evo_mean_time"],
                "evo_win_reps": row["evo_win_reps"],
                "smco_win_reps": row["smco_win_reps"],
                "tie_reps": row["tie_reps"],
                "smco_mean_gap": row["smco_mean_gap"],
                "evo_mean_gap": row["evo_mean_gap"],
                "gap_improvement": row["gap_improvement"],
                "decision_basis": decision_basis,
                "decision_metric": decision_metric,
                "outcome": outcome,
            }
        )
    wins = sum(1 for row in case_rows if row["outcome"] == "evo_better")
    losses = sum(1 for row in case_rows if row["outcome"] == "smco_better")
    ties = sum(1 for row in case_rows if row["outcome"] == "tie")
    total_evo_rep_wins = sum(int(row["evo_win_reps"]) for row in case_rows)
    total_smco_rep_wins = sum(int(row["smco_win_reps"]) for row in case_rows)
    total_tie_reps = sum(int(row["tie_reps"]) for row in case_rows)
    known_gap_rows = [row for row in case_rows if safe_float(str(row["gap_improvement"])) is not None]
    top_improved = sorted(
        [row for row in case_rows if row["outcome"] == "evo_better"],
        key=lambda row: float(row["decision_metric"]),
        reverse=True,
    )[:8]
    top_regressed = sorted(
        [row for row in case_rows if row["outcome"] == "smco_better"],
        key=lambda row: float(row["decision_metric"]),
    )[:8]
    overview = {
        "benchmarks": len(case_rows),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "rep_wins_evo": total_evo_rep_wins,
        "rep_wins_smco": total_smco_rep_wins,
        "rep_ties": total_tie_reps,
        "known_gap_rows": len(known_gap_rows),
    }
    write_csv(
        ANALYSIS_DIR / "direct_smco_vs_evo_cases.csv",
        case_rows,
        list(case_rows[0].keys()),
    )
    write_csv(
        ANALYSIS_DIR / "direct_smco_vs_evo_top_improved.csv",
        top_improved,
        list(top_improved[0].keys()) if top_improved else list(case_rows[0].keys()),
    )
    write_csv(
        ANALYSIS_DIR / "direct_smco_vs_evo_top_regressed.csv",
        top_regressed,
        list(top_regressed[0].keys()) if top_regressed else list(case_rows[0].keys()),
    )
    return overview, top_improved, top_regressed


def collect_pair_cases(source_name: str, csv_paths: Iterable[Path]) -> list[PairCase]:
    pair_cases: list[PairCase] = []
    for path in csv_paths:
        rows = read_csv(path)
        by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_label[row["label"]].append(row)
        for label_rows in by_label.values():
            by_algo = {row["algo"]: row for row in label_rows}
            for base_algo, evo_algo in PAIRS:
                if base_algo not in by_algo or evo_algo not in by_algo:
                    continue
                base_row = by_algo[base_algo]
                evo_row = by_algo[evo_algo]
                best_opt = safe_float(base_row.get("best_opt"))
                base_mean = safe_float(base_row.get("fopt_mean"))
                evo_mean = safe_float(evo_row.get("fopt_mean"))
                if best_opt is None or base_mean is None or evo_mean is None:
                    continue
                base_gap = abs(best_opt - base_mean)
                evo_gap = abs(best_opt - evo_mean)
                pair_cases.append(
                    PairCase(
                        source=source_name,
                        scenario=base_row["label"],
                        name=base_row["name"],
                        dim=int(base_row["dim"]),
                        maximize=parse_bool(base_row["maximize"]),
                        base_algo=base_algo,
                        evo_algo=evo_algo,
                        base_gap=base_gap,
                        evo_gap=evo_gap,
                        gap_improvement=base_gap - evo_gap,
                    )
                )
    return pair_cases


def pair_case_rows(cases: list[PairCase]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        rows.append(
            {
                "source": case.source,
                "scenario": case.scenario,
                "name": case.name,
                "dim": case.dim,
                "maximize": case.maximize,
                "base_algo": case.base_algo,
                "evo_algo": case.evo_algo,
                "base_gap": case.base_gap,
                "evo_gap": case.evo_gap,
                "gap_improvement": case.gap_improvement,
                "winner": case.evo_algo if case.gap_improvement > 0 else case.base_algo if case.gap_improvement < 0 else "tie",
            }
        )
    return rows


def summarize_pair_cases(cases: list[PairCase], tag: str) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str], list[PairCase]] = defaultdict(list)
    for case in cases:
        grouped[(case.source, case.base_algo, case.evo_algo)].append(case)
    for (source, base_algo, evo_algo), group in sorted(grouped.items()):
        wins = sum(1 for case in group if case.gap_improvement > 1e-12)
        losses = sum(1 for case in group if case.gap_improvement < -1e-12)
        ties = len(group) - wins - losses
        mean_improvement = sum(case.gap_improvement for case in group) / len(group)
        median_sorted = sorted(case.gap_improvement for case in group)
        median_improvement = median_sorted[len(group) // 2] if len(group) % 2 == 1 else (
            median_sorted[len(group) // 2 - 1] + median_sorted[len(group) // 2]
        ) / 2.0
        summary_rows.append(
            {
                "tag": tag,
                "source": source,
                "base_algo": base_algo,
                "evo_algo": evo_algo,
                "scenarios": len(group),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_rate": wins / len(group),
                "mean_gap_improvement": mean_improvement,
                "median_gap_improvement": median_improvement,
            }
        )
    return summary_rows


def summarize_pair_cases_by_group(cases: list[PairCase], group_fn) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str], list[PairCase]] = defaultdict(list)
    for case in cases:
        grouped[(group_fn(case), case.base_algo, case.evo_algo)].append(case)
    for (group_name, base_algo, evo_algo), group in sorted(grouped.items()):
        wins = sum(1 for case in group if case.gap_improvement > 1e-12)
        losses = sum(1 for case in group if case.gap_improvement < -1e-12)
        ties = len(group) - wins - losses
        mean_improvement = sum(case.gap_improvement for case in group) / len(group)
        median_sorted = sorted(case.gap_improvement for case in group)
        median_improvement = median_sorted[len(group) // 2] if len(group) % 2 == 1 else (
            median_sorted[len(group) // 2 - 1] + median_sorted[len(group) // 2]
        ) / 2.0
        summary_rows.append(
            {
                "group": group_name,
                "base_algo": base_algo,
                "evo_algo": evo_algo,
                "scenarios": len(group),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_rate": wins / len(group),
                "mean_gap_improvement": mean_improvement,
                "median_gap_improvement": median_improvement,
            }
        )
    return summary_rows


def best_family_counts(source_name: str, csv_paths: Iterable[Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    counts = Counter()
    rank_sums = Counter()
    rank_counts = Counter()
    case_rows: list[dict[str, object]] = []
    for path in csv_paths:
        rows = read_csv(path)
        by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_label[row["label"]].append(row)
        for label_rows in by_label.values():
            family_rows = [row for row in label_rows if row["algo"] in SMCO_FAMILY]
            if not family_rows:
                continue
            best_opt = safe_float(family_rows[0].get("best_opt"))
            if best_opt is None:
                continue
            scored = []
            for row in family_rows:
                mean = safe_float(row.get("fopt_mean"))
                if mean is None:
                    continue
                gap = abs(best_opt - mean)
                scored.append((row["algo"], gap))
            if not scored:
                continue
            scored.sort(key=lambda item: (item[1], item[0]))
            best_algo, best_gap = scored[0]
            counts[best_algo] += 1
            case_rows.append(
                {
                    "source": source_name,
                    "scenario": family_rows[0]["label"],
                    "name": family_rows[0]["name"],
                    "dim": family_rows[0]["dim"],
                    "best_algo": best_algo,
                    "best_gap": best_gap,
                }
            )
            for rank, (algo, gap) in enumerate(scored, start=1):
                rank_sums[algo] += rank
                rank_counts[algo] += 1
    count_rows = [
        {
            "source": source_name,
            "algo": algo,
            "best_count": counts[algo],
            "average_rank": rank_sums[algo] / rank_counts[algo] if rank_counts[algo] else None,
            "rank_count": rank_counts[algo],
        }
        for algo in SMCO_FAMILY
        if rank_counts[algo]
    ]
    return count_rows, case_rows


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "_No data._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def _reference_best_opt_map() -> dict[str, float]:
    reference: dict[str, float] = {}
    for path in sorted(PAPER_DIR.glob("*.csv")):
        rows = read_csv(path)
        if not rows:
            continue
        best = safe_float(rows[0].get("best_opt"))
        if best is not None:
            reference[rows[0]["label"]] = best
    synced_rows = read_csv(SYNCED_DIR / "summary.csv")
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in synced_rows:
        by_label[row["label"]].append(row)
    for label, rows in by_label.items():
        best = safe_float(rows[0].get("best_opt"))
        if best is not None:
            reference[label] = best
    return reference


def _collect_strategy_rows(strategy: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = STRATEGY_SWEEP_DIR / strategy
    paper_dir = base / "paper_tables_integrated"
    table3_dir = base / "table3"
    synced_csv = base / "r_synced" / "summary.csv"
    for csv_path in sorted(paper_dir.glob("*.csv")):
        for row in read_csv(csv_path):
            if row["algo"] in EVO_ALGOS:
                rows.append({"source": "paper_tables", "strategy": strategy, **row})
    for csv_path in sorted(table3_dir.glob("*.csv")):
        for row in read_csv(csv_path):
            if row["algo"] in EVO_ALGOS:
                rows.append({"source": "table3", "strategy": strategy, **row})
    if synced_csv.exists():
        for row in read_csv(synced_csv):
            if row["algo"] in EVO_ALGOS:
                rows.append({"source": "r_synced", "strategy": strategy, **row})
    return rows


def strategy_sweep_analysis() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if not STRATEGY_SWEEP_DIR.exists():
        return [], [], []
    strategies_found = [s for s in STRATEGIES if (STRATEGY_SWEEP_DIR / s).exists()]
    if not strategies_found:
        return [], [], []

    reference_best = _reference_best_opt_map()
    all_rows: list[dict[str, object]] = []
    for strategy in strategies_found:
        all_rows.extend(_collect_strategy_rows(strategy))
    if not all_rows:
        return [], [], []

    metric_rows_raw: list[dict[str, object]] = []
    for row in all_rows:
        label = row["label"]
        best = reference_best.get(label)
        mean = safe_float(row.get("fopt_mean"))
        if best is None or mean is None:
            continue
        metric_rows_raw.append(
            {
                "source": row["source"],
                "label": label,
                "name": row["name"],
                "dim": int(row["dim"]),
                "algo": row["algo"],
                "strategy": row["strategy"],
                "reference_best_opt": best,
                "fopt_mean": mean,
                "gap_to_reference": abs(best - mean),
            }
        )
    metric_rows_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in metric_rows_raw:
        key = (str(row["strategy"]), str(row["label"]), str(row["algo"]))
        existing = metric_rows_by_key.get(key)
        if existing is None:
            metric_rows_by_key[key] = row
            continue
        # Prefer the dedicated table3/r_synced rows over integrated duplicates.
        source_rank = {"r_synced": 2, "table3": 1, "paper_tables": 0}
        if source_rank.get(str(row["source"]), -1) > source_rank.get(str(existing["source"]), -1):
            metric_rows_by_key[key] = row
    metric_rows = list(metric_rows_by_key.values())
    if not metric_rows:
        return [], [], []

    write_csv(
        ANALYSIS_DIR / "strategy_sweep_evo_rows.csv",
        metric_rows,
        list(metric_rows[0].keys()),
    )

    by_key: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in metric_rows:
        by_key[(row["label"], row["algo"])][row["strategy"]] = row

    pairwise_rows: list[dict[str, object]] = []
    if "rand1bin" in strategies_found:
        for (label, algo), strategy_map in by_key.items():
            base = strategy_map.get("rand1bin")
            if not base:
                continue
            for strategy, row in strategy_map.items():
                if strategy == "rand1bin":
                    continue
                diff = float(base["gap_to_reference"]) - float(row["gap_to_reference"])
                pairwise_rows.append(
                    {
                        "label": label,
                        "algo": algo,
                        "baseline_strategy": "rand1bin",
                        "strategy": strategy,
                        "baseline_gap": base["gap_to_reference"],
                        "strategy_gap": row["gap_to_reference"],
                        "gap_improvement_vs_baseline": diff,
                        "winner": strategy if diff > 0 else "rand1bin" if diff < 0 else "tie",
                    }
                )
        if pairwise_rows:
            write_csv(
                ANALYSIS_DIR / "strategy_sweep_vs_rand1bin_cases.csv",
                pairwise_rows,
                list(pairwise_rows[0].keys()),
            )

    summary_rows: list[dict[str, object]] = []
    for strategy in strategies_found:
        subset = [row for row in metric_rows if row["strategy"] == strategy]
        if not subset:
            continue
        by_algo: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in subset:
            by_algo[str(row["algo"])].append(row)
        for algo, algo_rows in sorted(by_algo.items()):
            mean_gap = sum(float(row["gap_to_reference"]) for row in algo_rows) / len(algo_rows)
            median_sorted = sorted(float(row["gap_to_reference"]) for row in algo_rows)
            median_gap = median_sorted[len(median_sorted) // 2] if len(median_sorted) % 2 == 1 else (
                median_sorted[len(median_sorted) // 2 - 1] + median_sorted[len(median_sorted) // 2]
            ) / 2.0
            summary_rows.append(
                {
                    "strategy": strategy,
                    "algo": algo,
                    "cases": len(algo_rows),
                    "mean_gap_to_reference": mean_gap,
                    "median_gap_to_reference": median_gap,
                }
            )
    if summary_rows:
        write_csv(
            ANALYSIS_DIR / "strategy_sweep_summary.csv",
            summary_rows,
            list(summary_rows[0].keys()),
        )

    best_rows: list[dict[str, object]] = []
    by_label_algo: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        by_label_algo[(str(row["label"]), str(row["algo"]))].append(row)
    for (label, algo), rows in sorted(by_label_algo.items()):
        rows_sorted = sorted(rows, key=lambda item: (float(item["gap_to_reference"]), str(item["strategy"])))
        best_rows.append(
            {
                "label": label,
                "algo": algo,
                "best_strategy": rows_sorted[0]["strategy"],
                "best_gap_to_reference": rows_sorted[0]["gap_to_reference"],
            }
        )
    if best_rows:
        write_csv(
            ANALYSIS_DIR / "strategy_sweep_best_strategy_by_case.csv",
            best_rows,
            list(best_rows[0].keys()),
        )
    return metric_rows, summary_rows, best_rows


def format_pair_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    formatted = []
    for row in rows:
        formatted.append(
            {
                "source": row["source"],
                "pair": f"{row['base_algo']} -> {row['evo_algo']}",
                "scenarios": row["scenarios"],
                "wins": row["wins"],
                "losses": row["losses"],
                "ties": row["ties"],
                "win_rate": format_pct(float(row["win_rate"])),
                "mean_gap_improvement": format_float(float(row["mean_gap_improvement"]), 3),
                "median_gap_improvement": format_float(float(row["median_gap_improvement"]), 3),
            }
        )
    return formatted


def format_best_count_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    formatted = []
    for row in rows:
        formatted.append(
            {
                "source": row["source"],
                "algo": row["algo"],
                "best_count": row["best_count"],
                "average_rank": format_float(safe_float(str(row["average_rank"])), 3),
            }
        )
    return formatted


def format_group_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    formatted = []
    for row in rows:
        formatted.append(
            {
                "group": row["group"],
                "pair": f"{row['base_algo']} -> {row['evo_algo']}",
                "scenarios": row["scenarios"],
                "wins": row["wins"],
                "losses": row["losses"],
                "ties": row["ties"],
                "win_rate": format_pct(float(row["win_rate"])),
                "mean_gap_improvement": format_float(float(row["mean_gap_improvement"]), 3),
                "median_gap_improvement": format_float(float(row["median_gap_improvement"]), 3),
            }
        )
    return formatted


def build_report(
    direct_overview: dict[str, object],
    direct_top_improved: list[dict[str, object]],
    direct_top_regressed: list[dict[str, object]],
    pair_summary_rows: list[dict[str, object]],
    best_count_rows: list[dict[str, object]],
    paper_group_rows: list[dict[str, object]],
    strategy_metric_rows: list[dict[str, object]],
    strategy_summary_rows: list[dict[str, object]],
    strategy_best_rows: list[dict[str, object]],
) -> str:
    direct_rows = [
        {
            "metric": "18函数 head-to-head: evo 更优",
            "value": direct_overview["wins"],
        },
        {
            "metric": "18函数 head-to-head: 原版更优",
            "value": direct_overview["losses"],
        },
        {
            "metric": "18函数 head-to-head: 持平",
            "value": direct_overview["ties"],
        },
        {
            "metric": "重复层面: evo 胜",
            "value": direct_overview["rep_wins_evo"],
        },
        {
            "metric": "重复层面: 原版胜",
            "value": direct_overview["rep_wins_smco"],
        },
        {
            "metric": "重复层面: 平局",
            "value": direct_overview["rep_ties"],
        },
    ]
    top_improved_rows = [
        {
            "function": row["function"],
            "dim": row["dim"],
            "metric": format_float(float(row["decision_metric"]), 3),
            "basis": row["decision_basis"],
            "evo_win_reps": row["evo_win_reps"],
            "smco_win_reps": row["smco_win_reps"],
            "tie_reps": row["tie_reps"],
        }
        for row in direct_top_improved[:6]
    ]
    top_regressed_rows = [
        {
            "function": row["function"],
            "dim": row["dim"],
            "metric": format_float(float(row["decision_metric"]), 3),
            "basis": row["decision_basis"],
            "evo_win_reps": row["evo_win_reps"],
            "smco_win_reps": row["smco_win_reps"],
            "tie_reps": row["tie_reps"],
        }
        for row in direct_top_regressed[:6]
    ]
    paper_pair_rows = [row for row in pair_summary_rows if row["source"] == "paper_tables"]
    synced_pair_rows = [row for row in pair_summary_rows if row["source"] == "r_synced"]
    best_count_sorted = sorted(best_count_rows, key=lambda row: (row["source"], -int(row["best_count"]), row["algo"]))
    paper_group_focus = [
        row for row in paper_group_rows
        if row["pair"] in {"SMCO -> SMCO_EVO", "SMCO_R -> SMCO_R_EVO", "SMCO_BR -> SMCO_BR_EVO"}
    ]

    paper_xy_bars = []
    paper_xy_losses = []
    paper_labels = []
    for row in paper_pair_rows:
        paper_labels.append(f"\"{row['pair']}\"")
        paper_xy_bars.append(str(row["wins"]))
        paper_xy_losses.append(str(row["losses"]))
    synced_xy_bars = []
    synced_xy_losses = []
    synced_labels = []
    for row in synced_pair_rows:
        synced_labels.append(f"\"{row['pair']}\"")
        synced_xy_bars.append(str(row["wins"]))
        synced_xy_losses.append(str(row["losses"]))

    strategy_table_rows: list[dict[str, object]] = []
    for row in strategy_summary_rows:
        strategy_table_rows.append(
            {
                "strategy": row["strategy"],
                "algo": row["algo"],
                "cases": row["cases"],
                "mean_gap_to_reference": format_float(float(row["mean_gap_to_reference"]), 3),
                "median_gap_to_reference": format_float(float(row["median_gap_to_reference"]), 3),
            }
        )
    strategy_best_count: Counter[str] = Counter()
    for row in strategy_best_rows:
        strategy_best_count[str(row["best_strategy"])] += 1
    strategy_best_table_rows = [{"strategy": strategy, "best_case_count": strategy_best_count[strategy]} for strategy in STRATEGIES if strategy in strategy_best_count]
    strategy_coverage_rows: list[dict[str, object]] = []
    coverage_by_strategy: dict[str, set[str]] = defaultdict(set)
    for row in strategy_metric_rows:
        coverage_by_strategy[str(row["strategy"])].add(str(row["label"]))
    for strategy in STRATEGIES:
        case_set = coverage_by_strategy.get(strategy, set())
        strategy_coverage_rows.append(
            {
                "strategy": strategy,
                "covered_cases": len(case_set),
            }
        )
    pie_lines = []
    for row in strategy_best_table_rows:
        pie_lines.append(f'    "{row["strategy"]}" : {row["best_case_count"]}')
    pie_block = "\n".join(pie_lines) if pie_lines else '    "no-data" : 1'

    report = f"""# SMCO-EVO Results Report

## 1. 结果目录与覆盖范围

本次整理后的统一目录为 `result/smco-evo/`，结构如下：

- `source/smco-evo-vs-smco-2026-05-27/`: 原始 `smco` vs `smco_evo` head-to-head 结果
- `source/paper_tables_integrated_2026-05-27/`: 论文 Table 1/2/3 整合结果
- `source/r-synced-benchmarks-2026-05-28/`: 从上游 R 同步到 Python 的 18 个 benchmark 全量结果
- `source/evo_strategy_sweep/`: 四种进化策略（`rand1bin/current-to-best1bin/best1bin/sobol`）对应结果
- `analysis/`: 本次新增的汇总表、pairwise 对比表和六个 SMCO 变体的排名统计

```mermaid
flowchart LR
    A[smco-evo-vs-smco-2026-05-27] --> D[result/smco-evo]
    B[paper_tables_integrated_2026-05-27] --> D
    C[r-synced-benchmarks-2026-05-28] --> D
    G[evo_strategy_sweep/*] --> D
    D --> E[analysis/*.csv]
    D --> F[report.md]
```

本报告主要回答两个问题：

1. `smco-evo` 相比原版 `smco` 是否稳定更好？
2. 把视角放大到 `SMCO / SMCO_R / SMCO_BR` 三条主线后，evo 变体是否普遍带来收益？

## 2. 核心结论

{markdown_table(direct_rows, ["metric", "value"])}

结论可以先概括成一句话：`smco-evo` 在直接 head-to-head 上明显占优，但把范围扩展到论文表格和全量 R-synced benchmark 后，收益并不是“无条件单调提升”，而是更偏向 **原版 SMCO 主线明显受益，refine/boosted 两条支线增益更混合**。

```mermaid
pie title smco_evo 直连对比结果分布
    "EVO better" : {direct_overview["wins"]}
    "SMCO better" : {direct_overview["losses"]}
    "Tie" : {direct_overview["ties"]}
```

## 3. 直接对比：SMCO vs SMCO_EVO

这部分使用 `source/smco-evo-vs-smco-2026-05-27/summary.csv` 的 18 函数对照结果。这里的优势最直接，因为实验就是专门为了回答 `smco_evo` 是否比原版 `smco` 更有效。

### 3.1 改进最明显的函数

{markdown_table(top_improved_rows, ["function", "dim", "metric", "basis", "evo_win_reps", "smco_win_reps", "tie_reps"])}

### 3.2 回退或无增益最明显的函数

{markdown_table(top_regressed_rows, ["function", "dim", "metric", "basis", "evo_win_reps", "smco_win_reps", "tie_reps"])}

分析要点：

- `Eggholder`、`Qing10d`、`Shubert`、`Rosenbrock10d` 是这批数据里最明显的受益者，说明阶段性淘汰和补点在复杂地形、非凸或病态曲面上更能发挥作用。
- `Bukin6` 是最清晰的回退案例，说明 `rand1bin` 的补点在某些低维尖锐地形上可能打破了原版轨迹的局部精修节奏。
- `Michalewicz10d`、`Mishra6` 没有已知最优值字段，因此这里用重复层面的胜负差作为替代判据；它们总体仍偏向 `smco-evo`。

## 4. family 视角：论文 Table 1/2/3

这部分不只比较 `SMCO`，还比较 `SMCO_R` 与 `SMCO_BR` 的 evo 版本。统一指标是：

`mean gap = |best_opt - fopt_mean|`

gap 改进量定义为：

`gap improvement = gap(base) - gap(evo)`

值为正表示 evo 更好，值为负表示原版更好。

### 4.1 pairwise 汇总

{markdown_table(paper_pair_rows, ["pair", "scenarios", "wins", "losses", "ties", "win_rate", "mean_gap_improvement", "median_gap_improvement"])}

```mermaid
xychart-beta
    title "Paper tables: evo 胜出场景数"
    x-axis [{", ".join(paper_labels)}]
    y-axis "Wins / Losses" 0 --> 24
    bar [{", ".join(paper_xy_bars)}]
    line [{", ".join(paper_xy_losses)}]
```

### 4.2 六个 SMCO 变体谁最常拿到最优

{markdown_table([row for row in best_count_sorted if row["source"] == "paper_tables"], ["algo", "best_count", "average_rank"])}

### 4.3 按 Table 1 / Table 2 / Table 3 拆开看

{markdown_table(paper_group_focus, ["group", "pair", "scenarios", "wins", "losses", "ties", "win_rate", "mean_gap_improvement", "median_gap_improvement"])}

对论文表格的解释：

- `SMCO -> SMCO_EVO` 在论文表格里并不是稳定单调提升：`table1` 基本打平，`table2` 偏弱，`table3` 的均值收益主要来自少数 `EmpiricalWelfare` 大幅改善场景。
- `SMCO_R -> SMCO_R_EVO` 的总均值优势很大，但要注意其中有明显高维 outlier，不能只看 mean；从胜负场次看，它也不是“碾压式”提升。
- `SMCO_BR -> SMCO_BR_EVO` 在论文表格里是最混合的一组，说明 boosted 轨迹本身已经比较敏感，evo 注入的额外扰动不一定总是正收益。
- `table2` 的 200 维任务尤其关键，因为它们更能暴露中途淘汰是否真的改善了高维搜索，而不只是低维 lucky hit。

## 5. family 视角：R-synced 18 个 benchmark

这部分覆盖 18 个从上游 R 同步到 Python 的 benchmark，并且包含 comparison 方法全量结果。这里我们仍然只聚焦六个 SMCO 变体的内部对比。

### 5.1 pairwise 汇总

{markdown_table(synced_pair_rows, ["pair", "scenarios", "wins", "losses", "ties", "win_rate", "mean_gap_improvement", "median_gap_improvement"])}

```mermaid
xychart-beta
    title "R-synced benchmarks: evo 胜出场景数"
    x-axis [{", ".join(synced_labels)}]
    y-axis "Wins / Losses" 0 --> 18
    bar [{", ".join(synced_xy_bars)}]
    line [{", ".join(synced_xy_losses)}]
```

### 5.2 六个 SMCO 变体谁最常拿到最优

{markdown_table([row for row in best_count_sorted if row["source"] == "r_synced"], ["algo", "best_count", "average_rank"])}

这部分比论文表格更像“广覆盖压力测试”：

- 如果 `SMCO_EVO` 在这里仍能稳定压过 `SMCO`，说明它不是只对论文中的那几类任务有效。
- 如果 `SMCO_R_EVO` 或 `SMCO_BR_EVO` 的 best-count 和 average-rank 没有同步改善，那就说明 evo 的收益主要来自主线多起点搜索，而不是 refine / boosted 的后处理阶段。

## 6. 结论与建议

综合三类结果，`smco-evo` 的效果可以概括为：

1. **`smco-evo` 的正信号是真实存在的。** 直接 18 函数 head-to-head 是最干净的证据，`15/18` 函数更好，重复层面也是明显占优。
2. **但收益存在明显任务依赖。** 论文表格里，`SMCO_EVO` 和 `SMCO_BR_EVO` 都没有表现出“全场景稳定优于原版”的特征；`SMCO_R_EVO` 更强，但也受少数高维大收益点影响。
3. **在广覆盖的 R-synced benchmark 上，evo family 整体重新显示出强优势。** 这说明进化补点本身不是偶然有效，而是对更广泛的连续测试函数仍有价值。
4. **默认 `rand1bin` 只是其中一种可行策略，不应直接当作唯一结论。** 现在四种策略都已补齐，同一批基准上可以直接比较 `rand1bin / current-to-best1bin / best1bin / sobol` 的稳定性与收益分布。

## 7. 进化策略扫全量（新增）

本节比较 `rand1bin / current-to-best1bin / best1bin / sobol` 在已跑基准上的表现。  
统一指标仍是 `|reference_best_opt - fopt_mean|`，其中 `reference_best_opt` 来自主基线目录（不随策略变化）。

{markdown_table(strategy_coverage_rows, ["strategy", "covered_cases"])}

{markdown_table(strategy_table_rows, ["strategy", "algo", "cases", "mean_gap_to_reference", "median_gap_to_reference"])}

{markdown_table(strategy_best_table_rows, ["strategy", "best_case_count"])}

```mermaid
pie title 策略成为最优的case占比
{pie_block}
```

## 8. 生成文件清单

- `analysis/direct_smco_vs_evo_cases.csv`
- `analysis/direct_smco_vs_evo_top_improved.csv`
- `analysis/direct_smco_vs_evo_top_regressed.csv`
- `analysis/paper_pairwise_cases.csv`
- `analysis/paper_pairwise_summary.csv`
- `analysis/r_synced_pairwise_cases.csv`
- `analysis/r_synced_pairwise_summary.csv`
- `analysis/smco_family_best_counts.csv`
- `analysis/smco_family_best_cases.csv`
- `analysis/strategy_sweep_evo_rows.csv`
- `analysis/strategy_sweep_summary.csv`
- `analysis/strategy_sweep_best_strategy_by_case.csv`
"""
    return report


def main() -> None:
    ensure_rand1bin_strategy_snapshot()
    ensure_clean_dest()
    for src in (DIRECT_DIR, PAPER_DIR, SYNCED_DIR, STRATEGY_SWEEP_DIR):
        copy_source_tree(src, SOURCE_DIR)

    direct_overview, direct_top_improved, direct_top_regressed = direct_summary()

    paper_csvs = sorted(PAPER_DIR.glob("*.csv"))
    synced_csvs = [SYNCED_DIR / "summary.csv"]

    paper_cases = collect_pair_cases("paper_tables", paper_csvs)
    synced_cases = collect_pair_cases("r_synced", synced_csvs)

    write_csv(
        ANALYSIS_DIR / "paper_pairwise_cases.csv",
        pair_case_rows(paper_cases),
        [
            "source",
            "scenario",
            "name",
            "dim",
            "maximize",
            "base_algo",
            "evo_algo",
            "base_gap",
            "evo_gap",
            "gap_improvement",
            "winner",
        ],
    )
    write_csv(
        ANALYSIS_DIR / "r_synced_pairwise_cases.csv",
        pair_case_rows(synced_cases),
        [
            "source",
            "scenario",
            "name",
            "dim",
            "maximize",
            "base_algo",
            "evo_algo",
            "base_gap",
            "evo_gap",
            "gap_improvement",
            "winner",
        ],
    )

    pair_summary = summarize_pair_cases(paper_cases, "paper_tables") + summarize_pair_cases(synced_cases, "r_synced")
    write_csv(
        ANALYSIS_DIR / "paper_pairwise_summary.csv",
        [row for row in pair_summary if row["source"] == "paper_tables"],
        list(pair_summary[0].keys()),
    )
    write_csv(
        ANALYSIS_DIR / "r_synced_pairwise_summary.csv",
        [row for row in pair_summary if row["source"] == "r_synced"],
        list(pair_summary[0].keys()),
    )
    paper_group_summary = summarize_pair_cases_by_group(paper_cases, lambda case: case.scenario.split("_")[0])
    write_csv(
        ANALYSIS_DIR / "paper_pairwise_by_table.csv",
        paper_group_summary,
        list(paper_group_summary[0].keys()),
    )

    paper_best_counts, paper_best_cases = best_family_counts("paper_tables", paper_csvs)
    synced_best_counts, synced_best_cases = best_family_counts("r_synced", synced_csvs)
    write_csv(
        ANALYSIS_DIR / "smco_family_best_counts.csv",
        paper_best_counts + synced_best_counts,
        ["source", "algo", "best_count", "average_rank", "rank_count"],
    )
    write_csv(
        ANALYSIS_DIR / "smco_family_best_cases.csv",
        paper_best_cases + synced_best_cases,
        ["source", "scenario", "name", "dim", "best_algo", "best_gap"],
    )
    strategy_metric_rows, strategy_summary_rows, strategy_best_rows = strategy_sweep_analysis()

    report = build_report(
        direct_overview=direct_overview,
        direct_top_improved=direct_top_improved,
        direct_top_regressed=direct_top_regressed,
        pair_summary_rows=format_pair_summary_rows(pair_summary),
        best_count_rows=format_best_count_rows(paper_best_counts + synced_best_counts),
        paper_group_rows=format_group_summary_rows(paper_group_summary),
        strategy_metric_rows=strategy_metric_rows,
        strategy_summary_rows=strategy_summary_rows,
        strategy_best_rows=strategy_best_rows,
    )
    (DEST_DIR / "report.md").write_text(report)


if __name__ == "__main__":
    main()
