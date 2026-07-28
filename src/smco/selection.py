"""E1 global implementation selection (Task 9).

Selection picks ONE SMCO-EVO implementation globally across all functions,
dimensions and instances; per-function / per-dimension cherry-picking is
forbidden. The ranking cascade (experiment plan, E1 selection rule) is fixed:

    1. Primary:  ECDF-AUC of relative targets over log10(FE/d)   (higher better)
    2. If within 1%:  median normalized log-gap at B_max          (lower better)
    3. If still tied: failure rate                                (lower better)
    4. If still tied: median wall time, single-thread hardware    (lower better)

Every tie-break step that decides the winner is written into
``selection_report.md``. Until E1 results exist the primary score uses the
B-max target-hit rate as a monotone proxy; the full ECDF-AUC is a Gate-E
refinement (clearly noted in ``SELECTION_RULES``).
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Callable

from .experiment_manifests import e1_algorithm_configs

TARGETS = ("1e-1", "1e-2", "1e-3", "1e-5")

SELECTION_RULES = [
    "Primary: ECDF-AUC of relative targets over log10(FE/d) (higher is better). "
    "Until E1 results exist the B-max target-hit rate is used as a monotone proxy "
    "(Gate-E refines to the full ECDF-AUC).",
    "If primary within 1%: median normalized log-gap at B_max (lower is better).",
    "If still tied: failure rate (lower is better).",
    "If still tied: median wall time on fixed single-thread hardware (lower is better).",
]


def selection_candidates():
    """The 18 E1 candidate configurations; each yields one global score."""
    return e1_algorithm_configs()


def _is_success(run: dict) -> bool:
    return run.get("status") == "success"


def score_config(runs: list[dict]) -> dict:
    """Score one config across its runs (failures kept in the denominator)."""
    n = len(runs)
    if n == 0:
        return {
            "n_runs": 0, "target_hit_rate": 0.0, "median_log_gap": None,
            "failure_rate": 1.0, "median_wall_time": None,
        }
    successes = [r for r in runs if _is_success(r)]
    hit = 0
    total = 0
    for run in successes:
        th = run.get("target_hit_fe") or {}
        for target in TARGETS:
            total += 1
            if th.get(target) is not None:
                hit += 1
    target_hit_rate = hit / total if total else 0.0
    gaps = [r.get("normalized_gap") for r in successes if r.get("normalized_gap") is not None]
    median_log_gap = (
        statistics.median(math.log(max(float(g), 1e-12)) for g in gaps) if gaps else None
    )
    failure_rate = 1.0 - len(successes) / n
    walls = [r.get("wall_time_sec") for r in successes if r.get("wall_time_sec") is not None]
    median_wall_time = statistics.median(walls) if walls else None
    return {
        "n_runs": n,
        "target_hit_rate": target_hit_rate,
        "median_log_gap": median_log_gap,
        "failure_rate": failure_rate,
        "median_wall_time": median_wall_time,
    }


def rank_configs(scored: dict) -> list:
    """Rank configs by the fixed cascade; returns list of (algorithm_id, score)."""
    def key(item):
        _aid, score = item
        return (
            -(score.get("target_hit_rate") or 0.0),
            score["median_log_gap"] if score.get("median_log_gap") is not None else math.inf,
            score.get("failure_rate") if score.get("failure_rate") is not None else 1.0,
            score["median_wall_time"] if score.get("median_wall_time") is not None else math.inf,
        )

    return sorted(scored.items(), key=key)


def _default_loader(result_dir, candidates):
    """Group raw/<run_id>.json payloads by algorithm_id."""
    by_algo: dict[str, list] = {}
    if result_dir is None:
        return {c["algorithm_id"]: [] for c in candidates}
    for json_file in Path(result_dir).glob("*.json"):
        if json_file.name.startswith("_"):
            continue
        try:
            payload = json.loads(json_file.read_text())
        except Exception:
            continue
        task = payload.get("task") or {}
        aid = task.get("algorithm_id") or payload.get("algorithm_id")
        if aid:
            by_algo.setdefault(aid, []).append(payload)
    return by_algo


def _write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_candidates_csv(path, ranked):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "algorithm_id"])
        for rank, (aid, _score) in enumerate(ranked, 1):
            writer.writerow([rank, aid])


def _write_score_components_csv(path, scored, ranked):
    order = {aid: rank for rank, (aid, _s) in enumerate(ranked, 1)}
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "algorithm_id", "rank", "n_runs", "target_hit_rate",
            "median_log_gap", "failure_rate", "median_wall_time",
        ])
        for aid, score in scored.items():
            writer.writerow([
                aid, order.get(aid, ""), score["n_runs"],
                _fmt(score["target_hit_rate"]), _fmt(score["median_log_gap"]),
                _fmt(score["failure_rate"]), _fmt(score["median_wall_time"]),
            ])


def _tie_break_notes(ranked, scored):
    """Describe, for each adjacent pair, which rule first separates them."""
    notes = []
    fields = [
        ("target_hit_rate", "primary target-hit rate", 0.01, False),
        ("median_log_gap", "median normalized log-gap", None, True),
        ("failure_rate", "failure rate", None, True),
        ("median_wall_time", "median wall time", None, True),
    ]
    for i in range(len(ranked) - 1):
        aid_a, _ = ranked[i]
        aid_b, _ = ranked[i + 1]
        for key_name, label, _tol, lower_better in fields:
            va = scored[aid_a].get(key_name)
            vb = scored[aid_b].get(key_name)
            if va is None or vb is None or va == vb:
                continue
            better = (va < vb) if lower_better else (va > vb)
            if better:
                notes.append(f"- rank {i+1}>{i+2}: {aid_a} beats {aid_b} on {label}.")
                break
    return notes


def _write_report_dryrun(path, summary):
    lines = ["# E1 Selection Report (dry-run)", ""]
    lines.append("This is a dry-run: no E1 results were read, so no winner is chosen.")
    lines.append("")
    lines.append("## Ranking rules (fixed, global across all functions/dimensions/instances)")
    lines.append("")
    for rule in summary["rules"]:
        lines.append(f"- {rule}")
    lines.append("")
    lines.append(f"Candidates ({summary['n_candidates']}):")
    for aid in summary["candidates"]:
        lines.append(f"- {aid}")
    path.write_text("\n".join(lines) + "\n")


def _write_report(path, summary, ranked, scored):
    lines = ["# E1 Selection Report", ""]
    lines.append(f"Global winner: **{summary['winner']}**")
    lines.append("")
    lines.append("## Ranking rules (fixed)")
    lines.append("")
    for rule in summary["rules"]:
        lines.append(f"- {rule}")
    lines.append("")
    lines.append("## Tie-break decisions")
    lines.append("")
    notes = _tie_break_notes(ranked, scored)
    lines.extend(notes if notes else ["- (no ties needed to separate the winner)"])
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append("| rank | algorithm_id | target_hit_rate | median_log_gap | failure_rate | median_wall_time |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for rank, (aid, score) in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {aid} | {_fmt(score['target_hit_rate'])} | "
            f"{_fmt(score['median_log_gap'])} | {_fmt(score['failure_rate'])} | "
            f"{_fmt(score['median_wall_time'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def build_selection(
    result_dir,
    *,
    out_dir,
    dry_run: bool = False,
    candidates=None,
    loader: Callable | None = None,
) -> dict:
    """Run (or dry-run) selection and write the four outputs."""
    candidates = candidates if candidates is not None else selection_candidates()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        summary = {
            "dry_run": True,
            "n_candidates": len(candidates),
            "rules": list(SELECTION_RULES),
            "candidates": [c["algorithm_id"] for c in candidates],
        }
        _write_json(out_dir / "selection.json", summary)
        _write_report_dryrun(out_dir / "selection_report.md", summary)
        return summary

    runs_by_config = (loader or _default_loader)(result_dir, candidates)
    scored = {
        c["algorithm_id"]: score_config(runs_by_config.get(c["algorithm_id"], []))
        for c in candidates
    }
    ranked = rank_configs(scored)
    winner = ranked[0][0] if ranked else None
    summary = {
        "dry_run": False,
        "n_candidates": len(candidates),
        "winner": winner,
        "rules": list(SELECTION_RULES),
    }
    _write_json(out_dir / "selection.json", summary)
    _write_candidates_csv(out_dir / "selection_candidates.csv", ranked)
    _write_score_components_csv(out_dir / "selection_score_components.csv", scored, ranked)
    _write_report(out_dir / "selection_report.md", summary, ranked, scored)
    return summary


__all__ = [
    "SELECTION_RULES",
    "TARGETS",
    "selection_candidates",
    "score_config",
    "rank_configs",
    "build_selection",
]
