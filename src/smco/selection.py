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
import functools
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Callable

from .experiment_manifests import e1_algorithm_configs, load_manifest
from .paper_contract import NONE_TOKEN, canonical_json, parse_algorithm_id

TARGETS = ("1e-1", "1e-2", "1e-3", "1e-5")

SELECTION_RULES = [
    "Primary: ECDF-AUC of relative targets over log10(FE/dim) (higher is better). "
    "Failed/timeout runs are right-censored (never solved) but kept in the denominator.",
    "If AUC within 1%: median normalized log-gap at B_max (lower is better).",
    "If still tied: failure rate (lower is better).",
    "If still tied: median wall time on fixed single-thread hardware (lower is better).",
]

# ECDF-AUC integration window: B_max = 1000*d FE => FE/dim in [1, 1000].
_ECDF_LOG_LO = 0.0
_ECDF_LOG_HI = math.log10(1000.0)
_ECDF_N_GRID = 50


def selection_candidates():
    """The 18 E1 candidate configurations; each yields one global score."""
    return e1_algorithm_configs()


def _is_success(run: dict) -> bool:
    return run.get("status") == "success"


def ecdf_auc(runs: list[dict], targets=TARGETS, n_grid: int = _ECDF_N_GRID) -> float:
    """Area under the ECDF of relative targets over log10(FE/dim) in [0, 1].

    Each (run, target) pair is "solved" at the first FE where the run reaches the
    target; failed/timeout runs and unreached targets are right-censored (never
    solved) but kept in the denominator, so a config that fails often cannot win
    on a few lucky hits. The ECDF at FE/dim = 10^x is the fraction of pairs
    solved by then; AUC integrates it over log10(FE/dim) and normalises to
    [0, 1]. Higher is better. This is the E1 primary score (review A-02).
    """
    n_pairs = 0
    x_hits: list[float] = []
    for run in runs:
        dim = max(1, int(run.get("dimension") or run.get("dim") or 1))
        th = (run.get("target_hit_fe") or {}) if _is_success(run) else {}
        for target in targets:
            n_pairs += 1
            fe = th.get(target)
            if fe:
                x_hits.append(math.log10(float(fe) / dim))
    if n_pairs == 0 or _ECDF_LOG_HI <= _ECDF_LOG_LO:
        return 0.0
    width = _ECDF_LOG_HI - _ECDF_LOG_LO
    step = width / n_grid
    auc = 0.0
    for i in range(n_grid):
        xa = _ECDF_LOG_LO + i * step
        solved = sum(1 for x in x_hits if x <= xa)
        auc += (solved / n_pairs) * step
    return auc / width


def score_config(runs: list[dict]) -> dict:
    """Score one config across its runs (failures kept in the denominator)."""
    n = len(runs)
    if n == 0:
        return {
            "n_runs": 0, "ecdf_auc": 0.0, "target_hit_rate": 0.0,
            "median_log_gap": None, "failure_rate": 1.0, "median_wall_time": None,
        }
    hit = 0
    total = 0
    for run in runs:  # all runs; failures keep their target slots (right censored)
        th = (run.get("target_hit_fe") or {}) if _is_success(run) else {}
        for target in TARGETS:
            total += 1
            if th.get(target) is not None:
                hit += 1
    target_hit_rate = hit / total if total else 0.0
    successes = [r for r in runs if _is_success(r)]
    gaps = [r.get("normalized_gap") for r in successes if r.get("normalized_gap") is not None]
    median_log_gap = (
        statistics.median(math.log(max(float(g), 1e-12)) for g in gaps) if gaps else None
    )
    failure_rate = 1.0 - len(successes) / n
    walls = [r.get("wall_time_sec") for r in successes if r.get("wall_time_sec") is not None]
    median_wall_time = statistics.median(walls) if walls else None
    return {
        "n_runs": n,
        "ecdf_auc": ecdf_auc(runs),
        "target_hit_rate": target_hit_rate,
        "median_log_gap": median_log_gap,
        "failure_rate": failure_rate,
        "median_wall_time": median_wall_time,
    }


def rank_configs(scored: dict) -> list:
    """Rank configs by the fixed cascade; returns list of (algorithm_id, score).

    Primary is ECDF-AUC; the tiebreak cascade (log-gap, failure rate, wall time)
    only applies between configs whose AUC differs by less than 1% (plan E1
    rule). AUC differences of >= 1% fully determine the order.
    """
    def cmp(a, b):
        _aid_a, sa = a
        _aid_b, sb = b
        auc_a = sa.get("ecdf_auc") or 0.0
        auc_b = sb.get("ecdf_auc") or 0.0
        if abs(auc_a - auc_b) >= 0.01:
            return -1 if auc_a > auc_b else 1
        for key_name in ("median_log_gap", "failure_rate", "median_wall_time"):
            va = sa.get(key_name)
            vb = sb.get(key_name)
            if va is None or vb is None or va == vb:
                continue
            return -1 if va < vb else 1
        return 0

    return sorted(scored.items(), key=functools.cmp_to_key(cmp))


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


def _merged_loader(merged_dir, candidates, *, expected_stage="e1_development"):
    """Load E1 rows from merged/valid_runs.csv (R-05 canonical selection input).

    Returns ``(by_algo, dropped_other_stage)`` where ``by_algo`` is a
    {algorithm_id: [payload]} map shaped like :func:`_default_loader` and
    ``dropped_other_stage`` lists candidate rows whose stage is not the expected
    E1 stage (R5b isolation). Raises if provenance_audit.json did not pass.
    """
    merged_dir = Path(merged_dir)
    audit_path = merged_dir / "provenance_audit.json"
    if not audit_path.exists():
        raise ValueError(f"merged audit not found: {audit_path}")
    audit = json.loads(audit_path.read_text())
    if not audit.get("passed"):
        raise ValueError(
            f"provenance audit did not pass (failed: {audit.get('failed_checks')}); "
            f"selection refuses to freeze a winner over unaudited results"
        )
    rows = list(csv.DictReader(open(merged_dir / "valid_runs.csv")))
    by_algo: dict[str, list] = {c["algorithm_id"]: [] for c in candidates}
    dropped_other_stage: list[dict] = []
    for r in rows:
        aid = r.get("algorithm_id")
        if aid not in by_algo:
            continue
        if r.get("stage") != expected_stage:
            dropped_other_stage.append(r)  # R5b: other-stage rows are isolated, not scored
            continue
        th: dict[str, int] = {}
        for t in TARGETS:
            v = r.get(f"target_hit_fe_{t}")
            if v not in ("", None, NONE_TOKEN):
                try:
                    th[t] = int(float(v))
                except (TypeError, ValueError):
                    pass
        gap = r.get("normalized_gap")
        wall = r.get("wall_time_sec")
        dim = r.get("dimension")
        by_algo[aid].append({
            "status": r.get("status", "success"),
            "dimension": int(float(dim)) if dim not in ("", None) else 1,
            "normalized_gap": float(gap) if gap not in ("", None, NONE_TOKEN) else None,
            "wall_time_sec": float(wall) if wall not in ("", None) else None,
            "target_hit_fe": th,
            "run_id": r.get("run_id"),
            "task": {"configuration_hash": r.get("configuration_hash"),
                     "instance_hash": r.get("instance_hash")},
        })
    return by_algo, dropped_other_stage


def _load_e1_manifest_index(e1_manifest_paths) -> dict[str, dict[str, dict]]:
    """{algorithm_id: {run_id: task}} from the frozen E1 manifests (R5b)."""
    index: dict[str, dict[str, dict]] = {}
    for path in e1_manifest_paths:
        manifest = load_manifest(path)
        for task in manifest.get("tasks", []):
            aid = task.get("algorithm_id")
            rid = task.get("run_id")
            if aid and rid:
                index.setdefault(aid, {})[rid] = task
    return index


def _enforce_e1_manifest_closure(runs_by_config, candidates, e1_index, dropped_other_stage):
    """R5b: each candidate must carry EXACTLY its E1 manifest task set (matching
    run_ids + configuration_hash), and the merged input must be free of
    other-stage rows. A mixed-stage or wrong-count directory cannot freeze the
    E1 winner.
    """
    errors: list[str] = []
    if dropped_other_stage:
        stages = sorted({r.get("stage") for r in dropped_other_stage})
        errors.append(
            f"merged input has {len(dropped_other_stage)} non-E1 row(s) "
            f"(stages {stages}); canonical E1 selection refuses a contaminated input"
        )
    for c in candidates:
        aid = c["algorithm_id"]
        expected = e1_index.get(aid, {})
        expected_ids = set(expected)
        rows = runs_by_config.get(aid, [])
        kept = [r.get("run_id") for r in rows]
        kept_set = set(kept)
        if len(kept) != len(kept_set):
            errors.append(f"{aid}: duplicate run_id within E1 rows")
        missing = sorted(expected_ids - kept_set)
        extra = sorted(kept_set - expected_ids)
        if missing:
            errors.append(
                f"{aid}: missing {len(missing)} of {len(expected_ids)} E1 task(s) "
                f"({missing[:3]}); each candidate must carry exactly its manifest tasks")
        if extra:
            errors.append(
                f"{aid}: {len(extra)} row(s) not in the E1 manifest ({extra[:3]})")
        for r in rows:
            rid = r.get("run_id")
            mtask = expected.get(rid)
            if not mtask:
                continue
            row_hash = (r.get("task") or {}).get("configuration_hash")
            man_hash = mtask.get("configuration_hash")
            if (row_hash and man_hash and row_hash not in (None, NONE_TOKEN)
                    and man_hash not in (None, NONE_TOKEN) and row_hash != man_hash):
                errors.append(f"{aid}/{rid}: configuration_hash mismatch vs E1 manifest")
    if errors:
        raise ValueError("E1 canonical selection rejected: " + "; ".join(errors))


def _enforce_merged_completeness(runs_by_config, candidates):
    """R-05: reject incomplete or duplicate E1 coverage before freezing a winner."""
    errors: list[str] = []
    missing = [c["algorithm_id"] for c in candidates if not runs_by_config.get(c["algorithm_id"])]
    if missing:
        errors.append(f"candidates with no runs: {missing}")
    counts = {aid: len(r) for aid, r in runs_by_config.items() if r}
    if counts:
        n = max(counts.values())
        incomplete = {aid: c for aid, c in counts.items() if c < n}
        if incomplete:
            errors.append(f"incomplete coverage (max={n}): {incomplete}")
    all_ids = [r.get("run_id") for runs in runs_by_config.values() for r in runs if r.get("run_id")]
    dup = sorted({i for i in all_ids if all_ids.count(i) > 1})
    if dup:
        errors.append(f"duplicate run_id in merged/: {dup}")
    if errors:
        raise ValueError("E1 merged input rejected: " + "; ".join(errors))


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
            "algorithm_id", "rank", "n_runs", "ecdf_auc", "target_hit_rate",
            "median_log_gap", "failure_rate", "median_wall_time",
        ])
        for aid, score in scored.items():
            writer.writerow([
                aid, order.get(aid, ""), score["n_runs"],
                _fmt(score["ecdf_auc"]), _fmt(score["target_hit_rate"]),
                _fmt(score["median_log_gap"]), _fmt(score["failure_rate"]),
                _fmt(score["median_wall_time"]),
            ])


def _tie_break_notes(ranked, scored):
    """Describe, for each adjacent pair, which rule first separates them."""
    notes = []
    fields = [
        ("ecdf_auc", "primary ECDF-AUC", 0.01, False),
        ("median_log_gap", "median normalized log-gap", None, True),
        ("failure_rate", "failure rate", None, True),
        ("median_wall_time", "median wall time", None, True),
    ]
    for i in range(len(ranked) - 1):
        aid_a, _ = ranked[i]
        aid_b, _ = ranked[i + 1]
        for key_name, label, tol, lower_better in fields:
            va = scored[aid_a].get(key_name)
            vb = scored[aid_b].get(key_name)
            if va is None or vb is None:
                continue
            if tol is not None and abs(va - vb) < tol:
                continue  # within tolerance -> not separated by this field
            if va == vb:
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
    lines.append("| rank | algorithm_id | ecdf_auc | target_hit_rate | median_log_gap | failure_rate | median_wall_time |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for rank, (aid, score) in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {aid} | {_fmt(score['ecdf_auc'])} | {_fmt(score['target_hit_rate'])} | "
            f"{_fmt(score['median_log_gap'])} | {_fmt(score['failure_rate'])} | "
            f"{_fmt(score['median_wall_time'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _selection_hash(summary: dict) -> str:
    """Stable hash of the selection summary (excludes the hash field itself)."""
    payload = {k: v for k, v in summary.items() if k != "selection_hash"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def build_selection(
    result_dir=None,
    *,
    out_dir,
    dry_run: bool = False,
    candidates=None,
    loader: Callable | None = None,
    merged_dir=None,
    e1_manifest_paths=None,
    development: bool = False,
) -> dict:
    """Run (or dry-run) selection and write the four outputs.

    Canonical mode reads merged/ (``merged_dir``) AND the frozen E1 manifests
    (``e1_manifest_paths``): it enforces the provenance audit, isolates/rejects
    other-stage rows, and verifies each candidate carries EXACTLY its E1 manifest
    task set (matching run_ids + configuration_hash) before freezing a winner
    (R-05/R5b). Reading raw ``result_dir`` JSON is development-only
    (``development=True``); an explicit ``loader`` (test hook) bypasses both.
    """
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

    e1_manifest_validated = False
    if merged_dir:
        if not e1_manifest_paths:
            raise ValueError(
                "canonical E1 selection requires e1_manifest_paths (the frozen E1 "
                "manifests) to validate stage, run_ids and per-candidate task count; "
                "pass development=True for raw result_dir exploration"
            )
        e1_index = _load_e1_manifest_index(e1_manifest_paths)
        runs_by_config, dropped_other_stage = _merged_loader(merged_dir, candidates)
        _enforce_e1_manifest_closure(runs_by_config, candidates, e1_index, dropped_other_stage)
        e1_manifest_validated = True
    elif loader is not None:
        runs_by_config = loader(result_dir, candidates)
    else:
        if not development:
            raise ValueError(
                "reading raw result_dir is development-only; pass development=True "
                "or supply merged_dir (+ e1_manifest_paths) for canonical selection"
            )
        runs_by_config = _default_loader(result_dir, candidates)
    scored = {
        c["algorithm_id"]: score_config(runs_by_config.get(c["algorithm_id"], []))
        for c in candidates
    }
    ranked = rank_configs(scored)
    winner = ranked[0][0] if ranked else None
    winner_language = None
    if winner is not None:
        try:
            winner_language = parse_algorithm_id(winner)["language"]
        except Exception:
            winner_language = None
    winner_runs = runs_by_config.get(winner, []) if winner else []
    winner_config_hash = None
    if winner_runs:
        wtask = winner_runs[0].get("task") or {}
        winner_config_hash = wtask.get("configuration_hash")
    # A-02 part 2: record per-candidate coverage + a fingerprint of the result
    # set so a stale/incomplete directory cannot silently freeze a winner.
    candidate_ids = [c["algorithm_id"] for c in candidates]
    coverage = {aid: len(runs_by_config.get(aid, [])) for aid in candidate_ids}
    all_run_ids = sorted({
        r.get("run_id") for runs in runs_by_config.values() for r in runs if r.get("run_id")
    })
    results_hash = (hashlib.sha256(canonical_json(all_run_ids).encode("utf-8")).hexdigest()[:16]
                    if all_run_ids else None)
    summary = {
        "dry_run": False,
        "n_candidates": len(candidates),
        "winner": winner,
        "winner_language": winner_language,
        "winner_config_hash": winner_config_hash,
        "coverage": coverage,
        "n_results": len(all_run_ids),
        "results_hash": results_hash,
        "e1_manifest_validated": e1_manifest_validated,
        "rules": list(SELECTION_RULES),
    }
    summary["selection_hash"] = _selection_hash(summary)
    _write_json(out_dir / "selection.json", summary)
    _write_candidates_csv(out_dir / "selection_candidates.csv", ranked)
    _write_score_components_csv(out_dir / "selection_score_components.csv", scored, ranked)
    _write_report(out_dir / "selection_report.md", summary, ranked, scored)
    return summary


__all__ = [
    "SELECTION_RULES",
    "TARGETS",
    "ecdf_auc",
    "selection_candidates",
    "score_config",
    "rank_configs",
    "build_selection",
]
