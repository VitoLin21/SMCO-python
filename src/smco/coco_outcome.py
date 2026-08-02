"""COCO external-validation task-level outcome for E4/E5 (review §9, P3).

E4/E5 run the frozen winner + matched base + baselines on COCO (bbob-largescale /
bbob) as an **external** benchmark check. Their outcomes are NOT synthetic
high-dim instances — there is no ``instance_artifact_dir`` / transform hash.
Instead each task carries **benchmark provenance** (COCO suite, function,
dimension, instance, problem id, cocoex/cocopp versions, the known ``f_opt``,
the raw ``best_observed_fvalue1`` and ``evaluations``) plus the COCO-native
``final_target_hit`` truth flag.

To flow through the same merge/audit/analysis chain as E1/E2/E3, the outcome also
derives ``normalized_gap`` and ``target_hit_fe_<tau>`` from the full best-so-far
trace, using the SAME relative convention as the synthetic contract
(``f_opt + tau * (initial_ref - f_opt)``). Nothing is faked: the COCO-native
fields are preserved verbatim and the audit layer explicitly recognises the COCO
suite and validates benchmark provenance instead of instance hashes.
"""
from __future__ import annotations

from .paper_contract import NONE_TOKEN, SCHEMA_VERSION
from .selection import TARGETS

# COCO benchmark suites recognised by the audit layer (no synthetic instance
# contract applies to these).
COCO_SUITES = frozenset({"bbob-largescale", "bbob"})


def derive_gap_and_targets(best_trace, *, f_opt, initial_ref, fe_budget,
                           targets=TARGETS):
    """From a minimisation best-so-far ``best_trace`` = [(fe, best_value), ...]
    derive ``normalized_gap`` and ``{target: first_fe_hit}``.

    Targets are relative, identical to the synthetic contract:
    ``f_opt + tau * (initial_ref - f_opt)``. A target not reached by
    ``fe_budget`` maps to ``None`` (right-censored). ``initial_ref`` is the
    reference gap span (typically the best at the first evaluation, or the
    median of the start-point values).
    """
    gap_span = max(initial_ref - f_opt, 1e-12)
    final_best = best_trace[-1][1] if best_trace else None
    normalized_gap = (
        max(final_best - f_opt, 1e-12) / gap_span if final_best is not None else None)
    target_fe: dict[str, int | None] = {}
    for tau in targets:
        threshold = f_opt + float(tau) * gap_span
        hit = None
        for fe, best in best_trace:
            if best <= threshold:
                hit = int(fe)
                break
        target_fe[tau] = hit
    return normalized_gap, target_fe


def build_coco_outcome(
    task: dict,
    *,
    best_observed_fvalue1: float,
    evaluations: int,
    final_target_hit: bool,
    best_trace,
    f_opt: float,
    initial_ref: float,
    fe_budget: int,
    suite: str,
    problem_id: str,
    cocoex_version: str,
    cocopp_version: str | None,
    machine_id: str,
    git_commit: str,
    environment_hash: str,
    status: str = "success",
    failure_reason: str = NONE_TOKEN,
    wall_time_sec: float = 0.0,
    peak_memory_mb: float = 0.0,
    targets=TARGETS,
) -> dict:
    """Assemble one COCO external-validation task-level outcome payload.

    Preserves the COCO-native fields verbatim and derives the synthetic-style
    ``normalized_gap`` / ``target_hit_fe_<tau>`` / ``anytime`` from the
    best-so-far trace so the unified merge/audit/analysis chain can consume it.
    """
    normalized_gap, target_fe = derive_gap_and_targets(
        best_trace, f_opt=f_opt, initial_ref=initial_ref,
        fe_budget=fe_budget, targets=targets)
    anytime = [
        {"checkpoint_fe": int(fe), "best_value": float(best)}
        for fe, best in best_trace
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": task["run_id"],
        "status": status,
        "failure_reason": failure_reason,
        "best_value": float(best_observed_fvalue1),
        "known_optimum": float(f_opt),
        "normalized_gap": normalized_gap,
        "fe_used": int(evaluations),
        "fe_budget": int(fe_budget),
        "final_target_hit": bool(final_target_hit),
        "target_hit_fe": {tau: target_fe[tau] for tau in targets},
        "anytime": anytime,
        "wall_time_sec": float(wall_time_sec),
        "peak_memory_mb": float(peak_memory_mb),
        # benchmark provenance (replaces synthetic instance_artifact_dir / hashes)
        "benchmark": {
            "kind": "coco",
            "suite": suite,
            "function": task.get("function"),
            "dimension": int(task["dimension"]),
            "instance": int(task["instance"]),
            "problem_id": problem_id,
            "f_opt": float(f_opt),
            "best_observed_fvalue1": float(best_observed_fvalue1),
            "evaluations": int(evaluations),
            "cocoex_version": cocoex_version,
            "cocopp_version": cocopp_version,
        },
        "machine_id": machine_id,
        "git_commit": git_commit,
        "environment_hash": environment_hash,
        "task": task,
        "supersedes_run_id": NONE_TOKEN,
    }
    return payload


def coco_benchmark_provenance(outcome: dict) -> dict | None:
    """Return the COCO benchmark provenance block of an outcome, or None."""
    bench = outcome.get("benchmark")
    if isinstance(bench, dict) and bench.get("kind") == "coco":
        return bench
    return None


def coco_outcome_errors(outcome: dict) -> list[str]:
    """Validate a COCO outcome's benchmark provenance contract (review P3).

    The audit layer calls this for COCO-suite outcomes instead of the synthetic
    instance_artifact_dir / transform / start_points hash checks. Empty == ok.
    """
    errors: list[str] = []
    bench = coco_benchmark_provenance(outcome)
    if bench is None:
        return ["outcome has no COCO benchmark provenance block"]
    for field in ("suite", "function", "dimension", "instance", "problem_id",
                  "f_opt", "best_observed_fvalue1", "evaluations", "cocoex_version"):
        if field not in bench:
            errors.append(f"benchmark provenance missing {field}")
    if bench.get("suite") not in COCO_SUITES:
        errors.append(f"benchmark suite {bench.get('suite')!r} not in {sorted(COCO_SUITES)}")
    if not bench.get("problem_id"):
        errors.append("benchmark problem_id empty")
    if not outcome.get("git_commit") or not outcome.get("environment_hash") \
            or not outcome.get("machine_id"):
        errors.append("outcome missing machine_id/git_commit/environment_hash provenance")
    return errors


__all__ = [
    "COCO_SUITES",
    "derive_gap_and_targets",
    "build_coco_outcome",
    "coco_benchmark_provenance",
    "coco_outcome_errors",
]
