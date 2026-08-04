#!/usr/bin/env python
"""E5 low-dimensional non-degradation check (Task 10 / E5).

Runs the frozen E1 winner + matched non-EVO base on COCO ``bbob``
(24 functions, d in {5, 20}, official instances 1--5) under B_max = 2000*d FE,
and writes a supplement CSV (per func*dim*instance*algorithm metrics +
winner-vs-base summary). Unless severely degraded, low-dim results do not
overturn the high-dim winner. Requires cocoex (``pip install coco-experiment``).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smco.coco_runner import (  # noqa: E402
    aggregate_instance_summary,
    run_on_problem,
    write_run_provenance,
)
from smco.confirmatory import (  # noqa: E402
    E5_DIMENSIONS,
    confirmatory_coco_contract,
    confirmatory_run_matrix,
    enforce_confirmatory,
)
from smco.paper_contract import parse_algorithm_id  # noqa: E402

_FAM_TOKEN = {"smco": "SMCO", "smco_refine": "SMCO-REFINE", "smco_boost_refine": "SMCO-BOOST-REFINE"}


def _have_cocoex() -> bool:
    try:
        import cocoex  # noqa: F401
        return True
    except ImportError:
        return False


def to_py(winner: str) -> str:
    """Normalise an R-winner to its Py equivalent (E5 runs Python cocoex)."""
    parsed = parse_algorithm_id(winner)
    fam = _FAM_TOKEN[parsed["family"]]
    if not parsed["evolutionary"]:
        return f"PY-BASE-{fam}"
    slot = {"state_preserving": "SP", "restart": "RS"}[parsed["state_semantics"]]
    return f"PY-{slot}-{fam}-EVO"


def matched_base(winner_py: str) -> str:
    parsed = parse_algorithm_id(winner_py)
    return f"PY-BASE-{_FAM_TOKEN[parsed['family']]}"


def run_lowdim(*, winner, dims, instances, fe_budget_per_d, result_dir) -> dict:
    """Run winner + matched base over the bbob suite; write supplement CSVs."""
    import cocoex  # local: keeps the resolver importable without cocoex (R2b tests)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    winner_py = to_py(winner)
    base = matched_base(winner_py)
    # R-01: never silently swap an R winner for its Py equivalent on COCO. A Py
    # winner is the frozen winner's own external validation; an R winner is only
    # a "Python port external check" and must not back its main claim.
    original_language = parse_algorithm_id(winner)["language"]
    language_note = None
    external_check_kind = "frozen_winner"
    is_frozen_winner_validation = True
    if original_language != "python":
        external_check_kind = "python_port_external"
        is_frozen_winner_validation = False
        language_note = (f"Python port external check: {winner_py!r} is the Py equivalent "
                         f"of the frozen {original_language} winner, run on COCO because "
                         f"{original_language} cocoex is unavailable. This is NOT the frozen "
                         f"winner's own external validation; the {original_language} winner's "
                         f"main claim must not rest on E5.")
        print(f"WARNING [E5]: {language_note}", file=sys.stderr)
    suite = cocoex.Suite(
        "bbob",
        f"instances:{','.join(str(i) for i in instances)}",
        f"dimensions:{','.join(str(d) for d in dims)}",
    )

    rows: list[dict] = []
    for algo in [winner_py, base]:
        observer = cocoex.Observer("bbob", f"result_folder: {result_dir / 'cocoex' / algo}")
        for problem in suite:
            rows.append(run_on_problem(
                problem, algorithm_id=algo,
                fe_budget=fe_budget_per_d * int(problem.dimension),
                observer=observer,
            ))

    _write_csv(result_dir / "lowdim_degradation.csv", rows,
               ("function", "dimension", "instance", "algorithm_id",
                "best_observed_fvalue1", "final_target_hit", "evaluations"))
    _write_summary(result_dir / "lowdim_summary.csv", rows, winner_py, base)
    write_run_provenance(result_dir, kind="e5_lowdim_check", algorithms=[winner_py, base],
                         winner=winner_py, base=base, suite="bbob",
                         dims=dims, instances=instances, fe_budget_per_d=fe_budget_per_d,
                         original_winner=winner, original_language=original_language,
                         language_note=language_note, external_check_kind=external_check_kind,
                         is_frozen_winner_validation=is_frozen_winner_validation)
    return {"n_runs": len(rows), "winner": winner_py, "base": base}


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in fields})


def _write_summary(path, rows, winner, base):
    # A-06: aggregate over instances instead of keeping only the last one.
    out, fields = aggregate_instance_summary(rows, [winner, base])
    _write_csv(path, out, fields)


def _resolve_winner(args, parser):
    """R2b: canonical E5 requires --manifest AND --selection, locks the stage
    (e5_lowdim_check) + suite (bbob) and reads the run matrix ONLY from the
    manifest. Free --winner/--dims/... is development-only. Returns a resolved dict.
    """
    if args.manifest:
        if not args.selection:
            parser.error("canonical E5 requires --manifest AND --selection; "
                         "a manifest without its frozen selection is not confirmatory")
        manifest = json.loads(Path(args.manifest).read_text())
        sel = json.loads(Path(args.selection).read_text())
        enforce_confirmatory(manifest, selection=sel)
        matrix = confirmatory_run_matrix(
            manifest, expected_stage="e5_lowdim_check", expected_suite="bbob",
            expected_fe_budget_per_d=2000)
        winner = manifest["winner_algorithm"]
        base = manifest["matched_base_algorithm"]
        # R7c: the manifest must be the full winner+base x 24-function x {5,20} x
        # 5-instance E5 matrix (480 tasks), not a partial grid.
        confirmatory_coco_contract(
            manifest, expected_algos={winner, base}, expected_dims=E5_DIMENSIONS)
        instances = list(range(1, matrix["n_instances"] + 1))
        print(f"[E5] canonical: matrix from manifest dims={matrix['dims']} "
              f"instances={instances} fe_budget_per_d={matrix['fe_budget_per_d']} "
              f"(CLI --dims/--instances/--fe-budget-per-d ignored)", file=sys.stderr)
        return {"winner": winner, "suite": matrix["suite"], "dims": matrix["dims"],
                "instances": instances, "fe_budget_per_d": matrix["fe_budget_per_d"]}
    if not args.winner:
        parser.error("canonical E5 requires --manifest + --selection; "
                     "or use --development with --winner")
    if not args.development:
        parser.error("free --winner is development-only; pass --development")
    print("WARNING [E5]: development mode — output is NOT confirmatory.", file=sys.stderr)
    return {"winner": args.winner, "suite": "bbob", "dims": args.dims,
            "instances": args.instances, "fe_budget_per_d": args.fe_budget_per_d}


def _default_git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def _default_environment_hash() -> str:
    import platform
    payload = {"python": platform.python_version(), "numpy": np.__version__,
               "platform": platform.platform()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _run_e5_shard(args, parser) -> int:
    """Task-level E5 sharding mode (review P3/P5): run a manifest run_id shard on
    cocoex (suite bbob), emitting one COCO outcome JSON per run_id. Reuses the
    shared COCO outcome infrastructure (same as E4). No aggregate CSV."""
    if not args.manifest or not args.selection:
        parser.error("task-level --only-run-ids requires --manifest AND --selection")
    resolved = _resolve_winner(args, parser)  # validates manifest+selection+matrix+contract
    from smco.experiment_manifests import load_manifest, verify_manifest
    manifest = load_manifest(args.manifest)
    verify_manifest(manifest)
    wanted = set(args.only_run_ids)
    tasks = [t for t in manifest["tasks"] if t["run_id"] in wanted]
    missing = wanted - {t["run_id"] for t in tasks}
    if missing:
        parser.error(f"run_ids not in manifest: {sorted(missing)[:5]}")
    import cocoex
    suite_obj = cocoex.Suite(
        resolved["suite"],
        f"instances:{','.join(str(i) for i in resolved['instances'])}",
        f"dimensions:{','.join(str(d) for d in resolved['dims'])}",
    )
    from smco.coco_runner import dispatch_e4_tasks
    statuses = dispatch_e4_tasks(
        tasks, suite_obj, result_dir=args.result_dir, suite=resolved["suite"],
        machine_id=args.machine_id if args.machine_id is not None else socket.gethostname(),
        git_commit=args.git_commit if args.git_commit is not None else _default_git_commit(),
        environment_hash=(args.environment_hash if args.environment_hash is not None
                          else _default_environment_hash()))
    print(json.dumps(statuses, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner", default=None, help="Frozen E1 winner (development mode; canonical uses --manifest).")
    parser.add_argument("--manifest", default=None, help="Frozen confirmatory manifest: read winner (canonical E5).")
    parser.add_argument("--selection", default=None, help="selection.json to verify --manifest (canonical E5).")
    parser.add_argument("--development", action="store_true", help="Allow free --winner; output is flagged development, not confirmatory.")
    parser.add_argument("--dims", nargs="+", type=int, default=[5, 20])
    parser.add_argument("--instances", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--fe-budget-per-d", type=int, default=2000)
    parser.add_argument("--result-dir", required=True)
    # task-level sharding mode (review P3/P5): one COCO outcome JSON per run_id.
    parser.add_argument("--only-run-ids", dest="only_run_ids", nargs="+", default=None,
                        help="Task-level mode: run only these manifest run_ids (shard). "
                             "Emits <result-dir>/<run_id>.json per task instead of aggregate CSV.")
    parser.add_argument("--machine-id", default=None, help="Defaults to hostname.")
    parser.add_argument("--git-commit", default=None, help="Defaults to current HEAD.")
    parser.add_argument("--environment-hash", default=None, help="Defaults to a py/numpy hash.")
    args = parser.parse_args(argv)

    if not _have_cocoex():
        print("ERROR: cocoex not installed. Install with: pip install coco-experiment", file=sys.stderr)
        return 2

    if args.only_run_ids:
        return _run_e5_shard(args, parser)

    resolved = _resolve_winner(args, parser)
    summary = run_lowdim(
        winner=resolved["winner"], dims=resolved["dims"], instances=resolved["instances"],
        fe_budget_per_d=resolved["fe_budget_per_d"], result_dir=args.result_dir)
    print(f"E5 lowdim: {summary['n_runs']} runs ({summary['winner']} vs {summary['base']}) "
          f"-> {args.result_dir}/lowdim_degradation.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
