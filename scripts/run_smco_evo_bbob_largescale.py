#!/usr/bin/env python
"""E4 bbob-largescale external benchmark (Task 10 / E4).

Runs the frozen E1 winner + matched non-EVO base + 5 strong baselines
(DE/GA/PSO/SA/GenSA) on COCO ``bbob-largescale`` (24 functions, d in
{160, 320, 640}, instances 1--5) under B_max = 1000*d FE, writing the figure-5
data CSV (per func*dim*instance*algorithm metrics + summary). Requires cocoex.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cocoex  # noqa: E402

from smco.coco_runner import (  # noqa: E402
    aggregate_instance_summary,
    run_baseline_on_problem,
    run_on_problem,
    write_run_provenance,
)
from smco.confirmatory import enforce_confirmatory  # noqa: E402
from smco.paper_contract import parse_algorithm_id  # noqa: E402

_FAM_TOKEN = {"smco": "SMCO", "smco_refine": "SMCO-REFINE", "smco_boost_refine": "SMCO-BOOST-REFINE"}
BASELINES = ("DE", "GA", "PSO", "SA", "GenSA")


def _have_cocoex() -> bool:
    try:
        import cocoex  # noqa: F401
        return True
    except ImportError:
        return False


def to_py(winner: str) -> str:
    parsed = parse_algorithm_id(winner)
    fam = _FAM_TOKEN[parsed["family"]]
    if not parsed["evolutionary"]:
        return f"PY-BASE-{fam}"
    slot = {"state_preserving": "SP", "restart": "RS"}[parsed["state_semantics"]]
    return f"PY-{slot}-{fam}-EVO"


def matched_base(winner_py: str) -> str:
    parsed = parse_algorithm_id(winner_py)
    return f"PY-BASE-{_FAM_TOKEN[parsed['family']]}"


def run_bbob_largescale(*, winner, suite, dims, instances, fe_budget_per_d,
                        result_dir, baselines=BASELINES) -> dict:
    """Run winner + matched base + N baselines over the suite; write figure-5 CSVs."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    winner_py = to_py(winner)
    base = matched_base(winner_py)
    smco_algos = [winner_py, base]
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
                         f"main claim must not rest on E4.")
        print(f"WARNING [E4]: {language_note}", file=sys.stderr)
    suite_obj = cocoex.Suite(
        suite,
        f"instances:{'-'.join(str(i) for i in instances)}",
        f"dimensions:{','.join(str(d) for d in dims)}",
    )

    rows: list[dict] = []
    for algo in smco_algos:
        observer = cocoex.Observer(suite, f"result_folder: {result_dir / 'cocoex' / algo}")
        for problem in suite_obj:
            rows.append(run_on_problem(
                problem, algorithm_id=algo,
                fe_budget=fe_budget_per_d * int(problem.dimension), observer=observer))
    for algo in baselines:
        observer = cocoex.Observer(suite, f"result_folder: {result_dir / 'cocoex' / algo}")
        for problem in suite_obj:
            rows.append(run_baseline_on_problem(
                problem, algorithm_name=algo,
                fe_budget=fe_budget_per_d * int(problem.dimension), observer=observer))

    _write_csv(result_dir / "bbob_largescale.csv", rows,
               ("function", "dimension", "instance", "algorithm_id",
                "best_observed_fvalue1", "final_target_hit", "evaluations"))
    _write_summary(result_dir / "bbob_largescale_summary.csv", rows, smco_algos + list(baselines))
    write_run_provenance(result_dir, kind="e4_bbob_largescale",
                         algorithms=smco_algos + list(baselines), winner=winner_py, base=base,
                         suite=suite, dims=dims, instances=instances,
                         fe_budget_per_d=fe_budget_per_d, original_winner=winner,
                         original_language=original_language, language_note=language_note,
                         external_check_kind=external_check_kind,
                         is_frozen_winner_validation=is_frozen_winner_validation)
    return {"n_runs": len(rows), "algorithms": smco_algos + list(baselines)}


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in fields})


def _write_summary(path, rows, algorithms):
    # A-06: aggregate over instances (target-hit rate, mean best, n_instances)
    # instead of keeping only the last instance per (function, dim).
    out, fields = aggregate_instance_summary(rows, algorithms)
    _write_csv(path, out, fields)


def _resolve_winner_baselines(args, parser):
    """R-04: canonical E4 reads algorithms from a frozen manifest; free
    --winner/--baselines is development-only and must be acknowledged."""
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        if args.selection:
            sel = json.loads(Path(args.selection).read_text())
            enforce_confirmatory(manifest, selection=sel)
        winner = manifest["winner_algorithm"]
        baselines = manifest.get("baseline_algorithms") or list(BASELINES)
        return winner, baselines
    if not args.winner:
        parser.error("canonical E4 requires --manifest (+ --selection); "
                     "or use --development with --winner")
    if not args.development:
        parser.error("free --winner/--baselines is development-only; pass "
                     "--development to acknowledge the output is not confirmatory")
    print("WARNING [E4]: development mode (free --winner/--baselines) — "
          "output is NOT confirmatory.", file=sys.stderr)
    return args.winner, args.baselines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner", default=None, help="Frozen E1 winner (development mode; canonical uses --manifest).")
    parser.add_argument("--manifest", default=None, help="Frozen confirmatory manifest: read winner/baselines (canonical E4).")
    parser.add_argument("--selection", default=None, help="selection.json to verify --manifest (canonical E4).")
    parser.add_argument("--development", action="store_true", help="Allow free --winner/--baselines; output is flagged development, not confirmatory.")
    parser.add_argument("--suite", default="bbob-largescale", help="cocoex suite name (default bbob-largescale).")
    parser.add_argument("--dims", nargs="+", type=int, default=[160, 320, 640])
    parser.add_argument("--instances", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--fe-budget-per-d", type=int, default=1000)
    parser.add_argument("--baselines", nargs="+", default=list(BASELINES))
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args(argv)

    if not _have_cocoex():
        print("ERROR: cocoex not installed. Install with: pip install coco-experiment", file=sys.stderr)
        return 2

    winner, baselines = _resolve_winner_baselines(args, parser)
    summary = run_bbob_largescale(
        winner=winner, suite=args.suite, dims=args.dims, instances=args.instances,
        fe_budget_per_d=args.fe_budget_per_d, result_dir=args.result_dir, baselines=baselines)
    print(f"E4 bbob-largescale: {summary['n_runs']} runs "
          f"({', '.join(summary['algorithms'])}) -> {args.result_dir}/bbob_largescale.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
