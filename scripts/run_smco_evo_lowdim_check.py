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
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cocoex  # noqa: E402

from smco.coco_runner import (  # noqa: E402
    aggregate_instance_summary,
    run_on_problem,
    write_run_provenance,
)
from smco.confirmatory import enforce_confirmatory  # noqa: E402
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
        f"instances:{'-'.join(str(i) for i in instances)}",
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
    """R-04: canonical E5 reads the winner from a frozen manifest; free
    --winner is development-only and must be acknowledged."""
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        if args.selection:
            sel = json.loads(Path(args.selection).read_text())
            enforce_confirmatory(manifest, selection=sel)
        return manifest["winner_algorithm"]
    if not args.winner:
        parser.error("canonical E5 requires --manifest (+ --selection); "
                     "or use --development with --winner")
    if not args.development:
        parser.error("free --winner is development-only; pass --development")
    print("WARNING [E5]: development mode — output is NOT confirmatory.", file=sys.stderr)
    return args.winner


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
    args = parser.parse_args(argv)

    if not _have_cocoex():
        print("ERROR: cocoex not installed. Install with: pip install coco-experiment", file=sys.stderr)
        return 2

    winner = _resolve_winner(args, parser)
    summary = run_lowdim(
        winner=winner, dims=args.dims, instances=args.instances,
        fe_budget_per_d=args.fe_budget_per_d, result_dir=args.result_dir)
    print(f"E5 lowdim: {summary['n_runs']} runs ({summary['winner']} vs {summary['base']}) "
          f"-> {args.result_dir}/lowdim_degradation.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
