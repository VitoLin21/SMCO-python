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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cocoex  # noqa: E402

from smco.coco_runner import run_baseline_on_problem, run_on_problem  # noqa: E402
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
    return {"n_runs": len(rows), "algorithms": smco_algos + list(baselines)}


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in fields})


def _write_summary(path, rows, algorithms):
    by_key: dict[tuple, dict] = {}
    for r in rows:
        key = (int(r["function"]), int(r["dimension"]))
        by_key.setdefault(key, {})[r["algorithm_id"]] = r
    out = []
    for (func, dim), algos in sorted(by_key.items()):
        row = {"function": func, "dimension": dim}
        for algo in algorithms:
            rec = algos.get(algo)
            row[f"{algo}_target_hit"] = rec["final_target_hit"] if rec else ""
            row[f"{algo}_best"] = rec["best_observed_fvalue1"] if rec else ""
        out.append(row)
    fields = (["function", "dimension"]
              + [f"{a}_target_hit" for a in algorithms]
              + [f"{a}_best" for a in algorithms])
    _write_csv(path, out, fields)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner", required=True, help="Frozen E1 winner algorithm_id (Py or R; R auto-converted).")
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

    summary = run_bbob_largescale(
        winner=args.winner, suite=args.suite, dims=args.dims, instances=args.instances,
        fe_budget_per_d=args.fe_budget_per_d, result_dir=args.result_dir, baselines=args.baselines)
    print(f"E4 bbob-largescale: {summary['n_runs']} runs "
          f"({', '.join(summary['algorithms'])}) -> {args.result_dir}/bbob_largescale.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
