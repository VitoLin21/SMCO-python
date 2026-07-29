#!/usr/bin/env python
"""E5 low-dimensional non-degradation check (Task 10). REQUIRES cocoex.

Checks that the high-dimensional E1 winner does not systematically degrade at
low dimension: COCO ``bbob`` (all 24 functions, d in {5, 20}, official
instances 1--5), winner vs matched non-EVO base only, B_max = 2000*d FE
(experiment plan, E5). Low-dim results go to the supplement; unless severely
degraded they do not overturn the high-dim winner.

Like run_smco_evo_bbob_largescale.py this is a contract skeleton gated on the
optional ``cocoex`` dependency; it exits with code 2 when cocoex is absent.
"""

from __future__ import annotations

import argparse
import sys


def _have_cocoex() -> bool:
    try:
        import cocoex  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner", required=True, help="Frozen E1 winner algorithm_id.")
    parser.add_argument("--dims", nargs="+", type=int, default=[5, 20])
    parser.add_argument("--instances", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--fe-budget-per-d", type=int, default=2000)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args(argv)

    if not _have_cocoex():
        print(
            "ERROR: cocoex not installed. Install with: pip install cocoex cocopp\n"
            "Then implement bbob suite dispatch (winner vs matched base only).",
            file=sys.stderr,
        )
        return 2

    # TODO(cocoex): iterate bbob suite; run winner + matched base per problem
    # under fe_budget_per_d*dim; emit supplement low-dim degradation table.
    raise NotImplementedError("cocoex present but low-dim dispatch not yet implemented")


if __name__ == "__main__":
    sys.exit(main())
