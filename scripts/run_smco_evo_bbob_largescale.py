#!/usr/bin/env python
"""E4 BBOB large-scale runner (Task 10). REQUIRES cocoex (optional dependency).

Runs the frozen E1 winner + matched non-EVO base + 3 representative strong
baselines on COCO ``bbob-largescale`` (24 functions, d in {160, 320, 640},
official instances 1--5) under B_max = 1000*d FE, n_starts=8 for the SMCO
family (experiment plan, E4). COCO's own target / ERT / ECDF definitions feed
the external benchmark figure (plan section 8, figure 5).

``cocoex`` / ``cocopp`` are OPTIONAL dependencies (plan section 12). This file
is a contract skeleton: it checks availability and pins the CLI; the Suite/
Problem dispatch + FE-budget observer are filled in once cocoex is installed
in the run environment (e.g. a fleet node). Until then it exits with code 2
rather than silently no-op'ing.
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
    parser.add_argument("--dims", nargs="+", type=int, default=[160, 320, 640])
    parser.add_argument("--instances", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--fe-budget-per-d", type=int, default=1000)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--selection", default=None, help="selection.json for --confirmatory.")
    parser.add_argument("--confirmatory", action="store_true")
    args = parser.parse_args(argv)

    if not _have_cocoex():
        print(
            "ERROR: cocoex not installed. Install with: pip install cocoex cocopp\n"
            "Then implement Suite/Problem dispatch (winner+base+3 baselines under\n"
            "fe_budget_per_d*dim with the FE observer, writing COCO-format results).",
            file=sys.stderr,
        )
        return 2

    # TODO(cocoex): iterate bbob-largescale suite; for each problem run the
    # winner, its matched base, and 3 frozen strong baselines under the shared
    # FE budget; record COCO targets/ERT/ECDF via cocopp.
    raise NotImplementedError("cocoex present but bbob-largescale dispatch not yet implemented")


if __name__ == "__main__":
    sys.exit(main())
