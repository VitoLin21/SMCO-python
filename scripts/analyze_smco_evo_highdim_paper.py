#!/usr/bin/env python
"""SMCO-EVO high-dim paper analysis entry (Task 9 selection; Task 12 statistics).

Task 9 implements only the ``--selection-only`` path (global E1 implementation
selection, with a dry-run that needs no results). The full statistics / figures
(ECDF, ERT, hierarchical bootstrap, dimension trends) arrive in Task 12.
"""

from __future__ import annotations

import argparse
import json
import sys

from smco.selection import build_selection


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="e1-development", help="Analysis stage.")
    parser.add_argument("--result-dir", default=None, help="raw/ directory with result payloads.")
    parser.add_argument(
        "--out-dir",
        default="result/smco-evo-paper-highdim-2026/analysis",
        help="Where to write selection.* outputs.",
    )
    parser.add_argument("--selection-only", action="store_true", help="Run only the selection step.")
    parser.add_argument("--dry-run", action="store_true", help="No results needed; report rules + candidates.")
    parser.add_argument("--merged-dir", default=None, help="merged/ dir (valid_runs.csv + provenance_audit.json): canonical selection input.")
    parser.add_argument("--development", action="store_true", help="Allow raw --result-dir JSON (development only).")
    args = parser.parse_args(argv)

    if not args.selection_only:
        parser.error(
            "only --selection-only is implemented in Task 9; "
            "full statistics (ECDF/ERT/bootstrap/figures) is Task 12 (R-03)"
        )

    summary = build_selection(args.result_dir, out_dir=args.out_dir, dry_run=args.dry_run,
                              merged_dir=args.merged_dir, development=args.development)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
