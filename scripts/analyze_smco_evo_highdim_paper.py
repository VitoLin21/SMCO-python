#!/usr/bin/env python
"""SMCO-EVO high-dim paper analysis entry (Task 9 selection; Task 12 statistics).

--selection-only: global E1 implementation selection (dry-run needs no results).
--statistics: Task-12 primary table from merged/ (the provenance audit must
  pass) — per-algorithm ECDF-AUC, COCO ERT per target, bootstrap CI on the
  median log-gap, and failure rate.
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
        "--merged-dir", default=None,
        help="merged/ dir (valid_runs.csv + provenance_audit.json): canonical selection/statistics input.",
    )
    parser.add_argument(
        "--e1-manifest", dest="e1_manifest", nargs="+", default=None,
        help="Frozen E1 manifest path(s); required by canonical --selection-only --merged-dir to "
             "validate stage, run_ids and per-candidate task count (R5b).",
    )
    parser.add_argument(
        "--out-dir",
        default="result/smco-evo-paper-highdim-2026/analysis",
        help="Where to write selection.* / primary_table outputs.",
    )
    parser.add_argument("--selection-only", action="store_true", help="Run only the selection step.")
    parser.add_argument("--statistics", action="store_true", help="Compute the Task-12 primary table from merged/ (audit must pass).")
    parser.add_argument("--dry-run", action="store_true", help="No results needed; report rules + candidates.")
    parser.add_argument("--development", action="store_true", help="Allow raw --result-dir JSON (development only).")
    parser.add_argument(
        "--composite", default=None,
        help="E3 comparative composite JSON; REQUIRED for any E3-stage --statistics "
             "(review §6.3). The composite and the final merged/ are validated before "
             "any E3 statistics are produced.",
    )
    args = parser.parse_args(argv)

    if args.statistics:
        if not args.merged_dir:
            parser.error("--statistics requires --merged-dir (canonical merged/ input)")
        # P1c (review §6.3): an E3 comparative analysis must go through the
        # composite gate — no statistics without a validated 120+300=420 composite.
        is_e3 = args.composite is not None or "e3" in (args.stage or "").lower()
        if is_e3:
            if not args.composite:
                parser.error(
                    "E3 --statistics requires --composite <composite.json> "
                    "(review §6.3); E1/E2 analyses keep their original entry")
            try:
                from smco.confirmatory import enforce_e3_composite_gate
                enforce_e3_composite_gate(
                    composite_path=args.composite, merged_dir=args.merged_dir)
            except (ValueError, FileNotFoundError) as exc:
                parser.error(f"E3 composite gate failed: {exc}")
        from smco.paper_analysis import write_primary_table
        from smco.selection import selection_candidates
        algos = [c["algorithm_id"] for c in selection_candidates()]
        table = write_primary_table(args.merged_dir, args.out_dir, algos)
        print(f"wrote {args.out_dir}/primary_table.csv ({len(table)} algorithms)")
        return 0

    if not args.selection_only:
        parser.error("use --selection-only or --statistics")

    summary = build_selection(args.result_dir, out_dir=args.out_dir, dry_run=args.dry_run,
                              merged_dir=args.merged_dir, e1_manifest_paths=args.e1_manifest,
                              development=args.development)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
