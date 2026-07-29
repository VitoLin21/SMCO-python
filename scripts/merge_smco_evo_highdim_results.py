#!/usr/bin/env python
"""Merge SMCO-EVO high-dim raw outcomes into merged/ artefacts (Task 11).

Reads frozen manifests + raw outcome dirs, builds RESULT_COLUMNS rows from
outcome + manifest task at a single Python point, resolves supersedes, runs the
provenance audit, and writes merged/{all_attempts,valid_runs,missing_runs,
duplicate_runs,anytime}.csv + provenance_audit.{json,md}.

Usage:
    python scripts/merge_smco_evo_highdim_results.py \
        --manifest m1.json [m2.json ...] \
        --raw-dir raw1 [raw2 ...] \
        --merged-dir merged/

Exit code 0 if the audit passed, 2 if it failed (the analysis gate).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smco.merge_results import merge  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", nargs="+", required=True, help="Frozen manifest JSON(s).")
    parser.add_argument("--raw-dir", nargs="+", required=True, help="raw outcome dir(s).")
    parser.add_argument("--merged-dir", required=True, help="Output dir for merged/ artefacts.")
    args = parser.parse_args(argv)
    summary = merge(args.manifest, args.raw_dir, args.merged_dir)
    print(json.dumps(summary, indent=2))
    return 0 if summary["audit"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
