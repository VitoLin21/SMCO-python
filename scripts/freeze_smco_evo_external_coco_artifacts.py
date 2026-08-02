#!/usr/bin/env python
"""Freeze a formal E4 or E5 COCO-native external evidence index.

This writes a *separate* external index.  It intentionally never edits the
already frozen high-dimensional primary canonical index.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from smco.external_canonical_artifacts import (
    build_formal_external_index, validate_formal_external_index,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=("e4", "e5"), required=True)
    parser.add_argument("--manifest", required=True, help="Frozen formal manifest JSON.")
    parser.add_argument("--merged-dir", required=True, help="Merged formal COCO outcome directory.")
    parser.add_argument("--out", required=True, help="External index output JSON (new file only).")
    args = parser.parse_args(argv)
    prefix = args.campaign
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        git_commit = ""
    index = build_formal_external_index(
        {f"{prefix}_formal_manifest": args.manifest,
         f"{prefix}_formal_merged": args.merged_dir},
        campaign=prefix, git_commit=git_commit)
    errors = validate_formal_external_index(index)
    if errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"wrote {out}; validation: PASS ({args.campaign} external supporting evidence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
