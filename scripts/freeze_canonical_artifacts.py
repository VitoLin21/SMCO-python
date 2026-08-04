#!/usr/bin/env python
"""Freeze the canonical artifact index for the SMCO-EVO high-dim paper (review §8).

Reproducibly builds result/smco-evo-paper-highdim-2026/canonical_artifacts.json
from the fixed contract (smco.canonical_artifacts.CANONICAL_CONTRACT) + the
known result/ paths, then validates it. Task 12/13 read paths ONLY through this
index + an artifact key (review P0).

Usage:
  python scripts/freeze_canonical_artifacts.py [--root result/e*...] [--out PATH]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from smco.canonical_artifacts import (
    CANONICAL_CONTRACT,
    build_canonical_index,
    validate_canonical_index,
)


def _paths(root: str) -> dict:
    return {
        "e1_merged": f"{root}/e1-2026-07-30/merged_v2",
        "e1_selection": f"{root}/e1-2026-07-30/selection_v2/selection.json",
        "e2_manifest": f"{root}/e1-2026-07-30/confirmatory/e2_factorial_highdim__synthetic_highdim.json",
        "e2_merged": f"{root}/e2-2026-07-31/merged_v2",
        "e3_baseline_component_manifest": f"{root}/e3-2026-07-31/e3_baseline_component__synthetic_highdim.json",
        "e3_baseline_merged": f"{root}/e3-2026-07-31/merged_baseline_v2",
        "e3_composite": f"{root}/e3-2026-07-31/e3_comparative_composite.json",
        "e3_composite_merged": f"{root}/e3-2026-07-31/merged_composite",
        "e6_strategy_merged": f"{root}/e6-2026-07-31/strategy/merged_v2",
        "e6_start_count_manifest": f"{root}/e6-2026-07-31/start_count/e6_ablations__synthetic_highdim.json",
        "e6_start_count_merged": f"{root}/e6-2026-07-31/start_count/merged_v2",
        "e4_dev": f"{root}/e4-2026-07-31/bbob_largescale_all.csv",
        "e5_dev": f"{root}/e5-2026-07-31/raw/lowdim_degradation.csv",
        "e6_schedule": f"{root}/e6-2026-07-31/schedule",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="result", help="Result tree root.")
    parser.add_argument("--out", default="result/smco-evo-paper-highdim-2026/canonical_artifacts.json",
                        help="Output index path.")
    args = parser.parse_args(argv)

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        git_commit = ""

    index = build_canonical_index(_paths(args.root), git_commit=git_commit)
    errors = validate_canonical_index(index)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps(index, indent=2, ensure_ascii=False))

    missing = [a["key"] for a in index["artifacts"] if a.get("missing")]
    print(f"wrote {out} ({len(index['artifacts'])} artifacts, git={git_commit[:12]})")
    if missing:
        print(f"WARNING missing artifacts: {missing}", file=sys.stderr)
    if errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 2
    print("validation: PASS (all required artifacts present, hashes + 12-check audit intact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
