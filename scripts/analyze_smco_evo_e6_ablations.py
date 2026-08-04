#!/usr/bin/env python
"""Dedicated Task-12 E6.1/E6.2 ablation analysis from a canonical index."""
from __future__ import annotations

import argparse
import json
import sys

from smco.e6_analysis import write_e6_analysis


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-index", required=True,
                        help="Frozen canonical_artifacts.json; raw/merged paths are not accepted.")
    parser.add_argument("--out-dir", default="result/e6-2026-07-31/analysis_e6")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = write_e6_analysis(args.canonical_index, args.out_dir,
                                   n_boot=args.n_bootstrap, figures=not args.no_figures)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({"out_dir": args.out_dir, "h4": result["h4"],
                      "n_strategy_pairs": len(result["strategy_pairwise"])}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
