#!/usr/bin/env python
"""Generate frozen E2 H1/H2/H3 Task-12 analysis artifacts."""
from __future__ import annotations

import argparse

from smco.e2_hypotheses import DEFAULT_BOOTSTRAPS, DEFAULT_SEED, run_e2_hypotheses


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-index", required=True,
                        help="Frozen canonical_artifacts.json; bare merged paths are intentionally unsupported.")
    parser.add_argument("--instance-index", default="instances_index_confirmatory.json")
    parser.add_argument("--out-dir", default="result/e2-2026-07-31/analysis_e2")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.n_boot < 1:
        parser.error("--n-boot must be positive")
    run_e2_hypotheses(canonical_index_path=args.canonical_index, out_dir=args.out_dir,
                      instance_index_path=args.instance_index, n_boot=args.n_boot, seed=args.seed)
    print(f"wrote E2 hypotheses under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
