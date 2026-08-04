#!/usr/bin/env python
"""Analyze one validated formal COCO external artifact with native metrics only."""
from __future__ import annotations

import argparse
from smco.coco_external_analysis import write_coco_native_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-index", required=True)
    parser.add_argument("--artifact-key", required=True,
                        help="e4_formal_merged or e5_formal_merged")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    try:
        rows = write_coco_native_report(args.external_index, args.artifact_key, args.out_dir)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"wrote {args.out_dir}/coco_native_summary.csv ({len(rows)} algorithms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
