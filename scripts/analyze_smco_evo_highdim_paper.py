#!/usr/bin/env python
"""SMCO-EVO high-dim paper analysis entry (Task 9 selection; Task 12 statistics).

--selection-only: global E1 implementation selection (dry-run needs no results).
--statistics: Task-12 primary table from merged/ (the provenance audit must
  pass) — per-algorithm ECDF-AUC, COCO ERT per target, bootstrap CI on the
  median log-gap, and failure rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from smco.selection import build_selection

# Baseline algorithms only present in E3 comparative data. Used to detect E3
# data handed in via a bare --merged-dir (which must be rejected — E3 needs the
# composite gate, review P0).
_E3_BASELINE_ALGOS = {"DE", "GA", "PSO", "SA", "GenSA"}


def _merged_dir_is_e3(merged_dir) -> bool:
    """True if a merged dir's valid_runs.csv contains E3 baseline algorithms."""
    path = Path(merged_dir) / "valid_runs.csv"
    if not path.exists():
        return False
    with open(path, newline="") as handle:
        algos = {row.get("algorithm_id") for row in csv.DictReader(handle)}
    return bool(algos & _E3_BASELINE_ALGOS)


def _resolve_statistics_inputs(args, parser):
    """Resolve (merged_dir, algorithms) for Task-12 statistics (review P0).

    Preferred: ``--canonical-index`` + ``--artifact-key`` — the merged dir (and,
    for E3, the composite path) come from the validated index, never a --stage
    string or an arbitrary path. E3 statistics read the algorithm set from the
    validated composite and force the composite gate.

    Bare ``--merged-dir`` is development-only.  Confirmatory Task 12 always
    resolves through the index; this keeps every table tied to the frozen
    artifact contract rather than an arbitrary directory.
    """
    from smco.selection import selection_candidates

    if args.canonical_index:
        if not args.artifact_key:
            parser.error("--canonical-index requires --artifact-key")
        try:
            from smco.canonical_artifacts import (
                resolve_analysis_target, validate_canonical_index)
            index = json.loads(Path(args.canonical_index).read_text())
            errs = validate_canonical_index(index)
            if errs:
                parser.error("canonical index invalid:\n  " + "\n  ".join(errs))
            target = resolve_analysis_target(index, args.artifact_key)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(f"canonical index resolve failed: {exc}")
        merged_dir = target["merged_dir"]
        analysis_kind = target.get("analysis_kind")
        if analysis_kind == "comparative":
            from smco.confirmatory import enforce_e3_composite_gate
            try:
                enforce_e3_composite_gate(
                    composite_path=target["composite_path"], merged_dir=merged_dir)
            except (ValueError, FileNotFoundError) as exc:
                parser.error(f"E3 composite gate failed: {exc}")
            algos = json.loads(Path(target["composite_path"]).read_text())["algorithms"]
        elif analysis_kind == "selection_matrix":
            algos = [c["algorithm_id"] for c in selection_candidates()]
        elif analysis_kind == "winner_vs_base":
            manifest_path = target.get("source_manifest_path")
            if not manifest_path:
                parser.error("winner_vs_base analysis target has no frozen source manifest")
            try:
                manifest = json.loads(Path(manifest_path).read_text())
                algos = [manifest["winner_algorithm"], manifest["matched_base_algorithm"]]
            except (KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
                parser.error(f"cannot resolve frozen E2 winner/base algorithms: {exc}")
            if len(set(algos)) != 2 or not all(isinstance(a, str) and a for a in algos):
                parser.error("frozen E2 manifest must name exactly two distinct algorithms")
            actual_algos = _algorithms_in_merged_dir(merged_dir)
            if actual_algos != set(algos):
                parser.error(
                    "E2 merged algorithms do not exactly match its frozen winner/base manifest")
        elif analysis_kind in {"baseline_only", "strategy_ablation", "start_count_ablation"}:
            parser.error(
                f"artifact {target['key']!r} has analysis_kind={analysis_kind!r}; "
                "it requires its dedicated Task-12 analysis and cannot use primary_table")
        else:
            parser.error(f"artifact {target['key']!r} has no supported analysis_kind")
        return merged_dir, algos, analysis_kind

    if not args.development:
        parser.error(
            "confirmatory --statistics requires --canonical-index + --artifact-key; "
            "bare --merged-dir is development-only (pass --development)")
    if not args.merged_dir:
        parser.error(
            "--statistics --development requires --merged-dir")
    merged_dir = args.merged_dir
    # Defense-in-depth (review P0): default stage + E3 merged must still reject —
    # E3 data cannot bypass the composite gate via a bare --merged-dir.
    if _merged_dir_is_e3(merged_dir):
        if not args.composite:
            parser.error(
                "merged dir contains E3 comparative data (baselines); E3 statistics "
                "require --canonical-index --artifact-key e3_composite_merged (or "
                "--composite) so the 120+300=420 composite gate runs")
        from smco.confirmatory import enforce_e3_composite_gate
        try:
            enforce_e3_composite_gate(composite_path=args.composite, merged_dir=merged_dir)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(f"E3 composite gate failed: {exc}")
        algos = json.loads(Path(args.composite).read_text())["algorithms"]
        analysis_kind = "comparative"
    else:
        algos = [c["algorithm_id"] for c in selection_candidates()]
        analysis_kind = "selection_matrix"
    return merged_dir, algos, analysis_kind


def _algorithms_in_merged_dir(merged_dir) -> set[str]:
    """Read the observed algorithm IDs for a contract-bound algorithm check."""
    with open(Path(merged_dir) / "valid_runs.csv", newline="") as handle:
        return {row.get("algorithm_id") for row in csv.DictReader(handle)
                if row.get("algorithm_id")}


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
        help="E3 comparative composite JSON (with --merged-dir only); the composite and "
             "final merged/ are validated before any E3 statistics.",
    )
    parser.add_argument(
        "--canonical-index", dest="canonical_index", default=None,
        help="canonical_artifacts.json (review P0/§8): Task-12 statistics resolve their "
             "merged dir (and, for E3, the composite) from the index + --artifact-key, "
             "NOT from an arbitrary --merged-dir or a --stage string.",
    )
    parser.add_argument(
        "--artifact-key", dest="artifact_key", default=None,
        help="artifact key in the canonical index (e.g. e3_composite_merged, e1_merged). "
             "Required with --canonical-index.",
    )
    args = parser.parse_args(argv)

    if args.statistics:
        merged_dir, algos, analysis_kind = _resolve_statistics_inputs(args, parser)
        from smco.paper_analysis import write_pairwise_table, write_primary_table
        table = write_primary_table(merged_dir, args.out_dir, algos)
        print(f"wrote {args.out_dir}/primary_table.csv ({len(table)} algorithms)")
        # comparative analysis also emits the pairwise Holm/superiority table
        # (Task 12): is the proposed method significantly different from each
        # baseline / the matched base, including where a baseline beats it.
        if analysis_kind == "comparative":
            pairs = write_pairwise_table(merged_dir, args.out_dir, algos)
            print(f"wrote {args.out_dir}/pairwise_table.csv ({len(pairs)} pairs)")
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
