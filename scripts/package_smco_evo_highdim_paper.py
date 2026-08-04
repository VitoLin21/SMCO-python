#!/usr/bin/env python
"""Task-13 packaging: assemble the reproducible paper result bundle (R-03).

Reads the analysis outputs (selection.json, primary_table.csv) and writes a
report.md whose every number traces back to those files. The bundle is the
Gate-G artifact.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def build_report(analysis_dir, out_path) -> Path:
    analysis_dir = Path(analysis_dir)
    lines = ["# SMCO-EVO High-Dim Paper — Result Report", ""]

    sel_path = analysis_dir / "selection.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text())
        lines.append("## Selection")
        lines.append("")
        lines.append(f"- winner: **{sel.get('winner')}** (language: {sel.get('winner_language')})")
        lines.append(f"- selection_hash: `{sel.get('selection_hash')}`")
        lines.append(f"- winner_config_hash: `{sel.get('winner_config_hash')}`")
        lines.append(f"- n_results: {sel.get('n_results')}; results_hash: `{sel.get('results_hash')}`")
        lines.append("")

    pt_path = analysis_dir / "primary_table.csv"
    if pt_path.exists():
        rows = list(csv.DictReader(open(pt_path)))
        lines.append("## Primary table (from primary_table.csv)")
        lines.append("")
        lines.append("| algorithm_id | n_runs | ecdf_auc | median_log_gap | failure_rate | ert_1e-1 | ert_1e-5 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in rows:
            lines.append(
                f"| {r['algorithm_id']} | {r['n_runs']} | {r.get('ecdf_auc','')} | "
                f"{r.get('median_log_gap','')} | {r.get('failure_rate','')} | "
                f"{r.get('ert_1e-1','')} | {r.get('ert_1e-5','')} |"
            )
        lines.append("")

    pw_path = analysis_dir / "pairwise_table.csv"
    if pw_path.exists():
        pairs = list(csv.DictReader(open(pw_path)))
        lines.append("## Pairwise comparison (from pairwise_table.csv)")
        lines.append("")
        lines.append("Holm step-down adjusted p (paired by function x dimension x instance;")
        lines.append("median_log_gap_diff < 0 means algorithm_a reaches a lower gap).")
        lines.append("")
        lines.append("| algorithm_a | algorithm_b | n_pairs | median_log_gap_diff | prob_a_better | p_value | p_holm |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in pairs:
            lines.append(
                f"| {r.get('algorithm_a','')} | {r.get('algorithm_b','')} | "
                f"{r.get('n_pairs','')} | {r.get('median_log_gap_diff','')} | "
                f"{r.get('prob_a_better','')} | {r.get('p_value','')} | "
                f"{r.get('p_holm','')} |"
            )
        lines.append("")

    figures = sorted(analysis_dir.glob("ecdf_target_*.png"))
    if figures:
        lines.append("## ECDF figures")
        lines.append("")
        for fig in figures:
            rel = fig.relative_to(analysis_dir) if fig.is_relative_to(analysis_dir) else fig
            lines.append(f"![{fig.stem}]({rel})")
        lines.append("")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--out", default=None, help="report.md path (default: <analysis-dir>/report.md).")
    args = parser.parse_args(argv)
    out = args.out or str(Path(args.analysis_dir) / "report.md")
    build_report(args.analysis_dir, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
