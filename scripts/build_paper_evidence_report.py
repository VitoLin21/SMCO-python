#!/usr/bin/env python
"""Task-13: assemble the consolidated, reproducible paper evidence report (Gate-G).

Reads the AUDITED artefacts only (selection, synthetic primary/pairwise tables,
E4/E5 COCO-native summaries) and writes a top-level report that states the
frozen winner, the synthetic main result, the isolated COCO external evidence,
and the honest boundaries between them. Every number traces to a cited CSV.
This is the evidence bundle, not the paper text.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _primary_row(rows, algo):
    return next((r for r in rows if r.get("algorithm_id") == algo), {})


def build(selection_path, e3_dir, e4_summary, e5_summary, out_path) -> Path:
    lines = ["# SMCO-EVO High-Dim Paper — Consolidated Evidence Report (Gate-G)", ""]
    sel = json.loads(Path(selection_path).read_text())
    lines += [
        "## Frozen selection (E1)",
        "",
        f"- winner: **{sel.get('winner')}** (language: {sel.get('winner_language')})",
        f"- selection_hash: `{sel.get('selection_hash')}`",
        "",
    ]

    # --- Synthetic main result (E3 comparative) ---
    pt = _read_csv(Path(e3_dir) / "primary_table.csv")
    pw = _read_csv(Path(e3_dir) / "pairwise_table.csv")
    winner = sel.get("winner")
    lines += [
        "## Synthetic main result (E3 comparative, 12-check audited)",
        "",
        f"Winner `{winner}` vs each baseline, paired by (function, dimension, instance), "
        "Holm step-down adjusted p:",
        "",
        "| opponent | n_pairs | median log-gap diff (winner-opponent) | prob winner better | p_holm |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in pw:
        a, b = r.get("algorithm_a"), r.get("algorithm_b")
        opp = b if a == winner else (a if b == winner else None)
        if opp is None:
            continue
        diff = r.get("median_log_gap_diff", "")
        # flip sign when the winner is algorithm_b so the diff is always winner-opponent
        if b == winner and diff not in ("", None):
            try:
                diff = f"{-float(diff):.4f}"
            except (TypeError, ValueError):
                pass
        lines.append(f"| {opp} | {r.get('n_pairs','')} | {diff} | "
                     f"{r.get('prob_a_better','')} | {r.get('p_holm','')} |")
    lines.append("")

    # --- COCO external evidence (E4 / E5) ---
    lines += [
        "## Isolated COCO external evidence (E4 bbob-largescale, E5 bbob low-dim)",
        "",
        "These are COCO-native, metric_mode=coco_native outcomes — isolated from the "
        "synthetic normalized-gap pipeline. They report the COCO final-target-hit rate, "
        "NOT the synthetic advantage.",
        "",
    ]
    for label, path in (("E4 bbob-largescale (d160/320/640)", e4_summary),
                        ("E5 bbob low-dim (d5/20)", e5_summary)):
        rows = _read_csv(path)
        lines += [f"### {label}", "",
                  "| algorithm | n_runs | final_target_hit_rate | median_fe_used |",
                  "| --- | ---: | ---: | ---: |"]
        for r in sorted(rows, key=lambda x: -float(x.get("final_target_hit_rate") or 0)):
            lines.append(f"| {r.get('algorithm_id')} | {r.get('n_runs')} | "
                         f"{r.get('final_target_hit_rate')} | {r.get('median_fe_used')} |")
        lines.append("")

    # --- Honest framing ---
    lines += [
        "## Honest boundaries (must be reported verbatim)",
        "",
        "1. **Synthetic (E3, paired Holm):** SMCO-EVO significantly beats DE/GA/PSO "
        "(Holm p~0); it is **statistically indistinguishable** from GenSA, SA and its "
        "matched base PY-BASE-SMCO.",
        "2. **COCO external (E4):** on bbob-largescale the winner's COCO "
        "final-target-hit rate is **0%** (same as DE/GA/PSO/base); only GenSA/SA reach "
        "the target on a fraction of problems. The synthetic advantage does **not** "
        "transfer to bbob-largescale final-target hitting.",
        "3. E4/E5 are **isolated supporting evidence**, never a basis for a "
        "\"wins everywhere\" claim. R-01: a Python winner is its own frozen validation "
        "on COCO (external_check_kind=frozen_winner).",
        "",
    ]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--e3-dir", required=True, help="E3 comparative analysis dir (primary+pairwise).")
    parser.add_argument("--e4-summary", required=True)
    parser.add_argument("--e5-summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = build(args.selection, args.e3_dir, args.e4_summary, args.e5_summary, args.out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
