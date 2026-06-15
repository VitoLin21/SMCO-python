#!/usr/bin/env python3
"""Comprehensive seed sensitivity analysis for SMCO and SMCO-EVO.

Covers:
  - Paper Table 1: Rastrigin/Ackley/Griewank/Michalewicz d=2, max+min
  - Paper Table 2: Rastrigin/Ackley/Griewank/Michalewicz d=200, max+min
  - Paper Table 3: MaximumScore/EmpiricalWelfare d=5/10/20/50
  - Additional scalable: Rosenbrock/DixonPrice/Zakharov/Qing d=5/10/20/50/100
  - Fixed 2D: DropWave/Shubert/McCormick/SixHumpCamel/Eggholder/Bukin6/Easom/Mishra6

10 random seeds per (function, variant) pair.
Results saved to result/seed-test/.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from smco import smco, smco_br, smco_br_evo, smco_evo, smco_r, smco_r_evo
from smco.test_functions import assign_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 4096, 5555, 9999]
VARIANTS = ["smco", "smco_r", "smco_br", "smco_evo", "smco_r_evo", "smco_br_evo"]

VARIANT_FNS = {
    "smco": smco,
    "smco_r": smco_r,
    "smco_br": smco_br,
    "smco_evo": smco_evo,
    "smco_r_evo": smco_r_evo,
    "smco_br_evo": smco_br_evo,
}

OUT_DIR = REPO_ROOT / "result" / "seed-test"

# ---------------------------------------------------------------------------
# Benchmark case definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkCase:
    label: str           # e.g. "table1_Rastrigin_d2_max"
    func_name: str       # for assign_config or special factory
    dim: int
    maximize: bool
    iter_max: int
    n_starts: int
    group: str           # "table1" / "table2" / "table3" / "scalable" / "fixed2d"
    # For Table 3 random objectives: factory function to build a fixed objective
    objective_factory: Callable[[], tuple[Callable, np.ndarray, np.ndarray]] | None = None

    def get_objective(self) -> tuple[Callable[[np.ndarray], float], np.ndarray, np.ndarray]:
        """Return (f, bounds_lower, bounds_upper) ready for SMCO."""
        if self.objective_factory is not None:
            return self.objective_factory()
        config = assign_config(self.func_name, self.dim)
        # assign_config returns max-objective (negated for min-sense)
        f = config.f
        if not self.maximize:
            raw_f = f
            f = lambda x, _f=raw_f: -_f(x)
        return f, config.bounds_lower.copy(), config.bounds_upper.copy()

    def effective_n_starts(self) -> int:
        return self.n_starts


def _build_table1_cases() -> list[BenchmarkCase]:
    funcs = ["Rastrigin", "Ackley", "Griewank", "Michalewicz"]
    cases = []
    for fn in funcs:
        for direction in ("max", "min"):
            maximize = direction == "max"
            label = f"table1_{fn}_d2_{direction}"
            cases.append(BenchmarkCase(
                label=label, func_name=fn, dim=2, maximize=maximize,
                iter_max=500, n_starts=1, group="table1",
            ))
    return cases


def _build_table2_cases() -> list[BenchmarkCase]:
    funcs = ["Rastrigin", "Ackley", "Griewank", "Michalewicz"]
    cases = []
    for fn in funcs:
        for direction in ("max", "min"):
            maximize = direction == "max"
            label = f"table2_{fn}_d200_{direction}"
            cases.append(BenchmarkCase(
                label=label, func_name=fn, dim=200, maximize=maximize,
                iter_max=500, n_starts=14, group="table2",
            ))
    return cases


def _maximum_score_factory(dim: int, data_seed: int) -> Callable[[], tuple[Callable, np.ndarray, np.ndarray]]:
    """Build a factory that generates a fixed MaximumScore objective."""
    def factory():
        lower = np.full(dim, -20.0)
        upper = np.full(dim, 20.0)
        rng = np.random.default_rng(data_seed)
        n = 500
        x1 = rng.normal(loc=0.0, scale=np.sqrt(dim), size=n)
        xrest = rng.normal(loc=0.0, scale=1.0, size=(n, dim))
        beta0 = rng.normal(loc=0.0, scale=1.0, size=dim)
        eps = rng.normal(loc=0.0, scale=np.sqrt(dim), size=n)
        y = (x1 + xrest @ beta0 + eps >= 0.0).astype(float)

        def objective(beta: np.ndarray) -> float:
            beta = np.asarray(beta, dtype=float)
            return float(np.mean(y * (x1 + xrest @ beta >= 0.0)))

        return objective, lower, upper
    return factory


def _empirical_welfare_factory(dim: int, data_seed: int, cache_dir: Path) -> Callable[[], tuple[Callable, np.ndarray, np.ndarray]]:
    """Build a factory that generates a fixed EmpiricalWelfare objective."""
    def factory():
        import csv as _csv

        lower = np.full(dim, -20.0)
        upper = np.full(dim, 20.0)
        cache_dir.mkdir(parents=True, exist_ok=True)
        csv_path = cache_dir / "jtpa.csv"
        if not csv_path.exists():
            import urllib.request
            url = "https://zenodo.org/api/records/18836198/files/jtpa.csv/content"
            urllib.request.urlretrieve(url, csv_path)

        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
        assign = np.array([float(r["Z2"]) for r in rows])
        prevearn = np.array([float(r["prevearn"]) for r in rows])
        edu = np.array([float(r["edu"]) for r in rows])
        earnings = np.array([float(r["earnings"]) for r in rows])
        pscore = 2.0 / 3.0

        def standardize(col):
            c = col - np.mean(col)
            s = np.std(c)
            return c / s if s != 0 else c

        base_x = np.column_stack([standardize(prevearn), standardize(edu), standardize(edu**2), standardize(edu**3)])

        rng = np.random.default_rng(data_seed)
        if dim == 5:
            x = base_x
        else:
            extra = rng.normal(loc=0.0, scale=1.0, size=(base_x.shape[0], dim - 5))
            x = np.column_stack([base_x, extra])
        weights = assign / pscore - (1.0 - assign) / (1.0 - pscore)

        def objective(beta: np.ndarray) -> float:
            beta = np.asarray(beta, dtype=float)
            decision = beta[0] + x @ beta[1:]
            return float(np.mean(weights * earnings * (decision >= 0.0)))

        return objective, lower, upper
    return factory


def _build_table3_cases(cache_dir: Path) -> list[BenchmarkCase]:
    dims = [5, 10, 20, 50]
    cases = []
    for dim in dims:
        n_starts = int(round(math.sqrt(dim)))
        # MaximumScore
        cases.append(BenchmarkCase(
            label=f"table3_MaximumScore_d{dim}",
            func_name="MaximumScore", dim=dim, maximize=True,
            iter_max=300, n_starts=n_starts, group="table3",
            objective_factory=_maximum_score_factory(dim, data_seed=99900 + dim),
        ))
        # EmpiricalWelfare
        cases.append(BenchmarkCase(
            label=f"table3_EmpiricalWelfare_d{dim}",
            func_name="EmpiricalWelfare", dim=dim, maximize=True,
            iter_max=300, n_starts=n_starts, group="table3",
            objective_factory=_empirical_welfare_factory(dim, data_seed=88800 + dim, cache_dir=cache_dir),
        ))
    return cases


def _build_scalable_cases() -> list[BenchmarkCase]:
    funcs = ["Rosenbrock", "DixonPrice", "Zakharov", "Qing"]
    dims = [5, 10, 20, 50, 100]
    cases = []
    for fn in funcs:
        for dim in dims:
            n_starts = max(3, int(round(math.sqrt(dim))))
            cases.append(BenchmarkCase(
                label=f"scalable_{fn}_d{dim}",
                func_name=fn, dim=dim, maximize=True,
                iter_max=300, n_starts=n_starts, group="scalable",
            ))
    return cases


def _build_fixed2d_cases() -> list[BenchmarkCase]:
    funcs = ["DropWave", "Shubert", "McCormick", "SixHumpCamel", "Eggholder", "Bukin6", "Easom", "Mishra6"]
    cases = []
    for fn in funcs:
        cases.append(BenchmarkCase(
            label=f"fixed2d_{fn}_d2",
            func_name=fn, dim=2, maximize=True,
            iter_max=300, n_starts=5, group="fixed2d",
        ))
    return cases


def build_all_cases(cache_dir: Path) -> list[BenchmarkCase]:
    return (
        _build_table1_cases()
        + _build_table2_cases()
        + _build_table3_cases(cache_dir)
        + _build_scalable_cases()
        + _build_fixed2d_cases()
    )


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

CSV_FIELDS = ["case_label", "group", "func_name", "dim", "maximize", "variant", "seed",
              "f_optimal", "elapsed_s", "error"]


def run_single(case: BenchmarkCase, variant: str, seed: int) -> dict:
    f, lower, upper = case.get_objective()
    variant_fn = VARIANT_FNS[variant]
    t0 = time.perf_counter()
    try:
        result = variant_fn(
            f, lower, upper,
            n_starts=case.n_starts,
            iter_max=case.iter_max,
            seed=seed,
        )
        elapsed = time.perf_counter() - t0
        return {
            "case_label": case.label,
            "group": case.group,
            "func_name": case.func_name,
            "dim": case.dim,
            "maximize": case.maximize,
            "variant": variant,
            "seed": seed,
            "f_optimal": result.best_result.f_optimal,
            "elapsed_s": round(elapsed, 3),
            "error": "",
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "case_label": case.label,
            "group": case.group,
            "func_name": case.func_name,
            "dim": case.dim,
            "maximize": case.maximize,
            "variant": variant,
            "seed": seed,
            "f_optimal": "",
            "elapsed_s": round(elapsed, 3),
            "error": str(exc),
        }


def load_existing_results(csv_path: Path) -> dict[tuple, dict]:
    """Load completed runs for resume support. Returns {(label, variant, seed): row}."""
    done = {}
    if not csv_path.exists():
        return done
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["case_label"], row["variant"], int(row["seed"]))
            done[key] = row
    return done


def append_rows(rows: list[dict], csv_path: Path) -> None:
    need_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if need_header:
            writer.writeheader()
        writer.writerows(rows)


def run_all(cases: list[BenchmarkCase], csv_path: Path) -> list[dict]:
    done = load_existing_results(csv_path)
    total = len(cases) * len(VARIANTS) * len(SEEDS)
    skipped = 0
    new_rows: list[dict] = []
    run_count = 0

    for case in cases:
        for variant in VARIANTS:
            for seed in SEEDS:
                key = (case.label, variant, seed)
                if key in done:
                    skipped += 1
                    continue
                row = run_single(case, variant, seed)
                new_rows.append(row)
                run_count += 1

                status = "OK" if not row["error"] else "ERR"
                print(f"[{skipped + run_count}/{total}] {case.label} {variant} seed={seed} -> "
                      f"{row['f_optimal'] if isinstance(row['f_optimal'], float) else status}"
                      f"  ({row['elapsed_s']:.1f}s)")

                # Flush every 10 runs
                if run_count % 10 == 0:
                    append_rows(new_rows, csv_path)
                    new_rows = []

    if new_rows:
        append_rows(new_rows, csv_path)

    print(f"\nCompleted: {run_count} new runs, {skipped} resumed from cache.")
    # Return all rows (merged)
    all_rows = list(done.values())
    for r in new_rows:
        all_rows.append(r)
    return all_rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        if r.get("error") or r.get("f_optimal") == "":
            continue
        key = (r["case_label"], r["group"], r["func_name"], int(r["dim"]),
               r["maximize"], r["variant"])
        groups.setdefault(key, []).append(float(r["f_optimal"]))

    records = []
    for (label, group, func_name, dim, maximize, variant), vals in sorted(groups.items()):
        arr = np.array(vals, dtype=float)
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        span = max_v - min_v
        cv = std_v / abs(mean_v) if abs(mean_v) > 1e-15 else (0.0 if std_v < 1e-15 else float("inf"))
        records.append({
            "case_label": label,
            "group": group,
            "func_name": func_name,
            "dim": dim,
            "maximize": maximize,
            "variant": variant,
            "mean": mean_v,
            "std": std_v,
            "min": min_v,
            "max": max_v,
            "span": span,
            "cv": cv,
            "n_seeds": len(vals),
        })
    return records


def build_report(rows: list[dict], summary: list[dict]) -> str:
    lines: list[str] = []

    lines.append("# SMCO Seed Sensitivity Report (Full Benchmark)")
    lines.append("")
    lines.append(f"- Seeds ({len(SEEDS)}): {SEEDS}")
    lines.append(f"- Variants: {', '.join(VARIANTS)}")
    lines.append("")
    lines.append("## Benchmark Groups")
    lines.append("")

    group_counts: dict[str, int] = {}
    for s in summary:
        group_counts.setdefault(s["group"], 0)
        if s["variant"] == "smco":
            group_counts[s["group"]] += 1
    for g in ["table1", "table2", "table3", "scalable", "fixed2d"]:
        cnt = group_counts.get(g, 0)
        lines.append(f"- **{g}**: {cnt} benchmark cases")
    lines.append(f"- Total (function, variant) pairs: {len(summary)}")
    lines.append(f"- Total runs: {len(rows)}")
    lines.append("")

    # Per-group tables
    for group_name, group_title in [
        ("table1", "Table 1 (d=2)"),
        ("table2", "Table 2 (d=200)"),
        ("table3", "Table 3 (MaximumScore / EmpiricalWelfare)"),
        ("scalable", "Additional Scalable Functions"),
        ("fixed2d", "Fixed 2D Functions"),
    ]:
        group_summary = [s for s in summary if s["group"] == group_name]
        if not group_summary:
            continue

        lines.append(f"## {group_title}")
        lines.append("")
        lines.append("| Case | Variant | Mean | Std | Min | Max | Span | CV |")
        lines.append("|------|---------|------|-----|-----|-----|------|----|")
        for s in sorted(group_summary, key=lambda x: (x["case_label"], x["variant"])):
            lines.append(
                f"| {s['case_label']} | {s['variant']} | {s['mean']:.6f} | {s['std']:.6f} "
                f"| {s['min']:.6f} | {s['max']:.6f} | {s['span']:.6f} | {s['cv']:.4f} |"
            )
        lines.append("")

    # Aggregate
    cvs = [s["cv"] for s in summary if s["cv"] != float("inf") and not np.isnan(s["cv"])]
    cvs_arr = np.array(cvs) if cvs else np.array([0.0])

    lines.append("## Aggregate Statistics")
    lines.append("")
    lines.append(f"- (function, variant) pairs: {len(summary)}")
    lines.append(f"- Median CV: {float(np.median(cvs_arr)):.4f}")
    lines.append(f"- Mean CV: {float(np.mean(cvs_arr)):.4f}")
    lines.append(f"- Max CV: {float(np.max(cvs_arr)):.4f}")
    lines.append(f"- Pairs with CV < 0.01: {int(np.sum(cvs_arr < 0.01))} / {len(summary)}")
    lines.append(f"- Pairs with CV > 0.10: {int(np.sum(cvs_arr > 0.10))} / {len(summary)}")
    lines.append(f"- Pairs with CV > 0.50: {int(np.sum(cvs_arr > 0.50))} / {len(summary)}")
    lines.append("")

    # Per-variant
    lines.append("## Per-Variant Median CV")
    lines.append("")
    lines.append("| Variant | Median CV | Mean CV | Max CV |")
    lines.append("|---------|-----------|---------|--------|")
    for variant in VARIANTS:
        v_cvs = [s["cv"] for s in summary if s["variant"] == variant and s["cv"] != float("inf")]
        if v_cvs:
            v_arr = np.array(v_cvs)
            lines.append(
                f"| {variant} | {float(np.median(v_arr)):.4f} | {float(np.mean(v_arr)):.4f} "
                f"| {float(np.max(v_arr)):.4f} |"
            )
    lines.append("")

    # Per-group median CV
    lines.append("## Per-Group Median CV")
    lines.append("")
    lines.append("| Group | Median CV | Mean CV | Max CV | Cases |")
    lines.append("|-------|-----------|---------|--------|-------|")
    for group_name in ["table1", "table2", "table3", "scalable", "fixed2d"]:
        g_cvs = [s["cv"] for s in summary if s["group"] == group_name and s["cv"] != float("inf")]
        n_cases = len(set(s["case_label"] for s in summary if s["group"] == group_name))
        if g_cvs:
            g_arr = np.array(g_cvs)
            lines.append(
                f"| {group_name} | {float(np.median(g_arr)):.4f} | {float(np.mean(g_arr)):.4f} "
                f"| {float(np.max(g_arr)):.4f} | {n_cases} |"
            )
    lines.append("")

    # Evo vs base
    lines.append("## Evolutionary vs Base")
    lines.append("")
    lines.append("| Base | Evo | Base median CV | Evo median CV | Delta |")
    lines.append("|------|-----|----------------|---------------|-------|")
    for base, evo in [("smco", "smco_evo"), ("smco_r", "smco_r_evo"), ("smco_br", "smco_br_evo")]:
        b_cvs = [s["cv"] for s in summary if s["variant"] == base and s["cv"] != float("inf")]
        e_cvs = [s["cv"] for s in summary if s["variant"] == evo and s["cv"] != float("inf")]
        if b_cvs and e_cvs:
            b_med = float(np.median(b_cvs))
            e_med = float(np.median(e_cvs))
            lines.append(f"| {base} | {evo} | {b_med:.4f} | {e_med:.4f} | {e_med - b_med:+.4f} |")
    lines.append("")

    # Dimension effect
    lines.append("## Seed Sensitivity by Dimension")
    lines.append("")
    dims_present = sorted(set(s["dim"] for s in summary))
    lines.append("| Dim | Median CV | Mean CV | Pairs |")
    lines.append("|-----|-----------|---------|-------|")
    for dim in dims_present:
        d_cvs = [s["cv"] for s in summary if s["dim"] == dim and s["cv"] != float("inf")]
        if d_cvs:
            d_arr = np.array(d_cvs)
            lines.append(f"| {dim} | {float(np.median(d_arr)):.4f} | {float(np.mean(d_arr)):.4f} | {len(d_arr)} |")
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- **CV (Coefficient of Variation)** = std / |mean|. Lower means more stable across seeds.")
    lines.append("- **Span** = max - min across seeds.")
    lines.append("- CV < 0.01: negligible seed effect; CV > 0.10: meaningful seed dependence; CV > 0.50: strong seed dependence.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = OUT_DIR / "cache"

    csv_path = OUT_DIR / "seed_results.csv"
    summary_csv_path = OUT_DIR / "seed_summary.csv"
    report_path = OUT_DIR / "report.md"

    cases = build_all_cases(cache_dir)
    total = len(cases) * len(VARIANTS) * len(SEEDS)

    print("=" * 70)
    print("SMCO Seed Sensitivity Analysis (Full Benchmark)")
    print(f"Seeds ({len(SEEDS)}): {SEEDS}")
    print(f"Variants: {', '.join(VARIANTS)}")
    print(f"Benchmark cases: {len(cases)}")
    print(f"  table1: {sum(1 for c in cases if c.group == 'table1')}")
    print(f"  table2: {sum(1 for c in cases if c.group == 'table2')}")
    print(f"  table3: {sum(1 for c in cases if c.group == 'table3')}")
    print(f"  scalable: {sum(1 for c in cases if c.group == 'scalable')}")
    print(f"  fixed2d: {sum(1 for c in cases if c.group == 'fixed2d')}")
    print(f"Total runs: {total}")
    print(f"Output: {OUT_DIR}")
    print("=" * 70)

    # Run (with resume support)
    rows = run_all(cases, csv_path)

    # Summary
    summary = build_summary(rows)
    sum_fields = ["case_label", "group", "func_name", "dim", "maximize", "variant",
                  "mean", "std", "min", "max", "span", "cv", "n_seeds"]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=sum_fields)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Summary saved to {summary_csv_path}")

    # Report
    report = build_report(rows, summary)
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to {report_path}")
    print("Done!")


if __name__ == "__main__":
    main()
