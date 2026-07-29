# E4 bbob-largescale 外部基准 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 COCO `bbob-largescale`（d∈{160,320,640}）上跑 winner + matched base + 5 baselines（FE=1000·d），自算 figure-5 CSV 验证高维优势在外部基准复现。

**Architecture:** 复用 E5 的 `coco_runner`；新增 `run_baseline_on_problem`（`_CocoMinObserver`：minimization + FE hard stop + clip/nan）跑 DE/GA/PSO/SA/GenSA；bbob-largescale runner 7 趟独立遍历，写 figure-5 CSV。

**Tech Stack:** Python 3 + numpy + cocoex 2.8.2（`coco-experiment`）+ `comparison.methods`（baseline_worker 同源）+ pytest。

**Spec:** `docs/superpowers/specs/2026-07-29-e4-bbob-largescale-design.md`

**分支：** `feat/smco-evo-highdim-paper-2026`。本机已装 cocoex，可完整 TDD。

**全局约定：** TDD（先失败测试→跑→实现→跑→commit）；每 task commit，message 末尾加 `Co-Authored-By: Claude <noreply@anthropic.com>`；测试 `.venv/bin/python -m pytest <path> -v`。

---

## 文件结构

| 文件 | 责任 | 动作 |
|---|---|---|
| `src/smco/coco_runner.py` | 加 `run_baseline_on_problem` + `_CocoMinObserver` | 修改 |
| `scripts/run_smco_evo_bbob_largescale.py` | bbob-largescale 遍历 + 7 算法 + figure-5 CSV | 改（替换骨架） |
| `tests/test_coco_runner.py` | baseline 路径 + 端到端 | 修改 |

---

## Task 1: coco_runner.run_baseline_on_problem + _CocoMinObserver

**Files:**
- Modify: `src/smco/coco_runner.py`
- Test: `tests/test_coco_runner.py`

- [ ] **Step 1: 写失败测试** — 在 `tests/test_coco_runner.py` 追加：

```python
def test_run_baseline_on_problem_smoke():
    from smco.coco_runner import run_baseline_on_problem
    p = _first_problem(5)
    res = run_baseline_on_problem(p, algorithm_name="GenSA", fe_budget=200)
    assert res["dimension"] == 5
    assert res["evaluations"] <= 200
    assert res["evaluations"] > 0
    assert res["algorithm_id"] == "GenSA"
    assert isinstance(res["final_target_hit"], bool)


def test_run_baseline_fe_hard_stop():
    from smco.coco_runner import run_baseline_on_problem
    p = _first_problem(5)
    res = run_baseline_on_problem(p, algorithm_name="DE", fe_budget=30)
    assert res["evaluations"] <= 30


def test_run_baseline_rejects_unknown():
    import pytest
    from smco.coco_runner import run_baseline_on_problem
    p = _first_problem(5)
    with pytest.raises(ValueError):
        run_baseline_on_problem(p, algorithm_name="CMAES", fe_budget=50)
```

- [ ] **Step 2: 跑确认失败**

Run: `.venv/bin/python -m pytest tests/test_coco_runner.py -v -k baseline`
Expected: FAIL — `ImportError: cannot import name 'run_baseline_on_problem'`

- [ ] **Step 3: 实现** — 在 `src/smco/coco_runner.py`：

(a) 顶部 import 区追加（在 `from .optimizer import ...` 之后）：

```python
from comparison.methods.de import differential_evo
from comparison.methods.ga import genetic_algorithm
from comparison.methods.gensa import gensa
from comparison.methods.pso import particle_swarm
from comparison.methods.sa import simulated_annealing
from .evaluation import EvaluationBudgetExceeded
```

(b) 在 `_select_algorithm` 之后、`run_on_problem` 之前追加 baseline dispatch + observer + runner：

```python
_BASELINE_DISPATCH = {
    "DE": differential_evo,
    "GA": genetic_algorithm,
    "PSO": particle_swarm,
    "SA": simulated_annealing,
    "GenSA": gensa,
}


class _CocoMinObserver:
    """Minimisation objective over a cocoex problem with a FE hard stop.

    Clips probe points to the cocoex bounds and penalises non-finite values
    (mirroring the SMCO path in :func:`run_on_problem`). Raises
    :class:`EvaluationBudgetExceeded` at ``max_evals`` so the baseline loop stops.
    """

    def __init__(self, problem, max_evals: int) -> None:
        self.problem = problem
        self.max_evals = int(max_evals)
        self.fe = 0

    def __call__(self, x):
        if self.fe >= self.max_evals:
            raise EvaluationBudgetExceeded(
                f"cocoex FE budget {self.max_evals} reached"
            )
        self.fe += 1
        x = np.clip(np.asarray(x, dtype=float), self.problem.lower_bounds, self.problem.upper_bounds)
        if not np.all(np.isfinite(x)):
            return 1e10
        v = float(self.problem(x))
        if not np.isfinite(v):
            return 1e10
        return v


def run_baseline_on_problem(
    problem,
    *,
    algorithm_name: str,
    fe_budget: int,
    n_starts: int = 8,
    seed: int | None = None,
    observer: Any = None,
) -> dict:
    """Run one comparison baseline on a cocoex problem; return cocoex metrics.

    Minimisation (``maximize=False``); FE is hard-stopped by ``_CocoMinObserver``.
    """
    if algorithm_name not in _BASELINE_DISPATCH:
        raise ValueError(f"unknown baseline: {algorithm_name!r}")
    if observer is not None:
        problem.observe_with(observer)
    dim = int(problem.dimension)
    algorithm = _BASELINE_DISPATCH[algorithm_name]
    if seed is None:
        seed = problem_seed(problem, n_starts)
    rng = np.random.default_rng(seed)
    span = problem.upper_bounds - problem.lower_bounds
    starts = problem.lower_bounds + rng.uniform(size=(n_starts, dim)) * span

    observer_obj = _CocoMinObserver(problem, fe_budget)
    try:
        algorithm(
            observer_obj, problem.lower_bounds, problem.upper_bounds,
            start_points=starts, maximize=False, max_iter=int(fe_budget), seed=int(seed),
        )
    except EvaluationBudgetExceeded:
        pass  # expected hard stop at the FE budget

    return {
        "algorithm_id": algorithm_name,
        "function": int(problem.id_function),
        "dimension": dim,
        "instance": int(problem.id_instance),
        "best_observed_fvalue1": float(problem.best_observed_fvalue1),
        "final_target_hit": bool(problem.final_target_hit),
        "evaluations": int(problem.evaluations),
    }
```

(c) `__all__` 追加 `"run_baseline_on_problem"`。

- [ ] **Step 4: 跑通过**

Run: `.venv/bin/python -m pytest tests/test_coco_runner.py -v`
Expected: PASS（全部）

- [ ] **Step 5: Commit**

```bash
git add src/smco/coco_runner.py tests/test_coco_runner.py
git commit -m "feat(coco): run_baseline_on_problem + _CocoMinObserver (baseline on cocoex)" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: bbob-largescale runner + figure-5 CSV

**Files:**
- Modify: `scripts/run_smco_evo_bbob_largescale.py`（替换骨架）
- Test: `tests/test_coco_runner.py`（追加）

- [ ] **Step 1: 写失败测试** — 在 `tests/test_coco_runner.py` 追加（用 bbob d5 加速，验证 runner 逻辑不依赖 largescale）：

```python
def test_bbob_largescale_runner_small_subset(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "largescale_cli", Path("scripts/run_smco_evo_bbob_largescale.py"))
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    # bbob d5 (加速), 1 instance, winner+base+5 baselines = 7 algorithms x 24 func = 168 runs
    summary = cli.run_bbob_largescale(
        winner="PY-SP-SMCO-EVO", suite="bbob", dims=[5], instances=[1],
        fe_budget_per_d=50, result_dir=tmp_path)
    import csv
    rows = list(csv.DictReader(open(tmp_path / "bbob_largescale.csv")))
    assert len(rows) == 24 * 1 * 1 * 7  # 24 func x 1 inst x d5 x 7 algos
    algos = {r["algorithm_id"] for r in rows}
    assert algos == {"PY-SP-SMCO-EVO", "PY-BASE-SMCO", "DE", "GA", "PSO", "SA", "GenSA"}
    assert all(int(r["evaluations"]) <= 50 * 5 for r in rows)
    assert (tmp_path / "bbob_largescale_summary.csv").exists()
```

- [ ] **Step 2: 跑确认失败**

Run: `.venv/bin/python -m pytest tests/test_coco_runner.py::test_bbob_largescale_runner_small_subset -v`
Expected: FAIL — `AttributeError: module 'largescale_cli' has no attribute 'run_bbob_largescale'`

- [ ] **Step 3: 实现** — 替换 `scripts/run_smco_evo_bbob_largescale.py` 全文为：

```python
#!/usr/bin/env python
"""E4 bbob-largescale external benchmark (Task 10 / E4).

Runs the frozen E1 winner + matched non-EVO base + 5 strong baselines
(DE/GA/PSO/SA/GenSA) on COCO ``bbob-largescale`` (24 functions, d in
{160, 320, 640}, instances 1--5) under B_max = 1000*d FE, writing the figure-5
data CSV (per func*dim*instance*algorithm metrics + summary). Requires cocoex.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cocoex  # noqa: E402

from smco.coco_runner import run_baseline_on_problem, run_on_problem  # noqa: E402
from smco.paper_contract import parse_algorithm_id  # noqa: E402

_FAM_TOKEN = {"smco": "SMCO", "smco_refine": "SMCO-REFINE", "smco_boost_refine": "SMCO-BOOST-REFINE"}
BASELINES = ("DE", "GA", "PSO", "SA", "GenSA")


def _have_cocoex() -> bool:
    try:
        import cocoex  # noqa: F401
        return True
    except ImportError:
        return False


def to_py(winner: str) -> str:
    parsed = parse_algorithm_id(winner)
    fam = _FAM_TOKEN[parsed["family"]]
    if not parsed["evolutionary"]:
        return f"PY-BASE-{fam}"
    slot = {"state_preserving": "SP", "restart": "RS"}[parsed["state_semantics"]]
    return f"PY-{slot}-{fam}-EVO"


def matched_base(winner_py: str) -> str:
    parsed = parse_algorithm_id(winner_py)
    return f"PY-BASE-{_FAM_TOKEN[parsed['family']]}"


def run_bbob_largescale(*, winner, suite, dims, instances, fe_budget_per_d,
                        result_dir, baselines=BASELINES) -> dict:
    """Run winner + matched base + N baselines over the suite; write figure-5 CSVs."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    winner_py = to_py(winner)
    base = matched_base(winner_py)
    smco_algos = [winner_py, base]
    suite_obj = cocoex.Suite(
        suite,
        f"instances:{'-'.join(str(i) for i in instances)}",
        f"dimensions:{','.join(str(d) for d in dims)}",
    )

    rows: list[dict] = []
    for algo in smco_algos:
        observer = cocoex.Observer(suite, f"result_folder: {result_dir / 'cocoex' / algo}")
        for problem in suite_obj:
            rows.append(run_on_problem(
                problem, algorithm_id=algo,
                fe_budget=fe_budget_per_d * int(problem.dimension), observer=observer))
    for algo in baselines:
        observer = cocoex.Observer(suite, f"result_folder: {result_dir / 'cocoex' / algo}")
        for problem in suite_obj:
            rows.append(run_baseline_on_problem(
                problem, algorithm_name=algo,
                fe_budget=fe_budget_per_d * int(problem.dimension), observer=observer))

    _write_csv(result_dir / "bbob_largescale.csv", rows,
               ("function", "dimension", "instance", "algorithm_id",
                "best_observed_fvalue1", "final_target_hit", "evaluations"))
    _write_summary(result_dir / "bbob_largescale_summary.csv", rows, smco_algos + list(baselines))
    return {"n_runs": len(rows), "algorithms": smco_algos + list(baselines)}


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in fields})


def _write_summary(path, rows, algorithms):
    by_key: dict[tuple, dict] = {}
    for r in rows:
        key = (int(r["function"]), int(r["dimension"]))
        by_key.setdefault(key, {})[r["algorithm_id"]] = r
    out = []
    for (func, dim), algos in sorted(by_key.items()):
        row = {"function": func, "dimension": dim}
        for algo in algorithms:
            rec = algos.get(algo)
            row[f"{algo}_target_hit"] = rec["final_target_hit"] if rec else ""
            row[f"{algo}_best"] = rec["best_observed_fvalue1"] if rec else ""
        out.append(row)
    fields = ["function", "dimension"] + [f"{a}_target_hit" for a in algorithms] + [f"{a}_best" for a in algorithms]
    _write_csv(path, out, fields)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner", required=True, help="Frozen E1 winner algorithm_id (Py or R; R auto-converted).")
    parser.add_argument("--suite", default="bbob-largescale", help="cocoex suite name (default bbob-largescale).")
    parser.add_argument("--dims", nargs="+", type=int, default=[160, 320, 640])
    parser.add_argument("--instances", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--fe-budget-per-d", type=int, default=1000)
    parser.add_argument("--baselines", nargs="+", default=list(BASELINES))
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args(argv)

    if not _have_cocoex():
        print("ERROR: cocoex not installed. Install with: pip install coco-experiment", file=sys.stderr)
        return 2

    summary = run_bbob_largescale(
        winner=args.winner, suite=args.suite, dims=args.dims, instances=args.instances,
        fe_budget_per_d=args.fe_budget_per_d, result_dir=args.result_dir, baselines=args.baselines)
    print(f"E4 bbob-largescale: {summary['n_runs']} runs "
          f"({', '.join(summary['algorithms'])}) -> {args.result_dir}/bbob_largescale.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 跑通过**

Run: `.venv/bin/python -m pytest tests/test_coco_runner.py -v`
Expected: PASS（全部）

- [ ] **Step 5: Commit**

```bash
git add scripts/run_smco_evo_bbob_largescale.py tests/test_coco_runner.py
git commit -m "feat(e4): bbob-largescale runner + figure-5 CSV (winner+base+5 baselines)" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: 端到端冒烟 + 全量 pytest

**Files:** Verify only.

- [ ] **Step 1: CLI 冒烟（bbob-largescale d160 1 func 1 inst，小 budget 加速）**

Run:
```bash
.venv/bin/python scripts/run_smco_evo_bbob_largescale.py \
  --winner PY-SP-SMCO-EVO --suite bbob-largescale --dims 160 --instances 1 \
  --fe-budget-per-d 20 --result-dir /tmp/e4_smoke 2>&1 | tail -2
echo "--- CSV 行数 + 算法 ---"
wc -l /tmp/e4_smoke/bbob_largescale.csv
cut -d, -f4 /tmp/e4_smoke/bbob_largescale.csv | sort -u | head
```
Expected: 7 algorithms；CSV 24 func × 1 inst × d160 × 7 = 168 行。

- [ ] **Step 2: 全量 pytest**

Run: `.venv/bin/python -m pytest -q`
Expected: 全绿（364 + E4 新增；无 fail/error）。

---

## Self-Review（已完成）

- **Spec coverage**：§4.2 run_baseline_on_problem + _CocoMinObserver → Task 1；§4.3 runner + CSV → Task 2；§4.5 测试 → 各 Task TDD。全覆盖。
- **Placeholder scan**：无 TBD/TODO；每步含完整代码。
- **Type consistency**：`run_baseline_on_problem`/`run_bbob_largescale`/`_CocoMinObserver`/`to_py`/`matched_base` 跨 task 一致；`BASELINES` 与 `baseline_worker.BASELINE_NAMES` 同集。
- **测试加速**：Task 2 测试用 bbob d5（非 largescale）验证 runner 逻辑；Task 3 冒烟用 bbob-largescale d160 小 budget 验证 largescale 真能跑。
