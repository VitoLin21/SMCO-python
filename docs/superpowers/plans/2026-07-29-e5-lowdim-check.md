# E5 低维非退化检查 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 COCO `bbob` (d∈{5,20}) 上跑 E1 winner vs matched base（FE=2000·d），自算 supplement CSV 回答"低维是否退化"。

**Architecture:** 新 `coco_runner.run_on_problem` 把 cocoex problem 包装成 SMCO objective（`g=-problem(x)`），复用现有 `optimizer` API（不改核心），cocoex observer 自动记录；lowdim runner 两趟遍历 winner/base，写 supplement CSV。

**Tech Stack:** Python 3 + numpy + cocoex 2.8.2（`coco-experiment`）+ pytest。

**Spec:** `docs/superpowers/specs/2026-07-29-e5-lowdim-check-design.md`

**分支：** `feat/smco-evo-highdim-paper-2026`。本机已装 cocoex/cocopp，可完整 TDD。

**全局约定：** TDD（先失败测试→跑→实现→跑→commit）；每 task commit，message 末尾加 `Co-Authored-By: Claude <noreply@anthropic.com>`；测试 `.venv/bin/python -m pytest <path> -v`。

---

## 文件结构

| 文件 | 责任 | 动作 |
|---|---|---|
| `src/smco/coco_runner.py` | cocoex problem → SMCO dispatch + 指标收集 | 新建 |
| `scripts/run_smco_evo_lowdim_check.py` | bbob suite 遍历 + winner/base + supplement CSV | 改（替换骨架） |
| `tests/test_coco_runner.py` | coco_runner 单元 + 端到端 | 新建 |

---

## Task 1: coco_runner.run_on_problem

**Files:**
- Create: `src/smco/coco_runner.py`
- Test: `tests/test_coco_runner.py`

- [ ] **Step 1: 写失败测试** — 新建 `tests/test_coco_runner.py`：

```python
"""Tests for the COCO bbob runner (E5 low-dim non-degradation check)."""
from __future__ import annotations

import cocoex
import pytest

from smco.coco_runner import problem_seed, run_on_problem


def _first_problem(dim=5):
    suite = cocoex.Suite("bbob", "instances:1", f"dimensions:{dim}")
    return next(iter(suite))


def test_run_on_problem_base_smoke():
    p = _first_problem(5)
    res = run_on_problem(p, algorithm_id="PY-BASE-SMCO", fe_budget=200)
    assert res["dimension"] == 5
    assert res["evaluations"] <= 200
    assert res["evaluations"] > 0
    assert isinstance(res["final_target_hit"], bool)
    # minimization best must be no worse than a random feasible point
    import numpy as np
    rng = np.random.default_rng(0)
    x_rand = p.lower_bounds + rng.uniform(size=5) * (p.upper_bounds - p.lower_bounds)
    assert res["best_observed_fvalue1"] <= p(x_rand) + 1e-9


def test_run_on_problem_fe_hard_stop():
    p = _first_problem(5)
    res = run_on_problem(p, algorithm_id="PY-BASE-SMCO", fe_budget=30)
    assert res["evaluations"] <= 30


def test_problem_seed_is_stable_and_id_derived():
    p = _first_problem(5)
    assert problem_seed(p) == problem_seed(p)
    p2 = _first_problem(5)  # same id
    assert problem_seed(p) == problem_seed(p2)


def test_run_on_problem_rejects_r_language():
    p = _first_problem(5)
    with pytest.raises(ValueError):
        run_on_problem(p, algorithm_id="R-SP-SMCO-EVO", fe_budget=50)


def test_run_on_problem_evo_smoke():
    p = _first_problem(5)
    res = run_on_problem(p, algorithm_id="PY-SP-SMCO-EVO", fe_budget=200)
    assert res["evaluations"] <= 200
    assert res["algorithm_id"] == "PY-SP-SMCO-EVO"
```

- [ ] **Step 2: 跑确认失败**

Run: `.venv/bin/python -m pytest tests/test_coco_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'smco.coco_runner'`

- [ ] **Step 3: 实现** — 新建 `src/smco/coco_runner.py`：

```python
"""COCO bbob runner for the E5 low-dim non-degradation check.

Wraps a cocoex Problem as an SMCO objective (``g = -problem(x)``; cocoex is
minimisation, SMCO maximises) and reuses the existing optimizer API — the SMCO
core is not modified. cocoex records every evaluation via its observer; the
runner returns the cocoex-accumulated metrics (best_observed_fvalue1,
final_target_hit, evaluations). See
``docs/superpowers/specs/2026-07-29-e5-lowdim-check-design.md``.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .optimizer import smco, smco_br, smco_br_evo, smco_evo, smco_r, smco_r_evo
from .paper_contract import parse_algorithm_id

_BASE_DISPATCH = {
    ("python", "smco"): smco,
    ("python", "smco_refine"): smco_r,
    ("python", "smco_boost_refine"): smco_br,
}
_EVO_DISPATCH = {
    ("python", "smco"): smco_evo,
    ("python", "smco_refine"): smco_r_evo,
    ("python", "smco_boost_refine"): smco_br_evo,
}

_DEFAULT_EVO_POINTS = (0.5, 0.75)
_DEFAULT_ELIMINATION_RATE = 0.25
_DEFAULT_DE_FACTOR = 0.8
_DEFAULT_DE_CROSSOVER = 0.7
_DEFAULT_STRATEGY = "rand1bin"


def problem_seed(problem, n_starts: int = 8) -> int:
    """Stable 32-bit seed derived from the cocoex problem id (order-independent)."""
    key = f"{problem.id}:n{n_starts}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def _select_algorithm(algorithm_id: str):
    parsed = parse_algorithm_id(algorithm_id)
    if parsed["language"] != "python":
        raise ValueError(
            f"coco_runner is Python-only; algorithm_id {algorithm_id!r} is "
            f"{parsed['language']!r}. Convert R winners to their Py equivalent."
        )
    table = _EVO_DISPATCH if parsed["evolutionary"] else _BASE_DISPATCH
    key = ("python", parsed["family"])
    if key not in table:
        raise ValueError(f"no Python dispatch for family={parsed['family']!r}")
    return table[key], parsed


def run_on_problem(
    problem,
    *,
    algorithm_id: str,
    fe_budget: int,
    n_starts: int = 8,
    seed: int | None = None,
    observer: Any = None,
) -> dict:
    """Run one SMCO variant on a cocoex problem; return cocoex-accumulated metrics.

    ``problem(x)`` is minimisation; SMCO maximises ``g = -problem(x)``. Each
    evaluation is recorded by cocoex when an observer is attached. The returned
    ``best_observed_fvalue1`` is the minimisation best found during this run.
    """
    if observer is not None:
        problem.observe_with(observer)
    dim = int(problem.dimension)
    algorithm, parsed = _select_algorithm(algorithm_id)
    if seed is None:
        seed = problem_seed(problem, n_starts)
    rng = np.random.default_rng(seed)
    span = problem.upper_bounds - problem.lower_bounds
    starts = problem.lower_bounds + rng.uniform(size=(n_starts, dim)) * span

    iter_max = max(1, int(fe_budget) // (2 * dim + 1))
    control: dict = {
        "max_evals": int(fe_budget),
        "objective_sense": "maximize",
        "known_optimum": 0.0,  # SMCO convergence target; cocoex final_target_hit is authoritative
        "iter_max": iter_max,
        "seed": int(seed),
    }
    if parsed["family"] == "smco_refine":
        control["refine_search"] = True
        control["refine_ratio"] = 0.5
    elif parsed["family"] == "smco_boost_refine":
        control["refine_search"] = True
        control["iter_boost"] = 1000
        control["refine_ratio"] = 0.5
    if parsed["evolutionary"]:
        control["evolution_points"] = _DEFAULT_EVO_POINTS
        control["elimination_rate"] = _DEFAULT_ELIMINATION_RATE
        control["evolution_strategy"] = _DEFAULT_STRATEGY
        control["de_factor"] = _DEFAULT_DE_FACTOR
        control["de_crossover"] = _DEFAULT_DE_CROSSOVER
        control["state_semantics"] = parsed["state_semantics"]

    algorithm(lambda x: -problem(x), problem.lower_bounds, problem.upper_bounds, starts, **control)

    return {
        "algorithm_id": algorithm_id,
        "function": int(problem.id_function),
        "dimension": dim,
        "instance": int(problem.id_instance),
        "best_observed_fvalue1": float(problem.best_observed_fvalue1),
        "final_target_hit": bool(problem.final_target_hit),
        "evaluations": int(problem.evaluations),
    }


__all__ = ["problem_seed", "run_on_problem"]
```

- [ ] **Step 4: 跑通过**

Run: `.venv/bin/python -m pytest tests/test_coco_runner.py -v`
Expected: PASS（全部）

- [ ] **Step 5: Commit**

```bash
git add src/smco/coco_runner.py tests/test_coco_runner.py
git commit -m "feat(coco): coco_runner.run_on_problem (bbob problem -> SMCO, metrics)" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: lowdim runner + supplement CSV

**Files:**
- Modify: `scripts/run_smco_evo_lowdim_check.py`（替换骨架）
- Test: `tests/test_coco_runner.py`（追加）

- [ ] **Step 1: 写失败测试** — 在 `tests/test_coco_runner.py` 追加：

```python
def test_lowdim_runner_small_subset(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "lowdim_cli", Path("scripts/run_smco_evo_lowdim_check.py"))
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    summary = cli.run_lowdim(
        winner="PY-SP-SMCO-EVO", dims=[5], instances=[1],
        fe_budget_per_d=200, result_dir=tmp_path)
    # winner + base on 1 func x 1 instance x d5 = 2 rows
    import csv
    rows = list(csv.DictReader(open(tmp_path / "lowdim_degradation.csv")))
    assert len(rows) == 2
    assert {r["algorithm_id"] for r in rows} == {"PY-SP-SMCO-EVO", "PY-BASE-SMCO"}
    assert all(int(r["evaluations"]) <= 200 * 5 for r in rows)
    assert (tmp_path / "lowdim_summary.csv").exists()
```

- [ ] **Step 2: 跑确认失败**

Run: `.venv/bin/python -m pytest tests/test_coco_runner.py::test_lowdim_runner_small_subset -v`
Expected: FAIL — `AttributeError: module 'lowdim_cli' has no attribute 'run_lowdim'`

- [ ] **Step 3: 实现** — 替换 `scripts/run_smco_evo_lowdim_check.py` 全文为：

```python
#!/usr/bin/env python
"""E5 low-dimensional non-degradation check (Task 10 / E5).

Runs the frozen E1 winner + matched non-EVO base on COCO ``bbob``
(24 functions, d in {5, 20}, official instances 1--5) under B_max = 2000*d FE,
and writes a supplement CSV (per func×dim×instance×algorithm metrics +
winner-vs-base summary). Unless severely degraded, low-dim results do not
overturn the high-dim winner. Requires cocoex (``pip install coco-experiment``).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cocoex  # noqa: E402

from smco.coco_runner import run_on_problem  # noqa: E402
from smco.paper_contract import parse_algorithm_id  # noqa: E402

_FAM_TOKEN = {"smco": "SMCO", "smco_refine": "SMCO-REFINE", "smco_boost_refine": "SMCO-BOOST-REFINE"}


def _have_cocoex() -> bool:
    try:
        import cocoex  # noqa: F401
        return True
    except ImportError:
        return False


def to_py(winner: str) -> str:
    """Normalise an R-winner to its Py equivalent (E5 runs Python cocoex)."""
    parsed = parse_algorithm_id(winner)
    fam = _FAM_TOKEN[parsed["family"]]
    if not parsed["evolutionary"]:
        return f"PY-BASE-{fam}"
    slot = {"state_preserving": "SP", "restart": "RS"}[parsed["state_semantics"]]
    return f"PY-{slot}-{fam}-EVO"


def matched_base(winner_py: str) -> str:
    parsed = parse_algorithm_id(winner_py)
    return f"PY-BASE-{_FAM_TOKEN[parsed['family']]}"


def run_lowdim(*, winner, dims, instances, fe_budget_per_d, result_dir) -> dict:
    """Run winner + matched base over the bbob suite; write supplement CSVs."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    winner_py = to_py(winner)
    base = matched_base(winner_py)
    suite = cocoex.Suite(
        "bbob",
        f"instances:{'-'.join(str(i) for i in instances)}",
        f"dimensions:{','.join(str(d) for d in dims)}",
    )

    rows: list[dict] = []
    for algo in [winner_py, base]:
        observer = cocoex.Observer("bbob", f"result_folder: {result_dir / 'cocoex' / algo}")
        for problem in suite:
            rows.append(run_on_problem(
                problem, algorithm_id=algo,
                fe_budget=fe_budget_per_d * int(problem.dimension),
                observer=observer,
            ))

    _write_csv(result_dir / "lowdim_degradation.csv", rows,
               ("function", "dimension", "instance", "algorithm_id",
                "best_observed_fvalue1", "final_target_hit", "evaluations"))
    _write_summary(result_dir / "lowdim_summary.csv", rows, winner_py, base)
    return {"n_runs": len(rows), "winner": winner_py, "base": base}


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in fields})


def _write_summary(path, rows, winner, base):
    # Per (function, dimension): winner/base final_target_hit rate + median best.
    by_key: dict[tuple, dict] = {}
    for r in rows:
        key = (int(r["function"]), int(r["dimension"]))
        by_key.setdefault(key, {})[r["algorithm_id"]] = r
    out = []
    import statistics
    for (func, dim), algos in sorted(by_key.items()):
        w = algos.get(winner); b = algos.get(base)
        out.append({
            "function": func, "dimension": dim,
            "winner_target_hit": w["final_target_hit"] if w else "",
            "base_target_hit": b["final_target_hit"] if b else "",
            "winner_best": w["best_observed_fvalue1"] if w else "",
            "base_best": b["best_observed_fvalue1"] if b else "",
        })
    _write_csv(path, out, ("function", "dimension", "winner_target_hit",
                           "base_target_hit", "winner_best", "base_best"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner", required=True, help="Frozen E1 winner algorithm_id (Py or R; R auto-converted).")
    parser.add_argument("--dims", nargs="+", type=int, default=[5, 20])
    parser.add_argument("--instances", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--fe-budget-per-d", type=int, default=2000)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args(argv)

    if not _have_cocoex():
        print("ERROR: cocoex not installed. Install with: pip install coco-experiment", file=sys.stderr)
        return 2

    summary = run_lowdim(
        winner=args.winner, dims=args.dims, instances=args.instances,
        fe_budget_per_d=args.fe_budget_per_d, result_dir=args.result_dir)
    print(f"E5 lowdim: {summary['n_runs']} runs ({summary['winner']} vs {summary['base']}) "
          f"-> {args.result_dir}/lowdim_degradation.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 跑通过**

Run: `.venv/bin/python -m pytest tests/test_coco_runner.py -v`
Expected: PASS（全部）

- [ ] **Step 5: Commit**

```bash
git add scripts/run_smco_evo_lowdim_check.py tests/test_coco_runner.py
git commit -m "feat(e5): lowdim runner + supplement CSV (bbob winner vs base)" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: 端到端冒烟 + 全量 pytest

**Files:** Verify only.

- [ ] **Step 1: CLI 冒烟（小子集）**

Run:
```bash
.venv/bin/python scripts/run_smco_evo_lowdim_check.py \
  --winner PY-SP-SMCO-EVO --dims 5 --instances 1 \
  --fe-budget-per-d 200 --result-dir /tmp/e5_smoke 2>&1 | tail -2
echo "--- CSV ---"; head -3 /tmp/e5_smoke/lowdim_degradation.csv
```
Expected: `E5 lowdim: 2 runs ...`；CSV 含 winner+base 两行。

- [ ] **Step 2: 全量 pytest**

Run: `.venv/bin/python -m pytest -q`
Expected: 全绿（358 + coco_runner 新增；无 fail/error）。

- [ ] **Step 3: 记 pyproject optional deps（备注，不阻塞）**

确认 `coco-experiment`/`cocopp` 在本机已装；正式记入 `pyproject.toml` 的 `paper` optional deps 留 Task 12（统计阶段）一并处理，本计划不改 pyproject。

---

## Self-Review（已完成）

- **Spec coverage**：§4.2 coco_runner → Task 1；§4.3 lowdim runner + CSV → Task 2；§4.5 测试 → 各 Task TDD；§3 cocopp 非关键路径 → 不在本计划。全覆盖。
- **Placeholder scan**：无 TBD/TODO；每步含完整代码。
- **Type consistency**：`run_on_problem`/`run_lowdim`/`to_py`/`matched_base`/`problem_seed` 跨 task 签名一致；`run_lowdim` 在 Task 2 定义。
- **风险**：`known_optimum=0.0`（bbob 多数 optimal≈0；Schwefel 类不提前停，cocoex `final_target_hit` 是权威判定）；FE 由 SMCO `max_evals` hard stop，`problem.evaluations ≤ fe_budget`（Task 1 测试覆盖）。
