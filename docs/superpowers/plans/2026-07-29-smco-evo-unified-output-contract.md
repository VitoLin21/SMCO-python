# SMCO-EVO 全链条统一输出契约 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让三种 worker（Py SMCO / R SMCO / baseline）输出统一且详尽的 outcome payload，由 merge 在 Python 单点构建 `result_row`，消除原 Task 11 为 R 重建、为 baseline bypass 的两条特殊路径。

**Architecture:** 方案 B——worker 只产统一 outcome（含嵌入 `task`、新增 `best_so_far_trace`、保留全部 FE/质量/provenance 字段，缺失值统一 `null`）；`merge_results.py` 从 outcome + frozen manifest task 单点构建 `RESULT_COLUMNS` row，做 supersedes 解析 + 11 项 provenance audit，写 `merged/`。

**Tech Stack:** Python 3（stdlib + numpy）、R 4.3.2（jsonlite/qrng）、pytest。`paper_contract.py` 保持 stdlib-only。

**Spec:** `docs/superpowers/specs/2026-07-29-smco-evo-unified-output-contract-design.md`

**分支：** `feat/smco-evo-highdim-paper-2026`（不在 main 上做）。本机可跑全部测试与 R 端到端。

**全局约定：**
- TDD：先写失败测试 → 跑 → 实现 → 跑通过 → commit。
- 每个 task 结束 commit。commit message 末尾加 `Co-Authored-By: Claude <noreply@anthropic.com>`。
- 跑测试：`.venv/bin/python -m pytest <path> -v`（本机用 `.venv`；若无则 `python -m pytest`）。
- `result_row` / `RESULT_COLUMNS` / `validate_result_row` 来自 `smco.paper_contract`；`result_row_from_task` / `validate_result_against_task` / `derive_seed` / `load_manifest` / `verify_manifest` 来自 `smco.experiment_manifests`。

---

## 文件结构

| 文件 | 责任 | 动作 |
|---|---|---|
| `src/smco/paper_contract.py` | 加 `OUTCOME_FIELDS` + `validate_outcome` | 修改 |
| `src/smco/highdim_worker.py` | `run_task` 返回统一 outcome（去 `result_row`，加 trace/task/supersedes） | 修改 |
| `src/smco/baseline_worker.py` | `run_baseline_task` 补齐 outcome 字段（trace/termination/fe_counts/peak_memory/supersedes） | 修改 |
| `scripts/run_smco_evo_highdim_baselines.py` | `run_baseline_file` 嵌入 `task`；占位 payload 统一 | 修改 |
| `scripts/run_smco_evo_highdim_r.R` | outcome：`na="null"`、加 trace/supersedes、占位去 `result_row` | 修改 |
| `scripts/run_smco_evo_highdim_factorial.py` | infra 占位去 `result_row: None` | 修改 |
| `src/smco/merge_results.py` | 单点构建 row + supersedes + audit + 输出 | 新建 |
| `scripts/merge_smco_evo_highdim_results.py` | CLI | 新建 |
| `tests/test_paper_contract.py` | `validate_outcome` 测试 | 修改 |
| `tests/test_highdim_worker.py` | 断言 outcome（去 `result_row`） | 修改 |
| `tests/test_baseline_worker.py` | 断言 outcome 字段 + 端到端 `validate_outcome` | 修改 |
| `tests/test_merge_results.py` | merge 全流程 TDD | 新建 |
| `docs/gate-d-pilot-2026-07-29.md` | 字段表更新 | 修改 |

---

## Task 1: paper_contract — OUTCOME_FIELDS + validate_outcome

**Files:**
- Modify: `src/smco/paper_contract.py`
- Test: `tests/test_paper_contract.py`

- [ ] **Step 1: 写失败测试** — 在 `tests/test_paper_contract.py` 末尾追加：

```python
from smco.paper_contract import OUTCOME_FIELDS, validate_outcome


def _good_outcome():
    return {
        "run_id": "rabc123def456abcd",
        "status": "success",
        "failure_reason": "none",
        "fe_used": 100,
        "fe_budget": 200,
        "best_value": 1e-6,
        "known_optimum": 0.0,
        "normalized_gap": 0.01,
        "target_hit_fe": {"1e-1": 50, "1e-2": None, "1e-3": None, "1e-5": None},
        "anytime": [{"checkpoint_fe": 200, "fe_used": 100, "best_value": 1e-6, "normalized_gap": 0.01}],
        "best_so_far_trace": [[10, 1e-1], [50, 1e-6]],
        "termination_reason": "evaluation_budget",
        "fe_counts_by_event": {"initialization": 1},
        "wall_time_sec": 0.5,
        "peak_memory_mb": 12.0,
        "machine_id": "host",
        "git_commit": "abc",
        "environment_hash": "env",
        "task": {"run_id": "rabc123def456abcd"},
        "algorithm_id": "PY-SP-SMCO-EVO",
        "supersedes_run_id": "none",
    }


def test_validate_outcome_passes_good_payload():
    assert validate_outcome(_good_outcome()) == []


def test_validate_outcome_detects_missing_field():
    payload = _good_outcome()
    del payload["best_so_far_trace"]
    errors = validate_outcome(payload)
    assert errors and any("best_so_far_trace" in e for e in errors)


def test_validate_outcome_detects_bad_status():
    payload = _good_outcome()
    payload["status"] = "great"
    assert validate_outcome(payload) != []


def test_validate_outcome_detects_wrong_types():
    p = _good_outcome(); p["target_hit_fe"] = "x"
    assert any("target_hit_fe" in e for e in validate_outcome(p))
    p = _good_outcome(); p["best_so_far_trace"] = "x"
    assert any("best_so_far_trace" in e for e in validate_outcome(p))
    p = _good_outcome(); p["fe_counts_by_event"] = "x"
    assert any("fe_counts_by_event" in e for e in validate_outcome(p))
    p = _good_outcome(); p["task"] = "x"
    assert any("task" in e for e in validate_outcome(p))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_paper_contract.py -v -k validate_outcome`
Expected: FAIL — `ImportError: cannot import name 'OUTCOME_FIELDS'`

- [ ] **Step 3: 实现** — 在 `src/smco/paper_contract.py` 的 `RESULT_COLUMNS` 定义之后、`validate_result_row` 之前插入：

```python
# --- unified outcome payload (worker -> raw/<run_id>.json); contract section 7 ---
OUTCOME_FIELDS: tuple[str, ...] = (
    "run_id",
    "status",
    "failure_reason",
    "fe_used",
    "fe_budget",
    "best_value",
    "known_optimum",
    "normalized_gap",
    "target_hit_fe",
    "anytime",
    "best_so_far_trace",
    "termination_reason",
    "fe_counts_by_event",
    "wall_time_sec",
    "peak_memory_mb",
    "machine_id",
    "git_commit",
    "environment_hash",
    "task",
    "algorithm_id",
    "supersedes_run_id",
)


def validate_outcome(payload: Mapping[str, Any]) -> list[str]:
    """Return contract violations for an outcome payload (empty == ok).

    Only success/algorithm_failure outcomes must pass; infra_failure / timeout
    runner placeholders (``{run_id, status, failure_reason}``) are tolerated by
    the merge step and need not pass this check.
    """
    errors: list[str] = []
    for field in OUTCOME_FIELDS:
        if field not in payload:
            errors.append(f"missing outcome field: {field}")
    if errors:
        return errors
    if payload["status"] not in STATUSES:
        errors.append(f"status not in {STATUSES}")
    if not isinstance(payload["target_hit_fe"], dict):
        errors.append("target_hit_fe must be a dict")
    if not isinstance(payload["anytime"], list):
        errors.append("anytime must be a list")
    if not isinstance(payload["best_so_far_trace"], list):
        errors.append("best_so_far_trace must be a list")
    if not isinstance(payload["fe_counts_by_event"], dict):
        errors.append("fe_counts_by_event must be a dict")
    if not isinstance(payload["task"], dict):
        errors.append("task must be an embedded dict")
    return errors
```

并把 `__all__` 列表里追加 `"OUTCOME_FIELDS"`, `"validate_outcome"`。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_paper_contract.py -v`
Expected: PASS（全部，含原有测试）

- [ ] **Step 5: Commit**

```bash
git add src/smco/paper_contract.py tests/test_paper_contract.py
git commit -m "feat: add OUTCOME_FIELDS + validate_outcome to paper_contract" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Py worker — run_task 返回统一 outcome

**Files:**
- Modify: `src/smco/highdim_worker.py`（`run_task` 返回块，约第 207–243 行）

- [ ] **Step 1: 写失败测试** — `tests/test_highdim_worker.py` 顶部 import 改为：

```python
from smco.paper_contract import validate_outcome
```
（删除 `from smco.paper_contract import validate_result_row`。）

把 `test_run_task_base_smco_smoke` 末尾两行（`assert validate_result_row(res["result_row"]) == []` 与 `assert res["result_row"]["run_id"] == ...`）替换为：

```python
    assert validate_outcome(res) == []
    assert res["task"]["run_id"] == _base_task()["run_id"]
    assert res["supersedes_run_id"] == "none"
    assert isinstance(res["best_so_far_trace"], list)
```

把 `test_run_task_evo_sp_smoke`、`test_run_task_evo_restart_smoke`、`test_run_task_br_smoke` 里的 `assert validate_result_row(res["result_row"]) == []` 全部替换为 `assert validate_outcome(res) == []`。

把 `test_run_task_normalized_gap_in_unit_interval` 里 `float(res["result_row"]["normalized_gap"])` 改为 `float(res["normalized_gap"])`。

把 `test_run_task_fe_used_observed_equals_budget_cap_path` 里 `res["result_row"]["fe_used"]` 改为 `res["fe_used"]`（该行变为 `assert res["fe_used"] == res["fe_used"]`，删掉这行整行，因为同义）。

把 `test_run_task_file_end_to_end` 里 `payload["result_row"]["run_id"]` 改为 `payload["run_id"]`，并在其后加 `assert payload["task"]["run_id"] == task["run_id"]`。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_highdim_worker.py -v`
Expected: FAIL — `KeyError: 'result_row'`（worker 仍返回旧 payload）

- [ ] **Step 3: 实现** — 在 `src/smco/highdim_worker.py`：

(a) 把第 28 行 `from .experiment_manifests import result_row_from_task` 删除（不再使用）。

(b) 把 `run_task` 末尾的返回块（从 `result_row = result_row_from_task(` 到 `return {...}` 结束，约第 207–243 行）整体替换为：

```python
    return {
        "run_id": task["run_id"],
        "status": status,
        "failure_reason": failure_reason,
        "fe_used": fe_used,
        "fe_budget": fe_budget,
        "best_value": float(best_min),
        "known_optimum": known_optimum,
        "normalized_gap": normalized_gap,
        "target_hit_fe": target_hit,
        "anytime": anytime,
        "best_so_far_trace": [[int(fe), float(val)] for fe, val in observer.trace],
        "termination_reason": termination_reason,
        "fe_counts_by_event": evaluation_counts,
        "wall_time_sec": wall_time,
        "peak_memory_mb": peak_memory_mb,
        "machine_id": machine_id,
        "git_commit": git_commit,
        "environment_hash": environment_hash,
        "task": task,
        "algorithm_id": task["algorithm_id"],
        "supersedes_run_id": NONE_TOKEN,
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_highdim_worker.py -v`
Expected: PASS（全部）

- [ ] **Step 5: Commit**

```bash
git add src/smco/highdim_worker.py tests/test_highdim_worker.py
git commit -m "refactor: Py worker emits unified outcome (drop result_row, add trace/task)" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: baseline worker — 补齐 outcome 字段 + runner 嵌 task

**Files:**
- Modify: `src/smco/baseline_worker.py`（`run_baseline_task` 返回块，约第 162–180 行）
- Modify: `scripts/run_smco_evo_highdim_baselines.py`（`run_baseline_file`，约第 94 行）

- [ ] **Step 1: 写失败测试** — 在 `tests/test_baseline_worker.py`：

(a) 顶部加 import：`from smco.paper_contract import validate_outcome`

(b) 在 `test_run_baseline_task_smoke` 的断言后追加：

```python
    assert isinstance(res["best_so_far_trace"], list)
    assert res["supersedes_run_id"] == "none"
    assert res["termination_reason"] == "evaluation_budget"
    assert res["fe_counts_by_event"] == {}
    assert res["peak_memory_mb"] is None
```

(c) 在 `test_run_baseline_batch_end_to_end` 的循环里，对 success payload 追加校验（嵌入 task 使其通过 outcome 契约）：

```python
        if payload["status"] == "success":
            assert validate_outcome(payload) == []
            assert payload["task"]["algorithm"] in {"GenSA", "DE"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_baseline_worker.py -v`
Expected: FAIL — `KeyError: 'best_so_far_trace'` / `validate_outcome` 报缺字段

- [ ] **Step 3: 实现** —

(a) `src/smco/baseline_worker.py` 把 `run_baseline_task` 返回块替换为：

```python
    return {
        "algorithm_id": algorithm_name,
        "stage": stage,
        "function": instance.function_name,
        "dimension": instance.dimension,
        "known_optimum": known_optimum,
        "status": status,
        "failure_reason": failure_reason,
        "fe_budget": int(fe_budget),
        "fe_used": fe_used,
        "best_value": float(best_min),
        "normalized_gap": normalized_gap,
        "objective_sense": "minimize",
        "target_hit_fe": target_hit,
        "anytime": anytime,
        "best_so_far_trace": [[int(fe), float(val)] for fe, val in observer.trace],
        "termination_reason": "evaluation_budget",
        "fe_counts_by_event": {},
        "wall_time_sec": wall_time,
        "peak_memory_mb": None,
        "machine_id": machine_id,
        "git_commit": git_commit,
        "environment_hash": environment_hash,
        "supersedes_run_id": NONE_TOKEN,
    }
```

（`run_id` 与 `task` 仍由 runner 填充——见 (b)。）

(b) `scripts/run_smco_evo_highdim_baselines.py` 的 `run_baseline_file` 里，把

```python
    payload["run_id"] = run_id
    _atomic_write_json(result_dir / f"{run_id}.json", payload)
```

改为：

```python
    payload["run_id"] = run_id
    payload["task"] = task
    _atomic_write_json(result_dir / f"{run_id}.json", payload)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_baseline_worker.py -v`
Expected: PASS（全部）

- [ ] **Step 5: Commit**

```bash
git add src/smco/baseline_worker.py scripts/run_smco_evo_highdim_baselines.py tests/test_baseline_worker.py
git commit -m "refactor: baseline worker emits unified outcome + runner embeds task" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: R worker — 统一 outcome（na=null + trace + supersedes）

**Files:**
- Modify: `scripts/run_smco_evo_highdim_r.R`（`.payload` 构建块，约第 182–207 行）

R 无 Python 单元测试框架；靠本机端到端验证（Step 4）。

- [ ] **Step 1: 实现** — 在 `scripts/run_smco_evo_highdim_r.R`：

(a) 把成功分支的 `.payload <- list(...)`（约第 182–201 行）替换为：

```r
  .payload <- list(
    run_id = .run_id,
    status = "success",
    failure_reason = "none",
    fe_used = .fe_used,
    fe_budget = .fe_budget,
    best_value = .best_min,
    known_optimum = .known_optimum,
    normalized_gap = .norm_gap,
    objective_sense = "minimize",
    target_hit_fe = .target_hit,
    anytime = .anytime,
    best_so_far_trace = mapply(c, .obs$trace_fe, .obs$trace_val, SIMPLIFY = FALSE),
    termination_reason = .fe_summary$termination_reason %||% "evaluation_budget",
    fe_counts_by_event = as.list(.fe_summary$evaluation_counts_by_event %||% list()),
    wall_time_sec = as.numeric((proc.time() - .t0)["elapsed"]),
    peak_memory_mb = NA_real_,
    machine_id = Sys.info()[["nodename"]],
    git_commit = "",
    environment_hash = paste0("R-", R.version$major, ".", R.version$minor),
    task = .task,
    algorithm_id = .task$algorithm_id,
    supersedes_run_id = "none"
  )
```

说明：`mapply(c, ..., SIMPLIFY=FALSE)` 在 trace 为空时返回 `list()`，非空时返回 `list(c(fe,val), ...)` → JSON `[[fe,val],...]]`。

(b) 把 infra_failure 分支的 `.payload <<- list(...)`（约第 204–206 行）替换为（去掉 `result_row = NULL`）：

```r
  .payload <<- list(run_id = .run_id, status = "infra_failure",
                    failure_reason = paste0(class(e)[1], ": ", conditionMessage(e)))
```

(c) 把原子写那行（约第 211 行）的 `write_json` 加 `na = "null"`，确保未命中 target / NA 数值序列化为 JSON `null`：

```r
jsonlite::write_json(.payload, .tmp, auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null")
```

- [ ] **Step 2: R 语法校验**

Run: `Rscript -e 'parse(file = "scripts/run_smco_evo_highdim_r.R"); cat("parse OK\n")'`
Expected: `parse OK`

- [ ] **Step 3: 构造端到端 fixture 脚本** — 新建 `/tmp/verify_r_outcome.py`：

```python
import json, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from smco.experiment_manifests import build_algorithm_config, build_task
from smco.highdim_instances import generate_instance, write_instance_artifacts

root = Path("/tmp/r_outcome_verify"); root.mkdir(exist_ok=True)
inst = generate_instance("Rastrigin", 4, 0, seed=1)
rng = np.random.default_rng(5); span = inst.bounds_upper - inst.bounds_lower
starts = inst.bounds_lower + rng.uniform(size=(4, 4)) * span
art = root / "instances" / "dev_Rastrigin_d4_i0"
meta = write_instance_artifacts(inst, starts, art)
cfg = build_algorithm_config("r", "smco", False, "none", evolution_strategy="none",
    evolution_points=(), elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=4)
task = build_task("e0_contract", "contract", "Rastrigin", 4, 0, 0, config=cfg,
    fe_budget=200, checkpoints=(50, 100, 200), seed=42,
    instance_artifact_dir="instances/dev_Rastrigin_d4_i0",
    instance_hash=meta["transform_sha256"], start_points_hash=meta["file_hashes"]["starts"])
(root / "task.json").write_text(json.dumps(task))
# raw + log dirs live one level up so log_dir default resolves outside instances/
raw = root / "raw"; log = root / "logs"; raw.mkdir(exist_ok=True); log.mkdir(exist_ok=True)
subprocess.run(["Rscript", "scripts/run_smco_evo_highdim_r.R", "--task", str(root / "task.json"),
    "--instance-root", str(root), "--result-dir", str(raw), "--log-dir", str(log)], check=True)
payload = json.loads((raw / f"{task['run_id']}.json").read_text())
assert payload["status"] == "success", payload
assert isinstance(payload["best_so_far_trace"], list)
assert payload["supersedes_run_id"] == "none"
assert payload["task"]["run_id"] == task["run_id"]
# 未命中 target 必须是 JSON null (Python None)，不是字符串 "NA"
for k, v in payload["target_hit_fe"].items():
    assert v is None or isinstance(v, int), (k, v)
print("R outcome verify OK")
```

- [ ] **Step 4: 跑端到端验证**

Run: `.venv/bin/python /tmp/verify_r_outcome.py`
Expected: `R outcome verify OK`（若无 jsonlite，按记忆本机 R 4.3.2 已装 jsonlite 2.0.0 + qrng 0.0.11）

- [ ] **Step 5: Commit**

```bash
git add scripts/run_smco_evo_highdim_r.R
git commit -m "refactor: R worker emits unified outcome (na=null, trace, supersedes)" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: runner 占位 payload 统一（Py infra 去 result_row）

**Files:**
- Modify: `scripts/run_smco_evo_highdim_factorial.py`（`run_task_file` infra 分支，约第 171–179 行）

- [ ] **Step 1: 实现** — 把 `run_task_file` 里 infra_failure 占位的 `_atomic_write_json(...)` payload（约第 171–179 行）：

```python
            _atomic_write_json(
                result_dir / f"{run_id}.json",
                {
                    "run_id": run_id,
                    "status": "infra_failure",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "result_row": None,
                },
            )
```

改为（去掉 `"result_row": None,`）：

```python
            _atomic_write_json(
                result_dir / f"{run_id}.json",
                {
                    "run_id": run_id,
                    "status": "infra_failure",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                },
            )
```

- [ ] **Step 2: 跑受影响测试**

Run: `.venv/bin/python -m pytest tests/test_highdim_worker.py::test_run_task_file_rejects_instance_hash_mismatch tests/test_highdim_batch.py -v`
Expected: PASS（infra payload 仍写 `{run_id, status:"infra_failure", failure_reason}`，`is_run_complete`/merge 容错）

- [ ] **Step 3: Commit**

```bash
git add scripts/run_smco_evo_highdim_factorial.py
git commit -m "refactor: drop result_row from Py worker infra placeholder payload" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: merge_results — task_index + classify + SMCO/baseline row 构建

**Files:**
- Create: `src/smco/merge_results.py`
- Test: `tests/test_merge_results.py`

- [ ] **Step 1: 写失败测试** — 新建 `tests/test_merge_results.py`：

```python
"""Tests for the unified merge / provenance-audit step (Task 11, redesigned)."""
from __future__ import annotations
import json
import pytest

from smco.experiment_manifests import (
    build_algorithm_config, build_baseline_task, build_task, build_manifest,
    freeze_manifest,
)
from smco.merge_results import (
    baseline_row_from_outcome, build_task_index, classify_task,
    smco_row_from_outcome,
)
from smco.paper_contract import NONE_TOKEN, RESULT_COLUMNS, validate_result_against_task, validate_result_row


def _evo_task():
    cfg = build_algorithm_config("python", "smco", True, "state_preserving",
        evolution_strategy="rand1bin", evolution_points=(0.5, 0.75),
        elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=8)
    return build_task("e1_development", "synthetic_highdim", "Zakharov", 200, 0, 0,
        config=cfg, fe_budget=20000, checkpoints=(5000, 10000), seed=12345,
        instance_hash="ihash", start_points_hash="shash")


def _baseline_task():
    return build_baseline_task("e3_baselines_highdim", "synthetic_highdim", "Zakharov",
        200, 0, algorithm="DE", fe_budget=20000, checkpoints=(5000, 10000), seed=77,
        instance_hash="ihash", start_points_hash="shash")


def _smco_outcome(task):
    return {"run_id": task["run_id"], "status": "success", "failure_reason": "none",
        "fe_used": 19998, "fe_budget": 20000, "best_value": 1e-6, "known_optimum": 0.0,
        "normalized_gap": 0.001, "target_hit_fe": {"1e-1": 500, "1e-2": 5000, "1e-3": None, "1e-5": None},
        "anytime": [], "best_so_far_trace": [[500, 0.1]], "termination_reason": "evaluation_budget",
        "fe_counts_by_event": {"initialization": 1}, "wall_time_sec": 1.0, "peak_memory_mb": 10.0,
        "machine_id": "h", "git_commit": "abc", "environment_hash": "env", "task": task,
        "algorithm_id": task["algorithm_id"], "supersedes_run_id": "none"}


def test_classify_task():
    assert classify_task(_evo_task()) == "smco"
    assert classify_task(_baseline_task()) == "baseline"


def test_build_task_index_loads_all_manifests(tmp_path):
    m1 = freeze_manifest(build_manifest("e1_development", "synthetic_highdim", [_evo_task()]))
    m2 = freeze_manifest(build_manifest("e3_baselines_highdim", "synthetic_highdim", [_baseline_task()]))
    p1 = tmp_path / "m1.json"; p1.write_text(json.dumps(m1))
    p2 = tmp_path / "m2.json"; p2.write_text(json.dumps(m2))
    idx = build_task_index([p1, p2])
    assert set(idx) == {_evo_task()["run_id"], _baseline_task()["run_id"]}


def test_smco_row_from_outcome_is_contract_valid_and_consistent():
    task = _evo_task(); row = smco_row_from_outcome(_smco_outcome(task), task, manifest_id="m")
    assert set(row) == set(RESULT_COLUMNS)
    assert validate_result_row(row) == []
    assert validate_result_against_task(row, task) == []
    assert row["target_hit_fe_1e-3"] == NONE_TOKEN  # null -> NONE_TOKEN
    assert row["target_hit_fe_1e-1"] == 500


def test_smco_row_tolerates_null_best_value():
    task = _evo_task()
    oc = _smco_outcome(task); oc["best_value"] = None; oc["status"] = "infra_failure"
    row = smco_row_from_outcome(oc, task)
    assert row["status"] == "infra_failure"
    import math
    assert math.isnan(row["best_value"])


def test_baseline_row_from_outcome_has_columns_and_algorithm():
    task = _baseline_task()
    oc = {"run_id": task["run_id"], "status": "success", "failure_reason": "none",
        "fe_used": 20000, "best_value": 0.5, "known_optimum": 0.0, "normalized_gap": 0.5,
        "target_hit_fe": {"1e-1": 100, "1e-2": None, "1e-3": None, "1e-5": None},
        "anytime": [], "best_so_far_trace": [], "termination_reason": "evaluation_budget",
        "fe_counts_by_event": {}, "wall_time_sec": 2.0, "peak_memory_mb": None,
        "machine_id": "h", "git_commit": "", "environment_hash": "env",
        "supersedes_run_id": "none"}
    row = baseline_row_from_outcome(oc, task, manifest_id="m")
    assert set(row) == set(RESULT_COLUMNS)
    assert row["algorithm_id"] == "DE"
    assert row["family"] == NONE_TOKEN
    assert row["configuration_hash"] == NONE_TOKEN
    assert row["is_confirmatory"] is True  # e3 is confirmatory
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_merge_results.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'smco.merge_results'`

- [ ] **Step 3: 实现** — 新建 `src/smco/merge_results.py`：

```python
"""Merge + provenance audit for the SMCO-EVO high-dim paper (Task 11, redesigned).

All three workers (Py SMCO / R SMCO / baseline) emit one unified outcome payload.
This module is the single place that builds ``RESULT_COLUMNS`` rows from an
outcome plus its frozen manifest task, resolves supersedes, runs the provenance
audit and writes the ``merged/`` artefacts. See
``docs/superpowers/specs/2026-07-29-smco-evo-unified-output-contract-design.md``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .experiment_manifests import (
    derive_seed,
    load_manifest,
    result_row_from_task,
    verify_manifest,
)
from .paper_contract import NONE_TOKEN, RESULT_COLUMNS, SCHEMA_VERSION, STATUSES

_CONFIRMATORY_STAGES = {
    "e2_factorial_highdim", "e3_baselines_highdim",
    "e4_bbob_largescale", "e5_lowdim_check",
}
_NAN = float("nan")


def classify_task(task: dict) -> str:
    """'smco' if the task carries configuration_hash, else 'baseline'."""
    return "smco" if "configuration_hash" in task else "baseline"


def build_task_index(manifest_paths: Iterable[str]) -> dict[str, dict]:
    """Load + verify all manifests; return {run_id: task}."""
    index: dict[str, dict] = {}
    for path in manifest_paths:
        manifest = load_manifest(path)
        verify_manifest(manifest)
        for task in manifest.get("tasks", []):
            index[task["run_id"]] = task
    return index


def _num(value, default=_NAN):
    return default if value is None else value


def smco_row_from_outcome(outcome: dict, task: dict, manifest_id: str = "") -> dict:
    """Build a contract-valid SMCO RESULT_COLUMNS row from outcome + task."""
    th = {k: v for k, v in (outcome.get("target_hit_fe") or {}).items() if v is not None}
    gap = outcome.get("normalized_gap")
    return result_row_from_task(
        task,
        best_value=_num(outcome.get("best_value")),
        fe_used=int(outcome.get("fe_used") or 0),
        status=outcome.get("status", "infra_failure"),
        known_optimum=_num(outcome.get("known_optimum"), 0.0),
        normalized_gap=NONE_TOKEN if gap is None else gap,
        checkpoint_fe=task["fe_budget"],
        target_hit_fe=th,
        wall_time_sec=float(outcome.get("wall_time_sec") or 0.0),
        peak_memory_mb=float(outcome.get("peak_memory_mb") or 0.0),
        failure_reason=outcome.get("failure_reason", NONE_TOKEN),
        termination_reason=outcome.get("termination_reason", "evaluation_budget"),
        fe_counts_by_event=str(outcome.get("fe_counts_by_event") or {}),
        machine_id=outcome.get("machine_id", ""),
        git_commit=outcome.get("git_commit", ""),
        environment_hash=outcome.get("environment_hash", ""),
        objective_sense="minimize",
        manifest_id=manifest_id,
        supersedes_run_id=outcome.get("supersedes_run_id", NONE_TOKEN),
    )


def _th_cell(th: dict, label: str):
    v = (th or {}).get(label)
    return NONE_TOKEN if v is None else v


def baseline_row_from_outcome(outcome: dict, task: dict, manifest_id: str = "") -> dict:
    """Build a RESULT_COLUMNS row for a baseline run (algorithm_id = DE/GenSA/...).

    Baseline rows bypass ``validate_result_row``'s SMCO ``algorithm_id`` rebuild
    check; only field presence + numeric sanity is enforced (audit step).
    """
    th = outcome.get("target_hit_fe") or {}
    gap = outcome.get("normalized_gap")
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "stage": task["stage"],
        "suite": task.get("suite", "synthetic_highdim"),
        "function": task["function"],
        "dimension": int(task["dimension"]),
        "instance": int(task["instance"]),
        "replication": 0,
        "seed": int(task["seed"]),
        "language": "python",
        "state_semantics": NONE_TOKEN,
        "family": NONE_TOKEN,
        "evolutionary": "false",
        "evolution_strategy": NONE_TOKEN,
        "algorithm_id": task["algorithm"],
        "n_starts": 0,
        "fe_budget": int(task["fe_budget"]),
        "fe_used": int(outcome.get("fe_used") or 0),
        "checkpoint_fe": int(task["fe_budget"]),
        "best_value": _num(outcome.get("best_value")),
        "known_optimum": _num(outcome.get("known_optimum"), 0.0),
        "normalized_gap": NONE_TOKEN if gap is None else gap,
        "objective_sense": "minimize",
        "target_hit_fe_1e-1": _th_cell(th, "1e-1"),
        "target_hit_fe_1e-2": _th_cell(th, "1e-2"),
        "target_hit_fe_1e-3": _th_cell(th, "1e-3"),
        "target_hit_fe_1e-5": _th_cell(th, "1e-5"),
        "wall_time_sec": float(outcome.get("wall_time_sec") or 0.0),
        "peak_memory_mb": float(outcome.get("peak_memory_mb") or 0.0),
        "status": outcome.get("status", "infra_failure"),
        "failure_reason": outcome.get("failure_reason", NONE_TOKEN),
        "is_confirmatory": task["stage"] in _CONFIRMATORY_STAGES,
        "supersedes_run_id": outcome.get("supersedes_run_id", NONE_TOKEN),
        "machine_id": outcome.get("machine_id", ""),
        "git_commit": outcome.get("git_commit", ""),
        "environment_hash": outcome.get("environment_hash", ""),
        "start_points_hash": task.get("start_points_hash") or NONE_TOKEN,
        "instance_hash": task.get("instance_hash") or NONE_TOKEN,
        "configuration_hash": NONE_TOKEN,
        "run_id": task["run_id"],
        "termination_reason": outcome.get("termination_reason", "evaluation_budget"),
        "fe_counts_by_event": str(outcome.get("fe_counts_by_event") or {}),
    }


__all__ = [
    "classify_task", "build_task_index",
    "smco_row_from_outcome", "baseline_row_from_outcome",
]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_merge_results.py -v`
Expected: PASS（全部）

- [ ] **Step 5: Commit**

```bash
git add src/smco/merge_results.py tests/test_merge_results.py
git commit -m "feat(merge): single-point row construction (SMCO + baseline) from outcome+task" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: merge_results — supersedes 解析 + provenance audit

**Files:**
- Modify: `src/smco/merge_results.py`（追加函数）
- Test: `tests/test_merge_results.py`（追加测试）

- [ ] **Step 1: 写失败测试** — 在 `tests/test_merge_results.py` 追加：

```python
from smco.merge_results import resolve_supersedes, audit_payloads, _identity_key


def _row(run_id, **kw):
    base = {"function": "Zakharov", "dimension": 200, "instance": 0,
            "algorithm_id": "PY-SP-SMCO-EVO", "language": "python",
            "state_semantics": "state_preserving", "evolution_strategy": "rand1bin",
            "seed": 1, "run_id": run_id, "stage": "e2_factorial_highdim",
            "suite": "synthetic_highdim", "fe_budget": 1000, "fe_used": 999,
            "objective_sense": "minimize", "best_value": 1e-6, "known_optimum": 0.0,
            "normalized_gap": 0.01, "family": "smco", "evolutionary": "true",
            "configuration_hash": "cfg", "start_points_hash": "sh",
            "instance_hash": "ih", "supersedes_run_id": "none", "status": "success"}
    base.update(kw)
    return base


def test_resolve_supersedes_excludes_superseded():
    rows = [_row("r1"), _row("r2", supersedes_run_id="r1")]
    valid, superseded = resolve_supersedes(rows)
    assert [r["run_id"] for r in valid] == ["r2"]
    assert superseded == {"r1"}


def test_identity_key_detects_duplicate_identity():
    a = _row("r1"); b = _row("r2")  # same identity, different run_id
    assert _identity_key(a) == _identity_key(b)


def test_audit_passes_clean_rows():
    task = _evo_task()
    row = smco_row_from_outcome(_smco_outcome(task), task)
    audit = audit_payloads([row], {task["run_id"]: task})
    assert audit["passed"] is True, audit


def test_audit_flags_fe_over_budget():
    task = _evo_task()
    row = smco_row_from_outcome(_smco_outcome(task), task)
    row["fe_used"] = task["fe_budget"] + 1
    audit = audit_payloads([row], {task["run_id"]: task})
    assert audit["passed"] is False
    assert any("fe_over_budget" in c["name"] for c in audit["checks"])


def test_audit_flags_wrong_seed():
    task = _evo_task()
    row = smco_row_from_outcome(_smco_outcome(task), task)
    row["seed"] = task["seed"] + 1
    audit = audit_payloads([row], {task["run_id"]: task})
    assert audit["passed"] is False
    assert any("seed" in c["name"] for c in audit["checks"])


def test_audit_flags_duplicate_identity():
    task = _evo_task()
    row = smco_row_from_outcome(_smco_outcome(task), task)
    dup = dict(row); dup["run_id"] = "r_other"
    audit = audit_payloads([row, dup], {task["run_id"]: task, "r_other": task})
    assert audit["passed"] is False
    assert any("duplicate" in c["name"] for c in audit["checks"])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_merge_results.py -v -k "supersedes or audit or identity"`
Expected: FAIL — `ImportError: cannot import name 'resolve_supersedes'`

- [ ] **Step 3: 实现** — 在 `src/smco/merge_results.py` 的 `__all__` 之前追加：

```python
def resolve_supersedes(rows: list[dict]) -> tuple[list[dict], set[str]]:
    """Split rows into (valid, superseded_run_ids).

    A row whose ``supersedes_run_id`` is a real run_id removes that run_id from
    the valid set (it stays in all_attempts).
    """
    superseded: set[str] = set()
    for row in rows:
        sup = row.get("supersedes_run_id")
        if sup and sup != NONE_TOKEN:
            superseded.add(sup)
    valid = [r for r in rows if r["run_id"] not in superseded]
    return valid, superseded


def _identity_key(row: dict) -> tuple:
    """Identity (excluding run_id) — same key => duplicate unless supersedes."""
    return (
        row["function"], int(row["dimension"]), int(row["instance"]),
        row["algorithm_id"], row["language"], row["state_semantics"],
        row["evolution_strategy"], int(row["seed"]),
    )


def _check(name, rows, ok, errors):
    return {"name": name, "passed": ok, "n": len(rows), "errors": errors}


def audit_payloads(rows: list[dict], task_index: dict[str, dict]) -> dict:
    """Run the 11 provenance checks; return {passed, checks, summary}.

    ``passed=False`` does not crash the merge — the analysis layer (Task 12)
    refuses to build primary tables when the audit fails.
    """
    checks: list[dict] = []

    # 1. run_id uniqueness
    ids = [r["run_id"] for r in rows]
    dup_ids = sorted({i for i in ids if ids.count(i) > 1})
    checks.append(_check("run_id_uniqueness", rows, not dup_ids,
                         [f"duplicate run_id: {i}" for i in dup_ids]))

    # 2. manifest coverage (orphans: run_id not in task_index)
    orphans = [r["run_id"] for r in rows if r["run_id"] not in task_index]
    checks.append(_check("manifest_coverage", rows, not orphans,
                         [f"run_id not in any manifest: {o}" for o in orphans]))

    # 3. supersedes target exists
    known = set(ids) | set(task_index)
    dangling = [r["supersedes_run_id"] for r in rows
                if r.get("supersedes_run_id") not in (NONE_TOKEN, None)
                and r["supersedes_run_id"] not in known]
    checks.append(_check("supersedes_resolvable", rows, not dangling,
                         [f"supersedes unknown run_id: {d}" for d in dangling]))

    # 4. configuration_hash consistent with task (SMCO only)
    bad_cfg = []
    for r in rows:
        t = task_index.get(r["run_id"])
        if t and "configuration_hash" in t and r.get("configuration_hash") != t["configuration_hash"]:
            bad_cfg.append(r["run_id"])
    checks.append(_check("configuration_hash_consistent", rows, not bad_cfg,
                         [f"hash mismatch: {b}" for b in bad_cfg]))

    # 5. FE <= budget
    over = [r["run_id"] for r in rows if int(r["fe_used"]) > int(r["fe_budget"])]
    checks.append(_check("fe_within_budget", rows, not over,
                         [f"fe_over_budget: {o}" for o in over]))

    # 6. objective direction
    wrong_dir = [r["run_id"] for r in rows if r.get("objective_sense") != "minimize"]
    checks.append(_check("objective_direction", rows, not wrong_dir,
                         [f"non-minimize: {w}" for w in wrong_dir]))

    # 7. known_optimum / gap sanity (best >= optimum - tol in minimisation)
    bad_gap = []
    for r in rows:
        try:
            if r["best_value"] < r["known_optimum"] - 1e-6:
                bad_gap.append(r["run_id"])
        except TypeError:
            pass  # NaN best (infra/timeout) stays in the denominator, not a gap error
    checks.append(_check("gap_sanity", rows, not bad_gap,
                         [f"best<optimum: {b}" for b in bad_gap]))

    # 8. start_points_hash consistent within (function,dim,instance)
    by_inst: dict[tuple, set] = {}
    for r in rows:
        key = (r["function"], int(r["dimension"]), int(r["instance"]))
        by_inst.setdefault(key, set()).add(r.get("start_points_hash"))
    clash = [f"{k}" for k, v in by_inst.items() if len(v) > 1]
    checks.append(_check("start_points_hash_consistent", rows, not clash,
                         [f"instance has multiple starts hashes: {c}" for c in clash]))

    # 9. non-EVO rows not duplicated by strategy + identity duplicates
    bad_strategy = [r["run_id"] for r in rows
                    if r["evolutionary"] == "false" and r["evolution_strategy"] != NONE_TOKEN]
    seen: dict[tuple, list[str]] = {}
    for r in rows:
        seen.setdefault(_identity_key(r), []).append(r["run_id"])
    dups = [rids for rids in seen.values() if len(rids) > 1]
    checks.append(_check("no_pseudo_duplicates", rows, not bad_strategy and not dups,
                         [f"base row has strategy: {b}" for b in bad_strategy]
                         + [f"identity duplicated: {rids}" for rids in dups]))

    # 10. confirmatory seed equals derive_seed(stage,...,algorithm_id)
    bad_seed = []
    for r in rows:
        t = task_index.get(r["run_id"])
        if not t:
            continue
        algo = t.get("algorithm_id") or t.get("algorithm")
        expected = derive_seed(t["stage"], t.get("suite", "synthetic_highdim"),
                               t["function"], int(t["dimension"]), int(t["instance"]),
                               0, algo)
        if int(r["seed"]) != int(expected):
            bad_seed.append(r["run_id"])
    checks.append(_check("seed_matches_derive_seed", rows, not bad_seed,
                         [f"seed mismatch (possible dev seed): {b}" for b in bad_seed]))

    # 11. statuses are all in the contract vocabulary (kept in the denominator)
    bad_status = [r["run_id"] for r in rows if r["status"] not in STATUSES]
    checks.append(_check("status_vocabulary", rows, not bad_status,
                         [f"unknown status: {b}" for b in bad_status]))

    failed = [c["name"] for c in checks if not c["passed"]]
    return {
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "n_rows": len(rows),
    }
```

并把 `__all__` 追加 `"resolve_supersedes", "audit_payloads"`（`_identity_key` 是私有，不导出，但测试可直接引用）。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_merge_results.py -v`
Expected: PASS（全部）

- [ ] **Step 5: Commit**

```bash
git add src/smco/merge_results.py tests/test_merge_results.py
git commit -m "feat(merge): supersedes resolution + 11-check provenance audit" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: merge_results — 输出 + 主流程 + CLI + 端到端

**Files:**
- Modify: `src/smco/merge_results.py`（追加 `load_raw_outcomes` / `merge`）
- Create: `scripts/merge_smco_evo_highdim_results.py`
- Test: `tests/test_merge_results.py`（追加端到端测试）

- [ ] **Step 1: 写失败测试** — 在 `tests/test_merge_results.py` 追加：

```python
from smco.merge_results import merge


def _write(raw_dir, run_id, payload):
    (raw_dir / f"{run_id}.json").write_text(json.dumps(payload))


def test_merge_end_to_end_writes_all_artefacts(tmp_path):
    task = _evo_task()
    btask = _baseline_task()
    manifest = freeze_manifest(build_manifest("e1_development", "synthetic_highdim", [task]))
    bmanifest = freeze_manifest(build_manifest("e3_baselines_highdim", "synthetic_highdim", [btask]))
    mp = tmp_path / "m.json"; mp.write_text(json.dumps(manifest))
    bp = tmp_path / "bm.json"; bp.write_text(json.dumps(bmanifest))
    raw = tmp_path / "raw"; raw.mkdir()
    _write(raw, task["run_id"], _smco_outcome(task))
    boc = {"run_id": btask["run_id"], "status": "success", "failure_reason": "none",
        "fe_used": 20000, "fe_budget": 20000, "best_value": 0.4, "known_optimum": 0.0,
        "normalized_gap": 0.4, "target_hit_fe": {"1e-1": 100, "1e-2": None, "1e-3": None, "1e-5": None},
        "anytime": [], "best_so_far_trace": [], "termination_reason": "evaluation_budget",
        "fe_counts_by_event": {}, "wall_time_sec": 1.0, "peak_memory_mb": None,
        "machine_id": "h", "git_commit": "", "environment_hash": "env", "task": btask,
        "algorithm_id": "DE", "supersedes_run_id": "none"}
    _write(raw, btask["run_id"], boc)

    merged = tmp_path / "merged"
    summary = merge([mp, bp], [raw], merged)

    import csv
    all_rows = list(csv.DictReader(open(merged / "all_attempts.csv")))
    valid = list(csv.DictReader(open(merged / "valid_runs.csv")))
    missing = list(csv.DictReader(open(merged / "missing_runs.csv")))
    assert len(all_rows) == 2
    assert len(valid) == 2
    assert {r["algorithm_id"] for r in valid} == {task["algorithm_id"], "DE"}
    assert summary["audit"]["passed"] is True
    # missing = manifest tasks with no raw
    assert merged.joinpath("provenance_audit.json").exists()
    assert merged.joinpath("provenance_audit.md").exists()
    assert merged.joinpath("anytime.csv").exists()


def test_merge_reports_missing_runs(tmp_path):
    task = _evo_task()
    manifest = freeze_manifest(build_manifest("e1_development", "synthetic_highdim", [task]))
    mp = tmp_path / "m.json"; mp.write_text(json.dumps(manifest))
    raw = tmp_path / "raw"; raw.mkdir()
    summary = merge([mp], [raw], tmp_path / "merged")
    missing = list(csv.DictReader(open(tmp_path / "merged" / "missing_runs.csv")))
    assert len(missing) == 1
    assert missing[0]["run_id"] == task["run_id"]
```

（顶部 `import csv` 加到测试文件 import 区。）

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_merge_results.py -v -k merge`
Expected: FAIL — `ImportError: cannot import name 'merge'`

- [ ] **Step 3: 实现** — 在 `src/smco/merge_results.py` 的 `__all__` 之前追加：

```python
def load_raw_outcomes(raw_dirs: Iterable[str]):
    """Yield (path, payload) for every <run_id>.json across raw_dirs."""
    for raw_dir in raw_dirs:
        for path in sorted(Path(raw_dir).glob("*.json")):
            if path.name.startswith(".") or ".tmp" in path.name:
                continue
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            if isinstance(payload, dict) and "run_id" in payload:
                yield path, payload


def _write_csv(path: Path, columns, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _audit_md(audit: dict) -> str:
    lines = ["# Provenance Audit", "",
             f"**Passed:** {audit['passed']}", f"**Rows:** {audit['n_rows']}", ""]
    for c in audit["checks"]:
        flag = "PASS" if c["passed"] else "FAIL"
        lines.append(f"- [{flag}] {c['name']}")
        for e in c["errors"]:
            lines.append(f"    - {e}")
    return "\n".join(lines) + "\n"


def merge(manifest_paths, raw_dirs, merged_dir) -> dict:
    """Load outcomes, build rows, resolve supersedes, audit, write merged/."""
    import csv
    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    task_index = build_task_index(manifest_paths)

    attempts: list[dict] = []
    for _path, outcome in load_raw_outcomes(raw_dirs):
        run_id = outcome["run_id"]
        task = task_index.get(run_id)
        if task is None:
            continue  # orphan — recorded in audit via coverage; row dropped here
        if classify_task(task) == "smco":
            attempts.append(smco_row_from_outcome(outcome, task, manifest_id=run_id))
        else:
            attempts.append(baseline_row_from_outcome(outcome, task, manifest_id=run_id))

    valid, superseded = resolve_supersedes(attempts)
    audit = audit_payloads(attempts, task_index)

    # missing = manifest tasks with no raw outcome
    have = {r["run_id"] for r in attempts}
    missing = [{"run_id": t["run_id"], "stage": t["stage"], "function": t["function"],
                "dimension": t["dimension"], "instance": t["instance"],
                "algorithm_id": t.get("algorithm_id") or t.get("algorithm")}
               for t in task_index.values() if t["run_id"] not in have]

    # anytime long table from raw outcomes
    anytime_rows = []
    for _path, outcome in load_raw_outcomes(raw_dirs):
        for a in outcome.get("anytime") or []:
            anytime_rows.append({
                "run_id": outcome["run_id"],
                "checkpoint_fe": a.get("checkpoint_fe"),
                "fe_used": a.get("fe_used"),
                "best_value": a.get("best_value"),
                "normalized_gap": a.get("normalized_gap"),
            })

    _write_csv(merged_dir / "all_attempts.csv", RESULT_COLUMNS, attempts)
    _write_csv(merged_dir / "valid_runs.csv", RESULT_COLUMNS, valid)
    _write_csv(merged_dir / "missing_runs.csv",
               ("run_id", "stage", "function", "dimension", "instance", "algorithm_id"), missing)
    _write_csv(merged_dir / "duplicate_runs.csv", RESULT_COLUMNS,
               [r for r in attempts if r["run_id"] in {a for a in superseded}])
    _write_csv(merged_dir / "anytime.csv",
               ("run_id", "checkpoint_fe", "fe_used", "best_value", "normalized_gap"), anytime_rows)
    (merged_dir / "provenance_audit.json").write_text(json.dumps(audit, indent=2))
    (merged_dir / "provenance_audit.md").write_text(_audit_md(audit))

    return {"n_attempts": len(attempts), "n_valid": len(valid),
            "n_missing": len(missing), "audit": audit}
```

并把 `__all__` 追加 `"load_raw_outcomes"`, `"merge"`。

- [ ] **Step 4: 实现 CLI** — 新建 `scripts/merge_smco_evo_highdim_results.py`：

```python
#!/usr/bin/env python
"""Merge SMCO-EVO high-dim raw outcomes into merged/ artefacts (Task 11).

Reads frozen manifests + raw outcome dirs, builds RESULT_COLUMNS rows from
outcome + manifest task at a single Python point, resolves supersedes, runs the
provenance audit, and writes merged/{all_attempts,valid_runs,missing_runs,
duplicate_runs,anytime}.csv + provenance_audit.{json,md}.

Usage:
    python scripts/merge_smco_evo_highdim_results.py \
        --manifest m1.json [m2.json ...] \
        --raw-dir raw1 [raw2 ...] \
        --merged-dir merged/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smco.merge_results import merge  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", nargs="+", required=True, help="Frozen manifest JSON(s).")
    parser.add_argument("--raw-dir", nargs="+", required=True, help="raw outcome dir(s).")
    parser.add_argument("--merged-dir", required=True, help="Output dir for merged/ artefacts.")
    args = parser.parse_args(argv)
    summary = merge(args.manifest, args.raw_dir, args.merged_dir)
    print(json.dumps(summary, indent=2))
    return 0 if summary["audit"]["passed"] else 2  # exit 2 if audit failed (analysis gate)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_merge_results.py -v`
Expected: PASS（全部）

- [ ] **Step 6: 手动冒烟 CLI**

Run:
```bash
.venv/bin/python -c "
import json; from pathlib import Path
from smco.experiment_manifests import build_algorithm_config, build_task, build_manifest, freeze_manifest
cfg = build_algorithm_config('python','smco',False,'none',evolution_strategy='none',evolution_points=(),elimination_rate=0.25,de_factor=0.8,de_crossover=0.7,n_starts=4)
t = build_task('e0_contract','contract','Rastrigin',4,0,0,config=cfg,fe_budget=200,checkpoints=(100,200),seed=42,instance_hash='ih',start_points_hash='sh')
m = freeze_manifest(build_manifest('e0_contract','contract',[t]))
Path('/tmp/merge_smoke/m.json').parent.mkdir(parents=True,exist_ok=True)
Path('/tmp/merge_smoke/m.json').write_text(json.dumps(m))
oc = {'run_id':t['run_id'],'status':'success','failure_reason':'none','fe_used':199,'fe_budget':200,'best_value':1e-6,'known_optimum':0.0,'normalized_gap':0.01,'target_hit_fe':{'1e-1':50,'1e-2':None,'1e-3':None,'1e-5':None},'anytime':[],'best_so_far_trace':[[50,0.1]],'termination_reason':'evaluation_budget','fe_counts_by_event':{'initialization':1},'wall_time_sec':0.1,'peak_memory_mb':5.0,'machine_id':'h','git_commit':'a','environment_hash':'e','task':t,'algorithm_id':t['algorithm_id'],'supersedes_run_id':'none'}
Path('/tmp/merge_smoke/raw').mkdir(exist_ok=True)
Path('/tmp/merge_smoke/raw/'+t['run_id']+'.json').write_text(json.dumps(oc))
"
.venv/bin/python scripts/merge_smco_evo_highdim_results.py --manifest /tmp/merge_smoke/m.json --raw-dir /tmp/merge_smoke/raw --merged-dir /tmp/merge_smoke/merged
```
Expected: JSON summary with `"passed": true`, exit code 0; `ls /tmp/merge_smoke/merged` shows all 6 files.

- [ ] **Step 7: Commit**

```bash
git add src/smco/merge_results.py scripts/merge_smco_evo_highdim_results.py tests/test_merge_results.py
git commit -m "feat(merge): merge main flow + merged/ outputs + CLI" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: 全量验证 + Gate D pilot 文档更新

**Files:**
- Verify: whole test suite
- Modify: `docs/gate-d-pilot-2026-07-29.md`

- [ ] **Step 1: 全量 pytest**

Run: `.venv/bin/python -m pytest -q`
Expected: 全绿（之前 331 passed + 本计划新增测试；无 fail/error）。若有失败，修复后再继续——不要在红时 commit。

- [ ] **Step 2: R worker 端到端再确认**（若 Task 4 的 fixture 已清，重跑一次）

Run: `.venv/bin/python /tmp/verify_r_outcome.py`
Expected: `R outcome verify OK`

- [ ] **Step 3: 更新 Gate D pilot 字段表** — 在 `docs/gate-d-pilot-2026-07-29.md` 找到 result payload 字段表，把字段列表更新为统一 outcome 字段集（加 `best_so_far_trace`、`task`、`supersedes_run_id`；注明 `result_row` 已移至 merge 产出）。加一行注记："2026-07-29: 按 unified-output-contract 重构，worker 输出统一详尽 outcome，result_row 由 merge 单点构建。"

- [ ] **Step 4: Commit**

```bash
git add docs/gate-d-pilot-2026-07-29.md
git commit -m "docs: update Gate D pilot payload fields for unified outcome contract" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review（计划作者自检，已完成）

- **Spec coverage**：spec §4.2 字段表 → Task 1（契约）+ Task 2/3/4（worker）；§4.3 缺失值约定 → Task 6（`_num`/`_th_cell`）+ Task 4（R `na="null"`）；§4.4 worker 改动 → Task 2/3/4/5；§4.5 merge → Task 6/7/8；§4.6 `OUTCOME_FIELDS` → Task 1；§5 测试 → 各 Task 的 TDD 步骤；§6 影响 → 全覆盖。analysis 输出（§3 非目标）不在本计划。
- **Placeholder scan**：无 TBD/TODO；每个代码步含完整代码。
- **Type consistency**：`smco_row_from_outcome` / `baseline_row_from_outcome` / `resolve_supersedes` / `audit_payloads` / `merge` 在定义 task 与使用 task 中签名一致；`OUTCOME_FIELDS` 字段名与 worker 返回键一致。
