# SMCO-EVO 高维论文：全链条统一输出契约设计

- 日期：2026-07-29
- 分支：`feat/smco-evo-highdim-paper-2026`
- 关联：实现计划 `docs/smco-evo-highdim-implementation-plan-2026-07-28.md` 的 Task 8（Py/R worker）、Task 9（baseline worker）、Task 11（merge）
- 状态：设计已与用户对齐（方案 B + 详尽 raw + merge 单点构建 row）

## 1. 背景与动机

三种 worker 当前各自写 `raw/<run_id>.json`，字段结构不一致：

| worker | `result_row` | `peak_memory_mb` | 身份/配置 | `algorithm_id` |
|---|---|---|---|---|
| Python SMCO | 有（完整） | 有 | payload 扁平 | SMCO 标准 |
| R SMCO | 无（嵌入了 `task`） | `NA` | 嵌在 `task` 里 | SMCO 标准 |
| baseline | 无 | 缺字段 | 扁平但字段少 | 非标准（DE/GenSA） |

后果：Task 11 merge 必须为 R 用 `result_row_from_task` 重建 row、为 baseline bypass `validate_result_row` 的 SMCO 重建检查——两条特殊路径，是 bug 温床，也使链条缺乏统一契约。`best-so-far` 改进点序列目前只存在于 worker 内存中，落盘即丢失，导致 anytime / target_hit / ECDF 只能在固定 checkpoints 上分析。

**关键时机**：E1 全量（Gate E）尚未启动，**无历史 raw 需迁移**——现在统一是零成本窗口。

## 2. 目标

1. 一套统一的 outcome payload **形式**（字段名、类型、缺失值约定）贯穿 worker → merge → analysis。
2. raw payload **尽可能详细**：保留嵌入 `task`、新增 `best_so_far_trace`、保留全部 FE/质量/provenance 字段。信息只增不减。
3. 权威扁平记录 `result_row` 由 merge 在 **Python 单点**构建；R/baseline 不需懂 contract。

## 3. 非目标

- analysis 输出表/图契约（留 Task 12）。
- 重算/迁移任何已有结果（无历史 raw）。
- 记录每次函数评价（百万 FE 不可行；`best_so_far_trace` 只记改进点，已有界）。

## 4. 设计

### 4.1 数据流

```
worker(Py / R / baseline) ──统一详尽 outcome──▶ raw/<run_id>.json
                                                       │
            frozen manifest task ◀────────────────────┘
                          │
          merge：outcome + task → result_row（Python 单点构建）
                          │
          merged/all_attempts.csv        （全部 attempt，含被取代）
          merged/valid_runs.csv          （去重/去被取代 + audit 通过）
          merged/missing_runs.csv        （manifest 有 task、无 raw）
          merged/duplicate_runs.csv      （真重复 run_id）
          merged/anytime.csv             （长表：run_id × checkpoint）
          merged/provenance_audit.{json,md}
                          │
                    analysis（Task 12）
```

raw 在信息量上**严格 ⊇ result_row**：`result_row` 的每一列都可从 outcome + 嵌入 `task` 派生，而 `best_so_far_trace` 是 row 中没有的更详细原始材料。

### 4.2 统一详尽 outcome payload

权威字段集定义在 `paper_contract.OUTCOME_FIELDS`（worker 与 merge 共用的 single source of truth）。

| 字段 | 类型 | Py-SMCO | R-SMCO | baseline | 说明 |
|---|---|---|---|---|---|
| `run_id` | str | ✓ | ✓ | ✓ | 关联 manifest task |
| `status` | str | ✓ | ✓ | ✓ | success/algorithm_failure/infra_failure/timeout |
| `failure_reason` | str | ✓ | ✓ | ✓ | `"none"` 或错误描述 |
| `fe_used` `fe_budget` | int | ✓ | ✓ | ✓ | |
| `best_value` `known_optimum` `normalized_gap` | float\|null | ✓ | ✓ | ✓ | minimization 方向；未算出为 null |
| `target_hit_fe` | {1e-1,1e-2,1e-3,1e-5}→int\|null | ✓ | ✓ | ✓ | **未命中统一 null**（R 不再 NA） |
| `anytime` | list[{checkpoint_fe,fe_used,best_value,normalized_gap}] | ✓ | ✓ | ✓ | 固定 checkpoints |
| `best_so_far_trace` | list[[fe:int, best_min:float]] | **新增** | **新增** | **新增** | 改进点序列（有界） |
| `termination_reason` | str | ✓ | ✓ | **新增** | baseline 现补 |
| `fe_counts_by_event` | dict | ✓ | ✓ | `{}` | per-event FE 计数；统一 JSON object；baseline 无则 `{}` |
| `wall_time_sec` | float | ✓ | ✓ | ✓ | |
| `peak_memory_mb` | float\|null | ✓ | null | **新增(null)** | R/baseline 无可移植 ru_maxrss |
| `machine_id` `git_commit` `environment_hash` | str | ✓ | ✓ | ✓ | provenance |
| `task` | dict | **新增** | ✓(保留) | **新增** | 嵌入完整 task（含 `algorithm_config`） |
| `algorithm_id` | str | 冗余可省 | 冗余可省 | ✓(必需) | baseline 为 DE/GenSA |
| `supersedes_run_id` | str | **新增** | **新增** | **新增** | 默认 `"none"` |

**"新增" = 该 worker 当前未输出、本设计要求补齐。**

### 4.3 缺失值与类型约定

- 缺失数值统一用 JSON `null`（不再混用 R 的 `NA`）。merge 构建 row 时：`null` 数值 → `NaN`（保留进分母）；`null` target → `NONE_TOKEN`；`null` `peak_memory_mb` → `0.0`。
- `fe_counts_by_event` 统一为 JSON object（R 的 named list → object；baseline 无 per-event 计数 → `{}`）。merge 构建 row 时 `str(dict)` 写入该列（row 列层是字符串）。
- `best_value` 在 infra_failure/timeout 下可为 null → row 中 `NaN`，`status` 保留，进入失败/超时分母（不丢弃）。
- `task` 嵌入字段必须与 frozen manifest 中对应 task 的身份字段逐字一致（merge 的 audit 项）。

### 4.4 worker 改动

**Python `highdim_worker.run_task`**
- 删除 `result_row_from_task` 调用与返回的 `result_row` 键。
- 返回值改为统一 outcome：补 `supersedes_run_id="none"`、`git_commit`/`environment_hash`（runner 透传）、嵌入 `task`、新增 `best_so_far_trace`（来自 `observer.trace`，转 `[[fe, best_min], ...]`）。
- `termination_reason`/`fe_counts_by_event` 已有，保留。

**R `run_smco_evo_highdim_r.R`**
- 保留嵌入 `task`（不变）。
- `target_hit_fe` 未命中由 `NA` 改写为 `null`（`jsonlite` 写 `NULL`）。
- `peak_memory_mb` 保持 `null`。
- 新增 `best_so_far_trace`（`.obs$trace_fe`/`.obs$trace_val` → `list(c(fe, val))`）。
- 新增 `supersedes_run_id = "none"`。
- `fe_counts_by_event` 确保 object（named list）。
- `termination_reason` 已有，保留。

**baseline `baseline_worker.run_baseline_task`**
- 补 `peak_memory_mb=null`、`termination_reason`（如 `"evaluation_budget"`）、`failure_reason` 默认 `"none"`、`fe_counts_by_event={}`、`known_optimum`/`normalized_gap` 已有标准化、嵌入 `task`、新增 `best_so_far_trace`（来自 `_MinObserver.trace`）、`supersedes_run_id="none"`。
- 保留 `algorithm_id`（DE/GenSA）。

**runner 占位 payload（infra_failure / timeout）**
- 三个 runner 现写的占位 payload 统一为 `{run_id, status, failure_reason}`（可缺其余字段）；merge 容错构建 row（数值 → `NaN`，`status` 保留）。

### 4.5 merge（Task 11）重新定义

- 扫描 raw 目录所有 `<run_id>.json`（排除 `_tasks/`、`*.tmp*`）。
- 按 `run_id` 在合并的 manifest `task_index` 中查 task：task 含 `configuration_hash` → SMCO；含 `algorithm` → baseline。`run_id` 前缀 `r`/`b` 作双重保险；查不到 → orphan，进 audit 错误。
- **单一构建点**：`result_row_from_task(task, **outcome)` 构建 row。
- baseline：构建后 bypass `validate_result_row` 中 `build_algorithm_id` 重建检查（algorithm_id 非标准），其余 schema 检查仍跑。
- 原计划为 R 重建、为 baseline bypass 的**两条特殊路径合并为一条**：统一从 outcome + task 重建。
- supersedes 解析（`supersedes_run_id` DAG，被取代行进 `all_attempts` 不进 `valid_runs`）+ 11 项 audit（见实现计划第 11 节）+ 写 `merged/`。
- 新增 `merged/anytime.csv`：从各 raw 的 `anytime` 展开成长表，供 Task 12；`best_so_far_trace` 留 raw 供高分辨率回查。

### 4.6 outcome 契约权威位置

`paper_contract.OUTCOME_FIELDS`：字段名元组 + 一个 `validate_outcome(payload)` 函数（返回违规列表，空即合规），worker 测试与 merge 共用。保持 `paper_contract.py` stdlib-only。`validate_outcome` 仅校验 success/algorithm_failure 的完整 outcome；infra_failure/timeout 的 runner 占位 payload（`{run_id,status,failure_reason}`）由 merge 容错处理，不要求通过 `validate_outcome`。

## 5. 测试策略

- `paper_contract` 新增 `OUTCOME_FIELDS` + `validate_outcome` 单测。
- 更新 `tests/test_highdim_worker.py` / `tests/test_baseline_worker.py`：不再断言 `result_row`；改断言 outcome 字段集（含 `best_so_far_trace`、嵌入 `task`、`supersedes_run_id`）。
- 新增 `tests/test_merge_results.py`：outcome + task → row 重建（SMCO Py/R 形态、baseline 形态）、supersedes 解析、11 项 audit、`merged/` 输出（含 `anytime.csv`）、audit 不过时 merge 拒绝产出主表。
- R worker 端到端在本机重验（R 4.3.2 + jsonlite + qrng 已装）：确认 `null`/`trace`/`task` 落盘正确。
- `tests/test_r_instance_parity.py` 不受影响（实例 loader 未改）。

## 6. 影响评估

- **改动**：`src/smco/highdim_worker.py`、`src/smco/baseline_worker.py`、`scripts/run_smco_evo_highdim_r.R`、三个 runner 的占位 payload、`src/smco/paper_contract.py`（+ `OUTCOME_FIELDS`/`validate_outcome`）。
- **新增**：`src/smco/merge_results.py`、`scripts/merge_smco_evo_highdim_results.py`、`tests/test_merge_results.py`。
- **测试更新**：`test_highdim_worker.py`、`test_baseline_worker.py`、`test_paper_contract.py`。
- **文档**：`docs/gate-d-pilot-2026-07-29.md` 字段表更新；本 spec。
- **无历史 raw 迁移**（E1 未跑）。Gate D pilot 仅字段验证、未保留数据，重跑即更新。
- `is_run_complete`（confirmatory.py）仍以 `status=="success"` 为准，不受 payload 结构变化影响。

## 7. 决策记录

- **为何方案 B（merge 单点构建）而非 A（worker 自带 row）**：方案 A 要求 R 维护第二份 contract 逻辑（列序/枚举/run_id 重建），跨语言漂移风险高——正是 Gate A 花大力气对齐的同一类问题。方案 B 把 row 构建去重到 Python 一处，R/baseline 只产统一 outcome。
- **为何 raw 不含 result_row**：raw 信息量已严格 ⊇ row（含 `best_so_far_trace` + 嵌入 `task`，row 的每列都可从其派生）。row 是 merge 产出的扁平分析视图，放 `merged/`。raw 不含 row 不损失任何信息。
- **为何新增 best_so_far_trace**：改进点序列是 anytime / target_hit / ECDF 的原始材料，目前落盘即丢失。只记改进点（有界），即便 5000D 也是 KB 级；让 Task 12 能以任意分辨率重算，价值高、成本低。
- **为何保留并要求 Py 也嵌入 task**：完整 `algorithm_config` 快照是详细 provenance，使单个 raw 自含可审计的配置上下文，并与 frozen manifest 交叉校验。

## 8. 实施顺序（概要，详见 writing-plans 产出）

1. `paper_contract.OUTCOME_FIELDS` + `validate_outcome`（TDD）。
2. 改 Py worker 输出统一 outcome（含 trace/task）。
3. 改 R worker（null/trace/task/supersedes）+ 本机端到端重验。
4. 改 baseline worker（补字段 + trace/task）。
5. 更新 runner 占位 payload。
6. 更新 worker 测试。
7. `merge_results.py` + CLI（TDD）。
8. 全量 pytest + Gate D pilot 字段更新。
