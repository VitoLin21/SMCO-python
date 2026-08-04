# SMCO-EVO R True State-Preserving — 设计规格

- 日期：2026-07-28
- 状态：已批准（设计审批通过），待实现
- 分支：`feat/smco-evo-highdim-paper-2026`
- 父计划：[`docs/smco-evo-highdim-implementation-plan-2026-07-28.md`](../../smco-evo-highdim-implementation-plan-2026-07-28.md) 的 **Task 4**
- 对应提交边界（父计划 §2.3 第 3 项）：`feat: add stateful evolutionary scheduler in r`

## 1. 背景与动机

SMCO-EVO 高维论文战役采用 **2×2 设计**：语言 {Python, R} × 状态语义
{state-preserving (SP), restart (RS)}，需要四格都可运行、可对照。

- Python 侧（Task 3，commit `85a4f00`）已完成：`_run_evolutionary_states`
  （真 SP）与 `_run_evolutionary_restarts`（RS）双 runner，`state_semantics`
  参数贯穿 `smco_evo_multi` 与三个 wrapper。
- R 侧现状（`vendor/SMCO_R/main/SMCO_evo.R`）：`run_evolutionary_states`
  虽以 "state-preserving" 命名，**实际是 restart 语义**——每个演化边界都
  重新调用 `SMCO_single` 从 `x_runmax` 开始，递推累加器 `s_value` 不跨边界
  延续。状态记录只持轻量字段（`x`/`f`/`x_runmax`/`f_runmax`/`iterations`/
  `birth`），没有 `s_value`/`current_n`/`initial_n` 等真正的递推状态。

因此 R-SP 格缺失。Task 4 补齐它：新增真正的 state-preserving runner，把现有
restart 逻辑正名为 RS，并给公共入口加 `state_semantics` 选择。

## 2. 目标

1. 新增 `vendor/SMCO_R/main/SMCO_evo_stateful.R`，提供与 Python `SMCOState`
   对齐的 state 容器与 `initialize_smco_state` / `run_smco_state_until` /
   `smco_state_to_result` 三个函数；`run_smco_state_until` 直接延续递推，
   不调用 `SMCO_single`。
2. 把 `SMCO_evo.R` 中现有的 restart 调度重命名为 `run_evolutionary_restarts`，
   补齐全局 best archive、`restart_initialization` 事件标记、history 的
   `state_semantics` 字段，使其与 Python `_run_evolutionary_restarts` 对齐。
3. 新建 `run_evolutionary_states`（真正 SP）：survivor 跨边界保留 state 对象，
   replacement 以 `birth_iteration = boundary` 初始化。
4. 三个公共入口 `SMCO_EVO` / `SMCO_R_EVO` / `SMCO_BR_EVO` 增加
   `state_semantics` 参数，透传到对应 runner。
5. 新增 `vendor/SMCO_R/main/tests/test_evolution_semantics.R`，覆盖 plan Task 4
   的 R 测试清单与 Python 测试的关键场景。
6. 通过 **Gate B**。

## 3. 非目标

- 不重构 `SMCO_single`（保持字节级行为不变，Gate A 已锁）。
- 不修改 `vendor/SMCO_R/v1.0.0/`（冻结快照）。
- 不在本 Task 修改 `run_highdim_r.R`、`align/r_side.R` 等运行脚本去显式传
  `state_semantics`——那是 Task 8/9 worker 的事；本 Task 只保证遗留调用方
  依赖的默认值向后兼容。
- 不启动任何正式实验运行（父计划 §1：本文档不授权启动正式实验）。
- 不引入 `runmax_history`（R EVO 路径当前不消费 `record_history`，YAGNI）。

## 4. 关键设计决策

### 4.1 SP 递推独立实现，不碰 `SMCO_single`（方案 A）

`SMCO_evo_stateful.R` 的 `run_smco_state_until` 直接镜像 Python
`_run_smco_state_until`（`src/smco/optimizer.py:637`）的递推，数学上等价于
`SMCO_single` 的主循环（`S += Z; x_next = S/(n+1)`），但以"可从任意
`current_n` 延续到 `target_n`"的形态表达。

不采用"重构 `SMCO_single` 复用 `run_smco_state_until`"的方案 B，原因：

- `SMCO_single` 是带 `cmpfun` 的性能优化版（预计算 `h_steps_matrix`），Gate A
  已验证其事件级 FE 计数与 Python 一致；改动它有回归风险且违背"现有
  smco/smco_r/smco_br 行为不变"硬边界。
- Python 侧 `_single` 与 `_run_smco_state_until` 共享递推，是 Python 无
  `cmpfun` 历史包袱下的自然路径；R 移植不必照搬这一耦合。
- SP 递推与 `SMCO_single` 的数学重复由测试覆盖（与 Python 小型轨迹数值一致）
  保证等价，零回归风险。

### 4.2 R 公共入口默认 `state_semantics = "restart"`

与 Python 默认 `state_preserving` **故意不对称**。理由：

- 现有 R EVO 调用方（`run_highdim_r.R:169-184`、`align/r_side.R:21`、
  `test_evaluation_budget.R:91-108`）都不传 `state_semantics`，依赖默认值。
  默认 restart 使它们与已有 R-RS 结果保持一致（父计划 §6 Task 4："旧结果
  迁移时标记为 R-RS，不能改写原 CSV"）。
- 实验 runner（Task 8/9）会显式传 `state_semantics`，故默认值不影响 2×2
  实验正确性，只决定遗留脚本走哪条路径。
- 父计划 §6 Task 4 明确："现有入口第一次整合时可默认 restart；runner 必须
  显式传值，不能依赖默认；输出 metadata 必须写明语义。"

### 4.3 state 用 list，不用 environment

`smco_state` 是 per-trajectory、显式传递并返回新对象的值语义容器，用普通
named list 即可。只有 FE budget 需要 environment（跨函数共享同一个计数器，
避免 R 的 list 写时复制丢失计数），二者语义不同，不混用。

## 5. 文件结构与 source 链

```
vendor/SMCO_R/main/SMCO_evo_stateful.R   # 新增：state 容器 + 三函数 + SP runner
vendor/SMCO_R/main/SMCO_evo.R            # 修改：末尾 source stateful；重命名；入口
vendor/SMCO_R/main/SMCO.R                # 不改
vendor/SMCO_R/main/tests/test_evolution_semantics.R  # 新增
```

source 顺序（由 `SMCO_evo.R` 末尾 `source("SMCO_evo_stateful.R")` 自动接入
现有链，无需改 `run_highdim_r.R` / `align/r_side.R` / test 文件）：

```
evaluation_budget.R -> SMCO.R (末尾 cmpfun) -> SMCO_evo.R -> SMCO_evo_stateful.R
```

`SMCO_evo_stateful.R` 复用 `SMCO_evo.R` 的 `generate_evolution_points` /
`.evolution_boundaries` / `.n_eliminate` / `%||%`，故必须在其后 source。

## 6. `smco_state` 结构（镜像 Python `SMCOState`，`optimizer.py:35`）

named list，字段：

| 字段 | 类型 | 语义 |
|---|---|---|
| `x_current` | numeric(d) | 当前点 |
| `f_current` | numeric | 当前点目标值 |
| `s_value` | numeric(d) | 递推累加器（跨边界延续的核心） |
| `current_n` | integer | 下一个要执行的循环索引 n |
| `initial_n` | integer | 该 trajectory 的 n_boost_1 锚点（= iter_boost + iter_nstart） |
| `iter_boost` | integer | boost 偏移 |
| `x_runmax` | numeric(d) \| NULL | running-best 点 |
| `f_runmax` | numeric \| NULL | running-best 值 |
| `iterations` | integer | 最后完成 n 对应的 SingleResult iteration 计数 |
| `birth_iteration` | integer | 该 trajectory 创建时的全局 boundary |
| `stopped_target_n` | integer \| NULL | 收敛命中时记录的 target_n |

辅助取值函数（镜像 Python `ranking_value` / `ranking_point`）：

- `state_ranking_value(state)`：`f_runmax` 非 NULL 取 `f_runmax`，否则 `f_current`。
- `state_ranking_point(state)`：`x_runmax` 非 NULL 取 `x_runmax`，否则 `x_current`。

## 7. 三个核心函数

### 7.1 `initialize_smco_state`

镜像 Python `_initialize_smco_state`（`optimizer.py:602`）。

```r
initialize_smco_state <- function(f, start_point, iter_nstart, iter_boost,
                                  use_runmax, birth_iteration = 0L,
                                  budget = NULL, event = "initialization")
```

- `x_current <- start_point`；`f_current <- eval_fe(budget, f, x_current, event)`
- `n_boost_1 <- iter_boost + iter_nstart`
- `s_value <- start_point * n_boost_1`
- `use_runmax` 时 `x_runmax <- x_current`、`f_runmax <- f_current`，否则 NULL
- 返回 state list（`current_n = n_boost_1`、`initial_n = n_boost_1`、
  `iterations = 0`、`birth_iteration = birth_iteration`、`stopped_target_n = NULL`）

### 7.2 `run_smco_state_until`

镜像 Python `_run_smco_state_until`（`optimizer.py:637`）。**直接延续递推，
不调用 `SMCO_single`。**

```r
run_smco_state_until <- function(state, f, lo, hi, bounds_buffer, buffer_rand,
                                 iter_target, tol_conv, partial_option,
                                 use_runmax, budget = NULL)
```

行为：

- `initial_n <- state$initial_n %||% state$current_n`；`target_n <- initial_n + iter_target`
- 若 `state$stopped_target_n` 非 NULL 且 `target_n <= stopped_target_n`：直接返回（已收敛）
- `step_cost <- (2*d if partial_option=="center" else d) + 1`
- 循环 `while (state$current_n <= target_n)`：
  - 预检：`if (!is.null(budget) && !ctx_can_evaluate(budget, step_cost))` →
    `ctx_set_termination(budget, "evaluation_budget")`；break
  - `n <- state$current_n`；`h_step <- bounds_diff / (n + 1)`
  - `partial <- compute_partial_signs(...)`（复用 SMCO.R 的 cmpfun 版）
  - `buffer_rand` 时随机 pushout，否则固定 pushout
  - `Z <- signs * upper_out + (1 - signs) * lower_out`；`state$s_value <- state$s_value + Z`
  - `x_next <- state$s_value / (n + 1)`；`f_next <- eval_fe(budget, f, x_next, "iterate")`
  - `use_runmax` 时更新 `f_runmax`/`x_runmax`（取 `max(f_partial_best, f_next)`）
  - 推进 `state$f_current`/`x_current`/`current_n`/`iterations`
  - 收敛检测：`n >= iter_min_check && abs(f_current - f_prev) < tol_conv` →
    设 `stopped_target_n <- target_n`；break
- 返回更新后的 state（in-place 修改 list 字段并返回）

注：state 以 list 传递，函数内修改字段后返回同一 list 对象（调用方赋值），
等价于 Python 的可变 dataclass 语义。

### 7.3 `smco_state_to_result`

镜像 Python `SMCOState.to_result`（`optimizer.py:60`）。

```r
smco_state_to_result <- function(state)
```

返回 `list(x_optimal = x_current, f_optimal = f_current, iterations = iterations,
x_runmax = x_runmax, f_runmax = f_runmax)`（`use_runmax=FALSE` 时不带 runmax 字段）。

## 8. 双调度 runner

### 8.1 `run_evolutionary_states`（真正 SP）

镜像 Python `_run_evolutionary_states`（`optimizer.py:1091`）。函数签名沿用
现有 R 版（参数不变），但内部改为：

1. 每个 start 用 `initialize_smco_state(birth_iteration = 0L, event = "initialization")` 建 state。
2. 每个 boundary：对每个 state 调 `run_smco_state_until(state, ...,
   iter_target = boundary - state$birth_iteration, ...)`——**survivor 的
   `s_value`/`current_n` 跨边界延续**。
3. 按 `state_ranking_value` 降序排名，`.n_eliminate` 淘汰，从 survivor 经
   `generate_evolution_points` 生成 replacement。
4. replacement 用 `initialize_smco_state(birth_iteration = boundary,
   event = "replacement_initialization")` 建 state。
5. 最终每个 state 推到 `iter_max`：`run_smco_state_until(..., iter_target = iter_max - birth)`；
   `smco_state_to_result` → clip to bounds → promote runmax；iterations 按
   `birth_iteration` 归一化。
6. history 每条带 `state_semantics = "state_preserving"`。

### 8.2 `run_evolutionary_restarts`（RS，正名 + 补齐）

现有 `run_evolutionary_states` 的 restart 逻辑重命名而来，镜像 Python
`_run_evolutionary_restarts`（`optimizer.py:1235`）。保留其"每个 boundary 从
`x_runmax` 重新 `SMCO_single`"的核心，补三件事：

1. **全局 archive**：跨 boundary 保留 best-ever running-best（`archive_value`/
   `archive_point`），最终若 archive 优于所有 final state 则追加为额外 result。
2. **`restart_initialization` 事件**：survivor 续延段与最终段的重新 init 用
   `eval_fe(..., "restart_initialization")` 计数（replacement 仍用
   `replacement_initialization`）。
3. **history 字段**：每条带 `state_semantics = "restart"`。

### 8.3 调度分发

`.run_evolutionary_branch`（`SMCO_evo.R:354`）根据 `state_semantics` 选择
`run_evolutionary_states`（SP）或 `run_evolutionary_restarts`（RS），透传
`state_semantics` 到 `.run_evo_core` 与公共入口。

## 9. 公共入口

`SMCO_EVO` / `SMCO_R_EVO` / `SMCO_BR_EVO`（`SMCO_evo.R:280/462/474`）增加：

```r
state_semantics = "restart"
```

- 非法值（非 `"state_preserving"` / `"restart"`）`stop("state_semantics must be ...")`
- 透传到 `.run_evo_core` → `.run_evolutionary_branch`
- BR（`iter_boost > 0`）的 regular/boosted 两个 branch 共用同一 `state_semantics`

## 10. 测试矩阵（`tests/test_evolution_semantics.R`）

无 testthat 依赖，沿用 `test_evaluation_budget.R` 的 `check()`/`fail()` +
`stopifnot` 风格，非零 exit on failure。覆盖：

| 场景 | 来源 |
|---|---|
| staged SP run == one-shot SP run（分段 == 一次性） | plan Task4 R 测试 |
| SP survivor 跨边界保留 `s_value`/`current_n`（accumulator 延续） | plan Task4 |
| RS 跨边界重置（`s_value` 不延续，从 x_runmax 重启） | plan Task4 |
| running-best 与 current point 分开记录 | plan Task4 |
| replacement `birth_iteration == boundary` | plan Task4 |
| SP 与 RS 在 `evolution_points=(0.999,)`（无边界）时数值一致 | Python `test_sp_and_restart_agree_without_boundaries` |
| 非法 `state_semantics` 抛错（三个 family） | Python `test_invalid_state_semantics_raises` |
| 三个 family 都能跑 restart 端到端 | Python `test_all_evo_variants_support_restart` |
| RS 计 `restart_initialization` 与 `replacement_initialization` 事件 | Python `test_restart_counts_restart_initialization_event` |
| RS archive 保留全局 best（best == f_runmax） | Python `test_restart_preserves_global_best_in_archive` |
| RS 同 seed 可复现 | Python `test_restart_is_reproducible_with_same_seed` |
| BR split 预算：regular/boosted 各 ≤ budget/2+1 | Python `test_restart_br_respects_branch_budget` |
| 紧预算不抛错且 `termination_reason == "evaluation_budget"` | Python `test_restart_runs_under_tight_budget_without_raising` |
| **R-SP 与 Python-SP 小型轨迹数值一致（容差内）** | plan Task4 / Gate B 跨语言对齐 |

最后一项是跨语言烟雾测试：固定 seed、固定 starts、小型 deterministic objective，
对比 R `SMCO_EVO(..., state_semantics="state_preserving")` 与 Python
`smco_evo(..., state_semantics="state_preserving")` 的 `f_optimal`/`x_optimal`
在容差内一致。R 测试通过 `reticulate` 或子进程调 Python 太重，改为：R 测试
内嵌一组 Python 参考值（从 Python 实跑得到，写入测试常量），R 结果与之比。
参考值的生成与锁定在实现时记录。

## 11. Gate B 验收

```bash
.venv/bin/python -m pytest tests/test_evolution_semantics.py -v
Rscript vendor/SMCO_R/main/tests/test_evolution_semantics.R
.venv/bin/python -m pytest -q
```

Gate B 未通过不得生成 E1 manifest（父计划 §6）。

## 12. 风险与权衡

| 风险 | 缓解 |
|---|---|
| SP 递推与 `SMCO_single` 数学重复导致数值漂移 | 忠实镜像同一递推；跨语言数值一致测试覆盖 |
| R list 写时复制使 state 修改丢失 | 每次显式返回修改后的 state，调用方赋值（不依赖引用语义） |
| 重命名 `run_evolutionary_states` 破坏外部引用 | 探索确认无外部 R 脚本直接引用（仅 `SMCO_evo.R` 内部 `.run_evolutionary_branch`），更新内部调用即可 |
| 默认 restart 与 Python 默认 SP 不对称致混淆 | metadata 写明语义；实验 runner 显式传参；本 spec §4.2 记录理由 |
| 跨语言数值一致测试的 Python 参考值锁不定 | 实现时用固定 seed 实跑生成常量并记录 Python 版本 |

## 13. 与父计划的关系 / 后续衔接

- 本 spec 细化父计划 Task 4（§6）的实现路径，不改变其验收标准。
- 实现完成后更新战役进度内存 `smco-evo-highdim-paper-campaign.md`。
- 后续：Gate B → Task 5（跨语言 trace 合同）→ Task 6（高维实例）。
- 本 Task 仅交付算法 + 入口 + 测试；runner 接 `state_semantics` 在 Task 8/9。
