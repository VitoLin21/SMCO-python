# SMCO-EVO 高维论文实验实现计划

日期：2026-07-28  
状态：Task 0--4 已完成；下一主任务为高维实例、manifest 和 pilot
依赖实验设计：
[`docs/smco-evo-highdim-paper-experiment-plan-2026-07-28.md`](smco-evo-highdim-paper-experiment-plan-2026-07-28.md)

## 1. 目标和范围

本计划把实验方案拆成可逐项验收的工程任务。当前核心实现已经完成，
后续执行者需要集中完成实验基础设施和高维验证：

1. 高维 shift/asymmetry/permutation/block-rotation 实例；
2. 开发集选出一个 canonical SMCO-EVO 实现并冻结；
3. 高维 winner vs matched base 和强基线实验；
4. 关键机制消融；
5. 统一统计、图表和结果打包。

## 1.1 当前进度

| Task | 状态 | 提交/验证 |
| --- | --- | --- |
| Task 0：结果合同 | 已完成 | `04a155c` |
| Task 1：Python FE budget | 已完成 | `8d39122`；16 tests passed |
| Task 2：R FE budget | 已完成 | `8c46776`；R budget tests passed |
| Task 3：Python restart | 已完成 | `85a4f00`；13 semantics tests passed |
| Task 4：R state-preserving | 已完成 | `19e6393`；R semantics tests passed |
| Task 5：跨语言逐轨迹对齐 | 可选/P3 | 只在怀疑实现差异时执行 |
| Task 6--13 | 待完成 | 当前主线 |

Task 5 不再阻塞 Task 6、manifest、pilot 或正式高维实验。

本计划不要求本轮执行任何代码修改或实验。后续执行智能体应先阅读仓库根目录
`AGENTS.md`、实验方案、现有 EVO 设计文档和当前 `git status`。

## 2. 不可突破的边界

### 2.1 兼容性

- 现有 `smco`、`smco_r`、`smco_br` 行为必须保持不变；
- 现有 Python EVO API 默认保持 state-preserving；
- 新增 FE budget 前，未传 `max_evals` 的调用必须保持当前 `iter_max` 行为；
- `vendor/SMCO_R/v1.0.0/` 是上游参考快照，不修改；
- 不覆盖 `result/rerun-2026-07-20/`、`result/highdim-full-rerun-2026-07-20/`
  和 `result/r-highdim-rerun-2026-07-24/`；
- 当前工作树中已有修改属于用户，执行者必须先确认差异，不能回滚或顺手格式化。

### 2.2 实验边界

- 现有结果只进入 exploratory/development provenance；
- E1 选型完成后，确认性 manifest 必须冻结；
- 不允许按函数、维度或 seed 事后选择不同语言版本；
- 正式质量比较使用 FE budget，不使用表面 `iter_max`；
- 低维不是主实验，不应先大规模重跑低维套件。

### 2.3 推荐的提交边界

执行者应按任务分提交，避免把算法、runner 和大结果混成一个提交：

1. `feat: add exact objective evaluation budgets`
2. `feat: add restart semantics for python smco evo`
3. `feat: add stateful evolutionary scheduler in r`
4. `feat: add reproducible high-dimensional instances`
5. `feat: add frozen-manifest experiment runners`
6. `feat: add high-dimensional paper analysis`
7. `docs+results: package smco evo paper campaign`

若执行环境不应提交代码，可以保持同样的变更分组进行交付。

## 3. 最终文件结构

建议新增或修改以下文件。

### 3.1 Python 核心

```text
src/smco/evaluation.py                  # FE counter、budget、best-so-far trace
src/smco/optimizer.py                   # SP/RS 语义和 budget-aware SMCO
src/smco/results.py                     # 仅在必须增加稳定结果字段时修改
src/smco/highdim_instances.py           # 可复现高维实例
src/smco/__init__.py                    # 只导出必要公共 API
```

### 3.2 R 核心

```text
vendor/SMCO_R/main/SMCO.R
vendor/SMCO_R/main/SMCO_evo.R           # 明确为 restart 语义并兼容旧入口
vendor/SMCO_R/main/SMCO_evo_stateful.R  # true state-preserving runner
vendor/SMCO_R/main/evaluation_budget.R
```

### 3.3 实验与分析

```text
scripts/generate_smco_evo_manifests.py
scripts/run_smco_evo_contracts.py
scripts/run_smco_evo_highdim_factorial.py
scripts/run_smco_evo_highdim_baselines.py
scripts/run_smco_evo_bbob_largescale.py
scripts/run_smco_evo_lowdim_check.py
scripts/run_smco_evo_ablations.py
scripts/run_smco_evo_highdim_r.R
scripts/merge_smco_evo_highdim_results.py
scripts/analyze_smco_evo_highdim_paper.py
scripts/package_smco_evo_highdim_paper.py
```

### 3.4 测试

```text
tests/test_evaluation_budget.py
tests/test_evolution_semantics.py
tests/test_highdim_instances.py
tests/test_experiment_manifests.py
tests/test_experiment_result_schema.py
tests/test_cross_language_traces.py
vendor/SMCO_R/main/tests/test_evaluation_budget.R
vendor/SMCO_R/main/tests/test_evolution_semantics.R
```

R 侧目前没有完整 `testthat` 工程。第一版可以使用无额外依赖的 `Rscript` 断言文件，
失败时返回非零 exit code；不要为了测试引入重量级 R package 结构。

## 4. 统一术语和接口

## Task 0：冻结命名和结果合同（已完成）

### 文件

- 新增：`docs/smco-evo-result-contract.md`
- 修改：无算法文件

### 需要定义

```text
language = python | r
state_semantics = state_preserving | restart
family = smco | smco_refine | smco_boost_refine
evolutionary = true | false
strategy = rand1bin | current-to-best1bin | best1bin | sobol
```

算法 ID 由字段稳定拼接，例如：

```text
PY-SP-SMCO-EVO
R-RS-SMCO-REFINE-EVO
PY-BASE-SMCO-BOOST-REFINE
```

不得继续用 `SMCO_R` 同时表达 R 语言和 refine family。

### 结果行必填字段

使用实验计划第 7 节的字段，并补充：

```text
schema_version
manifest_id
stage
algorithm_id
objective_sense
checkpoint_fe
is_confirmatory
supersedes_run_id
```

### 验收

- 同一 run key 在 Python、R 和 merge 脚本中生成同一个 `run_id`；
- 字段枚举值在文档、Python 常量、R 常量和 CSV header 中一致；
- schema 增量通过 `schema_version` 管理，不静默改列含义。

## 5. 精确函数评价预算

## Task 1：Python objective counter 和 hard budget（已完成）

### 文件

- 新增：`src/smco/evaluation.py`
- 修改：`src/smco/optimizer.py`
- 修改：`src/smco/results.py`（仅当 summary 不足以承载字段时）
- 新增：`tests/test_evaluation_budget.py`
- 修改：`tests/test_optimizer.py`

### 建议接口

```python
class EvaluationBudgetExceeded(RuntimeError):
    pass


class EvaluationContext:
    max_evals: int | None
    evaluations: int
    best_value: float | None
    best_point: np.ndarray | None
    trace: list[EvaluationRecord]

    def can_evaluate(self, count: int = 1) -> bool: ...
    def require(self, count: int = 1) -> None: ...
    def evaluate(self, x: np.ndarray, *, event: str) -> float: ...
```

不要只用外层 closure 粗略计数，因为需要：

- 在一个有限差分迭代开始前预检剩余预算；
- 标记 `initialization`、`finite_difference`、`iterate`、
  `replacement_initialization`、`refine`、`clip_recheck`；
- 保存 target-hit FE 和 best-so-far；
- 防止预算耗尽时产生半个坐标更新。

### 原子步骤成本

执行者需要从真实代码计算，不能硬编码所有路径都为 `2d + 1`：

- `partial_option="center"`：有限差分通常为 `2d`；
- `partial_option="forward"`：通常为 `d`；
- 当前点初始化：1；
- 新 `x_next`：1；
- replacement 初始化：1；
- 越界裁剪后的重新评价：按实际发生计数；
- refine 和 boosted branches 使用同一 context 或显式子预算 context。

每次主循环开始前调用 `require(step_cost)`。如果预算不足：

- 不启动该循环；
- 保存当前 best-so-far；
- `termination_reason="evaluation_budget"`；
- `fe_used <= max_evals`。

### API 兼容

给 SMCO 公共入口增加可选参数：

```text
max_evals: int | None = None
record_evaluations: bool = False
```

默认 `None` 时维持当前迭代行为。不要移除 `iter_max`；实验 runner 在有 FE budget 时
以 `max_evals` 为硬停止，`iter_max` 只作为足够大的安全上限。

### summary 字段

```text
fe_budget
fe_used
termination_reason
best_so_far_trace
target_hit_evaluations
evaluation_counts_by_event
```

大规模正式运行时，`best_so_far_trace` 只保存变化点或预定义 checkpoint，
不要保存每一次无改进评价。

### 必测场景

1. 初始化预算不足；
2. 恰好完成一个 center-difference iteration；
3. 剩余预算不足以完成下一轮；
4. replacement 初始化计数；
5. refine 分段计数；
6. BR regular/boosted 两分支总和不超过全局预算；
7. clip re-evaluation 计数；
8. `max_evals=None` 与修改前结果一致；
9. 同 seed、同预算完全复现；
10. trace 中最后一个 FE 不超过 budget。

### 验证命令

```bash
.venv/bin/python -m pytest tests/test_evaluation_budget.py -v
.venv/bin/python -m pytest tests/test_optimizer.py -v
```

## Task 2：R objective counter 和 hard budget（已完成）

### 文件

- 新增：`vendor/SMCO_R/main/evaluation_budget.R`
- 修改：`vendor/SMCO_R/main/SMCO.R`
- 修改：`vendor/SMCO_R/main/SMCO_evo.R`
- 新增：`vendor/SMCO_R/main/tests/test_evaluation_budget.R`

### 实现要求

- R 与 Python 使用相同字段和事件名；
- 用 environment 保存可变 counter，避免 list copy 导致计数丢失；
- `evaluate_with_budget(ctx, f, x, event)` 是唯一正式评价入口；
- 在 `compute_partial_signs` 前做原子成本预检；
- `SMCO_single`、refine、boost 和 EVO replacement 全部接收同一 budget context；
- 不允许每个 branch 各创建一份完整 budget。

### 验证命令

```bash
Rscript vendor/SMCO_R/main/tests/test_evaluation_budget.R
```

### Gate A 验收

只有以下条件全部满足才能实现新语义：

- Python/R exact count 测试通过；
- 相同小型调用的事件级评价数量一致；
- 无 budget 调用保持兼容；
- 全部 Python 测试仍通过。

## 6. 两种 EVO 状态语义

## Task 3：Python restart 语义（已完成）

### 文件

- 修改：`src/smco/optimizer.py`
- 新增：`tests/test_evolution_semantics.py`
- 修改：`src/smco/__init__.py`（只有新增公共 wrapper 时）

### 推荐 API

优先在现有三个 EVO wrapper 中增加：

```text
state_semantics="state_preserving"
```

允许值：

```text
state_preserving
restart
```

保持默认值不变。内部拆成两个明确 runner：

```python
_run_evolutionary_states(...)
_run_evolutionary_restarts(...)
```

不要在同一循环中堆积大量条件分支，使两种语义无法审计。

### Restart 合同

在每个演化边界：

1. 所有 active trajectories 运行到边界；
2. 按 running-best 排序；
3. survivor 的 restart point 取 `x_runmax`；
4. survivor 和 replacement 都建立新状态；
5. 全局 archive 保留边界前 best；
6. 重启状态使用边界对应的 global iteration anchor；
7. 重启初始化评价计入 FE；
8. final result 是全局 archive 与最终状态的最好者。

### 必测场景

- SP survivor 的 `s_value/current_n` 对象连续；
- RS survivor 的状态重新初始化；
- SP 与 RS 在 `evolution_points=()` 或无边界时一致；
- 两种语义共享相同淘汰集合和 replacement 输入；
- RS 不丢失边界前 global best；
- 相同 seed 可复现；
- 非法 `state_semantics` 抛出清晰 `ValueError`；
- 三个 family 均支持两种语义；
- BR 总预算在两种语义下都正确。

## Task 4：R true state-preserving 语义（已完成）

### 文件

- 新增：`vendor/SMCO_R/main/SMCO_evo_stateful.R`
- 修改：`vendor/SMCO_R/main/SMCO_evo.R`
- 可能修改：`vendor/SMCO_R/main/SMCO.R`
- 新增：`vendor/SMCO_R/main/tests/test_evolution_semantics.R`

### 关键要求

当前 R EVO 通过重复调用 `SMCO_single` 从 `x_runmax` 开始，属于 restart。
执行者应先修正文档注释，不能继续声称它保留完整 state。

R-SP 需要新增与 Python `SMCOState` 对齐的 state：

```text
x_current
f_current
s_value
current_n
initial_n
iter_boost
x_runmax
f_runmax
iterations
birth_iteration
stopped_target_n
```

建议新增：

```r
initialize_smco_state(...)
run_smco_state_until(...)
smco_state_to_result(...)
```

`run_smco_state_until` 必须直接延续递推，不调用 `SMCO_single` 重新开始。

### 公共入口

R EVO wrapper 同样接受：

```text
state_semantics = "state_preserving" | "restart"
```

为了兼容已有 R 结果：

- 现有入口第一次整合时可默认 `restart`；
- runner 必须显式传值，不能依赖默认；
- 输出 metadata 必须写明语义；
- 旧结果迁移时标记为 `R-RS`，不能改写原 CSV。

### R 测试

- staged run 与 one-shot state run 一致；
- survivor 跨边界保留 accumulator；
- restart 确实重置；
- running-best 与 current point 分开；
- replacement 的 birth iteration 正确；
- refine 和 boost 预算正确；
- 与 Python 小型轨迹结果在容差内一致。

### Gate B 验收

```bash
.venv/bin/python -m pytest tests/test_evolution_semantics.py -v
Rscript vendor/SMCO_R/main/tests/test_evolution_semantics.R
.venv/bin/python -m pytest -q
```

Gate B 未通过不得生成 E1 manifest。

## 7. 跨语言轨迹合同

## Task 5：Portable random tape 与 trace comparator（可选诊断，P3）

本任务不属于论文最小可投稿版本，也不阻塞 Task 6--13。只有在以下情况之一发生时执行：

- Python/R 开发结果差距大到无法用随机波动解释；
- 怀疑某一实现违反 SP/RS 合同；
- 审稿人明确要求跨语言逐轨迹保真证据；
- 需要定位第一次实现分叉事件。

如果开发集已经选出表现稳定的 canonical implementation，后续高维实验可以直接使用该实现，
无需先完成 portable random tape。

### 文件

- 新增：`scripts/run_smco_evo_contracts.py`
- 新增：`vendor/SMCO_R/align/evo_trace_r.R`
- 新增：`tests/test_cross_language_traces.py`
- 扩展：`vendor/SMCO_R/align/`

### Random tape 内容

使用标准 CSV/JSON/GZip 格式，避免要求 R 读取 NumPy 私有格式：

```text
buffer_uniforms
parent_base_indices
parent_difference_indices
crossover_uniforms
forced_coordinates
sobol_replacement_points
```

随机决策按稳定 event key 索引：

```text
run_id / boundary / trajectory_id / replacement_id / coordinate_block
```

不能只保存一个顺序数组，因为 SP/RS 分支可能消耗不同数量的随机数。

### Trace 字段

```text
boundary
trajectory_id
event
x_current_hash
s_value_hash
current_n
ranking_value
survivor
parent_indices
replacement_point
fe_used
global_best
```

小维合同测试可以额外保存完整向量；正式高维运行只保存 hash 和摘要。

### 比较层级

1. 完全相同：边界、淘汰 ID、父代 ID、FE；
2. 数值容差：replacement point、best value；
3. 轨迹容差：小型 deterministic objective 的 state；
4. 若语言库浮点顺序造成分叉，记录第一个 divergence event。

### 验收

- Python-SP 与 R-SP 小型轨迹对齐；
- Python-RS 与 R-RS 小型轨迹对齐；
- 同语言 SP/RS 在预期边界前一致、边界后分叉；
- comparator 对故意篡改的 trace 能定位首个错误。

## 8. 高维实例生成

## Task 6：Shift/asymmetry/permutation/block rotation

### 文件

- 新增：`src/smco/highdim_instances.py`
- 新增：`tests/test_highdim_instances.py`
- 新增：`scripts/generate_smco_evo_manifests.py`

### 实例对象

建议：

```python
@dataclass(frozen=True)
class HighDimInstance:
    function_name: str
    dimension: int
    instance_id: int
    bounds_lower: np.ndarray
    bounds_upper: np.ndarray
    known_optimum_value: float
    known_optimum_x: np.ndarray
    transform_spec: TransformSpec
```

### TransformSpec

至少包含：

```text
shift
asymmetry_direction
asymmetry_strength
permutation
block_size
block_rotation_seeds or block_matrices
objective_scale
```

开发集和确认集使用不重叠的 instance ID/seed namespace。

### 计算要求

- `d=200` 支持完整正交旋转；
- `d>=500` 默认 block rotation；
- block size 固定并写入 manifest，例如 32 或 40；
- permutation 在 rotation 前后的顺序固定并测试；
- 不在每次 objective evaluation 时重新生成 rotation matrix；
- 不构造 `5000 x 5000` dense matrix；
- Python/R 必须消费同一变换 artifact。

### Artifact 格式

推荐：

```text
instances/<instance_id>/metadata.json
instances/<instance_id>/shift.csv.gz
instances/<instance_id>/permutation.csv.gz
instances/<instance_id>/rotation_blocks.csv.gz
instances/<instance_id>/starts.csv.gz
```

每个文件记录 SHA-256。manifest 保存相对路径和 hash。

### 必测场景

- 已知 optimum 在 bounds 内；
- optimum 经变换后达到已知最优值；
- 同 seed 生成相同 hash；
- 不同 instance ID 产生不同变换；
- block rotation 近似正交；
- inverse transform round-trip；
- Python/R 在固定点上的函数值一致；
- `d=5000` 实例构造不分配 dense `d x d` matrix。

## 9. Manifest 和运行状态

## Task 7：不可变 manifest

### 文件

- 新增：`scripts/generate_smco_evo_manifests.py`
- 新增：`tests/test_experiment_manifests.py`
- 新增：`tests/test_experiment_result_schema.py`

### Manifest 分层

```text
development.yaml/json
baseline_selection.yaml/json
confirmatory_factorial.yaml/json
confirmatory_baselines.yaml/json
bbob_largescale.yaml/json
lowdim_check.yaml/json
ablations.yaml/json
```

如果不新增 YAML 依赖，使用稳定排序的 JSON。不要手写多个 CSV 造成字段漂移。

### 每个 task 的稳定 key

```text
stage
suite
function
dimension
instance
replication
algorithm_id
seed
budget
configuration_hash
```

`run_id = sha256(canonical_json(task))` 的固定前缀。

### 冻结流程

1. 生成 development manifest；
2. 完成 E1；E1B 仅在确有多个可比 comparison implementations 时执行；
3. 运行 selection 脚本；
4. 生成 `selection.json`；
5. 用 selection 生成 confirmatory manifests；
6. 写入 git commit、environment hash、instance hashes；
7. 创建 `FROZEN` 标记和 manifest SHA-256；
8. runner 遇到 frozen manifest hash 不匹配时拒绝启动。

### Resume 语义

- 每个 run 独立写 `raw/<run_id>.json` 或 append-only shard；
- 完成状态以结果 artifact 和 schema validation 为准，不只看文件存在；
- `status=success|algorithm_failure|infra_failure|timeout`；
- infra retry 写新 attempt，不覆盖旧 attempt；
- merge 时按明确的 supersedes 规则选取有效 attempt。

## 10. 分阶段 runner

## Task 8：Python/R 单任务 worker

### 文件

- 新增：`scripts/run_smco_evo_highdim_factorial.py`
- 新增：`scripts/run_smco_evo_highdim_r.R`

### Python worker

输入一个 canonical task JSON，输出一个 result JSON。职责仅包括：

- 校验 manifest/task hash；
- 加载实例和共同 starts；
- 调用指定算法；
- 收集 FE、quality、time、memory、status；
- 原子写临时文件后 rename 为最终结果；
- 写独立 log。

### R worker

接口与 Python worker 相同：

```bash
Rscript scripts/run_smco_evo_highdim_r.R --task path/to/task.json
```

若 R 读取 JSON 需要 `jsonlite`，应在环境文档中固定版本。不要让 R runner
从用户 home 的绝对路径 source 代码；所有路径从 repo root 或 task manifest 解析。

### 进程隔离

- 每个 run 单进程；
- BLAS/OpenMP 线程固定为 1；
- worker 数由 orchestrator 控制；
- wall-time cap 在外层进程管理器执行；
- 记录 hostname、CPU model、R/Python version 和 package versions。

## Task 9：E0/E1/E1B runner

### 文件

- 新增：`scripts/run_smco_evo_contracts.py`
- 新增：`scripts/run_smco_evo_highdim_factorial.py`
- 新增：`scripts/run_smco_evo_highdim_baselines.py`

### 执行顺序

1. E0 contract；
2. E1 100 FE/d pilot；
3. 冻结 wall-time cap；
4. E1 全 1080 runs；
5. 如确有必要，执行可选 E1B（最多 600 runs）；
6. selection；
7. 确认性 manifest freeze。

### E1 selection 脚本

单独提供 dry-run：

```bash
.venv/bin/python scripts/analyze_smco_evo_highdim_paper.py \
  --stage e1-development \
  --selection-only \
  --dry-run
```

输出：

```text
selection_candidates.csv
selection_score_components.csv
selection.json
selection_report.md
```

selection tie-break 必须逐条写入报告，不能只输出 winner 名称。

## Task 10：E2/E3/E4/E5/E6 runner

### 文件

- 新增：`scripts/run_smco_evo_highdim_factorial.py`
- 新增：`scripts/run_smco_evo_highdim_baselines.py`
- 新增：`scripts/run_smco_evo_bbob_largescale.py`
- 新增：`scripts/run_smco_evo_lowdim_check.py`
- 新增：`scripts/run_smco_evo_ablations.py`

### 强制检查

每个确认性 runner 启动时：

1. 验证 `FROZEN`；
2. 验证 manifest hash；
3. 验证 selection hash；
4. 验证实例文件 hash；
5. 验证 git commit 和环境信息已记录；
6. 打印任务数、预计 FE、已完成数和缺失数；
7. 只执行 manifest 中的 task。

E4 的 manifest 固定为 7 个配置：冻结 winner、matched base，以及
`DE`、`GA`、`PSO`、`SA`、`GenSA`。runner 不提供可改变 canonical baseline 集合的
运行时参数；其 `summary` 必须从 instance-level raw results 显式聚合每个
`(function, dimension, algorithm)` 的 5 个官方 instances，不能覆盖前序 instance。

R winner 不实现 R COCO runner。E4/E5 若接收 R winner，只能执行对应 Python port，
并在 selection、manifest/provenance、CSV/图表和最终报告中写入
`python_port_external_check=true`、original winner/language 与实际 Python id；该
结果不得作为 R winner 的直接外部验证或并入其主结论。Python winner 的 E4/E5 才可
标作冻结实现的外部验证。

禁止 runner 根据已有结果动态：

- 删除表现差的算法；
- 增加 seed；
- 修改预算；
- 修改 strategy；
- 替换 winner。

### 推荐 CLI 合同

```text
--manifest PATH
--result-dir PATH
--workers N
--only-language
--only-dims
--only-run-ids
--resume
--dry-run
--validate-only
```

过滤选项只用于分布式分片，不能改变 canonical manifest。

## 11. 结果合并与统计

## Task 11：Merge 和 provenance audit

### 文件

- 新增：`scripts/merge_smco_evo_highdim_results.py`
- 新增测试：`tests/test_experiment_result_schema.py`

### Merge 检查

- run_id 唯一；
- manifest task 覆盖率；
- 多 attempt 解析；
- task/result configuration hash 一致；
- FE 不超过 budget；
- maximize/minimize 方向正确；
- known optimum 与 gap 一致；
- Python/R starts hash 一致；
- 非 EVO 行不被 strategy 复制成伪重复；
- confirmatory 结果没有 development seed；
- NaN、timeout、failure 保留在分母。

### 输出

```text
merged/all_attempts.csv
merged/valid_runs.csv
merged/missing_runs.csv
merged/duplicate_runs.csv
merged/provenance_audit.json
merged/provenance_audit.md
```

audit 未通过时，正式分析脚本拒绝生成主文表图。

## Task 12：统计分析

### 文件

- 新增：`scripts/analyze_smco_evo_highdim_paper.py`
- 可能修改：`pyproject.toml` 增加独立 `paper` optional dependencies

### 依赖建议

最小：

```text
numpy
scipy
matplotlib
```

可选：

```text
cocoex/cocopp
statsmodels
```

若增加依赖，固定版本并写环境 lock。分层 bootstrap 可直接使用 NumPy 实现，
避免分析关键路径过度依赖额外统计包。

### 必须实现的分析

- normalized gap；
- target-hit FE；
- ERT；
- ECDF 和 ECDF-AUC；
- winner vs matched base；
- paired gain 对 `log(d)` 的维度趋势；
- 非轴对齐实例上的 winner vs matched base；
- `n_starts=8` 对 `ceil(sqrt(d))` 的质量和 FE 效率；
- hierarchical bootstrap 95% CI；
- Holm 校正；
- probability of superiority；
- failure/timeout sensitivity；
- wall-time 与 FE/s 次要分析。

### Bootstrap 规则

- 固定并报告 bootstrap seed；
- 先按 function 重采样；
- 再在 function 内按 instance 重采样；
- 不把 checkpoint 当独立样本；
- 默认至少 10,000 bootstrap replicates；
- 保存 bootstrap distribution 摘要和配置。

### 结果表

```text
analysis/selection_summary.csv
analysis/primary_hypotheses.csv
analysis/factorial_effects.csv
analysis/dimension_trend.csv
analysis/baseline_comparison.csv
analysis/bbob_largescale_summary.csv
analysis/lowdim_degradation.csv
analysis/ablation_summary.csv
analysis/failures.csv
analysis/walltime.csv
```

### 图

```text
figures/e2_factorial_ecdf.*
figures/e3_baseline_ecdf.*
figures/evo_base_ratio_by_dimension.*
figures/anytime_d1000.*
figures/anytime_d3000.*
figures/anytime_d5000.*
figures/bbob_largescale_ecdf.*
figures/start_count_ablation.*
figures/state_component_ablation.*
```

同时输出 PNG 和矢量 PDF/SVG，主文图不得只依赖位图。

## 12. 打包和最终报告

## Task 13：结果打包

### 文件

- 新增：`scripts/package_smco_evo_highdim_paper.py`
- 生成：`result/smco-evo-paper-highdim-2026/report.md`
- 生成：`result/smco-evo-paper-highdim-2026/README.md`

### 打包规则

- raw results 不在打包阶段重算；
- manifests、selection、environment、provenance audit 一并复制；
- 报告中的每个数字都能追溯到 analysis CSV；
- 主文 selected implementation 与 supplement 全矩阵同时存在；
- 标记 exploratory、development、confirmatory；
- 不把旧 7 月结果并入确认性统计；
- 旧结果可以放在 `exploratory_context/` 并附 provenance。

### 最终报告必须回答

1. E1 选中了什么，为什么？
2. E2 上 winner 是否优于 matched base？
3. 最终使用的语言、代码版本和 state semantics 是什么？
4. EVO 收益是否随维度增强？
5. E3 是否优于强基线？
6. E4 外部基准是否复现主要结论？
7. 低维是否明显退化？
8. 起点、strategy、schedule 和 state 组成分别贡献什么？
9. 哪些 run 失败或超时？
10. 哪些结论是确认性的，哪些仍是探索性的？

## 13. 全流程验证命令

执行者应从窄测试逐步扩大。

### Python

```bash
.venv/bin/python -m pytest tests/test_evaluation_budget.py -v
.venv/bin/python -m pytest tests/test_evolution_semantics.py -v
.venv/bin/python -m pytest tests/test_highdim_instances.py -v
.venv/bin/python -m pytest tests/test_experiment_manifests.py -v
.venv/bin/python -m pytest tests/test_experiment_result_schema.py -v
.venv/bin/python -m pytest -q
```

Task 5 实际启动时再运行：

```bash
.venv/bin/python -m pytest tests/test_cross_language_traces.py -v
```

### R

```bash
Rscript vendor/SMCO_R/main/tests/test_evaluation_budget.R
Rscript vendor/SMCO_R/main/tests/test_evolution_semantics.R
```

Task 5 实际启动时再运行：

```bash
Rscript vendor/SMCO_R/align/evo_trace_r.R --smoke
```

### Dry-run

```bash
.venv/bin/python scripts/generate_smco_evo_manifests.py --stage e1 --dry-run
.venv/bin/python scripts/run_smco_evo_highdim_factorial.py \
  --manifest result/smco-evo-paper-highdim-2026/manifests/development.json \
  --dry-run
.venv/bin/python scripts/merge_smco_evo_highdim_results.py \
  --result-dir result/smco-evo-paper-highdim-2026 \
  --validate-only
```

## 14. 验收闸门

### Gate A：预算

- Python/R 精确 FE 计数；
- branch 总预算正确；
- 无 budget 行为兼容。

### Gate B：语义

- Python-SP、Python-RS、R-SP、R-RS 均通过合同测试；
- 当前 R restart 结果被正确标记；
- Task 5 portable trace 不属于 Gate B 必需条件。

Gate A/B 已于 2026-07-28 通过。

### Gate C：实例与 manifest

- 已知 optimum 验证；
- 无 dense 5000D rotation；
- development/confirmatory seed 不重叠；
- manifest hash 可冻结。

### Gate D：Pilot

- 方向、FE、schema、日志、resume、timeout 均正常；
- wall-time cap 和服务器分片规则冻结；
- pilot 不用于换超参数。

### Gate E：开发选型

- E1 覆盖完整；若启用 E1B，其覆盖和选择规则完整；
- selection tie-break 可复算；
- `selection.json` 和确认性 manifests 冻结。

### Gate F：确认性结果

- E2/E3 coverage audit 通过；
- E4/E5 按最小版或增强版范围完成；
- 缺失、失败和 timeout 全部披露。

### Gate G：论文产物

- 分析可从 raw 重建；
- 表图与 CSV 一致；
- selected winner 的选型记录、真实语言和语义完整披露；
- 文档清楚区分探索、开发和确认性证据。

## 15. 推荐执行顺序

严格按以下顺序交给执行智能体：

```text
Task 0--4 已完成
  -> Gate A/B 已通过
  -> Task 6    高维实例
  -> Task 7    manifest
  -> Gate C
  -> Task 8/9  worker、pilot、E1（E1B 可选）
  -> Gate D/E
  -> Task 10   E2--E6
  -> Task 11   merge/provenance
  -> Task 12   statistics/figures
  -> Task 13   package/report
  -> Gate F/G
  -> Task 5    仅在需要时追加跨语言诊断
```

不要让多个执行智能体同时修改 `src/smco/optimizer.py` 或
`vendor/SMCO_R/main/SMCO_evo.R`。可并行的部分应限制为：

- Python budget 与 R budget；
- Python runner 与 R runner；
- 实例生成与结果 schema；
- 在 core API 冻结后的分析脚本和 BBOB integration。

## 16. 交接给其他智能体时的最短说明

执行智能体至少需要收到：

1. 仓库根目录和当前 branch；
2. `AGENTS.md`；
3. 本实现计划；
4. 高维实验方案；
5. 当前 `git status`；
6. 禁止覆盖的结果目录；
7. 四种候选实现均已完成；E1 只选一个 canonical implementation，
   E2 之后不再研究语言差异；
8. “只在 E1 全局选一次赢家，E2 之后不能改”的规则。

若执行智能体提出改变预算、函数、seed、选择规则或确认性范围，应先修改计划并由用户确认，
不能在实验运行过程中自行扩大或缩小范围。
