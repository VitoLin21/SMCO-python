# SMCO-EVO 高维论文证据链复审与修复方案（2026-08-02）

## 1. 文档目的

本文档用于指导后续智能体修复当前 SMCO-EVO 高维论文实验链条。实现必须遵循：

- TDD：每项先写失败测试，再实现，再运行定向与全量测试；
- 每项独立提交，不把多个审查问题混进同一 commit；
- 不修改已冻结的 E2 task 字段、stage、run_id 或数值结果；
- 不通过改实验协议消除 SA/GenSA 优于 SMCO-EVO 的科学结果；
- 在本文 P1 阻断全部关闭前，不生成 Task 12 正式图表、Holm 结果或 Task 13 论文包；
- 保留旧结果目录，只新增明确命名的正式产物，禁止静默覆盖不可恢复的数据。

当前代码 HEAD：`cb1b8eb57ee2ccd926ce7446e0e6f0d1f6b8acd7`。

## 2. 当前复审结论

当前不能宣称“E1--E6 全部通过 12 项 audit”或“P1c composite 已完全闭环”。

已确认通过：

| 证据 | 当前正式候选路径 | 状态 |
| --- | --- | --- |
| E1 | `result/e1-2026-07-30/merged_v2/` | 1,080 行；12 项 audit 通过 |
| E1 selection | `result/e1-2026-07-30/selection_v2/selection.json` | winner/hash 与原选择完全一致 |
| E2 | `result/e2-2026-07-31/merged_v2/` | 120 行；12 项 audit 通过 |
| E3 baseline component | `result/e3-2026-07-31/merged_baseline_v2/` | 300 行；12 项 audit 通过 |
| E3 合并数据 | `result/e3-2026-07-31/merged_composite/` | 420 行；12 项普通 merge audit 通过 |
| E6 strategy | `result/e6-2026-07-31/strategy/merged_v2/` | 420 行；12 项 audit 通过 |
| Python 测试 | `.venv/bin/python -m pytest -q` | 463 passed |

仍被阻断：

1. P1c composite validator 可以接受被篡改的 composite/manifest 元数据；
2. baseline component 的 Gate-F 可被错误 selection 或不完整任务矩阵绕过；
3. composite 没有接入 manifest CLI 和 Task 12 analysis gate；
4. E6 start-count 仍是旧 11-check audit，180 个 raw 全部缺 git provenance；
5. 旧 `merged/` 与新 `merged_v2/` 并存，没有唯一 canonical artifact index；
6. E6 schedule 仍有 216 task/189 unique run-id，尚未去重；
7. E4/E5 虽分别已有 2,520/480 行 development CSV，但不属于统一 outcome/merge/audit 链。

## 3. 已复现的负例

### 3.1 Composite validator 接受篡改

对 `result/e3-2026-07-31/e3_comparative_composite.json` 做内存副本修改后，当前
`validate_composite()` 对以下情况均错误返回空错误列表：

- `composite_sha256` 改成 64 个 `0`；
- `frozen=False`；
- E2 component 的 `manifest_sha256` 改成 64 个 `0`；
- E2 component 的 `n_runs` 改成 `1`；
- E2 component 的 `stage` 改成任意字符串。

根因位于 `src/smco/confirmatory.py`：

- `validate_composite()` 没有重算 composite 自身 hash；
- 没有读取、验证两个源 manifest；
- 没有要求精确的 120/300/420 行合同；
- 没有比较 merged run-id 集合与 manifest task run-id 集合；
- 没有验证 merged 行的 stage/algorithm/selection 归属。

### 3.2 Baseline component Gate-F 可绕过

当前 `_is_baseline_extension()` 只检查 component 元数据，不检查任务矩阵。已经复现：

- 给 component 传入错误的 `selection_hash`，`confirmatory_errors()` 返回 `[]`；
- 将 component 缩减为 1 个 task、重算 manifest hash 后，`confirmatory_errors()` 仍返回 `[]`。

因此 `component_role="baseline_extension"` 尚未被完整任务合同保护。

### 3.3 E6 start-count 当前 audit 失败

使用当前 merge 代码对以下输入重新审计：

```text
manifest: result/e6-2026-07-31/start_count/e6_ablations__synthetic_highdim.json
raw:      result/e6-2026-07-31/start_count/raw_dedup/
```

结果为：

```text
n_rows = 180
前 11 项检查通过
provenance_complete = FAIL（180/180）
overall audit = FAIL
```

现有 `start_count/merged/provenance_audit.json` 只包含旧版 11 项检查，不得再标记为正式通过。

## 4. 修复任务 A：严格 baseline component 合同（P1）

### 4.1 文件范围

- `src/smco/confirmatory.py`
- `tests/test_confirmatory.py`

### 4.2 实现要求

新增独立函数，例如：

```python
def baseline_component_errors(manifest: dict, *, selection: dict | None = None) -> list[str]:
    ...
```

必须验证：

1. `frozen is True` 且 `manifest_sha256` 重算一致；
2. `component_role == "baseline_extension"`；
3. `stage == "e3_companion_baselines"`；
4. `suite == "synthetic_highdim"`；
5. `selection_hash` 非空；传入 selection 时必须与 `selection["selection_hash"]` 相等；
6. baseline 集合精确为 `DE, GA, PSO, SA, GenSA`；
7. 函数集合精确为 `Rastrigin, Ackley, Griewank, Zakharov`；
8. 维度集合精确为 `200, 500, 1000`；
9. instance 集合精确为 `0, 1, 2, 3, 4`；
10. 每个 `(algorithm,function,dimension,instance)` 恰有一条；
11. 总任务数精确为 300，run-id 数也为 300；
12. 所有 `instance_artifact_dir` 指向 `instances/confirmatory_*`；
13. 若传入 instance index，所有条目必须显式满足 `stage == "confirmatory"`；
14. task 中不得出现 winner/base 算法。

`confirmatory_errors()` 对 baseline component 不应简单跳过整个 selection 分支，而应调用
`baseline_component_errors()`。只有该函数无错误时，才跳过普通 manifest 的
winner-present 检查。

### 4.3 必需测试

- 正式 300-task component 通过；
- 错误 selection hash 被拒绝；
- 缺一个 task 被拒绝；
- 复制一个 task、仍保持总行数 300 时被拒绝；
- 仅 1 个 task 被拒绝；
- 错误函数、维度、instance 或 baseline 被拒绝；
- development/missing-stage instance index 被拒绝；
- component_role 单独存在但结构不完整时不能绕过 Gate-F。

### 4.4 提交建议

```text
fix(p1c): enforce exact E3 baseline component closure
```

## 5. 修复任务 B：冻结并严格验证 comparative composite（P1）

### 5.1 文件范围

- `src/smco/confirmatory.py`
- `tests/test_confirmatory.py`

### 5.2 Composite 顶层合同

Composite 至少包含：

```json
{
  "schema_version": "1",
  "composite_type": "comparative_composite",
  "stage": "e3_comparative_analysis",
  "suite": "synthetic_highdim",
  "selection_hash": "...",
  "frozen": true,
  "components": {},
  "algorithms": [],
  "total_runs": 420,
  "composite_sha256": "..."
}
```

实现统一的 `composite_sha256()`，计算时排除 `composite_sha256` 字段自身。禁止在 builder
和 validator 中分别手写不一致的 hash 逻辑。

### 5.3 Builder 必须验证后才能生成

`build_comparative_composite()` 必须：

1. 对 E2 manifest 调用 `verify_manifest()`；
2. 对 baseline component 调用 `verify_manifest()` 和任务 A 的严格 component validator；
3. 验证两个 source audit 文件实时 `passed is True`；
4. 要求 audit 含 `provenance_complete` 且该项通过，拒绝旧 11-check audit；
5. 验证 E2 manifest 为 `e2_factorial_highdim/synthetic_highdim`；
6. 验证 E2 精确含 winner/base × 4 functions × 3 dims × 5 instances = 120 tasks；
7. 验证 E2 和 baseline component 的 selection hash 一致；
8. 验证每个 source `valid_runs.csv` 的 run-id 集合与各自 manifest task 集合精确相等；
9. 验证 source CSV 中实际 algorithm/stage 与 manifest 一致；
10. 验证两组件 run-id 无交集；
11. 自动推导算法集合，并要求精确为 winner、matched base 和五个 baseline；
12. 记录 manifest、valid-runs、audit 和 run-id-set 的 SHA-256；
13. 最后设置 `frozen=True` 并生成 composite hash。

### 5.4 Validator 必须重新读取全部来源

`validate_composite()` 必须：

- 验证 schema/type/stage/suite/frozen；
- 重算并验证 composite hash；
- 从 composite 记录的 manifest 路径（或显式 override 路径）重新读取两个 manifest；
- 重算并验证 manifest hash，而不是只比较字符串字段；
- 重读 audit 与 valid-runs 文件并重算内容 hash；
- 检查实时 audit 的 `passed` 和 `provenance_complete`；
- 检查 `n_runs == 120/300`、`total_runs == 420`；
- 检查 source run-id 精确等于 manifest tasks；
- 检查算法、stage 和 selection hash；
- 检查双源无交集。

### 5.5 必需测试

下列任一改动必须导致验证失败：

- composite hash 改动；
- `frozen=False`；
- schema/type/stage/suite 改动；
- 任一 manifest 被修改或 manifest hash 不符；
- component `n_runs` 改成 1、119、299；
- 总数为 419 或 421；
- source CSV 保持同样行数但换入一个非 manifest run-id；
- source CSV 中 algorithm 或 stage 错误；
- source audit 为旧 11-check、`passed=False` 或 provenance check 失败；
- 两组件 run-id 重叠；
- selection hash 不一致。

现有允许 `5+5=10` composite 通过的测试必须删除或改成明确的 development helper；正式
validator 只能接受 120+300。

### 5.6 提交建议

```text
fix(p1c): freeze and validate exact E3 comparative composite
```

## 6. 修复任务 C：接入生成 CLI 与 Task 12 分析 Gate（P1）

### 6.1 文件范围

- `scripts/generate_smco_evo_manifests.py`
- `scripts/analyze_smco_evo_highdim_paper.py`
- `src/smco/confirmatory.py` 或新建 `src/smco/composite.py`
- `tests/test_confirmatory.py`
- 新增 CLI 测试文件（如 `tests/test_composite_cli.py`）

### 6.2 生成入口

提供可重复执行的 CLI，例如：

```text
generate_smco_evo_manifests.py --stage e3-baseline-component ...
generate_smco_evo_manifests.py --stage e3-composite \
  --e2-manifest ... --e2-merged-dir ... \
  --baseline-manifest ... --baseline-merged-dir ...
```

禁止依赖一次性 Python 片段手工生成正式 composite。

### 6.3 分析入口

E3 正式统计必须显式提供：

```text
--composite result/e3-2026-07-31/e3_comparative_composite.json
--merged-dir result/e3-2026-07-31/merged_composite
```

分析前必须：

1. 调用严格 `validate_composite()`；
2. 验证最终 `merged_composite/valid_runs.csv` 的 420 个 run-id 精确等于两个 component
   run-id 的并集；
3. 验证最终 merged audit 为当前 12-check 且通过；
4. 验证最终 merged 中保留 E2 的 `stage=e2_factorial_highdim`，不允许改写成 E3；
5. 验证 baseline 行为 `stage=e3_companion_baselines`；
6. 不满足任一条件时拒绝生成 primary table、图或报告。

对 E3 stage，`--statistics --merged-dir ...` 但不提供 `--composite` 必须失败；E1/E2
普通分析可保持原入口。

### 6.4 必需测试

- 合法 composite + 精确 420 行可进入统计；
- 不传 composite 时 E3 拒绝；
- composite 验证失败时分析拒绝；
- 最终 merged 缺 1 行或多 1 行时拒绝；
- 最终 merged 混入旧 stage=e3 winner/base 时拒绝；
- E2 行保持原 stage/run-id 时通过。

### 6.5 提交建议

```text
feat(p1c): wire E3 composite into generation and analysis gates
```

## 7. 修复任务 D：E6 start-count provenance 补跑（P1，结果任务）

### 7.1 前置条件

- 任务 A--C 合入并冻结新的完整 40-hex commit SHA；
- 全量测试通过；
- runner 使用 `--confirmatory` provenance fail-fast；
- 确认算法实现没有因 A--C 改动而改变。

### 7.2 重跑要求

- 精确重跑 180 个 start-count task；
- 使用 `--no-resume`，不能让旧 `status=success` 结果被跳过；
- 显式传入冻结的 `--git-commit <40-hex-SHA>`；
- 输出到新的目录，例如 `start_count/raw_dedup2/`；
- 不覆盖旧 `raw_dedup/`；
- merge 输出到新的 `start_count/merged_v2/`；
- 要求 180 行、12 项 audit 全部通过；
- 检查每行 git/environment/machine 非空且 run-id 与 manifest 精确匹配。

### 7.3 结果验收

```text
n_rows = 180
n_missing = 0
n_duplicate = 0
provenance_complete = PASS
overall audit = PASS
```

### 7.4 提交/记录建议

若 `result/` 被 gitignore，不强行提交大型 raw；提交运行记录、命令、冻结 SHA、结果 hash
和 canonical index 更新：

```text
results(e6): regenerate start-count evidence with complete provenance
```

## 8. 修复任务 E：冻结唯一 canonical artifact index（P2）

### 8.1 目的

避免旧 `merged/`、新 `merged_v2/`、`selection/`、`selection_v2/` 并存导致 Task 12 或
人工分析读取错误版本。

### 8.2 建议文件

```text
result/smco-evo-paper-highdim-2026/canonical_artifacts.json
```

至少记录：

- E1 manifest、merged_v2、selection_v2 路径及内容 hash；
- E2 manifest、merged_v2 路径及内容 hash；
- E3 baseline component、baseline merged、composite、composite merged 路径及内容 hash；
- E6 strategy/start-count 正式路径及内容 hash；
- 每项 row count、audit check count、audit passed；
- source git commit(s)；
- 明确标记 E4/E5 当前为 `development_only`；
- 明确标记 E6 schedule 为 `deferred`。

新增 validator，拒绝：路径不存在、hash 不符、audit 非 12 项或未通过。Task 12/13 必须从
该 index 读取，不允许通过默认路径猜测正式产物。

### 8.3 提交建议

```text
chore(results): freeze canonical highdim artifact index
```

## 9. 后续任务（不与当前 P1 修复混提）

### 9.1 E6 schedule 去重

当前 216 tasks 只有 189 个唯一 run-id。后续应：

- 去除 27 个完全相同的重复 task；
- 重新 freeze manifest 和 hash；
- 验证旧 61 条可复用 raw 的 task 内容与新 manifest 完全一致；
- 补跑剩余 128 个唯一任务；
- 使用完整 provenance，最终通过 12 项 audit。

若 spec instances 仍缺失，应继续标为 deferred，不得用 development instance 替代。

### 9.2 E4/E5 正式化

- E4 当前 2,520 行 CSV 完整，但属于 development COCO 输出；
- E5 当前 480 行 CSV 完整，但属于 development COCO 输出；
- 后续应输出统一 per-run outcome、provenance、merge/audit 或提供等价且经过审查的 COCO
  正式证据合同；
- 正式化前只能作为外部 exploratory/supporting evidence。

### 9.3 Task 12/13

只有满足以下条件才能开始：

- E1/E2/E3/E6 纳入论文的每个数据源均在 canonical index 中；
- 每项都通过当前 12 项 audit；
- E3 analysis 必须通过 composite gate；
- E6 未完成部分被明确标为 deferred/excluded；
- E4/E5 的 development/formal 边界写入报告。

之后再实现 ECDF、ERT、hierarchical bootstrap、H1--H4、pairwise/Holm、figures 和最终
Task 13 打包。

## 10. 推荐执行顺序与最终验收

按以下顺序执行，每步独立提交：

1. 任务 A：严格 baseline component；
2. 任务 B：严格 composite builder/validator；
3. 任务 C：接入生成 CLI 和 analysis gate；
4. 全量 pytest，冻结新的代码 SHA；
5. 任务 D：重跑 E6 start-count；
6. 任务 E：冻结 canonical artifact index；
7. 单独处理 E6 schedule；
8. 单独处理 E4/E5 正式化；
9. 最后进入 Task 12/13。

最终 Gate 必须同时满足：

```text
pytest: all pass
E1: 1080 rows, 12 checks pass, winner unchanged or按预注册规则重新冻结
E2: 120 rows, 12 checks pass
E3 baseline: 300 rows, 12 checks pass
E3 composite: exact 120+300=420, strict composite validation pass
E6 strategy: 420 rows, 12 checks pass
E6 start-count: 180 rows, 12 checks pass，或明确 excluded
canonical artifact index: hash validation pass
Task 12 E3 analysis: cannot run without validated composite
```

科学结论约束：SA/GenSA 在当前 E3 上优于 SMCO-EVO 是实验结果，不属于工程 bug，后续
分析与论文必须如实报告。
