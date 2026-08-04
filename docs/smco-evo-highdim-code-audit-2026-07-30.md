# SMCO-EVO 高维论文代码与结果全面审查（2026-07-31）

## 结论

**当前状态：不应打包为论文最终结果；E1/E2 可保留为有价值的证据，E3--E6 必须按本报告修正后再进入主结论。**

本轮审查覆盖当前工作树、冻结 manifest、E1/E2/E3/E6 merged 数据、E4/E5
产物、运行 provenance 和完整 Python 测试。代码的预算、实例隔离、冻结选择和
基础结果合同已经具备良好基础；但可复现性元数据、E3 对 E2 的复用、E4/E5 的
正式结果链条以及 E6 的 identity/manifest 问题，仍阻止形成可投稿的完整证据包。

## 审查范围与验证

| 项目 | 结果 |
| --- | --- |
| Python 测试 | `.venv/bin/python -m pytest -q`：**448 passed in 54.33s** |
| E1 | `result/e1-2026-07-30/merged/`：1,080 行，11 项 provenance audit 全部通过 |
| E2 | `result/e2-2026-07-31/merged/`：120 行，11 项 provenance audit 全部通过 |
| E3 | `result/e3-2026-07-31/merged/`：420 行，11 项基础 provenance audit 全部通过 |
| E6.1 start-count | 180 行，provenance audit 通过 |
| E6.2 strategy | 420 行，但 provenance audit **失败** |
| E4 | 有冻结 2,520-task manifest，尚无正式 COCO 原始/合并结果 |
| E5 | 有冻结 480-task manifest 和汇总 CSV；没有统一 raw outcome、merge 或 audit |

审查时不读取或修改未纳入本实验链条的 `exdata/`。

## 已通过的证据链

### E1：开发集全局选型

E1 使用 18 个候选配置、4 个函数（Rastrigin、Ackley、Griewank、Zakharov）、
3 个维度（200/500/1000）和每单元 5 个实例，共 1,080 个任务。其 merged
provenance audit 通过，selection 输入记录为：

- winner：`PY-SP-SMCO-EVO`；
- winner configuration hash：`0ca95d61e9e2b475`；
- selection hash：`bcf87965006220a0`；
- 选择规则：ECDF-AUC 为主指标，失败/超时按右删失保留在分母。

E1 的科学角色仅是**冻结候选实现**，不能单独被表述为 EVO 优于 SMCO 的确认性
结论。

### E2：独立高维确认性比较

E2 的 120 个任务覆盖冻结 winner 与其 matched base：

- `PY-SP-SMCO-EVO`；
- `PY-BASE-SMCO`；
- 4 个函数 × 3 个维度 × 5 个 confirmatory 实例 × 2 个算法。

冻结 manifest hash 为
`ab19176c5d2af2664f65894cf17976a4aa316d965c206315d6e563cd6f715569`。其实例目录
为 `instances/confirmatory_*`；与 E1 development suite 的 transform hash 和
start-points hash 交集均为 0。

E2 的全部 120 个任务均成功，函数评估均未超过各自预算。只做描述性统计时：

| 指标 | PY-SP-SMCO-EVO | PY-BASE-SMCO |
| --- | ---: | ---: |
| ECDF-AUC | 0.303750 | 0.253083 |
| median log normalized-gap | -4.740094 | -4.740094 |
| ERT, target 1e-1 | 569,988.5 | 571,655.6 |
| ERT, target 1e-5 | 1,724,929.0 | 1,744,794.3 |

最终 normalized-gap 的 60 个配对单元为 **11 胜 / 43 平 / 6 负**（EVO 相对
base，越低越好）。因此目前可谨慎写为“EVO 在部分目标达成速度上呈现改善迹象”，
但不应写成压倒性的最终精度优势；最终表述应等待预注册的 bootstrap、配对检验和
多重比较输出。

## 当前结果所显示的限制

### E3：强基线比较不支持“总体最优”结论

E3 覆盖 7 个算法、4 函数 × 3 维度 × 5 实例，共 420 个任务，基础 coverage/FE/
objective audit 通过。描述性 primary table 为：

| 算法 | ECDF-AUC | median log normalized-gap |
| --- | ---: | ---: |
| PY-SP-SMCO-EVO | 0.303750 | -4.740097 |
| PY-BASE-SMCO | 0.253083 | -4.740094 |
| DE | 0.248500 | -0.712487 |
| GA | 0.290167 | -1.795243 |
| PSO | 0.250000 | -1.144971 |
| SA | 0.370833 | -6.674094 |
| GenSA | 0.394667 | -7.008308 |

在当前强基线集合和统计口径下，GenSA、SA 的 ECDF-AUC 和最终 gap 都优于
SMCO-EVO。即使其余工程问题全部修复，论文也不能声称 SMCO-EVO 在这些高维
synthetic 基准上“总体优于所有强基线”。较可行的主张需要收缩为明确的机制、
适用条件或相对 matched SMCO 的改进，并如实报告该限制。

## 阻断与修复项

### P0：E6 strategy 结果 audit 失败

`result/e6-2026-07-31/strategy/merged/provenance_audit.json` 的
`no_pseudo_duplicates` 检查失败。420 个结果包含 7 个不同 configuration hash，
但 `src/smco/merge_results.py` 的 `_identity_key()` 没有纳入
`configuration_hash` 或完整消融参数，因此把不同策略/参数的合法任务判为重复。

**处置：**在 identity 中加入 configuration hash（或等价的完整算法配置），增加
“同 logical cell、不同消融配置不重复”的回归测试，然后重新 merge 并要求 audit
通过。当前 strategy 消融不得用于图、表或论文结论。

### P1：E1/E2/E3 结果的源码与环境 provenance 不完整

实际 merged 数据显示：

- E1 多数 Python 行的 `git_commit` 为空，且跨 `amax`/`amax-node1` 和多个环境 hash；
- E2 共 102/120 行的 `git_commit` 为空；
- E3 的全部 420 行 `git_commit` 为空；其中 DE/GA/PSO/SA/GenSA 共 300 行的
  `environment_hash` 也为空。

静态代码可复现该问题：`scripts/run_smco_evo_highdim_baselines.py` 调用
`run_baseline_file()` 时只传入 `machine_id`，没有传入默认 git commit 和 environment
hash。

**处置：**先修 baseline dispatcher，要求确认性结果的 git/environment/machine
字段非空且可审计；新增负例测试。对已缺失 provenance 的 E1/E2/E3，优先尝试由
原始作业日志恢复不可变来源；无法恢复时，应在修复后重跑对应阶段，不能把这些行当作
严格可复现的最终证据。

### P1：E3 没有复用 E2 winner/base 结果

计划规定 E2/E3 重复的 winner/base “只计算一次”。当前 E2 和 E3 各有相同的 120
个 `(algorithm,function,dimension,instance)` 逻辑单元，但 task seed 全部不同；E3
生成了另一组 stage 为 `e3_companion_baselines` 的结果，而非引用 E2 的冻结结果。

**处置：**E3 主分析应直接复用通过 E2 audit 的 winner/base 行，只补跑五个 baseline。
若保留 E3 的重复行，它们至多作为额外的开发性重复，不得与 E2 主确认性结果混合。

### P1：E4/E5 尚未形成正式、可审计的外部证据

- E4 的 `e4_bbob_largescale__bbob-largescale.json` 已冻结 2,520 个任务，但当前只
  有 development 日志，没有正式 raw/merged/audit 结果。
- E5 的 `e5_lowdim_check__bbob.json` 已冻结 480 个任务，且有
  `lowdim_summary.csv` 和 `provenance.json`；但不存在每运行统一 outcome、
  `merged/valid_runs.csv` 或通过的 provenance audit。它不能替代 Task 11/12 的
  正式输入。

**处置：**E4 完整运行；E5 改用统一结果合同、merge 与 audit 产物后再运行或导出。
二者完成前不得标为外部验证或低维非退化检查的正式结果。

### P1：E6 schedule manifest 含重复任务且运行不完整

`result/e6-2026-07-31/schedule/e6_ablations__synthetic_highdim.json` 记录 216 条
task，却只有 189 个唯一 `run_id`：27 个 run_id 各有一条**完全相同**的重复 task。
现有 `raw/` 只有 61 个成功结果。

**处置：**先去重并重新 freeze manifest；去重后为 189 个唯一任务，可保留已完成的
61 个有效 raw outcome，并继续完成余下 128 个。禁止直接按旧 manifest 的“216 次”
宣称覆盖完成。

### P2：confirmatory 实例 stage guard 应更严格

当前 guard 拒绝含有 `development` 的实例索引，但 stage 缺失的索引仍可能通过。

**处置：**要求所有 E2/E3/E6 confirmatory instance-index 条目显式满足
`stage == "confirmatory"`，并增加缺失 stage 的负例测试。

### P2：旧审查文本已失效

本文档替换了旧的未跟踪审查稿。旧稿末尾仍含“重新启动 E1 前”的初审结论，与当前
E1/E2/E3 结果和后续代码修复不一致，不应再用于描述项目状态。

## 下一步与最小重跑范围

1. 修复 E3 baseline provenance、E6 identity 与 schedule manifest 去重；补齐测试。
2. 将 E3 主表改为复用 E2 的 winner/base，并重新 merge 强基线结果。
3. 对无法恢复 git/environment provenance 的 E1/E2/E3 结果执行重跑；重新审计。
4. 完成 E4；将 E5 导出/重跑为统一结果合同，并通过 merge audit。
5. 对 E6 strategy 重新 merge，对 E6 schedule 完成去重后的剩余任务。
6. 仅在所有正式阶段 audit 通过后，生成 hierarchical bootstrap、pairwise/Holm、
   ECDF/ERT 图和最终 Task-13 打包报告。

## 当前可用的论文叙述边界

当前最稳妥的表述是：在独立高维 E2 中，冻结的 `PY-SP-SMCO-EVO` 相对其 matched
SMCO base 显示更高的 target-oriented ECDF-AUC，但最终 gap 多数配对持平；在 E3
的强基线集合中，SA 和 GenSA 优于它。故现阶段不能使用“高维总体领先”作为文章
主结论。
