# SMCO-EVO 高维论文实验方案

日期：2026-07-28  
状态：高维主线修订版；Task 1--4 已完成，确认性实验尚未启动
适用范围：SMCO-EVO 的 Python/R 实现、state-preserving/restart 语义及高维优化表现

配套实现计划：
[`docs/smco-evo-highdim-implementation-plan-2026-07-28.md`](smco-evo-highdim-implementation-plan-2026-07-28.md)

## 1. 论文定位

本文不把 SMCO-EVO 描述为对所有问题都无条件更好的通用优化器，而聚焦以下命题：

> 在中高维和超高维连续黑盒优化中，SMCO-EVO 通过阶段性淘汰低质量轨迹并从优质轨迹生成新起点，可以提高固定函数评价预算下的资源利用效率。

论文的主要实验维度为 `d = 200--5000`。低维实验只承担三项任务：

1. 验证候选实现和函数评价预算的正确性；
2. 检查 SMCO-EVO 是否在低维上出现系统性退化；
3. 与原始 SMCO 论文的实验设置建立联系。

现有 `result/rerun-2026-07-20/`、`result/smco-evo/` 和
`result/r-highdim-rerun-2026-07-24/` 视为探索性证据。它们可以用于提出假设、
估计算力和固定默认策略，但不进入新论文的确认性显著性检验。

## 2. 研究问题与预注册假设

### 2.1 研究问题

- **RQ1：EVO 增益。** 在相同起点和相同函数评价预算下，选定的 SMCO-EVO 是否优于对应非 EVO 基线？
- **RQ2：高维竞争力。** 在 `d = 1000, 3000, 5000` 上，SMCO-EVO 是否能与强全局优化算法竞争？
- **RQ3：维度效应。** EVO 相对基线的收益是否随维度增加而增强？
- **RQ4：非轴对齐稳健性。** 收益能否在 shift、asymmetry、permutation 和 block rotation 后保持？
- **RQ5：资源效率。** EVO 是否能用较少起点达到与 `ceil(sqrt(d))` 起点策略相当或更好的结果？
- **RQ6：机制来源。** 收益主要来自淘汰、补点、DE 变异还是状态记忆？

Python/R 和 state-preserving/restart 只作为开发集中的候选实现，不作为论文主要研究因素。
论文 Methods 必须披露最终选中的语言、代码版本和语义，但不需要证明不同语言逐轨迹一致，
也不需要对“语言效应”作统计结论。

### 2.2 确认性假设

- **H1（主要假设）：** 在 `d = 1000, 3000, 5000` 的保留测试集上，
  开发集选出的 SMCO-EVO 实现，其 ECDF-AUC 高于对应非 EVO 基线。
- **H2：** EVO 相对基线的配对收益随 `log(d)` 增加，维度-算法交互项为正。
- **H3：** 在新的非轴对齐高维实例上，选定 SMCO-EVO 的优势仍然存在。
- **H4：** 在相同 FE 预算下，固定 8 个起点的 SMCO-EVO 不劣于
  `ceil(sqrt(d))` 起点配置，并具有更高资源效率。

低维退化检查、四种进化策略比较、起点数和 evolution schedule 消融均属于次要分析。

## 3. 候选实现与选型边界

### 3.1 已完成的候选实现

| 语言实现 | state-preserving（SP） | restart（RS） |
| --- | --- | --- |
| Python | 已完成 | 已完成 |
| R | 已完成 | 已完成 |

Task 1--4 已完成，因此四种候选实现均可用于小规模开发选型。每个单元格均可包含三个 SMCO family：

- `SMCO_EVO`
- `SMCO_R_EVO`
- `SMCO_BR_EVO`

推荐使用不会与 `SMCO_R` 混淆的结果标识，例如：

- `PY-SP-SMCO-EVO`
- `PY-RS-SMCO-EVO`
- `R-SP-SMCO-EVO`
- `R-RS-SMCO-EVO`

结果表中不要把 `R` 同时用作语言和 refine family 的缩写。至少保留以下独立字段：

```text
language = python | r
state_semantics = state_preserving | restart
family = smco | smco_refine | smco_boost_refine
evolutionary = true | false
```

这张表不是确认性实验的完整因子设计。E1 只用它选择一个全局实现；E1 之后，
E2--E5 只运行冻结赢家及其 matched base，不再把语言或语义比较扩展成论文主问题。

### 3.2 两种语义的算法合同

**State-preserving：**

- survivor 保留 `x_current`、`f_current`、`s_value`、`current_n`、
  `x_runmax`、`f_runmax` 和停止状态；
- 只为被淘汰轨迹建立新状态；
- survivor 在边界后执行的是原递推的下一步。

**Restart：**

- 每个演化边界都从该轨迹的 `x_runmax` 重新初始化；
- 重置局部递推状态，但保留全局 best-so-far archive；
- 重启初始化产生的函数评价必须计入总预算。

这两个版本是并列算法变体。最终选中哪一个，就按其真实语义定义论文中的 SMCO-EVO；
不能在 Methods 中把 restart 写成 state-preserving，反之亦然。

## 4. 公平预算与共同运行协议

### 4.1 函数评价次数是主要预算

所有质量比较使用函数评价次数（FE）作为主预算，wall-clock time 只作为工程指标。
当前的 `iter_max` 不能作为跨算法公平预算，因为一次 SMCO center-difference 迭代
约包含 `2d + 1` 次目标函数评价，而且起点数、refine、boost 和 restart 都会改变
总评价次数。

需要加入精确 objective counter 和 hard budget：

- 初始化、有限差分、`x_next`、replacement 初始化、边界裁剪后的重新评价全部计入 FE；
- 当剩余预算不足以完成一个不可分割步骤时，不启动该步骤；
- 输出实际 `fe_used`，保证 `fe_used <= fe_budget`；
- 保存 best-so-far 相对于 FE 的轨迹；
- 演化点按搜索阶段 FE 预算的 `50%` 和 `75%` 触发，不再按表面 `iter_max` 触发。

对 SMCO-BR：

- regular branch 与 boosted branch 各分配总预算的 `50%`；
- 每个 branch 内部再按固定 `refine_ratio = 0.5` 分配搜索和 refine；
- 不允许两个 branch 各自使用一份完整预算。

### 4.2 默认 EVO 配置

现有探索性策略扫参中 `rand1bin` 的整体表现最好，因此确认性实验冻结：

```text
evolution_strategy = rand1bin
evolution_points = (0.5, 0.75)
elimination_rate = 0.25
de_factor = 0.8
de_crossover = 0.7
```

不在看到保留测试结果后修改这些参数。`best1bin`、
`current-to-best1bin` 和 `sobol` 仅进入预先定义的消融实验。

### 4.3 起点数

高维主实验统一使用：

```text
n_starts = 8
```

原因不是假定 8 一定最优，而是：

- 8 个起点在淘汰 25% 后仍有 6 个 survivor，可正常执行 `rand1bin`；
- 固定起点数避免 `ceil(sqrt(d))` 随维度增加而吞噬更多 FE；
- 在相同 FE 预算下，各维度可以获得更接近的有效迭代数；
- 起点数量本身另设消融，不与主效应混在一起。

起点消融使用 `8`、`16` 和 `ceil(sqrt(d))`，所有方案仍共享相同 FE 预算。

### 4.4 随机化与配对

每个 `(suite, function, dimension, instance, replication)` 生成一个不可变运行清单：

- E1 中所有候选实现共享完全相同的初始点矩阵；
- 候选实现使用各自独立的确定性随机流：种子由 run key（**含 `algorithm_id`**）稳定
  派生，同一算法在同一任务上完全可复现，但不同算法不强制共用随机流。异构优化器
  （SMCO / DE / GenSA / …）的共同随机数（CRN）既不可对齐也无必要，配对比较的方差
  控制由共享的初始点矩阵 + 多实例聚合承担，而非共享随机流；
- 所有输入矩阵和变换参数保存到磁盘并记录 SHA-256；
- 随机种子由稳定哈希从 run key 派生，不依赖任务执行顺序；
- 并行调度不能改变随机结果。

Python/R 可以使用各自实现的随机流。开发集选型依靠多实例总体表现，不要求逐轨迹一致。
Portable random tape 仅在出现无法解释的巨大差异、怀疑实现错误时作为可选诊断工具，
不阻塞高维主实验。

### 4.5 失败与重跑规则

- `NaN`、越界、异常退出、达到 wall-time cap 均记为失败；
- 算法性失败不能事后换种子重跑；
- 只有日志明确证明属于机器、文件系统或调度器故障时才允许同 seed 重跑；
- 重跑记录保留原 run id、失败原因和 superseded 标记；
- 任何算法不得因阶段性结果较差而提前从实验矩阵中删除。

## 5. 分阶段实验

### E0：核心实现合同验证（已完成）

目的：在大规模计算前证明 FE budget、Python restart 和 R state-preserving
均实现了预期合同。

配置：

- 函数：concave quadratic、Rastrigin、Rosenbrock；
- 维度：`d = 2, 10`；
- 起点：固定 8 个；
- seed：5 个；
- 短预算：足够跨过两个演化边界。

必须验证：

1. SP survivor 的内部状态在边界前后连续；
2. RS survivor 的递推状态确实重置；
3. replacement 只从 survivor 的 running-best points 生成；
4. 淘汰排序使用 running-best value；
5. 所有函数评价均被计数；
6. maximize/minimize 方向在所有入口一致；
7. 同一 seed 重跑完全可复现。

完成证据：

- Python FE budget：16 tests passed；
- Python semantics：13 tests passed；
- R budget：`ALL R BUDGET TESTS PASSED`；
- R semantics：`ALL R EVOLUTION-SEMANTICS TESTS PASSED`。

跨语言 portable-random-tape 不属于 E0 必需验收。

### E1：高维开发集与全局选型

目的：选择一个全局 SMCO family、语言实现和状态语义；不做确认性显著性结论。

问题集：

- 函数：Rastrigin、Ackley、Griewank、Zakharov；
- 维度：`d = 200, 500, 1000`；
- 每个函数-维度 5 个实例；
- 共 `4 x 3 x 5 = 60` 个问题实例。

> 注：E1 原列 Michalewicz，但高维 Michalewicz 全局最小值无解析解（`assign_config`
> 对任意维度返回 `known_min=None`），无法满足下方第 5 点"保存已知全局最优值"，
> 也会污染 `normalized_gap` 与 `target-hit` 指标分母。故改用 Zakharov
> （已知全局最小值 `0`，最优解 `x = 0`），保持 `4 x 3 x 5 = 60` 结构不变。
> 此变更于 2026-07-29 经用户确认。

实例变换遵循原始 SMCO 论文的思想：

1. 平移最优点；
2. 每个坐标独立进行左右非对称定义域变换；
3. `d = 200` 可使用完整正交旋转；
4. `d = 500, 1000` 使用固定块大小的随机 block rotation 加坐标置换，
   防止构造和评价成本变成 `O(d^2)`；
5. 变换后仍保存已知全局最优值和可验证的最优点。

候选矩阵：

- 12 个 EVO 单元：`2 languages x 2 semantics x 3 families`；
- 6 个非 EVO 基线：`2 languages x 3 families`；
- 共 18 个算法配置。

这里的完整矩阵只用于选择 canonical implementation，不用于在论文中研究语言或语义效应。
如果 pilot 表明全部 18 个配置成本过高，可以先在较小开发子集上筛到 3--4 个候选，
但筛选规则和子集必须在运行前固定。

预算：

```text
n_starts = 8
B_max = 1000 * d FE
checkpoints = {100, 250, 500, 1000} * d FE
```

总运行数：

```text
60 problem instances x 18 configurations = 1080 runs
```

### E1 选型规则

每个配置只产生一个跨所有函数、维度和实例的全局分数，禁止按函数或维度选择不同实现。

排序顺序：

1. 主要分数：相对目标上的 ECDF-AUC；
2. 若 AUC 差距小于 1%，比较 `B_max` 时的 median normalized log-gap；
3. 若仍相当，比较失败率；
4. 若仍相当，在固定硬件、单线程条件下比较 median wall time。

选择结果冻结为：

```text
selected_family
selected_language
selected_state_semantics
selected_configuration_hash
```

随后创建只读的确认性 manifest。主文只需要说明赢家的选型规则、语言、代码版本和语义；
E1 的完整候选结果放补充材料或开发记录。

### E1B：Comparison implementation 选择（可选）

语言对比不是论文目标。Comparison algorithms 默认直接使用项目中最成熟、接口最稳定的
canonical implementation，并在 manifest 中固定。只有同一个算法确实存在两个语义相符、
成本可控的成熟实现时，才按开发集规则选择一次：

- 使用 E1 的 60 个问题实例；
- 使用相同 FE budget、主种子和失败规则；
- 每个算法 family 只能选择一个全局语言实现；
- Python/R 两个候选的完整结果都进入补充材料；
- 不允许按函数或维度拼接“oracle best”；
- 如果两个实现的算法定义、初始化或停止条件不能对齐，则把它们视为不同算法，
  不把较好的一个称为该算法的语言版本。

候选 comparison family 最多包括 DE、PSO、SA、GenSA 和适合高维的 CMA-ES。
若五个 family 都具有两个可比语言实现，则 E1B 上限为：

```text
60 problem instances x 5 families x 2 languages = 600 runs
```

如果不执行 E1B，则在 pilot 前直接冻结 comparison implementation。无论是否执行，
都不能在 E2/E3 结果出来后更换实现。

### E2：1000--5000 维核心确认性实验

目的：在新的实例上验证冻结 SMCO-EVO 是否优于 matched non-EVO base。

问题集：

- 核心函数：Rastrigin、Ackley、Rosenbrock；
- 维度与独立重复数：
  - `d = 1000`：5 个实例；
  - `d = 3000`：3 个实例；
  - `d = 5000`：2 个实例；
- 所有实例 seed 与 E1 完全不重叠；
- 使用 shift + asymmetry + coordinate permutation + block rotation。

算法：

- E1 冻结的唯一 SMCO-EVO 赢家；
- 同语言、同 family 的 matched non-EVO base；
- 共 2 个配置。

预算：

```text
n_starts = 8
B_max = 2000 * d FE
checkpoints = {100, 250, 500, 1000, 2000} * d FE
```

额外将“冻结赢家”和其 matched base 运行到 `5000 * d FE`，作为预先声明的
长预算 anytime 扩展，不根据 `2000 * d` 的结果决定是否执行。

总运行数：

```text
3 functions x (5 + 3 + 2) instances x 2 configurations = 60 runs
```

### E3：高维强基线对比

目的：判断选中的 SMCO-EVO 是否只在 SMCO family 内部占优，还是能与强黑盒算法竞争。

问题集：

- Rastrigin、Ackley、Rosenbrock、Griewank、Zakharov；
- `d = 1000, 3000, 5000`；
- 重复数同 E2；
- 与 E2 使用相同实例以形成配对，但不重复计算 winner/base。

算法：

- E1 冻结的 SMCO-EVO 赢家；
- 其 matched non-EVO base；
- GenSA；
- SA；
- DE；
- PSO；
- 适用于高维的 separable/limited-memory CMA-ES。

所有比较算法的具体实现必须在 E2/E3 前冻结。禁止在确认性结果出来后按函数挑选
Python 或 R 版本。

预算与 E2 相同，主结果到 `2000 * d FE`。结果较差或运行较慢的算法仍保留，
超出预注册 wall-time cap 时按失败处理。

总运行数：

```text
5 functions x (5 + 3 + 2) instances x 7 configurations = 350 runs
```

E2 与 E3 重复的 winner/base 结果只计算一次。

### E4：独立外部验证 - BBOB large-scale

目的：避免结论只建立在仓库自带的五个函数上。

使用 COCO `bbob-largescale`：

- 全部 24 个函数；
- `d = 160, 320, 640`；
- 官方实例 `1--5`；
- 共 `24 x 3 x 5 = 360` 个问题实例。

算法：

- 冻结的 SMCO-EVO 赢家；
- matched base；
- E3 中预先定义的五个强基线：DE、GA、PSO、SA、GenSA；名称、版本和参数
  必须在 E4 manifest 启动前写死；
- 共 7 个配置。

这是相对原 5 配置 E4 设计的增强：保留全部五个 E3 baseline，以增加外部基准的
对照密度。该增强在确认性实验启动前冻结，不得由运行时 CLI 选择或删改 baseline。

预算：

```text
n_starts = 8 for SMCO family
B_max = 1000 * d FE
checkpoints = {100, 250, 500, 1000} * d FE
```

总运行数：

```text
360 problem instances x 7 configurations = 2520 runs
```

E4 使用 COCO 官方 ERT、ECDF 和 target definitions 生成外部基准图。原始结果必须
保留到 instance level；汇总结果必须显式聚合每个 `(function, dimension, algorithm)`
下的 5 个官方 instances，禁止以最后一个 instance 覆盖其余记录。

#### R winner 的 COCO 解释边界（已冻结决策）

COCO Python 接口可用，而本项目不实现 R COCO runner。若 E1 的冻结 winner 是 R
实现，E4/E5 运行对应的 Python family/semantics port，并将该产物固定标记为
**`Python port external check`**：

- 它不是冻结 R winner 的直接外部验证，不能用来加强或泛化 R winner 的主结论；
- R winner 的主证据仍仅来自 E2/E3 中的 R winner 与 matched R base；
- `selection.json`、E4/E5 `provenance.json`、图表标题/表注和报告正文必须保存
  original winner/language、实际 Python algorithm id，以及
  `python_port_external_check=true`；
- 若 winner 本身为 Python，E4/E5 才是冻结 winner 的直接外部验证。

### E5：低维非退化检查

低维不是主结论，只检查“高维有效是否以低维系统退化为代价”。

- BBOB 全部 24 个函数；
- `d = 5, 20`；
- 官方实例 `1--5`；
- 只比较冻结赢家与 matched base；
- `B_max = 2000 * d FE`；
- 共 `24 x 2 x 5 x 2 = 480` runs。

低维结果放补充材料。除非出现严重退化，不据此推翻或重新选择高维赢家。

### E6：机制消融

消融使用开发实例，不接触确认性实例。

### E6.1 起点数量

```text
n_starts in {8, 16, ceil(sqrt(d))}
d in {1000, 3000, 5000}
functions = {Rastrigin, Ackley, Rosenbrock}
```

比较质量必须保持相同 FE 预算，不能保持相同 `iter_max`。

### E6.2 进化操作

```text
strategy in {rand1bin, current-to-best1bin, best1bin, sobol}
```

`sobol` 是关键对照：如果 DE 策略未优于 Sobol replacement，则论文应把贡献描述为
“阶段性淘汰和补点”，而不是强调 differential evolution 机制。

### E6.3 调度和淘汰率

最小消融集合：

```text
evolution_points in {(0.5, 0.75), (0.25, 0.5, 0.75)}
elimination_rate in {0.25, 0.5}
```

### E6.4 状态组成

在 SP 实现中分别重置以下状态以解释机制：

- 只重置 `s_value/current_n`；
- 只重置 running-best archive；
- 完整 restart。

该消融能够区分收益来自历史递推、running-best 记忆，还是单纯从好点重新开始。

## 6. 指标和统计分析

### 6.1 主要质量指标

对自定义已知最优值问题，统一转换成 minimization gap，并以共同初始点的 gap 归一化：

```text
normalized_gap(t)
  = max(best_so_far(t) - f*, eps)
    / max(initial_reference - f*, eps)
```

其中 `initial_reference` 是共同起点集合的中位目标值，`eps = 1e-12`。

主要指标：

1. 相对目标 `{1e-1, 1e-2, 1e-3, 1e-5}` 的 target-hit FE；
2. ECDF 及其在 `log10(FE / d)` 上的 AUC；
3. ERT；
4. 各 checkpoint 的 normalized log-gap；
5. 成功率、NaN/timeout/越界率。

Wall time、峰值内存和 FE/s 作为次要工程指标，必须在相同机器类型、线程数和负载规则下采集。

### 6.2 统计单位

统计单位是独立的 `(function, dimension, instance)`，不是：

- 同一 run 的多个 FE checkpoint；
- 同一非 EVO 基线被复制到四个 strategy 标签后的重复行；
- 同一结果文件的多次汇总。

### 6.3 主要检验

- H1：winner EVO 对 matched base 的分层 bootstrap ECDF-AUC 差异；
- H2：paired gain 对 `log(d)` 的斜率，函数作为分层/随机效应；
- H3：在 shift/asymmetry/permutation/block-rotation 实例上，
  winner EVO 对 matched base 的配对效应；
- H4：`n_starts=8` 对 `ceil(sqrt(d))` 的非劣效与 FE 效率比较；
- 同时报告 median paired log-ratio、bootstrap 95% CI 和 probability of superiority；
- 有限组主检验使用 Holm 校正，`alpha = 0.05`；
- 函数和实例采用分层 bootstrap，先重采样函数，再在函数内重采样实例。

当 target 未命中时按右删失处理；不得把未命中样本删除后只在成功 run 上比较。

### 6.4 结果选择和展示规则

允许在主文突出“开发集选出的全局赢家”，但必须满足：

1. 赢家只在 E1 选择一次；
2. E2--E5 之前冻结；
3. 不按函数、维度、seed 或目标精度分别选择实现；
4. 主文明确标注选择过程；
5. 补充材料或开发记录报告 E1 候选矩阵，不把它解释为语言比较实验；
6. 若保留测试上赢家不再占优，必须如实报告，不回到 E1 选择第二名。

因此，主文可以说“selected implementation”，不能说“每个任务取 Python/R 中较好的结果”。

## 7. 结果文件和可复现性合同

每一行至少包含：

```text
run_id
suite
function
dimension
instance
replication
seed
language
state_semantics
family
evolutionary
evolution_strategy
n_starts
fe_budget
fe_used
best_value
known_optimum
normalized_gap
target_hit_fe_*
wall_time_sec
peak_memory_mb
status
failure_reason
machine_id
git_commit
environment_hash
start_points_hash
instance_hash
configuration_hash
```

推荐目录：

```text
result/smco-evo-paper-highdim-2026/
  manifests/
    development.yaml
    confirmatory.yaml
  raw/
    e0-contract/
    e1-development/
    e2-factorial-highdim/
    e3-baselines-highdim/
    e4-bbob-largescale/
    e5-lowdim-check/
    e6-ablations/
  merged/
  analysis/
  figures/
  logs/
  report.md
```

原始行只追加，不原地覆盖。合并和报告脚本必须从 raw artifacts 重建全部统计。

## 8. 论文表图规划

主文建议保留以下内容：

1. **算法图：** 冻结 SMCO-EVO 的淘汰、补点和继续搜索过程；
2. **表 1：** E1 的全局选型规则和冻结结果；
3. **图 1：** E2 中 winner 与 matched base 的高维 ECDF；
4. **图 2：** winner、matched base 与强基线在 E3 上的 ECDF；
5. **图 3：** EVO/base 性能比随维度变化；
6. **图 4：** `d = 1000, 3000, 5000` 的 anytime curves；
7. **表 2：** H1--H4 的效应量、置信区间和校正后 p 值；
8. **图 5：** BBOB-large-scale 函数组 ECDF。

补充材料：

- E1 全部 18 个配置；
- 全部函数-维度-实例结果；
- 起点数、策略、调度和状态组成消融；
- 低维非退化检查；
- wall time、内存、失败和 timeout；
- 最终实现的语言、版本、语义与环境；
- 如实际执行，再附 portable-random-tape 诊断结果。

## 9. 实施顺序与闸门

### Gate 1：实现和计数（已完成）

1. 增加统一 FE counter/hard budget；
2. 实现 Python-RS；
3. 实现真正的 R-SP；
4. 明确重命名或标记现有 R-RS；
5. 完成 E0 测试。

2026-07-28 已通过 Python/R budget 与 semantics 合同测试。Task 5 不再属于 Gate 1。

### Gate 2：小型性能 pilot

- 每个 E1 条件先跑 1 个实例和 `100 * d FE`；
- 检查结果 schema、预算、方向、失败和资源估算；
- 根据纯运行成本确定各维度 wall-time cap，并在完整实验前冻结；
- pilot 只修复基础设施问题，不据此换算法配置。

### Gate 3：开发集选型

- 完整执行 E1；
- 根据预注册规则选择全局赢家；
- comparison implementations 在 pilot 前冻结；E1B 如无必要可以跳过；
- 写入 `selected_configuration_hash`；
- 生成并冻结确认性 manifest。

### Gate 4：确认性实验

1. E2 winner vs matched base；
2. E3 高维强基线；
3. E4 BBOB-large-scale；
4. E5 低维检查；
5. E6 机制消融。

确认性 raw 文件生成后不得修改假设、主要指标、算法选择或排除规则。

## 10. 建议新增的代码边界

最终文件名可在实现计划中进一步确认，建议职责如下：

- `src/smco/evaluation.py`：objective counter、hard budget 和 trace；
- `src/smco/optimizer.py`：Python SP/RS 调度入口；
- `vendor/SMCO_R/main/SMCO_evo.R`：明确保留 R-RS；
- `vendor/SMCO_R/main/SMCO_evo_stateful.R`：真正的 R-SP；
- `src/smco/highdim_instances.py`：shift/asymmetry/block-rotation 实例；
- `scripts/run_smco_evo_highdim_factorial.py`：E1/E2；
- `scripts/run_smco_evo_highdim_baselines.py`：E3；
- `scripts/run_smco_evo_bbob_largescale.py`：E4；
- `scripts/analyze_smco_evo_highdim_paper.py`：唯一正式汇总入口；
- `tests/test_evaluation_budget.py`：FE 预算回归；
- `tests/test_evolution_semantics.py`：SP/RS 合同；
- `tests/test_cross_language_traces.py`：可选诊断，不是正式实验前置条件。

实现时先写详细代码计划，再逐步修改；不要直接改动现有
`result/rerun-2026-07-20/` 或覆盖已有高维结果。

## 11. 最小可投稿版本与增强版本

### 最小可投稿版本

- E0 全部通过；
- E1 完成并冻结赢家；
- E2 完整；
- E3 完整；
- E5 完成；
- E6 至少完成 strategy 和 start-count 消融；
- 现有高维结果作为探索性补充。

### 强化版本

- 加入完整 E4 BBOB-large-scale；
- 完成状态组成消融；
- 对随机 maximum-score/empirical-welfare 问题做独立验证；
- 在发现实现异常时增加 portable RNG 跨语言全轨迹诊断；
- 若理论工作可推进，给出 EVO 调度不破坏 survivor 单轨迹收敛性质的命题。

## 12. 方案的关键取舍

本方案有意作出以下取舍：

- 高维是主要结论，低维只做边界检查；
- 用 FE 而不是 `iter_max` 保证公平；
- 只选一个全局语言/语义实现，不按任务事后取最好值；
- 语言和语义只用于开发选型，不作为确认性研究因素；
- 使用 block rotation 让 1000--5000 维的非轴对齐测试可计算；
- 新确认性结果与已经观察过的 7 月结果严格分开。

只要上述边界不被实验结果反向修改，最终文章就可以同时强调 SMCO-EVO 的高维优势，
并避免“语言挑最好结果”“重复行当独立样本”或“迭代次数伪公平”等主要审稿风险。
