# SMCO-EVO 高维论文实验方案

日期：2026-07-28  
状态：实验设计草案；在确认性实验启动前冻结  
适用范围：SMCO-EVO 的 Python/R 实现、state-preserving/restart 语义及高维优化表现

配套实现计划：
[`docs/smco-evo-highdim-implementation-plan-2026-07-28.md`](smco-evo-highdim-implementation-plan-2026-07-28.md)

## 1. 论文定位

本文不把 SMCO-EVO 描述为对所有问题都无条件更好的通用优化器，而聚焦以下命题：

> 在中高维和超高维连续黑盒优化中，SMCO-EVO 通过阶段性淘汰低质量轨迹并从优质轨迹生成新起点，可以提高固定函数评价预算下的资源利用效率；这种收益受到状态延续语义和语言实现的共同影响。

论文的主要实验维度为 `d = 200--5000`。低维实验只承担三项任务：

1. 验证 Python/R 实现与两种状态语义的正确性；
2. 检查 SMCO-EVO 是否在低维上出现系统性退化；
3. 与原始 SMCO 论文的实验设置建立联系。

现有 `result/rerun-2026-07-20/`、`result/smco-evo/` 和
`result/r-highdim-rerun-2026-07-24/` 视为探索性证据。它们可以用于提出假设、
估计算力和固定默认策略，但不进入新论文的确认性显著性检验。

## 2. 研究问题与预注册假设

### 2.1 研究问题

- **RQ1：EVO 增益。** 在相同语言、相同 SMCO family、相同起点和相同函数评价预算下，EVO 是否优于非 EVO 基线？
- **RQ2：状态语义。** 保留完整轨迹状态和在演化边界重新启动，哪种语义在高维上更有效？
- **RQ3：语言实现。** Python 与 R 的实现包是否产生可复现的性能差异？
- **RQ4：交互效应。** 语言差异是否会改变 state-preserving 与 restart 的相对排序？
- **RQ5：维度效应。** EVO 相对非 EVO 的收益是否随维度增加而增强？
- **RQ6：资源效率。** EVO 是否能用较少起点达到与 `ceil(sqrt(d))` 起点策略相当或更好的结果？

这里的“语言效应”严格解释为“Python/R 实现包的总体差异”，其中可能包含
数值库、随机数实现和运行时开销；除非跨语言轨迹测试完成，否则不把它解释为
编程语言本身的因果效应。

### 2.2 确认性假设

- **H1（主要假设）：** 在 `d = 1000, 3000, 5000` 的保留测试集上，
  开发集选出的 SMCO-EVO 实现，其 ECDF-AUC 高于对应非 EVO 基线。
- **H2：** EVO 相对基线的配对收益随 `log(d)` 增加，维度-算法交互项为正。
- **H3：** 对选中的 SMCO family，state-preserving 与 restart 在保留测试集上存在非零性能差异；采用双侧检验，不预设方向。
- **H4：** 对选中的 SMCO family，语言与状态语义存在或不存在交互，以
  difference-in-differences 及其置信区间报告，不只给出单一 p 值。

低维退化检查、四种进化策略比较、起点数和 evolution schedule 消融均属于次要分析。

## 3. 2x2 实验因子与命名

### 3.1 两个核心因素

| 语言实现 | state-preserving（SP） | restart（RS） |
| --- | --- | --- |
| Python | 已有，需要加入严格 FE 计数 | 待实现 |
| R | 待实现 | 已有，需要明确标注为 RS |

每个单元格均包含三个 SMCO family：

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

这两个版本是并列算法变体，不能把 restart 称为 state-preserving 的“R 复现”。

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

- Python/R 共享完全相同的初始点矩阵；
- 四个 2x2 单元共享同一个主种子；
- 所有输入矩阵和变换参数保存到磁盘并记录 SHA-256；
- 随机种子由稳定哈希从 run key 派生，不依赖任务执行顺序；
- 并行调度不能改变随机结果。

跨语言性能实验允许 Python/R 使用各自实现的随机流，但必须使用相同主种子并进行
多实例配对。另做一个小型 portable-random-tape 轨迹实验，用来判断实现差异来自
算法语义还是 RNG/库差异。

### 4.5 失败与重跑规则

- `NaN`、越界、异常退出、达到 wall-time cap 均记为失败；
- 算法性失败不能事后换种子重跑；
- 只有日志明确证明属于机器、文件系统或调度器故障时才允许同 seed 重跑；
- 重跑记录保留原 run id、失败原因和 superseded 标记；
- 任何算法不得因阶段性结果较差而提前从实验矩阵中删除。

## 5. 分阶段实验

### E0：实现合同和跨语言轨迹验证

目的：在大规模计算前证明四个 2x2 单元确实实现了预期语义。

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
6. Python/R 在 portable random tape 下的淘汰集合、父代索引、replacement 点和
   best-so-far 轨迹在数值容差内一致；
7. maximize/minimize 方向在所有入口一致；
8. 同一 seed 重跑完全可复现。

E0 未通过时不得启动 E1。

### E1：高维开发集与全局选型

目的：选择一个全局 SMCO family、语言实现和状态语义；不做确认性显著性结论。

问题集：

- 函数：Rastrigin、Ackley、Griewank、Michalewicz；
- 维度：`d = 200, 500, 1000`；
- 每个函数-维度 5 个实例；
- 共 `4 x 3 x 5 = 60` 个问题实例。

实例变换遵循原始 SMCO 论文的思想：

1. 平移最优点；
2. 每个坐标独立进行左右非对称定义域变换；
3. `d = 200` 可使用完整正交旋转；
4. `d = 500, 1000` 使用固定块大小的随机 block rotation 加坐标置换，
   防止构造和评价成本变成 `O(d^2)`；
5. 变换后仍保存已知全局最优值和可验证的最优点。

算法矩阵：

- 12 个 EVO 单元：`2 languages x 2 semantics x 3 families`；
- 6 个非 EVO 基线：`2 languages x 3 families`；
- 共 18 个算法配置。

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

随后创建只读的确认性 manifest。主文可以重点展示该赢家，但补充材料必须展示
E1 的全部 18 个配置。

### E1B：可跨语言基线的实现选型

如果同一个 comparison algorithm 在 Python 和 R 中都有语义相符的成熟实现，
也按开发集规则选择一次全局实现：

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

E1B 选择在 E2/E3 前与 SMCO 赢家一起冻结。

### E2：1000--5000 维 2x2 确认性实验

目的：在新的实例上验证语言、状态语义以及二者交互。

问题集：

- 核心函数：Rastrigin、Ackley、Rosenbrock；
- 维度与独立重复数：
  - `d = 1000`：5 个实例；
  - `d = 3000`：3 个实例；
  - `d = 5000`：2 个实例；
- 所有实例 seed 与 E1 完全不重叠；
- 使用 shift + asymmetry + coordinate permutation + block rotation。

算法：

- E1 选中的 family 的四个 2x2 EVO 单元；
- Python 与 R 下对应的两个非 EVO family 基线；
- 共 6 个配置。

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
3 functions x (5 + 3 + 2) instances x 6 configurations = 180 runs
```

### E3：高维强基线对比

目的：判断选中的 SMCO-EVO 是否只在 SMCO family 内部占优，还是能与强黑盒算法竞争。

问题集：

- Rastrigin、Ackley、Rosenbrock、Griewank、Zakharov；
- `d = 1000, 3000, 5000`；
- 重复数同 E2；
- 与 E2 使用相同实例以形成配对，但不改变 E2 的 2x2 统计。

算法：

- E1 冻结的 SMCO-EVO 赢家；
- 其 matched non-EVO base；
- GenSA；
- SA；
- DE；
- PSO；
- 适用于高维的 separable/limited-memory CMA-ES。

对每个比较算法，若存在多个语言实现，在 E1B 上用同一规则选择一个全局实现，
但必须在 E2/E3 前冻结。禁止在确认性结果出来后按函数挑选 Python 或 R 版本。

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
- E1B 中排名最高的三个强基线，但这三个名称必须在 E4 manifest
  启动前写死；
- 共 5 个配置。

预算：

```text
n_starts = 8 for SMCO family
B_max = 1000 * d FE
checkpoints = {100, 250, 500, 1000} * d FE
```

总运行数：

```text
360 problem instances x 5 configurations = 1800 runs
```

E4 使用 COCO 官方 ERT、ECDF 和 target definitions 生成外部基准图。

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
- H3：在 Python 和 R 内分别比较 SP 与 RS；
- H4：计算
  `(R_SP - R_RS) - (PY_SP - PY_RS)` 的 difference-in-differences；
- 同时报告 median paired log-ratio、bootstrap 95% CI 和 probability of superiority；
- H3/H4 的有限组主检验使用 Holm 校正，`alpha = 0.05`；
- 函数和实例采用分层 bootstrap，先重采样函数，再在函数内重采样实例。

当 target 未命中时按右删失处理；不得把未命中样本删除后只在成功 run 上比较。

### 6.4 结果选择和展示规则

允许在主文突出“开发集选出的全局赢家”，但必须满足：

1. 赢家只在 E1 选择一次；
2. E2--E5 之前冻结；
3. 不按函数、维度、seed 或目标精度分别选择实现；
4. 主文明确标注选择过程；
5. 补充材料报告全部 2x2 单元和 E1 全部 family；
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

1. **算法图：** state-preserving 与 restart 在演化边界处的状态变化；
2. **表 1：** E1 的全局选型规则和冻结结果；
3. **图 1：** E2 中四个 2x2 单元的高维 ECDF；
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
- Python/R 环境与 portable-random-tape 对齐结果。

## 9. 实施顺序与闸门

### Gate 1：实现和计数

1. 增加统一 FE counter/hard budget；
2. 实现 Python-RS；
3. 实现真正的 R-SP；
4. 明确重命名或标记现有 R-RS；
5. 完成 E0 测试。

### Gate 2：小型性能 pilot

- 每个 E1 条件先跑 1 个实例和 `100 * d FE`；
- 检查结果 schema、预算、方向、失败和资源估算；
- 根据纯运行成本确定各维度 wall-time cap，并在完整实验前冻结；
- pilot 只修复基础设施问题，不据此换算法配置。

### Gate 3：开发集选型

- 完整执行 E1；
- 根据预注册规则选择全局赢家；
- 对存在可比 Python/R 实现的 comparison family 完成 E1B；
- 写入 `selected_configuration_hash`；
- 生成并冻结确认性 manifest。

### Gate 4：确认性实验

1. E2 高维 2x2；
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
- `tests/test_cross_language_traces.py`：读取 Python/R trace 后比较。

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
- 增加 portable RNG 后的跨语言全轨迹对齐；
- 若理论工作可推进，给出 EVO 调度不破坏 survivor 单轨迹收敛性质的命题。

## 12. 方案的关键取舍

本方案有意作出以下取舍：

- 高维是主要结论，低维只做边界检查；
- 用 FE 而不是 `iter_max` 保证公平；
- 只选一个全局语言/语义实现，不按任务事后取最好值；
- 保留全部 2x2 结果，使“选择较好实现”可审计；
- 使用 block rotation 让 1000--5000 维的非轴对齐测试可计算；
- 新确认性结果与已经观察过的 7 月结果严格分开。

只要上述边界不被实验结果反向修改，最终文章就可以同时强调 SMCO-EVO 的高维优势，
并避免“语言挑最好结果”“重复行当独立样本”或“迭代次数伪公平”等主要审稿风险。
