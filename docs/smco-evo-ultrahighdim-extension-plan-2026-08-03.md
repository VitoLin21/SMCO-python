# SMCO-EVO 缺失函数补充与超高维扩展计划

日期：2026-08-03  
状态：待实现、冻结和派跑  
定位：现有 E1--E6 之后的 prospective extension；不改写既有 canonical 结果

## 1. 研究目标与证据边界

本计划包含两个相互独立但可以复用结果的 campaign：

1. **E3-F：现有 E3 的缺失函数补充。** 在原 E3 的 `d={200,500,1000}`、5 个实例和 7 个算法上补齐 Rosenbrock、Levy、Schwefel 2.26、High-conditioned Ellipsoid。
2. **E7：超高维可扩展性实验。** 在 `d={1000,2000,3000,5000,10000}` 上比较 12 个算法，检验 SMCO-EVO 的质量、稳定性和 72 小时 operational deadline 下的可用性。

旧的 `result/rerun-2026-07-20/` 只用于估算计算量和提出假设。由于旧数据的实例、预算、实现版本和重复数与本计划不同，禁止直接并入 E3-F/E7 的确认性统计。

E3-F/E7 的函数、算法、实例、预算、假设、超时语义和统计规则必须在首个正式任务前冻结。正式结果出现后不得按函数、维度或语言挑选实现。

## 2. 函数集合

### 2.1 已有四个函数

- Rastrigin：强多峰、可分结构；
- Ackley：全局结构明显的多峰函数；
- Griewank：和项与乘积项耦合；
- Zakharov：平滑、非可分、尺度随维度增长。

### 2.2 新增四个函数

- Rosenbrock：弯曲窄谷，`x*=1`，`f*=0`，建议边界 `[-5,10]^d`；
- Levy：多峰，`x*=1`，`f*=0`，建议边界 `[-10,10]^d`；
- Schwefel 2.26：欺骗性多峰，`x*=420.968746...`，建议边界 `[-500,500]^d`；实现时将解析最优值平移为精确的 `known_optimum=0`；
- High-conditioned Ellipsoid：病态单峰，`x*=0`，`f*=0`，建议边界 `[-5,5]^d`，条件数固定为 `1e6`。

全部函数必须满足：

- 单次目标计算为 `O(d)`；
- 有可审计的已知最优点和最优值；
- 支持 shift、coordinate permutation、asymmetry 和 block rotation；
- `d=10000` 不分配 dense `d x d` 矩阵；block rotation 的 block size 固定且写入 configuration hash；
- 在最优点、边界点、随机点和 `d=10000` 上具有有限数值；
- Python/R 若都实现，必须通过固定 random tape/trace 的方向和数值一致性测试。

Michalewicz 不进入主集合，因为高维全局最优值没有可审计的解析形式，不适合当前 normalized-gap/target-hit 合同。

## 3. 算法集合

### 3.1 当前 E3 的七个算法

1. `PY-SP-SMCO-EVO`（冻结 winner）；
2. `PY-BASE-SMCO`（matched base）；
3. `GenSA`；
4. `SA`；
5. `DE`（当前冻结实现）；
6. `GA`；
7. `PSO`。

### 3.2 E7 新增的五个算法

8. `R-DEoptim`：与原 SMCO 论文中的 DEoptim 区分，不得冒充当前 `DE`；
9. `STOGO`：原论文 Group II 全局算法；
10. `L-BFGS`：可扩展的强局部基线，使用与 SMCO 相同的冻结 starts；
11. `SPSA`：低函数评估数的随机局部基线；
12. `SignGD`：与 SMCO sign-based 更新最相关的局部基线。

每个新增算法必须冻结 package/version、语言、默认参数的任何覆盖、边界处理、随机数生成、starts 语义和 FE 计数方法。内部一次 gradient/Jacobian 调用产生的所有 objective evaluations 都必须计入 FE。

GD、ADAM 和 BOBYQA 不进入 E7 主矩阵：前两者与 SignGD/SPSA 信息重复，BOBYQA 在 10000 维的内存和模型规模不可接受。可在补充材料说明其未纳入理由，不得把未运行描述成失败。

## 4. E3-F：缺失函数补充

### 4.1 矩阵

- functions：Rosenbrock、Levy、Schwefel、Ellipsoid；
- dimensions：`200,500,1000`；
- instances：每个函数和维度 5 个新的 confirmatory instances；
- algorithms：当前 E3 的 7 个算法；
- budget：`B_max = 1000*d FE`；
- checkpoints：`{0.1,0.25,0.5,0.75,1.0}*B_max`。

新增运行数：

```text
4 functions x 3 dimensions x 5 instances x 7 algorithms = 420 runs
```

完成后，原 E3 与 E3-F 的联合矩阵为：

```text
8 functions x 3 dimensions x 5 instances x 7 algorithms = 840 rows
```

E3-F 使用单独 frozen component manifest。联合分析通过 composite manifest 引用原 E3 与 E3-F，不改原 E3 的 run_id、stage、manifest 或 canonical index。

## 5. E7：超高维实验

### 5.1 逻辑分析矩阵

- functions：完整 8 函数；
- dimensions：`1000,2000,3000,5000,10000`；
- `d=1000`：每函数 5 个实例；
- `d={2000,3000,5000,10000}`：每函数 4 个实例；
- algorithms：完整 12 算法；
- budget：`B_max = 1000*d FE`。

逻辑分析总规模：

```text
d=1000:                    8 x 5 x 12 = 480 rows
d=2000/3000/5000/10000:   8 x 4 x 4 x 12 = 1536 rows
total:                                      2016 rows
```

### 5.2 结果复用与实际新增运行数

在 E3-F 完成后，`d=1000` 已有 8 函数 x 5 实例 x 7 算法 = 280 条可复用结果。E7 只补 5 个新增算法在这些 d=1000 cells 上的结果：

```text
d=1000 additional comparators: 8 x 5 x 5 = 200 runs
new ultra-high dimensions:     8 x 4 x 4 x 12 = 1536 runs
E7 physically new:                               1736 runs
```

因此 E7 实际新跑 **1736 runs**，与早期“约 1776 runs”的估计接近；E7 最终分析表为 2016 rows。禁止为了凑整数重复运行已有的 280 条结果。

两个 campaign 合计新执行：

```text
E3-F 420 + E7 1736 = 2156 runs
```

### 5.3 FE 与 anytime checkpoints

| dimension | FE budget | FE checkpoints |
|---:|---:|---|
| 1000 | 1,000,000 | 100k, 250k, 500k, 750k, 1m |
| 2000 | 2,000,000 | 200k, 500k, 1m, 1.5m, 2m |
| 3000 | 3,000,000 | 300k, 750k, 1.5m, 2.25m, 3m |
| 5000 | 5,000,000 | 500k, 1.25m, 2.5m, 3.75m, 5m |
| 10000 | 10,000,000 | 1m, 2.5m, 5m, 7.5m, 10m |

另外保存 wall-time checkpoints：1h、6h、24h、72h 和最终完成时。wall-time checkpoint 保存当时的 best-so-far、FE used、target-hit 和进程资源信息。

## 6. 72 小时语义与继续运行政策

72 小时是论文预注册的 **operational deadline**，不是进程 kill threshold。

每条 outcome 同时记录：

- `status`：最终执行状态，如 success/failure；
- `deadline_hours=72`；
- `deadline_exceeded`；
- `deadline_fe_used`；
- `deadline_best_value` 与 `deadline_normalized_gap`；
- `final_wall_time_sec`；
- `post_deadline_result=true/false`；
- heartbeat、checkpoint 和 attempt/supersedes provenance。

超过 72 小时的任务继续运行，直到达到冻结 FE budget 或算法正常终止。论文中进行两套分离分析：

1. **72h operational analysis**：超过 72h 视为 deadline timeout/censored，使用 72h 时保存的质量；
2. **eventual-completion analysis**：使用最终结果，但明确标记为 post-deadline，不得称为 72h 内完成。

这样可以写“GenSA 在 72h operational budget 下超时”，同时仍保留其十多天后完成的最终质量。不得把最终成功行的 `status` 伪造为 failure，也不得把 72h 后结果混入 72h 主分析。

任务不因超过 72h 被停止。只有以下基础设施异常可以重启：进程崩溃、节点失联、无 heartbeat 且 FE 在预设窗口内完全不增长。重启使用同一 run_id 的新 attempt，并通过 `supersedes_run_id`/attempt ledger 保留累计 wall time和失败原因。

## 7. 预注册假设

- **H5（维度收益）**：SMCO-EVO 相对 matched base 的 paired log-gap gain 随 `log(d)` 增长；
- **H6（超高维竞争力）**：在 `d>=5000` 时，SMCO-EVO 优于 DE、R-DEoptim、GA、PSO；
- **H7（72h 可用性）**：在 72h deadline 下，SMCO-EVO 的完成率和目标命中率高于 GenSA/STOGO；
- **H8（数值稳定性）**：SMCO-EVO 在 d=10000 不出现 failure、非有限值或 gap 爆炸；
- **H9（最终质量权衡）**：比较 SMCO-EVO 与 GenSA 的 72h 质量、最终质量和 time-to-target，允许结论为速度--质量权衡而非单一赢家。

H5 的主要统计单位是 `(function,dimension,instance)` paired cell；H6/H9 在 `d>=5000` 预先定义的 cells 上检验。所有 pairwise p-value 使用 Holm 校正，并同时报告 winner strict-win、tie 和 loss rate。

## 8. 统计分析

### 8.1 质量指标

- normalized gap；
- ECDF-AUC；
- target-hit rate at `1e-1,1e-2,1e-3,1e-5`；
- ERT/FE-to-target；
- failure 和 nonfinite rate；
- paired median log-gap difference 与 hierarchical bootstrap CI。

### 8.2 维度趋势

- 对 paired EVO-base gain 拟合 `gain ~ log(d)`；
- bootstrap 先按 function、再按 instance；
- 同时画每函数趋势，避免 pooled Simpson effect；
- d=1000 作为冻结锚点，d=2000--10000 为 prospective extension。

### 8.3 时间指标

- 72h deadline completion rate；
- 1h/6h/24h/72h fixed-time quality；
- eventual completion time；
- time-to-target survival/ECDF；
- 同一 problem bundle、同一节点内的 paired log runtime；
- CPU 型号或节点不同的绝对 wall time不直接混为算法排名。

所有最终表同时给出 deadline 和 eventual 两列，GenSA/STOGO 超过 72h 的记录保留在分母。

## 9. Fleet 分片与运行

### 9.1 分片单位

分片单位不是单个维度，而是：

```text
problem bundle = function x dimension x instance
```

同一 bundle 的全部算法尽量在同一节点运行，以控制硬件差异。bundle 按 pilot 估计的计算成本用 deterministic greedy bin packing 分到 fleet；每台机器必须包含多种函数、多个维度和全部算法，避免 node 与 dimension/algorithm 完全混杂。

### 9.2 节点预检

每节点冻结并保存：

- hostname、CPU model、物理核、内存；
- git SHA、dirty status；
- Python/R、NumPy/SciPy 和所有算法 package version；
- `pip freeze`/R sessionInfo hash；
- BLAS/OpenMP 后端；
- manifest、selection、instance index 和 shard hash；
- 本地结果根、开始时间和监控日志。

正式任务使用单线程 objective/BLAS（如 `OMP_NUM_THREADS=1`），按物理核和内存限制 workers，避免超售。每节点先运行不进入分析的 smoke bundle，确认 FE、known optimum、checkpoint、deadline monitor 和 provenance。

### 9.3 长任务运营

- 每 5--10 分钟原子写 heartbeat；
- 每个 FE/wall checkpoint 原子写 sidecar；
- coordinator 每小时汇总 planned/running/completed/deadline-exceeded/failed；
- 超过 72h 后继续运行并标红，不自动 kill；
- 节点维护或重启前保存 checkpoint；不支持算法内部恢复时，以新 attempt 从相同 seed 重跑并保留旧 attempt；
- 禁止删除慢任务、失败任务或只合并完成较快的算法。

## 10. Manifest、merge 与 canonical 合同

新增且互相隔离的 artifact：

1. E3-F component manifest（420 tasks）；
2. E3+E3-F comparative composite（840 rows）；
3. E7 new-task manifest（1736 tasks）；
4. E7 logical composite（2016 rows，引用 d=1000 的280条和 E7 新结果）；
5. deadline snapshot sidecar；
6. eventual outcome sidecar；
7. fleet dispatch ledger 和 attempt ledger；
8. E3-F/E7 各自的 canonical external/extension index。

merge/audit 至少检查：

- frozen manifest hash 与 selection hash；
- exact run-id coverage、physical row count、unique run-id；
- function/dimension/instance/algorithm grid；
- known optimum、instance/start/configuration hash；
- FE cap、objective direction、finite/gap sanity；
- deadline checkpoint 的存在性与不可后写；
- post-deadline 与最终结果的语义一致；
- machine/git/environment provenance；
- attempts/supersedes 可解析；
- 12-check 主 audit 加 deadline-specific checks 全通过。

任何一项失败时不得进入论文统计或 Task 13 package。

## 11. 实现任务顺序

1. **冻结 protocol**：本文件、H5--H9、函数/算法/预算/deadline schema；
2. **函数 TDD**：Levy、Schwefel、Ellipsoid；Rosenbrock 接入 highdim grid；
3. **10k instance TDD**：block rotation 内存、hash、known optimum、重放；
4. **新增算法 TDD**：R-DEoptim、STOGO、L-BFGS、SPSA、SignGD，统一 FE observer；
5. **deadline runner TDD**：72h snapshot、continue-after-deadline、heartbeat、attempt ledger；
6. **E3-F manifest**：生成、Gate-F、dry-run、冻结；
7. **E3-F fleet**：420 runs，merge/audit，建立 840-row composite；
8. **E7 manifest/composite**：1736 new tasks + 280 reused rows，冻结所有 hash；
9. **fleet preflight/pilot**：只测基础设施和吞吐，不用于算法选择；
10. **E7 正式派跑**：按 bundle 分片，长任务持续到最终 FE/正常终止；
11. **回传与 merge**：deadline/final 双结果、2016-row composite、audit；
12. **统计和图表**：H5--H9、dimension scaling、72h/final 双报告；
13. **Task 13 package**：更新综合证据报告并保留所有负结果和超时。

## 12. 验收标准

### E3-F

- 420/420 planned run-id 均有 outcome；
- 联合 E3 composite 为 840 unique rows；
- 8 函数 x 3 维度 x 5 实例 x 7 算法精确覆盖；
- audit 全通过。

### E7

- 1736 个新 run-id 全部达到最终终止状态；
- 超过 72h 的任务仍有 deadline snapshot 和 eventual outcome；
- logical composite 精确为 2016 unique rows；
- d=1000 的 280 条复用结果 hash 与来源 canonical 完全一致；
- 8 函数、5 维度、12 算法的预注册网格完整；
- 不因慢、失败或结果不利而删除 GenSA/STOGO/任何算法；
- deadline 与 eventual 两套统计均可从冻结 artifact 独立重建；
- full pytest、真实 smoke、merge audit、canonical validation 全通过。

## 13. 允许的论文表述

若结果支持，可以表述：

> 在统一 FE 合同和 72 小时 operational deadline 下，SMCO-EVO 在 5000--10000 维保持较高完成率和稳定质量；部分强全局方法虽然最终质量可能更高，但需要显著更长的计算时间。

不得只凭运行超过 72h 就声称 SMCO-EVO 的最终质量优于 GenSA，也不得把 72h 后完成的 GenSA 写成永久失败。最终结论必须同时展示 72h 截面和 eventual completion。
