# highdim-full 方向修正重跑 — 阶段报告（非5000D）

- **日期**：2026-07-23（启动 2026-07-20，第 3 天）
- **触发**：213（4000D+200D）完成空闲 → 分阶段交付（用户 2026-07-22 指示）
- **范围**：非5000D 的 8 个维度（50/100/200/500/1000/2000/3000/4000D），缺 5000D（217 跑中）+ 100D sobol strategy（217 batch7 待跑）
- **数据**：`result/highdim-full-rerun-2026-07-20/merged/all_results_no5000.csv`（17314 rows）

---

## 1. 方向铁证（成功标准）

修正后 `fopt = -raw`（SMCO 最大化 `config.f=-raw`），min 函数 fopt 应全负（raw>0）。

| 指标 | 结果 |
|---|---|
| 非5000D rows | 17314（去重后，无重复） |
| fopt 符号 | **pos=0 neg=17314 nan=0** |
| 结论 | **方向修正成功**，所有 min 函数 fopt 全负，0 正值 |

对比修正前（6/04 基准）：所有 fopt 全正（推边界），Rastrigin 1000D +29108、Rosenbrock 1000D +4.3 亿。修正后全负（接近真解 raw=0）。

## 2. 各维度数据量与质量

| dim | rows | fopt 中位数 | 含义 |
|---|---|---|---|
| 50 | 7200 | -19.3 | 全4strategy×5func×20rep×18algo |
| 100 | 4050 | -44.3 | **缺 sobol strategy**（3/4 strategy）|
| 200 | 3600 | -96.5 | 全4strategy |
| 500 | 880 | -129.6 | 全4strategy |
| 1000 | 660 | -238.6 | 全4strategy |
| 2000 | 396 | -1131 | 全4strategy |
| 3000 | 264 | -1514 | 全4strategy |
| 4000 | 264 | -1490 | 全4strategy |

fopt 中位数全负，随维度增大绝对值增大（高维更难，raw 残差更大，合理）。

## 3. 各函数质量（fopt 中位数）

| func | fopt 中位数 | 真解 raw |
|---|---|---|
| Ackley | -7.04 | 0 |
| Griewank | -0.93 | 0 |
| Rastrigin | -402.9 | 0（高维局部最优残差）|
| Rosenbrock | -211.6 | 0 |
| Zakharov | -194.7 | 0 |

全负（接近真解或合理局部最优残差）。

## 4. 算法覆盖（18 个）

SMCO 系列（6）：SMCO、SMCO_R、SMCO_BR、SMCO_EVO、SMCO_R_EVO、SMCO_BR_EVO
局部（7，≤200D）：GD、SignGD、ADAM、SPSA、optimLBFGS、BOBYQA、optimNM
全局（5，全维度）：GenSA、SA、DEoptim、GA、PSO

**全 18 算法**（用户要求），含慢全局算法。

## 5. 缺口与后续

- **5000D 全缺**：217 还在跑（batch6 best1bin 36-39/66），5000D 还需 batch6/8，预期 1-2 天。完成后 rsync 补充合并，更新为完整报告。
- **100D sobol 缺**：217 batch7（100D sobol）在 batch6（5000D best1bin）之后，5000D best1bin 完成后几小时内补齐。届时 100D 达 5400 rows（4 strategy 全）。

## 6. 分布式分工（本次重跑）

| 机 | 维度 | 状态 |
|---|---|---|
| 253（20核）| 50D | ✓ 完成 |
| 217（38核）| 100D + 5000D | 100D 缺sobol, 5000D 进行中 |
| 213（38核）| 200D + 4000D | ✓ 完成 |
| 215（38核）| 500D + 3000D | ✓ 完成 |
| 251（38核）| 1000D + 2000D | ✓ 完成 |

## 7. 结论

- **方向修正成功**：17314 rows fopt 全负（pos=0），8 维度 × 18 算法 × 多 strategy，修正前全正（推边界）→ 修正后全负（接近真解）。
- **数据可信**：非5000D 主体完整（仅 100D sobol + 5000D 待补），可先用于分析。
- **后续**：5000D 完成（+100D sobol）后合并完整版，更新本报告。
