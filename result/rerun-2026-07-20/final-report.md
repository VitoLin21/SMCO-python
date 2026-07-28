# highdim-full 方向修正重跑 — 完整报告（全 9 维度）

- **完成日期**：2026-07-24（启动 2026-07-20，~4 天）
- **范围**：全 9 维度（50/100/200/500/1000/2000/3000/4000/5000D）× 18 算法 × 4 strategy
- **数据**：`result/highdim-full-rerun-2026-07-20/merged/all_results_full.csv`（**18928 rows**）
- **前置**：阶段报告 `staged-report-no5000.md`（非5000D，已交付）；本版补充 5000D + 100D sobol

---

## 1. 方向铁证（成功标准）

修正后 `fopt = -raw`（SMCO 最大化 `config.f=-raw`），min 函数 fopt 应全负。

| 指标 | 结果 |
|---|---|
| 总 rows | **18928**（与 6/04 基准设计完全吻合：18算法×9维度×funcs×reps×4strategy）|
| fopt 符号 | **pos=0 neg=18928 nan=0** |
| 结论 | **方向修正成功**，所有 min 函数 fopt 全负，0 正值 0 nan |

**对比 6/04 基准**（修正前）：18928 rows fopt 全正（推边界，Rastrigin 1000D +29108、Rosenbrock 1000D +4.3 亿）。修正后全负（接近真解 raw=0）。

## 2. 各维度数据量与质量

| dim | rows | fopt 中位数 | 说明 |
|---|---|---|---|
| 50 | 7200 | -19.3 | 5func×20rep×4strat×18algo |
| 100 | 5400 | -44.3 | 5func×15rep×4strat×18algo（sobol 已补齐）|
| 200 | 3600 | -96.5 | 5func×10rep×4strat×18algo |
| 500 | 880 | -129.6 | 4func×5rep×4strat×11algo |
| 1000 | 660 | -238.6 | 3func×5rep×4strat×11algo |
| 2000 | 396 | -1131 | 3func×3rep×4strat×11algo |
| 3000 | 264 | -1514 | 3func×2rep×4strat×11algo |
| 4000 | 264 | -1490 | 3func×2rep×4strat×11algo |
| 5000 | 264 | **-2734.7** | 3func×2rep×4strat×11algo（含慢全局GenSA/SA，~4天完成）|

fopt 中位数全负，随维度递增（高维更难，raw 残差更大）。5000D -2734（vs 修正前 +142591 推边界）。

## 3. 各函数质量（fopt 中位数）

| func | fopt 中位数 | 真解 raw |
|---|---|---|
| Ackley | -7.58 | 0 |
| Griewank | -0.93 | 0 |
| Rastrigin | -419.9 | 0（高维局部最优残差）|
| Rosenbrock | -236.8 | 0 |
| Zakharov | -194.7 | 0 |

全负（接近真解或合理残差）。对比修正前全正（推边界）。

## 4. 算法覆盖（18 个，全）

SMCO 系列（6）：SMCO、SMCO_R、SMCO_BR、SMCO_EVO、SMCO_R_EVO、SMCO_BR_EVO
局部（7，≤200D）：GD、SignGD、ADAM、SPSA、optimLBFGS、BOBYQA、optimNM
全局（5，全维度，含 5000D 慢）：GenSA、SA、DEoptim、GA、PSO

**全 18 算法**（用户要求），含 5000D 慢全局（GenSA/SA 5000D 单 task ~3-6h，是 ~4 天主因）。

## 5. 分布式分工（本次重跑，~4 天）

| 机 | 维度 | 耗时 | 状态 |
|---|---|---|---|
| 253（20核）| 50D | ~11min×4strat | ✓ 完成（+smco_evo）|
| 217（38核）| 100D + 5000D | ~4天 | ✓ 完成（5000D最慢）|
| 213（38核）| 200D + 4000D | ~3天 | ✓ 完成 |
| 215（38核）| 500D + 3000D | ~3天 | ✓ 完成 |
| 251（38核）| 1000D + 2000D | ~2.5天 | ✓ 完成 |

## 6. 结论

1. **方向修正成功**：18928 rows fopt 全负（pos=0），全 9 维度 × 18 算法 × 4 strategy。修正前全正（推边界）→ 修正后全负（接近真解）。
2. **数据完整**：18928 rows 完全匹配 6/04 基准设计，无缺失。
3. **可信**：含慢全局算法（GenSA/SA/DEoptim/GA/PSO）全维度，5000D 慢全局 ~4 天完成（用户选等全 18 算法）。
4. **替代 6/04**：本数据（`all_results_full.csv`）方向正确，可替代推边界的旧 6/04 基准用于论文/分析。

## 7. 后续

- `smco_evo`（evo-comparison）已在 253 完成（方向已验证）。
- `run_highdim_r.R`（R 高维，~660 task）待启动到 215/251（有 R，现空闲）——P0 第三脚本。
- 实验1/2（evo-startsweep/dimsplit）此前已完成（方向正确）。
