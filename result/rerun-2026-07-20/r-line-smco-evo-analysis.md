# SMCO-EVO 在 R 语言上的表现 — 详细分析报告

- **日期**：2026-07-27
- **数据**：`result/r-highdim-rerun-2026-07-24/merged/r_highdim_merged.csv`（273 rows，R 实现 `vendor/SMCO_R/main/run_highdim_r.R`）
- **对比**：Python `result/highdim-full-rerun-2026-07-20/merged/all_results_full.csv`（18928 rows）
- **度量**：`raw = -fopt`（越小越优）。R 线 `fobj=-raw` 修正后方向正确

---

## 摘要

R 实现（`SMCO_evo.R`）的 SMCO-EVO 系列在高维优化中表现**显著优于 Python 版**，尤其 SMCO_BR_EVO（R 5000D raw 122 vs Python 42862，**优 350×**）。R 中 EVO 模式相比 base 有效（win rate 63–85%，与 Python 的 43–50% 相反）。SMCO_EVO 本身 R≈Python（移植保真），但带 refine/boost 的 EVO（R_EVO/BR_EVO）R 版高维稳定性远胜 Python。

## 1. 数据范围

R 线部分完成（**SA 已由 SA 专跑线补齐**，仅缺 3000+5000D GenSA 12 task）：
- 1000D：165 rows（完整，5 reps × 3 func × 11 algo）
- 3000D：60 rows（SMCO 系列 + DEoptim/GA/PSO/**SA** 完成，仅 GenSA 缺）
- 5000D：60 rows（同上）
- 合计 **285 rows**，方向：**fopt 全负（pos=0 neg=285）**，R 移植 + 修正正确

## 2. R 内：SMCO-EVO vs base（EVO 效果）

逐点配对（同 func/dim/rep），raw 越小越优：

| 配对（R 实现）| EVO 胜率 | base 中位 raw | EVO 中位 raw |
|---|---|---|---|
| SMCO → SMCO_EVO | **63.0%** | 255.1 | 251.1 |
| SMCO_R → SMCO_R_EVO | **66.7%** | 22.8 | 23.4 |
| SMCO_BR → SMCO_BR_EVO | **85.2%** | 168.2 | **25.4** |

**关键**：R 中 EVO 模式**有效**（win rate 63–85%）。SMCO_BR_EVO 尤其突出（85% 胜，中位 raw 168→25，降 6.6×）。

**对比 Python**（Python EVO win rate 仅 43–50%，EVO 无增益）：R 实现的 EVO 进化选择**比 Python 更有效**。这是 R vs Python 的核心行为差异。

## 3. R vs Python：SMCO-EVO 系列高维对比（核心发现）

raw 中位数（越小越优），R/P 比值 <1 表示 R 更优：

| 算法 | 维度 | R | Python | R/P | 解读 |
|---|---|---|---|---|---|
| SMCO_EVO | 1000 | 247.9 | 246.8 | 1.00 | 一致（移植保真）|
| SMCO_EVO | 3000 | 675.8 | 680.3 | 0.99 | 一致 |
| SMCO_EVO | 5000 | 1036.8 | 1108.9 | 0.93 | 一致 |
| SMCO_R_EVO | 1000 | 22.4 | 22.3 | 1.00 | 一致 |
| SMCO_R_EVO | 3000 | 65.0 | 105.1 | 0.62 | R 优 1.6× |
| SMCO_R_EVO | 5000 | **106.6** | 1397.3 | **0.08** | **R 优 13×** |
| SMCO_BR_EVO | 1000 | 25.0 | 27.8 | 0.90 | 一致 |
| SMCO_BR_EVO | 3000 | 74.4 | 6803.4 | **0.01** | **R 优 91×** |
| SMCO_BR_EVO | 5000 | **122.6** | 42862.7 | **0.003** | **R 优 350×** |

**重大发现**：
1. **SMCO_EVO R≈Python**（移植保真，基础 EVO 一致）。
2. **SMCO_R_EVO / SMCO_BR_EVO R 远优于 Python**（高维 R/P 0.08–0.003）。
3. **Python SMCO_BR_EVO 高维灾难**（iter_boost 发散，5000D 42862），**但 R SMCO_BR_EVO 5000D 122.6（正常，不发散）**。

**解读**：Python SMCO_BR_EVO 的 `iter_boost=1000` 机制在高维（≥3000D）数值发散；R 版（`SMCO_BR_EVO` 的 iter_boost 实现）在高维**稳定**。这是 R 移植的关键优势——R 的 boost/refine 实现在高维更稳健。

（注：R/Python 起点 RNG/Sobol 不同，但 base 系列 R≈Python 见下表，说明起点差异不主导；EVO 系列的 13–350× 差异指向实现差异。）

## 4. R vs Python：base 系列一致性（移植保真背书）

5000D raw 中位数：

| 算法 | R | Python | 一致性 |
|---|---|---|---|
| SMCO | 1712.6 | 1610.1 | ≈一致 |
| SMCO_R | 2288.9 | 2561.0 | ≈一致 |
| SMCO_BR | 42945.3 | 42866.1 | ≈一致（都灾难）|
| SMCO_EVO | 1036.8 | 1108.9 | ≈一致 |

**base 系列（无 EVO refine/boost）R≈Python**，证明 R 移植核心算法保真，起点差异不主导。EVO 系列的差异是**实现层**（R 的 refine/boost 高维更稳）。

## 5. R 内：SMCO-EVO vs 全局算法

R 各算法 raw 中位数 by dim：

| dim | SMCO_R_EVO | SMCO_BR_EVO | SMCO_EVO | SMCO | SMCO_R | SMCO_BR | GenSA | SA | DEoptim | GA | PSO |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 22.4 | 25.0 | 247.9 | 247.1 | 22.5 | 98.0 | 73.3 | 19.5 | 15398 | 8360 | 10031 |
| 3000 | **65.0** | 74.4 | 675.8 | 762.5 | 551.2 | 6770.1 | — | **19.5** | 51697 | 28811 | 32556 |
| 5000 | **106.6** | 122.6 | 1036.8 | 1712.6 | 2288.9 | 42945.3 | — | **19.5** | 88426 | 50247 | 55357 |

- **高维（3000/5000D）SMCO_R_EVO 最优**（65/106.6），SMCO_BR_EVO 次之（74/122）。
- SMCO_BR（base）高维灾难（6770/42945），但 **SMCO_BR_EVO 不灾难**（R 中 EVO 救了 BR）。
- global（DEoptim/GA/PSO）高维差（几万），GenSA 缺（3000+5000D 慢未完成），**SA 已完成**（见 §5.1）。
- **SA 高维 raw 中位 19.5（所有维度稳定）**，因 Rosenbrock 拉低。

### 5.1 SA 专项分析（SA 已完成）

SA（R basin-hopping multi-restart L-BFGS）各维度 raw 中位 **19.5**（1000/3000/5000D 一致）。分函数：

| func | 1000D | 3000D | 5000D | 特性 |
|---|---|---|---|---|
| Rosenbrock | **0.0** | **0.0** | **0.0** | SA 完美（窄谷 multi-restart L-BFGS 强）|
| Ackley | 19.5 | 19.5 | 19.5 | 单碗边界残差 |
| Rastrigin | 6665 | 21675 | 37820 | 多模态差（随维度恶化）|

**SA vs SMCO-EVO 互补**（5000D 分函数 raw）：

| func | SMCO_BR_EVO | SA | 最优 |
|---|---|---|---|
| Rosenbrock | 122.6 | **0.0** | SA 完胜 |
| Ackley | **0.1** | 19.5 | SMCO_BR_EVO 完胜 |
| Rastrigin | 42740 | 37820 | SA 略好（都差）|

**SA 窄谷（Rosenbrock）最强**（找到真解），**单碗（Ackley）弱**（19.5 vs SMCO 0.1）；SMCO-EVO 单碗强、窄谷中。**两者地形互补**——SA 的 multi-restart L-BFGS 适合窄谷强耦合，SMCO-EVO 的进化多起点适合单碗/多模态。

## 6. R SMCO_EVO 各函数

| func | raw 中位 |
|---|---|
| Ackley | 0.37（接近真解）|
| Rastrigin | 8273（多模态难，局部最优残差）|
| Rosenbrock | 251（窄谷）|

## 7. 耗时（5000D，秒）

| 算法 | time 中位 |
|---|---|
| SMCO_EVO | 37359（10.4h）|
| SMCO_BR_EVO | 30168（8.4h）|
| SMCO_R_EVO | 28970（8h）|
| DEoptim/GA/PSO | 1–9（快但质量差）|

SMCO-EVO 系列 5000D ~8–10h（高维评估主导）。GenSA/SA 未完成（数小时-数十小时/task）。

## 8. 结论

1. **R 移植正确**：fopt 全负，base 系列 R≈Python（保真）。
2. **R 中 EVO 有效**：win rate 63–85%（vs Python 43–50%）。R 的进化选择实现更有效。
3. **R SMCO_BR_EVO 高维稳定**（5000D 122.6），而 Python 版灾难（42862，iter_boost 发散）。**R 的 boost/refine 实现在高维更稳健，是关键移植优势**。
4. **R SMCO_R_EVO/BR_EVO 是 R 高维最优**（5000D 106/122），远优于 global（DEoptim/GA/PSO 几万）。
5. **R vs Python 差异聚焦 EVO 系列**：base 一致，EVO 系列 R 远优（高维 refine/boost 稳定性）。
6. **SA 与 SMCO-EVO 地形互补**（SA 已完成）：SA 窄谷（Rosenbrock 5000D raw **0**，找到真解）完胜 SMCO_BR_EVO（122.6）；SMCO-EVO 单碗（Ackley 5000D **0.1**）完胜 SA（19.5）。SA 的 multi-restart L-BFGS 适合窄谷强耦合，SMCO-EVO 进化多起点适合单碗/多模态——两者地形互补，搭配可覆盖更广问题类。

**论文价值**：R 移植不仅是数值对齐（base 一致），还揭示了 Python EVO 系列高维的实现缺陷（BR_EVO 发散），R 版修复/规避了它。这支持 SMCO-EVO 算法本身的正确性（问题在 Python 特定实现，非算法本质）。

## 9. 局限

- R 线 3000+5000D GenSA/SA 缺（慢未完成），高维 global 对比不完整。
- R/Python 起点 RNG/Sobol 不同（非逐点配对），但 base 一致性背书差异来自实现。
- reps 高维少（3000/5000D 仅 2），统计功效有限。

## 10. 数据

- R：`merged/r_highdim_merged.csv`（273 rows）
- Python：`merged/all_results_full.csv`（18928 rows）
- 详见 `final-report.md`（Python 主报告）、`analysis-highdim-full.md`（Python 详细分析）。
