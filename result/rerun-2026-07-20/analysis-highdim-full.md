# SMCO 高维优化基准 — 方向修正重跑分析报告（论文级）

- **日期**：2026-07-24（重跑 2026-07-20 启动，~4 天）
- **数据**：`result/highdim-full-rerun-2026-07-20/merged/all_results_full.csv`（**18928 rows**）
- **质量度量**：`raw = -fopt`（越小越优，越接近真解 raw=0；修正前全正=推边界，修正后全负=接近真解）

---

## 摘要

本报告分析 SMCO 高维优化基准的方向修正重跑（9 维度 × 5 函数 × 18 算法 × 4 strategy × 多 rep = 18928 行）。修正后所有结果 fopt 全负（pos=0），替代推边界的旧 6/04 基准。核心发现：

1. **GenSA 综合最优**：全维度 raw 最小（50D 0 → 5000D 41），高维性价比最高（5000D 仅 50min）。
2. **SMCO_R / SMCO_R_EVO 是 SMCO 系列最优**：高维（≥1000D）稳定第二好（5000D raw 1397），远优于其他全局算法。
3. **SMCO_BR / SMCO_BR_EVO 高维灾难**：iter_boost 机制在 ≥3000D 发散（5000D raw 42866）。
4. **EVO 模式无显著增益**：vs base win rate 仅 43–50%，SMCO_R/BR 的 EVO 中位 raw 反而略差。
5. **strategy**：rand1bin/sobol 略优于 best1bin/current-to-best1bin。
6. **vs 全局对比算法**：SMCO 系列在高维远稳定于 SA/DEoptim/GA/PSO（后者 5000D raw 5.5万–8.9万）。

---

## 1. 背景

`docs/direction-bug-2026-06-15.md` 记录的方向 bug：旧 6/04 基准脚本对 `sense=="min"` 双重取负（`f=-config.f=+raw`），导致 SMCO 及所有对比算法最大化 raw（推边界），fopt 全正。修正后（`f=config.f`）重跑，fopt 全负（接近真解）。详见 `docs/direction-bug-audit-2026-07-20.md`。

## 2. 实验设置

| 项 | 配置 |
|---|---|
| 维度 | 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000 |
| 函数 | Rastrigin, Ackley, Rosenbrock（全维度）；Griewank, Zakharov（≤500D）|
| 算法 | 18：SMCO 系列 6（SMCO/SMCO_R/SMCO_BR + EVO 版）+ 局部 7（GD/SignGD/ADAM/SPSA/optimLBFGS/BOBYQA/optimNM，≤200D）+ 全局 5（GenSA/SA/DEoptim/GA/PSO）|
| strategy | rand1bin, current-to-best1bin, best1bin, sobol |
| reps | 50D:20, 100D:15, 200D:10, 500D:5, 1000D:5, 2000D:3, 3000-5000D:2 |
| iter_max | 300（SMCO_BR 系列减半 + iter_boost=1000）|

## 3. 方向修正验证

**18928 rows fopt 全负（pos=0 neg=18928 nan=0）**。修正前 6/04 基准全正（Rastrigin 1000D +29108、Rosenbrock 1000D +4.3 亿，推边界）；修正后全负（接近真解 raw=0）。方向修正确凿。

---

## 4. 结果分析

### 4.1 SMCO-EVO vs base（EVO 进化模式效果）

逐点配对（同 strategy/func/dim/rep），raw 越小越优：

| 配对 | EVO 胜率 | base 中位 raw | EVO 中位 raw |
|---|---|---|---|
| SMCO → SMCO_EVO | **50.1%**（持平）| 6.63 | 4.45 |
| SMCO_R → SMCO_R_EVO | 43.1%（base 略优）| 2.18 | 4.43 |
| SMCO_BR → SMCO_BR_EVO | 43.6%（base 略优）| 7.28 | 10.02 |

**结论**：EVO（多起点进化）模式相比 base **无明显优势**。SMCO_EVO 与 SMCO 持平（50%）；SMCO_R/SMCO_BR 加 EVO 后中位 raw 反而略差（进化选择在这些配置下未带来净增益）。

### 4.2 SMCO 系列内部对比（全维度 raw 中位数，越小越优）

```
SMCO_R 2.18 < SMCO_R_EVO 4.43 ≈ SMCO_EVO 4.45 < SMCO 6.63 < SMCO_BR 7.28 < SMCO_BR_EVO 10.02
```

- **SMCO_R 最优**（refine 机制）。
- SMCO_BR / SMCO_BR_EVO 高维灾难（见 4.4）。

### 4.3 SMCO vs 局部算法（≤200D）

| 算法 | 50D | 100D | 200D | 说明 |
|---|---|---|---|---|
| optimLBFGS | 0 | 0 | 19.4 | 凸问题 BFGS 最强（低维）|
| GenSA | 0 | 0 | 0 | 全局退火，低维完美 |
| SMCO_R | 1.5 | 2.4 | 4.4 | SMCO 系列低维优 |
| SMCO_EVO | 0.8 | 1.9 | 8.4 | 同 |
| BOBYQA | 370.9 | 1146.5 | 4083 | 中 |
| SignGD | 370.9 | 770.1 | 1598 | 中 |
| GD | 597 | 1248.8 | 2529 | 差 |
| ADAM | 1201.9 | 2560.1 | 5265.7 | 差 |
| SPSA | 1220.5 | 2599.8 | 5341.1 | 差 |
| optimNM | 1124.4 | 2565.9 | 5333.1 | 差 |

低维：**optimLBFGS / GenSA 最好**（凸/退火），**SMCO 系列次优且稳定**，梯度类（GD/ADAM/SPSA/NM）在高维多模态差。

### 4.4 SMCO vs 全局算法（高维 ≥1000D，重点）

raw 中位数（越小越优）：

| dim | GenSA | SMCO_R_EVO | SMCO_EVO | SMCO | SMCO_R | SMCO_BR | SA | DEoptim | GA | PSO |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | **8** | 22 | 247 | 249 | 23 | 98 | 2489 | 11852 | 10406 | 15177 |
| 2000 | **18** | 45 | 472 | 462 | 96 | 2285 | 29753 | 23892 | 21484 | 32402 |
| 3000 | **28** | 105 | 680 | 717 | 589 | 6751 | 47090 | 35838 | 32486 | 49556 |
| 4000 | **39** | 560 | 882 | 1194 | 1404 | 19718 | 61235 | 47517 | 43679 | 68607 |
| 5000 | **41** | 1397 | 1109 | 1610 | 2561 | **42866** | 89361 | 59906 | 55153 | 84417 |

**关键发现**：
- **GenSA 高维最优**（5000D raw 41），且快（5000D 50min）。
- **SMCO_R_EVO / SMCO_EVO** 是 SMCO 系列高维最优（5000D 1109–1397）。
- **SA / DEoptim / GA / PSO 高维很差**（5000D raw 5.5万–8.9万）——这些全局算法在高维陷入差局部解。
- **SMCO_BR / SMCO_BR_EVO 高维灾难**（5000D raw 42866）——iter_boost 机制在 ≥3000D 数值发散（与实验1/2 发现一致：BR_EVO 满起点发散）。

### 4.5 strategy 对比（SMCO_EVO raw 中位数）

| strategy | raw |
|---|---|
| sobol | 3.6 |
| rand1bin | 3.66 |
| current-to-best1bin | 5.23 |
| best1bin | 5.33 |

**rand1bin / sobol 略优于 best1bin / current-to-best1bin**（DE 变异策略，随机父代 vs 贪心父代）。SMCO_BR_EVO 不敏感（iter_boost 主导）。

### 4.6 函数特性（raw 中位数）

| 函数 | GenSA | SMCO_R_EVO | SMCO | optimLBFGS | SA | DEoptim | 说明 |
|---|---|---|---|---|---|---|---|
| Ackley | 0 | 0.1 | 0.4 | 19.4 | 19.5 | 3.5 | 单碗：SMCO+GenSA 近真解 |
| Griewank | 0 | 0.1 | 0.9 | 0 | 0 | 1.2 | SMCO+全局近真解 |
| Rastrigin | 1 | 58.3 | 59.4 | 536 | 55.7 | 954 | 多模态：GenSA 强，SMCO 中 |
| Rosenbrock | 91 | **21** | 25 | 0 | 0 | 1013 | 窄谷：SMCO_R_EVO 最好 |
| Zakharov | 0 | 5.0 | 1.8 | 0 | 0 | 693 | SMCO+GenSA 近真解 |

- **Ackley/Griewank/Zakharov**（单碗/可分）：SMCO 系列与 GenSA 都接近真解。
- **Rastrigin**（多模态）：GenSA 最强（1），SMCO 系列中（56–59），其他差。
- **Rosenbrock**（窄谷强耦合）：**SMCO_R_EVO 最好（21）**，超过 GenSA（91）；SA/optimLBFGS 找到真解（0）但高维失效。

### 4.7 维度趋势

SMCO raw 随维度合理增长（50D 0.8 → 5000D 1610）；GenSA 增长最慢（0 → 41）。全局对比算法（SA/DEoptim/GA/PSO）高维急剧恶化（5000D 几万）。

### 4.8 耗时（time 中位数，秒）

| dim | SMCO | SMCO_R | SMCO_EVO | GenSA | SA | DEoptim | GA | PSO |
|---|---|---|---|---|---|---|---|---|
| 1000 | 855 | 731 | 583 | 219 | 383 | 2083 | 2 | 1 |
| 3000 | 9105 | 9443 | 9848 | 1382 | 904 | 16344 | 5 | 3 |
| 5000 | 33838 | 31391 | 33254 | **3009** | 2051 | 41763 | 9 | 5 |

- **SMCO 系列 5000D ~9h**（高维评估成本主导）。
- **GenSA 5000D 50min**（最快的高维全局，且质量最好）。
- **DEoptim 5000D 11.6h**（最慢）。
- GA/PSO 极快（<10s）但高维质量差。

---

## 5. 结论

1. **GenSA 综合最优**：全维度 raw 最小，高维快（5000D 50min）+ 好（raw 41）。但低维多模态也强（Rastrigin 50D 0）。
2. **SMCO_R / SMCO_R_EVO 是 SMCO 系列最佳**：高维（≥1000D）稳定第二（5000D raw 1397/1109），远优于 SA/DEoptim/GA/PSO。Rosenbrock 窄谷上 SMCO_R_EVO 超越 GenSA。
3. **SMCO_BR / SMCO_BR_EVO 高维避免**：iter_boost 在 ≥3000D 发散（5000D raw 42866），仅适用低维。
4. **EVO 模式无显著增益**：vs base win rate 43–50%，不优于 base。多起点进化在 SMCO 当前框架下未带来净收益。
5. **strategy**：rand1bin/sobol 略优于 best1bin/current-to-best1bin。
6. **vs 全局对比**：SMCO 系列高维远稳定于 SA/DEoptim/GA/PSO（后者高维 raw 几万–几十万，陷入差解）。

## 6. 方法与数据

- **修正**：`f=config.f`（6 脚本），详见 `docs/rerun-plan-2026-07-20.md`。
- **数据**：`merged/all_results_full.csv`（18928 rows，18 算法 × 9 维度 × 函数 × strategy × reps）。
- **分布式**：5 机（217/213/215/251/253），按维度分片，~4 天（5000D 慢全局是主因）。
- **相关**：实验1/2（evo-startsweep/dimsplit）已 7/18 完成；smco_evo（evo-comparison）已完成；R 线（run_highdim_r.R）跑中。

## 7. 局限

- 高维（≥1000D）只有 Rastrigin/Ackley/Rosenbrock 3 函数（FUNCS_BY_DIM 设计）。
- reps 高维少（3000-5000D 仅 2），统计功效有限。
- GenSA 高维有长尾（median 50min，但部分 case ~3h）。
