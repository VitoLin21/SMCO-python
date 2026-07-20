# 实验 3 执行记录与运维档案

> 实验 3（evo 迁移到 R + 高维碾压）的完整执行历史、单 task 耗时档案、运维教训。
> 生成于 2026-07-17（251 完整收官后）。配套分析见 `analysis.md`，整合结论见 `../summary-260615.md`。

---

## 一、部署架构

| 服务器 | 角色 | 任务范围 | 状态 |
|---|---|---|---|
| **10.25.40.251** | 实验3 主数据机 | 3000/5000D × 9 核心算法（SMCO/SMCO_R/SMCO_BR/SMCO_EVO + DEoptim/GA/PSO/GenSA/SA）× {Rastrigin,Ackley,Rosenbrock} × 2 rep | ✅ **12/12 收官**（07-15 21:05） |
| **10.16.144.215** | 复现 + 1000D + evo 扩展 | 1000D 全 11 算法 + 3000/5000D 的 R_EVO/BR_EVO + 核心9复现验证 | 🟡 仅差 5000D Rosenbrock GenSA/SA（~07/23） |
| 10.16.144.217 / 213 / 216 | 实验 1/2 专用 | （实验3 不涉及） | ✅ 已收官 |

- **脚本**：`vendor/SMCO_R/main/run_highdim_r.R`（mclappy 并行，含 filelock 实时 CSV flush 修复）+ `SMCO_evo.R`（333 行移植）。
- **方向**：与 Python 基线同向（最大化 raw，fopt↑ = 推边界更强，见 `analysis.md` §三）。
- **evo 参数**：`iter_max=300, elimination_rate=0.5, evolution_points=(0.5,0.75), de_factor=0.8, de_crossover=0.7`（R 与 Python 完全一致）。

---

## 二、执行时间线

| 日期 | 事件 |
|---|---|
| **06-15/16** | 实验3 部署启动：251（核心9）、215（1000D + R_EVO/BR_EVO + 复现） |
| 06-16 ~ 06-22 | 3000D GenSA 陆续完成（第 1–8 个）；DEoptim/GA/PSO/SA/SMCO 系秒级~小时级早早完成 |
| **06-20 23:09** | 首破 76h log 静默（5000D Ackley rep1 GenSA 落盘） |
| **06-22 23:21** | 第 8 个 GenSA（3000D Rastrigin rep0）落盘 → **3000D GenSA 全收官**；5000D 长尾（Rastrigin×2 + Rosenbrock×2）4 worker 启动 |
| 06-25 | 215 复现：5000D Ackley + 3000D Rastrigin 落盘（与 251 逐字节一致） |
| 06-26 ~ 07-13 | 251 4 worker 满载跑 5000D 长尾，log 长期静默（单 task 数百小时）；多次探测确认 `r=1.00` 满载非死循环 |
| **07-08** | 探测修正：5000D 仍只 Ackley×2 落盘，发现外推系数低估（曾用 2.69，实测 3.12–4.98），预测推迟 |
| **07-14** | 5000D 长尾大量落盘：251 Rastrigin rep0/1 + Rosenbrock rep1 → **11/12**；215 Rastrigin rep0/1 → **10/12** |
| **07-15 21:05** | 251 Rosenbrock rep0（694h）落盘 → **12/12**，log 打印 `DONE`，进程退出，load 归零 |
| **07-17** | 本档案 + analysis.md（最终版）+ summary-260615.md 更新完成 |

---

## 三、单 task 耗时档案（GenSA，全局瓶颈）

GenSA 是唯一耗时数百小时的算法。5000D 实测（251，单位小时）：

| 函数 | rep0 | rep1 | 备注 |
|---|---|---|---|
| Ackley | 90.6 | 85.7 | 最快（单深碗） |
| Rastrigin | 543.8 | 557.4 | 密集多极值 |
| **Rosenbrock** | **694.0** ⬅️ 全实验最慢 | 657.8 | 强耦合，rep0 = 28.9 天 |

3000D GenSA（对比）：Ackley 24.6–31.8h / Rastrigin 156.2–157.1h / Rosenbrock 131.5–140h。

### 维度增长系数（3000D → 5000D）

| 函数 | 系数 | 含义 |
|---|---|---|
| Ackley | 3.12× | 接近 O(D²)（理论 1.67²=2.78） |
| Rastrigin | 3.51× | 超 O(D²) |
| **Rosenbrock** | **4.98×** | 接近 O(D^2.6)，最陡 |

**难度排序随维度反转**：3000D 最慢是 Rastrigin（密集多极值）；5000D 变成 Rosenbrock（强耦合 + 维度增长系数最大）。

---

## 四、可复现性铁证（215 vs 251）

- **104 配对**（核心 9 算法 × 3000/5000D 共同行），**100% 逐字节一致，最大相对差 0.000**。
- 覆盖确定性（DEoptim/SMCO 系）与随机型（GenSA/SA/GA/PSO）全部 9 算法。
- 5000D Rastrigin GenSA 两机 fopt 逐字节全同（rep0=187432.179590326 / rep1=188862.966431347）。
- → R 移植版跨机**完全确定性收敛**。

---

## 五、运维教训（供未来高维实验参考）

1. **5000D × GenSA 极慢，函数依赖**：单 task 88–694h；Rosenbrock 增长最陡（~5×）。**未来高维实验优先用 3000D 封顶，或对 5000D 砍 GenSA/SA**（用 DEoptim/GA/PSO 秒级替代，虽弱但快）。
2. **外推系数勿用固定值**：曾用 2.69 外推致预测偏乐观，实测 3.12–4.98 波动大。应**按函数分别用已完成维度反推系数**。
3. **mclappy worker 跨 task 复用**：单 task 运行时间看 **log mtime**（= task 启动时刻），**绝不用 worker 进程 etime**（那是进程存活时间，跨多个 task）。
4. **log 长期静默 ≠ 死循环**：判据三件套——(a) `cpu/etime≈1.00`（100% 满载）；(b) `stat=R` worker 数稳定；(c) `load` 三档（1/5/15min）相等 ≈ 物理核数。三者满足即正常长尾。
5. **CPU 探测必须过滤 `stat=R`**：曾连续 4 轮被一个 103 天睡眠进程（`stat=Sl`）污染误读。正确写法见 memory `monitoring-probe-recipe.md`。
6. **mclappy CSV 批次刷新缺陷**：原 `run_highdim_r.R` 全部 task 完成才 flush CSV（长尾期 CSV 为空）；已修复为 filelock per-algo append（见 git 历史 + memory `r-highdim-csv-live-flush`）。**长尾期从 log 解析 fopt 绕过 CSV**。

---

## 六、产物清单（`result/r-highdim-2026-06-15/`）

| 文件 | 说明 |
|---|---|
| `analysis.md` | 实验3 详细分析报告（最终版） |
| `experiment-log.md` | 本档案（执行记录 + 运维教训） |
| `exp3_main.csv` | 合并主表（297 行：1000D 165 + 3000D 66 + 5000D 66） |
| `exp3_repro_validation.csv` | 215 vs 251 复现验证（104 配对，100% 逐字节） |
| `exp3_evo_advantage.csv` | SMCO_EVO vs 次强非 evo 优势表 |
| `r_highdim_results_251.csv` | 251 原始 CSV（3000/5000D 核心9，12/12 收官归档） |
| `run_251.log` | 251 完整运行 log（257 行，含 DONE） |
| `r_highdim_results_215_repro.csv` | 215 复现 CSV（3000/5000D 11 算法） |
| `r_highdim_results_215.csv` | 215 早期 CSV（1000D 全算法 165 行 + 部分 3000/5000D） |
| `figures/fig1-6.png` | 6 张图表（算法对比/evo优势/耗时/复现散点/R vs Python/GenSA维度增长） |

---

## 七、待办

1. 215 的 5000D Rosenbrock GenSA/SA 复现收官（~07/23，2 task），完成后复现配对数 104→108，预期仍 100% 逐字节。
2. R↔Python Rastrigin/Rosenbrock 绝对差异逐段定位（可选代码考古，见 `analysis.md` §7.3）。
3. 方向 bug 修正后的"真实最小化"重跑（用户决策暂不修，见 `docs/direction-bug-2026-06-15.md`）。
