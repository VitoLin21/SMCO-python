# 全量重跑方向修正实验 — 任务规划与执行记录

- **制定日期**：2026-07-20
- **修订**：v3 — v1/v2 分类有误（v1 混两套、v2 误把 `smco.run_benchmark` 归管道 B）；v3 区分三套路径，见 §1
- **相关**：`docs/direction-bug-2026-06-15.md`、`docs/direction-bug-audit-2026-07-20.md`

---

## 1. 三套 benchmark 路径（关键区分）

代码库有**三套**构造目标函数的路径，方向与输出标尺各异。`config.f` 对 `sense=="min"` 已是 `-raw`（SMCO 最大化约定，`_min_config`）：

### 管道 A — 双重取负 bug（真受害者）
调用方对 `sense=="min"` 再取一次负：
```python
config = assign_config(...)        # config.f = -raw  (min, SMCO 最大化目标)
f = config.f
if sense == "min":
    f = lambda x: -config.f(x)     # f = +raw  ← 双重取负！
result = variant_fn(f, ...)        # SMCO 最大化 +raw = 推边界
# 对比算法也 maximize=True + f=+raw → 一致推边界
fopt = result.best_result.f_optimal  # = max(+raw)，+raw 标尺（min 函数正值≈边界）
```
**fopt 输出 +raw 标尺**（min 函数正值，≈边界）。这是真 bug。

### 管道 B — `run_comparison()` 正确双向（非 bug）
`src/comparison/run_comparison.py:110-119`：
```python
raw_f = (-config.f) if sense=="min" else config.f   # 还原成原始 raw（-(-raw)=raw）
effective_f = raw_f if to_maximize else -raw_f       # 始终最大化；min 时 = -raw ✓
f = effective_f
# fopt 转回 raw 标尺 (line 173-174)
```
对 min 函数 + `to_maximize=False`：最大化 `-raw` = 最小化 raw ✓。**fopt 输出 raw 标尺**（正）。完全正确。`vendor/SMCO_R/main/benchmark.R:116-166` 是同一管道（`test_func=raw` + `to_maximize` 选 `raw`/`-raw` + fopt 转回 raw 标尺），也正确。

### 路径 C — `smco.run_benchmark` direct-SMCO（正确，-raw 标尺）
`src/smco/benchmark.py:83/91` 直接把 `config.f` 传给 optimizer（**不取负、无 `to_maximize`**，不经 `run_comparison`）。min：最大化 `-raw` = 最小化 raw ✓ **方向正确**。**fopt 输出 -raw 标尺**（min 函数负值，接近 0）。产出 `smco-evo-vs-smco-2026-05-27`、`benchmark-results-paper-comparison`。
- 与路径 A 区别：A 多取负→推边界（+raw 正边界值）；C 不取负→正确（-raw 负接近 0）。
- 与路径 B 区别：B 还原 raw+`to_maximize`+raw 标尺；C 直接 `config.f`、-raw 标尺、无 to_maximize。

**修正后的路径 A 6 脚本**（`f=config.f`）方向与标尺等同路径 C（-raw 标尺，fopt 负接近 0）。

**历史误判**：v1 用"fopt 正=bug"扫描，误判路径 B（raw 标尺正）；v2 又误把路径 C（`smco.run_benchmark`）归入路径 B。v3 区分三套。

## 2. 真 bug 受害者清单（管道 A，已修，需重跑）

| 脚本 | result 目录 | 估算 task |
|---|---|---|
| `run_highdim_full_comparison.py` | `highdim-full-comparison-2026-06-04` | **18,928** ⭐ |
| `run_smco_evo_comparison.py` | `evo-comparison-2026-05-30`（8 函数，`fopt` 多正=推边界） | ~3,024 |
| `run_highdim_comparison.py` | `highdim-comparison-2026-05-31` | ~1,348 |
| `run_highdim_v2.py` | `ultrahighdim-2026-06-01` | ~624 |
| `run_startsweep_base_comparison.py` | `startsweep-base-2026-06-18` | 324 |
| `vendor/SMCO_R/main/run_highdim_r.R` | `r-highdim-2026-06-15` | ~660 |

衍生聚合（重跑源后重打包）：
- `highdim-overall-2026-06-02` = `package_highdim_results.py` 聚合 `highdim-comparison-2026-05-31` + `ultrahighdim-2026-06-01`（**两者管道 A，不含 highdim-full**），重跑这两个源后重打包。
- `smco-evo` 打包物（`package_smco_evo_results.py`）不需重跑。直接对比 `smco-evo-vs-smco-2026-05-27`（18 函数、`best_value` **-raw 标尺**（min 函数负值）、`gap_to_known`≈0）由 `smco.run_benchmark`（**路径 C**，直接 `config.f`）产出——**不是 `run_smco_evo_comparison.py` 的产物**（后者路径 A，产出 `evo-comparison-2026-05-30`；脚本/schema/路径均不同，勿混）。其余 paper_tables/table3/r-synced/strategy_sweep 是路径 B（raw 标尺）。

旧版被取代（管道 A 旧版，可选）：`evo-startsweep-2026-06-15` / `evo-dimsplit-2026-06-15` 已被 7/18 正确版取代。

## 3. 非 bug（管道 B，正确，不改不重跑）

`run_paper_tables_with_evo.py`、`run_synced_r_benchmarks_with_evo.py`、`run_targeted_br_gensa.py`、`run_experiment.py`、`run_seed_sensitivity.py`、`run_evo_strategy_sweep_serial.py`（subprocess 调 paper_tables）、`run_paper_comparison_benchmark.py`（`smco.run_benchmark`）、`vendor/SMCO_R/main/benchmark.R`。

这些的 fopt 是 raw 标尺（min 函数为正，正常），**不是 bug**。对应的 `paper_tables_integrated` / `r-synced-benchmarks` / `comparison-targeted-br-gensa` / `comparison` / `seed-test` / `evo_strategy_sweep` / `table3` / `benchmark-results-paper-comparison` 均移出重跑清单。

## 4. 修正方案（管道 A 6 处，已完成 #45）

统一模式：调用方 `f = config.f`，删 `if sense=="min"` 取负分支。
- `run_highdim_full_comparison.py:234-236`、`run_highdim_comparison.py:81-84`、`run_highdim_v2.py:112-114`、`run_smco_evo_comparison.py:77-80`、`run_startsweep_base_comparison.py:219-221` → `f = config.f`
- `run_highdim_r.R:142` → `fobj <- function(x) -raw(x)`（+ 更新 line 16-22/137 注释）

**不改**：`test_functions.py`（`_min_config` 的 `-raw` 是 SMCO 最大化约定的正确部分）、所有管道 B 脚本、`benchmark.R`。

## 5. 重跑范围（收窄到管道 A）

参数：全 18 算法（含慢全局）、全 9 维 50-5000D、R 线修复。

- **P0 核心**：`run_highdim_full_comparison`（18,928）⭐、`run_highdim_r.R`（~660）、`run_smco_evo_comparison`（~3,024）
- **P1 补充**：`run_highdim_comparison`（~1,348）、`run_highdim_v2`（~624）、`run_startsweep_base_comparison`（324）

合计 **~25k task**（v1 的 55k 含管道 B，已移出）。

## 6. 资源与分工

- **5 远程机**：217 / 213 / 215 / 251 / **253(新)**，各 48 核，每机 **4/5 = 38 workers**，合计 **190 核**。
- 215/251/253（配 R 后）承担 `run_highdim_r.R` + Python。
- **排除**：本机（满载）、216（SSH 故障）。
- ⚠️ **253 环境未知**（密码新机 `Sdumt@10.25.40.253` / `kRohWd#92w`，本机无 sshpass），归入阶段 0；若装不了 python/R 则降级 4 机（152 核）。

## 7. 执行阶段与任务进度

| # | 任务 | 状态 | 备注 |
|---|---|---|---|
| 44 | 生成老实验方向错误汇总文档 | 🔄 待按 v2 重新分类 | `docs/direction-bug-audit-2026-07-20.md` |
| 45 | 修正管道 A 6 脚本方向 bug | ✅ 完成 | 5 Python + 1 R；管道 B 不动 |
| 46 | 配置新机 253 | ⏳ 待办 | 本机无 sshpass，需先装 |
| 47 | rsync 修正代码+venv 到 5 机 | ⏳ 待办 | 依赖 46 |
| 48 | 217 冒烟验证方向修正 | ⏳ 待办 | 依赖 47 |
| 49 | 启动 P0 分布式重跑（管道 A） | ⏳ 待办 | 依赖 48 |
| 50 | 监控重跑（cron 每 2h） | ⏳ 待办 | 依赖 49 |
| 51 | 收集合并+方向验证 | ⏳ 待办 | 依赖 50 |
| 52 | 写详细实验记录文档 | ⏳ 待办 | `result/rerun-2026-07-20/experiment-log.md` |
| 53 | 更新 direction-bug doc + memory | ⏳ 待办 | 标记已修 |

**阶段**：0 环境准备（装 sshpass→配 253→rsync）→ 1 修正已完成+冒烟 → 2 分布式重跑+监控 → 3 收集+记录+更新。

## 8. 启动陷阱（见 memory `exp12-direction-rerun-deploy`）

- `pkill -f pattern` 自杀 → `[s]` 正则技巧或脚本文件。
- `nohup` 不够 → `setsid bash -c "..." </dev/null >/dev/null 2>&1 &`。
- `mkdir && nohup` 竞态 → 用 `;`。
- 每机独立 FS → rsync 分发代码+venv；`PYTHONPATH=src` 双保险。

## 9. 验证标准（v2 精确化）

**仅对管道 A 6 脚本**（只跑 min 函数 Rastrigin/Ackley/Rosenbrock/Griewank/Zakharov）：
- **方向铁证**：改后 `fopt = -raw` 全负（raw>0），0 正 0 nan。
- **真解对比**：Rastrigin 1000D fopt≈-8,100（非 +29,130 边界值）；Rosenbrock 1000D fopt≈-31（非 +4.3 亿）。
- **R vs Python 同标尺**：`run_highdim_r.R` 与 `run_highdim_full_comparison` 都输出 -raw 标尺，fopt 同负、可逐行对比。
- **抽样逐点**：固定 seed task，修正后 raw ≪ 修正前（边界值）。

**取消** v1 的"所有重跑 fopt 全负"全局判据——管道 B 不在重跑范围，且其 raw 标尺本就为正。

## 10. 风险与规模

- **253 不确定性**：环境未知 → 降级 4 机（152 核）。
- **耗时**：~25k task，190 核预计 **3-6 天**；慢算法高维（~840 GenSA/SA task）是瓶颈。
- **磁盘**：各机 15-20TB，充足。
