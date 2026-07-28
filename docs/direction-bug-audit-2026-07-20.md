# 老实验方向错误审计汇总

- **审计日期**：2026-07-20
- **修订**：v3 — v1 用 fopt 符号扫描误判；v2 按脚本分类但误把 `smco.run_benchmark` 归管道 B；v3 区分三套路径（A bug / B run_comparison / C smco.run_benchmark），见 `docs/rerun-plan-2026-07-20.md` §1。
- **背景**：[[smpco-direction-bug]] —— 部分脚本对 `sense=="min"` 双重取负。

---

## 1. 审计方法（v3 修正）

代码库有**三套**目标函数构造路径（详见 `rerun-plan-2026-07-20.md` §1）。`config.f` 对 `sense=="min"` 已是 `-raw`（SMCO 最大化约定）：

- **路径 A（bug）**：调用方 `f = -config.f`（双重取负）→ 最大化 `+raw` → 推边界。fopt 输出 **+raw 标尺**（min 函数正值，≈边界）。
- **路径 B（正确）**：`run_comparison()` / `run_paper_tables_with_evo` / `benchmark.R` 还原 raw + `to_maximize` + 转回标尺。fopt 输出 **raw 标尺**（min 函数正值，接近 0）。
- **路径 C（正确，-raw 标尺）**：`smco.run_benchmark`（`src/smco/benchmark.py:83/91`）直接用 `config.f`（**不取负、无 to_maximize**）→ 最大化 `-raw` = 最小化 raw ✓。fopt 输出 **-raw 标尺**（min 函数负值，接近 0）。产出 `smco-evo-vs-smco-2026-05-27`、`benchmark-results-paper-comparison`。

v1 的"fopt 正 = bug"判据只对路径 A 有效（路径 B 的 raw 标尺正输出被误判）；v2 又误把路径 C 归入路径 B。v3 区分三套。**修正后的路径 A 6 脚本**（`f=config.f`）方向与标尺等同路径 C（-raw 标尺，fopt 负接近 0）。

## 2. 真 bug 受害者（管道 A，需重跑）

| 脚本（管道 A） | result 目录 | 估算 task |
|---|---|---|
| `run_highdim_full_comparison.py` | `highdim-full-comparison-2026-06-04`（+ analysis 衍生） | **18,928** ⭐ |
| `run_smco_evo_comparison.py` | `evo-comparison-2026-05-30`（8 函数，`fopt` 多正=推边界） | ~3,024 |
| `run_highdim_comparison.py` | `highdim-comparison-2026-05-31` | ~1,348 |
| `run_highdim_v2.py` | `ultrahighdim-2026-06-01` | ~624 |
| `run_startsweep_base_comparison.py` | `startsweep-base-2026-06-18` | 324 |
| `vendor/SMCO_R/main/run_highdim_r.R` | `r-highdim-2026-06-15` | ~660 |

衍生聚合（重跑源后重打包）：
- `highdim-overall-2026-06-02` = `package_highdim_results.py` 聚合 `highdim-comparison-2026-05-31` + `ultrahighdim-2026-06-01`（**两者管道 A，不含 highdim-full**）。
- `smco-evo` 打包物不需重跑。直接对比 `smco-evo-vs-smco-2026-05-27`（18 函数、`best_value` **-raw 标尺**（min 函数负值）、`gap_to_known`≈0）由 `smco.run_benchmark`（**路径 C**，直接 `config.f`）产出——**不是 `run_smco_evo_comparison.py` 的产物**（后者路径 A，产出 `evo-comparison-2026-05-30`；脚本/schema/路径均不同，勿混）。其余 paper_tables/table3/r-synced/strategy_sweep 是路径 B（raw 标尺）。

旧版被取代（管道 A 旧版，可选重跑）：`evo-startsweep-2026-06-15` / `evo-dimsplit-2026-06-15`（已被 7/18 正确版取代）。

## 3. 非 bug（管道 B，正确，v1 误判，现移出）

| 脚本（管道 B） | result 目录 | 说明 |
|---|---|---|
| `run_paper_tables_with_evo.py` | `paper_tables_integrated_2026-05-27` | `raw_f`+`to_maximize` 正确双向 |
| `run_synced_r_benchmarks_with_evo.py` | `r-synced-benchmarks-2026-05-28` | 走 `run_comparison` |
| `run_targeted_br_gensa.py` | `comparison-targeted-br-gensa-2026-05-14` | 走 `run_comparison` |
| `run_experiment.py` | `comparison/`、`comparison-rerun-2026-05-13[-d200]` | 走 `run_comparison` |
| `run_seed_sensitivity.py` | `seed-test/` | 已正确处理方向 |
| `run_evo_strategy_sweep_serial.py` | `evo_strategy_sweep/` | subprocess 调 paper_tables |
| `run_paper_comparison_benchmark.py` | `benchmark-results-paper-comparison` | `smco.run_benchmark` |
| `vendor/SMCO_R/main/benchmark.R` | — | `test_func=raw`+`to_maximize` 正确双向 |
| `smco.run_benchmark`（`src/smco/benchmark.py:83`） | `smco-evo-vs-smco-2026-05-27`、`benchmark-results-paper-comparison` | **路径 C**：直接 `config.f`，方向正确，18 函数 `best_value` **-raw 标尺**，`gap_to_known`≈0 |
| `package_smco_evo_results.py`（打包） | `smco-evo` | 打包 `smco-evo-vs-smco`+paper_tables+table3+r-synced+strategy_sweep，全管道 B |

这些的 fopt 是 raw 标尺（min 函数为正），**方向正确**，不重跑。

## 4. 方向正确（实验1/2，已 7/18 重跑，排除）

`evo-startsweep-2026-07-18`（405）、`evo-dimsplit-2026-07-18`（648）——管道 A 但已用修正代码重跑，fopt 全负。

## 5. 铁证（管道 A 推向边界）

| 函数/维度 | 老 6/04 基准 fopt | 7/18 正确重跑 fopt | 含义 |
|---|---|---|---|
| Rastrigin 1000D | **+29,130** | −8,100 | 边界 raw≈26,400 vs 真解 0 |
| Rosenbrock 1000D | **+4.26 亿** | −31.4 | 窄谷在中心、边界 raw 爆炸 |
| Rosenbrock 5000D | +17.6 亿 | −1,362.3 | |

老基准 fopt 全正、≈边界 raw 最大值 → 最大化 raw（推边界）。算法级铁证（`direction-bug-2026-06-15.md` §2）：Rastrigin 1000D SMCO=29108 / PSO=29130 / GA=21034 / GenSA=40353 全是大正值。

## 6. 影响评估（v2 修正）

1. **管道 A**：所有算法一致反向（SMCO 6 变体 + 对比算法 `maximize=True`+`f=+raw`）。**绝对值不可信**（fopt 实为边界 raw），相对排名在"推边界"语义下自洽。
2. **管道 B**：方向正确，**无 bug**。v1 基于 fopt 符号的误判已撤销。
3. **R 线**：`run_highdim_r.R` 是管道 A（bug，已修 `-raw`）；`benchmark.R` 是管道 B（正确，不动）。两者标尺不同（前者 -raw，后者 raw），各自与对应 Python 管道可比。

## 7. 处置

- **重跑**：管道 A 的 6 个脚本，用 `f = config.f` / `-raw(x)` 修正后重跑。计划见 `docs/rerun-plan-2026-07-20.md`。
- **不动**：管道 B 全部脚本（方向本就正确）+ 实验1/2（已重跑）。
- **修正方法**：统一 `f = config.f`（删 `if sense=="min"` 取负分支）；**不改** `test_functions.py`、管道 B 脚本、`benchmark.R`。

## 8. 相关文件

- 方向 bug 详档：`docs/direction-bug-2026-06-15.md`
- 重跑计划：`docs/rerun-plan-2026-07-20.md`
- 管道 A 根因：各 `scripts/run_highdim_*.py` / `run_smco_evo_comparison.py` / `run_startsweep_base_comparison.py` 调用方 + `run_highdim_r.R`
- 管道 B（正确）：`src/comparison/run_comparison.py:110-119`、`run_paper_tables_with_evo.py`、`vendor/SMCO_R/main/benchmark.R:116-166`
