# 全量重跑实验记录（方向修正）

- **起止**：2026-07-20 启动，2026-07-28 持续（R 线 GenSA 极慢）
- **目标**：用修正代码重跑所有受方向 bug 影响的实验，产出方向正确（fopt 全负）的基准数据，替代推边界的旧 6/04 基准
- **执行**：5 机分布式（217/213/215/251/253），本机仅编辑/分析
- **结果**：highdim-full 完整（18928 rows）+ R 线部分（285 rows，缺 GenSA 12）+ smco_evo 完成

---

## 1. 方向修正

**Bug**（`docs/direction-bug-2026-06-15.md`）：3+ 个脚本对 `sense=="min"` 双重取负（`f=-config.f=+raw`），SMCO 及对比算法实际最大化 raw（推边界），fopt 全正。铁证：Rastrigin 1000D 老 fopt=+29108（≈边界），真解 raw=0。

**改法**（统一模式，不改 `test_functions.py`）：调用方 `f=config.f`（删 `if sense=="min"` 取负分支）。修正 6 Python + 1 R：
- `run_highdim_full_comparison.py` / `run_highdim_comparison.py` / `run_highdim_v2.py` / `run_smco_evo_comparison.py` / `run_startsweep_base_comparison.py` / `run_paper_tables_with_evo.py`（核查）
- `vendor/SMCO_R/main/run_highdim_r.R:142`（`fobj<-function(x) -raw(x)`）

**三路径模型**（`docs/direction-bug-audit-2026-07-20.md`）：
- 路径 A（bug，`f=-config.f`，+raw 标尺正边界值）— 6 脚本，重跑对象
- 路径 B（`run_comparison`，raw 标尺）— 正确，不动
- 路径 C（`smco.run_benchmark`，-raw 标尺）— 正确，不动

**度量**：`raw=-fopt`（越小越优）。修正前全正（推边界），修正后全负（接近真解）。

## 2. 时间线

### 2026-07-20（启动日）
- 审计 + 规划文档（direction-bug-audit v3、rerun-plan v3）
- 修正 6 Python + 1 R 脚本（本机编辑）
- 新机访问与部署细节已移至受保护的本地运维记录。
- rsync 修正代码到 5 机
- 217 冒烟（50D rand1bin quick）：108 task fopt 全负 ✓（Rastrigin -305 vs 修正前 +1617）
- **启动 highdim-full 5 机分布式**：217→100+5000D, 213→200+4000D, 215→500+3000D, 251→1000+2000D, 253→50D（各 38 核，253 20 核，setsid 后台）
- 253 启动 smco_evo（evo-comparison）

### 2026-07-20 ~ 24（highdim-full 跑，~4 天）
- 5000D 慢算法瓶颈（GenSA/SA 单 task ~3-6h，Python scipy 5000D）
- 253 50D + smco_evo 先完成（~11min + 数小时）
- 213（200+4000D）/215（500+3000D）/251（1000+2000D）依次完成（~3 天）
- 217（100+5000D）最后，5000D 4 strategy batch 慢算法（rand1bin/cur-to-best1bin/best1bin/sobol）
- 监控：cron 每 1h（后改 2h）probe worker/CSV/fopt；多次确认 5000D global CPU 满（非死锁）
- 217 task 64（5000D GenSA）卡 ~3h 完成；多轮卡 global 但推进

### 2026-07-22（分阶段交付指示）
用户指示：4000D（213）完成后先合并非5000D + 阶段报告，5000D 继续后补充。
- 2026-07-23：213 完成触发分阶段合并 → `staged-report-no5000.md`（17314 rows，方向全负）

### 2026-07-24（highdim-full 完成 + R 线启动）
- **217 完成**（5000D 全 4 strategy，rows 5665）→ 完整合并 `all_results_full.csv`（18928 rows，fopt 全负）
- `final-report.md`（完整）+ `analysis-highdim-full.md`（论文级 SMCO-EVO 分析）
- **启动 R 线**（run_highdim_r.R）：215→1000D, 251→3000+5000D（fobj=-raw 修正）

### 2026-07-24 ~ 25（R 线问题）
- 251 缺 `filelock` 包 → rsync 215→251 补齐（尾斜杠内容复制）
- 215 启动命令反复漏 `cd`（vendor 相对路径）→ heredoc 经 stdin 传 cd 解决
- **215 宕机**（2026-07-24，ssh 超时 + ping 100% loss，持续数天，需现场）
- 1000D 移到 251 补跑（`r-highdim-rerun-1000D-251`，--dims 1000，10 核）

### 2026-07-25 ~ 27（R 线推进 + SA）
- 1000D 补跑（251，慢算法 GenSA ~8h/task）
- 2026-07-25：`legacy-boundary-exploration.md`（老反向 = 推边界能力，论文素材）
- 2026-07-26：1000D 补跑完成（166 rows）
- 2026-07-27：SA 线启动（251，--algos SA，20 核利用空闲）→ SA 完成（12 task）
- R 线 SMCO-EVO 分析（`r-line-smco-evo-analysis.md`，含 SA 互补）
- 主线 5000D GenSA 极慢（CPU 满，退火链极长，非死循环）

### 2026-07-28（当前）
- 主线 5000D GenSA 极慢持续（用户确认非死循环，继续等）
- R 线 285 rows（缺 3000+5000D GenSA 12 task）
- experiment-log.md 编写

## 3. 5 机分工（highdim-full）

| 机 | 核心 | 维度 | 完成时间 |
|---|---|---|---|
| 253（新）| 20 | 50D | 2026-07-20（~11min，+smco_evo）|
| 213 | 38 | 200+4000D | 2026-07-23 |
| 215 | 38 | 500+3000D | 2026-07-23 |
| 251 | 38 | 1000+2000D | 2026-07-23 |
| 217 | 38 | 100+5000D | 2026-07-24（5000D 慢算法）|

R 线：215（1000D，后宕机）→ 251 补（1000D + 3000+5000D + SA）

## 4. 关键决策点

1. **5000D 全局算法**（07-21）：5000D GenSA/SA 28 并行 8h+ 无完成，用户选"等全 18 算法"（~5 天预期）。
2. **分阶段交付**（07-22）：4000D 完成先合并非5000D + 阶段报告，5000D 后补。
3. **215 宕机**（07-24）：1000D 移 251 补跑。
4. **R 线 GenSA/SA 慢**（07-25+）：用户确认高维慢，不 kill，继续等。
5. **R 线 SA 补跑**（07-27）：251 空闲 CPU 启 SA 专跑线。

## 5. 异常处理

| 异常 | 处理 |
|---|---|
| 253 无 `/amax/math`/uv/python3.8 | rsync uv python + venv 到 ~/，修符号链接 + pyvenv.cfg + editable .pth |
| 251 缺 filelock | rsync 215→251（尾斜杠内容复制）|
| 反复漏 `cd`（vendor 相对路径）| heredoc 经 stdin 传 `cd` 第一行 |
| 215 宕机（硬件/网络）| 1000D 移 251 补跑；highdim-full 数据安全（早已 rsync）|
| 5000D GenSA/SA 极慢 | CPU 满确认非死锁；用户接受慢继续等 |
| pkill -f 自杀 / nohup 不够 / mkdir 竞态 | setsid + [s] 正则 + ; 分隔（见 memory exp12）|

## 6. 最终产出

**数据**：
- `result/highdim-full-rerun-2026-07-20/merged/all_results_full.csv`（**18928 rows**，highdim-full 完整，9 维度×18 算法×4 strategy，fopt 全负）
- `result/r-highdim-rerun-2026-07-24/merged/r_highdim_merged.csv`（**285 rows**，R 线，缺 GenSA 12）

**报告**：
- `final-report.md`（highdim-full 完整）+ `analysis-highdim-full.md`（论文级）
- `r-line-smco-evo-analysis.md`（R 线 SMCO-EVO + SA 互补 + R vs Python）
- `legacy-boundary-exploration.md`（老反向推边界，论文素材）
- `staged-report-no5000.md`（阶段，已取代）

**文档**：
- `docs/direction-bug-audit-2026-07-20.md`（三路径审计 v3）
- `docs/rerun-plan-2026-07-20.md`（重跑规划 v3）
- `docs/direction-bug-2026-06-15.md`（bug 详档）

## 7. 核心发现

1. **方向修正成功**：highdim-full 18928 rows fopt 全负（修正前全正推边界）。
2. **SMCO-EVO vs base**（Python）：EVO 无明显增益（win rate 43-50%）。
3. **SMCO_BR/BR_EVO Python 高维灾难**（iter_boost 发散，5000D raw 42862）。
4. **R SMCO_BR_EVO 高维稳定**（5000D 122.6 vs Python 42862）——R 移植修复了 Python 实现缺陷。
5. **SA 与 SMCO-EVO 互补**（SA Rosenbrock 完美 0，SMCO Ackley 强）。
6. **老反向 = 推边界能力对比**（GenSA 最强，论文素材）。

## 8. 经验教训

- **方向度量必须从数学定义确认**（raw=-fopt 越小越优，非"fopt 越负越好"）——初版报告结论全反。
- **三套 benchmark 管道**（A bug / B run_comparison / C smco.run_benchmark）勿混为一个 bug 家族。
- **setsid + heredoc cd** 解决 ssh 后台 + cwd 问题。
- **per-task 去重 + 增量保存** 支持断点续跑/分片。
- **R 高维 GenSA/SA 极慢**（单 task 数小时-数天），需提前评估或限 iter。

## 9. 未完成

- R 线 3000+5000D GenSA（12 task，极慢，继续等）
- 215 宕机（需现场恢复；1000D 已由 251 补齐）
- `direction-bug-2026-06-15.md` 标记"已修" + memory 更新（任务 #53）
