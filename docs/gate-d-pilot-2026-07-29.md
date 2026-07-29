# Gate D Pilot 报告（2026-07-29）

本机（R 4.3.2 + jsonlite 2.0.0 + qrng 0.0.11）代表性 pilot，验证 SMCO-EVO
高维实验基础设施在进入 E1 全量（Gate E）前的正确性与资源量级。pilot 只验证
基础设施，不调超参数（计划 Gate D 约定）。

## 1. 配置

- **d=200 全配置 pilot**：Rastrigin × d=200 × 1 实例 × 18 配置
  （9 Python + 9 R：2 语言 × {3 base + 6 EVO}）× 100 FE/d = 20000 FE/task。
- **d=1000 单 task 测时**：PY-BASE-SMCO 与 R-SP-SMCO-EVO，100 FE/d = 100000 FE。
- 共享起点 n_starts=8；EVO 默认 rand1bin / evolution_points (0.5,0.75) /
  elimination_rate 0.25 / de 0.8/0.7（方案 4.2 冻结值）。

## 2. Gate D 验收项

| 项 | 结果 |
| --- | --- |
| 方向 | `best_value` 为 minimization（Rastrigin ≥ 0）；budget/direction violations = **NONE** |
| FE 预算 | 所有 task `fe_used ≤ fe_budget`（d=200 全 ≈19665–19701/20000，hard stop 生效） |
| schema | result payload 字段齐：status/fe_used/best_value/known_optimum/normalized_gap/target_hit_fe/anytime/wall_time_sec/machine_id |
| 日志 | 每 run 写 `logs/<run_id>.log`（Py + R worker） |
| resume | `is_run_complete` 仅认 success；infra/timeout 重试（`tests/test_highdim_batch.py` 覆盖） |
| timeout | `run_batch --wall-time-cap` 经 subprocess `TimeoutExpired` → status=timeout（本 pilot 全 success 未触发，单测覆盖） |
| Python/R 对称 | 两 worker 同 FE 预算、同 observer、同 target_hit 相对阈值；Py/R instance loader 数值一致（`tests/test_r_instance_parity.py`，atol/rtol 1e-9） |

d=200 全 18 配置 **全部 success**（48.9s，workers=4）。

## 3. 资源量级（d=1000，100 FE/d）

| worker | wall time | peak RSS |
| --- | --- | --- |
| PY-BASE-SMCO | 35.7 s | 103 MB |
| R-SP-SMCO-EVO | 93.2 s | 126 MB |

R 约为 Python 的 2.6×。内存均 < 130 MB（block rotation 不构造 dense d×d，符合 Task 6 设计）。

## 4. E1 全量估算（Gate E，1080 runs）

方案 E1 = 4 函数 × {200,500,1000} × 5 实例 × 18 配置。按 d=1000 量级外推
（FE 与维度近似线性放大）：

- d=200（360 runs）：约 1 h
- d=500（360 runs）：约 2.5 h
- d=1000（360 runs）：约 6 h
- 单机串/并行合计约 **9 h**；fleet 分片（本机 + 253/251/213/214/215/217）可降到 1–2 h/节点。

> 注：E2 用 2000 FE/d 且含 d=3000/5000，单 task 显著更慢（历史 5000D 单任务
> 22–30h，见 `highdim-5000d-slow-tasks` 记忆），需单独估时与 wall-time cap。

## 5. 建议 wall-time cap

- E1（100 FE/d）：d≤1000 单 task cap **600 s**（R ~93 s 留 6× 余量）。
- E2/E3（2000 FE/d，d≤5000）：需按 d 分档 cap，pilot 前在 fleet 实测再冻结。

## 6. 结论

Gate D pilot 通过：方向 / FE / schema / 日志 / resume / timeout 基础设施均正常，
Python 与 R worker 端到端一致。可进入 Gate E（E1 全量 1080 runs + 全局选型 +
冻结 confirmatory manifest），**但 E1 全量需用户再次确认后启动**（战役约定：
不在同一轮自动启动确认性计算）。

pilot 过程中发现并修复的 R worker 问题（见 commit）：metadata 数值字段为
`_frepr` 字符串需 `as.numeric`；`seed` 32-bit 超 R integer max 需取模；
source 顺序需显式含 `evaluation_budget.R` + `SMCO.R`；运行环境需 `jsonlite` + `qrng`。
