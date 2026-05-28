# SMCO Python 基准与结果交接文档（给后续 LLM 代理）

## 1. 项目现状

- 仓库：`https://github.com/VitoLin21/SMCO-python`
- 当前分支：`main`
- 当前工作树已经包含：
  - SMCO / SMCO_R / SMCO_BR 主线实现
  - `smco_evo / smco_r_evo / smco_br_evo`
  - `smco_hb`
  - comparison methods
  - 完整的 `smco-evo` 结果打包与报告脚本

本仓库是对 R 版本 `wayne-y-gao/SMCO` 的 Python 迁移，目标是：

- 主 API 对齐 `v1.1.0` 风格；
- 保留 `v1.0.0` 论文复现参考路径（以 vendored R 源码形式）；
- 测试不依赖 R 运行时。
- benchmark 结果与报告允许依赖本地已生成的 `result/` 工件。

## 2. Python 环境（统一要求）

统一使用 `uv + Python 3.12`：

```bash
cd /amax/math/code/SMCO
uv python install 3.12
uv venv .venv --python 3.12
uv pip install -e ".[test]"
```

运行测试：

```bash
.venv/bin/python -m pytest -q
```

当前建议基线命令：

```bash
.venv/bin/python -m pytest -q
```

## 3. 代码结构

- 核心优化器：
  - `src/smco/optimizer.py`
- 结果类型：
  - `src/smco/results.py`
- benchmark 函数与配置：
  - `src/smco/test_functions.py`
- benchmark runner：
  - `src/smco/benchmark.py`
- evolutionary benchmark/report scripts：
  - `scripts/run_paper_tables_with_evo.py`
  - `scripts/run_synced_r_benchmarks_with_evo.py`
  - `scripts/package_smco_evo_results.py`
  - `scripts/run_evo_strategy_sweep_serial.py`
- 论文 smoke runner：
  - `src/smco/paper.py`
- 包导出：
  - `src/smco/__init__.py`
- 测试：
  - `tests/test_optimizer.py`
  - `tests/test_validation.py`
  - `tests/test_test_functions.py`
  - `tests/test_benchmark.py`

## 4. 已实现算法入口

- `smco(...)`
- `smco_r(...)`
- `smco_br(...)`
- `smco_evo(...)`
- `smco_r_evo(...)`
- `smco_br_evo(...)`
- `smco_multi(...)`
- `smco_hb(...)`

说明：

- 默认是“最大化”。
- 若要做最小化，传入负目标（benchmark 配置层已做包装）。

## 5. benchmark / 报告入口

通用入口：

```python
from smco import run_benchmark

run = run_benchmark(
    name="Rastrigin",
    dim=2,
    variant="smco_r",  # smco / smco_r / smco_br / smco_multi（大小写不敏感）
    repetitions=3,
    iter_max=100,
    n_starts=5,
    seed=123,
)
```

论文 smoke：

```python
from smco import run_paper_smoke
runs = run_paper_smoke(iter_max=100, n_starts=5, seed=123)
```

SMCO-EVO 结果工作流的主入口不是 Python API，而是脚本：

```bash
.venv/bin/python scripts/run_paper_tables_with_evo.py --table12-evo ...
.venv/bin/python scripts/run_paper_tables_with_evo.py --table3 --table3-evo-only ...
.venv/bin/python scripts/run_synced_r_benchmarks_with_evo.py --algo-set smco ...
.venv/bin/python scripts/package_smco_evo_results.py
```

## 6. 与 R 参考代码关系

已 vendored：

- `vendor/SMCO_R/main/*`（上游主线参考）
- `vendor/SMCO_R/v1.0.0/*`（论文复现参考）

注意：

- Python 测试不会执行 R。
- Sobol 序列来自 SciPy，不承诺与 R `qrng` bit-level 一致。

## 7. benchmark 兼容性要点（已处理）

`assign_config(...)` 已兼容并覆盖：

- 上游命名别名：`SqNorm`、`-SqNorm`、`Mishra06`；
- 2D bounds 与 vendored R 对齐修正：
  - `Shubert`: `[-5.12, 5.12]^2`
  - `Bukin6`: lower `[-15, -5]`, upper `[-3, 3]`

另外：

- `BenchmarkConfig.f` 会做输入维度检查；
- `known_best_value` 保留原 benchmark 标尺；
- `known_best_objective` 为 SMCO 最大化标尺（若可用）；
- `Mishra6/Mishra06` 的最优元数据标记为 `None`（避免传播不可信最优值）。

## 8. 当前已完成的 SMCO-EVO 结果交付

最终整理目录：

- `result/smco-evo/`

关键文件：

- `result/smco-evo/report.md`
- `result/smco-evo/analysis/direct_smco_vs_evo_cases.csv`
- `result/smco-evo/analysis/paper_pairwise_summary.csv`
- `result/smco-evo/analysis/r_synced_pairwise_summary.csv`
- `result/smco-evo/analysis/strategy_sweep_summary.csv`

已覆盖的结果范围：

1. `smco` vs `smco-evo` 直接 head-to-head
2. 论文 `Table 1 / Table 2 / Table 3`
3. 从 R 同步到 Python 的 18 个 benchmark
4. 四种进化策略：
   - `rand1bin`
   - `current-to-best1bin`
   - `best1bin`
   - `sobol`

策略 sweep 当前的最终口径：

- 每个策略都覆盖 `42` 个唯一 benchmark case
- 每个策略都补齐了：
  - `table1/2`: `16` 个 CSV
  - `table3`: `8` 个 CSV

## 9. 后续代理建议执行流程（benchmark / 报告）

1. 固定环境：确保只使用 `.venv/bin/python`。
2. 先跑全量测试，确保基线一致：
   ```bash
   .venv/bin/python -m pytest -q
   ```
3. 若只是刷新最终汇总，不要手工改 `result/smco-evo/report.md`；直接运行：
   ```bash
   .venv/bin/python scripts/package_smco_evo_results.py
   ```
4. 若是补跑论文或 R-synced 结果，优先用现有脚本并保持目录口径一致：
   ```bash
   .venv/bin/python scripts/run_paper_tables_with_evo.py ...
   .venv/bin/python scripts/run_synced_r_benchmarks_with_evo.py ...
   ```
5. 若是补跑策略 sweep 缺口，优先检查：
   - `result/evo_strategy_sweep/<strategy>/paper_tables_integrated/`
   - `result/evo_strategy_sweep/<strategy>/table3/`
   - `result/evo_strategy_sweep/<strategy>/r_synced/`
6. 对 `table2 d=200` 这类长任务，优先单标签运行或用 `scripts/run_evo_strategy_sweep_serial.py` 控制补跑，避免大批量后台任务难以审计。
7. 如需和 R 数值对照，新增可选脚本，不要破坏当前“不依赖 R”的测试主线。

## 10. 注意事项

- `.gitignore` 已加入，避免提交缓存和虚拟环境。
- 若遇到系统把 `.git` 挂只读的问题，不要在该目录直接硬改；可在可写目录重新 clone。
- 安全：不要在日志/文档中记录 token、密码、私钥。
- `scripts/package_smco_evo_results.py` 会重建 `result/smco-evo/`；如果报告统计异常，优先检查脚本统计口径，再重跑打包。
