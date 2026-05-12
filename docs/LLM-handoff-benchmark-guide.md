# SMCO Python 迁移交接文档（给后续 LLM 代理）

## 1. 项目现状

- 仓库：`https://github.com/VitoLin21/SMCO-python`
- 当前分支：`main`
- 当前基线提交：`0a35d65`（`docs+code: 中文化 README 并补充关键中文注释`）
- 本地已对齐远端，`git status` 为 clean。

本仓库是对 R 版本 `wayne-y-gao/SMCO` 的 Python 迁移，目标是：

- 主 API 对齐 `v1.1.0` 风格；
- 保留 `v1.0.0` 论文复现参考路径（以 vendored R 源码形式）；
- 测试不依赖 R 运行时。

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

当前测试规模：`70 passed`。

## 3. 代码结构

- 核心优化器：
  - `src/smco/optimizer.py`
- 结果类型：
  - `src/smco/results.py`
- benchmark 函数与配置：
  - `src/smco/test_functions.py`
- benchmark runner：
  - `src/smco/benchmark.py`
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
- `smco_multi(...)`

说明：

- 默认是“最大化”。
- 若要做最小化，传入负目标（benchmark 配置层已做包装）。

## 5. benchmark 入口

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

## 8. 后续代理建议执行流程（论文 benchmark）

1. 固定环境：确保只使用 `.venv/bin/python`。
2. 先跑全量测试，确保基线一致：
   ```bash
   .venv/bin/python -m pytest -q
   ```
3. 从小规模 benchmark 冒烟开始（低维、低 repetitions）。
4. 再扩展到论文维度/重复次数，分批运行并落盘结果（CSV/JSON）。
5. 如需和 R 数值对照，新增可选脚本，不要破坏当前“不依赖 R”的测试主线。

## 9. 注意事项

- `.gitignore` 已加入，避免提交缓存和虚拟环境。
- 若遇到系统把 `.git` 挂只读的问题，不要在该目录直接硬改；可在可写目录重新 clone。
- 安全：不要在日志/文档中记录 token、密码、私钥。

