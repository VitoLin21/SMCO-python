# SMCO Python（中文版）

本项目是对 `wayne-y-gao/SMCO`（R 版本）的 Python 迁移实现。

当前实现以 R `v1.1.0` 的 API 形态为主，同时保留了面向论文复现实验的 `v1.0.0` 参考路径。`vendor/SMCO_R` 中的 R 文件仅用于对照，不参与 Python 测试执行。

## 安装

推荐统一使用 `uv + Python 3.12`：

```bash
uv python install 3.12
uv venv .venv --python 3.12
uv pip install -e ".[test]"
```

后续命令统一使用本地虚拟环境解释器：

```bash
.venv/bin/python -m pytest -q
```

## 快速开始

```python
import numpy as np
from smco import smco_r

def objective(x):
    return -float(np.sum(x ** 2))

result = smco_r(
    objective,
    bounds_lower=[-5, -5],
    bounds_upper=[5, 5],
    iter_max=200,
    seed=123,
)

print(result.best_result.x_optimal)
print(result.best_result.f_optimal)
```

说明：SMCO 默认执行“最大化”。若要做最小化，请传入目标函数的相反数（`-f(x)`）。

## 算法入口

- `smco(...)`：基础单阶段搜索
- `smco_r(...)`：两阶段 refine 搜索
- `smco_br(...)`：regular + boost 搜索
- `smco_multi(...)`：多起点全控制入口

## Benchmark 示例

```python
from smco import run_benchmark

run = run_benchmark(
    "Rastrigin",
    dim=2,
    variant="smco_r",
    repetitions=3,
    iter_max=100,
    n_starts=5,
    seed=123,
)

print(run.best_x)
print(run.best_value)
```

说明：经典 benchmark 函数大多原始定义为最小化问题。项目内部已做包装，将其转换为 SMCO 的最大化目标。

- `BenchmarkConfig.known_best_value`：原 benchmark 标尺（通常是最小值）
- `BenchmarkConfig.known_best_objective`：转换到 SMCO 最大化后的标尺（可用时）

## 论文冒烟运行（Smoke）

```python
from smco import run_paper_smoke

runs = run_paper_smoke(iter_max=100, n_starts=5, seed=123)
print(runs.keys())
```

该入口仅用于快速验证流程。论文级别的大规模实验需要手动提高维度、重复次数和迭代预算。

## 测试

```bash
.venv/bin/python -m pytest -q
```

## 上游 R 参考代码

R 参考源码位于 `vendor/SMCO_R`：

- `main/`：上游当前主线实现（约等于 v1.1.0 语义）
- `v1.0.0/`：论文复现相关的历史实现与脚本

注意：上游 `main` README 提到的 `benchmark_version_comparison.R` 在归档时未从公开仓库获取到。
