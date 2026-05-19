# SMCO — 安装与使用指南

Strategic Monte Carlo Optimization (SMCO) 的 Python 实现，包含 SMCO 系列算法及 14 种对比优化算法。

## 安装

```bash
# 从源码安装（开发模式，推荐开发时使用）
# git clone <repo-url>
cd ~/code/SMCO
pip install -e .

# 或者直接安装
pip install .
```

**依赖**：Python >= 3.10，numpy >= 1.23，scipy >= 1.10。

安装完成后即可在任意项目中导入使用：

```python
import smco                    # SMCO 系列算法
from comparison import ...     # 对比优化算法
```

---

## SMCO 算法

### 三种变体

| 函数 | 说明 |
|------|------|
| `smco.smco()` | 基础版：单阶段搜索，无 refine |
| `smco.smco_r()` | 精修版：两阶段搜索（常规 + refine） |
| `smco.smco_br()` | 增强版：比较 regular 与 boosted 两条路径，取更优者 |

### 快速上手

```python
import numpy as np
import smco

# 定义目标函数（SMCO 默认最大化）
def objective(x):
    return -np.sum(x**2)

# 优化
result = smco.smco_r(
    objective,
    bounds_lower=[-5, -5],
    bounds_upper=[5, 5],
)

# 获取结果
print(f"最优值: {result.best_result.f_optimal}")
print(f"最优点: {result.best_result.x_optimal}")
```

### 公共接口

所有三种变体的调用签名相同：

```python
smco.smco(f, bounds_lower, bounds_upper, start_points=None, **kwargs) -> SMCOResult
smco.smco_r(f, bounds_lower, bounds_upper, start_points=None, **kwargs) -> SMCOResult
smco.smco_br(f, bounds_lower, bounds_upper, start_points=None, iter_boost=1000, **kwargs) -> SMCOResult
```

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `f` | `Callable[[np.ndarray], float]` | 目标函数（默认最大化） |
| `bounds_lower` | array-like (1D) | 搜索空间下界 |
| `bounds_upper` | array-like (1D) | 搜索空间上界 |
| `start_points` | array-like (2D) or None | 自定义起点；None 时自动用 Sobol 序列生成 |
| `iter_boost` | int | 仅 `smco_br`，boost 路径的额外迭代数，默认 1000 |
| `**kwargs` | | 控制参数，见下表 |

**控制参数**（通过 `**kwargs` 传入）：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `iter_max` | int | 500 | 最大迭代次数 |
| `n_starts` | int | `max(5, round(sqrt(dim)))` | 多起点数量（仅自动生成起点时生效） |
| `bounds_buffer` | float | 0.05 | 边界外扩比例 |
| `buffer_rand` | bool | False | 是否使用随机边界扰动 |
| `tol_conv` | float | 1e-8 | 收敛判定容差 |
| `refine_search` | bool | True | 是否启用精修阶段（`smco` 中被强制为 False） |
| `refine_ratio` | float | 0.5 | 精修阶段迭代占比（仅 `smco_r`、`smco_br`） |
| `partial_option` | str | "center" | 偏导数计算方式："center" 或 "forward" |
| `use_runmax` | bool | True | 是否追踪运行最优 |
| `use_parallel` | bool | False | 并行计算（预留接口） |
| `verbose` | bool | False | 打印详细信息 |
| `seed` | int or None | 123 | 随机种子 |

### 高级接口

```python
# smco_multi：完全控制所有参数
result = smco.smco_multi(
    objective,
    bounds_lower=[-5, -5, -5],
    bounds_upper=[5, 5, 5],
    start_points=[[1, 2, 3], [-1, 0, 1]],
    opt_control={
        "iter_max": 1000,
        "bounds_buffer": 0.1,
        "refine_ratio": 0.3,
        "seed": 42,
    },
)
```

### 结果结构

```python
@dataclass
class SMCOResult:
    best_result: SingleResult       # 所有起点中的最优结果
    all_results: list[SingleResult] # 每个起点的独立结果
    opt_control: dict               # 实际使用的控制参数
    summary: dict                   # 批次统计信息

@dataclass
class SingleResult:
    x_optimal: np.ndarray           # 最优点
    f_optimal: float                # 最优目标值
    iterations: int                 # 实际迭代次数
    x_runmax: np.ndarray | None     # 运行最优点（use_runmax=True 时）
    f_runmax: float | None          # 运行最优值
```

**summary 字典内容**：

```python
result.summary = {
    "n_starts": 5,
    "mean_iterations": 487.2,
    "std_values": 0.003,           # 各起点目标值标准差
    "endpoints": np.ndarray,       # 各起点最优点组成的矩阵
    "values": np.ndarray,          # 各起点最优值数组
}
```

---

## 对比优化算法

安装 smco 包后，14 种对比算法也可直接调用。

### 算法列表

**局部优化**（需要 `start_points`）：

| 注册名 | 说明 | 额外参数 |
|--------|------|----------|
| `GD` | 梯度下降 | `learning_rate=0.1`, `momentum=0.9`, `tol=1e-6` |
| `SignGD` | 符号梯度下降 | `learning_rate=0.1`, `decay_rate=0.995`, `tol=1e-6` |
| `ADAM` | Adam 优化器 | `learning_rate=0.01`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`, `tol=1e-6` |
| `SPSA` | 同时扰动随机逼近 | `A=50.0`, `a=0.10`, `c=1e-3`, `alpha=0.602`, `gamma=0.101`, `tol=1e-7` |
| `optimLBFGS` | L-BFGS-B | `tol=1e-6` |
| `optimNM` | Nelder-Mead | `tol=1e-6` |
| `BOBYQA` | COBYLA (近似) | `tol=1e-6` |

**全局优化**（`start_points` 可选）：

| 注册名 | 说明 | 额外参数 |
|--------|------|----------|
| `GenSA` | 模拟退火 (dual_annealing) | — |
| `SA` | 基于跳跃的模拟退火 (basinhopping) | — |
| `DEoptim` | 差分进化 | `popsize=15` |
| `GA` | 遗传算法 | `pop_size=50`, `mutation_rate=0.1`, `crossover_rate=0.8` |
| `PSO` | 粒子群优化 | `swarm_size=40`, `w=0.729`, `c1=1.49445`, `c2=1.49445` |

### 单独调用对比算法

```python
import numpy as np
from comparison import get_method, METHOD_REGISTRY

# 查看所有可用算法
print(METHOD_REGISTRY.keys())
# dict_keys(['GD', 'SignGD', 'ADAM', 'SPSA', 'optimLBFGS', 'optimNM', 'BOBYQA',
#            'GenSA', 'SA', 'DEoptim', 'GA', 'PSO'])

# 获取并调用某个算法
optimize = get_method("GenSA")
result = optimize(
    f=lambda x: -np.sum(x**2),   # 注意：对比算法默认最小化，要最大化传 maximize=True
    bounds_lower=np.array([-5.0, -5.0]),
    bounds_upper=np.array([5.0, 5.0]),
    maximize=True,
    max_iter=500,
    seed=42,
)
print(f"最优值: {result.f_optimal}")
print(f"最优点: {result.x_optimal}")
print(f"迭代数: {result.iterations}")
```

**对比算法公共签名**：

```python
def method(f, bounds_lower, bounds_upper, start_points=None, maximize=False, max_iter=500, ..., seed=None) -> OptimizerResult
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `f` | `Callable[[np.ndarray], float]` | 目标函数 |
| `bounds_lower` | `np.ndarray` (1D) | 下界 |
| `bounds_upper` | `np.ndarray` (1D) | 上界 |
| `start_points` | `np.ndarray` (2D) or None | 起点（局部算法必填） |
| `maximize` | bool | 是否最大化，默认 False（最小化） |
| `max_iter` | int | 最大迭代次数，默认 500 |
| `seed` | int or None | 随机种子 |

**返回值**：

```python
@dataclass
class OptimizerResult:
    x_optimal: np.ndarray   # 最优点
    f_optimal: float         # 最优值
    iterations: int          # 迭代次数
```

> **注意**：SMCO 系列默认**最大化**，而对比算法默认**最小化**。使用 `maximize=True` 可切换为最大化。

### 局部算法使用示例（需提供起点）

```python
from comparison import get_method
import numpy as np

optimize = get_method("optimLBFGS")
result = optimize(
    f=lambda x: np.sum(x**2),     # 最小化
    bounds_lower=np.array([-5.0, -5.0]),
    bounds_upper=np.array([5.0, 5.0]),
    start_points=np.array([[1.0, 2.0], [-1.0, 3.0]]),
    max_iter=500,
)
```

---

## 批量对比实验

使用 `run_comparison` 可一键运行多算法对比实验：

```python
from comparison import run_comparison

result = run_comparison(
    name_config="Rastrigin",    # 测试函数名称
    dim_config=5,               # 维度
    to_maximize=True,           # 是否最大化
    algo_names=["SMCO_R", "SMCO_BR", "GenSA", "DEoptim", "PSO", "GA"],
    n_replications=10,          # 重复次数
    smco_options={"iter_max": 300},
    seed=123,
)

# 查看结果
for algo in result.algo_names:
    s = result.sum_results[algo]
    print(f"{algo}: mean={s['fopt_mean']:.6f}, sd={s['fopt_sd']:.6f}, "
          f"time={s['time_avg']:.3f}s, rMSE={s['rMSE']:.6f}")
```

**`run_comparison` 参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `name_config` | str | 测试函数名称（见下方列表） |
| `dim_config` | int | 维度 |
| `to_maximize` | bool | 优化方向 |
| `algo_names` | list[str] | 算法名称列表 |
| `n_replications` | int | 重复实验次数 |
| `smco_options` | dict or None | SMCO 及各算法的迭代参数 |
| `domain_mod` | bool | 是否做域变换，默认 False |
| `truth_known` | bool | 是否已知理论最优值 |
| `true_opt` | float or None | 理论最优值 |
| `seed` | int | 随机种子 |

### 内置测试函数

通过 `smco.assign_config(name, dim)` 可获取测试函数配置，也可直接在 `run_comparison` 的 `name_config` 中使用：

**可缩放维度**：

| 名称 | 维度 | 最小值 | 搜索范围 |
|------|------|--------|----------|
| `Rastrigin` | 任意 | 0 | [-5.12, 5.12] |
| `Ackley` | 任意 | 0 | [-32.768, 32.768] |
| `Griewank` | 任意 | 0 | [-600, 600] |
| `Rosenbrock` | 任意 | 0 | [-5, 10] |
| `DixonPrice` | 任意 | 0 | [-10, 10] |
| `Zakharov` | 任意 | 0 | [-5, 10] |
| `Qing` | 任意 | 0 | [-500, 500] |
| `Michalewicz` | 任意 | — | [0, π] |

**固定 2D**：

| 名称 | 最小值 | 搜索范围 |
|------|--------|----------|
| `DropWave` | -1.0 | [-5.12, 5.12]² |
| `Shubert` | -186.7309 | [-5.12, 5.12]² |
| `McCormick` | -1.9132 | [-1.5, 4]×[-3, 4] |
| `SixHumpCamel` | -1.0316 | [-3, 3]×[-2, 2] |
| `Eggholder` | -959.6407 | [-512, 512]² |
| `Bukin6` | 0 | [-15, -3]×[-5, 3] |
| `Easom` | -1.0 | [-100, 100]² |
| `Mishra6` | — | [-10, 10]² |

---

## Benchmark 快捷接口

```python
import smco

# 运行单个 benchmark
run = smco.run_benchmark("Rastrigin", dim=5, variant="smco_r", repetitions=10)

# 论文 smoke test（多函数快速验证）
results = smco.run_paper_smoke(variant="smco_r", iter_max=80, n_starts=5)
for name, run in results.items():
    print(f"{name}: {run.best_value:.6f}")
```
