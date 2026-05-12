# Comparison 子系统代码审查记录

日期：2026-05-12

范围：`src/comparison/*`、`scripts/run_experiment.py`、`tests/test_comparison.py`

结论：SMCO 主体迁移基本可用，但后续新增的 comparison/benchmark 对照子系统存在明显语义错误。当前 `97 passed` 不能证明对照实验结果可信。`result/comparison/*` 与相关图表大概率需要在修复后重新生成。

## 1. 问题总览

本次审查发现 4 个高优先级问题：

1. `to_maximize` 参数实际上没有生效，`max/min` 两套实验会跑出同样结果。
2. 多个局部优化器在 `maximize=True` 时会返回更差的点，而不是更优点。
3. 至少两个 SciPy 包装器在 `maximize=False` 时返回值符号错误。
4. 现有测试没有覆盖上述语义错误，导致错误实现也能全量通过测试。

---

## 2. 详细问题

### 问题 A：`to_maximize` 未参与实际计算

受影响文件：

- [src/comparison/run_comparison.py](/amax/math/code/SMCO/src/comparison/run_comparison.py:43)
- [scripts/run_experiment.py](/amax/math/code/SMCO/scripts/run_experiment.py:75)

现象：

- `run_comparison(..., to_maximize=True, ...)`
- `run_comparison(..., to_maximize=False, ...)`

在同一随机种子下，会得到完全相同的 `fopt_algo` 和 `best_opt`。

原因：

- `assign_config()` 返回的 `config.f` 已经是统一成“最大化”语义后的目标函数。
- `run_comparison()` 内部没有根据 `to_maximize` 去切换目标方向、真值、误差定义或结果解释。
- `to_maximize` 最终只是被写入 `ComparisonResult.to_maximize`，没有影响实际优化流程。

影响：

- `table1_*_max` 与 `table1_*_min` 文件名不同，但内容语义上并未区分“最大化/最小化”。
- `table2_*_max` 与 `table2_*_min` 同样失真。
- 论文表格复现结果不可信。

最小复现：

```python
from comparison.run_comparison import run_comparison
import comparison.methods  # noqa

r1 = run_comparison("Rastrigin", 2, True, ["SMCO_R"], 2, {"iter_max": 30, "n_starts": 2}, seed=123)
r2 = run_comparison("Rastrigin", 2, False, ["SMCO_R"], 2, {"iter_max": 30, "n_starts": 2}, seed=123)

print(r1.fopt_algo)
print(r2.fopt_algo)
print(r1.best_opt, r2.best_opt)
```

本地复现结果：

- `r1.fopt_algo == r2.fopt_algo`
- `r1.best_opt == r2.best_opt`

修复建议：

1. 先决定 comparison 层到底要支持什么语义：
   - 方案 1：始终只比较“最大化后的统一目标”，删除 `to_maximize` 参数及相关文件命名。
   - 方案 2：同时支持 max/min，对 `assign_config()` 之外再增加方向控制层。
2. 如果保留 `to_maximize`：
   - 明确 `f` 应该传原始函数还是统一变换后的函数。
   - 明确 `best_opt`、`AE`、`rMSE` 应在原始标尺还是统一标尺上计算。
3. 修复后新增测试：
   - 同 seed 下，`to_maximize=True/False` 应产生不同且可解释的结果。
   - 输出文件名与内部语义必须一致。

---

### 问题 B：GD / SignGD / ADAM / SPSA 在 `maximize=True` 时按“最小值”更新最优解

受影响文件：

- [src/comparison/methods/gd.py](/amax/math/code/SMCO/src/comparison/methods/gd.py:18)
- [src/comparison/methods/sign_gd.py](/amax/math/code/SMCO/src/comparison/methods/sign_gd.py:10)
- [src/comparison/methods/adam.py](/amax/math/code/SMCO/src/comparison/methods/adam.py:10)
- [src/comparison/methods/spsa.py](/amax/math/code/SMCO/src/comparison/methods/spsa.py:9)

现象：

- 这些方法都通过 `sign = -1.0 if maximize else 1.0` 改变了步进方向。
- 但它们记录 `best_value` 时统一使用：

```python
if current_value < best_value:
    best_value = current_value
```

这在 `maximize=True` 时是错误的，应比较 `>`。

影响：

- 算法虽然朝“最大化”方向移动，但最终保存的是更差的目标值。
- comparison 结果会系统性低估这些方法的性能。

最小复现：

```python
import numpy as np
from comparison.methods.gd import gradient_descent

def neg_sphere(x):
    return float(-np.sum(x**2))

bl = np.array([-1.0, -1.0])
bu = np.array([1.0, 1.0])
starts = np.array([[0.1, 0.1], [0.5, 0.5]])
res = gradient_descent(neg_sphere, bl, bu, starts, maximize=True, max_iter=50)

print(res.x_optimal)
print(res.f_optimal)
```

本地复现结果：

- 返回 `x=[0.4, 0.4]`
- 返回 `f_optimal=-0.319999...`

而正确最大化目标 `-||x||^2` 应更接近原点，最优值应更接近 `0`。

修复建议：

1. 统一抽象“更优”的比较逻辑：

```python
is_better = current_value > best_value if maximize else current_value < best_value
```

2. 所有局部方法统一修复：
   - `GD`
   - `SignGD`
   - `ADAM`
   - `SPSA`
3. 增加针对 `maximize=True` 的回归测试：
   - 目标 `-sum(x**2)`，结果应靠近原点。
   - `f_optimal` 应大于某个阈值，例如 `>-0.05`。

---

### 问题 C：Nelder-Mead / SA 在 `maximize=False` 时返回值符号错误

受影响文件：

- [src/comparison/methods/nelder_mead.py](/amax/math/code/SMCO/src/comparison/methods/nelder_mead.py:8)
- [src/comparison/methods/sa.py](/amax/math/code/SMCO/src/comparison/methods/sa.py:8)

现象：

两处都写成了：

```python
val = float(-target(...))
```

但当 `maximize=False` 时，`target == f`，因此这里会把真实最小值 `f(x*)` 取负返回。

影响：

- 最小化问题返回值符号反了。
- 后续 `best_opt`、`AE`、`rMSE` 都可能被污染。
- 测试里只断言“小于某阈值”，负值也能通过，因此问题未被拦住。

最小复现：

```python
import numpy as np
from comparison.methods.sa import simulated_annealing

def sphere(x):
    return float(np.sum(x**2))

bl = np.array([-5.0, -5.0])
bu = np.array([5.0, 5.0])
res = simulated_annealing(sphere, bl, bu, maximize=False, max_iter=20, seed=42)

print(res.x_optimal)
print(res.f_optimal)
```

本地复现结果：

- `x_optimal` 接近原点
- `f_optimal = -3.49e-17`

这里返回值应接近 `0` 的非负数，而不是负数。

修复建议：

1. 统一返回值恢复逻辑：

```python
val = float(-res.fun if maximize else res.fun)
```

2. 检查所有基于 `target = -f if maximize else f` 的包装器，确认恢复逻辑一致：
   - `optimLBFGS`
   - `optimNM`
   - `BOBYQA`
   - `GenSA`
   - `SA`
   - `DEoptim`
   - `GA`
   - `PSO`
3. 对每个包装器补一条最小化与最大化的符号测试。

---

### 问题 D：测试覆盖方向错了，当前 `97 passed` 不能证明 comparison 结果正确

受影响文件：

- [tests/test_comparison.py](/amax/math/code/SMCO/tests/test_comparison.py:1)

具体问题：

1. `GD` 的 `maximize=True` 测试断言写反了：

```python
assert result.f_optimal < -0.01
```

这恰好把错误行为当成正确行为。

2. 多个最小化测试只断言：

```python
assert result.f_optimal < 1.0
```

如果返回错误负值，测试依然通过。

3. `run_comparison()` 只覆盖了 `to_maximize=True`，没有验证 `True/False` 的差异。

影响：

- 现有测试主要证明“能运行”，没有证明“语义正确”。

修复建议：

1. 为每个方法分别测：
   - `maximize=False` 时，返回值与目标语义一致。
   - `maximize=True` 时，返回值与目标语义一致。
2. 对明显有解析解的目标增加更强断言：
   - `sphere(x)=sum(x^2)` 最小值应接近 `0` 且非负。
   - `neg_sphere(x)=-sum(x^2)` 最大值应接近 `0` 且不应更负。
3. 为 `run_comparison()` 增加语义测试：
   - `to_maximize=True/False` 不应静默得到相同矩阵。
   - 若设计上只保留单一方向，则删除该参数并同步修改脚本和结果命名。

---

## 3. 修复优先级建议

建议按这个顺序修：

1. 明确 comparison 层的目标方向设计：
   - 保留 `to_maximize`
   - 或删除 `to_maximize`
2. 修复所有优化器的“更优比较”与“返回值恢复”逻辑。
3. 重写 `tests/test_comparison.py` 中与方向语义相关的断言。
4. 重新运行：
   - `pytest -q`
   - `scripts/run_experiment.py --quick`
5. 确认 quick 结果正常后，再重跑：
   - `table1`
   - `table2`
6. 删除旧结果并重建：
   - `result/comparison/*`
   - `figures/*`

---

## 4. 建议交给修复代理的任务说明

可直接给其他代理这段要求：

1. 修复 `src/comparison` 中 maximize/minimize 语义不一致的问题。
2. 修复 `run_comparison()` 中 `to_maximize` 未生效的问题，或者删除该参数并同步清理脚本。
3. 修复 `GD`、`SignGD`、`ADAM`、`SPSA` 在 `maximize=True` 时的最优值更新逻辑。
4. 修复 `optimNM`、`SA` 的返回值符号错误，并检查所有包装器返回逻辑是否统一。
5. 重写 `tests/test_comparison.py` 的方向语义测试，确保这些问题能被测试拦住。
6. 修复后先跑 `pytest -q`，再跑 `scripts/run_experiment.py --quick`，确认 max/min 输出语义正确。
7. 最后再决定是否重跑完整 benchmark 并更新 `result/` 与 `figures/`。

---

## 5. 当前验证状态

本地已执行：

```bash
.venv/bin/python -m pytest -q
```

结果：

- `97 passed`

说明：

- 当前测试集没有覆盖 comparison 层的关键语义错误。
- 修复后必须补测试，而不是只追求继续全绿。
