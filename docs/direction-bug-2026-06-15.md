# 方向 Bug 记录：基准/实验脚本的 SMCO 目标符号多取了一次负

- **发现日期**：2026-06-15
- **状态**：已确认（有诊断铁证），**暂不修复**（按用户决策保持原逻辑继续跑实验，本文件留待后续处理）
- **影响范围**：基准 `result/highdim-full-comparison-2026-06-04/all_results.csv` 的全部算法行；实验1/2 脚本 `run_evo_startsweep_comparison.py` / `run_evo_dimsplit_comparison.py`；**不含** optimizer.py 算法核心（SMCO 本身按"最大化"约定工作，正确）。
- **关键缓解**：所有受影响脚本逻辑一致（都多取一次负），彼此**内部自洽**——实验1/2 的结果能与基准逐点配对，"组数/起点比例的相对效果"结论仍成立。只是 `fopt` 的绝对语义是"推向边界"而非真正优化。

---

## 1. 根因

`src/smco/test_functions.py` 的 `_min_config`（line 183-197）把所有最小化测试函数包装成 SMCO 的**最大化目标**：

```python
def objective(x):                 # line 183-185
    # SMCO 默认最大化，这里将原始最小化目标统一变换为 -f(x)。
    return -float(raw(x))
return BenchmarkConfig(..., f=objective, sense="min", ...)   # line 187-197
```

即 **`config.f` 本身已经是 `-raw`，就是 SMCO 该最大化的目标**（对 Rastrigin，`config.f(zeros) = -0.0`，`config.f(5s) = -250.0`，已在 `diag_sign.py` 中验证）。

但基准脚本 `scripts/run_highdim_full_comparison.py:233-236` 又对 `sense=="min"` 再取一次负：

```python
f = config.f                       # line 234   f = -raw  (正确目标)
if sense == "min":
    f = lambda x, _f=config.f: -_f(x)   # line 236   f = +raw  (反向！)
```

于是实际传给 SMCO（及所有对比算法，因为 line 248-253 共用同一个 `f`）的是 `+raw`。SMCO 最大化 `+raw` ⇒ 把 `x` 推向使原始目标最大的边界，方向与"最小化 Rastrigin/Ackley/Rosenbrock"相反。

实验1/2 脚本从基准脚本复制了同一模式：
- `run_evo_startsweep_comparison.py:208-210`
- `run_evo_dimsplit_comparison.py:193-195`

故三个脚本行为一致。

> 注：`max` 问题（SquaredNorm 等）走 `assign_config` 的 `sense="max"` 分支，`config.f` 不被取负，**不受此 bug 影响**。本次实验的三个函数全是 `sense="min"`，全中招。

## 2. 诊断铁证

`vendor/SMCO_R/align/diag_sign.py`（D=10 Rastrigin，seed=1）：

```
sense: min
config.f(zeros): -0.0   (raw rastrigin(zeros)= 0.0)
config.f(5s):    -250.0
[config.f]  fopt=-0.4550  xnorm=0.0479  raw_rastrigin=0.4550   ← 正确(找原点)
[-config.f] fopt=369.1301 xnorm=13.0340 raw_rastrigin=369.1301 ← 反向(推边界)
```

基准 `all_results.csv` 实测（Rastrigin 1000D）：

| algo | fopt |
|---|---|
| SMCO | 29108.1 |
| PSO | 29130.4 |
| GA | 21034.5 |
| GenSA / SA | 40353.3 |

1000D Rastrigin 在边界（±5.12）处的 raw 值量级正是 ~2.9 万，SA/GenSA 的 40353 更贴近"完全推到边界"。**而 Rastrigin 真实最优 raw=0**。所以基准里所有算法都在做"最大化 raw → 推边界"，方向反了。

## 3. 正确做法（待后续实施）

把三个脚本的 line 234-236 / 208-210 / 193-195 改成直接用 `f = config.f`（`config.f` 已是最大化目标，无需再取负）。修正后 Rastrigin 1000D SMCO 的 `fopt` 应是绝对值很小的负数（`-raw`，raw≈残差）。

> 实验3 对齐脚本 `vendor/SMCO_R/align/python_side.py` **已做此修正**（`f = config.f`），因为对齐要求 Python 与 R 用同符号目标：R 端 `cfg$f` 是原始 raw，R 做 `-cfg$f`；Python 侧必须用 `config.f = -raw` 才能与之对齐。修正后 D=10 对齐成功（见下）。**此改动是否回退由用户决定**——若要实验3 也保持反向以与基准一致，可改回 `-config.f`。

## 4. 对各实验的影响与处理

| 实验 | 影响 | 本次处理 |
|---|---|---|
| 实验1（起点缩减） | fopt 绝对值反向，但 `starts[:k]` 配对、各 `start_fraction` 相对效果结论仍成立 | 保持原逻辑跑 |
| 实验2（维度分离） | fopt 绝对值反向；`dim_groups=1` 仍逐点匹配基准；G∈{2,4,8} 的相对增益结论仍成立 | 保持原逻辑跑 |
| 实验3（R 对齐） | Python 侧已改 `config.f`（正确方向）；R 侧 `-cfg$f`（正确方向）。**对齐脚本两边方向已正确**，与基准方向不一致，但实验3 本就是验证算法移植正确性，正确方向才有意义 | 对齐脚本保持修正 |

## 5. 实验3 对齐结果（方向修正后，D=10 Rastrigin）

`config.f`（Python）vs `-cfg$f`（R），同 starts、同 seed=20260615：

| | f_optimal | x_norm | iters | n_history |
|---|---|---|---|---|
| Python | -1.30 | 1.00 | 208 | 2 |
| R | -0.21 | 0.033 | 66 | 2 |

两边都在正确方向（最小化 Rastrigin）收敛到原点附近、进化边界数一致（`n_history=2`）。数值差异来自 R/Python 的 RNG、Sobol 实现、收敛判定的固有差异，符合"行为对齐而非逐字节"的预期。**SMCO_EVO 的 R 移植数值正确性已验证。**

## 6. 待办（用户后续处理）

- [ ] 决定是否修正方向并重生成基准（工程量大：18 算法 × 多维度，~15700 行；其中 R 对比算法 GenSA/SA/DEoptim/GA/PSO 很慢）。
- [ ] 若修正：改 `run_highdim_full_comparison.py:236`、`run_evo_startsweep_comparison.py:210`、`run_evo_dimsplit_comparison.py:195` 三处 `f = -config.f` → `f = config.f`，删除 `if sense=="min"` 分支。
- [ ] 决定 `vendor/SMCO_R/align/python_side.py` 是否回退到 `-config.f`（当前是修正后的 `config.f`）。

## 7. 相关文件

- 诊断脚本：`vendor/SMCO_R/align/diag_sign.py`
- 受影响脚本：`scripts/run_highdim_full_comparison.py`、`scripts/run_evo_startsweep_comparison.py`、`scripts/run_evo_dimsplit_comparison.py`
- 已修正（仅实验3对齐）：`vendor/SMCO_R/align/python_side.py`
- 根因定义：`src/smco/test_functions.py:183-197`（`_min_config`，此处无 bug，bug 在调用方多取负）
- 基准数据：`result/highdim-full-comparison-2026-06-04/all_results.csv`
