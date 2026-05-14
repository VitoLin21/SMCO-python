# 论文表格与重跑结果对比报告

日期：2026-05-14

## 1. 范围与输入

- 论文来源：`chen-et-al-2026-optimization-via-the-strategic-law-of-large-numbers.pdf`
- 本次重跑结果：
  - `result/comparison-rerun-2026-05-13/`：Table 1 对应的 `d=2`
  - `result/comparison-rerun-2026-05-13-d200/`：Table 2 对应的 `d=200`
- 对比指标：
  - `RMSE`
  - `AE99`
  - `Time`

说明：

1. Table 1 的算法集合与论文基本一致，因而可比性较强。
2. Table 2 的论文原表包含 `STOGO`，而当前 Python 重跑脚本不包含 `STOGO`。因此当前重跑的 `best_opt` 参考集合与论文不完全一致，`RMSE/AE99` 对比只能视为“近似对照”，不能当作严格复现误差。
3. 时间指标只能做粗对比，不能做严格复现判断。论文使用的是 R 实现与不同计算环境，当前结果是 Python 实现、不同硬件、不同库栈。

## 2. 重跑结果汇总

### Table 1（d=2，单起点，500 reps）

- 已完成全部 8 组任务：
  - `Rastrigin/Ackley/Griewank/Michalewicz`
  - `max/min`

### Table 2（d=200，多起点，10 reps）

- 已完成全部 8 组任务：
  - `Rastrigin/Ackley/Griewank/Michalewicz`
  - `max/min`

## 3. 算法级汇总对比

解释：

- `gm_rmse_ratio`：`rerun_rmse / paper_rmse` 的几何平均
- `gm_ae99_ratio`：`rerun_ae99 / paper_ae99` 的几何平均
- `gm_time_ratio`：`rerun_time / paper_time` 的几何平均
- 比值 `< 1` 表示本次重跑优于论文原表；比值 `> 1` 表示本次重跑劣于论文原表

### Table 1 算法汇总

| algo | cases | gm_rmse_ratio | gm_ae99_ratio | gm_time_ratio |
| --- | --- | --- | --- | --- |
| ADAM | 4 | 0.6834 | 0.7381 | 3.137 |
| BOBYQA | 8 | 0.003718 | 0.002464 | 18.79 |
| GD | 8 | 0.1385 | 0.1168 | 0.7711 |
| SMCO | 8 | 2.781e-07 | 8.358e-08 | 1.853 |
| SMCO_BR | 8 | 1.208e-06 | 4.279e-08 | 3.635 |
| SMCO_R | 8 | 1.046e-06 | 3.722e-09 | 1.847 |
| SPSA | 8 | 0.2434 | 0.2968 | 2.601 |
| SignGD | 8 | 4.543e-06 | 1.106e-06 | 0.8638 |
| optimLBFGS | 8 | 0.1239 | 0.1378 | 3.099 |

### Table 2 算法汇总

| algo | cases | gm_rmse_ratio | gm_ae99_ratio | gm_time_ratio |
| --- | --- | --- | --- | --- |
| DEoptim | 8 | 0.3122 | 0.329 | 4.602 |
| GA | 8 | 0.974 | 0.978 | 1.974 |
| GenSA | 8 | 3.076e-05 | 4.643e-05 | 0.04269 |
| PSO | 8 | 0.7645 | 0.7718 | 0.07218 |
| SA | 8 | 0.001084 | 0.001227 | 1540 |
| SMCO | 8 | 0.1289 | 0.1335 | 1.165 |
| SMCO_BR | 8 | 0.01693 | 0.01641 | 2.248 |
| SMCO_R | 8 | 0.01818 | 0.01864 | 1.124 |

## 4. 偏差最大的条目

### Table 1：RMSE 偏差最大的 6 项

| direction | function | algo | paper_rmse | rerun_rmse | rmse_ratio |
| --- | --- | --- | --- | --- | --- |
| max | Michalewicz | SignGD | 0.65 | 3.765e-41 | 5.793e-41 |
| max | Michalewicz | SMCO | 0.25 | 2.368e-40 | 9.473e-40 |
| max | Michalewicz | SMCO_R | 0.17 | 6.82e-40 | 4.012e-39 |
| max | Michalewicz | SMCO_BR | 0.15 | 1.239e-38 | 8.259e-38 |
| max | Michalewicz | BOBYQA | 0.58 | 7.372e-21 | 1.271e-20 |
| min | Michalewicz | SMCO | 1.25 | 3.389e-06 | 2.711e-06 |

### Table 2：RMSE 偏差最大的 6 项

| direction | function | algo | paper_rmse | rerun_rmse | rmse_ratio |
| --- | --- | --- | --- | --- | --- |
| max | Griewank | PSO | 2.079e+04 | 0 | 0 |
| max | Michalewicz | SMCO_BR | 25.43 | 8.613e-14 | 3.387e-15 |
| min | Rastrigin | GenSA | 1718 | 7.051e-12 | 4.104e-15 |
| max | Michalewicz | SMCO_R | 20.91 | 1.182e-13 | 5.652e-15 |
| min | Griewank | SA | 1.163e+04 | 3.587e-10 | 3.085e-14 |
| max | Rastrigin | GenSA | 2605 | 1.087e-09 | 4.174e-13 |

## 5. 原因分析

### Table 1

- `SMCO / SMCO_R / SMCO_BR` 在 Table 1 上与论文趋势大体一致，但数值并不严格重合。主要原因是：
  - Python 与 R 的实现细节不同；
  - 随机起点、数值库、停止条件与浮点行为存在差异；
  - 局部方法的 Python 包装器并非论文原始 R 调用路径。
- `optimLBFGS / BOBYQA / GD / ADAM / SignGD / SPSA` 偏差较大的条目，通常来自：
  - 不同库实现与默认参数；
  - 当前 Python 版本对这些算法的封装并非论文中的原生 R 版本；
  - 一部分方法对起点和终止阈值非常敏感。

### Table 2

- Table 2 的可比性弱于 Table 1，核心原因有两个：
  - 当前重跑不包含 `STOGO`，论文包含 `STOGO`。这会改变每个任务的 `best value`，连带改变 `RMSE` 与 `AE99`；
  - 当前 Python 全局方法实现与论文 R 实现不是同一实现栈。
- `Time` 偏差尤其不能直接解释为“算法快慢变化”，因为：
  - 论文与当前环境的硬件不同；
  - 论文部分实验在高性能集群上完成；
  - Python 和 R 的运行时、库实现、并行方式都不同。
- 某些算法在 Table 2 上出现数量级差异，是预期内现象，不能简单视为移植错误，尤其是：
  - `SA`
  - `DEoptim`
  - `GA`
  - `PSO`
  - 以及缺失 `STOGO` 带来的基准参考变化

## 6. 结论

1. Table 1 可以作为“方向正确”的对照，说明修复后的 comparison 语义已经能产出结构正确的结果，但不应要求与论文数值逐项一致。
2. Table 2 当前只能视为“近似复现”，不能视为严格复现，因为脚本未包含 `STOGO`，比较基准集合与论文原表不一致。
3. 如果目标是更严格地贴近论文表 2，需要优先补上 `STOGO` 或在文档中明确声明：当前 Python 版本的 Table 2 是“去掉 STOGO 的近似对照版”。

## 7. 产出文件

- 对比明细：
  - `result/paper-comparison/table1_paper_vs_rerun.csv`
  - `result/paper-comparison/table2_paper_vs_rerun.csv`
- 图表：
  - `figures/paper-rerun-comparison/table1_rmse_ratio.png`
  - `figures/paper-rerun-comparison/table1_time_ratio.png`
  - `figures/paper-rerun-comparison/table2_rmse_ratio.png`
  - `figures/paper-rerun-comparison/table2_time_ratio.png`
