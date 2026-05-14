# 本次重跑结果可视化说明

日期：2026-05-14

## 1. 范围

- `table1`：`result/comparison-rerun-2026-05-13/`
- `table2(d=200)`：`result/comparison-rerun-2026-05-13-d200/`

本说明只基于本次重跑结果，不引入论文原表指标。

## 2. 图表说明

- 热力图：
  - 横轴是任务（函数 + `max/min`）
  - 纵轴是算法
  - 单元格数字是原始指标值
  - 颜色按 `log10(metric)` 编码，方便同时看数量级差异
- 平均柱状图：
  - 使用几何平均汇总算法在全部任务上的指标表现
  - `rMSE` / `AE99` / `time` 都是越低越好

## 3. 算法汇总

### Table 1：rMSE 几何平均排序

| algo | gm |
| --- | --- |
| SMCO | 6.25e-07 |
| SMCO_R | 6.47e-07 |
| SMCO_BR | 8.247e-07 |
| SignGD | 9.398e-05 |
| BOBYQA | 0.03308 |
| optimLBFGS | 2.178 |
| GD | 2.599 |
| ADAM | 3.509 |
| SPSA | 4.448 |

### Table 1：time 几何平均排序

| algo | gm |
| --- | --- |
| optimLBFGS | 0.001817 |
| SPSA | 0.008224 |
| GD | 0.01121 |
| ADAM | 0.01804 |
| SignGD | 0.02515 |
| SMCO_R | 0.03084 |
| SMCO | 0.03099 |
| BOBYQA | 0.03954 |
| SMCO_BR | 0.06119 |

### Table 2(d=200)：rMSE 几何平均排序

| algo | gm |
| --- | --- |
| GenSA | 4.613e-06 |
| SMCO_BR | 0.1801 |
| SMCO_R | 0.1945 |
| SA | 0.2799 |
| SMCO | 2.229 |
| PSO | 26.7 |
| DEoptim | 73.91 |
| GA | 236.7 |

### Table 2(d=200)：time 几何平均排序

| algo | gm |
| --- | --- |
| PSO | 0.3857 |
| GA | 1.037 |
| GenSA | 15.76 |
| SMCO_R | 40.22 |
| SMCO | 42.08 |
| SMCO_BR | 81.26 |
| DEoptim | 161.1 |
| SA | 402.1 |

## 4. 产出图表

- `figures/rerun-only/table1_rmse_heatmap.png`
- `figures/rerun-only/table1_ae99_heatmap.png`
- `figures/rerun-only/table1_time_heatmap.png`
- `figures/rerun-only/table1_rmse_gm.png`
- `figures/rerun-only/table1_time_gm.png`
- `figures/rerun-only/table2_d200_rmse_heatmap.png`
- `figures/rerun-only/table2_d200_ae99_heatmap.png`
- `figures/rerun-only/table2_d200_time_heatmap.png`
- `figures/rerun-only/table2_d200_rmse_gm.png`
- `figures/rerun-only/table2_d200_time_gm.png`
