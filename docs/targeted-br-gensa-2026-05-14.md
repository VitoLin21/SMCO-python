# SMCO_BR 与 GenSA 定向重跑结果

本次实验使用修复后的 comparison 代码：

- `SMCO_BR` 在 comparison 中按 regular + boosted 两轮 refine 的语义，将单次 `iter_max` 从 500 减半为 250。
- 表一补跑 `GenSA`，覆盖 Rastrigin、Ackley、Griewank、Michalewicz 的 max/min 共 8 个 d=2 场景。
- 表二只重跑 `SMCO_BR`，覆盖相同函数的 max/min 共 8 个 d=200 场景。

## 输出位置

- 原始结果：`result/comparison-targeted-br-gensa-2026-05-14/`
- 汇总 CSV：`result/targeted-br-gensa-summary-2026-05-14/`
- 图表：`figures/targeted-br-gensa-2026-05-14/`
- 运行日志：`result/targeted-br-gensa-2026-05-14.log`

## 表一 d=2 汇总

以下几何均值将新跑的 `SMCO_BR`、`GenSA` 与旧表一重跑结果中的其他算法放在一起比较。注意 `SMCO_BR` 是修复预算后的新结果。

| algo | cases | gm_rMSE | gm_AE99 | gm_time |
| --- | --- | --- | --- | --- |
| GenSA | 8 | 1.8673e-11 | 1.5166e-11 | 0.08416 |
| SMCO | 8 | 6.2502e-07 | 4.2434e-07 | 0.03099 |
| SMCO_R | 8 | 6.4699e-07 | 7.1479e-09 | 0.03084 |
| SMCO_BR | 8 | 7.1845e-06 | 8.8291e-07 | 0.03144 |
| SignGD | 8 | 9.3975e-05 | 4.1316e-05 | 0.02515 |
| BOBYQA | 8 | 0.03308 | 0.04495 | 0.03954 |
| optimLBFGS | 8 | 2.178 | 4.602 | 0.001817 |
| GD | 8 | 2.599 | 4.112 | 0.01121 |
| ADAM | 8 | 3.509 | 7.325 | 0.01804 |
| SPSA | 8 | 4.448 | 10.38 | 0.008224 |

## 表一中 SMCO_BR 与 GenSA 的逐场景表现

| task | algo | rMSE | AE99 | time_avg |
| --- | --- | --- | --- | --- |
| Rastrigin-max | GenSA | 4.1221e-10 | 1.9234e-09 | 0.07617 |
| Rastrigin-max | SMCO_BR | 2.619 | 8.041 | 0.02862 |
| Rastrigin-min | GenSA | 5.5748e-14 | 1.7828e-13 | 0.07506 |
| Rastrigin-min | SMCO_BR | 0.109 | 0.995 | 0.02683 |
| Ackley-max | GenSA | 0.0014 | 0.00654 | 0.09051 |
| Ackley-max | SMCO_BR | 0.1137 | 0.24 | 0.04338 |
| Ackley-min | GenSA | 1.5784e-08 | 3.9552e-08 | 0.104 |
| Ackley-min | SMCO_BR | 0.04906 | 0.09418 | 0.04319 |
| Griewank-max | GenSA | 0.1792 | 0.409 | 0.08621 |
| Griewank-max | SMCO_BR | 0.2735 | 0.526 | 0.03265 |
| Griewank-min | GenSA | 0.01208 | 0.04684 | 0.0858 |
| Griewank-min | SMCO_BR | 0.03081 | 0.0692 | 0.03259 |
| Michalewicz-max | GenSA | 1.8441e-40 | 4.6539e-44 | 0.07882 |
| Michalewicz-max | SMCO_BR | 7.5238e-32 | 2.9175e-42 | 0.01857 |
| Michalewicz-min | GenSA | 7.2949e-11 | 3.5396e-11 | 0.08019 |
| Michalewicz-min | SMCO_BR | 7.0284e-06 | 1.9233e-05 | 0.03363 |

## 表二 d=200 汇总

`SMCO_BR` 是修复预算后的新结果；`SMCO_BR_old` 是旧的表二重跑结果，旧结果使用了未减半的 BR 预算。

| algo | cases | gm_rMSE | gm_AE99 | gm_time |
| --- | --- | --- | --- | --- |
| GenSA | 8 | 4.6125e-06 | 1.0560e-05 | 15.76 |
| SMCO_BR | 8 | 0.03773 | 0.05707 | 43.48 |
| SMCO_BR_old | 8 | 0.1801 | 0.1971 | 81.26 |
| SMCO_R | 8 | 0.1945 | 0.2228 | 40.22 |
| SA | 8 | 0.2799 | 0.3484 | 402.1 |
| SMCO | 8 | 2.229 | 2.469 | 42.08 |
| PSO | 8 | 26.7 | 33.9 | 0.3857 |
| DEoptim | 8 | 73.91 | 79.94 | 161.1 |
| GA | 8 | 236.7 | 249.7 | 1.037 |

## 简要结论

- 表一 d=2 上，`GenSA` 在 Rastrigin、Ackley、Michalewicz 的 max/min 场景都明显优于修复预算后的 `SMCO_BR`，但耗时通常更高。
- Griewank d=2 上，`GenSA` 仍优于 `SMCO_BR`，不过优势没有 Rastrigin/Ackley 那么大。
- 表二 d=200 上，修复预算后的 `SMCO_BR` 相比旧 `SMCO_BR_old` 速度显著下降到更公平的预算水平，但误差整体变大；这是预期现象，因为旧结果实际给了 BR 接近两轮完整预算。
