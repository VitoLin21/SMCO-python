# SMCO-EVO vs SMCO Comparison Report
Generated: 2026-05-31
Input: result/evo-comparison-2026-05-30

## Experiment 1: Core Comparison

Total EVO wins: 2202, Base wins: 2014, Ties: 224
Significant differences (p<0.05): 14/63

| Dim | EVO wins | Base wins | Ties |
|-----|----------|-----------|------|
| 2D | 665 | 656 | 179 |
| 10D | 1209 | 1146 | 45 |
| 50D | 269 | 181 | 0 |
| 200D | 59 | 31 | 0 |

### Wilcoxon Test Results (p < 0.05 marked with *)

| Pair | Function | Dim | EVO mean | Base mean | EVO wins | p-value |
|------|----------|-----|----------|-----------|----------|---------|
| SMCO_BR_vs_SMCO_BR_EVO | Ackley | 2D | 22.2515 | 22.2527 | 42/100 | 0.8773 |
| SMCO_BR_vs_SMCO_BR_EVO | Griewank | 2D | 180.5561 | 180.5668 | 45/100 | 0.9674 |
| SMCO_BR_vs_SMCO_BR_EVO | Michalewicz | 2D | -0.0000 | -0.0000 | 42/100 | 0.4116 |
| SMCO_BR_vs_SMCO_BR_EVO | Rastrigin | 2D | 80.7062 | 80.7062 | 48/100 | 0.9027 |
| SMCO_BR_vs_SMCO_BR_EVO | Rosenbrock | 2D | 1080188.9346 | 1061717.2485 | 57/100 | 0.0291 * |
| SMCO_BR_vs_SMCO_BR_EVO | Ackley | 10D | 22.1519 | 22.1455 | 51/100 | 0.0980 |
| SMCO_BR_vs_SMCO_BR_EVO | DixonPrice | 10D | 2106720.3344 | 2104852.6075 | 49/100 | 0.9069 |
| SMCO_BR_vs_SMCO_BR_EVO | Griewank | 10D | 871.8596 | 872.1500 | 46/100 | 0.3007 |
| SMCO_BR_vs_SMCO_BR_EVO | Michalewicz | 10D | -0.0000 | -0.0000 | 39/100 | 0.0094 * |
| SMCO_BR_vs_SMCO_BR_EVO | Qing | 10D | 585629679144.5173 | 586044053956.6036 | 45/100 | 0.2697 |
| SMCO_BR_vs_SMCO_BR_EVO | Rastrigin | 10D | 382.3157 | 381.4270 | 54/100 | 0.3750 |
| SMCO_BR_vs_SMCO_BR_EVO | Rosenbrock | 10D | 5905863.9655 | 5914284.8506 | 45/100 | 0.0856 |
| SMCO_BR_vs_SMCO_BR_EVO | Zakharov | 10D | 5187725885.7728 | 5191240827.0794 | 49/100 | 0.4453 |
| SMCO_BR_vs_SMCO_BR_EVO | Ackley | 50D | 22.1839 | 22.0724 | 30/30 | 0.0000 * |
| SMCO_BR_vs_SMCO_BR_EVO | Griewank | 50D | 4161.8033 | 4161.5022 | 17/30 | 0.8394 |
| SMCO_BR_vs_SMCO_BR_EVO | Michalewicz | 50D | -0.0000 | -0.0000 | 13/30 | 0.4522 |
| SMCO_BR_vs_SMCO_BR_EVO | Rastrigin | 50D | 1712.3183 | 1714.4802 | 12/30 | 0.2988 |
| SMCO_BR_vs_SMCO_BR_EVO | Rosenbrock | 50D | 25266864.9537 | 25257153.7605 | 15/30 | 0.7000 |
| SMCO_BR_vs_SMCO_BR_EVO | Ackley | 200D | 22.1698 | 21.9752 | 10/10 | 0.0020 * |
| SMCO_BR_vs_SMCO_BR_EVO | Griewank | 200D | 15576.3673 | 15578.7022 | 4/10 | 0.6953 |
| SMCO_BR_vs_SMCO_BR_EVO | Rastrigin | 200D | 6360.8898 | 6361.4051 | 4/10 | 0.3750 |
| SMCO_R_vs_SMCO_R_EVO | Ackley | 2D | 22.2501 | 22.2468 | 44/100 | 0.3185 |
| SMCO_R_vs_SMCO_R_EVO | Griewank | 2D | 180.9311 | 180.9359 | 43/100 | 0.9494 |
| SMCO_R_vs_SMCO_R_EVO | Michalewicz | 2D | -0.0000 | -0.0000 | 34/100 | 0.3447 |
| SMCO_R_vs_SMCO_R_EVO | Rastrigin | 2D | 80.5454 | 80.6258 | 47/100 | 0.3968 |
| SMCO_R_vs_SMCO_R_EVO | Rosenbrock | 2D | 1066402.4686 | 1028606.5296 | 40/100 | 0.6036 |
| SMCO_R_vs_SMCO_R_EVO | Ackley | 10D | 22.1388 | 22.1488 | 28/100 | 0.0024 * |
| SMCO_R_vs_SMCO_R_EVO | DixonPrice | 10D | 2187672.7660 | 2186850.2960 | 53/100 | 0.4639 |
| SMCO_R_vs_SMCO_R_EVO | Griewank | 10D | 886.9713 | 887.2731 | 49/100 | 0.2865 |
| SMCO_R_vs_SMCO_R_EVO | Michalewicz | 10D | -0.0000 | -0.0000 | 50/100 | 0.6401 |
| SMCO_R_vs_SMCO_R_EVO | Qing | 10D | 605734903366.2952 | 606166055359.6832 | 49/100 | 0.2943 |
| SMCO_R_vs_SMCO_R_EVO | Rastrigin | 10D | 382.3124 | 382.8955 | 50/100 | 0.6724 |
| SMCO_R_vs_SMCO_R_EVO | Rosenbrock | 10D | 6107690.5866 | 6109001.6565 | 46/100 | 0.7544 |
| SMCO_R_vs_SMCO_R_EVO | Zakharov | 10D | 5468900787.1604 | 5467196685.4081 | 57/100 | 0.3900 |
| SMCO_R_vs_SMCO_R_EVO | Ackley | 50D | 22.1510 | 22.0744 | 30/30 | 0.0000 * |
| SMCO_R_vs_SMCO_R_EVO | Griewank | 50D | 4330.5428 | 4333.9931 | 8/30 | 0.0028 * |
| SMCO_R_vs_SMCO_R_EVO | Michalewicz | 50D | -0.0000 | -0.0000 | 13/30 | 0.1519 |
| SMCO_R_vs_SMCO_R_EVO | Rastrigin | 50D | 1709.6572 | 1709.5237 | 13/30 | 0.8394 |
| SMCO_R_vs_SMCO_R_EVO | Rosenbrock | 50D | 27250216.1322 | 27152771.2402 | 17/30 | 0.0248 * |
| SMCO_R_vs_SMCO_R_EVO | Ackley | 200D | 22.1192 | 21.9540 | 10/10 | 0.0020 * |
| SMCO_R_vs_SMCO_R_EVO | Griewank | 200D | 16718.3280 | 16712.9812 | 6/10 | 0.2324 |
| SMCO_R_vs_SMCO_R_EVO | Rastrigin | 200D | 6326.5050 | 6326.5105 | 5/10 | 0.7695 |
| SMCO_vs_SMCO_EVO | Ackley | 2D | 22.2169 | 22.2040 | 48/100 | 0.0309 * |
| SMCO_vs_SMCO_EVO | Griewank | 2D | 180.9510 | 180.9727 | 49/100 | 0.8532 |
| SMCO_vs_SMCO_EVO | Michalewicz | 2D | -0.0000 | -0.0000 | 36/100 | 0.1186 |
| SMCO_vs_SMCO_EVO | Rastrigin | 2D | 80.3560 | 80.3241 | 45/100 | 0.2430 |
| SMCO_vs_SMCO_EVO | Rosenbrock | 2D | 1063965.8992 | 1059543.0820 | 45/100 | 0.0763 |
| SMCO_vs_SMCO_EVO | Ackley | 10D | 21.9984 | 21.9147 | 73/100 | 0.0000 * |
| SMCO_vs_SMCO_EVO | DixonPrice | 10D | 2208389.0456 | 2209028.4966 | 51/100 | 0.9124 |
| SMCO_vs_SMCO_EVO | Griewank | 10D | 892.7622 | 892.6552 | 53/100 | 0.4852 |
| SMCO_vs_SMCO_EVO | Michalewicz | 10D | -0.0000 | -0.0000 | 54/100 | 0.8232 |
| SMCO_vs_SMCO_EVO | Qing | 10D | 613621681699.4482 | 613482000260.4888 | 53/100 | 0.5158 |
| SMCO_vs_SMCO_EVO | Rastrigin | 10D | 382.2272 | 381.5613 | 55/100 | 0.5180 |
| SMCO_vs_SMCO_EVO | Rosenbrock | 10D | 6154510.0782 | 6156178.5291 | 48/100 | 0.4895 |
| SMCO_vs_SMCO_EVO | Zakharov | 10D | 5567718801.5446 | 5564385997.0253 | 62/100 | 0.0306 * |
| SMCO_vs_SMCO_EVO | Ackley | 50D | 22.1324 | 21.6248 | 30/30 | 0.0000 * |
| SMCO_vs_SMCO_EVO | Griewank | 50D | 4398.1982 | 4398.2853 | 19/30 | 0.7922 |
| SMCO_vs_SMCO_EVO | Michalewicz | 50D | -0.0000 | -0.0000 | 18/30 | 0.0919 |
| SMCO_vs_SMCO_EVO | Rastrigin | 50D | 1706.7270 | 1708.3092 | 16/30 | 0.9354 |
| SMCO_vs_SMCO_EVO | Rosenbrock | 50D | 28629952.9778 | 28576472.3160 | 18/30 | 0.4645 |
| SMCO_vs_SMCO_EVO | Ackley | 200D | 22.0797 | 21.4120 | 10/10 | 0.0020 * |
| SMCO_vs_SMCO_EVO | Griewank | 200D | 17218.3374 | 17222.6430 | 3/10 | 0.1055 |
| SMCO_vs_SMCO_EVO | Rastrigin | 200D | 6345.7254 | 6345.6118 | 7/10 | 0.3223 |

## Experiment 2: Convergence Curves

Convergence curves saved to exp2_convergence_curves.png

### Iterations to reach 95% of final value

| Function | Dim | SMCO_R iter | SMCO_R_EVO iter | Speedup |
|----------|-----|-------------|-----------------|---------|
| Ackley | 10D | 0 | 0 | 0.00x |
| Rastrigin | 2D | 26 | 26 | 1.00x |
| Rastrigin | 10D | 31 | 31 | 1.00x |
| Rastrigin | 50D | 32 | 33 | 0.97x |
| Rosenbrock | 10D | 73 | 71 | 1.03x |

## Experiment 3: Evolution Strategy Sensitivity

| Strategy | Ackley | Griewank | Rastrigin | Rosenbrock | Mean |
|----------|------|------|------|------|------
| best1bin | 22.1834 | 888.3622 | 382.4543 | 6282016.6669 | 1570827.4167 |
| current-to-best1bin | 22.1509 | 886.9440 | 379.5587 | 6084093.4971 | 1521345.5377 |
| rand1bin | 22.1419 | 887.2266 | 383.0568 | 6055627.9773 | 1514230.1007 |
| sobol | 22.1415 | 886.4113 | 382.1715 | 6008360.9692 | 1502412.9234 |

Friedman test (Ackley): stat=20.5200, p=0.0001
Friedman test (Griewank): stat=6.2640, p=0.0994
Friedman test (Rastrigin): stat=3.7680, p=0.2876
Friedman test (Rosenbrock): stat=4.5120, p=0.2112

## Experiment 4: Parameter Sensitivity

Heatmap saved: exp4_heatmap_Ackley.png
Heatmap saved: exp4_heatmap_Rastrigin.png
Heatmap saved: exp4_heatmap_Rosenbrock.png

### Best Parameters

- Ackley: elimination_rate=0.1, evolution_points=(0.5, 0.75), mean fopt=22.1518
- Rastrigin: elimination_rate=0.4, evolution_points=(0.25, 0.5, 0.75), mean fopt=387.1703
- Rosenbrock: elimination_rate=0.1, evolution_points=(0.25, 0.5, 0.75), mean fopt=6223535.7595
