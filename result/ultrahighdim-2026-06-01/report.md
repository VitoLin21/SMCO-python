# High-Dimensional SMCO-EVO vs SMCO Comparison Report
Generated: 2026-06-02
Input: result/ultrahighdim-2026-06-01

## Experiment A: Scaling Benchmark (SMCO_R vs SMCO_R_EVO)

### Win/Loss by Dimension

| Dim | N | EVO wins | Base wins | Ties | EVO win rate |
|-----|---|----------|-----------|------|-------------|
| 500D | 40 | 28 | 12 | 0 | 70.0% |
| 1000D | 20 | 13 | 7 | 0 | 65.0% |
| 2000D | 12 | 7 | 5 | 0 | 58.3% |
| 5000D | 6 | 5 | 1 | 0 | 83.3% |

### Wilcoxon Test by Function and Dimension

| Function | Dim | N | EVO mean | Base mean | Rel. impr. | p-value |
|----------|-----|---|----------|-----------|------------|---------|
| Ackley | 500D | 10 | 22.1362 | 21.9418 | +0.89% | 0.0020 ** |
| Ackley | 1000D | 5 | 22.1456 | 21.9374 | +0.95% | 0.0625 |
| Ackley | 2000D | 3 | 22.1514 | 21.8994 | +1.15% | N/A |
| Ackley | 5000D | 2 | 22.0957 | 21.7237 | +1.71% | N/A |
| Rastrigin | 500D | 10 | 14828.9057 | 14828.9178 | -0.00% | 0.4316 |
| Rastrigin | 1000D | 5 | 29243.3775 | 29243.3375 | +0.00% | 1.0000 |
| Rastrigin | 2000D | 3 | 57742.9328 | 57747.0039 | -0.01% | N/A |
| Rastrigin | 5000D | 2 | 142649.3605 | 142667.5975 | -0.01% | N/A |
| Rosenbrock | 500D | 10 | 228248743.2596 | 228165092.0340 | +0.04% | 0.1602 |
| Rosenbrock | 1000D | 5 | 422402178.9169 | 422480981.8960 | -0.02% | 0.8125 |
| Rosenbrock | 2000D | 3 | 774754360.4679 | 774828026.0801 | -0.01% | N/A |
| Rosenbrock | 5000D | 2 | 1742676220.0948 | 1688438348.8648 | +3.21% | N/A |
| Zakharov | 500D | 10 | 109424737783069422911488.0000 | 109315128449742107836416.0000 | +0.10% | 0.1602 |
| Zakharov | 1000D | 5 | 24739790779593522763792384.0000 | 24736117535239985600397312.0000 | +0.01% | 0.4375 |
| Zakharov | 2000D | 3 | 5351111221591706703751020544.0000 | 5351736069869670488304779264.0000 | -0.01% | N/A |

## Experiment B: All Variant Pairs at High Dimensions

### SMCO vs SMCO_EVO

Overall: EVO 50/78 wins (64.1%)
- 500D: 25/40 wins
- 1000D: 13/20 wins
- 2000D: 8/12 wins
- 5000D: 4/6 wins

### SMCO_R vs SMCO_R_EVO

Overall: EVO 53/78 wins (67.9%)
- 500D: 28/40 wins
- 1000D: 13/20 wins
- 2000D: 7/12 wins
- 5000D: 5/6 wins

### SMCO_BR vs SMCO_BR_EVO

Overall: EVO 57/78 wins (73.1%)
- 500D: 28/40 wins
- 1000D: 15/20 wins
- 2000D: 10/12 wins
- 5000D: 4/6 wins

Experiment C: No data.
