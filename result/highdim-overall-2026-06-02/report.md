# High-Dimensional SMCO-EVO Overall Report

- Generated: 2026-06-02
- Sources: `highdim-comparison-2026-05-31`, `ultrahighdim-2026-06-01`
- Pairing rule: match by `(pair, func, dim, rep)`; all win/loss decisions use the corrected SMCO maximize-oriented `fopt` semantics.

## Executive Summary

- `ExpA` (`SMCO_R` vs `SMCO_R_EVO`) gives EVO 307/507 wins (60.6%); mean relative improvement is +0.149%.
- `ExpB` (all three base/EVO pairs combined) gives EVO 382/579 wins (66.0%); mean relative improvement is +0.559%.
- Runtime overhead stays negligible: mean time ratio is 1.005x in `ExpA` and 1.009x in `ExpB`.
- Main pattern: evolutionary selection helps more consistently as dimension grows, but the benefit is strongly function-dependent.

## Dimension-Level Summary

| Dim | Cases | EVO wins | Base wins | EVO win rate | Mean rel. impr. | Mean time ratio |
|---|---:|---:|---:|---:|---:|---:|
| 50D | 210 | 119 | 91 | 56.7% | +0.076% | 1.004x |
| 100D | 140 | 81 | 59 | 57.9% | +0.089% | 0.998x |
| 200D | 50 | 33 | 17 | 66.0% | +0.207% | 1.002x |
| 500D | 60 | 42 | 18 | 70.0% | +0.268% | 1.028x |
| 1000D | 29 | 20 | 9 | 69.0% | +0.260% | 1.002x |
| 2000D | 12 | 7 | 5 | 58.3% | +0.281% | 1.017x |
| 5000D | 6 | 5 | 1 | 83.3% | +1.636% | 1.023x |

The dimension trend is directionally positive after the analysis fix: 50D and 100D are already above 55% EVO win rate,
500D reaches 70.0%, and the ultra-high-dimensional tail remains favorable despite the small sample counts at 2000D and 5000D.

## Function-Level Analysis

| Function | Cases | EVO wins | Base wins | EVO win rate | Mean rel. impr. |
|---|---:|---:|---:|---:|---:|
| Ackley | 88 | 88 | 0 | 100.0% | +0.634% |
| Zakharov | 83 | 60 | 23 | 72.3% | +0.217% |
| Rastrigin | 88 | 46 | 42 | 52.3% | +0.002% |
| Rosenbrock | 88 | 43 | 45 | 48.9% | +0.073% |
| Griewank | 60 | 29 | 31 | 48.3% | -0.007% |
| DixonPrice | 50 | 21 | 29 | 42.0% | -0.044% |
| Qing | 50 | 20 | 30 | 40.0% | -0.047% |

- `Ackley` is the clearest success case: EVO wins every paired run in the merged `ExpA` set and the relative gain increases with dimension.
- `Zakharov` also trends positive, with strong win rates and small but consistent mean gains.
- `Rastrigin` is close to neutral overall: wins are slightly positive, but mean relative change is almost zero.
- `DixonPrice` and `Qing` are the main regressions in this suite; they are the strongest evidence that EVO is not uniformly beneficial.
- `Rosenbrock` is mixed: win counts are not dominant, but a few very high-dimensional cases create larger positive mean shifts.

## Variant Comparison

| Pair | Cases | EVO wins | Base wins | EVO win rate | Mean rel. impr. | Mean time ratio |
|---|---:|---:|---:|---:|---:|---:|
| SMCO_vs_SMCO_EVO | 193 | 126 | 67 | 65.3% | +0.787% | 1.009x |
| SMCO_R_vs_SMCO_R_EVO | 193 | 126 | 67 | 65.3% | +0.250% | 1.013x |
| SMCO_BR_vs_SMCO_BR_EVO | 193 | 130 | 63 | 67.4% | +0.641% | 1.004x |

Per-dimension detail for `ExpB`:

| Pair | Dim | Cases | EVO wins | Base wins | EVO win rate |
|---|---|---:|---:|---:|---:|
| SMCO_vs_SMCO_EVO | 100D | 60 | 37 | 23 | 61.7% |
| SMCO_vs_SMCO_EVO | 200D | 40 | 28 | 12 | 70.0% |
| SMCO_vs_SMCO_EVO | 500D | 55 | 36 | 19 | 65.5% |
| SMCO_vs_SMCO_EVO | 1000D | 20 | 13 | 7 | 65.0% |
| SMCO_vs_SMCO_EVO | 2000D | 12 | 8 | 4 | 66.7% |
| SMCO_vs_SMCO_EVO | 5000D | 6 | 4 | 2 | 66.7% |
| SMCO_R_vs_SMCO_R_EVO | 100D | 60 | 36 | 24 | 60.0% |
| SMCO_R_vs_SMCO_R_EVO | 200D | 40 | 26 | 14 | 65.0% |
| SMCO_R_vs_SMCO_R_EVO | 500D | 55 | 39 | 16 | 70.9% |
| SMCO_R_vs_SMCO_R_EVO | 1000D | 20 | 13 | 7 | 65.0% |
| SMCO_R_vs_SMCO_R_EVO | 2000D | 12 | 7 | 5 | 58.3% |
| SMCO_R_vs_SMCO_R_EVO | 5000D | 6 | 5 | 1 | 83.3% |
| SMCO_BR_vs_SMCO_BR_EVO | 100D | 60 | 35 | 25 | 58.3% |
| SMCO_BR_vs_SMCO_BR_EVO | 200D | 40 | 28 | 12 | 70.0% |
| SMCO_BR_vs_SMCO_BR_EVO | 500D | 55 | 38 | 17 | 69.1% |
| SMCO_BR_vs_SMCO_BR_EVO | 1000D | 20 | 15 | 5 | 75.0% |
| SMCO_BR_vs_SMCO_BR_EVO | 2000D | 12 | 10 | 2 | 83.3% |
| SMCO_BR_vs_SMCO_BR_EVO | 5000D | 6 | 4 | 2 | 66.7% |

- `SMCO_BR_EVO` is the most stable family in the merged variant sweep, reaching 67.4% overall win rate.
- `SMCO_EVO` and `SMCO_R_EVO` are effectively tied overall at 65.3%.
- The gap between variants widens in the ultra-high-dimensional regime; `SMCO_BR_EVO` is strongest at 1000D and 2000D.

## Convergence Readout

| Function | Dim | SMCO_R 95% iter | SMCO_R_EVO 95% iter | Ratio |
|---|---:|---:|---:|---:|
| Ackley | 100D | 182 | 50 | 3.64x |
| Ackley | 200D | 182 | 67 | 2.72x |
| Ackley | 500D | 181 | 78 | 2.32x |
| Rastrigin | 100D | 63 | 64 | 0.98x |
| Rastrigin | 200D | 60 | 60 | 1.00x |
| Rastrigin | 500D | 57 | 56 | 1.02x |
| Rosenbrock | 100D | 121 | 122 | 0.99x |
| Rosenbrock | 200D | 133 | 133 | 1.00x |

Ackley is the main convergence-speed exception here: EVO reaches the 95% progress threshold notably earlier.
Rastrigin and Rosenbrock stay near 1.0x, so outside Ackley the current evidence still favors a quality-improvement interpretation over a broad convergence-speed claim.

## Notable Cases

Largest positive relative shifts in `ExpA`:

| Function | Dim | Rep | Source | Rel. impr. |
|---|---:|---:|---|---:|
| Rosenbrock | 5000D | 1 | ultrahighdim-2026-06-01 | +5.577% |
| Rastrigin | 50D | 27 | highdim-comparison-2026-05-31 | +1.912% |
| Zakharov | 200D | 2 | highdim-comparison-2026-05-31 | +1.775% |
| Ackley | 5000D | 1 | ultrahighdim-2026-06-01 | +1.714% |
| Ackley | 5000D | 0 | ultrahighdim-2026-06-01 | +1.711% |
| Zakharov | 50D | 27 | highdim-comparison-2026-05-31 | +1.701% |
| Zakharov | 50D | 4 | highdim-comparison-2026-05-31 | +1.355% |
| Ackley | 2000D | 1 | ultrahighdim-2026-06-01 | +1.167% |

Largest negative relative shifts in `ExpA`:

| Function | Dim | Rep | Source | Rel. impr. |
|---|---:|---:|---|---:|
| Rastrigin | 50D | 29 | highdim-comparison-2026-05-31 | -1.418% |
| Rastrigin | 50D | 15 | highdim-comparison-2026-05-31 | -1.295% |
| Rastrigin | 50D | 23 | highdim-comparison-2026-05-31 | -0.714% |
| Rastrigin | 50D | 3 | highdim-comparison-2026-05-31 | -0.708% |
| Rastrigin | 50D | 22 | highdim-comparison-2026-05-31 | -0.699% |
| DixonPrice | 50D | 21 | highdim-comparison-2026-05-31 | -0.699% |
| Rosenbrock | 50D | 21 | highdim-comparison-2026-05-31 | -0.689% |
| Rastrigin | 100D | 19 | highdim-comparison-2026-05-31 | -0.610% |

## Files

- `summary.csv`: compact machine-readable overview of all aggregate statistics.
- `analysis/expA_cases.csv`: all merged `ExpA` paired outcomes.
- `analysis/expA_by_function.csv`: per-function aggregates for merged `ExpA`.
- `analysis/expB_cases.csv`: all merged `ExpB` paired outcomes.
- `analysis/expB_by_pair.csv`: per-pair aggregates for merged `ExpB`.
- `analysis/expC_summary.csv`: convergence readout extracted from the corrected `ExpC` data.
