# SMCO vs SMCO-EVO Benchmark Comparison

- Date: 2026-05-27
- Functions: 18
- Repetitions per function/variant: 20
- iter_max: 300
- n_starts: 10
- Evolution strategy: rand1bin (default)
- Tie threshold for mean comparisons: 1e-09

## Overall

- Mean result better for `smco_evo` on 15 functions, for `smco` on 1 functions, tied on 2 functions.
- Paired repetition wins: `smco_evo` 133, `smco` 18, ties 209.
- Mean runtime per run: `smco` 0.2331s, `smco_evo` 0.2344s.

## Function Summary

| Function | Dim | SMCO mean | EVO mean | Mean diff (EVO-SMCO) | EVO win reps | SMCO win reps |
|---|---:|---:|---:|---:|---:|---:|
| Eggholder | 2 | 854.338237 | 903.650131 | 49.311895 | 10 | 0 |
| Qing | 10 | -79.914763 | -71.713369 | 8.201394 | 6 | 0 |
| Shubert | 2 | 175.650298 | 178.194504 | 2.544207 | 5 | 0 |
| Rosenbrock | 10 | -0.793295 | -0.703005 | 0.090290 | 10 | 2 |
| Ackley | 2 | -0.108972 | -0.072578 | 0.036394 | 9 | 0 |
| Michalewicz | 10 | 8.986432 | 9.022456 | 0.036024 | 4 | 0 |
| DropWave | 2 | 0.951777 | 0.970535 | 0.018758 | 10 | 1 |
| Rastrigin | 2 | -0.012907 | -0.006016 | 0.006890 | 9 | 0 |
| Mishra6 | 2 | 3.948108 | 3.953793 | 0.005685 | 10 | 0 |
| DixonPrice | 10 | -0.674738 | -0.673329 | 0.001409 | 11 | 0 |
| Griewank | 2 | -0.007246 | -0.006451 | 0.000796 | 8 | 2 |
| Zakharov | 10 | -0.001150 | -0.001141 | 0.000009 | 6 | 6 |
| SixHumpCamel | 2 | 1.031628 | 1.031628 | 0.000000 | 9 | 0 |
| NegativeSquaredNorm | 2 | -0.000001 | -0.000000 | 0.000000 | 10 | 0 |
| McCormick | 2 | 1.913223 | 1.913223 | 0.000000 | 11 | 0 |
| Easom | 2 | 0.398104 | 0.398104 | 0.000000 | 1 | 0 |
| SquaredNorm | 2 | 2.000000 | 2.000000 | 0.000000 | 0 | 0 |
| Bukin6 | 2 | -0.208184 | -0.223264 | -0.015080 | 4 | 7 |
