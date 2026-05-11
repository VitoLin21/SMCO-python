# Comparison Methods Migration & Experiment Reproduction

## Goal

Migrate all 12 comparison optimizer methods from R (`wayne-y-gao/SMCO`) to Python, implement the `run_comparison` framework, and reproduce the paper's experiments (Tables 1-3) with results saved to `./result`.

## Scope

### Phase 1: Comparison Methods (./comparison/)

Migrate 12 optimizer algorithms into a unified interface:

**Hand-written (from R source):**
- GD (Gradient Descent) — from `GD.R`
- SignGD (Sign Gradient Descent) — from `SignGD.R`
- SPSA (Simultaneous Perturbation SA) — from `SPSA.R`

**Python library equivalents (replacing R packages):**
- ADAM — `scipy.optimize.minimize` with Adam-style updates or custom implementation
- L-BFGS-B — `scipy.optimize.minimize(method='L-BFGS-B')`
- Nelder-Mead — `scipy.optimize.minimize(method='Nelder-Mead')`
- BOBYQA — `scipy.optimize.minimize(method='COBYLA')` or `nlopt` if available
- GenSA — `scipy.optimize.dual_annealing`
- SA — `scipy.optimize.basinhopping`
- DE — `scipy.optimize.differential_evolution`
- GA — `scipy.optimize.differential_evolution` variant or lightweight library
- PSO — `pyswarm.pso` or custom swarm implementation

**Unified interface per method:**

```python
def optimize(
    f: Callable[[np.ndarray], float],
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_points: np.ndarray | None = None,
    maximize: bool = True,
    **kwargs,
) -> OptimizerResult
```

Returns a consistent `OptimizerResult` with fields: `x_optimal`, `f_optimal`, `iterations`, `history` (optional).

### Phase 2: Run Comparison Framework

Migrate `RunComparison.R` to Python:

- `run_comparison()` — core comparison loop across algorithms and test configs
- `generate_unif_starts()` — uniform starting point generation
- `generate_sobol_starts()` — Sobol sequence starting points (already exists in SMCO)
- Domain modification: rotation + shifting + asymmetrization (from R code)
- Metrics: RMSE, AE50, AE95, AE99, Time per run
- `print_results()` — output to JSON/CSV

### Phase 3: Experiment Execution

**Quick validation (--quick):**
- 4 test functions: Rastrigin, Ackley, Griewank, Michalewicz
- d=2, maximize only, 5 repetitions, 3 starting points
- All 15 algorithms (3 SMCO variants + 12 comparison)

**Full Table 1:**
- d=2, 4 functions x 2 (min/max), 500 reps, 1 starting point
- Group I local algorithms only

**Full Table 2:**
- d=200, 4 functions x 2 (min/max), 10 reps
- Group II global algorithms only

**Full Table 3:**
- Random functions (Maximum Score + Empirical Welfare)
- d=5,10,20,50, variable reps
- Both groups

## File Structure

```
comparison/
  __init__.py
  methods/
    __init__.py
    base.py          # OptimizerResult, base interface
    gd.py
    sign_gd.py
    spsa.py
    adam.py
    lbfgs.py
    nelder_mead.py
    bobyqa.py
    gensa.py
    sa.py
    de.py
    ga.py
    pso.py
  run_comparison.py  # core comparison framework
  domain_mod.py      # domain rotation/shift/asymmetrization
scripts/
  run_experiment.py  # entry point: --quick / --table1 / --table2 / --table3
result/
  comparison/        # experiment outputs
```

## Dependencies

- **Existing:** numpy, scipy (already installed)
- **New lightweight:** pyswarm (for PSO) — optional, can fallback to custom
- **Optional:** nlopt (for BOBYQA/STOGO) — fallback to scipy COBYLA

## Key Design Decisions

1. **Maximize convention**: SMCO maximizes internally. Comparison methods receive `f` as-is; the framework handles sign flipping via `maximize` flag.
2. **Multi-start**: All local methods run from the same starting points as SMCO; global methods run without specified starts (or one random start).
3. **Reproducibility**: Seeds are controlled per-replication with offset, matching R behavior.
4. **STOGO replacement**: No exact Python equivalent; use `scipy.optimize.dual_annealing` as the nearest global optimizer.
5. **CMAES**: Use `cma` library if available, otherwise skip gracefully.
