# SMCO Python Migration Design

Date: 2026-05-11

## Context

The current workspace at `/amax/math/code/SMCO` contains the SMCO paper PDF and no usable source project. The directory has a `.git` folder, but `git status` reports that it is not a valid Git repository.

The upstream R implementation is `https://github.com/wayne-y-gao/SMCO`. Its current `main` branch is version 1.1.0. The upstream README states that paper replication should use the archived 1.0.0 code. Version 1.1.0 keeps equivalent solution quality while improving defaults, wrappers, validation, and maintainability.

## Goal

Build a minimal runnable Python migration of SMCO that:

- Uses the upstream 1.1.0 API and behavior as the primary implementation target.
- Preserves a paper replication path based on the upstream 1.0.0 scripts and configurations.
- Uses `numpy`, `scipy`, and `pytest`.
- Does not require R for tests.
- Provides enough tests to validate algorithm behavior, inputs, benchmark functions, and runnable benchmark entry points.

## Non-Goals

- Do not fully reproduce every large-scale table from the paper by default.
- Do not require bit-level parity with R's `qrng::sobol`.
- Do not require R to be installed.
- Do not build a full PyPI-ready package with CI and extensive documentation.
- Do not redesign SMCO as a new optimization library beyond the minimal Python migration.

## Project Structure

Create this structure under `/amax/math/code/SMCO`:

```text
pyproject.toml
README.md
src/
  smco/
    __init__.py
    optimizer.py
    results.py
    test_functions.py
    benchmark.py
    paper.py
tests/
  test_optimizer.py
  test_test_functions.py
  test_benchmark.py
  test_validation.py
vendor/
  SMCO_R/
    main/
    v1.0.0/
```

`vendor/SMCO_R/` will store downloaded R source for reference. Tests will not import or execute it.

## Core API

Expose these functions from `smco`:

- `smco(...)`: basic single-phase optimization, corresponding to R `SMCO()`.
- `smco_r(...)`: two-phase refinement, corresponding to R `SMCO_R()`.
- `smco_br(...)`: regular plus boosted refinement, corresponding to R `SMCO_BR()`.
- `smco_multi(...)`: full control entry point, corresponding to R `SMCO_multi()`.

The implementation will maximize the objective function by default. Minimization is handled by passing a negated objective function or by benchmark helpers that wrap minimization functions.

## Algorithm Behavior

The Python port will preserve the R 1.1.0 algorithm semantics:

- Validate callable objective, compatible numeric bounds, start point dimensions, and control parameters.
- Default `n_starts` to `max(5, round(sqrt(dim)))`.
- Generate Sobol start points with `scipy.stats.qmc.Sobol`, scaled to the user bounds.
- Support `partial_option="center"` and `partial_option="forward"`.
- Preserve running maximum behavior controlled by `use_runmax`.
- Preserve refinement behavior with `refine_ratio`, `bounds_buffer`, `buffer_rand`, `iter_boost`, `iter_nstart`, `tol_conv`, and `iter_max`.
- Clip out-of-bound final candidates back into original bounds where the R code does so.
- Keep `use_parallel` as a user-facing option. The minimal implementation may run serially by default; if implemented, parallel execution will use standard Python facilities only.

The code may be Pythonic internally, but the update formulas, stopping condition, and returned fields should stay close to the R implementation.

## Results Model

Use lightweight dataclasses in `results.py`:

- `SingleResult`: stores `x_optimal`, `f_optimal`, `iterations`, and optionally `x_runmax`, `f_runmax`.
- `SMCOResult`: stores `best_result`, `all_results`, `opt_control`, and `summary`.

The dataclasses should be easy to inspect in Python while preserving names that map clearly to the R list output.

## Benchmark And Paper Layer

`test_functions.py` will migrate the standard benchmark functions and configuration helpers from upstream `testfuncs.R`, including at least:

- squared norm and negative squared norm
- Rastrigin
- Ackley
- Griewank
- Rosenbrock
- Michalewicz
- Dixon-Price
- Zakharov
- Qing
- selected 2D functions such as Dropwave, Shubert, McCormick, Six-Hump Camel, Eggholder, Bukin No. 6, Easom, and Mishra No. 6 where practical

`benchmark.py` will provide a small configurable benchmark runner that accepts function name, dimension, variant, repetitions, seed, and iteration controls.

`paper.py` will provide paper-oriented configuration aliases and a small `run_paper_smoke(...)` entry point. Full paper-scale experiments will be documented as manually configurable workloads rather than default tests.

## Testing Strategy

Use `pytest` with three layers:

1. Unit tests:
   - Input validation failures.
   - Bounds checking and clipping.
   - Finite-difference sign behavior.
   - Sobol start point shape, bounds, and seed reproducibility.

2. Algorithm behavior tests:
   - Simple concave quadratic maximization converges near the known optimum.
   - Basic, refinement, and boosted variants return stable result structures.
   - Results stay inside requested bounds after final clipping.
   - Small-dimensional Rastrigin or Ackley runs produce reasonable objective values.

3. Integration smoke tests:
   - Benchmark runner executes a tiny configuration.
   - Paper smoke runner executes a tiny configuration.
   - Public package imports expose the expected API.

Tests should avoid fragile exact numeric parity with one stochastic run. Assertions should focus on reproducibility under the same Python seed, valid bounds, stable structure, and reasonable optimization quality.

## Documentation

`README.md` will include:

- Installation with `pip install -e .`.
- Quick examples for `smco`, `smco_r`, and `smco_br`.
- Explanation that SMCO maximizes by default.
- Benchmark example.
- Note that the R source is vendored only for reference.
- Note that full paper-scale replication requires manually increasing repetitions, dimensions, iterations, and optional parallelism.

## Source Archiving

Download upstream R files into `vendor/SMCO_R/`:

- `main/SMCO.R`
- `main/testfuncs.R`
- `main/benchmark.R`
- `main/README.md`
- `main/NEWS.md`
- `main/LICENSE`
- `v1.0.0/SMCO.R`
- `v1.0.0/testfuncs.R`
- `v1.0.0/testfunc_NN1L.R`
- `v1.0.0/RunComparison.R`
- `v1.0.0/GD.R`
- `v1.0.0/SignGD.R`
- `v1.0.0/SPSA.R`
- `v1.0.0/README.md`
- `v1.0.0/LICENSE`

## Risks And Decisions

- Sobol sequences may differ between R and SciPy. This is acceptable because tests do not assert R parity.
- Some benchmark functions are minimization benchmarks in their original definitions. The Python layer must document whether each config is for maximizing a transformed objective or reports a known minimum.
- High-dimensional paper experiments can be expensive. Default tests must stay small.
- Parallel behavior can introduce serialization constraints for user objective functions. Keep serial execution as the default.

## Git Status

The design could not be committed because `/amax/math/code/SMCO` is not currently a valid Git repository. The observed error was:

```text
fatal: not a git repository (or any parent up to mount point /)
Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).
```
