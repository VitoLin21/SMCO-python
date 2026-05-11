# SMCO Python

Minimal Python migration of the Strategic Monte Carlo Optimization algorithms from `wayne-y-gao/SMCO`.

This package ports the current R `v1.1.0` API shape while preserving a paper-oriented path for the archived `v1.0.0` replication code. The vendored R files are reference material only; Python tests do not execute R.

## Install

```bash
uv python install 3.12
uv venv .venv --python 3.12
uv pip install -e ".[test]"
```

Run commands with the local Python 3.12 environment:

```bash
.venv/bin/python -m pytest -q
```

## Quick Start

```python
import numpy as np
from smco import smco_r

def objective(x):
    return -float(np.sum(x ** 2))

result = smco_r(
    objective,
    bounds_lower=[-5, -5],
    bounds_upper=[5, 5],
    iter_max=200,
    seed=123,
)

print(result.best_result.x_optimal)
print(result.best_result.f_optimal)
```

SMCO maximizes by default. To minimize a function, pass its negative.

## Algorithm Variants

- `smco(...)`: basic single-phase search.
- `smco_r(...)`: two-phase refinement search.
- `smco_br(...)`: regular plus boosted refinement search.
- `smco_multi(...)`: full-control multi-start entry point.

## Benchmarks

```python
from smco import run_benchmark

run = run_benchmark(
    "Rastrigin",
    dim=2,
    variant="smco_r",
    repetitions=3,
    iter_max=100,
    n_starts=5,
    seed=123,
)

print(run.best_x)
print(run.best_value)
```

Classic minimization benchmarks are wrapped as maximization objectives internally. `BenchmarkConfig.known_best_value` keeps the raw benchmark scale, while `known_best_objective` stores the value in SMCO's maximization scale when known.

## Paper Smoke Run

```python
from smco import run_paper_smoke

runs = run_paper_smoke(iter_max=100, n_starts=5, seed=123)
print(runs.keys())
```

The smoke runner is intentionally small. Full paper-scale experiments require manually increasing dimensions, repetitions, iteration counts, and optional parallel settings.

## Testing

```bash
pytest -q
```

## Upstream R Source

Reference R source files are stored in `vendor/SMCO_R`:

- `main/`: current upstream R implementation.
- `v1.0.0/`: archived paper replication implementation and comparison scripts.

The upstream `main` README references `benchmark_version_comparison.R`, but that file was not available from the current public repository when this archive was created.
