# SMCO Evolutionary Multi-Start Design

## Goal

Add a new evolutionary multi-start SMCO family based on the original `smco` algorithm. The existing `smco_hb` / Hyperband implementation is considered deprecated for this work and must not be used as a design reference.

The new algorithm adds population-level elimination and regeneration at scheduled iteration progress points while preserving the internal SMCO state for surviving trajectories.

## Public API

Add three public wrappers aligned with the existing SMCO family:

```python
smco_evo(...)
smco_r_evo(...)
smco_br_evo(...)
```

They should mirror `smco`, `smco_r`, and `smco_br` defaults, with additional evolutionary controls:

```python
evolution_points=(0.5, 0.75)
elimination_rate=0.25
evolution_strategy="rand1bin"
de_factor=0.8
de_crossover=0.7
```

Supported `evolution_strategy` values:

- `"rand1bin"`: default, exploratory differential evolution.
- `"current-to-best1bin"`: moves each parent toward the best survivor with differential perturbation.
- `"best1bin"`: greedier strategy around the best survivor.
- `"sobol"`: non-DE baseline that regenerates points with Sobol sampling.

## Execution Model

Each start point becomes a stateful SMCO trajectory. A trajectory stores enough information to pause and resume the original `_single` loop:

```text
x_current
f_current
s_value
current_n / last_n
x_runmax
f_runmax
```

At each scheduled evolution point, for example 50% and 75% of `iter_max`:

1. Run every active trajectory until the scheduled iteration boundary.
2. Rank trajectories by `f_runmax`.
3. Eliminate the worst `ceil(n_active * elimination_rate)` trajectories.
4. Keep survivor states unchanged so they continue the same SMCO recursion.
5. Generate replacement start points from survivor `x_runmax` values.
6. Initialize each replacement as a fresh SMCO state at the generated point.
7. Continue all active trajectories to the next boundary.

The final stage runs all active trajectories to `iter_max`, then returns the best result by `f_optimal` using the same maximize convention as the original SMCO implementation.

## Regeneration Strategies

All DE strategies use survivor `x_runmax` values as parents and survivor `f_runmax` values for best-parent selection.

`rand1bin`:

```text
mutant = a + F * (b - c)
```

`current-to-best1bin`:

```text
mutant = parent + F * (best - parent) + F * (b - c)
```

`best1bin`:

```text
mutant = best + F * (b - c)
```

Each DE mutant is combined with a base parent using binomial crossover. At least one dimension must come from the mutant. Generated points are clipped to the original bounds.

If the survivor pool is too small for the chosen DE strategy, fall back to Sobol regeneration.

## Results and Metadata

Return the existing `SMCOResult` type. Do not introduce a new top-level result class in the first version.

Add evolutionary metadata to `summary`:

```python
summary["evolution_history"] = [
    {
        "iteration": 50,
        "strategy": "rand1bin",
        "survivor_count": 6,
        "eliminated_count": 2,
        "generated_count": 2,
        "best_before": -0.01,
        "best_after_generation": -0.01,
    }
]
```

Also include final `summary["evolution_strategy"]`, `summary["evolution_points"]`, and `summary["elimination_rate"]`.

## Validation Plan

Tests should focus on `tests/test_optimizer.py`:

- Public exports for `smco_evo`, `smco_r_evo`, and `smco_br_evo`.
- Evolution triggers at the requested iteration progress points.
- Survivor states continue from their existing internal state instead of restarting.
- Elimination uses `f_runmax`, not `f_current`.
- `rand1bin`, `current-to-best1bin`, `best1bin`, and `sobol` are accepted strategies.
- Invalid rates, points, and strategy names raise clean `ValueError`s.
- Repeated runs with the same seed are reproducible.
- A smoke optimization on a concave quadratic returns a bounded, finite result.

Run focused tests first, then finish with:

```bash
.venv/bin/python -m pytest -q
```
