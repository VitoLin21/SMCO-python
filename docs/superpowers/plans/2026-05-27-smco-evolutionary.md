# SMCO Evolutionary Multi-Start Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `smco_evo`, `smco_r_evo`, and `smco_br_evo` as state-preserving evolutionary multi-start variants of the original SMCO algorithm.

**Architecture:** Keep existing `smco`, `smco_r`, `smco_br`, and `smco_multi` behavior unchanged. Add a stateful internal trajectory runner for the original `_single` loop, then build an evolutionary scheduler that eliminates low-`f_runmax` trajectories at progress boundaries and regenerates replacements with `rand1bin` by default. Treat existing `smco_hb` as deprecated and unrelated.

**Tech Stack:** Python 3.10+, `dataclasses`, `numpy`, existing Sobol helper, existing `pytest` suite.

---

## File Structure

- Modify `src/smco/optimizer.py`: add internal state dataclass, stateful step/run helpers, evolutionary strategy validation, DE/Sobol generation, evolutionary multi-start runner, and three public wrappers.
- Modify `src/smco/__init__.py`: export `smco_evo`, `smco_r_evo`, and `smco_br_evo`.
- Modify `tests/test_optimizer.py`: add public API, validation, strategy, metadata, reproducibility, and smoke optimization tests.
- Optionally modify `tests/test_validation.py`: add low-level validation tests only if keeping them separate makes failures clearer.
- Modify `docs/usage.md`: add a short usage section for the new wrappers after implementation passes tests.

## Implementation Notes

- Evolution applies to the original `_single` loop. For `smco_r_evo`, split iterations with `_split_refine_iterations(...)`, run evolution during the initial search segment, then refine each final trajectory from its best point with the existing zero-buffer refine pass.
- For `smco_br_evo`, run a regular evolutionary path and a boosted evolutionary path, then return the better one by `f_optimal`, mirroring `_single_boost`.
- Selection uses `f_runmax`. Regeneration parents use survivor `x_runmax`.
- Replacement trajectories are fresh states initialized at generated points. Survivor trajectories continue with unchanged state.
- If a DE strategy lacks enough survivor parents, fall back to Sobol regeneration.

---

### Task 1: Public API and Control Validation

**Files:**
- Modify: `tests/test_optimizer.py`
- Modify: `src/smco/optimizer.py`
- Modify: `src/smco/__init__.py`

- [ ] **Step 1: Write failing public export and validation tests**

Append to `tests/test_optimizer.py`:

```python
from smco import smco_br_evo, smco_evo, smco_r_evo
```

Add tests:

```python
def test_public_api_exports_evolutionary_smco_variants():
    assert callable(smco_evo)
    assert callable(smco_r_evo)
    assert callable(smco_br_evo)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"evolution_points": [0.0]}, "evolution_points"),
        ({"evolution_points": [1.0]}, "evolution_points"),
        ({"evolution_points": [0.75, 0.5]}, "evolution_points"),
        ({"elimination_rate": 0.0}, "elimination_rate"),
        ({"elimination_rate": 1.0}, "elimination_rate"),
        ({"evolution_strategy": "rand2bin"}, "evolution_strategy"),
        ({"de_factor": 0.0}, "de_factor"),
        ({"de_crossover": 1.5}, "de_crossover"),
    ],
)
def test_smco_evo_rejects_invalid_evolution_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        smco_evo(
            lambda x: -float(x[0] ** 2),
            [-1.0],
            [1.0],
            n_starts=4,
            iter_max=8,
            seed=123,
            **kwargs,
        )
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_optimizer.py::test_public_api_exports_evolutionary_smco_variants tests/test_optimizer.py::test_smco_evo_rejects_invalid_evolution_controls -v
```

Expected: import failure or callable assertion failure for missing evolutionary wrappers.

- [ ] **Step 3: Add control constants and validation helpers**

In `src/smco/optimizer.py`, add near `DEFAULT_CONTROL`:

```python
EVOLUTION_STRATEGIES = {"rand1bin", "current-to-best1bin", "best1bin", "sobol"}


def _normalize_evolution_points(value: Any) -> tuple[float, ...]:
    points = tuple(float(point) for point in value)
    if not points:
        raise ValueError("evolution_points must not be empty")
    if any((not math.isfinite(point)) or point <= 0.0 or point >= 1.0 for point in points):
        raise ValueError("evolution_points must contain values between 0 and 1")
    if tuple(sorted(points)) != points or len(set(points)) != len(points):
        raise ValueError("evolution_points must be strictly increasing")
    return points


def _validate_evolution_control(
    evolution_points: Any,
    elimination_rate: Any,
    evolution_strategy: str,
    de_factor: Any,
    de_crossover: Any,
) -> tuple[tuple[float, ...], float, str, float, float]:
    points = _normalize_evolution_points(evolution_points)
    rate = float(elimination_rate)
    if not math.isfinite(rate) or rate <= 0.0 or rate >= 1.0:
        raise ValueError("elimination_rate must be between 0 and 1")
    strategy = str(evolution_strategy)
    if strategy not in EVOLUTION_STRATEGIES:
        raise ValueError("evolution_strategy must be one of current-to-best1bin, rand1bin, best1bin, sobol")
    factor = float(de_factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("de_factor must be positive")
    crossover = float(de_crossover)
    if not math.isfinite(crossover) or crossover < 0.0 or crossover > 1.0:
        raise ValueError("de_crossover must be between 0 and 1")
    return points, rate, strategy, factor, crossover
```

- [ ] **Step 4: Add temporary wrappers that validate controls**

In `src/smco/optimizer.py`, add after `smco_br(...)`:

```python
def smco_evo(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    **kwargs: Any,
) -> SMCOResult:
    control = dict(kwargs)
    _validate_evolution_control(
        control.pop("evolution_points", (0.5, 0.75)),
        control.pop("elimination_rate", 0.25),
        control.pop("evolution_strategy", "rand1bin"),
        control.pop("de_factor", 0.8),
        control.pop("de_crossover", 0.7),
    )
    control["refine_search"] = False
    control["iter_boost"] = 0
    return smco_multi(f, bounds_lower, bounds_upper, start_points, control)


def smco_r_evo(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    **kwargs: Any,
) -> SMCOResult:
    control = dict(kwargs)
    _validate_evolution_control(
        control.pop("evolution_points", (0.5, 0.75)),
        control.pop("elimination_rate", 0.25),
        control.pop("evolution_strategy", "rand1bin"),
        control.pop("de_factor", 0.8),
        control.pop("de_crossover", 0.7),
    )
    control["refine_search"] = True
    control["iter_boost"] = 0
    control.setdefault("refine_ratio", 0.5)
    return smco_multi(f, bounds_lower, bounds_upper, start_points, control)


def smco_br_evo(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    iter_boost: int = 1000,
    **kwargs: Any,
) -> SMCOResult:
    control = dict(kwargs)
    _validate_evolution_control(
        control.pop("evolution_points", (0.5, 0.75)),
        control.pop("elimination_rate", 0.25),
        control.pop("evolution_strategy", "rand1bin"),
        control.pop("de_factor", 0.8),
        control.pop("de_crossover", 0.7),
    )
    control["refine_search"] = True
    control["iter_boost"] = iter_boost
    control.setdefault("refine_ratio", 0.5)
    return smco_multi(f, bounds_lower, bounds_upper, start_points, control)
```

- [ ] **Step 5: Export wrappers**

In `src/smco/__init__.py`, change the optimizer import to:

```python
from .optimizer import smco, smco_br, smco_br_evo, smco_evo, smco_multi, smco_r, smco_r_evo
```

Add names to `__all__`:

```python
"smco_br_evo",
"smco_evo",
"smco_r_evo",
```

- [ ] **Step 6: Run tests and verify Task 1 passes**

Run:

```bash
.venv/bin/python -m pytest tests/test_optimizer.py::test_public_api_exports_evolutionary_smco_variants tests/test_optimizer.py::test_smco_evo_rejects_invalid_evolution_controls -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/smco/optimizer.py src/smco/__init__.py tests/test_optimizer.py
git commit -m "feat: add evolutionary SMCO API validation"
```

---

### Task 2: Stateful SMCO Trajectory Runner

**Files:**
- Modify: `tests/test_optimizer.py`
- Modify: `src/smco/optimizer.py`

- [ ] **Step 1: Write failing state-preservation tests**

Add imports in `tests/test_optimizer.py`:

```python
from smco.optimizer import _initialize_smco_state, _run_smco_state_until
```

Add tests:

```python
def test_stateful_smco_runner_matches_single_stage_result_without_evolution():
    def objective(x):
        return -float((x[0] - 0.2) ** 2)

    lower = np.array([-1.0])
    upper = np.array([1.0])
    rng = np.random.default_rng(123)
    state = _initialize_smco_state(
        objective,
        lower,
        upper,
        np.array([0.8]),
        bounds_buffer=0.05,
        buffer_rand=False,
        iter_nstart=1,
        iter_boost=0,
        use_runmax=True,
    )
    _run_smco_state_until(
        state,
        objective,
        lower,
        upper,
        bounds_buffer=0.05,
        buffer_rand=False,
        iter_target=40,
        tol_conv=1e-12,
        partial_option="center",
        use_runmax=True,
        rng=rng,
    )
    state_result = state.to_result()

    full = smco(
        objective,
        lower,
        upper,
        start_points=[[0.8]],
        iter_max=40,
        n_starts=1,
        iter_nstart=1,
        seed=123,
        tol_conv=1e-12,
    ).best_result

    assert np.allclose(state_result.x_optimal, full.x_optimal)
    assert state_result.f_optimal == pytest.approx(full.f_optimal)
    assert state_result.f_runmax == pytest.approx(full.f_runmax)


def test_stateful_smco_runner_resumes_without_reinitializing_state():
    def objective(x):
        return -float((x[0] - 0.2) ** 2)

    lower = np.array([-1.0])
    upper = np.array([1.0])
    rng = np.random.default_rng(123)
    state = _initialize_smco_state(
        objective,
        lower,
        upper,
        np.array([0.8]),
        bounds_buffer=0.05,
        buffer_rand=False,
        iter_nstart=1,
        iter_boost=0,
        use_runmax=True,
    )
    _run_smco_state_until(state, objective, lower, upper, 0.05, False, 20, 1e-12, "center", True, rng)
    s_value_after_20 = np.array(state.s_value, copy=True)
    _run_smco_state_until(state, objective, lower, upper, 0.05, False, 40, 1e-12, "center", True, rng)

    assert state.current_n >= 40
    assert not np.allclose(state.s_value, s_value_after_20)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_optimizer.py::test_stateful_smco_runner_matches_single_stage_result_without_evolution tests/test_optimizer.py::test_stateful_smco_runner_resumes_without_reinitializing_state -v
```

Expected: import failure for missing `_initialize_smco_state`.

- [ ] **Step 3: Add state dataclass and initializer**

In `src/smco/optimizer.py`, add near other dataclasses:

```python
@dataclass
class SMCOState:
    x_current: np.ndarray
    f_current: float
    s_value: np.ndarray
    current_n: int
    iter_boost: int
    x_runmax: np.ndarray | None
    f_runmax: float | None
    iterations: int = 0

    def ranking_value(self) -> float:
        if self.f_runmax is not None:
            return float(self.f_runmax)
        return float(self.f_current)

    def ranking_point(self) -> np.ndarray:
        if self.x_runmax is not None:
            return np.array(self.x_runmax, dtype=float, copy=True)
        return np.array(self.x_current, dtype=float, copy=True)

    def to_result(self) -> SingleResult:
        return SingleResult(
            x_optimal=np.array(self.x_current, dtype=float, copy=True),
            f_optimal=float(self.f_current),
            iterations=int(self.iterations),
            x_runmax=None if self.x_runmax is None else np.array(self.x_runmax, dtype=float, copy=True),
            f_runmax=None if self.f_runmax is None else float(self.f_runmax),
        )
```

Add initializer:

```python
def _initialize_smco_state(
    f: Objective,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_point: np.ndarray,
    *,
    bounds_buffer: float,
    buffer_rand: bool,
    iter_nstart: int,
    iter_boost: int,
    use_runmax: bool,
) -> SMCOState:
    x_current = np.array(start_point, dtype=float, copy=True)
    f_current = float(f(x_current))
    n_boost_1 = int(iter_boost) + int(iter_nstart)
    x_runmax = np.array(x_current, copy=True) if use_runmax else None
    f_runmax = float(f_current) if use_runmax else None
    return SMCOState(
        x_current=x_current,
        f_current=f_current,
        s_value=x_current * n_boost_1,
        current_n=n_boost_1,
        iter_boost=int(iter_boost),
        x_runmax=x_runmax,
        f_runmax=f_runmax,
        iterations=0,
    )
```

- [ ] **Step 4: Add resumable run helper**

Add to `src/smco/optimizer.py`:

```python
def _run_smco_state_until(
    state: SMCOState,
    f: Objective,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    bounds_buffer: float,
    buffer_rand: bool,
    iter_target: int,
    tol_conv: float,
    partial_option: str,
    use_runmax: bool,
    rng: np.random.Generator,
) -> None:
    bounds_diff = bounds_upper - bounds_lower
    fixed_pushout = float(bounds_buffer) * bounds_diff
    fixed_upper_out = bounds_upper + fixed_pushout
    fixed_lower_out = bounds_lower - fixed_pushout
    target_n = int(state.iter_boost) + int(iter_target)

    while state.current_n < target_n:
        n = state.current_n
        h_step = bounds_diff / (n + 1)
        partial = compute_partial_signs(
            f,
            state.x_current,
            state.f_current,
            h_step,
            bounds_lower,
            bounds_upper,
            partial_option,
            use_runmax,
        )
        if buffer_rand:
            pushout = float(bounds_buffer) * bounds_diff * rng.uniform(-1.0, 1.0, size=bounds_diff.size)
            bounds_upper_out = bounds_upper + pushout
            bounds_lower_out = bounds_lower - pushout
        else:
            bounds_upper_out = fixed_upper_out
            bounds_lower_out = fixed_lower_out

        z_value = partial.signs * bounds_upper_out + (1.0 - partial.signs) * bounds_lower_out
        state.s_value = state.s_value + z_value
        x_next = state.s_value / (n + 1)
        f_next = float(f(x_next))

        if use_runmax:
            f_next_best = max(partial.f_partial_best, f_next)
            if state.f_runmax is None or f_next_best > state.f_runmax:
                state.f_runmax = float(f_next_best)
                state.x_runmax = np.array(
                    partial.x_partial_best if partial.f_partial_best > f_next else x_next,
                    copy=True,
                )

        f_prev = state.f_current
        state.f_current = f_next
        state.x_current = x_next
        state.current_n = n + 1
        state.iterations = int(state.current_n - state.iter_boost)
        if state.iterations >= int(math.ceil(iter_target / 2)) and abs(state.f_current - f_prev) < tol_conv:
            break
```

- [ ] **Step 5: Keep existing `_single` behavior by delegating to state helper**

Replace the body of `_single(...)` with:

```python
    state = _initialize_smco_state(
        f,
        bounds_lower,
        bounds_upper,
        start_point,
        bounds_buffer=bounds_buffer,
        buffer_rand=buffer_rand,
        iter_nstart=iter_nstart,
        iter_boost=iter_boost,
        use_runmax=use_runmax,
    )
    _run_smco_state_until(
        state,
        f,
        bounds_lower,
        bounds_upper,
        bounds_buffer,
        buffer_rand,
        iter_max,
        tol_conv,
        partial_option,
        use_runmax,
        rng,
    )
    return state.to_result()
```

- [ ] **Step 6: Run tests and verify Task 2 passes**

Run:

```bash
.venv/bin/python -m pytest tests/test_optimizer.py::test_stateful_smco_runner_matches_single_stage_result_without_evolution tests/test_optimizer.py::test_stateful_smco_runner_resumes_without_reinitializing_state tests/test_optimizer.py::test_smco_r_optimizes_concave_quadratic_near_origin -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/smco/optimizer.py tests/test_optimizer.py
git commit -m "refactor: add resumable SMCO trajectory state"
```

---

### Task 3: Regeneration Strategies

**Files:**
- Modify: `tests/test_optimizer.py`
- Modify: `src/smco/optimizer.py`

- [ ] **Step 1: Write failing regeneration tests**

Add import:

```python
from smco.optimizer import _generate_evolution_points
```

Add tests:

```python
@pytest.mark.parametrize("strategy", ["rand1bin", "current-to-best1bin", "best1bin", "sobol"])
def test_generate_evolution_points_supports_configured_strategies(strategy):
    parents = np.array(
        [
            [-0.8, -0.4],
            [-0.2, 0.1],
            [0.3, 0.5],
            [0.7, -0.6],
        ],
        dtype=float,
    )
    scores = np.array([0.1, 0.5, 0.9, 0.2], dtype=float)
    generated = _generate_evolution_points(
        parents,
        scores,
        n_new=3,
        strategy=strategy,
        bounds_lower=np.array([-1.0, -1.0]),
        bounds_upper=np.array([1.0, 1.0]),
        de_factor=0.8,
        de_crossover=0.7,
        rng=np.random.default_rng(123),
    )

    assert generated.shape == (3, 2)
    assert np.all(generated >= -1.0)
    assert np.all(generated <= 1.0)


def test_generate_evolution_points_falls_back_to_sobol_when_parent_pool_is_too_small():
    generated = _generate_evolution_points(
        np.array([[0.0, 0.0], [0.5, 0.5]], dtype=float),
        np.array([1.0, 0.5], dtype=float),
        n_new=2,
        strategy="rand1bin",
        bounds_lower=np.array([-1.0, -1.0]),
        bounds_upper=np.array([1.0, 1.0]),
        de_factor=0.8,
        de_crossover=0.7,
        rng=np.random.default_rng(123),
    )

    assert generated.shape == (2, 2)
    assert np.all(generated >= -1.0)
    assert np.all(generated <= 1.0)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_optimizer.py::test_generate_evolution_points_supports_configured_strategies tests/test_optimizer.py::test_generate_evolution_points_falls_back_to_sobol_when_parent_pool_is_too_small -v
```

Expected: import failure for missing `_generate_evolution_points`.

- [ ] **Step 3: Implement strategy helper**

Add to `src/smco/optimizer.py`:

```python
def _sobol_replacements(
    n_new: int,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    return generate_sobol_points(n_new, bounds_lower, bounds_upper, seed=int(rng.integers(0, 2**31)))


def _binomial_crossover(
    base: np.ndarray,
    mutant: np.ndarray,
    de_crossover: float,
    rng: np.random.Generator,
) -> np.ndarray:
    dim = int(base.size)
    forced_j = int(rng.integers(0, dim))
    mask = rng.uniform(size=dim) < de_crossover
    mask[forced_j] = True
    return np.where(mask, mutant, base)


def _generate_evolution_points(
    parents: np.ndarray,
    scores: np.ndarray,
    *,
    n_new: int,
    strategy: str,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    de_factor: float,
    de_crossover: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_new = int(n_new)
    if n_new < 1:
        return np.empty((0, bounds_lower.size), dtype=float)
    parents = np.asarray(parents, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if strategy == "sobol" or parents.shape[0] < 3:
        return _sobol_replacements(n_new, bounds_lower, bounds_upper, rng)

    best = parents[int(np.argmax(scores))]
    children: list[np.ndarray] = []
    for _ in range(n_new):
        base_index = int(rng.integers(0, parents.shape[0]))
        base = parents[base_index]
        if strategy == "rand1bin":
            a_idx, b_idx, c_idx = rng.choice(parents.shape[0], size=3, replace=False)
            mutant = parents[a_idx] + de_factor * (parents[b_idx] - parents[c_idx])
        elif strategy == "current-to-best1bin":
            choices = [idx for idx in range(parents.shape[0]) if idx != base_index]
            b_idx, c_idx = rng.choice(choices, size=2, replace=False)
            mutant = base + de_factor * (best - base) + de_factor * (parents[b_idx] - parents[c_idx])
        elif strategy == "best1bin":
            b_idx, c_idx = rng.choice(parents.shape[0], size=2, replace=False)
            mutant = best + de_factor * (parents[b_idx] - parents[c_idx])
        else:
            raise ValueError("evolution_strategy must be one of current-to-best1bin, rand1bin, best1bin, sobol")
        child = _binomial_crossover(base, mutant, de_crossover, rng)
        children.append(np.clip(child, bounds_lower, bounds_upper))
    return np.vstack(children)
```

- [ ] **Step 4: Run tests and verify Task 3 passes**

Run:

```bash
.venv/bin/python -m pytest tests/test_optimizer.py::test_generate_evolution_points_supports_configured_strategies tests/test_optimizer.py::test_generate_evolution_points_falls_back_to_sobol_when_parent_pool_is_too_small -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/smco/optimizer.py tests/test_optimizer.py
git commit -m "feat: add evolutionary replacement strategies"
```

---

### Task 4: Evolutionary Multi-Start Scheduler

**Files:**
- Modify: `tests/test_optimizer.py`
- Modify: `src/smco/optimizer.py`

- [ ] **Step 1: Write failing scheduler tests**

Add tests:

```python
def test_smco_evo_records_evolution_history_and_uses_rand1bin_by_default():
    result = smco_evo(
        lambda x: -float(np.sum((x - 0.1) ** 2)),
        [-1.0, -1.0],
        [1.0, 1.0],
        n_starts=8,
        iter_max=40,
        evolution_points=(0.5, 0.75),
        elimination_rate=0.25,
        seed=123,
        tol_conv=1e-12,
    )

    history = result.summary["evolution_history"]
    assert result.summary["evolution_strategy"] == "rand1bin"
    assert result.summary["evolution_points"] == (0.5, 0.75)
    assert len(history) == 2
    assert [event["iteration"] for event in history] == [20, 30]
    assert all(event["eliminated_count"] == 2 for event in history)
    assert all(event["generated_count"] == 2 for event in history)
    assert len(result.all_results) == 8


def test_smco_evo_is_reproducible_with_same_seed():
    def objective(x):
        return -float(np.sum((x + 0.15) ** 2))

    first = smco_evo(objective, [-2.0, -2.0], [2.0, 2.0], n_starts=8, iter_max=35, seed=321)
    second = smco_evo(objective, [-2.0, -2.0], [2.0, 2.0], n_starts=8, iter_max=35, seed=321)

    assert np.allclose(first.best_result.x_optimal, second.best_result.x_optimal)
    assert first.best_result.f_optimal == pytest.approx(second.best_result.f_optimal)
    assert np.allclose(first.summary["endpoints"], second.summary["endpoints"])
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_optimizer.py::test_smco_evo_records_evolution_history_and_uses_rand1bin_by_default tests/test_optimizer.py::test_smco_evo_is_reproducible_with_same_seed -v
```

Expected: failure because temporary wrappers do not populate `evolution_history`.

- [ ] **Step 3: Implement evolutionary state runner**

Add to `src/smco/optimizer.py`:

```python
def _evolution_boundaries(iter_max: int, evolution_points: tuple[float, ...]) -> list[int]:
    boundaries = [max(1, int(round(iter_max * point))) for point in evolution_points]
    return sorted(set(boundary for boundary in boundaries if boundary < iter_max))


def _run_evolutionary_states(
    f: Objective,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    starts: np.ndarray,
    control: dict[str, Any],
    *,
    evolution_points: tuple[float, ...],
    elimination_rate: float,
    evolution_strategy: str,
    de_factor: float,
    de_crossover: float,
    iter_max: int,
    iter_boost: int,
    rng: np.random.Generator,
) -> tuple[list[SingleResult], list[dict[str, Any]]]:
    states = [
        _initialize_smco_state(
            f,
            bounds_lower,
            bounds_upper,
            np.asarray(start, dtype=float),
            bounds_buffer=control["bounds_buffer"],
            buffer_rand=bool(control["buffer_rand"]),
            iter_nstart=control["iter_nstart"],
            iter_boost=iter_boost,
            use_runmax=bool(control["use_runmax"]),
        )
        for start in starts
    ]
    history: list[dict[str, Any]] = []
    for boundary in _evolution_boundaries(iter_max, evolution_points):
        for state in states:
            _run_smco_state_until(
                state,
                f,
                bounds_lower,
                bounds_upper,
                control["bounds_buffer"],
                bool(control["buffer_rand"]),
                boundary,
                control["tol_conv"],
                str(control["partial_option"]),
                bool(control["use_runmax"]),
                rng,
            )
            _clip_result_to_bounds(state_result := state.to_result(), f, bounds_lower, bounds_upper)
            state.x_current = np.array(state_result.x_optimal, copy=True)
            state.f_current = float(state_result.f_optimal)
            state.x_runmax = None if state_result.x_runmax is None else np.array(state_result.x_runmax, copy=True)
            state.f_runmax = state_result.f_runmax

        ranked = sorted(states, key=lambda state: state.ranking_value(), reverse=True)
        n_eliminate = min(len(ranked) - 1, max(1, int(math.ceil(len(ranked) * elimination_rate))))
        survivors = ranked[: len(ranked) - n_eliminate]
        eliminated = ranked[len(ranked) - n_eliminate :]
        parents = np.vstack([state.ranking_point() for state in survivors])
        scores = np.array([state.ranking_value() for state in survivors], dtype=float)
        generated = _generate_evolution_points(
            parents,
            scores,
            n_new=len(eliminated),
            strategy=evolution_strategy,
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            de_factor=de_factor,
            de_crossover=de_crossover,
            rng=rng,
        )
        replacements = [
            _initialize_smco_state(
                f,
                bounds_lower,
                bounds_upper,
                point,
                bounds_buffer=control["bounds_buffer"],
                buffer_rand=bool(control["buffer_rand"]),
                iter_nstart=control["iter_nstart"],
                iter_boost=iter_boost + boundary,
                use_runmax=bool(control["use_runmax"]),
            )
            for point in generated
        ]
        best_before = float(ranked[0].ranking_value())
        states = survivors + replacements
        history.append(
            {
                "iteration": int(boundary),
                "strategy": evolution_strategy,
                "survivor_count": len(survivors),
                "eliminated_count": len(eliminated),
                "generated_count": int(generated.shape[0]),
                "best_before": best_before,
                "best_after_generation": float(max(state.ranking_value() for state in states)),
            }
        )

    for state in states:
        _run_smco_state_until(
            state,
            f,
            bounds_lower,
            bounds_upper,
            control["bounds_buffer"],
            bool(control["buffer_rand"]),
            iter_max,
            control["tol_conv"],
            str(control["partial_option"]),
            bool(control["use_runmax"]),
            rng,
        )
        result = state.to_result()
        _clip_result_to_bounds(result, f, bounds_lower, bounds_upper)
        state.x_current = np.array(result.x_optimal, copy=True)
        state.f_current = float(result.f_optimal)
        state.x_runmax = None if result.x_runmax is None else np.array(result.x_runmax, copy=True)
        state.f_runmax = result.f_runmax

    results = [state.to_result() for state in states]
    if bool(control["use_runmax"]):
        for result in results:
            _promote_runmax(result)
    return results, history
```

- [ ] **Step 4: Implement `smco_evo_multi`**

Add to `src/smco/optimizer.py` before public wrappers:

```python
def smco_evo_multi(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    opt_control: dict[str, Any] | None = None,
    *,
    evolution_points: Any = (0.5, 0.75),
    elimination_rate: float = 0.25,
    evolution_strategy: str = "rand1bin",
    de_factor: float = 0.8,
    de_crossover: float = 0.7,
) -> SMCOResult:
    points, rate, strategy, factor, crossover = _validate_evolution_control(
        evolution_points,
        elimination_rate,
        evolution_strategy,
        de_factor,
        de_crossover,
    )
    lower, upper, starts = validate_smco_inputs(f, bounds_lower, bounds_upper, start_points, opt_control)
    control = _merge_control(opt_control)
    dim = int(lower.size)
    if control.get("n_starts") is None:
        control["n_starts"] = max(5, round(math.sqrt(dim)))
    control["n_starts"] = _as_positive_integer("n_starts", control["n_starts"])
    control["iter_max"] = _as_positive_integer("iter_max", control["iter_max"])
    control["iter_boost"] = _as_non_negative_integer("iter_boost", control["iter_boost"])
    control["bounds_buffer"] = float(control["bounds_buffer"])
    control["tol_conv"] = float(control["tol_conv"])
    control["refine_ratio"] = float(control["refine_ratio"])
    if control["seed"] is not None:
        control["seed"] = _as_integer("seed", control["seed"])
    if starts is None:
        starts = generate_sobol_points(control["n_starts"], lower, upper, control["seed"])
    else:
        control["n_starts"] = int(starts.shape[0])
    if control.get("iter_nstart") is None:
        control["iter_nstart"] = control["n_starts"]
    control["iter_nstart"] = _as_positive_integer("iter_nstart", control["iter_nstart"])

    rng = np.random.default_rng(control["seed"])
    results, evolution_history = _run_evolutionary_states(
        f,
        lower,
        upper,
        starts,
        control,
        evolution_points=points,
        elimination_rate=rate,
        evolution_strategy=strategy,
        de_factor=factor,
        de_crossover=crossover,
        iter_max=control["iter_max"],
        iter_boost=control["iter_boost"],
        rng=rng,
    )
    values = np.array([result.f_optimal for result in results], dtype=float)
    best_idx = int(np.argmax(values))
    endpoints = np.vstack([result.x_optimal for result in results])
    iterations = np.array([result.iterations for result in results], dtype=float)
    summary = {
        "n_starts": control["n_starts"],
        "mean_iterations": float(np.mean(iterations)),
        "std_values": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "endpoints": endpoints,
        "values": values,
        "evolution_history": evolution_history,
        "evolution_strategy": strategy,
        "evolution_points": points,
        "elimination_rate": rate,
    }
    return SMCOResult(best_result=results[best_idx], all_results=results, opt_control=control, summary=summary)
```

- [ ] **Step 5: Wire `smco_evo` to scheduler**

Replace temporary `smco_evo(...)` body with:

```python
    control = dict(kwargs)
    evolution_points = control.pop("evolution_points", (0.5, 0.75))
    elimination_rate = control.pop("elimination_rate", 0.25)
    evolution_strategy = control.pop("evolution_strategy", "rand1bin")
    de_factor = control.pop("de_factor", 0.8)
    de_crossover = control.pop("de_crossover", 0.7)
    control["refine_search"] = False
    control["iter_boost"] = 0
    return smco_evo_multi(
        f,
        bounds_lower,
        bounds_upper,
        start_points,
        control,
        evolution_points=evolution_points,
        elimination_rate=elimination_rate,
        evolution_strategy=evolution_strategy,
        de_factor=de_factor,
        de_crossover=de_crossover,
    )
```

- [ ] **Step 6: Run tests and verify Task 4 passes**

Run:

```bash
.venv/bin/python -m pytest tests/test_optimizer.py::test_smco_evo_records_evolution_history_and_uses_rand1bin_by_default tests/test_optimizer.py::test_smco_evo_is_reproducible_with_same_seed -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add src/smco/optimizer.py tests/test_optimizer.py
git commit -m "feat: add evolutionary SMCO scheduler"
```

---

### Task 5: Refine and Boost Wrappers

**Files:**
- Modify: `tests/test_optimizer.py`
- Modify: `src/smco/optimizer.py`

- [ ] **Step 1: Write failing wrapper behavior tests**

Add tests:

```python
def test_smco_r_evo_runs_refine_search_and_records_evolution_metadata():
    result = smco_r_evo(
        lambda x: -float(np.sum((x - 0.25) ** 2)),
        [-1.0, -1.0],
        [1.0, 1.0],
        n_starts=8,
        iter_max=60,
        refine_ratio=0.5,
        seed=456,
        tol_conv=1e-12,
    )

    assert result.opt_control["refine_search"] is True
    assert result.summary["evolution_strategy"] == "rand1bin"
    assert len(result.summary["evolution_history"]) >= 1
    assert np.isfinite(result.best_result.f_optimal)


def test_smco_br_evo_preserves_iter_boost_control():
    result = smco_br_evo(
        lambda x: -float(np.sum((x - 0.3) ** 2)),
        [-1.0, -1.0],
        [1.0, 1.0],
        n_starts=6,
        iter_max=40,
        iter_boost=20,
        seed=789,
    )

    assert result.opt_control["iter_boost"] == 20
    assert result.summary["evolution_strategy"] == "rand1bin"
    assert np.isfinite(result.best_result.f_optimal)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_optimizer.py::test_smco_r_evo_runs_refine_search_and_records_evolution_metadata tests/test_optimizer.py::test_smco_br_evo_preserves_iter_boost_control -v
```

Expected: failure because temporary wrappers still call `smco_multi`.

- [ ] **Step 3: Add evolutionary refine helper**

Add to `src/smco/optimizer.py`:

```python
def _refine_evolutionary_results(
    f: Objective,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    results: list[SingleResult],
    control: dict[str, Any],
    iter_max_refine: int,
    rng: np.random.Generator,
) -> list[SingleResult]:
    refined: list[SingleResult] = []
    for result in results:
        if (
            bool(control["use_runmax"])
            and result.x_runmax is not None
            and result.f_runmax is not None
            and result.f_runmax > result.f_optimal
        ):
            start_point_refine = np.array(result.x_runmax, copy=True)
        else:
            start_point_refine = np.array(result.x_optimal, copy=True)
        refine_result = _single(
            f,
            bounds_lower,
            bounds_upper,
            start_point_refine,
            bounds_buffer=0.0,
            buffer_rand=bool(control["buffer_rand"]),
            iter_max=iter_max_refine,
            iter_nstart=control["iter_nstart"],
            iter_boost=control["iter_boost"] + 1000,
            tol_conv=control["tol_conv"],
            partial_option=str(control["partial_option"]),
            use_runmax=bool(control["use_runmax"]),
            rng=rng,
        )
        _clip_result_to_bounds(refine_result, f, bounds_lower, bounds_upper)
        if bool(control["use_runmax"]):
            _promote_runmax(refine_result)
        refined.append(refine_result)
    return refined
```

- [ ] **Step 4: Update `smco_evo_multi` refine handling**

Inside `smco_evo_multi(...)`, before `_run_evolutionary_states(...)`, compute:

```python
    iter_max_initial, iter_max_refine = _split_refine_iterations(
        control["iter_max"],
        control["refine_ratio"],
        bool(control["refine_search"]),
    )
```

Pass `iter_max=iter_max_initial` to `_run_evolutionary_states(...)`.

After `_run_evolutionary_states(...)`, add:

```python
    if bool(control["refine_search"]) and iter_max_refine > 0:
        results = _refine_evolutionary_results(f, lower, upper, results, control, iter_max_refine, rng)
```

- [ ] **Step 5: Wire `smco_r_evo` and `smco_br_evo` to scheduler**

Replace temporary wrapper bodies with the same `control.pop(...)` pattern used by `smco_evo`, setting:

```python
control["refine_search"] = True
control["iter_boost"] = 0
control.setdefault("refine_ratio", 0.5)
```

for `smco_r_evo`, and:

```python
control["refine_search"] = True
control["iter_boost"] = iter_boost
control.setdefault("refine_ratio", 0.5)
```

for `smco_br_evo`.

- [ ] **Step 6: Run tests and verify Task 5 passes**

Run:

```bash
.venv/bin/python -m pytest tests/test_optimizer.py::test_smco_r_evo_runs_refine_search_and_records_evolution_metadata tests/test_optimizer.py::test_smco_br_evo_preserves_iter_boost_control -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

```bash
git add src/smco/optimizer.py tests/test_optimizer.py
git commit -m "feat: add refined evolutionary SMCO wrappers"
```

---

### Task 6: Documentation and Full Verification

**Files:**
- Modify: `docs/usage.md`
- Modify: `tests/test_optimizer.py`

- [ ] **Step 1: Add a final smoke test for optimization quality**

Add to `tests/test_optimizer.py`:

```python
def test_smco_evo_optimizes_concave_quadratic_near_origin():
    result = smco_evo(
        lambda x: -float(np.sum(x**2)),
        [-5.0, -5.0],
        [5.0, 5.0],
        n_starts=10,
        iter_max=100,
        evolution_points=(0.5, 0.75),
        elimination_rate=0.25,
        seed=123,
        tol_conv=1e-12,
    )

    assert result.best_result.f_optimal > -0.2
    assert np.linalg.norm(result.best_result.x_optimal) < 0.6
    assert len(result.summary["evolution_history"]) == 2
```

- [ ] **Step 2: Run focused optimizer tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_optimizer.py -v
```

Expected: PASS.

- [ ] **Step 3: Add usage documentation**

In `docs/usage.md`, add a section after the existing SMCO algorithm overview:

```markdown
### Evolutionary Multi-Start Variants

`smco_evo()`, `smco_r_evo()`, and `smco_br_evo()` add population-level elimination and regeneration to the original multi-start SMCO flow. At scheduled iteration progress points, the worst trajectories are removed by `f_runmax`, survivors keep their internal SMCO state, and replacement trajectories are initialized from differential-evolution or Sobol-generated points.

```python
from smco import smco_evo

result = smco_evo(
    objective,
    bounds_lower=[-5, -5],
    bounds_upper=[5, 5],
    n_starts=8,
    iter_max=200,
    evolution_points=(0.5, 0.75),
    elimination_rate=0.25,
    evolution_strategy="rand1bin",
    seed=123,
)
```

Supported `evolution_strategy` values are `"rand1bin"` (default), `"current-to-best1bin"`, `"best1bin"`, and `"sobol"`.
```

- [ ] **Step 4: Run full test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: PASS.

- [ ] **Step 5: Check worktree and diff**

Run:

```bash
git status --short
git diff --check
git diff --stat
```

Expected: only intentional files changed, no whitespace errors.

- [ ] **Step 6: Commit Task 6**

```bash
git add src/smco/optimizer.py src/smco/__init__.py tests/test_optimizer.py docs/usage.md
git commit -m "docs: add evolutionary SMCO usage"
```

---

## Self-Review

- Spec coverage: public wrappers, default `rand1bin`, four strategies, state-preserving survivors, fresh replacement states, `f_runmax` elimination, existing `SMCOResult`, summary metadata, and validation tests are covered by Tasks 1-6.
- Type consistency: all new public wrappers return `SMCOResult`; internal state uses `SMCOState`; strategy names match the spec.
- Scope: implementation is isolated to original optimizer flow and does not reuse deprecated `smco_hb`.
