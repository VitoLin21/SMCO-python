# SMCO Python Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal runnable Python migration of the R SMCO project, with a v1.1.0-style public API and a v1.0.0 paper replication path.

**Architecture:** Implement a small installable Python package under `src/smco`. Keep the optimizer core in `optimizer.py`, result containers in `results.py`, benchmark functions in `test_functions.py`, benchmark runners in `benchmark.py`, and paper-oriented smoke entry points in `paper.py`. Vendor the upstream R sources for reference only; Python tests do not execute R.

**Tech Stack:** Python 3.10+, `numpy`, `scipy`, `pytest`, optional standard-library `concurrent.futures` for future parallel execution.

---

## File Map

- `pyproject.toml`: editable package metadata and dependencies.
- `README.md`: install, quick start, benchmark, and paper smoke instructions.
- `src/smco/__init__.py`: public API exports.
- `src/smco/results.py`: dataclasses for single-start and multi-start results.
- `src/smco/optimizer.py`: SMCO helper functions, single-start loop, refinement/boosting, multi-start wrappers.
- `src/smco/test_functions.py`: benchmark functions and `BenchmarkConfig` helpers.
- `src/smco/benchmark.py`: small configurable benchmark runner.
- `src/smco/paper.py`: paper-oriented smoke configurations and runner.
- `tests/test_validation.py`: validation, bounds, Sobol, and partial sign tests.
- `tests/test_optimizer.py`: core optimizer and wrapper behavior tests.
- `tests/test_test_functions.py`: benchmark function value/config tests.
- `tests/test_benchmark.py`: benchmark and paper smoke tests.
- `vendor/SMCO_R/main/*`: upstream v1.1.0 source reference.
- `vendor/SMCO_R/v1.0.0/*`: upstream v1.0.0 paper replication source reference.

Current workspace note: `git status` fails because `/amax/math/code/SMCO` is not a valid git repository. Commit steps are included for when the repository is initialized or repaired; until then, run the stated verification commands and skip the commit command.

---

### Task 1: Package Scaffold And Result Types

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/smco/__init__.py`
- Create: `src/smco/results.py`
- Test: `tests/test_optimizer.py`

- [ ] **Step 1: Write the failing public API and result tests**

Create `tests/test_optimizer.py` with:

```python
import numpy as np

from smco import SMCOResult, SingleResult, smco, smco_br, smco_multi, smco_r


def test_public_api_exports_expected_symbols():
    assert callable(smco)
    assert callable(smco_r)
    assert callable(smco_br)
    assert callable(smco_multi)


def test_single_result_fields_are_numpy_friendly():
    result = SingleResult(
        x_optimal=np.array([1.0, 2.0]),
        f_optimal=3.0,
        iterations=7,
        x_runmax=np.array([1.5, 2.5]),
        f_runmax=4.0,
    )

    assert np.allclose(result.x_optimal, [1.0, 2.0])
    assert result.f_optimal == 3.0
    assert result.iterations == 7
    assert np.allclose(result.x_runmax, [1.5, 2.5])
    assert result.f_runmax == 4.0


def test_smco_result_groups_best_all_control_and_summary():
    single = SingleResult(x_optimal=np.array([0.0]), f_optimal=0.0, iterations=1)
    result = SMCOResult(
        best_result=single,
        all_results=[single],
        opt_control={"iter_max": 1},
        summary={"n_starts": 1},
    )

    assert result.best_result is single
    assert result.all_results == [single]
    assert result.opt_control["iter_max"] == 1
    assert result.summary["n_starts"] == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
pytest tests/test_optimizer.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'smco'`.

- [ ] **Step 3: Create package metadata**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "smco"
version = "0.1.0"
description = "Python migration of Strategic Monte Carlo Optimization"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.23",
  "scipy>=1.10",
]

[project.optional-dependencies]
test = ["pytest>=7"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 4: Create the initial README**

Create `README.md`:

```markdown
# SMCO Python

Minimal Python migration of the Strategic Monte Carlo Optimization algorithms from the R project `wayne-y-gao/SMCO`.

## Install

```bash
pip install -e ".[test]"
```

## Quick Start

```python
import numpy as np
from smco import smco_r

def objective(x):
    return -float(np.sum(x ** 2))

result = smco_r(objective, [-5, -5], [5, 5], iter_max=200, seed=123)
print(result.best_result.x_optimal)
print(result.best_result.f_optimal)
```

SMCO maximizes by default. To minimize a function, pass its negative.

## Paper Replication

The package includes small paper-oriented smoke runners. Full paper-scale experiments require increasing dimensions, repetitions, iteration counts, and optional parallel settings manually.

The upstream R source is vendored under `vendor/SMCO_R` for reference only. Tests do not require R.
```
```

- [ ] **Step 5: Implement result containers and placeholder public functions**

Create `src/smco/results.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SingleResult:
    x_optimal: np.ndarray
    f_optimal: float
    iterations: int
    x_runmax: np.ndarray | None = None
    f_runmax: float | None = None


@dataclass
class SMCOResult:
    best_result: SingleResult
    all_results: list[SingleResult]
    opt_control: dict[str, Any]
    summary: dict[str, Any]
```

Create `src/smco/__init__.py`:

```python
from .optimizer import smco, smco_br, smco_multi, smco_r
from .results import SMCOResult, SingleResult

__all__ = [
    "SMCOResult",
    "SingleResult",
    "smco",
    "smco_br",
    "smco_multi",
    "smco_r",
]
```

Create initial `src/smco/optimizer.py`:

```python
from __future__ import annotations


def smco(*args, **kwargs):
    raise NotImplementedError("smco is implemented in Task 3")


def smco_r(*args, **kwargs):
    raise NotImplementedError("smco_r is implemented in Task 3")


def smco_br(*args, **kwargs):
    raise NotImplementedError("smco_br is implemented in Task 3")


def smco_multi(*args, **kwargs):
    raise NotImplementedError("smco_multi is implemented in Task 3")
```

- [ ] **Step 6: Run scaffold tests**

Run:

```bash
pytest tests/test_optimizer.py -q
```

Expected: PASS for the three scaffold tests.

- [ ] **Step 7: Commit when git is available**

Run:

```bash
git status --short
git add pyproject.toml README.md src/smco/__init__.py src/smco/results.py src/smco/optimizer.py tests/test_optimizer.py
git commit -m "chore: scaffold smco python package"
```

Expected in current workspace: `git status` fails because the repository is invalid. Skip commit until git is repaired.

---

### Task 2: Optimizer Helpers

**Files:**
- Modify: `src/smco/optimizer.py`
- Test: `tests/test_validation.py`

- [ ] **Step 1: Write failing helper tests**

Create `tests/test_validation.py`:

```python
import numpy as np
import pytest

from smco.optimizer import (
    check_bounds,
    compute_partial_signs,
    generate_sobol_points,
    validate_smco_inputs,
)


def test_validate_rejects_non_callable_objective():
    with pytest.raises(TypeError, match="f must be callable"):
        validate_smco_inputs(12, [0.0], [1.0], None, {})


def test_validate_rejects_mismatched_bounds():
    with pytest.raises(ValueError, match="same length"):
        validate_smco_inputs(lambda x: 0.0, [0.0], [1.0, 2.0], None, {})


def test_validate_rejects_non_increasing_bounds():
    with pytest.raises(ValueError, match="strictly less"):
        validate_smco_inputs(lambda x: 0.0, [1.0], [1.0], None, {})


def test_validate_rejects_bad_start_point_shape():
    with pytest.raises(ValueError, match="2 columns"):
        validate_smco_inputs(lambda x: 0.0, [0.0, 0.0], [1.0, 1.0], [[0.5]], {})


def test_check_bounds_clips_out_of_range_values():
    checked = check_bounds(
        np.array([-1.0, 0.5, 3.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
    )

    assert checked.is_out is True
    assert np.allclose(checked.x_in, [0.0, 0.5, 1.0])


def test_generate_sobol_points_is_reproducible_and_inside_bounds():
    lower = np.array([-2.0, 10.0])
    upper = np.array([2.0, 20.0])

    first = generate_sobol_points(5, lower, upper, seed=123)
    second = generate_sobol_points(5, lower, upper, seed=123)

    assert first.shape == (5, 2)
    assert np.all(first >= lower)
    assert np.all(first <= upper)
    assert np.allclose(first, second)


def test_compute_partial_signs_center_tracks_best_perturbation():
    def objective(x):
        return -float((x[0] - 0.75) ** 2 + (x[1] + 0.25) ** 2)

    x = np.array([0.0, 0.0])
    result = compute_partial_signs(
        objective,
        x,
        objective(x),
        h_step=np.array([0.1, 0.1]),
        bounds_lower=np.array([-1.0, -1.0]),
        bounds_upper=np.array([1.0, 1.0]),
        partial_option="center",
        use_runmax=True,
    )

    assert np.allclose(result.signs, [1.0, 0.0])
    assert result.f_partial_best > objective(x)
    assert np.allclose(x, [0.0, 0.0])


def test_compute_partial_signs_forward_handles_upper_boundary():
    def objective(x):
        return -float((x[0] - 0.5) ** 2)

    x = np.array([1.0])
    result = compute_partial_signs(
        objective,
        x,
        objective(x),
        h_step=np.array([0.1]),
        bounds_lower=np.array([0.0]),
        bounds_upper=np.array([1.0]),
        partial_option="forward",
        use_runmax=False,
    )

    assert np.allclose(result.signs, [1.0])
```

- [ ] **Step 2: Run helper tests to verify they fail**

Run:

```bash
pytest tests/test_validation.py -q
```

Expected: FAIL because helper functions are not defined.

- [ ] **Step 3: Implement helper dataclasses and functions**

Replace `src/smco/optimizer.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import math

import numpy as np
from scipy.stats import qmc

from .results import SMCOResult, SingleResult

Objective = Callable[[np.ndarray], float]


@dataclass
class BoundsCheck:
    is_out: bool
    x_in: np.ndarray


@dataclass
class PartialSignResult:
    signs: np.ndarray
    f_partial_best: float
    x_partial_best: np.ndarray


def _as_1d_float_array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional numeric vector")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_start_points(start_points: Any, dim: int) -> np.ndarray:
    array = np.asarray(start_points, dtype=float)
    if array.ndim == 1:
        if dim == 1:
            array = array.reshape(-1, 1)
        else:
            array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("start_points must be a two-dimensional array")
    if array.shape[1] != dim:
        raise ValueError(f"start_points must have {dim} columns")
    if array.shape[0] < 1:
        raise ValueError("start_points must have at least one row")
    if not np.all(np.isfinite(array)):
        raise ValueError("start_points must contain only finite values")
    return array


def validate_smco_inputs(
    f: Any,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    opt_control: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not callable(f):
        raise TypeError("f must be callable")
    lower = _as_1d_float_array("bounds_lower", bounds_lower)
    upper = _as_1d_float_array("bounds_upper", bounds_upper)
    if lower.shape != upper.shape:
        raise ValueError("bounds_lower and bounds_upper must have the same length")
    if np.any(lower >= upper):
        raise ValueError("bounds_lower must be strictly less than bounds_upper")

    starts = None
    if start_points is not None:
        starts = _as_start_points(start_points, lower.size)

    control = opt_control or {}
    if control.get("n_starts") is not None and int(control["n_starts"]) < 1:
        raise ValueError("n_starts must be a positive integer")
    if control.get("iter_max") is not None and int(control["iter_max"]) < 1:
        raise ValueError("iter_max must be a positive integer")
    if control.get("bounds_buffer") is not None and float(control["bounds_buffer"]) < 0:
        raise ValueError("bounds_buffer must be non-negative")
    if control.get("refine_ratio") is not None:
        ratio = float(control["refine_ratio"])
        if ratio < 0 or ratio > 1:
            raise ValueError("refine_ratio must be between 0 and 1")
    if control.get("partial_option") is not None and control["partial_option"] not in {"center", "forward"}:
        raise ValueError("partial_option must be 'center' or 'forward'")

    return lower, upper, starts


def generate_sobol_points(
    n_starts: int,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    n_starts = int(n_starts)
    dim = int(bounds_lower.size)
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    m = int(math.ceil(math.log2(max(1, n_starts))))
    points = sampler.random_base2(m=m)[:n_starts]
    return qmc.scale(points, bounds_lower, bounds_upper)


def check_bounds(
    x: np.ndarray,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
) -> BoundsCheck:
    is_out = bool(np.any((x > bounds_upper) | (x < bounds_lower)))
    return BoundsCheck(is_out=is_out, x_in=np.clip(x, bounds_lower, bounds_upper))


def compute_partial_signs(
    f: Objective,
    x: np.ndarray,
    fx: float,
    h_step: np.ndarray,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    partial_option: str = "center",
    use_runmax: bool = True,
) -> PartialSignResult:
    x_work = np.array(x, dtype=float, copy=True)
    partial_signs = np.zeros_like(x_work, dtype=float)
    f_partial_best = float(fx)
    best_j: int | None = None
    best_value: float | None = None

    if partial_option == "center":
        for j in range(x_work.size):
            original = float(x_work[j])
            x_work[j] = min(original + h_step[j], bounds_upper[j])
            f_plus = float(f(x_work))
            x_work[j] = max(original - h_step[j], bounds_lower[j])
            f_minus = float(f(x_work))
            sign = f_plus > f_minus
            partial_signs[j] = 1.0 if sign else 0.0

            if use_runmax:
                if sign and f_plus > f_partial_best:
                    f_partial_best = f_plus
                    best_j = j
                    best_value = min(original + h_step[j], bounds_upper[j])
                elif (not sign) and f_minus > f_partial_best:
                    f_partial_best = f_minus
                    best_j = j
                    best_value = max(original - h_step[j], bounds_lower[j])
            x_work[j] = original
    elif partial_option == "forward":
        for j in range(x_work.size):
            original = float(x_work[j])
            if original >= bounds_upper[j]:
                x_work[j] = max(original - h_step[j], bounds_lower[j])
                f_perturb = float(f(x_work))
                sign = fx > f_perturb
            else:
                x_work[j] = min(original + h_step[j], bounds_upper[j])
                f_perturb = float(f(x_work))
                sign = f_perturb > fx
            partial_signs[j] = 1.0 if sign else 0.0

            if use_runmax and f_perturb > f_partial_best:
                f_partial_best = f_perturb
                best_j = j
                best_value = float(x_work[j])
            x_work[j] = original
    else:
        raise ValueError("partial_option must be 'center' or 'forward'")

    x_partial_best = np.array(x, dtype=float, copy=True)
    if best_j is not None and best_value is not None:
        x_partial_best[best_j] = best_value
    return PartialSignResult(
        signs=partial_signs,
        f_partial_best=f_partial_best,
        x_partial_best=x_partial_best,
    )


def smco(*args, **kwargs):
    raise NotImplementedError("smco is implemented in Task 3")


def smco_r(*args, **kwargs):
    raise NotImplementedError("smco_r is implemented in Task 3")


def smco_br(*args, **kwargs):
    raise NotImplementedError("smco_br is implemented in Task 3")


def smco_multi(*args, **kwargs):
    raise NotImplementedError("smco_multi is implemented in Task 3")
```

- [ ] **Step 4: Run helper tests**

Run:

```bash
pytest tests/test_validation.py -q
```

Expected: PASS.

- [ ] **Step 5: Run scaffold and helper tests together**

Run:

```bash
pytest tests/test_optimizer.py tests/test_validation.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit when git is available**

Run:

```bash
git status --short
git add src/smco/optimizer.py tests/test_validation.py
git commit -m "feat: add smco optimizer helpers"
```

Expected in current workspace: `git status` fails. Skip commit until git is repaired.

---

### Task 3: Core SMCO Algorithm And Wrappers

**Files:**
- Modify: `src/smco/optimizer.py`
- Modify: `tests/test_optimizer.py`

- [ ] **Step 1: Add failing optimizer behavior tests**

Append to `tests/test_optimizer.py`:

```python

def test_smco_r_optimizes_concave_quadratic_near_origin():
    def objective(x):
        return -float(np.sum(x ** 2))

    result = smco_r(
        objective,
        [-5.0, -5.0],
        [5.0, 5.0],
        n_starts=6,
        iter_max=180,
        seed=123,
        tol_conv=1e-12,
    )

    assert result.best_result.f_optimal > -0.05
    assert np.linalg.norm(result.best_result.x_optimal) < 0.25
    assert result.summary["n_starts"] == 6
    assert len(result.all_results) == 6


def test_smco_basic_returns_points_inside_bounds():
    def objective(x):
        return -float((x[0] - 0.2) ** 2)

    result = smco(
        objective,
        [-1.0],
        [1.0],
        start_points=[[0.9]],
        iter_max=80,
        seed=1,
    )

    x = result.best_result.x_optimal
    assert x.shape == (1,)
    assert -1.0 <= x[0] <= 1.0


def test_smco_br_uses_boosted_search_and_returns_valid_structure():
    def objective(x):
        return -float(np.sum((x - 0.3) ** 2))

    result = smco_br(
        objective,
        [-1.0, -1.0],
        [1.0, 1.0],
        n_starts=5,
        iter_max=120,
        iter_boost=100,
        seed=456,
    )

    assert result.best_result.f_optimal > -0.1
    assert isinstance(result.opt_control, dict)
    assert result.opt_control["iter_boost"] == 100
    assert np.all(result.summary["endpoints"] >= -1.0)
    assert np.all(result.summary["endpoints"] <= 1.0)


def test_smco_multi_is_reproducible_with_same_seed():
    def objective(x):
        return -float(np.sum((x + 0.1) ** 2))

    first = smco_multi(objective, [-2.0, -2.0], [2.0, 2.0], opt_control={"iter_max": 80, "seed": 99})
    second = smco_multi(objective, [-2.0, -2.0], [2.0, 2.0], opt_control={"iter_max": 80, "seed": 99})

    assert np.allclose(first.best_result.x_optimal, second.best_result.x_optimal)
    assert first.best_result.f_optimal == second.best_result.f_optimal
```

- [ ] **Step 2: Run optimizer tests to verify they fail**

Run:

```bash
pytest tests/test_optimizer.py -q
```

Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Add core algorithm functions**

Append these functions before the public wrapper functions in `src/smco/optimizer.py`, then replace the four placeholder public functions with the wrapper definitions shown here:

```python
DEFAULT_CONTROL: dict[str, Any] = {
    "iter_max": 500,
    "iter_boost": 0,
    "bounds_buffer": 0.05,
    "buffer_rand": False,
    "tol_conv": 1e-8,
    "refine_search": True,
    "refine_ratio": 0.5,
    "partial_option": "center",
    "use_runmax": True,
    "use_parallel": False,
    "verbose": False,
    "seed": 123,
}


def _merge_control(user_control: dict[str, Any] | None) -> dict[str, Any]:
    control = dict(DEFAULT_CONTROL)
    if user_control:
        control.update(user_control)
    return control


def _single(
    f: Objective,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_point: np.ndarray,
    *,
    bounds_buffer: float,
    buffer_rand: bool,
    iter_max: int,
    iter_nstart: int,
    iter_boost: int,
    tol_conv: float,
    partial_option: str,
    use_runmax: bool,
    rng: np.random.Generator,
) -> SingleResult:
    x_current = np.array(start_point, dtype=float, copy=True)
    f_current = float(f(x_current))
    f_runmax = f_current
    x_runmax = np.array(x_current, copy=True)

    bounds_diff = bounds_upper - bounds_lower
    n_boost_1 = int(iter_boost) + int(iter_nstart)
    n_boost_max = n_boost_1 + int(iter_max)
    s_value = x_current * n_boost_1

    if not buffer_rand:
        fixed_out = float(bounds_buffer) * bounds_diff
        fixed_upper_out = bounds_upper + fixed_out
        fixed_lower_out = bounds_lower - fixed_out

    iter_min_check = n_boost_1 + int(math.ceil(iter_max / 2))
    last_n = n_boost_1

    for n in range(n_boost_1, n_boost_max + 1):
        last_n = n
        h_step = bounds_diff / (n + 1)
        partial = compute_partial_signs(
            f,
            x_current,
            f_current,
            h_step,
            bounds_lower,
            bounds_upper,
            partial_option,
            use_runmax,
        )

        if buffer_rand:
            bound_out = float(bounds_buffer) * bounds_diff * rng.uniform(-1.0, 1.0, size=bounds_diff.size)
            bounds_upper_out = bounds_upper + bound_out
            bounds_lower_out = bounds_lower - bound_out
        else:
            bounds_upper_out = fixed_upper_out
            bounds_lower_out = fixed_lower_out

        z_value = partial.signs * bounds_upper_out + (1.0 - partial.signs) * bounds_lower_out
        s_value = s_value + z_value
        x_next = s_value / (n + 1)
        f_next = float(f(x_next))

        if use_runmax:
            f_next_best = max(partial.f_partial_best, f_next)
            if f_next_best > f_runmax:
                f_runmax = f_next_best
                x_runmax = np.array(partial.x_partial_best if partial.f_partial_best > f_next else x_next, copy=True)

        f_prev = f_current
        f_current = f_next
        x_current = x_next

        if n >= iter_min_check and abs(f_current - f_prev) < tol_conv:
            break

    result = SingleResult(
        x_optimal=np.array(x_current, copy=True),
        f_optimal=float(f_current),
        iterations=int(last_n - iter_boost),
    )
    if use_runmax:
        result.x_runmax = np.array(x_runmax, copy=True)
        result.f_runmax = float(f_runmax)
    return result


def _single_refine(
    f: Objective,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_point: np.ndarray,
    *,
    bounds_buffer: float,
    buffer_rand: bool,
    iter_max: int,
    iter_nstart: int,
    iter_boost: int,
    tol_conv: float,
    refine_search: bool,
    refine_ratio: float,
    partial_option: str,
    use_runmax: bool,
    rng: np.random.Generator,
) -> SingleResult:
    ratio = float(refine_ratio) if refine_search else 0.0
    iter_max_initial = int(round(iter_max * (1.0 - ratio)))
    iter_max_initial = max(1, iter_max_initial)

    result = _single(
        f,
        bounds_lower,
        bounds_upper,
        start_point,
        bounds_buffer=bounds_buffer,
        buffer_rand=buffer_rand,
        iter_max=iter_max_initial,
        iter_nstart=iter_nstart,
        iter_boost=iter_boost,
        tol_conv=tol_conv,
        partial_option=partial_option,
        use_runmax=use_runmax,
        rng=rng,
    )

    checked = check_bounds(result.x_optimal, bounds_lower, bounds_upper)
    if checked.is_out:
        result.x_optimal = checked.x_in
        result.f_optimal = float(f(checked.x_in))

    if use_runmax and result.x_runmax is not None:
        checked_runmax = check_bounds(result.x_runmax, bounds_lower, bounds_upper)
        if checked_runmax.is_out:
            result.x_runmax = checked_runmax.x_in
            result.f_runmax = float(f(checked_runmax.x_in))

    if not refine_search:
        if use_runmax and result.f_runmax is not None and result.f_runmax > result.f_optimal:
            result.x_optimal = np.array(result.x_runmax, copy=True)
            result.f_optimal = float(result.f_runmax)
        return result

    iter_max_refine = max(1, int(round(iter_max * ratio)))
    if use_runmax and result.f_runmax is not None and result.f_runmax > result.f_optimal:
        start_point_refine = np.array(result.x_runmax, copy=True)
    else:
        start_point_refine = np.array(result.x_optimal, copy=True)

    refine_result = _single(
        f,
        bounds_lower,
        bounds_upper,
        start_point_refine,
        bounds_buffer=0.0,
        buffer_rand=buffer_rand,
        iter_max=iter_max_refine,
        iter_nstart=iter_nstart,
        iter_boost=iter_boost + 1000,
        tol_conv=tol_conv,
        partial_option=partial_option,
        use_runmax=use_runmax,
        rng=rng,
    )

    checked_refine = check_bounds(refine_result.x_optimal, bounds_lower, bounds_upper)
    if checked_refine.is_out:
        refine_result.x_optimal = checked_refine.x_in
        refine_result.f_optimal = float(f(checked_refine.x_in))

    if use_runmax and refine_result.x_runmax is not None and refine_result.f_runmax is not None:
        checked_refine_runmax = check_bounds(refine_result.x_runmax, bounds_lower, bounds_upper)
        if checked_refine_runmax.is_out:
            refine_result.x_runmax = checked_refine_runmax.x_in
            refine_result.f_runmax = float(f(checked_refine_runmax.x_in))
        if refine_result.f_runmax > refine_result.f_optimal:
            refine_result.x_optimal = np.array(refine_result.x_runmax, copy=True)
            refine_result.f_optimal = float(refine_result.f_runmax)

    return refine_result


def _single_boost(
    f: Objective,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_point: np.ndarray,
    *,
    bounds_buffer: float,
    buffer_rand: bool,
    iter_max: int,
    iter_nstart: int,
    iter_boost: int,
    tol_conv: float,
    refine_search: bool,
    refine_ratio: float,
    partial_option: str,
    use_runmax: bool,
    rng: np.random.Generator,
) -> SingleResult:
    regular = _single_refine(
        f,
        bounds_lower,
        bounds_upper,
        start_point,
        bounds_buffer=bounds_buffer,
        buffer_rand=buffer_rand,
        iter_max=iter_max,
        iter_nstart=iter_nstart,
        iter_boost=0,
        tol_conv=tol_conv,
        refine_search=refine_search,
        refine_ratio=refine_ratio,
        partial_option=partial_option,
        use_runmax=use_runmax,
        rng=rng,
    )
    if iter_boost <= 0:
        return regular
    boosted = _single_refine(
        f,
        bounds_lower,
        bounds_upper,
        start_point,
        bounds_buffer=bounds_buffer,
        buffer_rand=buffer_rand,
        iter_max=iter_max,
        iter_nstart=iter_nstart,
        iter_boost=iter_boost,
        tol_conv=tol_conv,
        refine_search=refine_search,
        refine_ratio=refine_ratio,
        partial_option=partial_option,
        use_runmax=use_runmax,
        rng=rng,
    )
    return boosted if boosted.f_optimal > regular.f_optimal else regular


def smco_multi(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    opt_control: dict[str, Any] | None = None,
) -> SMCOResult:
    lower, upper, starts = validate_smco_inputs(f, bounds_lower, bounds_upper, start_points, opt_control)
    control = _merge_control(opt_control)
    dim = lower.size
    if control.get("n_starts") is None:
        control["n_starts"] = max(5, round(math.sqrt(dim)))
    control["n_starts"] = int(control["n_starts"])
    control["iter_max"] = int(control["iter_max"])
    control["iter_boost"] = int(control["iter_boost"])
    control["bounds_buffer"] = float(control["bounds_buffer"])
    control["tol_conv"] = float(control["tol_conv"])
    control["refine_ratio"] = float(control["refine_ratio"])

    if starts is None:
        starts = generate_sobol_points(control["n_starts"], lower, upper, control["seed"])
    else:
        control["n_starts"] = int(starts.shape[0])

    if control.get("iter_nstart") is None:
        control["iter_nstart"] = control["n_starts"]
    control["iter_nstart"] = int(control["iter_nstart"])

    rng = np.random.default_rng(control["seed"])
    results: list[SingleResult] = []
    for start in starts:
        results.append(
            _single_boost(
                f,
                lower,
                upper,
                np.asarray(start, dtype=float),
                bounds_buffer=control["bounds_buffer"],
                buffer_rand=bool(control["buffer_rand"]),
                iter_max=control["iter_max"],
                iter_nstart=control["iter_nstart"],
                iter_boost=control["iter_boost"],
                tol_conv=control["tol_conv"],
                refine_search=bool(control["refine_search"]),
                refine_ratio=control["refine_ratio"],
                partial_option=str(control["partial_option"]),
                use_runmax=bool(control["use_runmax"]),
                rng=rng,
            )
        )

    f_values = np.array([result.f_optimal for result in results], dtype=float)
    best_idx = int(np.argmax(f_values))
    endpoints = np.vstack([result.x_optimal for result in results])
    iterations = np.array([result.iterations for result in results], dtype=float)
    summary = {
        "n_starts": control["n_starts"],
        "mean_iterations": float(np.mean(iterations)),
        "std_values": float(np.std(f_values, ddof=1)) if f_values.size > 1 else 0.0,
        "endpoints": endpoints,
        "values": f_values,
    }
    return SMCOResult(
        best_result=results[best_idx],
        all_results=results,
        opt_control=control,
        summary=summary,
    )


def smco(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    **kwargs: Any,
) -> SMCOResult:
    control = dict(kwargs)
    control["refine_search"] = False
    control["iter_boost"] = 0
    return smco_multi(f, bounds_lower, bounds_upper, start_points, control)


def smco_r(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    **kwargs: Any,
) -> SMCOResult:
    control = dict(kwargs)
    control["refine_search"] = True
    control["iter_boost"] = 0
    control.setdefault("refine_ratio", 0.5)
    return smco_multi(f, bounds_lower, bounds_upper, start_points, control)


def smco_br(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    iter_boost: int = 1000,
    **kwargs: Any,
) -> SMCOResult:
    control = dict(kwargs)
    control["refine_search"] = True
    control["iter_boost"] = iter_boost
    control.setdefault("refine_ratio", 0.5)
    return smco_multi(f, bounds_lower, bounds_upper, start_points, control)
```

- [ ] **Step 4: Remove placeholder wrapper definitions**

Ensure `src/smco/optimizer.py` has exactly one definition each for `smco`, `smco_r`, `smco_br`, and `smco_multi`, and none of them raises `NotImplementedError`.

- [ ] **Step 5: Run optimizer tests**

Run:

```bash
pytest tests/test_optimizer.py tests/test_validation.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit when git is available**

Run:

```bash
git status --short
git add src/smco/optimizer.py tests/test_optimizer.py
git commit -m "feat: port core smco optimizer"
```

Expected in current workspace: `git status` fails. Skip commit until git is repaired.

---

### Task 4: Benchmark Test Functions

**Files:**
- Create: `src/smco/test_functions.py`
- Create: `tests/test_test_functions.py`
- Modify: `src/smco/__init__.py`

- [ ] **Step 1: Write failing benchmark function tests**

Create `tests/test_test_functions.py`:

```python
import math

import numpy as np
import pytest

from smco.test_functions import (
    BenchmarkConfig,
    ackley,
    assign_config,
    griewank,
    negative_squared_norm,
    rastrigin,
    rosenbrock,
    squared_norm,
)


def test_core_function_known_values():
    zeros = np.zeros(3)
    assert squared_norm(zeros) == 0.0
    assert negative_squared_norm(zeros) == -0.0
    assert rastrigin(zeros) == pytest.approx(0.0)
    assert ackley(zeros) == pytest.approx(0.0, abs=1e-12)
    assert griewank(zeros) == pytest.approx(0.0)
    assert rosenbrock(np.ones(4)) == pytest.approx(0.0)


def test_assign_config_returns_maximization_wrapper_for_minimum_benchmarks():
    config = assign_config("Rastrigin", 2)

    assert isinstance(config, BenchmarkConfig)
    assert config.name == "Rastrigin2d"
    assert config.dim == 2
    assert np.allclose(config.bounds_lower, [-5.12, -5.12])
    assert np.allclose(config.bounds_upper, [5.12, 5.12])
    assert config.f(np.zeros(2)) == pytest.approx(-0.0)
    assert config.known_best_value == pytest.approx(0.0)
    assert config.sense == "min"


def test_assign_config_supports_multiple_names_case_insensitively():
    names = ["Ackley", "Griewank", "Rosenbrock", "DixonPrice", "Zakharov", "Qing", "Michalewicz"]
    for name in names:
        config = assign_config(name.lower(), 3)
        assert config.dim == 3
        assert config.bounds_lower.shape == (3,)
        assert config.bounds_upper.shape == (3,)
        assert math.isfinite(config.f(np.zeros(3)))


def test_assign_config_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown benchmark"):
        assign_config("does-not-exist", 2)
```

- [ ] **Step 2: Run benchmark function tests to verify they fail**

Run:

```bash
pytest tests/test_test_functions.py -q
```

Expected: FAIL because `smco.test_functions` does not exist.

- [ ] **Step 3: Implement benchmark functions and configs**

Create `src/smco/test_functions.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    dim: int
    bounds_lower: np.ndarray
    bounds_upper: np.ndarray
    f: Callable[[np.ndarray], float]
    sense: str
    known_best_value: float | None = None
    known_best_x: np.ndarray | None = None


def squared_norm(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))


def negative_squared_norm(x: np.ndarray) -> float:
    return -squared_norm(x)


def rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    d = x.size
    return float(10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi) -> float:
    x = np.asarray(x, dtype=float)
    d = x.size
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(c * x))
    return float(-a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.e)


def griewank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1.0)


def rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))


def dixon_price(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    i = np.arange(2, x.size + 1, dtype=float)
    return float((x[0] - 1.0) ** 2 + np.sum(i * (2.0 * x[1:] ** 2 - x[:-1]) ** 2))


def zakharov(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1, dtype=float)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(0.5 * i * x)
    return float(sum1 + sum2 ** 2 + sum2 ** 4)


def qing(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum((x ** 2 - i) ** 2))


def michalewicz(x: np.ndarray, m: float = 10.0) -> float:
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1, dtype=float)
    return float(-np.sum(np.sin(x) * np.sin(i * x ** 2 / np.pi) ** (2 * m)))


def dropwave(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    r2 = x[0] ** 2 + x[1] ** 2
    return float(-(1.0 + np.cos(12.0 * np.sqrt(r2))) / (0.5 * r2 + 2.0))


def shubert(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    i = np.arange(1, 6, dtype=float)
    sum1 = np.sum(i * np.cos((i + 1.0) * x[0] + i))
    sum2 = np.sum(i * np.cos((i + 1.0) * x[1] + i))
    return float(sum1 * sum2)


def mccormick(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1.0)


def six_hump_camel(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    return float((4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2)


def eggholder(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    return float(-(x2 + 47) * np.sin(np.sqrt(abs(x2 + x1 / 2 + 47))) - x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47)))))


def bukin6(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    return float(100 * np.sqrt(abs(x2 - 0.01 * x1 ** 2)) + 0.01 * abs(x1 + 10))


def easom(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(-np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi) ** 2) - ((x[1] - np.pi) ** 2)))


def mishra6(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    return float(-np.log((np.sin((np.cos(x1) + np.cos(x2)) ** 2) ** 2 - np.cos((np.sin(x1) + np.sin(x2)) ** 2) ** 2 + x1) ** 2 + 0.1))


def _min_config(
    name: str,
    dim: int,
    lower: float | list[float],
    upper: float | list[float],
    raw: Callable[[np.ndarray], float],
    known_min: float | None,
    known_min_x: np.ndarray | None = None,
) -> BenchmarkConfig:
    bounds_lower = np.full(dim, lower, dtype=float) if np.isscalar(lower) else np.asarray(lower, dtype=float)
    bounds_upper = np.full(dim, upper, dtype=float) if np.isscalar(upper) else np.asarray(upper, dtype=float)
    return BenchmarkConfig(
        name=name,
        dim=dim,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        f=lambda x: -float(raw(np.asarray(x, dtype=float))),
        sense="min",
        known_best_value=known_min,
        known_best_x=known_min_x,
    )


def assign_config(name: str, dim: int) -> BenchmarkConfig:
    key = name.lower().replace("_", "").replace("-", "")
    dim = int(dim)

    if key in {"norm2x", "squarednorm", "normsquared"}:
        return BenchmarkConfig("Norm Squared", dim, np.full(dim, -1.0), np.full(dim, 1.0), squared_norm, "max", float(dim), np.ones(dim))
    if key in {"negativesquarednorm", "minusnormsquared", "norm2xminus"}:
        return BenchmarkConfig("Minus Norm Squared", dim, np.full(dim, -1.0), np.full(dim, 1.0), negative_squared_norm, "max", 0.0, np.zeros(dim))
    if key == "rastrigin":
        return _min_config(f"Rastrigin{dim}d", dim, -5.12, 5.12, rastrigin, 0.0, np.zeros(dim))
    if key == "ackley":
        return _min_config(f"Ackley{dim}d", dim, -32.768, 32.768, ackley, 0.0, np.zeros(dim))
    if key == "griewank":
        return _min_config(f"Griewank{dim}d", dim, -600.0, 600.0, griewank, 0.0, np.zeros(dim))
    if key == "rosenbrock":
        return _min_config(f"Rosenbrock{dim}d", dim, -5.0, 10.0, rosenbrock, 0.0, np.ones(dim))
    if key == "dixonprice":
        return _min_config(f"DixonPrice{dim}d", dim, -10.0, 10.0, dixon_price, 0.0, None)
    if key == "zakharov":
        return _min_config(f"Zakharov{dim}d", dim, -5.0, 10.0, zakharov, 0.0, np.zeros(dim))
    if key == "qing":
        return _min_config(f"Qing{dim}d", dim, -500.0, 500.0, qing, 0.0, None)
    if key == "michalewicz":
        return _min_config(f"Michalewicz{dim}d", dim, 0.0, np.pi, michalewicz, None, None)
    if key == "dropwave":
        return _min_config("Dropwave", 2, -5.12, 5.12, dropwave, -1.0, np.zeros(2))
    if key == "shubert":
        return _min_config("Shubert", 2, -5.12, 5.12, shubert, -186.7309, None)
    if key == "mccormick":
        return _min_config("McCormick", 2, [-1.5, -3.0], [4.0, 4.0], mccormick, -1.9133, None)
    if key in {"sixhumpcamel", "camel6"}:
        return _min_config("Six-Hump Camel", 2, [-3.0, -2.0], [3.0, 2.0], six_hump_camel, -1.0316, None)
    if key == "eggholder":
        return _min_config("Eggholder", 2, -512.0, 512.0, eggholder, -959.6407, None)
    if key == "bukin6":
        return _min_config("Bukin No. 6", 2, [-15.0, -3.0], [-5.0, 3.0], bukin6, 0.0, np.array([-10.0, 1.0]))
    if key == "easom":
        return _min_config("Easom", 2, -100.0, 100.0, easom, -1.0, np.array([np.pi, np.pi]))
    if key == "mishra6":
        return _min_config("Mishra No. 6", 2, -10.0, 10.0, mishra6, None, None)

    raise ValueError(f"Unknown benchmark: {name}")
```

- [ ] **Step 4: Export benchmark config helpers**

Modify `src/smco/__init__.py` to include:

```python
from .test_functions import BenchmarkConfig, assign_config
```

and add `"BenchmarkConfig"` and `"assign_config"` to `__all__`.

- [ ] **Step 5: Run benchmark function tests**

Run:

```bash
pytest tests/test_test_functions.py -q
```

Expected: PASS.

- [ ] **Step 6: Run all tests so far**

Run:

```bash
pytest tests/test_optimizer.py tests/test_validation.py tests/test_test_functions.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit when git is available**

Run:

```bash
git status --short
git add src/smco/test_functions.py src/smco/__init__.py tests/test_test_functions.py
git commit -m "feat: add smco benchmark functions"
```

Expected in current workspace: `git status` fails. Skip commit until git is repaired.

---

### Task 5: Benchmark And Paper Smoke Runners

**Files:**
- Create: `src/smco/benchmark.py`
- Create: `src/smco/paper.py`
- Modify: `src/smco/__init__.py`
- Create: `tests/test_benchmark.py`

- [ ] **Step 1: Write failing runner tests**

Create `tests/test_benchmark.py`:

```python
import numpy as np

from smco.benchmark import BenchmarkRun, run_benchmark
from smco.paper import run_paper_smoke


def test_run_benchmark_executes_tiny_configuration():
    run = run_benchmark(
        name="Rastrigin",
        dim=2,
        variant="smco_r",
        repetitions=2,
        iter_max=40,
        n_starts=5,
        seed=10,
    )

    assert isinstance(run, BenchmarkRun)
    assert run.config.name == "Rastrigin2d"
    assert len(run.results) == 2
    assert run.best_value == max(result.best_result.f_optimal for result in run.results)
    assert np.isfinite(run.best_value)


def test_run_benchmark_supports_all_variants():
    for variant in ["smco", "smco_r", "smco_br"]:
        run = run_benchmark(
            name="Minus Norm Squared",
            dim=2,
            variant=variant,
            repetitions=1,
            iter_max=30,
            n_starts=5,
            seed=20,
        )
        assert len(run.results) == 1
        assert run.best_x.shape == (2,)


def test_run_paper_smoke_returns_named_runs():
    runs = run_paper_smoke(iter_max=30, n_starts=5, seed=30)

    assert set(runs) == {"Rastrigin", "Ackley", "Griewank"}
    assert all(len(run.results) == 1 for run in runs.values())
```

- [ ] **Step 2: Run runner tests to verify they fail**

Run:

```bash
pytest tests/test_benchmark.py -q
```

Expected: FAIL because `smco.benchmark` and `smco.paper` do not exist.

- [ ] **Step 3: Implement benchmark runner**

Create `src/smco/benchmark.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .optimizer import smco, smco_br, smco_multi, smco_r
from .results import SMCOResult
from .test_functions import BenchmarkConfig, assign_config


@dataclass
class BenchmarkRun:
    config: BenchmarkConfig
    variant: str
    results: list[SMCOResult]
    best_value: float
    best_x: np.ndarray


def _variant_function(variant: str):
    key = variant.lower()
    if key == "smco":
        return smco
    if key == "smco_r":
        return smco_r
    if key == "smco_br":
        return smco_br
    if key == "smco_multi":
        return smco_multi
    raise ValueError("variant must be one of: smco, smco_r, smco_br, smco_multi")


def run_benchmark(
    name: str,
    dim: int,
    *,
    variant: str = "smco_r",
    repetitions: int = 1,
    seed: int = 123,
    **smco_options: Any,
) -> BenchmarkRun:
    config = assign_config(name, dim)
    runner = _variant_function(variant)
    results: list[SMCOResult] = []

    for rep in range(int(repetitions)):
        options = dict(smco_options)
        options["seed"] = int(seed) + rep
        if variant.lower() == "smco_multi":
            result = runner(config.f, config.bounds_lower, config.bounds_upper, opt_control=options)
        else:
            result = runner(config.f, config.bounds_lower, config.bounds_upper, **options)
        results.append(result)

    values = np.array([result.best_result.f_optimal for result in results], dtype=float)
    best_idx = int(np.argmax(values))
    return BenchmarkRun(
        config=config,
        variant=variant,
        results=results,
        best_value=float(values[best_idx]),
        best_x=np.array(results[best_idx].best_result.x_optimal, copy=True),
    )
```

- [ ] **Step 4: Implement paper smoke runner**

Create `src/smco/paper.py`:

```python
from __future__ import annotations

from typing import Any

from .benchmark import BenchmarkRun, run_benchmark


PAPER_SMOKE_CONFIGS: tuple[tuple[str, int], ...] = (
    ("Rastrigin", 2),
    ("Ackley", 2),
    ("Griewank", 2),
)


def run_paper_smoke(
    *,
    variant: str = "smco_r",
    iter_max: int = 80,
    n_starts: int = 5,
    seed: int = 123,
    **smco_options: Any,
) -> dict[str, BenchmarkRun]:
    options = dict(smco_options)
    options["iter_max"] = iter_max
    options["n_starts"] = n_starts
    return {
        name: run_benchmark(name, dim, variant=variant, repetitions=1, seed=seed + idx, **options)
        for idx, (name, dim) in enumerate(PAPER_SMOKE_CONFIGS)
    }
```

- [ ] **Step 5: Export runner symbols**

Modify `src/smco/__init__.py` to include:

```python
from .benchmark import BenchmarkRun, run_benchmark
from .paper import run_paper_smoke
```

and add `"BenchmarkRun"`, `"run_benchmark"`, and `"run_paper_smoke"` to `__all__`.

- [ ] **Step 6: Run runner tests**

Run:

```bash
pytest tests/test_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 7: Run all Python tests**

Run:

```bash
pytest -q
```

Expected: PASS.

- [ ] **Step 8: Commit when git is available**

Run:

```bash
git status --short
git add src/smco/benchmark.py src/smco/paper.py src/smco/__init__.py tests/test_benchmark.py
git commit -m "feat: add benchmark and paper smoke runners"
```

Expected in current workspace: `git status` fails. Skip commit until git is repaired.

---

### Task 6: Vendor Upstream R Sources

**Files:**
- Create: `vendor/SMCO_R/main/SMCO.R`
- Create: `vendor/SMCO_R/main/testfuncs.R`
- Create: `vendor/SMCO_R/main/benchmark.R`
- Create: `vendor/SMCO_R/main/README.md`
- Create: `vendor/SMCO_R/main/NEWS.md`
- Create: `vendor/SMCO_R/main/LICENSE`
- Create: `vendor/SMCO_R/v1.0.0/SMCO.R`
- Create: `vendor/SMCO_R/v1.0.0/testfuncs.R`
- Create: `vendor/SMCO_R/v1.0.0/testfunc_NN1L.R`
- Create: `vendor/SMCO_R/v1.0.0/RunComparison.R`
- Create: `vendor/SMCO_R/v1.0.0/GD.R`
- Create: `vendor/SMCO_R/v1.0.0/SignGD.R`
- Create: `vendor/SMCO_R/v1.0.0/SPSA.R`
- Create: `vendor/SMCO_R/v1.0.0/README.md`
- Create: `vendor/SMCO_R/v1.0.0/LICENSE`

- [ ] **Step 1: Download upstream source files**

Run:

```bash
mkdir -p vendor/SMCO_R/main vendor/SMCO_R/v1.0.0
curl -LfsS https://raw.githubusercontent.com/wayne-y-gao/SMCO/main/SMCO.R -o vendor/SMCO_R/main/SMCO.R
curl -LfsS https://raw.githubusercontent.com/wayne-y-gao/SMCO/main/testfuncs.R -o vendor/SMCO_R/main/testfuncs.R
curl -LfsS https://raw.githubusercontent.com/wayne-y-gao/SMCO/main/benchmark.R -o vendor/SMCO_R/main/benchmark.R
curl -LfsS https://raw.githubusercontent.com/wayne-y-gao/SMCO/main/README.md -o vendor/SMCO_R/main/README.md
curl -LfsS https://raw.githubusercontent.com/wayne-y-gao/SMCO/main/NEWS.md -o vendor/SMCO_R/main/NEWS.md
curl -LfsS https://raw.githubusercontent.com/wayne-y-gao/SMCO/main/LICENSE -o vendor/SMCO_R/main/LICENSE
curl -LfsS https://raw.githubusercontent.com/wayne-y-gao/SMCO/main/v1.0.0/SMCO.R -o vendor/SMCO_R/v1.0.0/SMCO.R
curl -LfsS https://raw.githubusercontent.com/wayne-y-gao/SMCO/main/v1.0.0/testfuncs.R -o vendor/SMCO_R/v1.0.0/testfuncs.R
curl -LfsS https://raw.githubusercontent.com/wayne-y-gao/SMCO/main/v1.0.0/testfunc_NN1L.R -o vendor/SMCO_R/v1.0.0/testfunc_NN1L.R
curl -LfsS https://raw.githubusercontent.com/wayne-y-gao/SMCO/main/v1.0.0/RunComparison.R -o vendor/SMCO_R/v1.0.0/RunComparison.R
curl -LfsS https://raw.githubusercontent.com/wayne-y-gao/SMCO/main/v1.0.0/GD.R -o vendor/SMCO_R/v1.0.0/GD.R
curl -LfsS https://raw.githubusercontent.com/wayne-y-gao/SMCO/main/v1.0.0/SignGD.R -o vendor/SMCO_R/v1.0.0/SignGD.R
curl -LfsS https://raw.githubusercontent.com/wayne-y-gao/SMCO/main/v1.0.0/SPSA.R -o vendor/SMCO_R/v1.0.0/SPSA.R
curl -LfsS https://raw.githubusercontent.com/wayne-y-gao/SMCO/main/v1.0.0/README.md -o vendor/SMCO_R/v1.0.0/README.md
curl -LfsS https://raw.githubusercontent.com/wayne-y-gao/SMCO/main/v1.0.0/LICENSE -o vendor/SMCO_R/v1.0.0/LICENSE
```

Expected: all commands complete with exit code 0.

- [ ] **Step 2: Verify vendored files exist and are non-empty**

Run:

```bash
find vendor/SMCO_R -type f -maxdepth 3 -print0 | xargs -0 wc -l
```

Expected: every listed file has at least one line.

- [ ] **Step 3: Run all Python tests to confirm vendoring did not affect behavior**

Run:

```bash
pytest -q
```

Expected: PASS.

- [ ] **Step 4: Commit when git is available**

Run:

```bash
git status --short
git add vendor/SMCO_R
git commit -m "chore: vendor upstream smco r sources"
```

Expected in current workspace: `git status` fails. Skip commit until git is repaired.

---

### Task 7: README Polish And Final Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README with final usage documentation**

Replace `README.md` with:

```markdown
# SMCO Python

Minimal Python migration of the Strategic Monte Carlo Optimization algorithms from `wayne-y-gao/SMCO`.

This package ports the current R `v1.1.0` API shape while preserving a paper-oriented path for the archived `v1.0.0` replication code. The vendored R files are reference material only; Python tests do not execute R.

## Install

```bash
pip install -e ".[test]"
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

Classic minimization benchmarks are wrapped as maximization objectives internally.

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
```
```

- [ ] **Step 2: Run import smoke check**

Run:

```bash
python - <<'PY'
from smco import smco, smco_r, smco_br, smco_multi, run_benchmark, run_paper_smoke
print("imports ok")
PY
```

Expected: prints `imports ok`.

- [ ] **Step 3: Run all tests**

Run:

```bash
pytest -q
```

Expected: PASS.

- [ ] **Step 4: Run a tiny end-to-end example**

Run:

```bash
python - <<'PY'
import numpy as np
from smco import smco_r

def objective(x):
    return -float(np.sum(x ** 2))

result = smco_r(objective, [-5, -5], [5, 5], iter_max=80, n_starts=5, seed=123)
print(result.best_result.x_optimal)
print(result.best_result.f_optimal)
assert result.best_result.f_optimal > -0.2
PY
```

Expected: exits with code 0 and prints a point near the origin plus a finite objective value.

- [ ] **Step 5: Commit when git is available**

Run:

```bash
git status --short
git add README.md
git commit -m "docs: document smco python usage"
```

Expected in current workspace: `git status` fails. Skip commit until git is repaired.

---

## Self-Review

- Spec coverage: The plan covers package scaffold, v1.1.0 public API, optimizer semantics, result model, benchmark functions, paper smoke path, vendored R source, README, and tests that do not require R.
- Placeholder scan: No task contains undefined placeholders such as TBD or TODO.
- Type consistency: `SingleResult`, `SMCOResult`, `BenchmarkConfig`, and `BenchmarkRun` names are defined before use. Public functions are consistently named `smco`, `smco_r`, `smco_br`, and `smco_multi`.
- Known limitation: Commit commands are included for a repaired git repository, but cannot run successfully in the current invalid git state.
