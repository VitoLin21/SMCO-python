# Comparison Methods Migration & Experiment Reproduction

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate 12 comparison optimizer methods from R to Python, build a comparison framework, and reproduce the paper's experiments.

**Architecture:** A `comparison/` package with a `methods/` sub-package for individual optimizer wrappers, each exposing a unified `optimize()` function. A `run_comparison.py` orchestrates all algorithms across test configs and computes paper metrics. An entry script `scripts/run_experiment.py` drives experiments with `--quick`/`--table1`/`--table2`/`--table3` modes.

**Tech Stack:** Python 3.10+, numpy, scipy (already in venv). No additional dependencies — PSO/GA implemented from scratch, BOBYQA uses scipy COBYLA.

---

## File Structure

```
comparison/
  __init__.py              # Re-exports run_comparison, domain_mod, method registry
  methods/
    __init__.py            # METHOD_REGISTRY dict mapping names to optimize functions
    base.py                # OptimizerResult dataclass
    gd.py                  # Gradient Descent (from GD.R)
    sign_gd.py             # Sign Gradient Descent (from SignGD.R)
    spsa.py                # SPSA (from SPSA.R)
    adam.py                # Adam optimizer (custom implementation)
    lbfgs.py               # L-BFGS-B (scipy.optimize.minimize)
    nelder_mead.py         # Nelder-Mead (scipy.optimize.minimize)
    bobyqa.py              # BOBYQA ≈ COBYLA (scipy.optimize.minimize)
    gensa.py               # GenSA ≈ dual_annealing (scipy.optimize)
    sa.py                  # SA ≈ basinhopping (scipy.optimize)
    de.py                  # DE (scipy.optimize.differential_evolution)
    ga.py                  # Genetic Algorithm (custom implementation)
    pso.py                 # Particle Swarm (custom implementation)
  run_comparison.py        # Core comparison framework
  domain_mod.py            # Domain rotation/shift/asymmetrization
scripts/
  run_experiment.py        # CLI entry point
tests/
  test_comparison.py       # Tests for comparison methods and framework
```

---

### Task 1: Create directory structure and base types

**Files:**
- Create: `comparison/__init__.py`
- Create: `comparison/methods/__init__.py`
- Create: `comparison/methods/base.py`
- Create: `tests/test_comparison.py`

- [ ] **Step 1: Create comparison package directories**

```bash
mkdir -p comparison/methods
```

- [ ] **Step 2: Write `comparison/methods/base.py` with OptimizerResult**

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)
class OptimizerResult:
    x_optimal: np.ndarray
    f_optimal: float
    iterations: int
```

- [ ] **Step 3: Write `comparison/methods/__init__.py` with method registry**

```python
from __future__ import annotations

from typing import Callable

import numpy as np

from .base import OptimizerResult

OptimizerFn = Callable[
    [Callable[[np.ndarray], float], np.ndarray, np.ndarray, np.ndarray | None, bool],
    OptimizerResult,
]

METHOD_REGISTRY: dict[str, OptimizerFn] = {}


def register(name: str) -> Callable[[OptimizerFn], OptimizerFn]:
    def decorator(fn: OptimizerFn) -> OptimizerFn:
        METHOD_REGISTRY[name] = fn
        return fn
    return decorator


def get_method(name: str) -> OptimizerFn:
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown optimizer method: {name}. Available: {list(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[name]
```

- [ ] **Step 4: Write `comparison/__init__.py`**

```python
from .methods import METHOD_REGISTRY, get_method
from .methods.base import OptimizerResult
```

- [ ] **Step 5: Write initial test file `tests/test_comparison.py`**

```python
from __future__ import annotations

import numpy as np
import pytest

from comparison.methods import METHOD_REGISTRY, get_method
from comparison.methods.base import OptimizerResult


def test_optimizer_result_creation():
    result = OptimizerResult(
        x_optimal=np.array([1.0, 2.0]),
        f_optimal=3.0,
        iterations=10,
    )
    assert result.f_optimal == 3.0
    assert result.iterations == 10
    assert result.x_optimal.shape == (2,)


def test_registry_empty_at_start():
    assert isinstance(METHOD_REGISTRY, dict)


def test_get_method_raises_for_unknown():
    with pytest.raises(ValueError, match="Unknown optimizer method"):
        get_method("nonexistent_method_xyz")
```

- [ ] **Step 6: Run tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_comparison.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add comparison/ tests/test_comparison.py
git commit -m "feat: add comparison package skeleton with base types and registry"
```

---

### Task 2: Implement Gradient Descent (GD)

**Files:**
- Create: `comparison/methods/gd.py`
- Modify: `tests/test_comparison.py`

This migrates `GD.R` — gradient descent with momentum, adaptive learning rate, and bound constraints.

- [ ] **Step 1: Write `comparison/methods/gd.py`**

```python
from __future__ import annotations

import numpy as np

from . import register
from .base import OptimizerResult


def _numerical_gradient(f, x, eps=1e-8):
    grad = np.zeros_like(x)
    for i in range(x.size):
        h = np.zeros_like(x)
        h[i] = eps
        grad[i] = (f(x + h) - f(x - h)) / (2.0 * eps)
    return grad


@register("GD")
def gradient_descent(
    f,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_points: np.ndarray | None = None,
    maximize: bool = False,
    max_iter: int = 500,
    tol: float = 1e-6,
    learning_rate: float = 0.1,
    momentum: float = 0.9,
) -> OptimizerResult:
    if start_points is None:
        raise ValueError("GD requires start_points")

    sign = -1.0 if maximize else 1.0

    best_x = None
    best_value = np.inf
    total_iters = 0

    for sp in start_points:
        x = np.array(sp, dtype=float, copy=True)
        velocity = np.zeros_like(x)
        lr = learning_rate
        prev_value = float(f(x))

        for it in range(1, max_iter + 1):
            grad = _numerical_gradient(f, x)
            velocity = momentum * velocity + sign * lr * grad
            x_new = np.clip(x - velocity, bounds_lower, bounds_upper)
            current_value = float(f(x_new))

            if current_value < prev_value:
                pass
            else:
                lr *= 0.5
                velocity *= 0.5

            if current_value < best_value:
                best_value = current_value
                best_x = np.array(x_new, copy=True)

            if np.linalg.norm(x_new - x) < tol:
                total_iters += it
                break

            x = x_new
            prev_value = current_value
            total_iters += 1
        else:
            total_iters += max_iter

    if best_x is None:
        best_x = np.array(start_points[0], dtype=float, copy=True)
        best_value = float(f(best_x))

    return OptimizerResult(x_optimal=best_x, f_optimal=best_value, iterations=total_iters)
```

- [ ] **Step 2: Add GD tests to `tests/test_comparison.py`**

Append to the end of the file:

```python
from comparison.methods.gd import gradient_descent


def _sphere_min(x):
    return float(np.sum(x**2))


class TestGD:
    def test_finds_minimum(self):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        starts = np.array([[2.0, 2.0], [-2.0, -2.0]])
        result = gradient_descent(_sphere_min, bl, bu, starts, maximize=False, max_iter=500)
        assert result.f_optimal < 0.5
        assert result.x_optimal.shape == (2,)

    def test_registered(self):
        assert "GD" in METHOD_REGISTRY
        fn = get_method("GD")
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        starts = np.array([[1.0, 1.0]])
        result = fn(_sphere_min, bl, bu, starts, maximize=False, max_iter=200)
        assert isinstance(result, OptimizerResult)

    def test_maximize(self):
        def neg_sphere(x):
            return float(-np.sum(x**2))

        bl = np.array([-1.0, -1.0])
        bu = np.array([1.0, 1.0])
        starts = np.array([[0.1, 0.1]])
        result = gradient_descent(neg_sphere, bl, bu, starts, maximize=True, max_iter=500)
        assert result.f_optimal < -0.01
```

- [ ] **Step 3: Run tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_comparison.py::TestGD -v
```

Expected: All 3 GD tests PASS.

- [ ] **Step 4: Commit**

```bash
git add comparison/methods/gd.py tests/test_comparison.py
git commit -m "feat: add Gradient Descent optimizer migrated from GD.R"
```

---

### Task 3: Implement Sign Gradient Descent (SignGD)

**Files:**
- Create: `comparison/methods/sign_gd.py`
- Modify: `tests/test_comparison.py`

Migrates `SignGD.R` — uses only the sign of the gradient, with learning rate decay and no-improvement early stopping.

- [ ] **Step 1: Write `comparison/methods/sign_gd.py`**

```python
from __future__ import annotations

import numpy as np

from . import register
from .base import OptimizerResult
from .gd import _numerical_gradient


@register("SignGD")
def sign_gradient_descent(
    f,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_points: np.ndarray | None = None,
    maximize: bool = False,
    max_iter: int = 500,
    tol: float = 1e-6,
    learning_rate: float = 0.1,
    decay_rate: float = 0.995,
) -> OptimizerResult:
    if start_points is None:
        raise ValueError("SignGD requires start_points")

    sign = -1.0 if maximize else 1.0

    best_x = None
    best_value = np.inf
    total_iters = 0

    for sp in start_points:
        x = np.array(sp, dtype=float, copy=True)
        lr = learning_rate
        no_improve = 0

        for it in range(1, max_iter + 1):
            grad = _numerical_gradient(f, x)
            grad_sign = np.sign(grad)
            x_new = np.clip(x - sign * lr * grad_sign, bounds_lower, bounds_upper)
            new_value = float(f(x_new))

            if new_value < best_value:
                best_value = new_value
                best_x = np.array(x_new, copy=True)
                no_improve = 0
            else:
                no_improve += 1

            lr *= decay_rate

            if np.linalg.norm(x_new - x) < tol:
                total_iters += it
                break
            if no_improve > 50:
                total_iters += it
                break

            x = x_new
            total_iters += 1
        else:
            total_iters += max_iter

    if best_x is None:
        best_x = np.array(start_points[0], dtype=float, copy=True)
        best_value = float(f(best_x))

    return OptimizerResult(x_optimal=best_x, f_optimal=best_value, iterations=total_iters)
```

- [ ] **Step 2: Add SignGD tests to `tests/test_comparison.py`**

Append:

```python
from comparison.methods.sign_gd import sign_gradient_descent


class TestSignGD:
    def test_finds_minimum(self):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        starts = np.array([[2.0, 2.0], [-2.0, -2.0]])
        result = sign_gradient_descent(_sphere_min, bl, bu, starts, maximize=False, max_iter=500)
        assert result.f_optimal < 1.0
        assert result.x_optimal.shape == (2,)

    def test_registered(self):
        assert "SignGD" in METHOD_REGISTRY
```

- [ ] **Step 3: Run tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_comparison.py::TestSignGD -v
```

Expected: All 2 SignGD tests PASS.

- [ ] **Step 4: Commit**

```bash
git add comparison/methods/sign_gd.py tests/test_comparison.py
git commit -m "feat: add Sign Gradient Descent optimizer migrated from SignGD.R"
```

---

### Task 4: Implement SPSA

**Files:**
- Create: `comparison/methods/spsa.py`
- Modify: `tests/test_comparison.py`

Migrates `SPSA.R` — simultaneous perturbation stochastic approximation with Bernoulli perturbation.

- [ ] **Step 1: Write `comparison/methods/spsa.py`**

```python
from __future__ import annotations

import numpy as np

from . import register
from .base import OptimizerResult


@register("SPSA")
def spsa(
    f,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_points: np.ndarray | None = None,
    maximize: bool = False,
    A: float = 50.0,
    a: float = 0.10,
    c: float = 1e-3,
    alpha: float = 0.602,
    gamma: float = 0.101,
    max_iter: int = 500,
    tol: float = 1e-7,
    seed: int | None = None,
) -> OptimizerResult:
    if start_points is None:
        raise ValueError("SPSA requires start_points")

    rng = np.random.default_rng(seed)
    sign = -1.0 if maximize else 1.0

    best_x = None
    best_value = np.inf
    total_iters = 0

    for sp in start_points:
        x = np.array(sp, dtype=float, copy=True)
        d = x.size
        prev_value = float(f(x))

        for k in range(1, max_iter + 1):
            ak = a / (A + k) ** alpha
            ck = c / k ** gamma

            delta = 2 * rng.binomial(1, 0.5, size=d) - 1
            x_plus = x + ck * delta
            x_minus = x - ck * delta
            f_plus = float(f(x_plus))
            f_minus = float(f(x_minus))

            with np.errstate(divide="ignore", invalid="ignore"):
                g_hat = (f_plus - f_minus) / (2.0 * ck * delta)
            g_hat = np.nan_to_num(g_hat, nan=0.0, posinf=0.0, neginf=0.0)

            x = np.clip(x - sign * ak * g_hat, bounds_lower, bounds_upper)
            current_value = float(f(x))

            if current_value < best_value:
                best_value = current_value
                best_x = np.array(x, copy=True)

            if k > 1 and abs(current_value - prev_value) < tol:
                total_iters += k
                break

            prev_value = current_value
            total_iters += 1
        else:
            total_iters += max_iter

    if best_x is None:
        best_x = np.array(start_points[0], dtype=float, copy=True)
        best_value = float(f(best_x))

    return OptimizerResult(x_optimal=best_x, f_optimal=best_value, iterations=total_iters)
```

- [ ] **Step 2: Add SPSA tests to `tests/test_comparison.py`**

Append:

```python
from comparison.methods.spsa import spsa


class TestSPSA:
    def test_finds_minimum(self):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        starts = np.array([[2.0, 2.0]])
        result = spsa(_sphere_min, bl, bu, starts, maximize=False, max_iter=500, seed=42)
        assert result.f_optimal < 2.0
        assert result.x_optimal.shape == (2,)

    def test_registered(self):
        assert "SPSA" in METHOD_REGISTRY
```

- [ ] **Step 3: Run tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_comparison.py::TestSPSA -v
```

Expected: All 2 SPSA tests PASS.

- [ ] **Step 4: Commit**

```bash
git add comparison/methods/spsa.py tests/test_comparison.py
git commit -m "feat: add SPSA optimizer migrated from SPSA.R"
```

---

### Task 5: Implement Adam optimizer

**Files:**
- Create: `comparison/methods/adam.py`
- Modify: `tests/test_comparison.py`

Custom Adam implementation (replaces R's `optimg` package ADAM method). Standard Adam with configurable beta1, beta2, epsilon.

- [ ] **Step 1: Write `comparison/methods/adam.py`**

```python
from __future__ import annotations

import numpy as np

from . import register
from .base import OptimizerResult
from .gd import _numerical_gradient


@register("ADAM")
def adam(
    f,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_points: np.ndarray | None = None,
    maximize: bool = False,
    max_iter: int = 500,
    tol: float = 1e-6,
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> OptimizerResult:
    if start_points is None:
        raise ValueError("ADAM requires start_points")

    sign = -1.0 if maximize else 1.0

    best_x = None
    best_value = np.inf
    total_iters = 0

    for sp in start_points:
        x = np.array(sp, dtype=float, copy=True)
        m = np.zeros_like(x)
        v = np.zeros_like(x)

        for t in range(1, max_iter + 1):
            grad = _numerical_gradient(f, x)
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad**2
            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)

            x_new = np.clip(x - sign * learning_rate * m_hat / (np.sqrt(v_hat) + epsilon), bounds_lower, bounds_upper)
            current_value = float(f(x_new))

            if current_value < best_value:
                best_value = current_value
                best_x = np.array(x_new, copy=True)

            if np.linalg.norm(x_new - x) < tol:
                total_iters += t
                break

            x = x_new
            total_iters += 1
        else:
            total_iters += max_iter

    if best_x is None:
        best_x = np.array(start_points[0], dtype=float, copy=True)
        best_value = float(f(best_x))

    return OptimizerResult(x_optimal=best_x, f_optimal=best_value, iterations=total_iters)
```

- [ ] **Step 2: Add ADAM tests to `tests/test_comparison.py`**

Append:

```python
from comparison.methods.adam import adam


class TestAdam:
    def test_finds_minimum(self):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        starts = np.array([[2.0, 2.0]])
        result = adam(_sphere_min, bl, bu, starts, maximize=False, max_iter=500)
        assert result.f_optimal < 0.5
        assert result.x_optimal.shape == (2,)

    def test_registered(self):
        assert "ADAM" in METHOD_REGISTRY
```

- [ ] **Step 3: Run tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_comparison.py::TestAdam -v
```

Expected: All 2 ADAM tests PASS.

- [ ] **Step 4: Commit**

```bash
git add comparison/methods/adam.py tests/test_comparison.py
git commit -m "feat: add Adam optimizer (replaces R optimg ADAM)"
```

---

### Task 6: Implement scipy-based local optimizers (L-BFGS-B, Nelder-Mead, BOBYQA)

**Files:**
- Create: `comparison/methods/lbfgs.py`
- Create: `comparison/methods/nelder_mead.py`
- Create: `comparison/methods/bobyqa.py`
- Modify: `tests/test_comparison.py`

All three wrap `scipy.optimize.minimize` with different methods. L-BFGS-B and BOBYQA support bounds; Nelder-Mead doesn't natively so we clip results.

- [ ] **Step 1: Write `comparison/methods/lbfgs.py`**

```python
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from . import register
from .base import OptimizerResult


@register("optimLBFGS")
def lbfgs(
    f,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_points: np.ndarray | None = None,
    maximize: bool = False,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> OptimizerResult:
    if start_points is None:
        raise ValueError("optimLBFGS requires start_points")

    target = (lambda x: -f(x)) if maximize else f
    bounds = list(zip(bounds_lower, bounds_upper))

    best_x = None
    best_value = np.inf
    total_iters = 0

    for sp in start_points:
        res = scipy_minimize(
            target, np.array(sp, dtype=float), method="L-BFGS-B",
            bounds=bounds, options={"maxiter": max_iter, "ftol": tol},
        )
        total_iters += res.nit
        val = float(-res.fun if maximize else res.fun)
        if val < best_value:
            best_value = val
            best_x = np.array(res.x, copy=True)

    if best_x is None:
        best_x = np.array(start_points[0], dtype=float, copy=True)
        best_value = float(f(best_x))

    return OptimizerResult(x_optimal=best_x, f_optimal=best_value, iterations=total_iters)
```

- [ ] **Step 2: Write `comparison/methods/nelder_mead.py`**

```python
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from . import register
from .base import OptimizerResult


@register("optimNM")
def nelder_mead(
    f,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_points: np.ndarray | None = None,
    maximize: bool = False,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> OptimizerResult:
    if start_points is None:
        raise ValueError("optimNM requires start_points")

    target = (lambda x: -f(x)) if maximize else f

    best_x = None
    best_value = np.inf
    total_iters = 0

    for sp in start_points:
        res = scipy_minimize(
            target, np.array(sp, dtype=float), method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": tol, "fatol": tol},
        )
        total_iters += res.nit
        x_clipped = np.clip(res.x, bounds_lower, bounds_upper)
        val = float(-target(x_clipped))
        if val < best_value:
            best_value = val
            best_x = np.array(x_clipped, copy=True)

    if best_x is None:
        best_x = np.array(start_points[0], dtype=float, copy=True)
        best_value = float(f(best_x))

    return OptimizerResult(x_optimal=best_x, f_optimal=best_value, iterations=total_iters)
```

- [ ] **Step 3: Write `comparison/methods/bobyqa.py`**

```python
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from . import register
from .base import OptimizerResult


@register("BOBYQA")
def bobyqa(
    f,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_points: np.ndarray | None = None,
    maximize: bool = False,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> OptimizerResult:
    if start_points is None:
        raise ValueError("BOBYQA requires start_points")

    target = (lambda x: -f(x)) if maximize else f
    bounds = list(zip(bounds_lower, bounds_upper))

    best_x = None
    best_value = np.inf
    total_iters = 0

    for sp in start_points:
        res = scipy_minimize(
            target, np.array(sp, dtype=float), method="COBYLA",
            bounds=bounds, options={"maxiter": max_iter, "rhobeg": 0.1},
        )
        total_iters += res.nfev
        val = float(-res.fun if maximize else res.fun)
        if val < best_value:
            best_value = val
            best_x = np.array(res.x, copy=True)

    if best_x is None:
        best_x = np.array(start_points[0], dtype=float, copy=True)
        best_value = float(f(best_x))

    return OptimizerResult(x_optimal=best_x, f_optimal=best_value, iterations=total_iters)
```

- [ ] **Step 4: Add scipy-wrapper tests to `tests/test_comparison.py`**

Append:

```python
from comparison.methods.lbfgs import lbfgs
from comparison.methods.nelder_mead import nelder_mead
from comparison.methods.bobyqa import bobyqa


class TestScipyMethods:
    @pytest.mark.parametrize("name,fn", [
        ("optimLBFGS", lbfgs),
        ("optimNM", nelder_mead),
        ("BOBYQA", bobyqa),
    ])
    def test_minimize_sphere(self, name, fn):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        starts = np.array([[2.0, 2.0]])
        result = fn(_sphere_min, bl, bu, starts, maximize=False, max_iter=500)
        assert result.f_optimal < 1.0
        assert result.x_optimal.shape == (2,)

    def test_all_registered(self):
        for name in ("optimLBFGS", "optimNM", "BOBYQA"):
            assert name in METHOD_REGISTRY
```

- [ ] **Step 5: Run tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_comparison.py::TestScipyMethods -v
```

Expected: All 4 tests PASS (3 parametrized + 1 registration).

- [ ] **Step 6: Commit**

```bash
git add comparison/methods/lbfgs.py comparison/methods/nelder_mead.py comparison/methods/bobyqa.py tests/test_comparison.py
git commit -m "feat: add L-BFGS-B, Nelder-Mead, BOBYQA scipy wrappers"
```

---

### Task 7: Implement scipy-based global optimizers (GenSA, SA, DE)

**Files:**
- Create: `comparison/methods/gensa.py`
- Create: `comparison/methods/sa.py`
- Create: `comparison/methods/de.py`
- Modify: `tests/test_comparison.py`

GenSA maps to `dual_annealing`, SA maps to `basinhopping`, DE maps to `differential_evolution`. Global methods may ignore start_points or use one random start.

- [ ] **Step 1: Write `comparison/methods/gensa.py`**

```python
from __future__ import annotations

import numpy as np
from scipy.optimize import dual_annealing

from . import register
from .base import OptimizerResult


@register("GenSA")
def gensa(
    f,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_points: np.ndarray | None = None,
    maximize: bool = False,
    max_iter: int = 500,
    seed: int | None = None,
) -> OptimizerResult:
    target = (lambda x: -f(x)) if maximize else f
    bounds = list(zip(bounds_lower, bounds_upper))

    x0 = None
    if start_points is not None and len(start_points) > 0:
        x0 = np.array(start_points[0], dtype=float)

    res = dual_annealing(target, bounds, x0=x0, maxiter=max_iter, seed=seed)
    val = float(-res.fun if maximize else res.fun)
    return OptimizerResult(x_optimal=np.array(res.x, copy=True), f_optimal=val, iterations=res.nit)
```

- [ ] **Step 2: Write `comparison/methods/sa.py`**

```python
from __future__ import annotations

import numpy as np
from scipy.optimize import basinhopping

from . import register
from .base import OptimizerResult


@register("SA")
def simulated_annealing(
    f,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_points: np.ndarray | None = None,
    maximize: bool = False,
    max_iter: int = 500,
    seed: int | None = None,
) -> OptimizerResult:
    target = (lambda x: -f(x)) if maximize else f
    rng = np.random.default_rng(seed)

    x0 = np.zeros_like(bounds_lower)
    if start_points is not None and len(start_points) > 0:
        x0 = np.array(start_points[rng.integers(len(start_points))], dtype=float)
    else:
        x0 = bounds_lower + rng.uniform(size=bounds_lower.size) * (bounds_upper - bounds_lower)

    class BoundsCheck:
        def __call__(self, **kwargs):
            x = kwargs.get("x_new")
            if x is None:
                return True
            return bool(np.all((x >= bounds_lower) & (x <= bounds_upper)))

    res = basinhopping(target, x0, niter=max_iter, accept_test=BoundsCheck(), seed=seed)
    x_clipped = np.clip(res.x, bounds_lower, bounds_upper)
    val = float(-target(x_clipped))
    return OptimizerResult(x_optimal=np.array(x_clipped, copy=True), f_optimal=val, iterations=res.nit)
```

- [ ] **Step 3: Write `comparison/methods/de.py`**

```python
from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

from . import register
from .base import OptimizerResult


@register("DEoptim")
def differential_evo(
    f,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_points: np.ndarray | None = None,
    maximize: bool = False,
    max_iter: int = 500,
    seed: int | None = None,
    popsize: int = 15,
) -> OptimizerResult:
    target = (lambda x: -f(x)) if maximize else f
    bounds = list(zip(bounds_lower, bounds_upper))

    res = differential_evolution(
        target, bounds, maxiter=max_iter, seed=seed,
        popsize=popsize, tol=1e-7, polish=False,
    )
    val = float(-res.fun if maximize else res.fun)
    return OptimizerResult(x_optimal=np.array(res.x, copy=True), f_optimal=val, iterations=res.nit)
```

- [ ] **Step 4: Add global optimizer tests to `tests/test_comparison.py`**

Append:

```python
from comparison.methods.gensa import gensa
from comparison.methods.sa import simulated_annealing
from comparison.methods.de import differential_evo


class TestGlobalMethods:
    @pytest.mark.parametrize("name,fn", [
        ("GenSA", gensa),
        ("SA", simulated_annealing),
        ("DEoptim", differential_evo),
    ])
    def test_minimize_sphere(self, name, fn):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        result = fn(_sphere_min, bl, bu, maximize=False, max_iter=50, seed=42)
        assert result.f_optimal < 1.0
        assert result.x_optimal.shape == (2,)

    def test_all_registered(self):
        for name in ("GenSA", "SA", "DEoptim"):
            assert name in METHOD_REGISTRY
```

- [ ] **Step 5: Run tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_comparison.py::TestGlobalMethods -v
```

Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add comparison/methods/gensa.py comparison/methods/sa.py comparison/methods/de.py tests/test_comparison.py
git commit -m "feat: add GenSA, SA, DE global optimizer wrappers"
```

---

### Task 8: Implement GA and PSO from scratch

**Files:**
- Create: `comparison/methods/ga.py`
- Create: `comparison/methods/pso.py`
- Modify: `tests/test_comparison.py`

Simple GA (real-valued, tournament selection, blend crossover, gaussian mutation) and PSO (standard velocity update with inertia weight). No external dependencies.

- [ ] **Step 1: Write `comparison/methods/ga.py`**

```python
from __future__ import annotations

import numpy as np

from . import register
from .base import OptimizerResult


@register("GA")
def genetic_algorithm(
    f,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_points: np.ndarray | None = None,
    maximize: bool = False,
    max_iter: int = 500,
    pop_size: int = 50,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    seed: int | None = None,
) -> OptimizerResult:
    target = (lambda x: -f(x)) if maximize else f
    rng = np.random.default_rng(seed)
    dim = bounds_lower.size

    pop = rng.uniform(bounds_lower, bounds_upper, size=(pop_size, dim))
    if start_points is not None:
        n_inject = min(len(start_points), pop_size)
        pop[:n_inject] = start_points[:n_inject]

    fitness = np.array([target(ind) for ind in pop])
    best_idx = int(np.argmin(fitness))
    best_x = np.array(pop[best_idx], copy=True)
    best_value = float(fitness[best_idx])

    for _ in range(max_iter):
        parents_idx = np.array([
            int(np.argmin(rng.choice(fitness, size=3, replace=False)))
            for _ in range(pop_size)
        ])

        offspring = np.empty_like(pop)
        for i in range(0, pop_size - 1, 2):
            p1, p2 = pop[parents_idx[i]], pop[parents_idx[i + 1]]
            if rng.random() < crossover_rate:
                alpha = rng.uniform(size=dim)
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = alpha * p2 + (1 - alpha) * p1
            else:
                c1, c2 = np.array(p1, copy=True), np.array(p2, copy=True)
            offspring[i], offspring[i + 1] = c1, c2
        if pop_size % 2 == 1:
            offspring[-1] = pop[parents_idx[-1]]

        mutation_mask = rng.random(size=(pop_size, dim)) < mutation_rate
        noise = rng.normal(0, 0.1 * (bounds_upper - bounds_lower), size=offspring.shape)
        offspring[mutation_mask] += noise[mutation_mask]
        offspring = np.clip(offspring, bounds_lower, bounds_upper)

        pop = offspring
        fitness = np.array([target(ind) for ind in pop])
        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < best_value:
            best_value = float(fitness[gen_best_idx])
            best_x = np.array(pop[gen_best_idx], copy=True)

    val = float(-best_value if maximize else best_value)
    return OptimizerResult(x_optimal=best_x, f_optimal=val, iterations=max_iter)
```

- [ ] **Step 2: Write `comparison/methods/pso.py`**

```python
from __future__ import annotations

import numpy as np

from . import register
from .base import OptimizerResult


@register("PSO")
def particle_swarm(
    f,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_points: np.ndarray | None = None,
    maximize: bool = False,
    max_iter: int = 500,
    swarm_size: int = 40,
    w: float = 0.729,
    c1: float = 1.49445,
    c2: float = 1.49445,
    seed: int | None = None,
) -> OptimizerResult:
    target = (lambda x: -f(x)) if maximize else f
    rng = np.random.default_rng(seed)
    dim = bounds_lower.size
    bounds_diff = bounds_upper - bounds_lower

    positions = rng.uniform(bounds_lower, bounds_upper, size=(swarm_size, dim))
    if start_points is not None:
        n_inject = min(len(start_points), swarm_size)
        positions[:n_inject] = start_points[:n_inject]

    velocities = rng.uniform(-bounds_diff, bounds_diff, size=(swarm_size, dim))

    fitness = np.array([target(p) for p in positions])
    pbest_pos = np.array(positions, copy=True)
    pbest_val = np.array(fitness, copy=True)

    gbest_idx = int(np.argmin(fitness))
    gbest_pos = np.array(positions[gbest_idx], copy=True)
    gbest_val = float(fitness[gbest_idx])

    for _ in range(max_iter):
        r1 = rng.uniform(size=(swarm_size, dim))
        r2 = rng.uniform(size=(swarm_size, dim))

        velocities = (
            w * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos - positions)
        )
        positions = np.clip(positions + velocities, bounds_lower, bounds_upper)
        fitness = np.array([target(p) for p in positions])

        improved = fitness < pbest_val
        pbest_pos[improved] = positions[improved]
        pbest_val[improved] = fitness[improved]

        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < gbest_val:
            gbest_val = float(fitness[gen_best_idx])
            gbest_pos = np.array(positions[gen_best_idx], copy=True)

    val = float(-gbest_val if maximize else gbest_val)
    return OptimizerResult(x_optimal=gbest_pos, f_optimal=val, iterations=max_iter)
```

- [ ] **Step 3: Add GA/PSO tests to `tests/test_comparison.py`**

Append:

```python
from comparison.methods.ga import genetic_algorithm
from comparison.methods.pso import particle_swarm


class TestGAPSO:
    @pytest.mark.parametrize("name,fn", [
        ("GA", genetic_algorithm),
        ("PSO", particle_swarm),
    ])
    def test_minimize_sphere(self, name, fn):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        result = fn(_sphere_min, bl, bu, maximize=False, max_iter=50, seed=42)
        assert result.f_optimal < 2.0
        assert result.x_optimal.shape == (2,)

    def test_all_registered(self):
        for name in ("GA", "PSO"):
            assert name in METHOD_REGISTRY
```

- [ ] **Step 4: Run tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_comparison.py::TestGAPSO -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add comparison/methods/ga.py comparison/methods/pso.py tests/test_comparison.py
git commit -m "feat: add GA and PSO implementations (no external deps)"
```

---

### Task 9: Implement domain modification

**Files:**
- Create: `comparison/domain_mod.py`
- Modify: `tests/test_comparison.py`

Migrates the rotation + shifting + asymmetrization logic from `RunComparison.R` lines 106-127. Uses `scipy.stats.ortho_group` for random rotation.

- [ ] **Step 1: Write `comparison/domain_mod.py`**

```python
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import ortho_group


def modify_domain(
    f: Callable[[np.ndarray], float],
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    dim: int,
    seed: int | None = None,
) -> tuple[Callable[[np.ndarray], float], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Q = ortho_group.rvs(dim, random_state=rng)

    bounds_diff = bounds_upper - bounds_lower
    origin_shift = bounds_diff * rng.normal(size=dim)
    push_right = rng.binomial(1, 0.5, size=dim)
    rand_small = rng.uniform(0.2, 0.3, size=dim) * bounds_diff
    rand_big = rng.uniform(0.4, 0.6, size=dim) * bounds_diff

    lower_push = push_right * rand_small - (1 - push_right) * rand_big
    upper_push = push_right * rand_big - (1 - push_right) * rand_small

    new_lower = bounds_lower + origin_shift + lower_push
    new_upper = bounds_upper + origin_shift + upper_push

    def modified_f(x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=float)
        return float(f(Q @ (x_arr - origin_shift)))

    return modified_f, new_lower, new_upper
```

- [ ] **Step 2: Add domain_mod tests to `tests/test_comparison.py`**

Append:

```python
from comparison.domain_mod import modify_domain


class TestDomainMod:
    def test_preserves_shape(self):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])

        def identity(x):
            return float(np.sum(x))

        f_mod, new_bl, new_bu = modify_domain(identity, bl, bu, dim=2, seed=42)
        assert new_bl.shape == (2,)
        assert new_bu.shape == (2,)
        assert np.all(new_bl < new_bu)

    def test_output_is_scalar(self):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        f_mod, _, _ = modify_domain(_sphere_min, bl, bu, dim=2, seed=1)
        val = f_mod(np.array([1.0, 2.0]))
        assert isinstance(val, float)
        assert np.isfinite(val)
```

- [ ] **Step 3: Run tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_comparison.py::TestDomainMod -v
```

Expected: All 2 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add comparison/domain_mod.py tests/test_comparison.py
git commit -m "feat: add domain modification (rotation/shift/asymmetrization)"
```

---

### Task 10: Implement run_comparison framework

**Files:**
- Create: `comparison/run_comparison.py`
- Modify: `tests/test_comparison.py`

Core comparison framework: loops over algorithms × replications, records f_optimal and time per run, computes RMSE/AE metrics.

- [ ] **Step 1: Write `comparison/run_comparison.py`**

```python
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from smco import run_benchmark
from smco.test_functions import assign_config

from .domain_mod import modify_domain
from .methods import METHOD_REGISTRY, get_method


SMCO_VARIANTS = {"SMCO", "SMCO_R", "SMCO_BR"}


@dataclass(eq=False)
class ComparisonResult:
    name: str
    dim: int
    to_maximize: bool
    algo_names: list[str]
    n_replications: int
    best_opt: float
    fopt_algo: np.ndarray       # (n_algo, n_replications)
    time_algo: np.ndarray       # (n_algo, n_replications)
    sum_results: dict            # per-algorithm metrics


def generate_unif_starts(
    n_starts: int,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    d = bounds_lower.size
    return bounds_lower + rng.uniform(size=(n_starts, d)) * (bounds_upper - bounds_lower)


def _run_smco_variant(
    variant: str,
    f,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_points: np.ndarray,
    smco_options: dict,
) -> tuple[float, list[float]]:
    """Run one SMCO variant. Returns (best_f_optimal, [f_optimal per start])."""
    result = run_benchmark(
        name="__direct__",
        dim=int(bounds_lower.size),
        variant=variant,
        repetitions=1,
        seed=smco_options.get("seed", 123),
        start_points=start_points,
        iter_max=smco_options.get("iter_max", 300),
        n_starts=len(start_points),
    )
    best = result.best_value
    all_vals = [r.best_result.f_optimal for r in result.results]
    return best, all_vals


def run_comparison(
    name_config: str,
    dim_config: int,
    to_maximize: bool,
    algo_names: list[str],
    n_replications: int,
    smco_options: dict | None = None,
    domain_mod: bool = False,
    truth_known: bool = False,
    true_opt: float | None = None,
    seed: int = 123,
) -> ComparisonResult:
    smco_options = smco_options or {}
    rng = np.random.default_rng(seed)
    config = assign_config(name_config, dim_config)

    bounds_lower = config.bounds_lower.copy()
    bounds_upper = config.bounds_upper.copy()
    f = config.f

    if domain_mod:
        f, bounds_lower, bounds_upper = modify_domain(f, bounds_lower, bounds_upper, dim_config, seed=rng.integers(0, 2**31))

    n_algo = len(algo_names)
    fopt_algo = np.full((n_algo, n_replications), np.nan)
    time_algo = np.full((n_algo, n_replications), np.nan)

    for i in range(n_replications):
        n_starts = smco_options.get("n_starts", max(3, int(np.ceil(np.sqrt(dim_config)))))
        start_points = generate_unif_starts(n_starts, bounds_lower, bounds_upper, rng)
        rand_start_idx = rng.integers(0, n_starts)

        for algo_idx, algo_name in enumerate(algo_names):
            t0 = time.perf_counter()

            if algo_name in SMCO_VARIANTS:
                opts = dict(smco_options)
                opts["seed"] = int(rng.integers(0, 2**31))
                fopt, _ = _run_smco_variant(algo_name, f, bounds_lower, bounds_upper, start_points, opts)
            elif algo_name in ("GenSA", "DEoptim", "GA", "PSO"):
                opt_fn = get_method(algo_name)
                result = opt_fn(
                    f, bounds_lower, bounds_upper,
                    maximize=to_maximize,
                    max_iter=smco_options.get("iter_max", 300),
                    seed=int(rng.integers(0, 2**31)),
                )
                fopt = result.f_optimal
            else:
                opt_fn = get_method(algo_name)
                result = opt_fn(
                    f, bounds_lower, bounds_upper,
                    start_points=start_points,
                    maximize=to_maximize,
                    max_iter=smco_options.get("iter_max", 300),
                )
                fopt = result.f_optimal

            time_algo[algo_idx, i] = time.perf_counter() - t0
            if to_maximize:
                # f already returns maximization-form (SMCO convention), stored as-is
                pass
            fopt_algo[algo_idx, i] = fopt

    # Compute "best opt" across all algorithms and replications
    if truth_known and true_opt is not None and np.isfinite(true_opt):
        best_opt = true_opt
    elif to_maximize:
        best_opt = float(np.nanmax(fopt_algo))
    else:
        best_opt = float(np.nanmin(fopt_algo))

    # Compute metrics
    AE = np.abs(fopt_algo - best_opt)
    sum_results = {}
    for algo_idx, algo_name in enumerate(algo_names):
        ae_row = AE[algo_idx]
        sum_results[algo_name] = {
            "time_avg": float(np.nanmean(time_algo[algo_idx])),
            "rMSE": float(np.sqrt(np.nanmean(ae_row**2))),
            "AE50": float(np.nanpercentile(ae_row, 50)),
            "AE95": float(np.nanpercentile(ae_row, 95)),
            "AE99": float(np.nanpercentile(ae_row, 99)),
            "fopt_mean": float(np.nanmean(fopt_algo[algo_idx])),
            "fopt_sd": float(np.nanstd(fopt_algo[algo_idx], ddof=1)) if n_replications > 1 else 0.0,
        }

    return ComparisonResult(
        name=name_config,
        dim=dim_config,
        to_maximize=to_maximize,
        algo_names=list(algo_names),
        n_replications=n_replications,
        best_opt=best_opt,
        fopt_algo=fopt_algo,
        time_algo=time_algo,
        sum_results=sum_results,
    )
```

- [ ] **Step 2: Add run_comparison tests to `tests/test_comparison.py`**

Append:

```python
from comparison.run_comparison import run_comparison, generate_unif_starts


class TestRunComparison:
    def test_generate_starts(self):
        bl = np.array([-5.0, -5.0])
        bu = np.array([5.0, 5.0])
        starts = generate_unif_starts(3, bl, bu)
        assert starts.shape == (3, 2)
        assert np.all(starts >= bl) and np.all(starts <= bu)

    def test_quick_comparison(self):
        result = run_comparison(
            name_config="Rastrigin",
            dim_config=2,
            to_maximize=True,
            algo_names=["SMCO_R", "GD"],
            n_replications=2,
            smco_options={"iter_max": 50, "n_starts": 3},
            seed=42,
        )
        assert len(result.algo_names) == 2
        assert result.fopt_algo.shape == (2, 2)
        assert "SMCO_R" in result.sum_results
        assert "GD" in result.sum_results
        assert "rMSE" in result.sum_results["SMCO_R"]
```

- [ ] **Step 3: Run tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_comparison.py::TestRunComparison -v
```

Expected: All 2 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add comparison/run_comparison.py tests/test_comparison.py
git commit -m "feat: add run_comparison framework with metrics computation"
```

---

### Task 11: Wire up method imports in `comparison/methods/__init__.py`

**Files:**
- Modify: `comparison/methods/__init__.py`

All method modules must be imported so their `@register` decorators fire and populate `METHOD_REGISTRY`.

- [ ] **Step 1: Update `comparison/methods/__init__.py`**

The file should contain the registry code AND import all method modules at the bottom:

```python
from __future__ import annotations

from typing import Callable

import numpy as np

from .base import OptimizerResult

OptimizerFn = Callable[
    [Callable[[np.ndarray], float], np.ndarray, np.ndarray, np.ndarray | None, bool],
    OptimizerResult,
]

METHOD_REGISTRY: dict[str, OptimizerFn] = {}


def register(name: str) -> Callable[[OptimizerFn], OptimizerFn]:
    def decorator(fn: OptimizerFn) -> OptimizerFn:
        METHOD_REGISTRY[name] = fn
        return fn
    return decorator


def get_method(name: str) -> OptimizerFn:
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown optimizer method: {name}. Available: {list(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[name]


# Import all method modules to trigger @register decorators
from . import adam as _adam  # noqa: E402, F401
from . import bobyqa as _bobyqa  # noqa: E402, F401
from . import de as _de  # noqa: E402, F401
from . import ga as _ga  # noqa: E402, F401
from . import gd as _gd  # noqa: E402, F401
from . import gensa as _gensa  # noqa: E402, F401
from . import lbfgs as _lbfgs  # noqa: E402, F401
from . import nelder_mead as _nelder_mead  # noqa: E402, F401
from . import pso as _pso  # noqa: E402, F401
from . import sa as _sa  # noqa: E402, F401
from . import sign_gd as _sign_gd  # noqa: E402, F401
from . import spsa as _spsa  # noqa: E402, F401
```

- [ ] **Step 2: Run all tests to verify everything wires up**

```bash
source .venv/bin/activate && python -m pytest tests/test_comparison.py -v
```

Expected: All tests PASS. Verify all 12 methods are registered by checking test output.

- [ ] **Step 3: Commit**

```bash
git add comparison/methods/__init__.py
git commit -m "feat: wire up all method module imports in registry"
```

---

### Task 12: Create experiment entry script

**Files:**
- Create: `scripts/run_experiment.py`

CLI entry point with `--quick` mode for validation and `--table1`/`--table2`/`--table3` for full experiments.

- [ ] **Step 1: Write `scripts/run_experiment.py`**

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path

import numpy as np

# Ensure methods are registered
import comparison.methods  # noqa: F401
from comparison.run_comparison import run_comparison

FUNC_NAMES = ["Rastrigin", "Ackley", "Griewank", "Michalewicz"]

GROUP_I_LOCAL = ["SMCO", "SMCO_R", "SMCO_BR", "GD", "SignGD", "ADAM", "SPSA", "optimLBFGS", "BOBYQA"]
GROUP_II_GLOBAL = ["SMCO", "SMCO_R", "SMCO_BR", "GenSA", "SA", "DEoptim", "GA", "PSO"]
ALL_ALGOS = GROUP_I_LOCAL + ["GenSA", "SA", "DEoptim", "GA", "PSO"]


def save_results(comp_result, output_dir: Path, label: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for algo_name in comp_result.algo_names:
        metrics = comp_result.sum_results[algo_name]
        rows.append({
            "label": label,
            "name": comp_result.name,
            "dim": comp_result.dim,
            "maximize": comp_result.to_maximize,
            "n_replications": comp_result.n_replications,
            "best_opt": comp_result.best_opt,
            "algo": algo_name,
            **metrics,
        })

    json_path = output_dir / f"{label}.json"
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    csv_path = output_dir / f"{label}.csv"
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writerows(rows)

    print(f"\n--- {label} ---")
    for row in rows:
        print(f"  {row['algo']:15s}  rMSE={row['rMSE']:.6f}  AE99={row['AE99']:.6f}  time={row['time_avg']:.4f}s")
    print(f"  Saved: {json_path}, {csv_path}")


def run_quick(output_dir: Path):
    print("=== Quick Validation ===")
    for func_name in FUNC_NAMES:
        for to_max in [True]:
            label = f"quick_{func_name}_d2_{'max' if to_max else 'min'}"
            comp = run_comparison(
                name_config=func_name,
                dim_config=2,
                to_maximize=to_max,
                algo_names=ALL_ALGOS,
                n_replications=3,
                smco_options={"iter_max": 100, "n_starts": 3},
                seed=2026,
            )
            save_results(comp, output_dir, label)


def run_table1(output_dir: Path):
    print("=== Table 1: d=2, Group I ===")
    for func_name in FUNC_NAMES:
        for to_max in [True, False]:
            label = f"table1_{func_name}_d2_{'max' if to_max else 'min'}"
            comp = run_comparison(
                name_config=func_name,
                dim_config=2,
                to_maximize=to_max,
                algo_names=GROUP_I_LOCAL,
                n_replications=500,
                smco_options={"iter_max": 500, "n_starts": 1},
                seed=2026,
            )
            save_results(comp, output_dir, label)


def run_table2(output_dir: Path):
    print("=== Table 2: d=200, Group II ===")
    for func_name in FUNC_NAMES:
        for to_max in [True, False]:
            label = f"table2_{func_name}_d200_{'max' if to_max else 'min'}"
            comp = run_comparison(
                name_config=func_name,
                dim_config=200,
                to_maximize=to_max,
                algo_names=GROUP_II_GLOBAL,
                n_replications=10,
                smco_options={"iter_max": 500, "n_starts": 14},
                seed=2026,
            )
            save_results(comp, output_dir, label)


def main():
    parser = argparse.ArgumentParser(description="Run comparison experiments")
    parser.add_argument("--quick", action="store_true", help="Quick validation (small scale)")
    parser.add_argument("--table1", action="store_true", help="Reproduce Table 1 (d=2, Group I)")
    parser.add_argument("--table2", action="store_true", help="Reproduce Table 2 (d=200, Group II)")
    parser.add_argument("--output", default="result/comparison", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.quick:
        run_quick(output_dir)
    if args.table1:
        run_table1(output_dir)
    if args.table2:
        run_table2(output_dir)
    if not (args.quick or args.table1 or args.table2):
        print("No experiment selected. Use --quick, --table1, or --table2.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run quick validation**

```bash
source .venv/bin/activate && cd /amax/math/code/SMCO && PYTHONPATH=src:. python scripts/run_experiment.py --quick
```

Expected: Prints per-algorithm metrics for each function, creates JSON/CSV files in `result/comparison/`.

- [ ] **Step 3: Verify output files exist**

```bash
ls -la result/comparison/
```

Expected: JSON and CSV files for each quick test config.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_experiment.py
git commit -m "feat: add experiment entry script with --quick/--table1/--table2 modes"
```

---

### Task 13: Run quick validation and verify results

**Files:**
- No new files; runs the experiment and inspects results.

- [ ] **Step 1: Run the quick experiment**

```bash
source .venv/bin/activate && cd /amax/math/code/SMCO && PYTHONPATH=src:. python scripts/run_experiment.py --quick
```

- [ ] **Step 2: Inspect results**

```bash
source .venv/bin/activate && cd /amax/math/code/SMCO && python -c "
import json, pathlib
for p in sorted(pathlib.Path('result/comparison').glob('*.json')):
    data = json.loads(p.read_text())
    print(f'\n=== {p.name} ===')
    for row in data:
        print(f\"  {row['algo']:15s}  rMSE={row['rMSE']:.6f}  AE99={row['AE99']:.6f}  time={row['time_avg']:.4f}s\")
"
```

Expected: Each config shows SMCO variants and all 12 comparison methods with finite metric values.

- [ ] **Step 3: Run all tests one final time**

```bash
source .venv/bin/activate && cd /amax/math/code/SMCO && python -m pytest tests/test_comparison.py -v
```

Expected: All tests PASS.

- [ ] **Step 4: Commit result files**

```bash
git add result/comparison/
git commit -m "results: add quick validation comparison results"
```

---

## Self-Review

**Spec coverage:**
- 12 comparison methods: GD, SignGD, SPSA, ADAM, L-BFGS-B, Nelder-Mead, BOBYQA, GenSA, SA, DE, GA, PSO — all covered in Tasks 2-8.
- Unified interface: Task 1 (base.py), all methods use `optimize(f, bounds_lower, bounds_upper, start_points, maximize) -> OptimizerResult`.
- RunComparison framework: Task 10.
- Domain modification: Task 9.
- Experiment entry point: Task 12.
- Quick validation: Task 13.

**Placeholder scan:** No TBD, TODO, or placeholder patterns found. All code blocks contain complete implementations.

**Type consistency:** All methods return `OptimizerResult` with `x_optimal: np.ndarray`, `f_optimal: float`, `iterations: int`. The registry uses `OptimizerFn` type alias consistently. `run_comparison` accesses these fields uniformly across SMCO variants and comparison methods.
