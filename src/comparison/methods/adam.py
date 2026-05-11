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
