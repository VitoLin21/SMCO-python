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
