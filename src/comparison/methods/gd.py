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
    is_better = (lambda cur, best: cur > best) if maximize else (lambda cur, best: cur < best)
    best_x = None
    best_value = -np.inf if maximize else np.inf
    total_iters = 0
    for sp in start_points:
        x = np.array(sp, dtype=float, copy=True)
        velocity = np.zeros_like(x)
        lr = learning_rate
        prev_value = float(f(x))
        if is_better(prev_value, best_value):
            best_value = prev_value
            best_x = np.array(x, copy=True)
        for it in range(1, max_iter + 1):
            grad = _numerical_gradient(f, x)
            velocity = momentum * velocity + sign * lr * grad
            x_new = np.clip(x - velocity, bounds_lower, bounds_upper)
            current_value = float(f(x_new))
            if not is_better(current_value, prev_value) and it > 1:
                lr *= 0.5
                velocity *= 0.5
            if is_better(current_value, best_value):
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
