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
    is_better = (lambda cur, best: cur > best) if maximize else (lambda cur, best: cur < best)
    best_x = None
    best_value = -np.inf if maximize else np.inf
    total_iters = 0
    for sp in start_points:
        x = np.array(sp, dtype=float, copy=True)
        d = x.size
        prev_value = float(f(x))
        if is_better(prev_value, best_value):
            best_value = prev_value
            best_x = np.array(x, copy=True)
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
            if is_better(current_value, best_value):
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
