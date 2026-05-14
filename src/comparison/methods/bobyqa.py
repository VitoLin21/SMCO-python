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
    is_better = (lambda cur, best: cur > best) if maximize else (lambda cur, best: cur < best)
    best_x = None
    best_value = -np.inf if maximize else np.inf
    total_iters = 0
    for sp in start_points:
        res = scipy_minimize(
            target, np.array(sp, dtype=float), method="COBYLA",
            bounds=bounds, options={"maxiter": max_iter, "rhobeg": 0.1},
        )
        total_iters += res.nfev
        val = float(-res.fun if maximize else res.fun)
        if is_better(val, best_value):
            best_value = val
            best_x = np.array(res.x, copy=True)
    if best_x is None:
        best_x = np.array(start_points[0], dtype=float, copy=True)
        best_value = float(f(best_x))
    return OptimizerResult(x_optimal=best_x, f_optimal=best_value, iterations=total_iters)
