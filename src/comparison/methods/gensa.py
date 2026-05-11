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
