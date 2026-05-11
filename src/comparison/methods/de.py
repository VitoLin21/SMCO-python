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
