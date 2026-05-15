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
    def target(x):
        x_arr = np.asarray(x, dtype=float)
        if np.any((x_arr < bounds_lower) | (x_arr > bounds_upper)):
            distance = np.maximum(bounds_lower - x_arr, 0.0) + np.maximum(x_arr - bounds_upper, 0.0)
            return float(1e100 + np.sum(distance**2))
        value = float(f(x_arr))
        return -value if maximize else value

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

    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "bounds": list(zip(bounds_lower, bounds_upper)),
    }
    res = basinhopping(
        target,
        x0,
        niter=max_iter,
        accept_test=BoundsCheck(),
        minimizer_kwargs=minimizer_kwargs,
        seed=seed,
    )
    x_clipped = np.clip(res.x, bounds_lower, bounds_upper)
    val = float(f(x_clipped))
    return OptimizerResult(x_optimal=np.array(x_clipped, copy=True), f_optimal=val, iterations=res.nit)
