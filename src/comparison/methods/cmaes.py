from __future__ import annotations

import numpy as np

from . import register
from .base import OptimizerResult


@register("CMA-ES")
def cma_es(
    f,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_points: np.ndarray | None = None,
    maximize: bool = False,
    max_iter: int = 500,
    seed: int | None = None,
) -> OptimizerResult:
    """Separable (diagonal-covariance) CMA-ES for high dimensions.

    Uses cma's ``CMA_diagonal`` option so only the covariance diagonal is
    adapted (O(d) memory), scaling to the paper's d<=5000 regime where
    full-covariance CMA-ES is infeasible — the "limited-memory" variant the
    experiment plan (sec 4.3) calls for. FE is bounded by ``max_iter`` (cma
    ``maxfevals``); an ``EvaluationBudgetExceeded`` raised by the objective
    propagates out of the ask/tell loop and is caught by the worker, which
    then reports the best-so-far from its own observer.
    """
    import cma

    target = (lambda x: -f(x)) if maximize else f
    lo = np.asarray(bounds_lower, dtype=float)
    hi = np.asarray(bounds_upper, dtype=float)
    if start_points is not None and len(start_points) > 0:
        x0 = np.mean(np.asarray(start_points, dtype=float), axis=0)
    else:
        x0 = (lo + hi) / 2
    sigma0 = float(np.mean(hi - lo) / 4)
    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            "maxfevals": int(max_iter),
            "seed": int(seed) if seed is not None else 0,
            "verbose": -9,
            "bounds": [list(lo), list(hi)],
            "CMA_diagonal": True,
        },
    )
    while not es.stop():
        xs = es.ask()
        vals = [target(x) for x in xs]  # propagates EvaluationBudgetExceeded
        es.tell(xs, vals)
    res = es.result
    val = float(-res.fbest if maximize else res.fbest)
    return OptimizerResult(
        x_optimal=np.array(res.xbest, copy=True),
        f_optimal=val,
        iterations=int(res.evaluations),
    )
