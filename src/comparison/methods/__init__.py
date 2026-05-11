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


# Method modules will be imported here once created:
# from . import gd, sign_gd, spsa, adam, lbfgs, nelder_mead, bobyqa, gensa, sa, de, ga, pso
