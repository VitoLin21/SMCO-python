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
from . import gd, sign_gd, spsa, adam  # noqa: F401
from . import lbfgs as _lbfgs  # noqa: F401
from . import nelder_mead as _nelder_mead  # noqa: F401
from . import bobyqa as _bobyqa  # noqa: F401
from . import gensa as _gensa  # noqa: F401
from . import sa as _sa  # noqa: F401
from . import de as _de  # noqa: F401
from . import ga as _ga  # noqa: F401
from . import pso as _pso  # noqa: F401
