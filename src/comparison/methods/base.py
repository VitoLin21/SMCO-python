from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)
class OptimizerResult:
    x_optimal: np.ndarray
    f_optimal: float
    iterations: int
