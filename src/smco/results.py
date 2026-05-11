from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(eq=False)
class SingleResult:
    x_optimal: np.ndarray
    f_optimal: float
    iterations: int
    x_runmax: np.ndarray | None = None
    f_runmax: float | None = None


@dataclass(eq=False)
class SMCOResult:
    best_result: SingleResult
    all_results: list[SingleResult]
    opt_control: dict[str, Any]
    summary: dict[str, Any]
