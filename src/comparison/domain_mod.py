from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import ortho_group


def modify_domain(
    f: Callable[[np.ndarray], float],
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    dim: int,
    seed: int | None = None,
) -> tuple[Callable[[np.ndarray], float], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Q = ortho_group.rvs(dim, random_state=rng)

    bounds_diff = bounds_upper - bounds_lower
    origin_shift = bounds_diff * rng.normal(size=dim)
    push_right = rng.binomial(1, 0.5, size=dim)
    rand_small = rng.uniform(0.2, 0.3, size=dim) * bounds_diff
    rand_big = rng.uniform(0.4, 0.6, size=dim) * bounds_diff

    lower_push = push_right * rand_small - (1 - push_right) * rand_big
    upper_push = push_right * rand_big - (1 - push_right) * rand_small

    new_lower = bounds_lower + origin_shift + lower_push
    new_upper = bounds_upper + origin_shift + upper_push

    def modified_f(x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=float)
        return float(f(Q @ (x_arr - origin_shift)))

    return modified_f, new_lower, new_upper
