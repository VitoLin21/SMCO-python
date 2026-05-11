from __future__ import annotations
import numpy as np
from . import register
from .base import OptimizerResult


@register("PSO")
def particle_swarm(
    f,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_points: np.ndarray | None = None,
    maximize: bool = False,
    max_iter: int = 500,
    swarm_size: int = 40,
    w: float = 0.729,
    c1: float = 1.49445,
    c2: float = 1.49445,
    seed: int | None = None,
) -> OptimizerResult:
    target = (lambda x: -f(x)) if maximize else f
    rng = np.random.default_rng(seed)
    dim = bounds_lower.size
    bounds_diff = bounds_upper - bounds_lower
    positions = rng.uniform(bounds_lower, bounds_upper, size=(swarm_size, dim))
    if start_points is not None:
        n_inject = min(len(start_points), swarm_size)
        positions[:n_inject] = start_points[:n_inject]
    velocities = rng.uniform(-bounds_diff, bounds_diff, size=(swarm_size, dim))
    fitness = np.array([target(p) for p in positions])
    pbest_pos = np.array(positions, copy=True)
    pbest_val = np.array(fitness, copy=True)
    gbest_idx = int(np.argmin(fitness))
    gbest_pos = np.array(positions[gbest_idx], copy=True)
    gbest_val = float(fitness[gbest_idx])
    for _ in range(max_iter):
        r1 = rng.uniform(size=(swarm_size, dim))
        r2 = rng.uniform(size=(swarm_size, dim))
        velocities = w * velocities + c1 * r1 * (pbest_pos - positions) + c2 * r2 * (gbest_pos - positions)
        positions = np.clip(positions + velocities, bounds_lower, bounds_upper)
        fitness = np.array([target(p) for p in positions])
        improved = fitness < pbest_val
        pbest_pos[improved] = positions[improved]
        pbest_val[improved] = fitness[improved]
        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < gbest_val:
            gbest_val = float(fitness[gen_best_idx])
            gbest_pos = np.array(positions[gen_best_idx], copy=True)
    val = float(-gbest_val if maximize else gbest_val)
    return OptimizerResult(x_optimal=gbest_pos, f_optimal=val, iterations=max_iter)
