from __future__ import annotations
import numpy as np
from . import register
from .base import OptimizerResult


@register("GA")
def genetic_algorithm(
    f,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_points: np.ndarray | None = None,
    maximize: bool = False,
    max_iter: int = 500,
    pop_size: int = 50,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    seed: int | None = None,
) -> OptimizerResult:
    target = (lambda x: -f(x)) if maximize else f
    rng = np.random.default_rng(seed)
    dim = bounds_lower.size
    pop = rng.uniform(bounds_lower, bounds_upper, size=(pop_size, dim))
    if start_points is not None:
        n_inject = min(len(start_points), pop_size)
        pop[:n_inject] = start_points[:n_inject]
    fitness = np.array([target(ind) for ind in pop])
    best_idx = int(np.argmin(fitness))
    best_x = np.array(pop[best_idx], copy=True)
    best_value = float(fitness[best_idx])
    for _ in range(max_iter):
        parents_idx = np.array([
            int(np.argmin(rng.choice(fitness, size=min(3, pop_size), replace=False)))
            for _ in range(pop_size)
        ])
        offspring = np.empty_like(pop)
        for i in range(0, pop_size - 1, 2):
            p1, p2 = pop[parents_idx[i]], pop[parents_idx[i + 1]]
            if rng.random() < crossover_rate:
                alpha = rng.uniform(size=dim)
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = alpha * p2 + (1 - alpha) * p1
            else:
                c1, c2 = np.array(p1, copy=True), np.array(p2, copy=True)
            offspring[i], offspring[i + 1] = c1, c2
        if pop_size % 2 == 1:
            offspring[-1] = pop[parents_idx[-1]]
        mutation_mask = rng.random(size=(pop_size, dim)) < mutation_rate
        noise = rng.normal(0, 0.1 * (bounds_upper - bounds_lower), size=offspring.shape)
        offspring[mutation_mask] += noise[mutation_mask]
        offspring = np.clip(offspring, bounds_lower, bounds_upper)
        pop = offspring
        fitness = np.array([target(ind) for ind in pop])
        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < best_value:
            best_value = float(fitness[gen_best_idx])
            best_x = np.array(pop[gen_best_idx], copy=True)
    val = float(-best_value if maximize else best_value)
    return OptimizerResult(x_optimal=best_x, f_optimal=val, iterations=max_iter)
