#!/usr/bin/env python
"""Generate fixed start points + run Python smco_evo for R↔Python alignment.

Writes starts to a .csv, then runs smco_evo with those starts and prints the
result so the R side (which reads the same starts) can be compared. The goal is
*behavioral* alignment (both converge near the global optimum, same
evolution_history structure), not bitwise equality — R and Python use different
RNGs and Sobol implementations.
"""
import csv
import sys

import numpy as np

from smco import smco_evo
from smco.test_functions import assign_config


def main():
    func = sys.argv[1] if len(sys.argv) > 1 else "Rastrigin"
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    n_starts = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    starts_csv = sys.argv[4] if len(sys.argv) > 4 else "starts.csv"

    config = assign_config(func, dim)
    rng = np.random.default_rng(20260615)
    starts = config.bounds_lower + rng.uniform(size=(n_starts, dim)) * (
        config.bounds_upper - config.bounds_lower
    )

    with open(starts_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        for row in starts:
            w.writerow([f"{v:.17g}" for v in row])
    print(f"wrote {n_starts}x{dim} starts to {starts_csv}")

    # config.f is ALREADY the SMCO maximization target: assign_config's
    # _min_config wraps the raw minimizer as objective(x) = -raw(x). Do NOT
    # negate again -- that would flip Rastrigin into "push to boundary".
    f = config.f
    result = smco_evo(
        f, config.bounds_lower.copy(), config.bounds_upper.copy(),
        start_points=starts,
        iter_max=200, bounds_buffer=0.05, buffer_rand=True, tol_conv=1e-8,
        evolution_points=(0.5, 0.75), elimination_rate=0.25,
        evolution_strategy="rand1bin", de_factor=0.8, de_crossover=0.7,
        seed=20260615,
    )
    print(f"PY_RESULT f_optimal={result.best_result.f_optimal!r}")
    print(f"PY_RESULT iters={result.best_result.iterations}")
    print(f"PY_RESULT x_norm={np.linalg.norm(result.best_result.x_optimal)!r}")
    print(f"PY_RESULT n_history={len(result.summary['evolution_history'])}")


if __name__ == "__main__":
    main()
