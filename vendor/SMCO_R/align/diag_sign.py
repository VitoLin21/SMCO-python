"""Diagnose config.f sign + which objective finds the Rastrigin optimum."""
import numpy as np
from smco.test_functions import assign_config, rastrigin
from smco import smco_evo

cfg = assign_config("Rastrigin", 10)
print("sense:", cfg.sense)
print("config.f(zeros):", cfg.f(np.zeros(10)), "(raw rastrigin(zeros)=", rastrigin(np.zeros(10)), ")")
print("config.f(5s):   ", cfg.f(np.full(10, 5.0)))
print("known_best_objective:", cfg.known_best_objective)

for label, f in [("config.f", cfg.f), ("-config.f", lambda x: -cfg.f(x))]:
    r = smco_evo(f, cfg.bounds_lower.copy(), cfg.bounds_upper.copy(),
                 iter_max=200, seed=1)
    x = r.best_result.x_optimal
    print(f"[{label}] fopt={r.best_result.f_optimal:.4f} "
          f"xnorm={np.linalg.norm(x):.4f} raw_rastrigin={rastrigin(x):.4f}")
