from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from smco import run_benchmark
from smco.test_functions import assign_config

from .domain_mod import modify_domain
from .methods import METHOD_REGISTRY, get_method


SMCO_VARIANTS = {"SMCO", "SMCO_R", "SMCO_BR"}


@dataclass(eq=False)
class ComparisonResult:
    name: str
    dim: int
    to_maximize: bool
    algo_names: list[str]
    n_replications: int
    best_opt: float
    fopt_algo: np.ndarray
    time_algo: np.ndarray
    sum_results: dict


def generate_unif_starts(
    n_starts: int,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    d = bounds_lower.size
    return bounds_lower + rng.uniform(size=(n_starts, d)) * (bounds_upper - bounds_lower)


def run_comparison(
    name_config: str,
    dim_config: int,
    to_maximize: bool,
    algo_names: list[str],
    n_replications: int,
    smco_options: dict | None = None,
    domain_mod: bool = False,
    truth_known: bool = False,
    true_opt: float | None = None,
    seed: int = 123,
) -> ComparisonResult:
    smco_options = smco_options or {}
    rng = np.random.default_rng(seed)
    config = assign_config(name_config, dim_config)

    bounds_lower = config.bounds_lower.copy()
    bounds_upper = config.bounds_upper.copy()
    f = config.f

    if domain_mod:
        f, bounds_lower, bounds_upper = modify_domain(
            f, bounds_lower, bounds_upper, dim_config,
            seed=int(rng.integers(0, 2**31)),
        )

    n_algo = len(algo_names)
    fopt_algo = np.full((n_algo, n_replications), np.nan)
    time_algo = np.full((n_algo, n_replications), np.nan)

    for i in range(n_replications):
        n_starts = smco_options.get("n_starts", max(3, int(np.ceil(np.sqrt(dim_config)))))
        start_points = generate_unif_starts(n_starts, bounds_lower, bounds_upper, rng)

        for algo_idx, algo_name in enumerate(algo_names):
            t0 = time.perf_counter()

            if algo_name in SMCO_VARIANTS:
                # SMCO 变体：使用 run_benchmark
                # run_benchmark 内部已处理最大化（config.f 已是最大化形式）
                result = run_benchmark(
                    name=name_config,
                    dim=dim_config,
                    variant=algo_name.lower(),
                    repetitions=1,
                    seed=int(rng.integers(0, 2**31)),
                    start_points=start_points,
                    iter_max=smco_options.get("iter_max", 300),
                    n_starts=n_starts,
                )
                fopt = result.best_value
            elif algo_name in ("GenSA", "DEoptim", "GA", "PSO"):
                # 全局方法：不需要 start_points
                opt_fn = get_method(algo_name)
                # config.f 已经是最大化形式；全局方法传入 maximize=True
                result = opt_fn(
                    f, bounds_lower, bounds_upper,
                    maximize=True,
                    max_iter=smco_options.get("iter_max", 300),
                    seed=int(rng.integers(0, 2**31)),
                )
                fopt = result.f_optimal
            else:
                # 局部对比方法：需要 start_points
                opt_fn = get_method(algo_name)
                # config.f 已经是最大化形式；传入 maximize=True
                result = opt_fn(
                    f, bounds_lower, bounds_upper,
                    start_points=start_points,
                    maximize=True,
                    max_iter=smco_options.get("iter_max", 300),
                )
                fopt = result.f_optimal

            time_algo[algo_idx, i] = time.perf_counter() - t0
            fopt_algo[algo_idx, i] = fopt

    # 计算 best_opt
    if truth_known and true_opt is not None and np.isfinite(true_opt):
        best_opt = true_opt
    else:
        best_opt = float(np.nanmax(fopt_algo))

    # 计算指标
    AE = np.abs(fopt_algo - best_opt)
    sum_results = {}
    for algo_idx, algo_name in enumerate(algo_names):
        ae_row = AE[algo_idx]
        sum_results[algo_name] = {
            "time_avg": float(np.nanmean(time_algo[algo_idx])),
            "rMSE": float(np.sqrt(np.nanmean(ae_row**2))),
            "AE50": float(np.nanpercentile(ae_row, 50)),
            "AE95": float(np.nanpercentile(ae_row, 95)),
            "AE99": float(np.nanpercentile(ae_row, 99)),
            "fopt_mean": float(np.nanmean(fopt_algo[algo_idx])),
            "fopt_sd": float(np.nanstd(fopt_algo[algo_idx], ddof=1)) if n_replications > 1 else 0.0,
        }

    return ComparisonResult(
        name=name_config,
        dim=dim_config,
        to_maximize=to_maximize,
        algo_names=list(algo_names),
        n_replications=n_replications,
        best_opt=best_opt,
        fopt_algo=fopt_algo,
        time_algo=time_algo,
        sum_results=sum_results,
    )
