from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from smco.optimizer import smco, smco_br, smco_r
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

    # config.f 与 raw 的关系：sense="min" 时 config.f = -raw，sense="max" 时 config.f = raw
    raw_f = (lambda x, _f=config.f: -_f(x)) if config.sense == "min" else config.f

    # 构造 effective_f：始终是要最大化的函数
    if to_maximize:
        effective_f = raw_f
    else:
        effective_f = lambda x, _f=raw_f: -_f(x)

    f = effective_f
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
            algo_seed = int(rng.integers(0, 2**31))

            if algo_name in SMCO_VARIANTS:
                # SMCO 变体必须走各自公开包装器，不能统一退化成 smco_multi。
                variant_map = {"SMCO": smco, "SMCO_R": smco_r, "SMCO_BR": smco_br}
                variant_fn = variant_map[algo_name]
                variant_options = {
                    "iter_max": smco_options.get("iter_max", 300),
                    "bounds_buffer": 0.05,
                    "buffer_rand": True,
                    "tol_conv": 1e-8,
                    "seed": algo_seed,
                }
                if algo_name in ("SMCO_R", "SMCO_BR"):
                    variant_options["refine_ratio"] = 0.5
                if algo_name == "SMCO_BR":
                    variant_options["iter_boost"] = 99
                smco_result = variant_fn(
                    f,
                    bounds_lower,
                    bounds_upper,
                    start_points=start_points,
                    **variant_options,
                )
                fopt = smco_result.best_result.f_optimal
            elif algo_name in ("GenSA", "DEoptim", "GA", "PSO"):
                opt_fn = get_method(algo_name)
                result = opt_fn(
                    f, bounds_lower, bounds_upper,
                    maximize=True,
                    max_iter=smco_options.get("iter_max", 300),
                    seed=algo_seed,
                )
                fopt = result.f_optimal
            else:
                opt_fn = get_method(algo_name)
                result = opt_fn(
                    f, bounds_lower, bounds_upper,
                    start_points=start_points,
                    maximize=True,
                    max_iter=smco_options.get("iter_max", 300),
                )
                fopt = result.f_optimal

            time_algo[algo_idx, i] = time.perf_counter() - t0
            # fopt 是 effective_f(x*) 的最大值，转换回 raw 标尺
            if to_maximize:
                raw_value = fopt
            else:
                raw_value = -fopt
            fopt_algo[algo_idx, i] = raw_value

    # 计算 best_opt（在 raw 标尺上）
    if truth_known and true_opt is not None and np.isfinite(true_opt):
        best_opt = true_opt
    elif to_maximize:
        best_opt = float(np.nanmax(fopt_algo))
    else:
        best_opt = float(np.nanmin(fopt_algo))

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
