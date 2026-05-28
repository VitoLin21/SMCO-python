from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from smco.optimizer import smco, smco_br, smco_br_evo, smco_evo, smco_r, smco_r_evo
from smco.test_functions import assign_config

from .domain_mod import modify_domain
from .methods import METHOD_REGISTRY, get_method


SMCO_VARIANT_MAP = {
    "SMCO": smco,
    "SMCO_R": smco_r,
    "SMCO_BR": smco_br,
    "SMCO_EVO": smco_evo,
    "SMCO_R_EVO": smco_r_evo,
    "SMCO_BR_EVO": smco_br_evo,
}
SMCO_VARIANTS = set(SMCO_VARIANT_MAP)
GLOBAL_METHODS = {"GenSA", "DEoptim", "GA", "PSO"}
COMPARISON_SMCO_DEFAULTS: dict[str, Any] = {
    "iter_max": 300,
    "bounds_buffer": 0.05,
    "buffer_rand": True,
    "tol_conv": 1e-8,
}


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


def _call_with_supported_kwargs(fn, *args, **kwargs):
    signature = inspect.signature(fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        supported = kwargs
    else:
        supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return fn(*args, **supported)


def _smco_variant_options(algo_name: str, smco_options: dict, algo_seed: int) -> dict:
    options = dict(COMPARISON_SMCO_DEFAULTS)
    # n_starts 只用于生成共享起点，不应继续透传给 SMCO。
    options.update({key: value for key, value in smco_options.items() if key != "n_starts"})
    # Evolution-only controls should not leak into non-evo wrappers.
    if algo_name not in ("SMCO_EVO", "SMCO_R_EVO", "SMCO_BR_EVO"):
        for key in ("evolution_strategy", "evolution_points", "elimination_rate", "de_factor", "de_crossover"):
            options.pop(key, None)
    if algo_name in ("SMCO_R", "SMCO_BR", "SMCO_R_EVO", "SMCO_BR_EVO"):
        options.setdefault("refine_ratio", 0.5)
    if algo_name in ("SMCO_BR", "SMCO_BR_EVO"):
        base_iter_max = int(options["iter_max"])
        # SMCO-BR 会执行 regular 和 boosted 两次 refine；比较实验中将单次预算减半。
        options["iter_max"] = int(math.ceil(base_iter_max / 2.0))
        options.setdefault("iter_boost", 99)
    else:
        options.pop("iter_boost", None)
    options["seed"] = algo_seed
    return options


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
                variant_fn = SMCO_VARIANT_MAP[algo_name]
                variant_options = _smco_variant_options(algo_name, smco_options, algo_seed)
                smco_result = variant_fn(
                    f,
                    bounds_lower,
                    bounds_upper,
                    start_points=start_points,
                    **variant_options,
                )
                fopt = smco_result.best_result.f_optimal
            elif algo_name in GLOBAL_METHODS:
                opt_fn = get_method(algo_name)
                result = _call_with_supported_kwargs(
                    opt_fn,
                    f, bounds_lower, bounds_upper,
                    maximize=True,
                    max_iter=smco_options.get("iter_max", 300),
                    seed=algo_seed,
                )
                fopt = result.f_optimal
            else:
                opt_fn = get_method(algo_name)
                result = _call_with_supported_kwargs(
                    opt_fn,
                    f, bounds_lower, bounds_upper,
                    start_points=start_points,
                    maximize=True,
                    max_iter=smco_options.get("iter_max", 300),
                    seed=algo_seed,
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
