from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .optimizer import smco, smco_br, smco_br_evo, smco_evo, smco_multi, smco_r, smco_r_evo
from .results import SMCOResult
from .test_functions import BenchmarkConfig, assign_config


@dataclass(eq=False)
class BenchmarkRun:
    config: BenchmarkConfig
    variant: str
    results: list[SMCOResult]
    best_value: float
    best_x: np.ndarray


def _variant_function(variant: str) -> Callable[..., SMCOResult]:
    if not isinstance(variant, str):
        raise TypeError("variant must be a string")
    normalized_variant = variant.lower()
    variants = {
        "smco": smco,
        "smco_r": smco_r,
        "smco_br": smco_br,
        "smco_evo": smco_evo,
        "smco_r_evo": smco_r_evo,
        "smco_br_evo": smco_br_evo,
        "smco_multi": smco_multi,
    }
    try:
        return variants[normalized_variant]
    except KeyError as exc:
        raise ValueError(f"unknown SMCO variant: {variant}") from exc


def _validate_repetitions(repetitions: int) -> int:
    if isinstance(repetitions, (bool, np.bool_)) or not isinstance(repetitions, (int, np.integer)):
        raise ValueError("repetitions must be a positive integer")
    repetitions = int(repetitions)
    if repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    return repetitions


def _validate_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer or None")
    return int(seed)


def run_benchmark(
    name: str,
    dim: int,
    *,
    variant: str = "smco_r",
    repetitions: int = 1,
    seed: int | None = 123,
    **smco_options: Any,
) -> BenchmarkRun:
    # 统一入口：先解析 benchmark 配置，再按 variant 重复运行并汇总最优结果。
    config = assign_config(name, dim)
    variant_fn = _variant_function(variant)
    normalized_variant = variant.lower()
    repetitions = _validate_repetitions(repetitions)
    base_seed = _validate_seed(seed)
    base_options = dict(smco_options)
    start_points = base_options.pop("start_points", None)

    results: list[SMCOResult] = []
    for rep in range(repetitions):
        options = dict(base_options)
        # 每次重复在 base seed 上偏移，保证可复现且彼此不同。
        options["seed"] = None if base_seed is None else base_seed + rep
        if normalized_variant == "smco_multi":
            result = variant_fn(
                config.f,
                config.bounds_lower,
                config.bounds_upper,
                start_points=start_points,
                opt_control=options,
            )
        else:
            result = variant_fn(
                config.f,
                config.bounds_lower,
                config.bounds_upper,
                start_points=start_points,
                **options,
            )
        results.append(result)

    values = np.array([result.best_result.f_optimal for result in results], dtype=float)
    best_idx = int(np.argmax(values))
    best_result = results[best_idx].best_result
    return BenchmarkRun(
        config=config,
        variant=normalized_variant,
        results=results,
        best_value=float(best_result.f_optimal),
        best_x=np.array(best_result.x_optimal, dtype=float, copy=True),
    )
