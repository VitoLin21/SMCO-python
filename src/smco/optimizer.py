from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import math

import numpy as np
import scipy.stats.qmc as qmc

from .results import SMCOResult, SingleResult

Objective = Callable[[np.ndarray], float]


@dataclass
class BoundsCheck:
    is_out: bool
    x_in: np.ndarray


@dataclass
class PartialSignResult:
    signs: np.ndarray
    f_partial_best: float
    x_partial_best: np.ndarray


def _as_1d_float_array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional numeric vector")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=float, copy=True)


def _as_start_points(start_points: Any, dim: int) -> np.ndarray:
    array = np.asarray(start_points, dtype=float)
    if array.ndim == 1:
        if dim == 1:
            array = array.reshape(-1, 1)
        else:
            array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("start_points must be a two-dimensional array")
    if array.shape[1] != dim:
        raise ValueError(f"start_points must have {dim} columns")
    if array.shape[0] < 1:
        raise ValueError("start_points must have at least one row")
    if not np.all(np.isfinite(array)):
        raise ValueError("start_points must contain only finite values")
    return np.array(array, dtype=float, copy=True)


def _as_positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer") from None
    if not math.isfinite(number) or not number.is_integer() or number < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


def _as_non_negative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a non-negative integer") from None
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(number)


def _as_integer(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer") from None
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _validate_bounds(bounds_lower: Any, bounds_upper: Any) -> tuple[np.ndarray, np.ndarray]:
    lower = _as_1d_float_array("bounds_lower", bounds_lower)
    upper = _as_1d_float_array("bounds_upper", bounds_upper)
    if lower.shape != upper.shape:
        raise ValueError("bounds_lower and bounds_upper must have the same length")
    if np.any(lower >= upper):
        raise ValueError("bounds_lower must be strictly less than bounds_upper")
    return lower, upper


def validate_smco_inputs(
    f: Any,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    opt_control: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not callable(f):
        raise TypeError("f must be callable")

    lower, upper = _validate_bounds(bounds_lower, bounds_upper)

    starts = None
    if start_points is not None:
        starts = _as_start_points(start_points, lower.size)
        if np.any((starts < lower) | (starts > upper)):
            raise ValueError("start_points must be within bounds")

    control = opt_control or {}
    if control.get("n_starts") is not None:
        _as_positive_integer("n_starts", control["n_starts"])
    if control.get("iter_max") is not None:
        _as_positive_integer("iter_max", control["iter_max"])
    if control.get("iter_boost") is not None:
        _as_non_negative_integer("iter_boost", control["iter_boost"])
    if control.get("iter_nstart") is not None:
        _as_positive_integer("iter_nstart", control["iter_nstart"])
    if control.get("bounds_buffer") is not None:
        bounds_buffer = float(control["bounds_buffer"])
        if not math.isfinite(bounds_buffer) or bounds_buffer < 0:
            raise ValueError("bounds_buffer must be non-negative")
    if control.get("refine_ratio") is not None:
        ratio = float(control["refine_ratio"])
        if not math.isfinite(ratio) or ratio < 0 or ratio > 1:
            raise ValueError("refine_ratio must be between 0 and 1")
    if control.get("tol_conv") is not None:
        tol_conv = float(control["tol_conv"])
        if not math.isfinite(tol_conv) or tol_conv <= 0:
            raise ValueError("tol_conv must be positive")
    if control.get("seed") is not None:
        _as_integer("seed", control["seed"])
    for key in ("buffer_rand", "refine_search", "use_runmax", "use_parallel", "verbose"):
        if key in control and not isinstance(control[key], bool):
            raise ValueError(f"{key} must be bool")
    if control.get("partial_option") is not None and control["partial_option"] not in {"center", "forward"}:
        raise ValueError("partial_option must be 'center' or 'forward'")

    return lower, upper, starts


def generate_sobol_points(
    n_starts: int,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    n_starts = _as_positive_integer("n_starts", n_starts)
    bounds_lower, bounds_upper = _validate_bounds(bounds_lower, bounds_upper)
    dim = int(bounds_lower.size)
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    m = int(math.ceil(math.log2(max(1, n_starts))))
    points = sampler.random_base2(m=m)[:n_starts]
    return qmc.scale(points, bounds_lower, bounds_upper)


def check_bounds(
    x: np.ndarray,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
) -> BoundsCheck:
    is_out = bool(np.any((x < bounds_lower) | (x > bounds_upper)))
    return BoundsCheck(is_out=is_out, x_in=np.clip(x, bounds_lower, bounds_upper))


def compute_partial_signs(
    f: Objective,
    x: np.ndarray,
    fx: float,
    h_step: np.ndarray,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    partial_option: str = "center",
    use_runmax: bool = True,
) -> PartialSignResult:
    x_work = np.array(x, dtype=float, copy=True)
    partial_signs = np.zeros_like(x_work, dtype=float)
    f_partial_best = float(fx)
    best_j: int | None = None
    best_value: float | None = None

    if partial_option == "center":
        for j in range(x_work.size):
            original = float(x_work[j])

            x_work[j] = min(original + h_step[j], bounds_upper[j])
            f_plus = float(f(x_work))

            x_work[j] = max(original - h_step[j], bounds_lower[j])
            f_minus = float(f(x_work))

            sign = f_plus > f_minus
            partial_signs[j] = 1.0 if sign else 0.0

            if use_runmax:
                if sign and f_plus > f_partial_best:
                    f_partial_best = f_plus
                    best_j = j
                    best_value = min(original + h_step[j], bounds_upper[j])
                elif (not sign) and f_minus > f_partial_best:
                    f_partial_best = f_minus
                    best_j = j
                    best_value = max(original - h_step[j], bounds_lower[j])

            x_work[j] = original
    elif partial_option == "forward":
        for j in range(x_work.size):
            original = float(x_work[j])

            if original >= bounds_upper[j]:
                x_work[j] = max(original - h_step[j], bounds_lower[j])
                f_perturb = float(f(x_work))
                sign = fx > f_perturb
            else:
                x_work[j] = min(original + h_step[j], bounds_upper[j])
                f_perturb = float(f(x_work))
                sign = f_perturb > fx

            partial_signs[j] = 1.0 if sign else 0.0

            if use_runmax and f_perturb > f_partial_best:
                f_partial_best = f_perturb
                best_j = j
                best_value = float(x_work[j])

            x_work[j] = original
    else:
        raise ValueError("partial_option must be 'center' or 'forward'")

    x_partial_best = np.array(x, dtype=float, copy=True)
    if best_j is not None and best_value is not None:
        x_partial_best[best_j] = best_value

    return PartialSignResult(
        signs=partial_signs,
        f_partial_best=f_partial_best,
        x_partial_best=x_partial_best,
    )


DEFAULT_CONTROL: dict[str, Any] = {
    "iter_max": 500,
    "iter_boost": 0,
    "bounds_buffer": 0.05,
    "buffer_rand": False,
    "tol_conv": 1e-8,
    "refine_search": True,
    "refine_ratio": 0.5,
    "partial_option": "center",
    "use_runmax": True,
    "use_parallel": False,
    "verbose": False,
    "seed": 123,
}


def _merge_control(user_control: dict[str, Any] | None) -> dict[str, Any]:
    control = dict(DEFAULT_CONTROL)
    if user_control:
        control.update(user_control)
    return control


def _split_refine_iterations(iter_max: int, ratio: float, refine_search: bool) -> tuple[int, int]:
    if not refine_search:
        return int(iter_max), 0
    return int(round(iter_max * (1.0 - ratio))), int(round(iter_max * ratio))


def _single(
    f: Objective,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_point: np.ndarray,
    *,
    bounds_buffer: float,
    buffer_rand: bool,
    iter_max: int,
    iter_nstart: int,
    iter_boost: int,
    tol_conv: float,
    partial_option: str,
    use_runmax: bool,
    rng: np.random.Generator,
) -> SingleResult:
    # 单起点 SMCO 主循环：维护当前点、运行最优(runmax)及累计平均状态。
    x_current = np.array(start_point, dtype=float, copy=True)
    f_current = float(f(x_current))
    f_runmax = f_current
    x_runmax = np.array(x_current, copy=True)

    bounds_diff = bounds_upper - bounds_lower
    n_boost_1 = int(iter_boost) + int(iter_nstart)
    n_boost_max = n_boost_1 + int(iter_max)
    s_value = x_current * n_boost_1

    if not buffer_rand:
        # 与上游 R 一致：先把边界向外推，迭代中允许临时越界取样，最终再裁剪回原边界。
        fixed_pushout = float(bounds_buffer) * bounds_diff
        fixed_upper_out = bounds_upper + fixed_pushout
        fixed_lower_out = bounds_lower - fixed_pushout

    iter_min_check = n_boost_1 + int(math.ceil(iter_max / 2))
    last_n = n_boost_1

    for n in range(n_boost_1, n_boost_max + 1):
        last_n = n
        h_step = bounds_diff / (n + 1)
        partial = compute_partial_signs(
            f,
            x_current,
            f_current,
            h_step,
            bounds_lower,
            bounds_upper,
            partial_option,
            use_runmax,
        )

        if buffer_rand:
            # 随机缓冲：每轮对推边界加入随机扰动，提高探索多样性。
            pushout = float(bounds_buffer) * bounds_diff * rng.uniform(
                -1.0,
                1.0,
                size=bounds_diff.size,
            )
            bounds_upper_out = bounds_upper + pushout
            bounds_lower_out = bounds_lower - pushout
        else:
            bounds_upper_out = fixed_upper_out
            bounds_lower_out = fixed_lower_out

        z_value = partial.signs * bounds_upper_out + (1.0 - partial.signs) * bounds_lower_out
        s_value = s_value + z_value
        # 递推平均，与 R 版的 S/(n+1) 更新形式保持一致。
        x_next = s_value / (n + 1)
        f_next = float(f(x_next))

        if use_runmax:
            f_next_best = max(partial.f_partial_best, f_next)
            if f_next_best > f_runmax:
                f_runmax = float(f_next_best)
                x_runmax = np.array(
                    partial.x_partial_best if partial.f_partial_best > f_next else x_next,
                    copy=True,
                )

        f_prev = f_current
        f_current = f_next
        x_current = x_next

        if n >= iter_min_check and abs(f_current - f_prev) < tol_conv:
            break

    result = SingleResult(
        x_optimal=np.array(x_current, copy=True),
        f_optimal=float(f_current),
        iterations=int(last_n - iter_boost),
    )
    if use_runmax:
        result.x_runmax = np.array(x_runmax, copy=True)
        result.f_runmax = float(f_runmax)
    return result


def _clip_result_to_bounds(
    result: SingleResult,
    f: Objective,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
) -> None:
    # 输出阶段统一做边界裁剪，避免返回值落在用户定义边界之外。
    checked = check_bounds(result.x_optimal, bounds_lower, bounds_upper)
    if checked.is_out:
        result.x_optimal = checked.x_in
        result.f_optimal = float(f(checked.x_in))

    if result.x_runmax is not None:
        checked_runmax = check_bounds(result.x_runmax, bounds_lower, bounds_upper)
        if checked_runmax.is_out:
            result.x_runmax = checked_runmax.x_in
            result.f_runmax = float(f(checked_runmax.x_in))


def _promote_runmax(result: SingleResult) -> None:
    # 若 runmax 更优，则提升为最终最优解（与 use_runmax 语义一致）。
    if (
        result.x_runmax is not None
        and result.f_runmax is not None
        and result.f_runmax > result.f_optimal
    ):
        result.x_optimal = np.array(result.x_runmax, copy=True)
        result.f_optimal = float(result.f_runmax)


def _single_refine(
    f: Objective,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_point: np.ndarray,
    *,
    bounds_buffer: float,
    buffer_rand: bool,
    iter_max: int,
    iter_nstart: int,
    iter_boost: int,
    tol_conv: float,
    refine_search: bool,
    refine_ratio: float,
    partial_option: str,
    use_runmax: bool,
    rng: np.random.Generator,
) -> SingleResult:
    # refine 模式分为两段：先常规搜索，再以第一段最优点做零缓冲精修。
    ratio = float(refine_ratio) if refine_search else 0.0
    iter_max_initial, iter_max_refine = _split_refine_iterations(iter_max, ratio, refine_search)

    result = _single(
        f,
        bounds_lower,
        bounds_upper,
        start_point,
        bounds_buffer=bounds_buffer,
        buffer_rand=buffer_rand,
        iter_max=iter_max_initial,
        iter_nstart=iter_nstart,
        iter_boost=iter_boost,
        tol_conv=tol_conv,
        partial_option=partial_option,
        use_runmax=use_runmax,
        rng=rng,
    )
    _clip_result_to_bounds(result, f, bounds_lower, bounds_upper)

    if not refine_search:
        if use_runmax:
            _promote_runmax(result)
        return result

    if (
        use_runmax
        and result.x_runmax is not None
        and result.f_runmax is not None
        and result.f_runmax > result.f_optimal
    ):
        start_point_refine = np.array(result.x_runmax, copy=True)
    else:
        start_point_refine = np.array(result.x_optimal, copy=True)

    refine_result = _single(
        f,
        bounds_lower,
        bounds_upper,
        start_point_refine,
        bounds_buffer=0.0,
        buffer_rand=buffer_rand,
        iter_max=iter_max_refine,
        iter_nstart=iter_nstart,
        iter_boost=iter_boost + 1000,
        tol_conv=tol_conv,
        partial_option=partial_option,
        use_runmax=use_runmax,
        rng=rng,
    )
    _clip_result_to_bounds(refine_result, f, bounds_lower, bounds_upper)

    if use_runmax:
        _promote_runmax(refine_result)
    return refine_result


def _single_boost(
    f: Objective,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    start_point: np.ndarray,
    *,
    bounds_buffer: float,
    buffer_rand: bool,
    iter_max: int,
    iter_nstart: int,
    iter_boost: int,
    tol_conv: float,
    refine_search: bool,
    refine_ratio: float,
    partial_option: str,
    use_runmax: bool,
    rng: np.random.Generator,
) -> SingleResult:
    # boost 模式：比较 regular 与 boosted 两条路径，返回目标值更优者。
    regular = _single_refine(
        f,
        bounds_lower,
        bounds_upper,
        start_point,
        bounds_buffer=bounds_buffer,
        buffer_rand=buffer_rand,
        iter_max=iter_max,
        iter_nstart=iter_nstart,
        iter_boost=0,
        tol_conv=tol_conv,
        refine_search=refine_search,
        refine_ratio=refine_ratio,
        partial_option=partial_option,
        use_runmax=use_runmax,
        rng=rng,
    )
    if iter_boost <= 0:
        return regular

    boosted = _single_refine(
        f,
        bounds_lower,
        bounds_upper,
        start_point,
        bounds_buffer=bounds_buffer,
        buffer_rand=buffer_rand,
        iter_max=iter_max,
        iter_nstart=iter_nstart,
        iter_boost=iter_boost,
        tol_conv=tol_conv,
        refine_search=refine_search,
        refine_ratio=refine_ratio,
        partial_option=partial_option,
        use_runmax=use_runmax,
        rng=rng,
    )
    return boosted if boosted.f_optimal > regular.f_optimal else regular


def smco_multi(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    opt_control: dict[str, Any] | None = None,
) -> SMCOResult:
    lower, upper, starts = validate_smco_inputs(f, bounds_lower, bounds_upper, start_points, opt_control)
    control = _merge_control(opt_control)
    dim = int(lower.size)

    if control.get("n_starts") is None:
        control["n_starts"] = max(5, round(math.sqrt(dim)))
    control["n_starts"] = _as_positive_integer("n_starts", control["n_starts"])
    control["iter_max"] = _as_positive_integer("iter_max", control["iter_max"])
    control["iter_boost"] = _as_non_negative_integer("iter_boost", control["iter_boost"])
    control["bounds_buffer"] = float(control["bounds_buffer"])
    control["tol_conv"] = float(control["tol_conv"])
    control["refine_ratio"] = float(control["refine_ratio"])
    if control["seed"] is not None:
        control["seed"] = _as_integer("seed", control["seed"])

    if starts is None:
        # 未指定起点时使用 Sobol 低差异序列自动生成多起点。
        starts = generate_sobol_points(control["n_starts"], lower, upper, control["seed"])
    else:
        control["n_starts"] = int(starts.shape[0])

    if control.get("iter_nstart") is None:
        control["iter_nstart"] = control["n_starts"]
    control["iter_nstart"] = _as_positive_integer("iter_nstart", control["iter_nstart"])

    rng = np.random.default_rng(control["seed"])
    results: list[SingleResult] = []
    for start in starts:
        results.append(
            _single_boost(
                f,
                lower,
                upper,
                np.asarray(start, dtype=float),
                bounds_buffer=control["bounds_buffer"],
                buffer_rand=bool(control["buffer_rand"]),
                iter_max=control["iter_max"],
                iter_nstart=control["iter_nstart"],
                iter_boost=control["iter_boost"],
                tol_conv=control["tol_conv"],
                refine_search=bool(control["refine_search"]),
                refine_ratio=control["refine_ratio"],
                partial_option=str(control["partial_option"]),
                use_runmax=bool(control["use_runmax"]),
                rng=rng,
            )
        )

    values = np.array([result.f_optimal for result in results], dtype=float)
    best_idx = int(np.argmax(values))
    # summary 提供批次统计，便于 benchmark/论文实验后处理。
    endpoints = np.vstack([result.x_optimal for result in results])
    iterations = np.array([result.iterations for result in results], dtype=float)
    summary = {
        "n_starts": control["n_starts"],
        "mean_iterations": float(np.mean(iterations)),
        "std_values": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "endpoints": endpoints,
        "values": values,
    }

    return SMCOResult(
        best_result=results[best_idx],
        all_results=results,
        opt_control=control,
        summary=summary,
    )


def smco(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    **kwargs: Any,
) -> SMCOResult:
    control = dict(kwargs)
    control["refine_search"] = False
    control["iter_boost"] = 0
    return smco_multi(f, bounds_lower, bounds_upper, start_points, control)


def smco_r(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    **kwargs: Any,
) -> SMCOResult:
    control = dict(kwargs)
    control["refine_search"] = True
    control["iter_boost"] = 0
    control.setdefault("refine_ratio", 0.5)
    return smco_multi(f, bounds_lower, bounds_upper, start_points, control)


def smco_br(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    iter_boost: int = 1000,
    **kwargs: Any,
) -> SMCOResult:
    control = dict(kwargs)
    control["refine_search"] = True
    control["iter_boost"] = iter_boost
    control.setdefault("refine_ratio", 0.5)
    return smco_multi(f, bounds_lower, bounds_upper, start_points, control)
