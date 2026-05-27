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


@dataclass
class SMCOState:
    x_current: np.ndarray
    f_current: float
    s_value: np.ndarray
    current_n: int  # Next loop index n to execute.
    iter_boost: int
    x_runmax: np.ndarray | None
    f_runmax: float | None
    iterations: int = 0  # SingleResult iteration count for the last completed n.
    initial_n: int = 0  # Fixed n_boost_1 anchor for absolute target boundaries.
    stopped_target_n: int | None = None
    birth_iteration: int = 0  # Global evolutionary boundary where this trajectory was created.

    def ranking_value(self) -> float:
        if self.f_runmax is not None:
            return float(self.f_runmax)
        return float(self.f_current)

    def ranking_point(self) -> np.ndarray:
        if self.x_runmax is not None:
            return np.array(self.x_runmax, dtype=float, copy=True)
        return np.array(self.x_current, dtype=float, copy=True)

    def to_result(self) -> SingleResult:
        return SingleResult(
            x_optimal=np.array(self.x_current, dtype=float, copy=True),
            f_optimal=float(self.f_current),
            iterations=int(self.iterations),
            x_runmax=None if self.x_runmax is None else np.array(self.x_runmax, dtype=float, copy=True),
            f_runmax=None if self.f_runmax is None else float(self.f_runmax),
        )


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


EVOLUTION_STRATEGIES = {"rand1bin", "current-to-best1bin", "best1bin", "sobol"}


def _normalize_evolution_points(value: Any) -> tuple[float, ...]:
    try:
        points = tuple(float(point) for point in value)
    except (TypeError, ValueError):
        raise ValueError("evolution_points must contain numeric values") from None
    if not points:
        raise ValueError("evolution_points must not be empty")
    if any((not math.isfinite(point)) or point <= 0.0 or point >= 1.0 for point in points):
        raise ValueError("evolution_points must contain values between 0 and 1")
    if tuple(sorted(points)) != points or len(set(points)) != len(points):
        raise ValueError("evolution_points must be strictly increasing")
    return points


def _as_evolution_float(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric") from None
    return number


def _validate_evolution_control(
    evolution_points: Any,
    elimination_rate: Any,
    evolution_strategy: str,
    de_factor: Any,
    de_crossover: Any,
) -> tuple[tuple[float, ...], float, str, float, float]:
    points = _normalize_evolution_points(evolution_points)
    rate = _as_evolution_float("elimination_rate", elimination_rate)
    if not math.isfinite(rate) or rate <= 0.0 or rate >= 1.0:
        raise ValueError("elimination_rate must be between 0 and 1")
    strategy = str(evolution_strategy)
    if strategy not in EVOLUTION_STRATEGIES:
        raise ValueError("evolution_strategy must be one of current-to-best1bin, rand1bin, best1bin, sobol")
    factor = _as_evolution_float("de_factor", de_factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("de_factor must be positive")
    crossover = _as_evolution_float("de_crossover", de_crossover)
    if not math.isfinite(crossover) or crossover < 0.0 or crossover > 1.0:
        raise ValueError("de_crossover must be between 0 and 1")
    return points, rate, strategy, factor, crossover


def _sobol_replacements(
    n_new: int,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    return generate_sobol_points(n_new, bounds_lower, bounds_upper, seed=int(rng.integers(0, 2**31)))


def _binomial_crossover(
    base: np.ndarray,
    mutant: np.ndarray,
    de_crossover: float,
    rng: np.random.Generator,
) -> np.ndarray:
    dim = int(base.size)
    changed_coordinates = np.flatnonzero(~np.isclose(mutant, base))
    if changed_coordinates.size:
        forced_j = int(rng.choice(changed_coordinates))
    else:
        forced_j = int(rng.integers(0, dim))
    mask = rng.uniform(size=dim) < de_crossover
    mask[forced_j] = True
    return np.where(mask, mutant, base)


def _choice_excluding(
    n_parents: int,
    excluded_indices: set[int],
    *,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = np.array([idx for idx in range(n_parents) if idx not in excluded_indices], dtype=int)
    return rng.choice(candidates, size=size, replace=False)


def _generate_evolution_points(
    parents: np.ndarray,
    scores: np.ndarray,
    *,
    n_new: int,
    strategy: str,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    de_factor: float,
    de_crossover: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_new = int(n_new)
    bounds_lower, bounds_upper = _validate_bounds(bounds_lower, bounds_upper)
    if n_new < 1:
        return np.empty((0, bounds_lower.size), dtype=float)

    parents = np.asarray(parents, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if parents.ndim != 2 or parents.shape[1] != bounds_lower.size:
        raise ValueError("parents must be a two-dimensional array with one column per bound")
    if scores.ndim != 1 or scores.shape[0] != parents.shape[0]:
        raise ValueError("scores must be a one-dimensional array with one value per parent")
    if not np.all(np.isfinite(parents)) or not np.all(np.isfinite(scores)):
        raise ValueError("parents and scores must contain only finite values")
    if strategy not in EVOLUTION_STRATEGIES:
        raise ValueError("evolution_strategy must be one of current-to-best1bin, rand1bin, best1bin, sobol")
    if (
        strategy == "sobol"
        or (strategy in {"rand1bin", "current-to-best1bin"} and parents.shape[0] < 4)
        or (strategy == "best1bin" and parents.shape[0] < 3)
    ):
        return _sobol_replacements(n_new, bounds_lower, bounds_upper, rng)

    best_index = int(np.argmax(scores))
    best = parents[best_index]
    children: list[np.ndarray] = []
    for _ in range(n_new):
        base_index = int(rng.integers(0, parents.shape[0]))
        base = parents[base_index]
        if strategy == "rand1bin":
            a_idx, b_idx, c_idx = _choice_excluding(
                parents.shape[0],
                {base_index},
                size=3,
                rng=rng,
            )
            mutant = parents[a_idx] + de_factor * (parents[b_idx] - parents[c_idx])
        elif strategy == "current-to-best1bin":
            b_idx, c_idx = _choice_excluding(
                parents.shape[0],
                {base_index, best_index},
                size=2,
                rng=rng,
            )
            mutant = base + de_factor * (best - base) + de_factor * (parents[b_idx] - parents[c_idx])
        elif strategy == "best1bin":
            b_idx, c_idx = _choice_excluding(
                parents.shape[0],
                {best_index},
                size=2,
                rng=rng,
            )
            mutant = best + de_factor * (parents[b_idx] - parents[c_idx])
        child = _binomial_crossover(base, mutant, de_crossover, rng)
        children.append(np.clip(child, bounds_lower, bounds_upper))
    return np.vstack(children)


def _merge_control(user_control: dict[str, Any] | None) -> dict[str, Any]:
    control = dict(DEFAULT_CONTROL)
    if user_control:
        control.update(user_control)
    return control


def _split_refine_iterations(iter_max: int, ratio: float, refine_search: bool) -> tuple[int, int]:
    if not refine_search:
        return int(iter_max), 0
    return int(round(iter_max * (1.0 - ratio))), int(round(iter_max * ratio))


def _evolution_boundaries(iter_max: int, evolution_points: tuple[float, ...]) -> list[int]:
    boundaries = [max(1, int(round(iter_max * point))) for point in evolution_points]
    return sorted(set(boundary for boundary in boundaries if boundary < iter_max))


def _initialize_smco_state(
    f: Objective,
    start_point: np.ndarray,
    *,
    iter_nstart: int,
    iter_boost: int,
    use_runmax: bool,
    birth_iteration: int = 0,
) -> SMCOState:
    x_current = np.array(start_point, dtype=float, copy=True)
    f_current = float(f(x_current))
    n_boost_1 = int(iter_boost) + int(iter_nstart)
    x_runmax = np.array(x_current, copy=True) if use_runmax else None
    f_runmax = float(f_current) if use_runmax else None
    return SMCOState(
        x_current=x_current,
        f_current=f_current,
        s_value=x_current * n_boost_1,
        current_n=n_boost_1,
        iter_boost=int(iter_boost),
        x_runmax=x_runmax,
        f_runmax=f_runmax,
        iterations=0,
        initial_n=n_boost_1,
        birth_iteration=int(birth_iteration),
    )


def _run_smco_state_until(
    state: SMCOState,
    f: Objective,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    bounds_buffer: float,
    buffer_rand: bool,
    iter_target: int,
    tol_conv: float,
    partial_option: str,
    use_runmax: bool,
    rng: np.random.Generator,
) -> None:
    bounds_diff = bounds_upper - bounds_lower
    fixed_pushout = float(bounds_buffer) * bounds_diff
    fixed_upper_out = bounds_upper + fixed_pushout
    fixed_lower_out = bounds_lower - fixed_pushout
    initial_n = int(state.initial_n or state.current_n)
    target_n = initial_n + int(iter_target)
    iter_min_check = initial_n + int(math.ceil(iter_target / 2))
    if state.stopped_target_n is not None and target_n <= state.stopped_target_n:
        return

    while state.current_n <= target_n:
        n = state.current_n
        h_step = bounds_diff / (n + 1)
        partial = compute_partial_signs(
            f,
            state.x_current,
            state.f_current,
            h_step,
            bounds_lower,
            bounds_upper,
            partial_option,
            use_runmax,
        )
        if buffer_rand:
            pushout = float(bounds_buffer) * bounds_diff * rng.uniform(-1.0, 1.0, size=bounds_diff.size)
            bounds_upper_out = bounds_upper + pushout
            bounds_lower_out = bounds_lower - pushout
        else:
            bounds_upper_out = fixed_upper_out
            bounds_lower_out = fixed_lower_out

        z_value = partial.signs * bounds_upper_out + (1.0 - partial.signs) * bounds_lower_out
        state.s_value = state.s_value + z_value
        x_next = state.s_value / (n + 1)
        f_next = float(f(x_next))

        if use_runmax:
            f_next_best = max(partial.f_partial_best, f_next)
            if state.f_runmax is None or f_next_best > state.f_runmax:
                state.f_runmax = float(f_next_best)
                state.x_runmax = np.array(
                    partial.x_partial_best if partial.f_partial_best > f_next else x_next,
                    copy=True,
                )

        f_prev = state.f_current
        state.f_current = f_next
        state.x_current = x_next
        state.current_n = n + 1
        state.iterations = int(n - state.iter_boost)
        if n >= iter_min_check and abs(state.f_current - f_prev) < tol_conv:
            state.stopped_target_n = target_n
            break


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
    state = _initialize_smco_state(
        f,
        start_point,
        iter_nstart=iter_nstart,
        iter_boost=iter_boost,
        use_runmax=use_runmax,
    )
    _run_smco_state_until(
        state,
        f,
        bounds_lower,
        bounds_upper,
        bounds_buffer,
        buffer_rand,
        iter_max,
        tol_conv,
        partial_option,
        use_runmax,
        rng,
    )
    return state.to_result()


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


def _sync_state_from_clipped_result(
    state: SMCOState,
    result: SingleResult,
) -> None:
    state.x_current = np.array(result.x_optimal, dtype=float, copy=True)
    state.f_current = float(result.f_optimal)
    state.s_value = state.x_current * state.current_n
    state.x_runmax = None if result.x_runmax is None else np.array(result.x_runmax, dtype=float, copy=True)
    state.f_runmax = None if result.f_runmax is None else float(result.f_runmax)


def _promote_runmax(result: SingleResult) -> None:
    # 若 runmax 更优，则提升为最终最优解（与 use_runmax 语义一致）。
    if (
        result.x_runmax is not None
        and result.f_runmax is not None
        and result.f_runmax > result.f_optimal
    ):
        result.x_optimal = np.array(result.x_runmax, copy=True)
        result.f_optimal = float(result.f_runmax)


def _normalize_evolutionary_result_iterations(
    result: SingleResult,
    state: SMCOState,
) -> None:
    # evolutionary 公共结果统一按全局 boundary 口径报告 iterations，
    # 使 survivor 与 replacement 在 summary 中可直接比较。
    result.iterations = int(result.iterations + state.birth_iteration)


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


def _prepare_multi_start_inputs(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    opt_control: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
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

    return lower, upper, starts, control


def smco_multi(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    opt_control: dict[str, Any] | None = None,
) -> SMCOResult:
    lower, upper, starts, control = _prepare_multi_start_inputs(
        f,
        bounds_lower,
        bounds_upper,
        start_points,
        opt_control,
    )
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


def _build_multi_result(
    results: list[SingleResult],
    control: dict[str, Any],
    *,
    extra_summary: dict[str, Any] | None = None,
) -> SMCOResult:
    values = np.array([result.f_optimal for result in results], dtype=float)
    best_idx = int(np.argmax(values))
    endpoints = np.vstack([result.x_optimal for result in results])
    iterations = np.array([result.iterations for result in results], dtype=float)
    summary = {
        "n_starts": control["n_starts"],
        "mean_iterations": float(np.mean(iterations)),
        "std_values": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "endpoints": endpoints,
        "values": values,
    }
    if extra_summary:
        summary.update(extra_summary)

    return SMCOResult(
        best_result=results[best_idx],
        all_results=results,
        opt_control=control,
        summary=summary,
    )


def _run_evolutionary_states(
    f: Objective,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    starts: np.ndarray,
    control: dict[str, Any],
    *,
    evolution_points: tuple[float, ...],
    elimination_rate: float,
    evolution_strategy: str,
    de_factor: float,
    de_crossover: float,
    iter_max: int,
    iter_boost: int,
    rng: np.random.Generator,
) -> tuple[list[SingleResult], list[dict[str, Any]]]:
    states = [
        _initialize_smco_state(
            f,
            np.asarray(start, dtype=float),
            iter_nstart=control["iter_nstart"],
            iter_boost=iter_boost,
            use_runmax=bool(control["use_runmax"]),
            birth_iteration=0,
        )
        for start in starts
    ]
    history: list[dict[str, Any]] = []

    for boundary in _evolution_boundaries(iter_max, evolution_points):
        for state in states:
            _run_smco_state_until(
                state,
                f,
                bounds_lower,
                bounds_upper,
                control["bounds_buffer"],
                bool(control["buffer_rand"]),
                boundary - int(state.birth_iteration),
                control["tol_conv"],
                str(control["partial_option"]),
                bool(control["use_runmax"]),
                rng,
            )
            state_result = state.to_result()
            _clip_result_to_bounds(state_result, f, bounds_lower, bounds_upper)
            _sync_state_from_clipped_result(state, state_result)

        ranked = sorted(states, key=lambda state: state.ranking_value(), reverse=True)
        n_eliminate = min(len(ranked) - 1, max(1, int(math.ceil(len(ranked) * elimination_rate))))
        survivors = ranked[: len(ranked) - n_eliminate]
        eliminated = ranked[len(ranked) - n_eliminate :]
        generated = np.empty((0, bounds_lower.size), dtype=float)
        if eliminated:
            parents = np.vstack([state.ranking_point() for state in survivors])
            scores = np.array([state.ranking_value() for state in survivors], dtype=float)
            generated = _generate_evolution_points(
                parents,
                scores,
                n_new=len(eliminated),
                strategy=evolution_strategy,
                bounds_lower=bounds_lower,
                bounds_upper=bounds_upper,
                de_factor=de_factor,
                de_crossover=de_crossover,
                rng=rng,
            )
        replacements = [
            _initialize_smco_state(
                f,
                point,
                iter_nstart=control["iter_nstart"],
                iter_boost=iter_boost + boundary,
                use_runmax=bool(control["use_runmax"]),
                birth_iteration=boundary,
            )
            for point in generated
        ]
        best_before = float(ranked[0].ranking_value())
        states = survivors + replacements
        history.append(
            {
                "iteration": int(boundary),
                "strategy": evolution_strategy,
                "survivor_count": len(survivors),
                "eliminated_count": len(eliminated),
                "generated_count": int(generated.shape[0]),
                "best_before": best_before,
                "best_after_generation": float(max(state.ranking_value() for state in states)),
            }
        )

    results: list[SingleResult] = []
    for state in states:
        _run_smco_state_until(
            state,
            f,
            bounds_lower,
            bounds_upper,
            control["bounds_buffer"],
            bool(control["buffer_rand"]),
            iter_max - int(state.birth_iteration),
            control["tol_conv"],
            str(control["partial_option"]),
            bool(control["use_runmax"]),
            rng,
        )
        result = state.to_result()
        _normalize_evolutionary_result_iterations(result, state)
        _clip_result_to_bounds(result, f, bounds_lower, bounds_upper)
        if bool(control["use_runmax"]):
            _promote_runmax(result)
        results.append(result)

    return results, history


def _refine_evolutionary_results(
    f: Objective,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    results: list[SingleResult],
    control: dict[str, Any],
    *,
    iter_max_refine: int,
    rng: np.random.Generator,
) -> list[SingleResult]:
    refined_results: list[SingleResult] = []
    for result in results:
        if (
            bool(control["use_runmax"])
            and result.x_runmax is not None
            and result.f_runmax is not None
            and result.f_runmax > result.f_optimal
        ):
            start_point_refine = np.array(result.x_runmax, dtype=float, copy=True)
        else:
            start_point_refine = np.array(result.x_optimal, dtype=float, copy=True)

        refine_result = _single(
            f,
            bounds_lower,
            bounds_upper,
            start_point_refine,
            bounds_buffer=0.0,
            buffer_rand=bool(control["buffer_rand"]),
            iter_max=iter_max_refine,
            iter_nstart=control["iter_nstart"],
            iter_boost=control["iter_boost"] + 1000,
            tol_conv=control["tol_conv"],
            partial_option=str(control["partial_option"]),
            use_runmax=bool(control["use_runmax"]),
            rng=rng,
        )
        _clip_result_to_bounds(refine_result, f, bounds_lower, bounds_upper)
        if bool(control["use_runmax"]):
            _promote_runmax(refine_result)
        refined_results.append(refine_result)

    return refined_results


def _run_evolutionary_multi_branch(
    f: Objective,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    starts: np.ndarray,
    control: dict[str, Any],
    *,
    evolution_points: tuple[float, ...],
    elimination_rate: float,
    evolution_strategy: str,
    de_factor: float,
    de_crossover: float,
) -> SMCOResult:
    rng = np.random.default_rng(control["seed"])
    iter_max_initial, iter_max_refine = _split_refine_iterations(
        control["iter_max"],
        control["refine_ratio"],
        bool(control["refine_search"]),
    )
    results, evolution_history = _run_evolutionary_states(
        f,
        bounds_lower,
        bounds_upper,
        starts,
        control,
        evolution_points=evolution_points,
        elimination_rate=elimination_rate,
        evolution_strategy=evolution_strategy,
        de_factor=de_factor,
        de_crossover=de_crossover,
        iter_max=iter_max_initial,
        iter_boost=control["iter_boost"],
        rng=rng,
    )

    if bool(control["refine_search"]) and iter_max_refine > 0:
        results = _refine_evolutionary_results(
            f,
            bounds_lower,
            bounds_upper,
            results,
            control,
            iter_max_refine=iter_max_refine,
            rng=rng,
        )

    return _build_multi_result(
        results,
        control,
        extra_summary={
            "evolution_history": evolution_history,
            "evolution_strategy": evolution_strategy,
            "evolution_points": evolution_points,
            "elimination_rate": elimination_rate,
        },
    )


def smco_evo_multi(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    opt_control: dict[str, Any] | None = None,
    *,
    evolution_points: Any = (0.5, 0.75),
    elimination_rate: float = 0.25,
    evolution_strategy: str = "rand1bin",
    de_factor: float = 0.8,
    de_crossover: float = 0.7,
) -> SMCOResult:
    points, rate, strategy, factor, crossover = _validate_evolution_control(
        evolution_points,
        elimination_rate,
        evolution_strategy,
        de_factor,
        de_crossover,
    )
    lower, upper, starts, control = _prepare_multi_start_inputs(
        f,
        bounds_lower,
        bounds_upper,
        start_points,
        opt_control,
    )
    rng = np.random.default_rng(control["seed"])
    control["evolution_points"] = points
    control["elimination_rate"] = rate
    control["evolution_strategy"] = strategy
    control["de_factor"] = factor
    control["de_crossover"] = crossover

    if control["iter_boost"] <= 0:
        return _run_evolutionary_multi_branch(
            f,
            lower,
            upper,
            starts,
            control,
            evolution_points=points,
            elimination_rate=rate,
            evolution_strategy=strategy,
            de_factor=factor,
            de_crossover=crossover,
        )

    regular_control = dict(control)
    regular_control["iter_boost"] = 0
    regular = _run_evolutionary_multi_branch(
        f,
        lower,
        upper,
        starts,
        regular_control,
        evolution_points=points,
        elimination_rate=rate,
        evolution_strategy=strategy,
        de_factor=factor,
        de_crossover=crossover,
    )
    boosted = _run_evolutionary_multi_branch(
        f,
        lower,
        upper,
        starts,
        control,
        evolution_points=points,
        elimination_rate=rate,
        evolution_strategy=strategy,
        de_factor=factor,
        de_crossover=crossover,
    )
    winner = boosted if boosted.best_result.f_optimal > regular.best_result.f_optimal else regular
    return SMCOResult(
        best_result=winner.best_result,
        all_results=winner.all_results,
        opt_control=dict(control),
        summary=dict(winner.summary),
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


def smco_evo(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    **kwargs: Any,
) -> SMCOResult:
    control = dict(kwargs)
    evolution_points = control.pop("evolution_points", (0.5, 0.75))
    elimination_rate = control.pop("elimination_rate", 0.25)
    evolution_strategy = control.pop("evolution_strategy", "rand1bin")
    de_factor = control.pop("de_factor", 0.8)
    de_crossover = control.pop("de_crossover", 0.7)
    control["refine_search"] = False
    control["iter_boost"] = 0
    return smco_evo_multi(
        f,
        bounds_lower,
        bounds_upper,
        start_points,
        control,
        evolution_points=evolution_points,
        elimination_rate=elimination_rate,
        evolution_strategy=evolution_strategy,
        de_factor=de_factor,
        de_crossover=de_crossover,
    )


def smco_r_evo(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    **kwargs: Any,
) -> SMCOResult:
    control = dict(kwargs)
    evolution_points = control.pop("evolution_points", (0.5, 0.75))
    elimination_rate = control.pop("elimination_rate", 0.25)
    evolution_strategy = control.pop("evolution_strategy", "rand1bin")
    de_factor = control.pop("de_factor", 0.8)
    de_crossover = control.pop("de_crossover", 0.7)
    control["refine_search"] = True
    control["iter_boost"] = 0
    control.setdefault("refine_ratio", 0.5)
    return smco_evo_multi(
        f,
        bounds_lower,
        bounds_upper,
        start_points,
        control,
        evolution_points=evolution_points,
        elimination_rate=elimination_rate,
        evolution_strategy=evolution_strategy,
        de_factor=de_factor,
        de_crossover=de_crossover,
    )


def smco_br_evo(
    f: Objective,
    bounds_lower: Any,
    bounds_upper: Any,
    start_points: Any = None,
    iter_boost: int = 1000,
    **kwargs: Any,
) -> SMCOResult:
    control = dict(kwargs)
    evolution_points = control.pop("evolution_points", (0.5, 0.75))
    elimination_rate = control.pop("elimination_rate", 0.25)
    evolution_strategy = control.pop("evolution_strategy", "rand1bin")
    de_factor = control.pop("de_factor", 0.8)
    de_crossover = control.pop("de_crossover", 0.7)
    control["refine_search"] = True
    control["iter_boost"] = iter_boost
    control.setdefault("refine_ratio", 0.5)
    return smco_evo_multi(
        f,
        bounds_lower,
        bounds_upper,
        start_points,
        control,
        evolution_points=evolution_points,
        elimination_rate=elimination_rate,
        evolution_strategy=evolution_strategy,
        de_factor=de_factor,
        de_crossover=de_crossover,
    )
