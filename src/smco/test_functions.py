from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math
import re

import numpy as np


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    dim: int
    bounds_lower: np.ndarray
    bounds_upper: np.ndarray
    f: Callable[[np.ndarray], float]
    sense: str
    known_best_value: float | None = None
    known_best_x: np.ndarray | None = None
    known_best_objective: float | None = None

    def __post_init__(self) -> None:
        dim = int(self.dim)
        object.__setattr__(self, "dim", dim)

        def readonly_vector(field_name: str, values: np.ndarray) -> np.ndarray:
            array = np.asarray(values, dtype=float)
            if array.shape != (dim,):
                raise ValueError(f"{field_name} must have dimension {dim}")
            copy = np.array(array, dtype=float, copy=True)
            copy.setflags(write=False)
            return copy

        object.__setattr__(self, "bounds_lower", readonly_vector("bounds_lower", self.bounds_lower))
        object.__setattr__(self, "bounds_upper", readonly_vector("bounds_upper", self.bounds_upper))
        if self.known_best_x is not None:
            object.__setattr__(self, "known_best_x", readonly_vector("known_best_x", self.known_best_x))

        raw_objective = self.f

        def objective(x: np.ndarray) -> float:
            array = np.asarray(x, dtype=float)
            if array.shape != (dim,):
                raise ValueError(f"x must have dimension {dim}")
            return float(raw_objective(array))

        object.__setattr__(self, "f", objective)


def squared_norm(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))


def negative_squared_norm(x: np.ndarray) -> float:
    return -squared_norm(x)


def rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(10.0 * x.size + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    sum_sq = np.mean(x**2)
    sum_cos = np.mean(np.cos(2.0 * np.pi * x))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(sum_sq)) - np.exp(sum_cos) + 20.0 + math.e)


def griewank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    indices = np.arange(1, x.size + 1, dtype=float)
    return float(1.0 + np.sum(x**2) / 4000.0 - np.prod(np.cos(x / np.sqrt(indices))))


def rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def dixon_price(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    indices = np.arange(2, x.size + 1, dtype=float)
    return float((x[0] - 1.0) ** 2 + np.sum(indices * (2.0 * x[1:] ** 2 - x[:-1]) ** 2))


def zakharov(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    indices = np.arange(1, x.size + 1, dtype=float)
    linear = np.sum(0.5 * indices * x)
    return float(np.sum(x**2) + linear**2 + linear**4)


def qing(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    indices = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum((x**2 - indices) ** 2))


def michalewicz(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    indices = np.arange(1, x.size + 1, dtype=float)
    return float(-np.sum(np.sin(x) * np.sin(indices * x**2 / np.pi) ** 20))


def dropwave(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    radius = np.sqrt(x1**2 + x2**2)
    return float(-(1.0 + np.cos(12.0 * radius)) / (0.5 * radius**2 + 2.0))


def shubert(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    indices = np.arange(1, 6, dtype=float)
    first = np.sum(indices * np.cos((indices + 1.0) * x1 + indices))
    second = np.sum(indices * np.cos((indices + 1.0) * x2 + indices))
    return float(first * second)


def mccormick(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    return float(np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1.0)


def six_hump_camel(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    return float((4.0 - 2.1 * x1**2 + x1**4 / 3.0) * x1**2 + x1 * x2 + (-4.0 + 4.0 * x2**2) * x2**2)


def eggholder(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    return float(
        -(x2 + 47.0) * np.sin(np.sqrt(abs(x1 / 2.0 + x2 + 47.0)))
        - x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47.0))))
    )


def bukin6(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    return float(100.0 * np.sqrt(abs(x2 - 0.01 * x1**2)) + 0.01 * abs(x1 + 10.0))


def easom(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    return float(-np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi) ** 2 + (x2 - np.pi) ** 2)))


def mishra6(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    term = np.sin((np.cos(x1) + np.cos(x2)) ** 2) ** 2
    term -= np.cos((np.sin(x1) + np.sin(x2)) ** 2) ** 2
    term += x1
    return float(-np.log(term**2) + 0.01 * ((x1 - 1.0) ** 2 + (x2 - 1.0) ** 2))


def _min_config(
    name: str,
    dim: int,
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    raw: Callable[[np.ndarray], float],
    known_min: float | None,
    known_min_x: np.ndarray | None = None,
) -> BenchmarkConfig:
    bounds_lower = np.full(dim, lower, dtype=float) if np.isscalar(lower) else np.asarray(lower, dtype=float)
    bounds_upper = np.full(dim, upper, dtype=float) if np.isscalar(upper) else np.asarray(upper, dtype=float)
    known_best_x = None if known_min_x is None else np.asarray(known_min_x, dtype=float)

    def objective(x: np.ndarray) -> float:
        return -float(raw(x))

    return BenchmarkConfig(
        name=name,
        dim=dim,
        bounds_lower=np.array(bounds_lower, dtype=float, copy=True),
        bounds_upper=np.array(bounds_upper, dtype=float, copy=True),
        f=objective,
        sense="min",
        known_best_value=known_min,
        known_best_x=known_best_x,
        known_best_objective=None if known_min is None else -known_min,
    )


def _key(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _validate_dim(dim: int) -> int:
    if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)):
        raise TypeError("dim must be a positive integer")
    dim = int(dim)
    if dim < 1:
        raise ValueError("dim must be a positive integer")
    return dim


def assign_config(name: str, dim: int) -> BenchmarkConfig:
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    negative_sqnorm_alias = name.strip().lower() in {"-sqnorm", "- sqnorm"}
    key = _key(name)
    dim = _validate_dim(dim)

    if negative_sqnorm_alias or key in {"negativesquarednorm", "minusnorm", "minussqnorm", "minusnormsquared", "norm2xminus"}:
        return BenchmarkConfig(
            name=f"NegativeSquaredNorm{dim}d",
            dim=dim,
            bounds_lower=np.full(dim, -1.0),
            bounds_upper=np.full(dim, 1.0),
            f=negative_squared_norm,
            sense="max",
            known_best_value=0.0,
            known_best_x=np.zeros(dim),
            known_best_objective=0.0,
        )

    if key in {"norm", "norm2x", "sqnorm", "squarednorm", "normsquared"}:
        return BenchmarkConfig(
            name=f"SquaredNorm{dim}d",
            dim=dim,
            bounds_lower=np.full(dim, -1.0),
            bounds_upper=np.full(dim, 1.0),
            f=squared_norm,
            sense="max",
            known_best_value=float(dim),
            known_best_x=np.ones(dim),
            known_best_objective=float(dim),
        )

    scalable = {
        "rastrigin": ("Rastrigin", -5.12, 5.12, rastrigin, 0.0, np.zeros(dim)),
        "ackley": ("Ackley", -32.768, 32.768, ackley, 0.0, np.zeros(dim)),
        "griewank": ("Griewank", -600.0, 600.0, griewank, 0.0, np.zeros(dim)),
        "rosenbrock": ("Rosenbrock", -5.0, 10.0, rosenbrock, 0.0, np.ones(dim)),
        "dixonprice": ("DixonPrice", -10.0, 10.0, dixon_price, 0.0, None),
        "zakharov": ("Zakharov", -5.0, 10.0, zakharov, 0.0, np.zeros(dim)),
        "qing": ("Qing", -500.0, 500.0, qing, 0.0, None),
        "michalewicz": ("Michalewicz", 0.0, np.pi, michalewicz, None, None),
    }
    if key in scalable:
        display, lower, upper, raw, known_min, known_min_x = scalable[key]
        return _min_config(f"{display}{dim}d", dim, lower, upper, raw, known_min, known_min_x)

    two_dimensional = {
        "dropwave": ("DropWave", -5.12, 5.12, dropwave, -1.0, np.zeros(2)),
        "shubert": ("Shubert", -5.12, 5.12, shubert, -186.7309, None),
        "mccormick": (
            "McCormick",
            np.array([-1.5, -3.0]),
            np.array([4.0, 4.0]),
            mccormick,
            -1.913222954981037,
            np.array([-0.5471975602214493, -1.547197559268372]),
        ),
        "sixhumpcamel": ("SixHumpCamel", np.array([-3.0, -2.0]), np.array([3.0, 2.0]), six_hump_camel, -1.0316, None),
        "camel6": ("SixHumpCamel", np.array([-3.0, -2.0]), np.array([3.0, 2.0]), six_hump_camel, -1.0316, None),
        "eggholder": ("Eggholder", -512.0, 512.0, eggholder, -959.6407, np.array([512.0, 404.2319])),
        "bukin6": ("Bukin6", np.array([-15.0, -5.0]), np.array([-3.0, 3.0]), bukin6, 0.0, np.array([-10.0, 1.0])),
        "easom": ("Easom", -100.0, 100.0, easom, -1.0, np.array([np.pi, np.pi])),
        # Upstream-style smoke config only; do not advertise an untrusted optimum.
        "mishra6": ("Mishra6", -10.0, 10.0, mishra6, None, None),
        "mishra06": ("Mishra06", -10.0, 10.0, mishra6, None, None),
    }
    if key in two_dimensional:
        if dim != 2:
            raise ValueError(f"{name} is a 2-dimensional benchmark; dim must be 2")
        display, lower, upper, raw, known_min, known_min_x = two_dimensional[key]
        return _min_config(f"{display}2d", 2, lower, upper, raw, known_min, known_min_x)

    raise ValueError(f"Unknown benchmark: {name}")
