#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Prevent BLAS/OpenMP oversubscription when running multiple benchmarks in parallel.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

import comparison.methods  # noqa: F401
from comparison.methods import get_method
from comparison.run_comparison import run_comparison
from smco import smco, smco_br, smco_br_evo, smco_evo, smco_r, smco_r_evo
from smco.test_functions import assign_config


Objective = Callable[[np.ndarray], float]

TABLE12_EVO_ALGOS = ["SMCO_EVO", "SMCO_R_EVO", "SMCO_BR_EVO"]
TABLE3_EVO_ALGOS = ["SMCO_EVO", "SMCO_R_EVO", "SMCO_BR_EVO"]
GROUP_I_LOCAL = ["SMCO", "SMCO_R", "SMCO_BR", "GD", "SignGD", "ADAM", "SPSA", "optimLBFGS", "BOBYQA"]
GROUP_II_GLOBAL = ["SMCO", "SMCO_R", "SMCO_BR", "GenSA", "SA", "DEoptim", "GA", "PSO"]
TABLE1_ALL_ALGOS = GROUP_I_LOCAL + TABLE12_EVO_ALGOS
TABLE2_ALL_ALGOS = GROUP_II_GLOBAL + TABLE12_EVO_ALGOS
TABLE3_ALL_ALGOS = GROUP_I_LOCAL + ["SMCO_EVO", "SMCO_R_EVO", "SMCO_BR_EVO", "GenSA", "SA", "DEoptim", "GA", "PSO"]
GLOBAL_METHODS = {"GenSA", "DEoptim", "GA", "PSO"}
SMCO_VARIANT_MAP = {
    "SMCO": smco,
    "SMCO_R": smco_r,
    "SMCO_BR": smco_br,
    "SMCO_EVO": smco_evo,
    "SMCO_R_EVO": smco_r_evo,
    "SMCO_BR_EVO": smco_br_evo,
}
COMPARISON_SMCO_DEFAULTS: dict[str, Any] = {
    "iter_max": 300,
    "bounds_buffer": 0.05,
    "buffer_rand": True,
    "tol_conv": 1e-8,
}
JTPA_URL = "https://zenodo.org/api/records/18836198/files/jtpa.csv/content"


@dataclass(frozen=True)
class ObjectiveConfig:
    name: str
    dim: int
    bounds_lower: np.ndarray
    bounds_upper: np.ndarray
    raw_objective: Objective


def generate_unif_starts(
    n_starts: int,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    d = bounds_lower.size
    return bounds_lower + rng.uniform(size=(n_starts, d)) * (bounds_upper - bounds_lower)


def _call_with_supported_kwargs(fn, *args, **kwargs):
    signature = inspect.signature(fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        supported = kwargs
    else:
        supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return fn(*args, **supported)


def _smco_variant_options(algo_name: str, smco_options: dict[str, Any], algo_seed: int) -> dict[str, Any]:
    options = dict(COMPARISON_SMCO_DEFAULTS)
    options.update({key: value for key, value in smco_options.items() if key != "n_starts"})
    if algo_name in ("SMCO_R", "SMCO_BR", "SMCO_R_EVO", "SMCO_BR_EVO"):
        options.setdefault("refine_ratio", 0.5)
    if algo_name in ("SMCO_BR", "SMCO_BR_EVO"):
        base_iter_max = int(options["iter_max"])
        options["iter_max"] = int(math.ceil(base_iter_max / 2.0))
        options.setdefault("iter_boost", 99)
    else:
        options.pop("iter_boost", None)
    options["seed"] = algo_seed
    return options


def compute_metrics(values: np.ndarray, best_opt: float) -> dict[str, float]:
    ae = np.abs(values - best_opt)
    return {
        "rMSE": float(np.sqrt(np.mean(ae**2))),
        "AE50": float(np.percentile(ae, 50)),
        "AE95": float(np.percentile(ae, 95)),
        "AE99": float(np.percentile(ae, 99)),
        "fopt_mean": float(np.mean(values)),
        "fopt_sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    }


def run_algo_replications(
    config_factory: Callable[[int], ObjectiveConfig],
    *,
    to_maximize: bool,
    algo_names: list[str],
    n_replications: int,
    smco_options: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_algo = len(algo_names)
    fopt_algo = np.full((n_algo, n_replications), np.nan)
    time_algo = np.full((n_algo, n_replications), np.nan)

    for rep in range(n_replications):
        config = config_factory(rep)
        bounds_lower = config.bounds_lower
        bounds_upper = config.bounds_upper
        raw_f = config.raw_objective
        effective_f = raw_f if to_maximize else (lambda x, _f=raw_f: -_f(x))

        n_starts = int(smco_options.get("n_starts", max(3, int(np.ceil(np.sqrt(config.dim))))))
        start_points = generate_unif_starts(n_starts, bounds_lower, bounds_upper, rng)

        for algo_idx, algo_name in enumerate(algo_names):
            t0 = time.perf_counter()
            algo_seed = int(rng.integers(0, 2**31))
            if algo_name in SMCO_VARIANT_MAP:
                variant_fn = SMCO_VARIANT_MAP[algo_name]
                variant_options = _smco_variant_options(algo_name, smco_options, algo_seed)
                smco_result = variant_fn(
                    effective_f,
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
                    effective_f,
                    bounds_lower,
                    bounds_upper,
                    maximize=True,
                    max_iter=smco_options.get("iter_max", 300),
                    seed=algo_seed,
                )
                fopt = result.f_optimal
            else:
                opt_fn = get_method(algo_name)
                result = _call_with_supported_kwargs(
                    opt_fn,
                    effective_f,
                    bounds_lower,
                    bounds_upper,
                    start_points=start_points,
                    maximize=True,
                    max_iter=smco_options.get("iter_max", 300),
                    seed=algo_seed,
                )
                fopt = result.f_optimal

            time_algo[algo_idx, rep] = time.perf_counter() - t0
            fopt_algo[algo_idx, rep] = fopt if to_maximize else -fopt

    return fopt_algo, time_algo


def deterministic_config_factory(name: str, dim: int) -> Callable[[int], ObjectiveConfig]:
    config = assign_config(name, dim)
    raw_f = (lambda x, _f=config.f: -_f(x)) if config.sense == "min" else config.f
    lower = config.bounds_lower.copy()
    upper = config.bounds_upper.copy()

    def factory(_: int) -> ObjectiveConfig:
        return ObjectiveConfig(name=name, dim=dim, bounds_lower=lower, bounds_upper=upper, raw_objective=raw_f)

    return factory


def maximum_score_config_factory(dim: int, base_seed: int) -> Callable[[int], ObjectiveConfig]:
    lower = np.full(dim, -20.0, dtype=float)
    upper = np.full(dim, 20.0, dtype=float)

    def factory(rep: int) -> ObjectiveConfig:
        rng = np.random.default_rng(base_seed + rep)
        n = 500
        x1 = rng.normal(loc=0.0, scale=np.sqrt(dim), size=n)
        xrest = rng.normal(loc=0.0, scale=1.0, size=(n, dim))
        beta0 = rng.normal(loc=0.0, scale=1.0, size=dim)
        eps = rng.normal(loc=0.0, scale=np.sqrt(dim), size=n)
        y = (x1 + xrest @ beta0 + eps >= 0.0).astype(float)

        def objective(beta: np.ndarray) -> float:
            beta = np.asarray(beta, dtype=float)
            return float(np.mean(y * (x1 + xrest @ beta >= 0.0)))

        return ObjectiveConfig(
            name="MaximumScore",
            dim=dim,
            bounds_lower=lower,
            bounds_upper=upper,
            raw_objective=objective,
        )

    return factory


def ensure_jtpa_dataset(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "jtpa.csv"
    if not target.exists():
        urllib.request.urlretrieve(JTPA_URL, target)
    return target


def load_jtpa_columns(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return {
        "assign": np.array([float(row["Z2"]) for row in rows], dtype=float),
        "prevearn": np.array([float(row["prevearn"]) for row in rows], dtype=float),
        "edu": np.array([float(row["edu"]) for row in rows], dtype=float),
        "earnings": np.array([float(row["earnings"]) for row in rows], dtype=float),
    }


def empirical_welfare_config_factory(dim: int, base_seed: int, cache_dir: Path) -> Callable[[int], ObjectiveConfig]:
    lower = np.full(dim, -20.0, dtype=float)
    upper = np.full(dim, 20.0, dtype=float)
    cols = load_jtpa_columns(ensure_jtpa_dataset(cache_dir))
    assign = cols["assign"]
    prevearn = cols["prevearn"]
    edu = cols["edu"]
    earnings = cols["earnings"]
    pscore = 2.0 / 3.0

    def standardize(column: np.ndarray) -> np.ndarray:
        centered = column - np.mean(column)
        scale = np.std(centered)
        if scale == 0.0:
            return centered
        return centered / scale

    base_x = np.column_stack(
        [
            standardize(prevearn),
            standardize(edu),
            standardize(edu**2),
            standardize(edu**3),
        ]
    )

    def factory(rep: int) -> ObjectiveConfig:
        rng = np.random.default_rng(base_seed + rep)
        if dim < 5:
            raise ValueError("Empirical welfare configuration requires dim >= 5")
        if dim == 5:
            x = base_x
        else:
            extra = rng.normal(loc=0.0, scale=1.0, size=(base_x.shape[0], dim - 5))
            x = np.column_stack([base_x, extra])
        weights = assign / pscore - (1.0 - assign) / (1.0 - pscore)

        def objective(beta: np.ndarray) -> float:
            beta = np.asarray(beta, dtype=float)
            intercept = beta[0]
            coef = beta[1:]
            decision = intercept + x @ coef
            return float(np.mean(weights * earnings * (decision >= 0.0)))

        return ObjectiveConfig(
            name="EmpiricalWelfare",
            dim=dim,
            bounds_lower=lower,
            bounds_upper=upper,
            raw_objective=objective,
        )

    return factory


def normalize_existing_row(row: dict[str, str]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["dim"] = int(float(row["dim"]))
    normalized["maximize"] = str(row["maximize"]).lower() == "true"
    if "best_opt" in row and row["best_opt"] != "":
        normalized["best_opt"] = float(row["best_opt"])
    if "n_replications" in row and row["n_replications"] != "":
        normalized["n_replications"] = int(float(row["n_replications"]))
    return normalized


def read_existing_table_rows(result_dir: Path, label: str) -> list[dict[str, Any]]:
    csv_path = result_dir / f"{label}.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = [normalize_existing_row(row) for row in csv.DictReader(f)]
    return rows


def fill_existing_table_rows(
    rows: list[dict[str, Any]],
    *,
    label: str,
    name: str,
    dim: int,
    maximize: bool,
    n_replications: int,
    best_opt: float,
) -> list[dict[str, Any]]:
    filled: list[dict[str, Any]] = []
    for row in rows:
        current = dict(row)
        current.setdefault("label", label)
        current.setdefault("name", name)
        current.setdefault("dim", dim)
        current.setdefault("maximize", maximize)
        current.setdefault("n_replications", n_replications)
        current.setdefault("best_opt", best_opt)
        filled.append(current)
    return filled


def write_rows_csv_json(rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def build_result_rows(
    *,
    label: str,
    name: str,
    dim: int,
    maximize: bool,
    n_replications: int,
    best_opt: float,
    algo_names: list[str],
    fopt_algo: np.ndarray,
    time_algo: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, algo_name in enumerate(algo_names):
        metrics = compute_metrics(fopt_algo[idx], best_opt)
        rows.append(
            {
                "label": label,
                "name": name,
                "dim": dim,
                "maximize": maximize,
                "n_replications": n_replications,
                "best_opt": best_opt,
                "algo": algo_name,
                "time_avg": float(np.mean(time_algo[idx])),
                **metrics,
            }
        )
    return rows


def run_table12_evo(
    base_result_dir: Path,
    evo_result_dir: Path,
    integrated_dir: Path,
    *,
    label_filters: set[str] | None = None,
    evolution_strategy: str = "rand1bin",
) -> None:
    scenarios = [
        ("table1_Rastrigin_d2_max", "Rastrigin", 2, True, 500, {"iter_max": 500, "n_starts": 1}, 20260527),
        ("table1_Rastrigin_d2_min", "Rastrigin", 2, False, 500, {"iter_max": 500, "n_starts": 1}, 20260528),
        ("table1_Ackley_d2_max", "Ackley", 2, True, 500, {"iter_max": 500, "n_starts": 1}, 20260529),
        ("table1_Ackley_d2_min", "Ackley", 2, False, 500, {"iter_max": 500, "n_starts": 1}, 20260530),
        ("table1_Griewank_d2_max", "Griewank", 2, True, 500, {"iter_max": 500, "n_starts": 1}, 20260531),
        ("table1_Griewank_d2_min", "Griewank", 2, False, 500, {"iter_max": 500, "n_starts": 1}, 20260532),
        ("table1_Michalewicz_d2_max", "Michalewicz", 2, True, 500, {"iter_max": 500, "n_starts": 1}, 20260533),
        ("table1_Michalewicz_d2_min", "Michalewicz", 2, False, 500, {"iter_max": 500, "n_starts": 1}, 20260534),
        ("table2_Rastrigin_d200_max", "Rastrigin", 200, True, 10, {"iter_max": 500, "n_starts": 14}, 20260627),
        ("table2_Rastrigin_d200_min", "Rastrigin", 200, False, 10, {"iter_max": 500, "n_starts": 14}, 20260628),
        ("table2_Ackley_d200_max", "Ackley", 200, True, 10, {"iter_max": 500, "n_starts": 14}, 20260629),
        ("table2_Ackley_d200_min", "Ackley", 200, False, 10, {"iter_max": 500, "n_starts": 14}, 20260630),
        ("table2_Griewank_d200_max", "Griewank", 200, True, 10, {"iter_max": 500, "n_starts": 14}, 20260631),
        ("table2_Griewank_d200_min", "Griewank", 200, False, 10, {"iter_max": 500, "n_starts": 14}, 20260632),
        ("table2_Michalewicz_d200_max", "Michalewicz", 200, True, 10, {"iter_max": 500, "n_starts": 14}, 20260633),
        ("table2_Michalewicz_d200_min", "Michalewicz", 200, False, 10, {"iter_max": 500, "n_starts": 14}, 20260634),
    ]

    for label, name, dim, maximize, n_replications, smco_options, seed in scenarios:
        if label_filters and label not in label_filters:
            continue
        current_smco_options = dict(smco_options)
        current_smco_options["evolution_strategy"] = evolution_strategy
        comp = run_comparison(
            name_config=name,
            dim_config=dim,
            to_maximize=maximize,
            algo_names=TABLE12_EVO_ALGOS,
            n_replications=n_replications,
            smco_options=current_smco_options,
            seed=seed,
        )
        existing_rows = fill_existing_table_rows(
            read_existing_table_rows(base_result_dir, label),
            label=label,
            name=name,
            dim=dim,
            maximize=maximize,
            n_replications=n_replications,
            best_opt=float(comp.best_opt),
        )
        evo_rows = [
            {
                "label": label,
                "name": name,
                "dim": dim,
                "maximize": maximize,
                "n_replications": n_replications,
                "best_opt": float(existing_rows[0]["best_opt"]),
                "algo": algo_name,
                **comp.sum_results[algo_name],
            }
            for algo_name in TABLE12_EVO_ALGOS
        ]
        write_rows_csv_json(
            evo_rows,
            evo_result_dir / f"{label}.csv",
            evo_result_dir / f"{label}.json",
        )
        integrated_rows = existing_rows + evo_rows
        write_rows_csv_json(
            integrated_rows,
            integrated_dir / f"{label}.csv",
            integrated_dir / f"{label}.json",
        )
        print(f"WROTE {label} with {len(evo_rows)} evo rows")


def run_table12_evo_manual(
    base_result_dir: Path,
    evo_result_dir: Path,
    integrated_dir: Path,
    *,
    label_filters: set[str] | None = None,
) -> None:
    """Fallback helper kept for debugging; normal table1/table2 runs use run_comparison()."""
    scenarios = [
        ("table1_Rastrigin_d2_max", "Rastrigin", 2, True, 500, {"iter_max": 500, "n_starts": 1}, 20260527),
        ("table1_Rastrigin_d2_min", "Rastrigin", 2, False, 500, {"iter_max": 500, "n_starts": 1}, 20260528),
        ("table1_Ackley_d2_max", "Ackley", 2, True, 500, {"iter_max": 500, "n_starts": 1}, 20260529),
        ("table1_Ackley_d2_min", "Ackley", 2, False, 500, {"iter_max": 500, "n_starts": 1}, 20260530),
        ("table1_Griewank_d2_max", "Griewank", 2, True, 500, {"iter_max": 500, "n_starts": 1}, 20260531),
        ("table1_Griewank_d2_min", "Griewank", 2, False, 500, {"iter_max": 500, "n_starts": 1}, 20260532),
        ("table1_Michalewicz_d2_max", "Michalewicz", 2, True, 500, {"iter_max": 500, "n_starts": 1}, 20260533),
        ("table1_Michalewicz_d2_min", "Michalewicz", 2, False, 500, {"iter_max": 500, "n_starts": 1}, 20260534),
        ("table2_Rastrigin_d200_max", "Rastrigin", 200, True, 10, {"iter_max": 500, "n_starts": 14}, 20260627),
        ("table2_Rastrigin_d200_min", "Rastrigin", 200, False, 10, {"iter_max": 500, "n_starts": 14}, 20260628),
        ("table2_Ackley_d200_max", "Ackley", 200, True, 10, {"iter_max": 500, "n_starts": 14}, 20260629),
        ("table2_Ackley_d200_min", "Ackley", 200, False, 10, {"iter_max": 500, "n_starts": 14}, 20260630),
        ("table2_Griewank_d200_max", "Griewank", 200, True, 10, {"iter_max": 500, "n_starts": 14}, 20260631),
        ("table2_Griewank_d200_min", "Griewank", 200, False, 10, {"iter_max": 500, "n_starts": 14}, 20260632),
        ("table2_Michalewicz_d200_max", "Michalewicz", 200, True, 10, {"iter_max": 500, "n_starts": 14}, 20260633),
        ("table2_Michalewicz_d200_min", "Michalewicz", 200, False, 10, {"iter_max": 500, "n_starts": 14}, 20260634),
    ]

    for label, name, dim, maximize, n_replications, smco_options, seed in scenarios:
        if label_filters and label not in label_filters:
            continue
        existing_rows = read_existing_table_rows(base_result_dir, label)
        best_opt = float(existing_rows[0]["best_opt"])
        factory = deterministic_config_factory(name, dim)
        fopt_algo, time_algo = run_algo_replications(
            factory,
            to_maximize=maximize,
            algo_names=TABLE12_EVO_ALGOS,
            n_replications=n_replications,
            smco_options=smco_options,
            seed=seed,
        )
        evo_rows = build_result_rows(
            label=label,
            name=name,
            dim=dim,
            maximize=maximize,
            n_replications=n_replications,
            best_opt=best_opt,
            algo_names=TABLE12_EVO_ALGOS,
            fopt_algo=fopt_algo,
            time_algo=time_algo,
        )
        write_rows_csv_json(
            evo_rows,
            evo_result_dir / f"{label}.csv",
            evo_result_dir / f"{label}.json",
        )
        integrated_rows = existing_rows + evo_rows
        write_rows_csv_json(
            integrated_rows,
            integrated_dir / f"{label}.csv",
            integrated_dir / f"{label}.json",
        )
        print(f"WROTE {label} with {len(evo_rows)} evo rows")


def run_table12_full(output_dir: Path, *, label_filters: set[str] | None = None) -> None:
    scenarios = [
        ("table1_Rastrigin_d2_max", "Rastrigin", 2, True, 500, TABLE1_ALL_ALGOS, {"iter_max": 500, "n_starts": 1}, 20260527),
        ("table1_Rastrigin_d2_min", "Rastrigin", 2, False, 500, TABLE1_ALL_ALGOS, {"iter_max": 500, "n_starts": 1}, 20260528),
        ("table1_Ackley_d2_max", "Ackley", 2, True, 500, TABLE1_ALL_ALGOS, {"iter_max": 500, "n_starts": 1}, 20260529),
        ("table1_Ackley_d2_min", "Ackley", 2, False, 500, TABLE1_ALL_ALGOS, {"iter_max": 500, "n_starts": 1}, 20260530),
        ("table1_Griewank_d2_max", "Griewank", 2, True, 500, TABLE1_ALL_ALGOS, {"iter_max": 500, "n_starts": 1}, 20260531),
        ("table1_Griewank_d2_min", "Griewank", 2, False, 500, TABLE1_ALL_ALGOS, {"iter_max": 500, "n_starts": 1}, 20260532),
        ("table1_Michalewicz_d2_max", "Michalewicz", 2, True, 500, TABLE1_ALL_ALGOS, {"iter_max": 500, "n_starts": 1}, 20260533),
        ("table1_Michalewicz_d2_min", "Michalewicz", 2, False, 500, TABLE1_ALL_ALGOS, {"iter_max": 500, "n_starts": 1}, 20260534),
        ("table2_Rastrigin_d200_max", "Rastrigin", 200, True, 10, TABLE2_ALL_ALGOS, {"iter_max": 500, "n_starts": 14}, 20260627),
        ("table2_Rastrigin_d200_min", "Rastrigin", 200, False, 10, TABLE2_ALL_ALGOS, {"iter_max": 500, "n_starts": 14}, 20260628),
        ("table2_Ackley_d200_max", "Ackley", 200, True, 10, TABLE2_ALL_ALGOS, {"iter_max": 500, "n_starts": 14}, 20260629),
        ("table2_Ackley_d200_min", "Ackley", 200, False, 10, TABLE2_ALL_ALGOS, {"iter_max": 500, "n_starts": 14}, 20260630),
        ("table2_Griewank_d200_max", "Griewank", 200, True, 10, TABLE2_ALL_ALGOS, {"iter_max": 500, "n_starts": 14}, 20260631),
        ("table2_Griewank_d200_min", "Griewank", 200, False, 10, TABLE2_ALL_ALGOS, {"iter_max": 500, "n_starts": 14}, 20260632),
        ("table2_Michalewicz_d200_max", "Michalewicz", 200, True, 10, TABLE2_ALL_ALGOS, {"iter_max": 500, "n_starts": 14}, 20260633),
        ("table2_Michalewicz_d200_min", "Michalewicz", 200, False, 10, TABLE2_ALL_ALGOS, {"iter_max": 500, "n_starts": 14}, 20260634),
    ]

    for label, name, dim, maximize, n_replications, algo_names, smco_options, seed in scenarios:
        if label_filters and label not in label_filters:
            continue
        comp = run_comparison(
            name_config=name,
            dim_config=dim,
            to_maximize=maximize,
            algo_names=algo_names,
            n_replications=n_replications,
            smco_options=smco_options,
            seed=seed,
        )
        rows = []
        for algo_name in algo_names:
            rows.append(
                {
                    "label": label,
                    "name": name,
                    "dim": dim,
                    "maximize": maximize,
                    "n_replications": n_replications,
                    "best_opt": float(comp.best_opt),
                    "algo": algo_name,
                    **comp.sum_results[algo_name],
                }
            )
        write_rows_csv_json(rows, output_dir / f"{label}.csv", output_dir / f"{label}.json")
        print(f"WROTE full rerun {label} with {len(rows)} rows")


def run_table3(
    table3_dir: Path,
    cache_dir: Path,
    *,
    label_filters: set[str] | None = None,
    evolution_strategy: str = "rand1bin",
    algo_names: list[str] | None = None,
) -> None:
    algo_names = list(algo_names or TABLE3_ALL_ALGOS)
    dims = [5, 10, 20, 50]
    random_functions = [
        ("table3_MaximumScore", "MaximumScore", maximum_score_config_factory),
        ("table3_EmpiricalWelfare", "EmpiricalWelfare", lambda dim, seed: empirical_welfare_config_factory(dim, seed, cache_dir)),
    ]

    for prefix, name, factory_builder in random_functions:
        for dim in dims:
            n_replications = 500 // dim
            label = f"{prefix}_d{dim}"
            if label_filters and label not in label_filters:
                continue
            smco_options = {
                "iter_max": 300,
                "n_starts": int(round(math.sqrt(dim))),
                "evolution_strategy": evolution_strategy,
            }
            seed = 20270000 + dim * 100 + (0 if name == "MaximumScore" else 1)
            factory = factory_builder(dim, seed)
            fopt_algo, time_algo = run_algo_replications(
                factory,
                to_maximize=True,
                algo_names=algo_names,
                n_replications=n_replications,
                smco_options=smco_options,
                seed=seed + 5000,
            )
            best_opt = float(np.max(fopt_algo))
            rows = build_result_rows(
                label=label,
                name=name,
                dim=dim,
                maximize=True,
                n_replications=n_replications,
                best_opt=best_opt,
                algo_names=algo_names,
                fopt_algo=fopt_algo,
                time_algo=time_algo,
            )
            write_rows_csv_json(rows, table3_dir / f"{label}.csv", table3_dir / f"{label}.json")
            print(f"WROTE {label}")


def render_report(base_result_dir: Path, evo_result_dir: Path, integrated_dir: Path, table3_dir: Path, report_path: Path) -> None:
    lines = [
        "# Paper Table Results With SMCO-EVO",
        "",
        "## Inputs",
        "",
        f"- Existing table1/table2 results reused from `{base_result_dir}`.",
        f"- SMCO-EVO supplement results written to `{evo_result_dir}`.",
        f"- Integrated table1/table2 rows written to `{integrated_dir}`.",
        f"- Table3 reruns written to `{table3_dir}`.",
        "",
    ]

    def summarize_dir(title: str, directory: Path) -> None:
        csv_files = sorted(directory.glob("*.csv"))
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- Files: {len(csv_files)}")
        for path in csv_files[:20]:
            lines.append(f"- `{path.name}`")
        lines.append("")

    summarize_dir("Table1/Table2 Evo Rows", evo_result_dir)
    summarize_dir("Integrated Table1/Table2", integrated_dir)
    summarize_dir("Table3", table3_dir)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SMCO-EVO paper-table experiments and integration")
    parser.add_argument("--table12-evo", action="store_true", help="Run SMCO-EVO variants on table1/table2 settings")
    parser.add_argument("--table12-full", action="store_true", help="Rerun full table1/table2 with baseline + comparators + SMCO-EVO")
    parser.add_argument("--table3", action="store_true", help="Run paper table3 random-function experiments")
    parser.add_argument("--table3-evo-only", action="store_true", help="Run table3 with only SMCO evo family")
    parser.add_argument("--report", action="store_true", help="Render an inventory report after experiments")
    parser.add_argument("--base-result-dir", default="result/comparison", help="Existing table1/table2 result directory")
    parser.add_argument("--evo-result-dir", default="result/comparison_evo", help="Output directory for evo-only rows")
    parser.add_argument(
        "--integrated-dir",
        default="result/paper_tables_integrated_2026-05-27",
        help="Output directory for integrated table1/table2 rows",
    )
    parser.add_argument("--table3-dir", default="result/table3_2026-05-27", help="Output directory for table3 rows")
    parser.add_argument("--cache-dir", default="result/table3_cache", help="Cache directory for downloaded data")
    parser.add_argument(
        "--evolution-strategy",
        default="rand1bin",
        help="Evolution strategy used by SMCO_EVO variants: rand1bin/current-to-best1bin/best1bin/sobol",
    )
    parser.add_argument(
        "--only-labels",
        nargs="*",
        default=None,
        help="Optional exact labels to run, e.g. table2_Ackley_d200_max table3_MaximumScore_d10",
    )
    args = parser.parse_args()

    base_result_dir = Path(args.base_result_dir)
    evo_result_dir = Path(args.evo_result_dir)
    integrated_dir = Path(args.integrated_dir)
    table3_dir = Path(args.table3_dir)
    cache_dir = Path(args.cache_dir)

    label_filters = set(args.only_labels) if args.only_labels else None

    if args.table12_evo:
        run_table12_evo(
            base_result_dir,
            evo_result_dir,
            integrated_dir,
            label_filters=label_filters,
            evolution_strategy=args.evolution_strategy,
        )
    if args.table12_full:
        run_table12_full(integrated_dir, label_filters=label_filters)
    if args.table3:
        run_table3(
            table3_dir,
            cache_dir,
            label_filters=label_filters,
            evolution_strategy=args.evolution_strategy,
            algo_names=TABLE3_EVO_ALGOS if args.table3_evo_only else TABLE3_ALL_ALGOS,
        )
    if args.report:
        render_report(
            base_result_dir,
            evo_result_dir,
            integrated_dir,
            table3_dir,
            Path("result/paper_tables_integrated_2026-05-27.md"),
        )


if __name__ == "__main__":
    main()
