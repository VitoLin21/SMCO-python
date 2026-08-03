"""Frozen adapters for the five prospective E7 comparison algorithms.

This module deliberately keeps the E7 algorithm identifiers separate from the
older E3 ``DE`` identifier.  All adapters receive one observed objective; no
adapter is allowed to unwrap it or evaluate an instance objective directly.
Consequently scipy finite differences, SPSA perturbations, SignGD numerical
gradients, and callbacks made by R packages all consume the same hard FE pool.

The R algorithms use their actual CRAN packages through an optional ``rpy2``
bridge.  Missing runtimes/packages or a version mismatch are unsupported
dependencies, never a signal to substitute a Python optimizer.  Tests may
inject a small backend implementing ``preflight`` and ``run``.
"""

from __future__ import annotations

from copy import deepcopy
from importlib import metadata as importlib_metadata
from typing import Callable, Protocol

import numpy as np

from comparison.methods.lbfgs import lbfgs
from comparison.methods.sign_gd import sign_gradient_descent
from comparison.methods.spsa import spsa


E7_ALGORITHM_IDS = (
    "R-DEoptim",
    "STOGO",
    "L-BFGS",
    "SPSA",
    "SignGD",
)

# This dictionary is also embedded verbatim in every E7 comparator outcome and
# manifest task.  Values are JSON-compatible and intentionally explicit about
# dynamic budget-dependent controls.
E7_ALGORITHM_METADATA: dict[str, dict] = {
    "R-DEoptim": {
        "language": "r",
        "package": "DEoptim",
        "package_version": "2.2-8",
        "hyperparameters": {
            "strategy": 2,
            "NP": "max(n_starts, min(512, max(50, 10*d)))",
            "F": 0.8,
            "CR": 0.5,
            "itermax": "max(1, floor(remaining_fe_budget/NP))",
            "trace": False,
            "storepopfrom": "itermax+1",
            "steptol": 50,
            "reltol": 1e-8,
        },
        "bounds_handling": "DEoptim native box constraints; observed callback clips defensively",
        "rng": "R 4.3.2 Mersenne-Twister; set.seed(seed mod 2147483647)",
        "starts_semantics": "all frozen starts prefix initialpop; seeded uniform fill to NP",
        "fe_counting": "every objective callback, including initial population, uses the shared observer",
    },
    "STOGO": {
        "language": "r",
        "package": "nloptr",
        "package_version": "2.2.1",
        "hyperparameters": {
            "method": "stogo",
            "maxeval": "balanced_split_of_remaining_fe_budget_across_starts",
            "nl.info": False,
        },
        "bounds_handling": "nloptr::stogo native box constraints; observed callback clips defensively",
        "rng": "R 4.3.2 Mersenne-Twister; set.seed(seed mod 2147483647)",
        "starts_semantics": "all frozen starts in stored order, sharing one FE budget",
        "fe_counting": "every objective callback, including package-internal search calls, uses the shared observer",
    },
    "L-BFGS": {
        "language": "python",
        "package": "scipy",
        "package_version": "1.17.1",
        "hyperparameters": {
            "method": "L-BFGS-B",
            "jac": None,
            "max_iter_per_start": 500,
            "ftol": 1e-6,
        },
        "bounds_handling": "scipy L-BFGS-B native box constraints; observed callback clips defensively",
        "rng": "deterministic; seed recorded but algorithm consumes no RNG",
        "starts_semantics": "all frozen SMCO starts in stored order, sharing one FE budget",
        "fe_counting": "every objective callback, including scipy finite-difference gradient calls, uses the shared observer",
    },
    "SPSA": {
        "language": "python",
        "package": "smco",
        "package_version": "0.1.0",
        "hyperparameters": {
            "A": 50.0,
            "a": 0.10,
            "c": 1e-3,
            "alpha": 0.602,
            "gamma": 0.101,
            "max_iter_per_start": 500,
            "tol": 1e-7,
        },
        "bounds_handling": "clip perturbations and iterates to the frozen instance bounds",
        "rng": "numpy.random.Generator(PCG64) via default_rng(seed)",
        "starts_semantics": "all frozen SMCO starts in stored order, sharing one FE budget",
        "fe_counting": "every objective callback, including both SPSA perturbations and iterate checks, uses the shared observer",
    },
    "SignGD": {
        "language": "python",
        "package": "smco",
        "package_version": "0.1.0",
        "hyperparameters": {
            "gradient": "two-sided coordinate finite difference",
            "gradient_eps": 1e-8,
            "learning_rate": 0.1,
            "decay_rate": 0.995,
            "max_iter_per_start": 500,
            "tol": 1e-6,
            "no_improve_limit": 50,
        },
        "bounds_handling": "clip finite-difference probes and iterates to the frozen instance bounds",
        "rng": "deterministic; seed recorded but algorithm consumes no RNG",
        "starts_semantics": "all frozen SMCO starts in stored order, sharing one FE budget",
        "fe_counting": "every objective callback, including all 2*d finite-difference calls, uses the shared observer",
    },
}


class UnsupportedAlgorithmError(RuntimeError):
    """The frozen implementation cannot run in the current environment."""


class RAlgorithmBackend(Protocol):
    def preflight(self, *, algorithm_id: str, metadata: dict) -> None: ...

    def run(
        self,
        *,
        algorithm_id: str,
        objective: Callable[[np.ndarray], float],
        bounds_lower: np.ndarray,
        bounds_upper: np.ndarray,
        start_points: np.ndarray,
        seed: int,
        max_iter: int,
        metadata: dict,
    ) -> None: ...


def algorithm_metadata(algorithm_id: str) -> dict:
    """Return a mutation-safe copy of one frozen metadata record."""
    try:
        return deepcopy(E7_ALGORITHM_METADATA[algorithm_id])
    except KeyError as exc:
        raise ValueError(f"not an E7 comparator: {algorithm_id!r}") from exc


def _installed_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError as exc:
        raise UnsupportedAlgorithmError(
            f"required Python package {distribution!r} is not installed"
        ) from exc


def _preflight_python(algorithm_id: str, metadata: dict) -> None:
    installed = _installed_version(metadata["package"])
    expected = metadata["package_version"]
    if installed != expected:
        raise UnsupportedAlgorithmError(
            f"{algorithm_id} requires {metadata['package']}=={expected}; "
            f"installed={installed}"
        )


class _Rpy2Backend:
    """Thin callback bridge to the frozen native R implementations."""

    _R_VERSION = "4.3.2"
    _RPY2_VERSION = "3.6.4"

    def __init__(self) -> None:
        try:
            import rpy2.robjects as ro
            from rpy2.rinterface import FloatSexpVector, rternalize
            from rpy2.robjects import conversion, default_converter, numpy2ri
            from rpy2.robjects.conversion import localconverter
        except (ImportError, OSError) as exc:
            raise UnsupportedAlgorithmError(
                "R-DEoptim/STOGO require R 4.3.2 and rpy2==3.6.4; no Python "
                "optimizer substitution is permitted"
            ) from exc
        bridge_version = _installed_version("rpy2")
        if bridge_version != self._RPY2_VERSION:
            raise UnsupportedAlgorithmError(
                f"R-DEoptim/STOGO require rpy2=={self._RPY2_VERSION}; "
                f"installed={bridge_version}"
            )
        self.ro = ro
        self.FloatSexpVector = FloatSexpVector
        self.rternalize = rternalize
        self.conversion = conversion
        self.converter = default_converter + numpy2ri.converter
        self.localconverter = localconverter

    def preflight(self, *, algorithm_id: str, metadata: dict) -> None:
        version = str(self.ro.r("paste(R.version$major, R.version$minor, sep='.')")[0])
        if version != self._R_VERSION:
            raise UnsupportedAlgorithmError(
                f"{algorithm_id} requires R {self._R_VERSION}; installed={version}"
            )
        package = metadata["package"]
        available = bool(self.ro.r(
            f"requireNamespace({package!r}, quietly=TRUE)"
        )[0])
        if not available:
            raise UnsupportedAlgorithmError(
                f"{algorithm_id} requires R package {package}=="
                f"{metadata['package_version']}; package is not installed"
            )
        installed = str(self.ro.r(f"as.character(packageVersion({package!r}))")[0])
        expected = metadata["package_version"]
        if installed != expected:
            raise UnsupportedAlgorithmError(
                f"{algorithm_id} requires R package {package}=={expected}; "
                f"installed={installed}"
            )

    def _to_r(self, value):
        with self.localconverter(self.converter):
            return self.conversion.py2rpy(value)

    def run(
        self,
        *,
        algorithm_id: str,
        objective: Callable[[np.ndarray], float],
        bounds_lower: np.ndarray,
        bounds_upper: np.ndarray,
        start_points: np.ndarray,
        seed: int,
        max_iter: int,
        metadata: dict,
    ) -> None:
        @self.rternalize
        def r_objective(x):
            value = objective(np.asarray(x, dtype=float))
            return self.FloatSexpVector([float(value)])

        lower_r = self._to_r(np.asarray(bounds_lower, dtype=float))
        upper_r = self._to_r(np.asarray(bounds_upper, dtype=float))
        starts_r = self._to_r(np.asarray(start_points, dtype=float))
        r_seed = int(seed) % 2147483647
        if r_seed == 0:
            r_seed = 1

        if algorithm_id == "R-DEoptim":
            driver = self.ro.r(
                """
                function(fn, lower, upper, starts, seed, itermax) {
                  set.seed(seed)
                  d <- length(lower)
                  np <- max(nrow(starts), min(512L, max(50L, 10L * d)))
                  n_extra <- np - nrow(starts)
                  if (n_extra > 0L) {
                    u <- matrix(runif(n_extra * d), nrow=n_extra, ncol=d)
                    fill <- sweep(u, 2L, upper - lower, `*`)
                    fill <- sweep(fill, 2L, lower, `+`)
                    initialpop <- rbind(starts, fill)
                  } else initialpop <- starts
                  n_generations <- max(1L, as.integer(floor(itermax / np)))
                  ctrl <- DEoptim::DEoptim.control(
                    strategy=2L, NP=np, F=0.8, CR=0.5,
                    itermax=n_generations, trace=FALSE, initialpop=initialpop,
                    storepopfrom=n_generations + 1L, steptol=50L, reltol=1e-8)
                  invisible(DEoptim::DEoptim(fn=fn, lower=lower, upper=upper,
                                             control=ctrl))
                }
                """
            )
            driver(r_objective, lower_r, upper_r, starts_r, r_seed, int(max_iter))
            return

        if algorithm_id == "STOGO":
            driver = self.ro.r(
                """
                function(fn, lower, upper, starts, seed, maxeval) {
                  set.seed(seed)
                  n_starts <- nrow(starts)
                  base_budget <- as.integer(maxeval %/% n_starts)
                  extra <- as.integer(maxeval %% n_starts)
                  for (i in seq_len(nrow(starts))) {
                    start_budget <- base_budget + ifelse(i <= extra, 1L, 0L)
                    if (start_budget <= 0L) next
                    invisible(nloptr::stogo(
                      x0=starts[i,], fn=fn, lower=lower, upper=upper,
                      maxeval=start_budget, nl.info=FALSE))
                  }
                }
                """
            )
            driver(r_objective, lower_r, upper_r, starts_r, r_seed, int(max_iter))
            return
        raise ValueError(f"unknown R E7 comparator: {algorithm_id!r}")


def default_r_backend() -> RAlgorithmBackend:
    return _Rpy2Backend()


def prepare_e7_adapter(
    algorithm_id: str,
    *,
    r_backend: RAlgorithmBackend | None = None,
) -> tuple[Callable, dict]:
    """Validate frozen dependencies and return a unified optimizer callable."""
    metadata = algorithm_metadata(algorithm_id)
    if metadata["language"] == "r":
        backend = r_backend if r_backend is not None else default_r_backend()
        backend.preflight(algorithm_id=algorithm_id, metadata=metadata)

        def run_r(f, bounds_lower, bounds_upper, *, start_points=None,
                  maximize=False, max_iter=500, seed=None):
            if maximize:
                raise ValueError("E7 R comparators are frozen for minimization")
            if start_points is None:
                raise ValueError(f"{algorithm_id} requires frozen start_points")
            backend.run(
                algorithm_id=algorithm_id,
                objective=f,
                bounds_lower=np.asarray(bounds_lower, dtype=float),
                bounds_upper=np.asarray(bounds_upper, dtype=float),
                start_points=np.asarray(start_points, dtype=float),
                seed=int(seed),
                max_iter=int(max_iter),
                metadata=metadata,
            )

        return run_r, metadata

    _preflight_python(algorithm_id, metadata)
    if algorithm_id == "L-BFGS":
        def run_lbfgs(f, bounds_lower, bounds_upper, *, start_points=None,
                      maximize=False, max_iter=500, seed=None):
            return lbfgs(
                f, bounds_lower, bounds_upper, start_points=start_points,
                maximize=maximize, max_iter=500, tol=1e-6,
            )
        return run_lbfgs, metadata
    if algorithm_id == "SPSA":
        def run_spsa(f, bounds_lower, bounds_upper, *, start_points=None,
                     maximize=False, max_iter=500, seed=None):
            return spsa(
                f, bounds_lower, bounds_upper, start_points=start_points,
                maximize=maximize, A=50.0, a=0.10, c=1e-3,
                alpha=0.602, gamma=0.101, max_iter=500, tol=1e-7,
                seed=int(seed),
            )
        return run_spsa, metadata
    if algorithm_id == "SignGD":
        def run_signgd(f, bounds_lower, bounds_upper, *, start_points=None,
                       maximize=False, max_iter=500, seed=None):
            return sign_gradient_descent(
                f, bounds_lower, bounds_upper, start_points=start_points,
                maximize=maximize, max_iter=500, tol=1e-6,
                learning_rate=0.1, decay_rate=0.995,
            )
        return run_signgd, metadata
    raise ValueError(f"not an E7 comparator: {algorithm_id!r}")


__all__ = [
    "E7_ALGORITHM_IDS",
    "E7_ALGORITHM_METADATA",
    "RAlgorithmBackend",
    "UnsupportedAlgorithmError",
    "algorithm_metadata",
    "default_r_backend",
    "prepare_e7_adapter",
]
