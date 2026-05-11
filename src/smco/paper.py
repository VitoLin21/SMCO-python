from __future__ import annotations

from typing import Any

from .benchmark import BenchmarkRun, run_benchmark


PAPER_SMOKE_CONFIGS = (("Rastrigin", 2), ("Ackley", 2), ("Griewank", 2))


def run_paper_smoke(
    *,
    variant: str = "smco_r",
    iter_max: int = 80,
    n_starts: int = 5,
    seed: int | None = 123,
    **smco_options: Any,
) -> dict[str, BenchmarkRun]:
    options = dict(smco_options)
    options["iter_max"] = iter_max
    options["n_starts"] = n_starts

    runs: dict[str, BenchmarkRun] = {}
    for idx, (name, dim) in enumerate(PAPER_SMOKE_CONFIGS):
        runs[name] = run_benchmark(
            name,
            dim,
            variant=variant,
            repetitions=1,
            seed=None if seed is None else seed + idx,
            **options,
        )
    return runs
