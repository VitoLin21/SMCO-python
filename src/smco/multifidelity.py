from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import math
import time
from typing import Any, Callable

import numpy as np

from .optimizer import generate_sobol_points, smco, smco_br, smco_r


EvaluateFn = Callable[..., Any]


@dataclass(eq=False)
class HyperbandBracket:
    bracket_id: int
    budgets: list[float]
    rung_counts: list[int]


@dataclass(eq=False)
class HyperbandTrial:
    config_id: str
    config: np.ndarray
    budget: float
    score: float
    status: str
    cost: float
    bracket_id: int
    rung_index: int
    promoted: bool = False
    checkpoint: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False)
class SMCOHBResult:
    incumbent_config: np.ndarray
    incumbent_score: float
    incumbent_budget: float
    history: list[HyperbandTrial]
    brackets: list[HyperbandBracket]
    trajectory: list[dict[str, Any]]
    metadata: dict[str, Any]


def _as_float_array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a one-dimensional numeric vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=float, copy=True)


def _normalize_config_space(config_space: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(config_space, dict):
        if "bounds_lower" in config_space and "bounds_upper" in config_space:
            lower = config_space["bounds_lower"]
            upper = config_space["bounds_upper"]
        else:
            raise ValueError("config_space must describe a continuous numeric space")
    elif isinstance(config_space, (tuple, list)) and len(config_space) == 2:
        lower, upper = config_space
    else:
        raise ValueError("config_space must describe a continuous numeric space")

    bounds_lower = _as_float_array("bounds_lower", lower)
    bounds_upper = _as_float_array("bounds_upper", upper)
    if bounds_lower.shape != bounds_upper.shape or np.any(bounds_lower >= bounds_upper):
        raise ValueError("config_space must describe a continuous numeric space")
    return bounds_lower, bounds_upper


def build_hyperband_brackets(min_budget: float, max_budget: float, eta: float) -> list[HyperbandBracket]:
    min_budget = float(min_budget)
    max_budget = float(max_budget)
    eta = float(eta)
    if not math.isfinite(min_budget) or min_budget <= 0:
        raise ValueError("min_budget must be positive")
    if not math.isfinite(max_budget) or max_budget < min_budget:
        raise ValueError("max_budget must be at least min_budget")
    if not math.isfinite(eta) or eta <= 1:
        raise ValueError("eta must be greater than 1")

    s_max = int(math.floor(math.log(max_budget / min_budget, eta)))
    total_budget = (s_max + 1) * max_budget
    brackets: list[HyperbandBracket] = []
    for bracket_id, s in enumerate(range(0, s_max + 1)):
        n = int(math.ceil(total_budget / max_budget / (s + 1) * (eta**s)))
        budgets = [float(max_budget / (eta ** (s - i))) for i in range(s + 1)]
        rung_counts = [max(1, int(math.floor(n / (eta**i)))) for i in range(s + 1)]
        brackets.append(HyperbandBracket(bracket_id=bracket_id, budgets=budgets, rung_counts=rung_counts))
    return brackets


def _config_id(config: np.ndarray) -> str:
    rounded = np.round(np.asarray(config, dtype=float), 12)
    return hashlib.sha1(rounded.tobytes()).hexdigest()[:16]


def _candidate_distance(candidate: np.ndarray, pool: list[np.ndarray], scales: np.ndarray) -> float:
    if not pool:
        return math.inf
    distances = [np.linalg.norm((candidate - other) / scales) for other in pool]
    return float(min(distances))


class _RunHistory:
    def __init__(self) -> None:
        self._trials: list[HyperbandTrial] = []

    @property
    def trials(self) -> list[HyperbandTrial]:
        return self._trials

    def add(self, trial: HyperbandTrial) -> None:
        self._trials.append(trial)

    def latest_checkpoint(self, config_id: str) -> Any:
        checkpoint = None
        max_budget = -math.inf
        for trial in self._trials:
            if trial.config_id == config_id and trial.status == "success" and trial.checkpoint is not None:
                if trial.budget >= max_budget:
                    max_budget = trial.budget
                    checkpoint = trial.checkpoint
        return checkpoint

    def successful_configs(self) -> tuple[np.ndarray, np.ndarray]:
        configs: list[np.ndarray] = []
        scores: list[float] = []
        best_seen: dict[str, tuple[float, float, np.ndarray]] = {}
        for trial in self._trials:
            if trial.status != "success":
                continue
            current = best_seen.get(trial.config_id)
            if current is None or trial.budget > current[0] or (trial.budget == current[0] and trial.score > current[1]):
                best_seen[trial.config_id] = (
                    float(trial.budget),
                    float(trial.score),
                    np.array(trial.config, dtype=float, copy=True),
                )
        for _, score, config in best_seen.values():
            scores.append(score)
            configs.append(config)
        if not configs:
            return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float)
        return np.vstack(configs), np.array(scores, dtype=float)

    def promote(self, bracket_id: int, rung_index: int, next_count: int) -> list[np.ndarray]:
        candidates = [trial for trial in self._trials if trial.bracket_id == bracket_id and trial.rung_index == rung_index]
        successes = [trial for trial in candidates if trial.status == "success" and np.isfinite(trial.score)]
        successes.sort(key=lambda trial: trial.score, reverse=True)
        promoted = successes[: max(0, next_count)]
        for trial in promoted:
            trial.promoted = True
        return [np.array(trial.config, dtype=float, copy=True) for trial in promoted]

    def incumbent(self) -> tuple[np.ndarray, float, float]:
        successes = [trial for trial in self._trials if trial.status == "success"]
        if not successes:
            raise RuntimeError("SMCO-HB produced no successful trials")
        max_budget = max(trial.budget for trial in successes)
        finalists = [trial for trial in successes if trial.budget == max_budget]
        best = max(finalists, key=lambda trial: trial.score)
        return np.array(best.config, dtype=float, copy=True), float(best.score), float(best.budget)

    def trajectory(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        best_score = -math.inf
        best_budget = -math.inf
        for trial in self._trials:
            if trial.status != "success":
                continue
            if trial.budget > best_budget or (trial.budget == best_budget and trial.score > best_score):
                best_score = trial.score
                best_budget = trial.budget
            out.append(
                {
                    "budget": float(trial.budget),
                    "score": float(trial.score),
                    "best_budget": float(best_budget),
                    "best_score": float(best_score),
                }
            )
        return out


class _SMCOProposer:
    def __init__(
        self,
        suggest_variant: str,
        bounds_lower: np.ndarray,
        bounds_upper: np.ndarray,
        proposer_options: dict[str, Any],
        seed: int | None,
    ) -> None:
        suggest_variant = suggest_variant.lower()
        variant_map = {"smco": smco, "smco_r": smco_r, "smco_br": smco_br}
        if suggest_variant not in variant_map:
            raise ValueError("suggest_variant must be one of smco, smco_r, smco_br")
        self._variant = variant_map[suggest_variant]
        self._variant_name = suggest_variant
        self._lower = bounds_lower
        self._upper = bounds_upper
        self._options = dict(proposer_options)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._scales = np.maximum(bounds_upper - bounds_lower, 1e-12)

    def propose(self, n_configs: int, history: _RunHistory) -> list[np.ndarray]:
        n_configs = int(n_configs)
        if n_configs < 1:
            return []
        existing_configs, existing_scores = history.successful_configs()
        existing_pool = [np.array(config, dtype=float, copy=True) for config in existing_configs] if existing_configs.size else []
        chosen: list[np.ndarray] = []
        if existing_configs.size == 0:
            samples = generate_sobol_points(n_configs, self._lower, self._upper, seed=self._next_seed())
            return [np.array(sample, dtype=float, copy=True) for sample in samples]

        while len(chosen) < n_configs:
            candidate = self._optimize_surrogate(existing_configs, existing_scores, chosen)
            pool = existing_pool + chosen
            attempts = 0
            while _candidate_distance(candidate, pool, self._scales) < 1e-3 and attempts < 20:
                candidate = self._random_candidate()
                attempts += 1
            chosen.append(np.array(candidate, dtype=float, copy=True))
        return chosen

    def _random_candidate(self) -> np.ndarray:
        return self._lower + self._rng.uniform(size=self._lower.size) * (self._upper - self._lower)

    def _next_seed(self) -> int:
        return int(self._rng.integers(0, 2**31))

    def _optimize_surrogate(
        self,
        configs: np.ndarray,
        scores: np.ndarray,
        chosen: list[np.ndarray],
    ) -> np.ndarray:
        score_min = float(np.min(scores))
        score_span = float(np.max(scores) - score_min)
        normalized_scores = (scores - score_min) / score_span if score_span > 1e-12 else np.ones_like(scores)
        bandwidth = float(self._options.get("bandwidth", 0.2))
        exploration = float(self._options.get("exploration_weight", 0.15))
        distance_penalty = float(self._options.get("distance_penalty", 2.0))

        def surrogate(x: np.ndarray) -> float:
            delta = (configs - x) / self._scales
            sq_dist = np.sum(delta * delta, axis=1)
            weights = np.exp(-sq_dist / max(1e-12, 2.0 * bandwidth * bandwidth * x.size))
            weighted = float(np.sum(weights * normalized_scores) / max(np.sum(weights), 1e-12))
            nearest_history = float(np.min(np.sqrt(sq_dist)))
            novelty = nearest_history / math.sqrt(x.size)
            penalty = 0.0
            if chosen:
                penalty = max(0.0, 1.0 - _candidate_distance(x, chosen, self._scales)) * distance_penalty
            return weighted + exploration * novelty - penalty

        pool_size = int(self._options.get("pool_size", max(3, int(math.ceil(math.sqrt(self._lower.size))))))
        starts = [self._random_candidate()]
        top_indices = np.argsort(scores)[::-1][: max(1, pool_size - 1)]
        for idx in top_indices:
            jitter = self._rng.normal(scale=0.05, size=self._lower.size) * self._scales
            starts.append(np.clip(configs[idx] + jitter, self._lower, self._upper))
        start_points = np.vstack(starts[:pool_size])
        variant_options = {
            "iter_max": int(self._options.get("iter_max", 40)),
            "n_starts": int(self._options.get("pool_size", pool_size)),
            "seed": self._next_seed(),
        }
        if self._variant_name == "smco_br":
            variant_options["iter_boost"] = int(self._options.get("iter_boost", 20))
        result = self._variant(
            surrogate,
            self._lower,
            self._upper,
            start_points=start_points,
            **variant_options,
        )
        return np.array(result.best_result.x_optimal, dtype=float, copy=True)


def _normalize_evaluation_output(output: Any, elapsed: float) -> tuple[float, str, float, Any, dict[str, Any]]:
    if isinstance(output, dict):
        if "score" not in output:
            raise ValueError("evaluate result dict must contain 'score'")
        score = float(output["score"])
        status = str(output.get("status", "success"))
        cost = float(output.get("cost", elapsed))
        checkpoint = output.get("checkpoint")
        metadata = dict(output.get("metadata", {}))
    else:
        score = float(output)
        status = "success"
        cost = float(elapsed)
        checkpoint = None
        metadata = {}

    if status != "success" or not math.isfinite(score):
        return float("nan"), "failed", cost, checkpoint, metadata
    return score, "success", cost, checkpoint, metadata


def _run_trials(
    evaluate: EvaluateFn,
    configs: list[np.ndarray],
    budget: float,
    bracket_id: int,
    rung_index: int,
    history: _RunHistory,
    seed_rng: np.random.Generator,
    n_workers: int,
) -> list[HyperbandTrial]:
    requests = []
    for config in configs:
        config = np.array(config, dtype=float, copy=True)
        config_id = _config_id(config)
        requests.append(
            {
                "config_id": config_id,
                "config": config,
                "budget": float(budget),
                "checkpoint": history.latest_checkpoint(config_id),
                "seed": int(seed_rng.integers(0, 2**31)),
            }
        )

    def execute(request: dict[str, Any]) -> HyperbandTrial:
        t0 = time.perf_counter()
        try:
            output = evaluate(
                np.array(request["config"], dtype=float, copy=True),
                request["budget"],
                seed=request["seed"],
                checkpoint=request["checkpoint"],
            )
            score, status, cost, checkpoint, metadata = _normalize_evaluation_output(output, time.perf_counter() - t0)
        except Exception as exc:  # pragma: no cover - failure path is covered via status contract
            score = float("nan")
            status = "failed"
            cost = time.perf_counter() - t0
            checkpoint = request["checkpoint"]
            metadata = {"error": repr(exc)}
        return HyperbandTrial(
            config_id=request["config_id"],
            config=request["config"],
            budget=request["budget"],
            score=score,
            status=status,
            cost=cost,
            bracket_id=bracket_id,
            rung_index=rung_index,
            checkpoint=checkpoint,
            metadata=metadata,
        )

    if n_workers <= 1 or len(requests) <= 1:
        return [execute(request) for request in requests]

    ordered: list[HyperbandTrial | None] = [None] * len(requests)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_map = {executor.submit(execute, request): index for index, request in enumerate(requests)}
        for future in as_completed(future_map):
            ordered[future_map[future]] = future.result()
    return [trial for trial in ordered if trial is not None]


def smco_hb(
    *,
    suggest_variant: str,
    config_space: Any,
    evaluate: EvaluateFn,
    min_budget: float,
    max_budget: float,
    eta: float = 3,
    seed: int | None = 123,
    n_workers: int = 1,
    proposer_options: dict[str, Any] | None = None,
    runner_options: dict[str, Any] | None = None,
) -> SMCOHBResult:
    if not callable(evaluate):
        raise TypeError("evaluate must be callable")
    if isinstance(n_workers, bool) or int(n_workers) < 1:
        raise ValueError("n_workers must be a positive integer")

    bounds_lower, bounds_upper = _normalize_config_space(config_space)
    brackets = build_hyperband_brackets(min_budget=min_budget, max_budget=max_budget, eta=eta)
    proposer = _SMCOProposer(
        suggest_variant=suggest_variant,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        proposer_options=proposer_options or {},
        seed=seed,
    )
    history = _RunHistory()
    seed_rng = np.random.default_rng(seed)

    for bracket in brackets:
        configs = proposer.propose(bracket.rung_counts[0], history)
        for rung_index, budget in enumerate(bracket.budgets):
            trials = _run_trials(
                evaluate=evaluate,
                configs=configs,
                budget=budget,
                bracket_id=bracket.bracket_id,
                rung_index=rung_index,
                history=history,
                seed_rng=seed_rng,
                n_workers=int(n_workers),
            )
            for trial in trials:
                history.add(trial)
            if rung_index + 1 >= len(bracket.budgets):
                break
            configs = history.promote(bracket.bracket_id, rung_index, bracket.rung_counts[rung_index + 1])
            if len(configs) < bracket.rung_counts[rung_index + 1]:
                needed = bracket.rung_counts[rung_index + 1] - len(configs)
                configs.extend(proposer.propose(needed, history))

    incumbent_config, incumbent_score, incumbent_budget = history.incumbent()
    metadata = {
        "suggest_variant": suggest_variant.lower(),
        "seed": seed,
        "n_workers": int(n_workers),
        "runner_options": dict(runner_options or {}),
        "proposer_options": dict(proposer_options or {}),
    }
    return SMCOHBResult(
        incumbent_config=incumbent_config,
        incumbent_score=incumbent_score,
        incumbent_budget=incumbent_budget,
        history=history.trials,
        brackets=brackets,
        trajectory=history.trajectory(),
        metadata=metadata,
    )
