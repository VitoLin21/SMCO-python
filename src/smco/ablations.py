"""E6 mechanism-ablation configuration generation (Task 10).

Ablations vary ONE control of the E1 winner at a time on DEVELOPMENT instances
(never confirmatory). This module generates the config variants for:

* E6.2 strategy: {rand1bin, best1bin, current-to-best1bin, sobol}
  (rand1bin is the default/winner control; sobol is the key contrast — if DE
  strategies do not beat Sobol replacement, the gain is "elimination + restart",
  not DE).
* E6.3 schedule: evolution_points x elimination_rate grid.

E6.1 start-count ablation ({8, 16, ceil(sqrt(d))}) needs per-n_starts start
artifacts (the instance artifact fixes n_starts=8) and is therefore generated
by a dedicated path that also emits matching start artifacts. E6.4 state-
component resets (reset s_value/current_n vs reset archive vs full restart)
need SMCO SP hooks and are out of scope for this config generator. Both are
flagged in ``ABLATION_DIMENSIONS`` for later.
"""

from __future__ import annotations

from .experiment_manifests import build_algorithm_config
from .paper_contract import parse_algorithm_id

# Dimensions with code support now; "start_count"/"state" are flagged for later.
ABLATION_DIMENSIONS = ("strategy", "schedule", "start_count", "state")

STRATEGY_VARIANTS = ("rand1bin", "best1bin", "current-to-best1bin", "sobol")
SCHEDULE_VARIANTS = (
    ((0.5, 0.75), 0.25),       # default schedule
    ((0.25, 0.5, 0.75), 0.25),
    ((0.5, 0.75), 0.5),
    ((0.25, 0.5, 0.75), 0.5),
)

_EVO_DEFAULTS = dict(de_factor=0.8, de_crossover=0.7, n_starts=8)


def ablation_configs(winner_algorithm_id: str) -> list[tuple[str, str, dict]]:
    """Return ``(dimension, variant_label, config)`` tuples for the E1 winner.

    Empty for a non-EVO winner (ablations are EVO mechanism studies).
    """
    parsed = parse_algorithm_id(winner_algorithm_id)
    if not parsed["evolutionary"]:
        return []
    language, family, semantics = (
        parsed["language"], parsed["family"], parsed["state_semantics"]
    )
    configs: list[tuple[str, str, dict]] = []

    # E6.2 strategy (rand1bin is the default/winner control included for contrast)
    for strategy in STRATEGY_VARIANTS:
        cfg = build_algorithm_config(
            language, family, True, semantics,
            evolution_strategy=strategy, evolution_points=(0.5, 0.75),
            elimination_rate=0.25, **_EVO_DEFAULTS,
        )
        configs.append(("strategy", strategy, cfg))

    # E6.3 schedule grid (evolution_points x elimination_rate)
    for points, rate in SCHEDULE_VARIANTS:
        cfg = build_algorithm_config(
            language, family, True, semantics,
            evolution_strategy="rand1bin", evolution_points=points,
            elimination_rate=rate, **_EVO_DEFAULTS,
        )
        configs.append(("schedule", f"points{points}_rate{rate}", cfg))

    return configs


__all__ = ["ABLATION_DIMENSIONS", "ablation_configs"]
