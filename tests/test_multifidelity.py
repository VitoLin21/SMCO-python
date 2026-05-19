import threading
import time

import numpy as np
import pytest

from smco import smco_hb
from smco.multifidelity import build_hyperband_brackets


def test_build_hyperband_brackets_produces_expected_budgets():
    brackets = build_hyperband_brackets(min_budget=1, max_budget=9, eta=3)

    assert [bracket.budgets for bracket in brackets] == [
        [9.0],
        [3.0, 9.0],
        [1.0, 3.0, 9.0],
    ]


def test_smco_hb_rejects_non_numeric_config_space():
    with pytest.raises(ValueError, match="continuous numeric"):
        smco_hb(
            suggest_variant="smco",
            config_space={"choices": ["a", "b"]},
            evaluate=lambda config, budget, **_: {"score": 0.0},
            min_budget=1,
            max_budget=3,
            eta=3,
            seed=1,
        )


def test_smco_hb_promotes_configs_and_reuses_checkpoint_when_available():
    seen = []

    def evaluate(config, budget, seed=None, checkpoint=None):
        seen.append((float(config[0]), float(budget), checkpoint))
        return {
            "score": 1.0 - abs(float(config[0]) - 0.25) + 0.01 * float(budget),
            "checkpoint": f"ckpt@{float(config[0]):.3f}:{float(budget):.1f}",
        }

    result = smco_hb(
        suggest_variant="smco",
        config_space=([-1.0], [1.0]),
        evaluate=evaluate,
        min_budget=1,
        max_budget=3,
        eta=3,
        seed=7,
        n_workers=1,
        proposer_options={"pool_size": 3, "iter_max": 20},
    )

    assert result.incumbent_budget == 3.0
    assert any(budget == 3.0 and checkpoint is not None for _, budget, checkpoint in seen)
    assert any(trial.promoted for trial in result.history)


def test_smco_hb_uses_multiple_workers_within_a_rung():
    active = 0
    max_active = 0
    lock = threading.Lock()

    def evaluate(config, budget, seed=None, checkpoint=None):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {"score": 1.0 - abs(float(config[0]))}

    smco_hb(
        suggest_variant="smco",
        config_space=([-1.0], [1.0]),
        evaluate=evaluate,
        min_budget=1,
        max_budget=3,
        eta=3,
        seed=9,
        n_workers=2,
        proposer_options={"pool_size": 4, "iter_max": 10},
    )

    assert max_active >= 2


def test_smco_hb_finds_reasonable_incumbent_on_budgeted_quadratic():
    def evaluate(config, budget, seed=None, checkpoint=None):
        x = float(config[0])
        return {"score": 1.0 - (x - 0.2) ** 2 - 0.2 / float(budget)}

    result = smco_hb(
        suggest_variant="smco_r",
        config_space=([-1.0], [1.0]),
        evaluate=evaluate,
        min_budget=1,
        max_budget=9,
        eta=3,
        seed=3,
        n_workers=2,
        proposer_options={"pool_size": 9, "iter_max": 25},
    )

    assert result.incumbent_budget == 9.0
    assert abs(float(result.incumbent_config[0]) - 0.2) < 0.35
    assert result.metadata["suggest_variant"] == "smco_r"


def test_smco_hb_history_budgets_are_sorted_per_config():
    def evaluate(config, budget, seed=None, checkpoint=None):
        return {"score": 1.0 - abs(float(config[0])) + 0.01 * float(budget)}

    result = smco_hb(
        suggest_variant="smco_br",
        config_space=([-1.0], [1.0]),
        evaluate=evaluate,
        min_budget=1,
        max_budget=9,
        eta=3,
        seed=5,
        n_workers=1,
        proposer_options={"pool_size": 5, "iter_max": 20},
    )

    budgets_by_config = {}
    for trial in result.history:
        budgets_by_config.setdefault(trial.config_id, []).append(trial.budget)

    for budgets in budgets_by_config.values():
        assert budgets == sorted(budgets)
