"""Tests for paper-level statistics (Task 12)."""
from __future__ import annotations

import math

import numpy as np

from smco.paper_stats import (
    bootstrap_ci,
    expected_running_time,
    hierarchical_bootstrap_ci,
    holm_correction,
    probability_of_superiority,
)


def test_ert_coco_definition():
    # 2 reached (FE 100, 200) + 1 unreached (budget 1000) -> (100+200+1000)/2
    assert expected_running_time([100, 200, None], 1000) == 650.0
    # no run reached -> inf
    assert math.isinf(expected_running_time([None, None], 1000))
    # all reached
    assert expected_running_time([100, 300], 1000) == 200.0


def test_ert_uses_per_run_budgets():
    # R3b: each unreached run must contribute its OWN fe_budget, not a shared max.
    # reached FE 500 (budget 1_000_000) + unreached (budget 200_000)
    #   per-run ERT = (500 + 200_000) / 1 = 200_500
    #   max-budget bug:   (500 + 1_000_000) / 1 = 1_000_500
    assert expected_running_time([500, None], [1_000_000, 200_000]) == 200_500.0
    # scalar budget still broadcasts (backward compatible)
    assert expected_running_time([500, None], 200_000) == 200_500.0


def test_bootstrap_ci_brackets_point():
    point, lo, hi = bootstrap_ci([1, 2, 3, 4, 5], stat=np.median, n_boot=1000, seed=0)
    assert point == 3.0
    assert lo <= point <= hi
    # empty -> all None
    assert bootstrap_ci([]) == (None, None, None)


def test_hierarchical_bootstrap_ci_brackets_point():
    groups = [[1.0, 2.0, 3.0], [4.0, 5.0], [6.0, 7.0, 8.0, 9.0]]
    point, lo, hi = hierarchical_bootstrap_ci(groups, stat=np.mean, n_boot=500, seed=1)
    assert point is not None
    assert lo <= point <= hi
    # empty -> all None
    assert hierarchical_bootstrap_ci([]) == (None, None, None)


def test_hierarchical_bootstrap_ci_pool_uses_pooled_point():
    # R3b: pool=True reports the statistic on the pooled observations (the value
    # the table actually reports), not the median/mean of per-group statistics.
    # F1=[1,2,3,100] (median 2.5), F2=[4,5,6,7] (median 5.5)
    groups = [[1.0, 2.0, 3.0, 100.0], [4.0, 5.0, 6.0, 7.0]]
    pooled_median = np.median(np.concatenate([np.asarray(g) for g in groups]))  # 4.5
    median_of_medians = np.median([np.median(g) for g in groups])              # 4.0
    assert pooled_median != median_of_medians
    point, lo, hi = hierarchical_bootstrap_ci(groups, stat=np.median, n_boot=500, seed=1, pool=True)
    assert point == pooled_median  # pooled median, not median-of-medians
    assert lo <= point <= hi


def test_holm_correction_step_down():
    # p = [0.01, 0.04, 0.03]; Holm adjusted = [0.03, 0.06, 0.06]
    adj = holm_correction([0.01, 0.04, 0.03])
    assert adj == [0.03, 0.06, 0.06]
    # never exceeds 1.0
    big = holm_correction([0.5, 0.6])
    assert all(p <= 1.0 for p in big)


def test_probability_of_superiority_paired():
    assert probability_of_superiority([1, 2, 3], [2, 3, 4]) == 1.0  # a always < b
    assert probability_of_superiority([2, 2], [2, 2]) == 0.0       # ties don't count
    assert probability_of_superiority([3, 2], [2, 3]) == 0.5
    assert probability_of_superiority([], [1, 2]) is None
