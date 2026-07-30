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
