"""Frozen-cohort regression tests for the dedicated E6 Task-12 analysis."""
from __future__ import annotations

import math

import pytest

from smco.e6_analysis import (
    H4_MARGIN,
    START_COUNT_EIGHT,
    START_COUNT_SIXTEEN,
    START_COUNT_SQRT,
    STRATEGY_EXCLUDED_RAND1BIN,
    STRATEGY_PRIMARY,
    start_count_analysis,
    start_count_cohorts,
    strategy_cohorts,
)


FUNCTIONS = ("Rastrigin", "Ackley", "Griewank", "Zakharov")
DIMS = (200, 500, 1000)


def _cells():
    return [(func, dim, inst) for func in FUNCTIONS for dim in DIMS for inst in range(5)]


def _row(func, dim, instance, config_hash, *, strategy="rand1bin", n_starts=8, gap=0.1):
    return {
        "function": func, "dimension": str(dim), "instance": str(instance),
        "configuration_hash": config_hash, "evolution_strategy": strategy,
        "n_starts": str(n_starts), "normalized_gap": str(gap), "fe_used": "100",
        "fe_budget": "1000", "status": "success", "algorithm_id": "PY-SP-SMCO-EVO",
    }


def _strategy_rows():
    rows = []
    for i, (label, config_hash) in enumerate(STRATEGY_PRIMARY.items()):
        rows += [_row(*cell, config_hash, strategy=label, gap=0.1 + i * .01) for cell in _cells()]
    # The real input contains these schedule candidates. They must not alter the
    # primary cohort size or rand1bin values.
    for config_hash in STRATEGY_EXCLUDED_RAND1BIN:
        rows += [_row(*cell, config_hash, gap=99.0) for cell in _cells()]
    return rows


def _start_rows():
    rows = [_row(*cell, START_COUNT_EIGHT, n_starts=8, gap=0.1) for cell in _cells()]
    rows += [_row(*cell, START_COUNT_SIXTEEN, n_starts=16, gap=0.2) for cell in _cells()]
    for func, dim, instance in _cells():
        n, config_hash = START_COUNT_SQRT[dim]
        rows.append(_row(func, dim, instance, config_hash, n_starts=n, gap=0.1))
    return rows


def test_strategy_primary_cohort_is_four_matched_groups_and_excludes_rand_variants():
    cohorts = strategy_cohorts(_strategy_rows())
    assert set(cohorts) == set(STRATEGY_PRIMARY)
    assert all(len(group) == 60 for group in cohorts.values())
    assert {r["configuration_hash"] for r in cohorts["rand1bin"]} == {STRATEGY_PRIMARY["rand1bin"]}
    assert all(r["normalized_gap"] != "99.0" for group in cohorts.values() for r in group)


def test_strategy_rejects_incomplete_problem_matching():
    rows = _strategy_rows()
    rows.pop()
    # pop only affects an excluded cohort; remove a real primary row instead.
    rows = [r for r in rows if not (r["configuration_hash"] == STRATEGY_PRIMARY["sobol"]
                                    and r["instance"] == "4")]
    with pytest.raises(ValueError, match="expected exactly"):
        strategy_cohorts(rows)


def test_start_count_primary_is_8_vs_sqrt_and_n16_is_secondary():
    cohorts = start_count_cohorts(_start_rows())
    assert [len(cohorts[k]) for k in ("n_starts=8", "sqrt(d)", "n_starts=16 (secondary)")] == [60, 60, 60]
    summary, h4 = start_count_analysis(_start_rows(), n_boot=100)
    assert h4["n_pairs"] == 60
    assert h4["secondary_n16_included_in_h4"] is False
    assert {r["analysis_role"] for r in summary} == {"H4_control", "H4_comparator", "secondary_sensitivity"}


def test_h4_noninferiority_uses_frozen_log_110_margin():
    _summary, h4 = start_count_analysis(_start_rows(), n_boot=100)
    assert h4["noninferiority_margin"] == pytest.approx(math.log(1.10))
    assert h4["one_sided_95_upper"] <= H4_MARGIN
    assert h4["noninferior"] is True
