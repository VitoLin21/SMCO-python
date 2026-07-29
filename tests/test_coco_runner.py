"""Tests for the COCO bbob runner (E5 low-dim non-degradation check)."""
from __future__ import annotations

import cocoex
import pytest

from smco.coco_runner import problem_seed, run_on_problem


def _first_problem(dim=5):
    suite = cocoex.Suite("bbob", "instances:1", f"dimensions:{dim}")
    return next(iter(suite))


def test_run_on_problem_base_smoke():
    p = _first_problem(5)
    res = run_on_problem(p, algorithm_id="PY-BASE-SMCO", fe_budget=200)
    assert res["dimension"] == 5
    assert res["evaluations"] <= 200
    assert res["evaluations"] > 0
    assert isinstance(res["final_target_hit"], bool)
    # minimization best must be no worse than a random feasible point
    import numpy as np
    rng = np.random.default_rng(0)
    x_rand = p.lower_bounds + rng.uniform(size=5) * (p.upper_bounds - p.lower_bounds)
    assert res["best_observed_fvalue1"] <= p(x_rand) + 1e-9


def test_run_on_problem_fe_hard_stop():
    p = _first_problem(5)
    res = run_on_problem(p, algorithm_id="PY-BASE-SMCO", fe_budget=30)
    assert res["evaluations"] <= 30


def test_problem_seed_is_stable_and_id_derived():
    p = _first_problem(5)
    assert problem_seed(p) == problem_seed(p)
    p2 = _first_problem(5)  # same id
    assert problem_seed(p) == problem_seed(p2)


def test_run_on_problem_rejects_r_language():
    p = _first_problem(5)
    with pytest.raises(ValueError):
        run_on_problem(p, algorithm_id="R-SP-SMCO-EVO", fe_budget=50)


def test_run_on_problem_evo_smoke():
    p = _first_problem(5)
    res = run_on_problem(p, algorithm_id="PY-SP-SMCO-EVO", fe_budget=200)
    assert res["evaluations"] <= 200
    assert res["algorithm_id"] == "PY-SP-SMCO-EVO"


def test_lowdim_runner_small_subset(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "lowdim_cli", Path("scripts/run_smco_evo_lowdim_check.py"))
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    summary = cli.run_lowdim(
        winner="PY-SP-SMCO-EVO", dims=[5], instances=[1],
        fe_budget_per_d=200, result_dir=tmp_path)
    # 24 bbob functions x 1 instance x d5 x {winner, base} = 48 rows
    import csv
    rows = list(csv.DictReader(open(tmp_path / "lowdim_degradation.csv")))
    assert len(rows) == 24 * 1 * 1 * 2
    assert {r["algorithm_id"] for r in rows} == {"PY-SP-SMCO-EVO", "PY-BASE-SMCO"}
    assert all(int(r["evaluations"]) <= 200 * 5 for r in rows)
    assert (tmp_path / "lowdim_summary.csv").exists()
