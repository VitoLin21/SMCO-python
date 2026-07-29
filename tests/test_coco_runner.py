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
    import json
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert prov["winner"] == "PY-SP-SMCO-EVO"
    assert prov["matched_base"] == "PY-BASE-SMCO"


def test_run_baseline_on_problem_smoke():
    from smco.coco_runner import run_baseline_on_problem
    p = _first_problem(5)
    res = run_baseline_on_problem(p, algorithm_name="GenSA", fe_budget=200)
    assert res["dimension"] == 5
    assert res["evaluations"] <= 200
    assert res["evaluations"] > 0
    assert res["algorithm_id"] == "GenSA"
    assert isinstance(res["final_target_hit"], bool)


def test_run_baseline_fe_hard_stop():
    from smco.coco_runner import run_baseline_on_problem
    p = _first_problem(5)
    res = run_baseline_on_problem(p, algorithm_name="DE", fe_budget=30)
    assert res["evaluations"] <= 30


def test_run_baseline_cma_es_smoke():
    # A-10: CMA-ES baseline dispatches + respects the FE hard stop.
    from smco.coco_runner import run_baseline_on_problem
    p = _first_problem(5)
    res = run_baseline_on_problem(p, algorithm_name="CMA-ES", fe_budget=200)
    assert res["algorithm_id"] == "CMA-ES"
    assert 0 < res["evaluations"] <= 200
    assert isinstance(res["final_target_hit"], bool)


def test_run_baseline_rejects_unknown():
    import pytest
    from smco.coco_runner import run_baseline_on_problem
    p = _first_problem(5)
    with pytest.raises(ValueError):
        run_baseline_on_problem(p, algorithm_name="CMAES", fe_budget=50)


def test_bbob_largescale_runner_small_subset(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "largescale_cli", Path("scripts/run_smco_evo_bbob_largescale.py"))
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    # bbob d5 (加速), 1 instance, winner+base+5 baselines = 7 algorithms x 24 func = 168 runs
    summary = cli.run_bbob_largescale(
        winner="PY-SP-SMCO-EVO", suite="bbob", dims=[5], instances=[1],
        fe_budget_per_d=50, result_dir=tmp_path)
    import csv
    rows = list(csv.DictReader(open(tmp_path / "bbob_largescale.csv")))
    assert len(rows) == 24 * 1 * 1 * 7  # 24 func x 1 inst x d5 x 7 algos
    algos = {r["algorithm_id"] for r in rows}
    assert algos == {"PY-SP-SMCO-EVO", "PY-BASE-SMCO", "DE", "GA", "PSO", "SA", "GenSA"}
    assert all(int(r["evaluations"]) <= 50 * 5 for r in rows)
    assert (tmp_path / "bbob_largescale_summary.csv").exists()
    import json
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert prov["winner"] == "PY-SP-SMCO-EVO"
    assert "DE" in prov["algorithms"]


def test_aggregate_instance_summary_collapses_instances():
    from smco.coco_runner import aggregate_instance_summary
    rows = [
        {"function": 1, "dimension": 5, "algorithm_id": "A", "final_target_hit": True, "best_observed_fvalue1": 1.0},
        {"function": 1, "dimension": 5, "algorithm_id": "A", "final_target_hit": False, "best_observed_fvalue1": 2.0},
        {"function": 1, "dimension": 5, "algorithm_id": "B", "final_target_hit": True, "best_observed_fvalue1": 0.5},
    ]
    out, fields = aggregate_instance_summary(rows, ["A", "B"])
    assert len(out) == 1  # one (function, dim), instances aggregated
    row = out[0]
    assert row["A_target_hit_rate"] == 0.5  # 1 of 2 instances hit
    assert row["A_mean_best"] == 1.5  # (1.0 + 2.0) / 2
    assert row["A_n_instances"] == 2
    assert row["B_target_hit_rate"] == 1.0
    assert "A_target_hit_rate" in fields and "A_n_instances" in fields


def test_aggregate_instance_summary_separates_dimensions():
    from smco.coco_runner import aggregate_instance_summary
    rows = [
        {"function": 1, "dimension": 5, "algorithm_id": "A", "final_target_hit": True, "best_observed_fvalue1": 1.0},
        {"function": 1, "dimension": 10, "algorithm_id": "A", "final_target_hit": False, "best_observed_fvalue1": 5.0},
    ]
    out, _ = aggregate_instance_summary(rows, ["A"])
    assert len(out) == 2  # d5 and d10 are distinct rows
