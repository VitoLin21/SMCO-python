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
    assert prov["original_language"] == "python"
    assert prov["external_check_kind"] == "frozen_winner"
    assert prov["is_frozen_winner_validation"] is True


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


def test_lowdim_r_winner_records_language_note(tmp_path):
    # A-04: an R winner is not silently swapped; provenance records the original
    # language and a note that the Py equivalent was evaluated (R cocoex unavailable).
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "lowdim_cli_r", Path("scripts/run_smco_evo_lowdim_check.py"))
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    cli.run_lowdim(winner="R-SP-SMCO-EVO", dims=[5], instances=[1],
                   fe_budget_per_d=50, result_dir=tmp_path)
    import json
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert prov["original_language"] == "r"
    assert prov["original_winner"] == "R-SP-SMCO-EVO"
    assert prov["winner"] == "PY-SP-SMCO-EVO"  # Py equivalent actually run
    assert prov["external_check_kind"] == "python_port_external"
    assert prov["is_frozen_winner_validation"] is False
    assert "Python port external check" in prov["language_note"]
    assert "NOT the frozen" in prov["language_note"]


def test_lowdim_resolver_canonical_reads_winner(tmp_path):
    # R-04/R7c: canonical E5 reads the winner from a frozen FULL contract manifest
    # (+ selection). (Resolver-level: the cocoex execution smoke is covered by
    # test_lowdim_runner_small_subset; running the full 480-task matrix here would
    # be too slow for a unit test.)
    import argparse
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "lowdim_cli_m", Path("scripts/run_smco_evo_lowdim_check.py"))
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    _e5_full_manifest(tmp_path)
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(
        manifest=str(tmp_path / "manifest.json"),
        selection=str(tmp_path / "selection.json"),
        winner=None, development=False,
        dims=[5], instances=[1], fe_budget_per_d=2000)
    resolved = cli._resolve_winner(args, parser)
    assert resolved["winner"] == "PY-SP-SMCO-EVO"
    assert resolved["dims"] == [5, 20]


def test_lowdim_main_free_winner_requires_development(tmp_path):
    # R-04: free --winner is rejected unless --development acknowledges it.
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "lowdim_cli_d", Path("scripts/run_smco_evo_lowdim_check.py"))
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    with pytest.raises(SystemExit):
        cli.main(["--winner", "PY-SP-SMCO-EVO", "--dims", "5", "--instances", "1",
                  "--fe-budget-per-d", "50", "--result-dir", str(tmp_path / "out")])


# --- R2b: canonical E4/E5 must lock stage + suite + matrix from the manifest ---

def _load_runner(module_name, script):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(module_name, Path(script))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def _e4_manifest(tmp_path, *, stage="e4_bbob_largescale", suite="bbob-largescale",
                 dims=(160, 320), n_instances=2, baselines=("DE", "GA")):
    import json as _json
    from smco.confirmatory import build_confirmatory_manifest
    sel = {"winner": "PY-SP-SMCO-EVO", "winner_language": "python",
           "selection_hash": "s1", "winner_config_hash": "cfg"}
    manifest = build_confirmatory_manifest(
        sel, stage=stage, suite=suite, functions=["Rastrigin"], dims=list(dims),
        n_instances=n_instances, fe_budget_per_d=1000, checkpoints_per_d=(1000,),
        baselines=baselines)
    sel["winner_config_hash"] = manifest["winner_config_hash"]
    (tmp_path / "manifest.json").write_text(_json.dumps(manifest))
    (tmp_path / "selection.json").write_text(_json.dumps(sel))


def _e4_full_manifest(tmp_path, *, baselines=("DE", "GA", "PSO", "SA", "GenSA"),
                      functions=None, dims=(160, 320, 640), n_instances=5):
    """A full E4 contract manifest (7 algos x 24 funcs x 3 dims x 5 instances)."""
    import json as _json
    from smco.confirmatory import build_confirmatory_manifest
    if functions is None:
        functions = [f"f{i}" for i in range(1, 25)]
    sel = {"winner": "PY-SP-SMCO-EVO", "winner_language": "python",
           "selection_hash": "s1", "winner_config_hash": "cfg"}
    manifest = build_confirmatory_manifest(
        sel, stage="e4_bbob_largescale", suite="bbob-largescale",
        functions=list(functions), dims=list(dims), n_instances=n_instances,
        fe_budget_per_d=1000, checkpoints_per_d=(1000,), baselines=baselines)
    sel["winner_config_hash"] = manifest["winner_config_hash"]
    (tmp_path / "manifest.json").write_text(_json.dumps(manifest))
    (tmp_path / "selection.json").write_text(_json.dumps(sel))


def test_e4_resolver_requires_selection_with_manifest(tmp_path):
    # R2b: --manifest without --selection must not be confirmatory.
    import argparse
    cli = _load_runner("e4_r2b_a", "scripts/run_smco_evo_bbob_largescale.py")
    _e4_manifest(tmp_path)
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(manifest=str(tmp_path / "manifest.json"), selection=None,
                              winner=None, baselines=["DE"], development=False)
    with pytest.raises(SystemExit):
        cli._resolve_winner_baselines(args, parser)


def test_e4_resolver_reads_matrix_from_manifest_ignores_cli(tmp_path):
    # R2b/R6c: dims/instances/budget/baselines come ONLY from the manifest; the
    # manifest must be the full E4 contract; CLI overrides are ignored.
    import argparse
    cli = _load_runner("e4_r2b_b", "scripts/run_smco_evo_bbob_largescale.py")
    _e4_full_manifest(tmp_path)
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(
        manifest=str(tmp_path / "manifest.json"),
        selection=str(tmp_path / "selection.json"),
        winner=None, baselines=["SPURIOUS"], development=False,
        suite="bbob-largescale", dims=[999], instances=[999], fe_budget_per_d=9999)
    resolved = cli._resolve_winner_baselines(args, parser)
    assert resolved["winner"] == "PY-SP-SMCO-EVO"
    assert resolved["dims"] == [160, 320, 640]      # manifest, not CLI 999
    assert resolved["instances"] == [1, 2, 3, 4, 5]  # 5 instances -> COCO ids 1..5
    assert resolved["fe_budget_per_d"] == 1000       # manifest, not CLI 9999
    assert resolved["baselines"] == ["DE", "GA", "PSO", "SA", "GenSA"]  # manifest, not CLI
    assert resolved["suite"] == "bbob-largescale"


def test_e4_resolver_rejects_wrong_stage(tmp_path):
    # R2b: a frozen E2 manifest must not drive the E4 runner.
    import argparse
    cli = _load_runner("e4_r2b_c", "scripts/run_smco_evo_bbob_largescale.py")
    _e4_manifest(tmp_path, stage="e2_factorial_highdim", suite="synthetic_highdim",
                 dims=(200,), n_instances=1, baselines=("DE",))
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(
        manifest=str(tmp_path / "manifest.json"),
        selection=str(tmp_path / "selection.json"),
        winner=None, baselines=["DE"], development=False,
        suite="bbob-largescale", dims=[160], instances=[1], fe_budget_per_d=1000)
    with pytest.raises(ValueError, match="stage"):
        cli._resolve_winner_baselines(args, parser)


def test_e4_resolver_rejects_partial_baselines(tmp_path):
    # R6c reviewer repro: a manifest with only DE (not the plan's 5 baselines)
    # must not run as canonical E4.
    import argparse
    cli = _load_runner("e4_r6c_b", "scripts/run_smco_evo_bbob_largescale.py")
    _e4_full_manifest(tmp_path, baselines=("DE",))
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(
        manifest=str(tmp_path / "manifest.json"),
        selection=str(tmp_path / "selection.json"),
        winner=None, baselines=["DE"], development=False,
        suite="bbob-largescale", dims=[160], instances=[1], fe_budget_per_d=1000)
    with pytest.raises(ValueError, match="baseline"):
        cli._resolve_winner_baselines(args, parser)


def test_e4_resolver_rejects_partial_function_matrix(tmp_path):
    # R6c: 7 algos + correct dims/instances but only 1 function -> not the 24-fn
    # E4 matrix -> reject.
    import argparse
    cli = _load_runner("e4_r6c_c", "scripts/run_smco_evo_bbob_largescale.py")
    _e4_full_manifest(tmp_path, functions=["f1"])
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(
        manifest=str(tmp_path / "manifest.json"),
        selection=str(tmp_path / "selection.json"),
        winner=None, baselines=["DE"], development=False,
        suite="bbob-largescale", dims=[160], instances=[1], fe_budget_per_d=1000)
    with pytest.raises(ValueError, match="functions"):
        cli._resolve_winner_baselines(args, parser)


def test_e5_resolver_requires_selection_with_manifest(tmp_path):
    import argparse
    cli = _load_runner("e5_r2b_a", "scripts/run_smco_evo_lowdim_check.py")
    _e4_manifest(tmp_path)  # reuse: builds a manifest + selection
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(manifest=str(tmp_path / "manifest.json"), selection=None,
                              winner=None, development=False)
    with pytest.raises(SystemExit):
        cli._resolve_winner(args, parser)


def _e5_full_manifest(tmp_path, *, functions=None, dims=(5, 20), n_instances=5,
                      fe_budget_per_d=2000):
    """A full E5 contract manifest (winner+base x 24 funcs x {5,20} x 5 instances)."""
    import json as _json
    from smco.confirmatory import build_confirmatory_manifest
    if functions is None:
        functions = [f"f{i}" for i in range(1, 25)]
    sel = {"winner": "PY-SP-SMCO-EVO", "winner_language": "python",
           "selection_hash": "s1", "winner_config_hash": "cfg"}
    manifest = build_confirmatory_manifest(
        sel, stage="e5_lowdim_check", suite="bbob", functions=list(functions),
        dims=list(dims), n_instances=n_instances,
        fe_budget_per_d=fe_budget_per_d, checkpoints_per_d=(fe_budget_per_d,))
    sel["winner_config_hash"] = manifest["winner_config_hash"]
    (tmp_path / "manifest.json").write_text(_json.dumps(manifest))
    (tmp_path / "selection.json").write_text(_json.dumps(sel))


def test_e5_resolver_reads_matrix_from_manifest_ignores_cli(tmp_path):
    # R7c: E5 reads the full contract matrix from the manifest; CLI ignored.
    import argparse
    cli = _load_runner("e5_r7c_b", "scripts/run_smco_evo_lowdim_check.py")
    _e5_full_manifest(tmp_path)
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(
        manifest=str(tmp_path / "manifest.json"),
        selection=str(tmp_path / "selection.json"),
        winner=None, development=False,
        dims=[999], instances=[999], fe_budget_per_d=9999)
    resolved = cli._resolve_winner(args, parser)
    assert resolved["winner"] == "PY-SP-SMCO-EVO"
    assert resolved["dims"] == [5, 20]            # manifest, not CLI 999
    assert resolved["instances"] == [1, 2, 3, 4, 5]  # 5 instances -> COCO ids 1..5
    assert resolved["fe_budget_per_d"] == 2000     # manifest, not CLI 9999


def test_e5_resolver_rejects_partial_matrix(tmp_path):
    # R7c reviewer repro: a single-function, d=5, single-instance E5 manifest
    # must not be accepted as canonical E5.
    import argparse
    cli = _load_runner("e5_r7c_c", "scripts/run_smco_evo_lowdim_check.py")
    _e5_full_manifest(tmp_path, functions=["f1"], dims=(5,), n_instances=1)
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(
        manifest=str(tmp_path / "manifest.json"),
        selection=str(tmp_path / "selection.json"),
        winner=None, development=False,
        dims=[5], instances=[1], fe_budget_per_d=2000)
    with pytest.raises(ValueError, match="functions|instances|dims"):
        cli._resolve_winner(args, parser)


def test_e5_resolver_rejects_wrong_stage(tmp_path):
    import argparse
    cli = _load_runner("e5_r2b_c", "scripts/run_smco_evo_lowdim_check.py")
    _e4_manifest(tmp_path, stage="e4_bbob_largescale", suite="bbob-largescale",
                 dims=(160,), n_instances=1, baselines=("DE",))
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(
        manifest=str(tmp_path / "manifest.json"),
        selection=str(tmp_path / "selection.json"),
        winner=None, development=False,
        dims=[5], instances=[1], fe_budget_per_d=2000)
    with pytest.raises(ValueError, match="stage"):
        cli._resolve_winner(args, parser)
