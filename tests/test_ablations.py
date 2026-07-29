"""Tests for E6 ablation configuration generation (Task 10).

Ablations use DEVELOPMENT instances (never confirmatory) and vary ONE control of
the E1 winner at a time. This module covers the config-generation level
(E6.2 strategy, E6.3 schedule); E6.1 start-count needs per-n_starts start
artifacts and E6.4 state-component resets need SMCO SP hooks, both flagged for
later. The runner itself reuses the factorial batch + dev-instance manifest.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from smco.ablations import ABLATION_DIMENSIONS, ablation_configs
from smco.paper_contract import parse_algorithm_id

_ABLATIONS_RUNNER = (
    Path(__file__).resolve().parent.parent / "scripts" / "run_smco_evo_ablations.py"
)


def _load_ablations_cli():
    spec = importlib.util.spec_from_file_location("smco_evo_ablations_cli", _ABLATIONS_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ablation_configs_cover_strategy_and_schedule():
    configs = ablation_configs("PY-SP-SMCO-EVO")
    dims = {label for label, _variant, _cfg in configs}
    assert "strategy" in dims
    assert "schedule" in dims


def test_ablation_configs_have_distinct_hashes():
    configs = ablation_configs("PY-SP-SMCO-EVO")
    hashes = [cfg["configuration_hash"] for _l, _v, cfg in configs]
    # The default winner config appears once as strategy=rand1bin and once as
    # the schedule default (ep0.5,0.75 er0.25); every other variant is distinct.
    assert len(configs) == 8
    assert len(set(hashes)) >= len(configs) - 1


def test_ablation_configs_strategy_variants_are_the_four_strategies():
    configs = ablation_configs("R-RS-SMCO-REFINE-EVO")
    strat = [v for label, v, _cfg in configs if label == "strategy"]
    assert set(strat) == {"rand1bin", "best1bin", "current-to-best1bin", "sobol"}


def test_ablation_configs_empty_for_non_evo_winner():
    # Ablations are EVO-only; a base winner yields no ablation configs.
    assert ablation_configs("PY-BASE-SMCO") == []


def test_ablation_configs_preserve_winner_language_family_semantics():
    configs = ablation_configs("PY-SP-SMCO-EVO")
    assert configs, "expected non-empty ablations for an EVO winner"
    for _label, _variant, cfg in configs:
        parsed = parse_algorithm_id(cfg["algorithm_id"])
        assert parsed["language"] == "python"
        assert parsed["family"] == "smco"
        assert parsed["state_semantics"] == "state_preserving"
        assert cfg["evolutionary"] == "true"


def test_ablation_dimensions_documented():
    assert "strategy" in ABLATION_DIMENSIONS
    assert "schedule" in ABLATION_DIMENSIONS


def test_build_ablation_manifest(tmp_path):
    cli = _load_ablations_cli()
    manifest = cli.build_ablation_manifest(
        winner="PY-SP-SMCO-EVO", functions=["Rastrigin"], dims=[4], n_instances=1,
        fe_budget_per_d=50, checkpoints_per_d=(50,), out_dir=tmp_path,
    )
    assert manifest["frozen"] is True
    n_configs = len(ablation_configs("PY-SP-SMCO-EVO"))
    assert len(manifest["tasks"]) == n_configs * 1 * 1 * 1
    assert (tmp_path / "e6_ablations__synthetic_highdim.json").exists()
    # non-EVO winner is rejected
    import pytest
    with pytest.raises(ValueError):
        cli.build_ablation_manifest(
            winner="PY-BASE-SMCO", functions=["Rastrigin"], dims=[4],
            n_instances=1, fe_budget_per_d=50, checkpoints_per_d=(50,),
        )


def test_start_count_configs_three_tiers():
    from smco.ablations import start_count_configs
    configs = start_count_configs("PY-SP-SMCO-EVO", 1000)
    labels = [label for label, _cfg in configs]
    ns = sorted({cfg["n_starts"] for _label, cfg in configs})
    assert ns == [8, 16, 32]  # ceil(sqrt(1000)) = 32
    assert set(labels) == {"n8", "n16", "n32"}
    # different n_starts → different configuration_hash
    hashes = {cfg["configuration_hash"] for _label, cfg in configs}
    assert len(hashes) == 3


def test_start_count_configs_non_evo_empty():
    from smco.ablations import start_count_configs
    assert start_count_configs("PY-BASE-SMCO", 1000) == []


def test_build_start_count_ablation_manifest_per_dim(tmp_path):
    cli = _load_ablations_cli()
    import json
    idx_path = tmp_path / "idx.json"
    idx_path.write_text(json.dumps({"instances": [
        {"function": "Rastrigin", "dimension": 1000, "instance_id": 0,
         "artifact_dir": "art", "transform_sha256": "ih", "start_points_hash": "h8",
         "extra_starts": {"16": {"hash": "h16"}, "32": {"hash": "h32"}}}]}))
    manifest = cli.build_start_count_ablation_manifest(
        winner="PY-SP-SMCO-EVO", functions=["Rastrigin"], dims=[1000], n_instances=1,
        fe_budget_per_d=100, checkpoints_per_d=(100,), instances_index=idx_path)
    assert manifest["frozen"] is True
    ns = sorted({t["n_starts"] for t in manifest["tasks"]})
    assert ns == [8, 16, 32]  # ceil(sqrt(1000)) = 32
    by_n = {t["n_starts"]: t["start_points_hash"] for t in manifest["tasks"]}
    assert by_n == {8: "h8", 16: "h16", 32: "h32"}
