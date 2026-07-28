"""Contract tests for the SMCO-EVO paper result schema (Task 0)."""

from __future__ import annotations

import pytest

from smco import paper_contract as pc


@pytest.mark.parametrize(
    "language,family,evolutionary,state_semantics,expected",
    [
        ("python", "smco", True, "state_preserving", "PY-SP-SMCO-EVO"),
        ("python", "smco_boost_refine", True, "restart", "PY-RS-SMCO-BOOST-REFINE-EVO"),
        ("r", "smco_refine", True, "state_preserving", "R-SP-SMCO-REFINE-EVO"),
        ("r", "smco", True, "restart", "R-RS-SMCO-EVO"),
        ("python", "smco", False, None, "PY-BASE-SMCO"),
        ("r", "smco_boost_refine", False, None, "R-BASE-SMCO-BOOST-REFINE"),
    ],
)
def test_algorithm_id_build_and_parse_roundtrip(
    language, family, evolutionary, state_semantics, expected
):
    built = pc.build_algorithm_id(language, family, evolutionary, state_semantics)
    assert built == expected
    parsed = pc.parse_algorithm_id(built)
    assert parsed["language"] == language
    assert parsed["family"] == family
    assert parsed["evolutionary"] is evolutionary
    if evolutionary:
        assert parsed["state_semantics"] == state_semantics
    else:
        assert parsed["state_semantics"] == pc.NONE_TOKEN


def test_algorithm_id_rejects_bad_inputs():
    with pytest.raises(ValueError):
        pc.build_algorithm_id("julia", "smco", True, "state_preserving")
    with pytest.raises(ValueError):
        pc.build_algorithm_id("python", "smco_xx", True, "state_preserving")
    with pytest.raises(ValueError):
        pc.build_algorithm_id("python", "smco", True, None)  # EVO needs semantics
    with pytest.raises(ValueError):
        pc.parse_algorithm_id("PY-SMCO")  # malformed
    with pytest.raises(ValueError):
        pc.parse_algorithm_id("PY-XX-SMCO-EVO")  # bad slot


def test_run_id_deterministic_and_cross_language_stable():
    config = {
        "algorithm_id": "PY-SP-SMCO-EVO",
        "evolution_strategy": "rand1bin",
        "n_starts": 8,
        "evolution_points": "0.50,0.75",
        "elimination_rate": "0.25",
        "de_factor": "0.80",
        "de_crossover": "0.70",
        "fe_budget_per_d": 1000,
        "state_semantics": "state_preserving",
        "refine_ratio": "none",
        "objective_sense": "minimize",
    }
    cfg_hash = pc.compute_configuration_hash(config)
    base_task = {
        "stage": "e1_development",
        "suite": "synthetic_highdim",
        "function": "rastrigin",
        "dimension": 1000,
        "instance": 1,
        "replication": 0,
        "algorithm_id": "PY-SP-SMCO-EVO",
        "evolution_strategy": "rand1bin",
        "seed": 11,
        "fe_budget": 1000 * 1000,
        "n_starts": 8,
        "configuration_hash": cfg_hash,
    }
    r1 = pc.compute_run_id(base_task)
    r2 = pc.compute_run_id(dict(base_task))
    assert r1 == r2
    assert r1.startswith("r") and len(r1) == 17
    # Reordering the dict must not change the id (canonical json sorts keys).
    reversed_task = {k: base_task[k] for k in reversed(list(base_task.keys()))}
    assert pc.compute_run_id(reversed_task) == r1
    # A different algorithm_id (R variant) must yield a different run_id.
    other = dict(base_task)
    other["algorithm_id"] = "R-RS-SMCO-EVO"
    assert pc.compute_run_id(other) != r1


def test_configuration_hash_stable_across_float_formatting():
    cfg = {
        "de_factor": pc.format_cfg_float(0.8),
        "de_crossover": pc.format_cfg_float(0.7),
        "elimination_rate": pc.format_cfg_float(0.25),
    }
    assert pc.compute_configuration_hash(cfg) == pc.compute_configuration_hash(cfg)
    # Same numeric value formatted identically -> same hash.
    cfg2 = {
        "de_factor": pc.format_cfg_float(0.80),
        "de_crossover": pc.format_cfg_float(0.70),
        "elimination_rate": pc.format_cfg_float(0.250),
    }
    assert pc.compute_configuration_hash(cfg2) == pc.compute_configuration_hash(cfg)


def _good_row(**overrides):
    row = {
        "schema_version": pc.SCHEMA_VERSION,
        "manifest_id": "m" + "0" * 15,
        "stage": "e1_development",
        "suite": "synthetic_highdim",
        "function": "rastrigin",
        "dimension": 1000,
        "instance": 1,
        "replication": 0,
        "seed": 11,
        "language": "python",
        "state_semantics": "state_preserving",
        "family": "smco",
        "evolutionary": "true",
        "evolution_strategy": "rand1bin",
        "algorithm_id": "PY-SP-SMCO-EVO",
        "n_starts": 8,
        "fe_budget": 1000000,
        "fe_used": 999981,
        "checkpoint_fe": 999981,
        "best_value": -1.23,
        "known_optimum": 0.0,
        "normalized_gap": 0.01,
        "objective_sense": "minimize",
        "target_hit_fe_1e-1": 12000,
        "target_hit_fe_1e-2": "",
        "target_hit_fe_1e-3": "",
        "target_hit_fe_1e-5": "",
        "wall_time_sec": 12.3,
        "peak_memory_mb": 220.0,
        "status": "success",
        "failure_reason": "",
        "is_confirmatory": "false",
        "supersedes_run_id": "",
        "machine_id": "host-a",
        "git_commit": "deadbeef",
        "environment_hash": "e" * 16,
        "start_points_hash": "s" * 16,
        "instance_hash": "i" * 16,
        "configuration_hash": "c" * 16,
        "run_id": "r" + "a" * 16,
        "termination_reason": "evaluation_budget",
        "fe_counts_by_event": '{"initialization":8}',
    }
    row.update(overrides)
    return row


def test_validate_result_row_accepts_good_row():
    assert pc.validate_result_row(_good_row()) == []


def test_validate_result_row_catches_violations():
    bad = _good_row(
        schema_version="2",
        language="julia",
    )
    errors = pc.validate_result_row(bad)
    assert errors
    assert any("schema_version" in e for e in errors)
    assert any("language" in e for e in errors)
    # Bad language makes algorithm_id rebuild impossible; validator must not crash.
    assert any("algorithm_id cannot be rebuilt" in e for e in errors)


def test_validate_result_row_flags_algorithm_id_mismatch():
    # Valid language/family/semantics but algorithm_id string disagrees.
    bad = _good_row(algorithm_id="PY-BASE-SMCO")  # row claims EVO though
    errors = pc.validate_result_row(bad)
    assert any("algorithm_id mismatch" in e for e in errors)
