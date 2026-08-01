"""CLI tests for the P1c E3 composite generation + analysis gate (review §6).

Covers the two new generate stages (e3-baseline-component, e3-composite) and the
analyze --composite gate (E3 statistics refused without a validated composite).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

# Reuse the real-manifest composite builders from the confirmatory test module
# (pytest makes sibling test modules importable by name).
from test_confirmatory import (
    _build_full_baseline_component,
    _build_full_e2,
    _build_valid_composite,
    _write_component_merged,
    _confirmatory_instance_index,
)


def _load(rel):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, Path(rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --- generation: --stage e3-baseline-component ---

def _write_confirmatory_index_json(path, functions, dims, n_instances):
    entries = []
    for fn in functions:
        for d in dims:
            for i in range(n_instances):
                entries.append({
                    "function": fn, "dimension": int(d), "instance_id": i,
                    "stage": "confirmatory",
                    "artifact_dir": f"instances/confirmatory_{fn}_d{int(d)}_i{i}",
                    "transform_sha256": f"th_{fn}_{d}_{i}",
                    "file_hashes": {"starts": f"sph_{fn}_{d}_{i}"},
                })
    path.write_text(json.dumps({"instances": entries}))


def test_generate_e3_baseline_component_writes_300(tmp_path):
    gen = _load("scripts/generate_smco_evo_manifests.py")
    sel = {"winner": "PY-SP-SMCO-EVO", "winner_language": "python",
           "selection_hash": "bcf87965006220a0"}
    (tmp_path / "sel.json").write_text(json.dumps(sel))
    idx = tmp_path / "instances_index.json"
    _write_confirmatory_index_json(idx, ["Rastrigin", "Ackley", "Griewank", "Zakharov"],
                                   [200, 500, 1000], 5)
    out = tmp_path / "out"
    rc = gen.main([
        "--stage", "e3-baseline-component", "--selection", str(tmp_path / "sel.json"),
        "--instances-index", str(idx), "--out-dir", str(out),
    ])
    assert rc == 0
    written = out / "e3_baseline_component__synthetic_highdim.json"
    assert written.exists()
    manifest = json.loads(written.read_text())
    assert manifest["n_tasks"] == 300
    assert manifest["component_role"] == "baseline_extension"
    from smco.confirmatory import baseline_component_errors
    assert baseline_component_errors(manifest) == []


def test_generate_e3_baseline_component_requires_selection(tmp_path):
    gen = _load("scripts/generate_smco_evo_manifests.py")
    idx = tmp_path / "instances_index.json"
    _write_confirmatory_index_json(idx, ["Rastrigin"], [200], 1)
    with pytest.raises(SystemExit):
        gen.main(["--stage", "e3-baseline-component",
                  "--instances-index", str(idx), "--out-dir", str(tmp_path / "out")])


# --- generation: --stage e3-composite ---

def test_generate_e3_composite_writes_420(tmp_path):
    gen = _load("scripts/generate_smco_evo_manifests.py")
    e2 = _build_full_e2()
    bc = _build_full_baseline_component()
    e2_mp = tmp_path / "e2m.json"; e2_mp.write_text(json.dumps(e2))
    bc_mp = tmp_path / "bcm.json"; bc_mp.write_text(json.dumps(bc))
    e2_dir = _write_component_merged(tmp_path / "e2", e2)
    bc_dir = _write_component_merged(tmp_path / "bc", bc)
    comp_out = tmp_path / "composite.json"
    rc = gen.main([
        "--stage", "e3-composite",
        "--e2-manifest", str(e2_mp), "--e2-merged-dir", str(e2_dir),
        "--baseline-manifest", str(bc_mp), "--baseline-merged-dir", str(bc_dir),
        "--composite-out", str(comp_out),
    ])
    assert rc == 0
    composite = json.loads(comp_out.read_text())
    assert composite["total_runs"] == 420
    assert composite["frozen"] is True
    from smco.confirmatory import validate_composite
    assert validate_composite(composite) == []


def test_generate_e3_composite_rejects_missing_args(tmp_path):
    gen = _load("scripts/generate_smco_evo_manifests.py")
    with pytest.raises(SystemExit):
        gen.main(["--stage", "e3-composite", "--e2-manifest", str(tmp_path / "x.json")])


def test_generate_e3_composite_rejects_bad_sources(tmp_path):
    # selection_hash mismatch between E2 and baseline -> builder raises -> CLI exits
    gen = _load("scripts/generate_smco_evo_manifests.py")
    e2 = _build_full_e2(sel_hash="HASH_A")
    bc = _build_full_baseline_component(sel_hash="HASH_B")
    e2_mp = tmp_path / "e2m.json"; e2_mp.write_text(json.dumps(e2))
    bc_mp = tmp_path / "bcm.json"; bc_mp.write_text(json.dumps(bc))
    e2_dir = _write_component_merged(tmp_path / "e2", e2)
    bc_dir = _write_component_merged(tmp_path / "bc", bc)
    with pytest.raises(ValueError, match="selection_hash"):
        gen.main([
            "--stage", "e3-composite",
            "--e2-manifest", str(e2_mp), "--e2-merged-dir", str(e2_dir),
            "--baseline-manifest", str(bc_mp), "--baseline-merged-dir", str(bc_dir),
            "--composite-out", str(tmp_path / "composite.json"),
        ])


# --- analysis: E3 --statistics resolves via canonical index / composite gate (review P0) ---

def _e3_final_merged(tmp_path, e2, bc, **kw):
    from test_confirmatory import _write_final_merged
    return _write_final_merged(tmp_path / "final", e2, bc, **kw)


def test_analyze_default_stage_e3_merged_is_rejected(tmp_path):
    # review P0 regression: default stage + an E3 merged dir + NO composite must
    # still be rejected (the gate cannot be bypassed by omitting --stage/--composite).
    ana = _load("scripts/analyze_smco_evo_highdim_paper.py")
    comp, *_, e2, bc = _build_valid_composite(tmp_path)
    final = _e3_final_merged(tmp_path, e2, bc)  # contains DE/GA/PSO/SA/GenSA
    with pytest.raises(SystemExit):
        ana.main(["--statistics", "--merged-dir", str(final)])  # default stage, no composite


def test_analyze_e3_bare_merged_requires_composite(tmp_path):
    ana = _load("scripts/analyze_smco_evo_highdim_paper.py")
    comp, *_, e2, bc = _build_valid_composite(tmp_path)
    final = _e3_final_merged(tmp_path, e2, bc)
    with pytest.raises(SystemExit):
        ana.main(["--statistics", "--merged-dir", str(final)])  # no composite


def test_analyze_e3_rejects_invalid_composite(tmp_path):
    ana = _load("scripts/analyze_smco_evo_highdim_paper.py")
    comp, *_, e2, bc = _build_valid_composite(tmp_path)
    final = _e3_final_merged(tmp_path, e2, bc)
    (tmp_path / "comp.json").write_text(json.dumps({"frozen": False}))  # invalid
    with pytest.raises(SystemExit):
        ana.main(["--statistics", "--merged-dir", str(final),
                  "--composite", str(tmp_path / "comp.json")])


def test_analyze_e3_rejects_when_merged_misses_row(tmp_path):
    ana = _load("scripts/analyze_smco_evo_highdim_paper.py")
    comp, *_, e2, bc = _build_valid_composite(tmp_path)
    comp_path = tmp_path / "comp.json"
    comp_path.write_text(json.dumps(comp))
    final = _e3_final_merged(tmp_path, e2, bc, drop_run_id=e2["tasks"][0]["run_id"])
    with pytest.raises(SystemExit):
        ana.main(["--statistics", "--merged-dir", str(final), "--composite", str(comp_path)])


def _stub_primary_table(monkeypatch):
    import smco.paper_analysis
    called = {}

    def _fake(md, out, algos):
        called["ran"] = (str(md), list(algos))
        return []

    monkeypatch.setattr(smco.paper_analysis, "write_primary_table", _fake)
    return called


def test_analyze_e1_statistics_does_not_require_composite(tmp_path, monkeypatch):
    # E1 data (no baselines) via bare --merged-dir proceeds with selection_candidates.
    import csv as _csv
    called = _stub_primary_table(monkeypatch)
    ana = _load("scripts/analyze_smco_evo_highdim_paper.py")
    merged = tmp_path / "merged"; merged.mkdir()
    with open(merged / "valid_runs.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["run_id", "algorithm_id", "stage"])
        w.writeheader()
        for i in range(3):
            w.writerow({"run_id": f"r{i}", "algorithm_id": "PY-SP-SMCO-EVO",
                        "stage": "e1_development"})
    rc = ana.main(["--statistics", "--merged-dir", str(merged)])
    assert rc == 0
    assert called.get("ran")  # statistics ran, no composite gate


def test_analyze_e3_runs_after_valid_gate_uses_composite_algos(tmp_path, monkeypatch):
    # review P0: valid composite + 420 merged -> statistics runs with the
    # composite's 7 algorithms (NOT the 18 E1 candidates).
    called = _stub_primary_table(monkeypatch)
    ana = _load("scripts/analyze_smco_evo_highdim_paper.py")
    comp, *_, e2, bc = _build_valid_composite(tmp_path)
    comp_path = tmp_path / "comp.json"
    comp_path.write_text(json.dumps(comp))
    final = _e3_final_merged(tmp_path, e2, bc)
    rc = ana.main(["--statistics", "--merged-dir", str(final), "--composite", str(comp_path)])
    assert rc == 0
    md, algos = called["ran"]
    assert set(algos) == set(comp["algorithms"])  # 7 composite algorithms, not 18
    assert "DE" in algos and "GenSA" in algos  # baselines NOT dropped


def test_analyze_e3_via_canonical_index(tmp_path, monkeypatch):
    # review P0: Task-12 resolves E3 inputs from the canonical index + key; the
    # composite gate runs and algorithms come from the validated composite.
    import smco.canonical_artifacts as ca
    called = _stub_primary_table(monkeypatch)
    ana = _load("scripts/analyze_smco_evo_highdim_paper.py")
    comp, *_, e2, bc = _build_valid_composite(tmp_path)
    comp_path = tmp_path / "comp.json"
    comp_path.write_text(json.dumps(comp))
    final = _e3_final_merged(tmp_path, e2, bc)
    idx_path = tmp_path / "index.json"
    idx_path.write_text(json.dumps({"schema_version": "1", "frozen": True, "artifacts": []}))
    monkeypatch.setattr(ca, "validate_canonical_index", lambda index, **k: [])
    monkeypatch.setattr(ca, "resolve_analysis_target",
                        lambda index, key, **k: {"key": key, "merged_dir": str(final),
                                                 "is_e3": True, "composite_path": str(comp_path)})
    rc = ana.main(["--statistics", "--canonical-index", str(idx_path),
                   "--artifact-key", "e3_composite_merged"])
    assert rc == 0
    _, algos = called["ran"]
    assert set(algos) == set(comp["algorithms"])