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


# --- analysis: E3 --statistics requires --composite and a valid gate ---

def test_analyze_e3_statistics_requires_composite(tmp_path):
    ana = _load("scripts/analyze_smco_evo_highdim_paper.py")
    merged = tmp_path / "merged"; merged.mkdir()
    with pytest.raises(SystemExit):
        ana.main(["--stage", "e3-comparative-analysis", "--statistics",
                  "--merged-dir", str(merged)])


def test_analyze_e3_statistics_rejects_invalid_composite(tmp_path):
    ana = _load("scripts/analyze_smco_evo_highdim_paper.py")
    merged = tmp_path / "merged"; merged.mkdir()
    (tmp_path / "comp.json").write_text(json.dumps({"frozen": False}))  # invalid
    with pytest.raises(SystemExit):
        ana.main(["--stage", "e3-comparative-analysis", "--statistics",
                  "--merged-dir", str(merged), "--composite", str(tmp_path / "comp.json")])


def test_analyze_e3_statistics_rejects_when_merged_misses_row(tmp_path):
    ana = _load("scripts/analyze_smco_evo_highdim_paper.py")
    comp, e2_mp, bc_mp, e2_dir, bc_dir, e2, bc = _build_valid_composite(tmp_path)
    comp_path = tmp_path / "comp.json"
    comp_path.write_text(json.dumps(comp))
    # final merged missing one E2 row -> gate fails -> CLI exits before statistics
    from test_confirmatory import _write_final_merged
    final = _write_final_merged(tmp_path / "final", e2, bc,
                                drop_run_id=e2["tasks"][0]["run_id"])
    with pytest.raises(SystemExit):
        ana.main(["--stage", "e3-comparative-analysis", "--statistics",
                  "--merged-dir", str(final), "--composite", str(comp_path)])


def test_analyze_e1_statistics_does_not_require_composite(tmp_path, monkeypatch):
    # E1/E2 analyses keep their original entry: --stage e1 without --composite
    # must NOT trip the E3 gate — statistics proceeds (write_primary_table called).
    import smco.paper_analysis
    ana = _load("scripts/analyze_smco_evo_highdim_paper.py")
    called = {}

    def _fake(md, out, algos):
        called["ran"] = True
        return []

    monkeypatch.setattr(smco.paper_analysis, "write_primary_table", _fake)
    merged = tmp_path / "merged"; merged.mkdir()
    rc = ana.main(["--stage", "e1-development", "--statistics", "--merged-dir", str(merged)])
    assert rc == 0
    assert called.get("ran")  # statistics ran, no composite gate


def test_analyze_e3_statistics_runs_after_valid_gate(tmp_path, monkeypatch):
    # review §6.4: a valid composite + exact 420 merged -> statistics runs.
    import smco.paper_analysis
    ana = _load("scripts/analyze_smco_evo_highdim_paper.py")
    called = {}

    def _fake(md, out, algos):
        called["ran"] = (str(md), algos)
        return []

    monkeypatch.setattr(smco.paper_analysis, "write_primary_table", _fake)
    comp, e2_mp, bc_mp, e2_dir, bc_dir, e2, bc = _build_valid_composite(tmp_path)
    comp_path = tmp_path / "comp.json"
    comp_path.write_text(json.dumps(comp))
    from test_confirmatory import _write_final_merged
    final = _write_final_merged(tmp_path / "final", e2, bc)
    rc = ana.main(["--stage", "e3-comparative-analysis", "--statistics",
                   "--merged-dir", str(final), "--composite", str(comp_path)])
    assert rc == 0
    assert called.get("ran")  # gate passed, statistics executed
