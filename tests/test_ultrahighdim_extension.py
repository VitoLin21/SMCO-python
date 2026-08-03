"""Prospective E3-F/E7 orchestration and evidence-chain contracts."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from smco.experiment_manifests import (
    E1_FUNCTIONS,
    E3F_FUNCTIONS,
    E7_FUNCTIONS,
)
from smco.ultrahighdim_extension import (
    DEADLINE_HOURS,
    E3F_ALGORITHMS,
    E7_ALGORITHMS,
    E7_NEW_ALGORITHMS,
    WALL_CHECKPOINT_HOURS,
    AttemptLedger,
    ProgressReporter,
    audit_extension_records,
    build_e3f_manifest,
    build_e7_manifest,
    build_extension_composite,
    build_extension_index,
    build_shards,
    deadline_evidence_errors,
    expected_logical_grid,
    recover_stalled_attempt,
    stalled_attempt_errors,
    supervise_command,
    validate_e3f_manifest,
    validate_e7_manifest,
    validate_extension_index,
    validate_extension_composite,
    validate_shards,
)


def _selection():
    return {
        "winner": "PY-SP-SMCO-EVO",
        "winner_language": "python",
        "selection_hash": "a" * 64,
    }


def _production_style_selection():
    selection = _selection()
    selection["selection_hash"] = "bcf87965006220a0"
    return selection


def _instance_index(functions, dims, n_instances=5):
    return {
        (function, dim, instance): {
            "stage": "extension_confirmatory",
            "artifact_dir": f"instances/extension_{function}_d{dim}_i{instance}",
            "transform_sha256": f"instance-{function}-{dim}-{instance}",
            "start_points_hash": f"starts-{function}-{dim}-{instance}",
        }
        for function in functions
        for dim in dims
        for instance in range(n_instances)
    }


def test_e3f_manifest_is_exact_420_run_extension_grid():
    manifest = build_e3f_manifest(
        _selection(),
        instance_index=_instance_index(E3F_FUNCTIONS, (200, 500, 1000)),
    )
    assert manifest["frozen"] is True
    assert manifest["n_tasks"] == 420
    assert len({task["run_id"] for task in manifest["tasks"]}) == 420
    assert {task["algorithm_id"] for task in manifest["tasks"]} == set(E3F_ALGORITHMS)
    assert manifest["deadline_hours"] == DEADLINE_HOURS == 72
    assert manifest["wall_checkpoint_hours"] == list(WALL_CHECKPOINT_HOURS)
    assert validate_e3f_manifest(manifest) == []


def test_extension_manifest_accepts_project_selection_id_and_binds_payload():
    selection = _production_style_selection()
    manifest = build_e3f_manifest(
        selection,
        instance_index=_instance_index(E3F_FUNCTIONS, (200, 500, 1000)),
    )
    assert manifest["selection_hash"] == "bcf87965006220a0"
    assert len(manifest["selection_payload_sha256"]) == 64
    assert validate_e3f_manifest(manifest) == []

    bad = copy.deepcopy(manifest)
    bad.pop("selection_payload_sha256")
    from smco.experiment_manifests import manifest_sha256

    bad["manifest_sha256"] = manifest_sha256(bad)
    assert any("selection_payload_sha256" in error
               for error in validate_e3f_manifest(bad))

    bad = copy.deepcopy(manifest)
    bad["tasks"][-1] = copy.deepcopy(bad["tasks"][0])
    # Even a re-hashed document cannot hide a duplicated/missing grid cell.
    from smco.experiment_manifests import manifest_sha256

    bad["manifest_sha256"] = manifest_sha256(bad)
    errors = validate_e3f_manifest(bad)
    assert any("grid" in error or "run_id" in error for error in errors)


def test_e7_manifest_has_1736_physical_tasks_and_2016_logical_cells():
    manifest = build_e7_manifest(
        _selection(),
        instance_index=_instance_index(
            E7_FUNCTIONS, (1000, 2000, 3000, 5000, 10000)
        ),
    )
    assert manifest["n_tasks"] == 1736
    assert validate_e7_manifest(manifest) == []
    d1000 = [task for task in manifest["tasks"] if task["dimension"] == 1000]
    assert len(d1000) == 200
    assert {task["algorithm_id"] for task in d1000} == set(E7_NEW_ALGORITHMS)
    assert len(expected_logical_grid("e7")) == 2016
    assert set(E3F_FUNCTIONS).isdisjoint(E1_FUNCTIONS)
    assert len(E7_ALGORITHMS) == 12


def test_deterministic_greedy_sharding_keeps_problem_bundles_intact():
    manifest = build_e3f_manifest(
        _selection(),
        instance_index=_instance_index(E3F_FUNCTIONS, (200, 500, 1000)),
    )
    first = build_shards(manifest, n_shards=7)
    second = build_shards(manifest, n_shards=7)
    assert first == second
    assert validate_shards(first, manifest) == []
    assert sum(shard["n_tasks"] for shard in first["shards"]) == 420

    owner = {}
    for shard in first["shards"]:
        for task in shard["tasks"]:
            bundle = (task["function"], task["dimension"], task["instance"])
            owner.setdefault(bundle, set()).add(shard["shard_id"])
    assert all(len(shards) == 1 for shards in owner.values())
    assert all(len({t["algorithm_id"] for t in shard["tasks"]}) == 7
               for shard in first["shards"] for t in shard["tasks"][:1])

    tampered = copy.deepcopy(first)
    tampered["shards"][0]["tasks"][0]["fe_budget"] = 1
    from smco.ultrahighdim_extension import _hash_document

    tampered["shard_sha256"] = _hash_document(tampered, "shard_sha256")
    assert any("task payload" in error for error in validate_shards(tampered, manifest))


def test_progress_reporter_checkpoint_and_deadline_snapshot_are_append_only(tmp_path):
    reporter = ProgressReporter(
        tmp_path,
        run_id="r1",
        attempt_id="r1.a001",
        wall_checkpoints_sec=(1.0, 6.0, 24.0, 72.0),
        deadline_sec=72.0,
        heartbeat_interval_sec=0.0,
    )
    reporter.record(fe_used=10, best_value=5.0, normalized_gap=0.5,
                    target_hit_fe={}, elapsed_sec=1.1)
    reporter.record(fe_used=20, best_value=3.0, normalized_gap=0.3,
                    target_hit_fe={}, elapsed_sec=72.1)
    deadline_path = tmp_path / "deadline_snapshot.json"
    before = deadline_path.read_bytes()
    reporter.record(fe_used=30, best_value=2.0, normalized_gap=0.2,
                    target_hit_fe={}, elapsed_sec=80.0)
    assert deadline_path.read_bytes() == before

    outcome = reporter.finalize({
        "run_id": "r1", "status": "success", "fe_used": 40,
        "best_value": 1.0, "normalized_gap": 0.1,
        "target_hit_fe": {}, "wall_time_sec": 90.0,
    }, elapsed_sec=90.0)
    assert outcome["deadline_exceeded"] is True
    assert outcome["deadline_fe_used"] == 20
    assert outcome["deadline_best_value"] == 3.0
    assert outcome["post_deadline_result"] is True
    assert outcome["status"] == "success"
    assert (tmp_path / "checkpoints" / "1h.json").exists()
    assert (tmp_path / "checkpoints" / "final.json").exists()
    assert (tmp_path / "heartbeat.json").exists()
    assert deadline_evidence_errors(outcome, tmp_path) == []


def test_attempt_ledger_preserves_same_run_id_retries_and_hash_chain(tmp_path):
    ledger = AttemptLedger(tmp_path / "attempt_ledger.json", run_id="r1")
    first = ledger.start(machine_id="node-a", git_commit="g", environment_hash="e")
    assert first["attempt_id"] == "r1.a001"
    ledger.finish(first["attempt_id"], status="infra_failure", failure_reason="node lost")
    second = ledger.start(machine_id="node-b", git_commit="g", environment_hash="e")
    assert second["attempt_id"] == "r1.a002"
    assert second["supersedes_attempt_id"] == first["attempt_id"]
    assert second["supersedes_run_id"] == "r1"
    ledger.finish(second["attempt_id"], status="success")
    assert ledger.validate() == []

    data = json.loads((tmp_path / "attempt_ledger.json").read_text())
    data["events"][0]["machine_id"] = "tampered"
    (tmp_path / "attempt_ledger.json").write_text(json.dumps(data))
    assert any("hash" in error for error in ledger.validate())


def test_stalled_recovery_requires_stale_heartbeat_and_no_fe_growth(tmp_path):
    from smco.ultrahighdim_extension import _hash_document

    run_dir = tmp_path / "r2"
    ledger = AttemptLedger(run_dir / "attempt_ledger.json", run_id="r2")
    first = ledger.start(machine_id="node", git_commit="git", environment_hash="env")
    history = run_dir / "attempts" / first["attempt_id"] / "heartbeats"
    history.mkdir(parents=True)
    for number, captured in enumerate((0.0, 700.0)):
        heartbeat = {
            "kind": "heartbeat", "run_id": "r2", "attempt_id": first["attempt_id"],
            "captured_unix_sec": captured, "fe_used": 50, "best_value": 2.0,
        }
        heartbeat["sidecar_sha256"] = _hash_document(heartbeat, "sidecar_sha256")
        (history / f"{number}.json").write_text(json.dumps(heartbeat))
    assert stalled_attempt_errors(tmp_path, "r2", now_unix_sec=2000.0) == []
    recovered = recover_stalled_attempt(tmp_path, "r2", now_unix_sec=2000.0)
    assert recovered["status"] == "stalled"
    second = ledger.start(machine_id="node2", git_commit="git", environment_hash="env")
    assert second["supersedes_attempt_id"] == first["attempt_id"]

    # FE growth means the infrastructure restart gate stays closed.
    run3 = tmp_path / "r3"
    ledger3 = AttemptLedger(run3 / "attempt_ledger.json", run_id="r3")
    attempt3 = ledger3.start(machine_id="node", git_commit="git", environment_hash="env")
    history3 = run3 / "attempts" / attempt3["attempt_id"] / "heartbeats"
    history3.mkdir(parents=True)
    for number, (captured, fe) in enumerate(((0.0, 50), (700.0, 51))):
        heartbeat = {"kind": "heartbeat", "run_id": "r3",
                     "attempt_id": attempt3["attempt_id"],
                     "captured_unix_sec": captured, "fe_used": fe, "best_value": 2.0}
        heartbeat["sidecar_sha256"] = _hash_document(heartbeat, "sidecar_sha256")
        (history3 / f"{number}.json").write_text(json.dumps(heartbeat))
    assert any("FE increased" in error for error in
               stalled_attempt_errors(tmp_path, "r3", now_unix_sec=2000.0))


def test_supervisor_crosses_deadline_without_killing_process(tmp_path):
    code = r'''
import json, os, time
from pathlib import Path
p = Path(os.environ["SMCO_PROGRESS_PATH"])
r = Path(os.environ["SMCO_RESULT_PATH"])
p.write_text(json.dumps({"fe_used": 5, "best_value": 4.0,
                         "normalized_gap": 0.4, "target_hit_fe": {}}))
time.sleep(0.12)
r.write_text(json.dumps({"run_id": "r1", "status": "success", "fe_used": 10,
                         "best_value": 2.0, "normalized_gap": 0.2,
                         "target_hit_fe": {}, "wall_time_sec": 0.12}))
'''
    outcome = supervise_command(
        [sys.executable, "-c", code],
        run_id="r1",
        evidence_root=tmp_path,
        deadline_sec=0.05,
        wall_checkpoints_sec=(0.02, 0.05),
        poll_interval_sec=0.01,
        heartbeat_interval_sec=0.01,
        machine_id="node", git_commit="git", environment_hash="env",
    )
    assert outcome["status"] == "success"
    assert outcome["deadline_exceeded"] is True
    assert outcome["post_deadline_result"] is True
    assert outcome["final_wall_time_sec"] >= 0.1
    assert deadline_evidence_errors(
        outcome, tmp_path / "r1" / "attempts" / "r1.a001"
    ) == []
    attempt_dir = tmp_path / "r1" / "attempts" / "r1.a001"
    snapshot = json.loads((attempt_dir / "deadline_snapshot.json").read_text())
    snapshot["best_value"] = -999
    # Even re-hashing the sidecar cannot evade the finish event's hash binding.
    from smco.ultrahighdim_extension import _hash_document

    snapshot["sidecar_sha256"] = _hash_document(snapshot, "sidecar_sha256")
    (attempt_dir / "deadline_snapshot.json").write_text(json.dumps(snapshot))
    assert any("ledger evidence hash" in error
               for error in deadline_evidence_errors(outcome, attempt_dir))


def test_supervisor_streams_large_worker_output_without_pipe_deadlock(tmp_path):
    code = r'''
import json, os, sys
from pathlib import Path
sys.stdout.write("x" * 200000)
sys.stderr.write("y" * 200000)
Path(os.environ["SMCO_RESULT_PATH"]).write_text(json.dumps({
    "run_id": "large-output", "status": "success", "fe_used": 1,
    "best_value": 0.0, "normalized_gap": 0.0, "target_hit_fe": {},
    "wall_time_sec": 0.0, "git_commit": "", "environment_hash": "",
    "machine_id": "",
}))
'''
    outcome = supervise_command(
        [sys.executable, "-c", code], run_id="large-output",
        evidence_root=tmp_path, poll_interval_sec=0.01,
        machine_id="node", git_commit="a" * 40, environment_hash="env",
    )
    attempt_dir = tmp_path / "large-output" / "attempts" / "large-output.a001"
    assert outcome["status"] == "success"
    assert outcome["git_commit"] == "a" * 40
    assert outcome["environment_hash"] == "env"
    assert outcome["machine_id"] == "node"
    assert (attempt_dir / "stdout.log").stat().st_size == 200000
    assert (attempt_dir / "stderr.log").stat().st_size == 200000


def test_supervisor_deadline_clock_is_cumulative_across_retries(tmp_path):
    first = "import time; time.sleep(0.07)"
    failed = supervise_command(
        [sys.executable, "-c", first], run_id="retry-clock",
        evidence_root=tmp_path, deadline_sec=0.1,
        wall_checkpoints_sec=(0.05, 0.1), poll_interval_sec=0.01,
        machine_id="node", git_commit="a" * 40, environment_hash="env",
    )
    assert failed["status"] == "infra_failure"
    second = r'''
import json, os, time
from pathlib import Path
time.sleep(0.07)
Path(os.environ["SMCO_RESULT_PATH"]).write_text(json.dumps({
  "run_id":"retry-clock", "status":"success", "fe_used":1,
  "best_value":0.0, "normalized_gap":0.0, "target_hit_fe":{},
  "wall_time_sec":0.07,
}))
'''
    outcome = supervise_command(
        [sys.executable, "-c", second], run_id="retry-clock",
        evidence_root=tmp_path, deadline_sec=0.1,
        wall_checkpoints_sec=(0.05, 0.1), poll_interval_sec=0.01,
        machine_id="node", git_commit="a" * 40, environment_hash="env",
    )
    assert outcome["attempt_id"].endswith("a002")
    assert outcome["final_wall_time_sec"] >= 0.14
    assert outcome["deadline_exceeded"] is True


def test_extension_index_hash_binds_manifest_merge_and_sidecars(tmp_path):
    manifest = build_e3f_manifest(
        _selection(),
        instance_index=_instance_index(E3F_FUNCTIONS, (200, 500, 1000)),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "valid_runs.csv").write_text("run_id\n")
    (merged / "deadline_snapshots.csv").write_text("run_id\n")
    (merged / "attempt_ledger.json").write_text("{}")
    (merged / "provenance_audit.json").write_text(json.dumps({
        "passed": False, "n_rows": 0, "checks": [], "deadline_checks": [],
    }))
    index = build_extension_index(
        "e3f", manifest_path=manifest_path, merged_dir=merged,
        composite_path=None, root=tmp_path,
    )
    assert any("audit" in error for error in validate_extension_index(index, root=tmp_path))

    # Content mutation is detected independently of the frozen index hash.
    (merged / "deadline_snapshots.csv").write_text("run_id\nr-extra\n")
    errors = validate_extension_index(index, root=tmp_path)
    assert any("deadline_snapshots" in error for error in errors)


def _write_grid_csv(path, cells, prefix):
    import csv

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "run_id", "function", "dimension", "instance", "algorithm_id",
            "instance_hash", "start_points_hash",
        ))
        writer.writeheader()
        for number, (function, dimension, instance, algorithm) in enumerate(sorted(cells)):
            writer.writerow({
                "run_id": f"{prefix}-{number}", "function": function,
                "dimension": dimension, "instance": instance,
                "algorithm_id": algorithm,
                "instance_hash": f"instance-{function}-{dimension}-{instance}",
                "start_points_hash": f"starts-{function}-{dimension}-{instance}",
            })


def _write_manifest_result_source(tmp_path, manifest, name):
    import csv

    manifest_path = tmp_path / f"{name}_manifest.json"
    audit_path = tmp_path / f"{name}_audit.json"
    csv_path = tmp_path / f"{name}.csv"
    manifest_path.write_text(json.dumps(manifest))
    audit_path.write_text(json.dumps({
        "passed": True, "n_rows": manifest["n_tasks"],
        "manifest_sha256": manifest["manifest_sha256"],
        "checks": [{"name": f"check-{i}", "passed": True} for i in range(12)],
        "deadline_checks": [{"name": "deadline", "passed": True}],
    }))
    fields = (
        "run_id", "function", "dimension", "instance", "algorithm_id",
        "configuration_hash", "instance_hash", "start_points_hash",
        "fe_budget", "fe_used", "status", "best_value", "known_optimum",
        "normalized_gap",
    )
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in manifest["tasks"]:
            writer.writerow({
                "run_id": task["run_id"], "function": task["function"],
                "dimension": task["dimension"], "instance": task["instance"],
                "algorithm_id": task["algorithm_id"],
                "configuration_hash": task["configuration_hash"],
                "instance_hash": task["instance_hash"],
                "start_points_hash": task["start_points_hash"],
                "fe_budget": task["fe_budget"], "fe_used": task["fe_budget"],
                "status": "success", "best_value": 1.0,
                "known_optimum": 0.0, "normalized_gap": 0.1,
            })
    return csv_path, manifest_path, audit_path


def test_logical_composites_bind_exact_840_and_2016_row_grids(tmp_path):
    from smco.ultrahighdim_extension import _hash_document

    original_cells = {
        cell for cell in expected_logical_grid("e3_combined")
        if cell[0] in E1_FUNCTIONS
    }
    old_csv = tmp_path / "e3.csv"
    _write_grid_csv(old_csv, original_cells, "old")
    e3f_manifest = build_e3f_manifest(
        _selection(),
        instance_index=_instance_index(E3F_FUNCTIONS, (200, 500, 1000)),
    )
    e3f_csv, e3f_manifest_path, e3f_audit_path = _write_manifest_result_source(
        tmp_path, e3f_manifest, "e3f",
    )
    old_index_path = tmp_path / "old_canonical_index.json"
    old_index = {"schema_version": "1", "frozen": True, "artifacts": []}
    old_index["index_sha256"] = _hash_document(old_index, "index_sha256")
    old_index_path.write_text(json.dumps(old_index))
    e3_composite = build_extension_composite(
        "e3f", selection_hash="a" * 64,
        sources=[
            {"role": "original_e3", "valid_runs_path": old_csv},
            {"role": "e3f", "valid_runs_path": e3f_csv,
             "manifest_path": e3f_manifest_path, "audit_path": e3f_audit_path},
        ],
        source_documents=[old_index_path],
        output_csv=tmp_path / "e3_combined.csv",
    )
    assert e3_composite["total_rows"] == 840
    assert validate_extension_composite(e3_composite) == []

    e7_manifest = build_e7_manifest(
        _selection(),
        instance_index=_instance_index(
            E7_FUNCTIONS, (1000, 2000, 3000, 5000, 10000),
        ),
    )
    e7_new_csv, e7_manifest_path, e7_audit_path = _write_manifest_result_source(
        tmp_path, e7_manifest, "e7_new",
    )
    e3_composite_path = tmp_path / "e3_composite.json"
    e3_composite_path.write_text(json.dumps(e3_composite))
    e7_composite = build_extension_composite(
        "e7", selection_hash="a" * 64,
        sources=[
            {"role": "reused_d1000", "valid_runs_path": tmp_path / "e3_combined.csv",
             "filter": {"dimension": 1000}},
            {"role": "physically_new", "valid_runs_path": e7_new_csv,
             "manifest_path": e7_manifest_path, "audit_path": e7_audit_path},
        ],
        source_documents=[e3_composite_path],
        output_csv=tmp_path / "e7_logical.csv",
    )
    assert e7_composite["sources"][0]["n_rows"] == 280
    assert e7_composite["total_rows"] == 2016
    assert validate_extension_composite(e7_composite) == []

    # A nominally complete logical grid is invalid if reused and new
    # algorithms were evaluated on different transforms or frozen starts.
    rows = _read_csv_for_test(e7_new_csv)
    rows[0]["instance_hash"] = "different-instance"
    _write_dict_rows_for_test(e7_new_csv, rows)
    with pytest.raises(ValueError, match="instance_hash differs from manifest"):
        build_extension_composite(
            "e7", selection_hash="a" * 64,
            sources=[
                {"role": "reused_d1000", "valid_runs_path": tmp_path / "e3_combined.csv",
                 "filter": {"dimension": 1000}},
                {"role": "physically_new", "valid_runs_path": e7_new_csv,
                 "manifest_path": e7_manifest_path, "audit_path": e7_audit_path},
            ],
            source_documents=[e3_composite_path],
            output_csv=tmp_path / "bad_e7_logical.csv",
        )

    e7_new_csv, _, _ = _write_manifest_result_source(tmp_path, e7_manifest, "e7_new")

    rows = (tmp_path / "e7_new.csv").read_text().splitlines()
    (tmp_path / "e7_new.csv").write_text("\n".join(rows[:-1]) + "\n")
    assert any("hash" in error or "grid" in error
               for error in validate_extension_composite(e7_composite))


def _read_csv_for_test(path):
    import csv

    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _write_dict_rows_for_test(path, rows):
    import csv

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_extension_audit_keeps_12_main_checks_and_separate_deadline_checks(tmp_path):
    manifest = build_e3f_manifest(
        _selection(),
        instance_index=_instance_index(E3F_FUNCTIONS, (200, 500, 1000)),
    )
    audit = audit_extension_records(manifest, [], evidence_root=tmp_path)
    assert len(audit["checks"]) == 12
    assert len(audit["deadline_checks"]) == 5
    assert audit["passed"] is False
    assert "manifest_run_id_coverage" in audit["failed_checks"]


def test_cli_manifest_dry_run_validates_exact_grid_without_writing(tmp_path):
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(_selection()))
    entries = []
    for (function, dimension, instance), entry in _instance_index(
        E3F_FUNCTIONS, (200, 500, 1000)
    ).items():
        entries.append({
            **entry, "function": function, "dimension": dimension,
            "instance_id": instance,
            "file_hashes": {"starts": entry["start_points_hash"]},
        })
    index_path = tmp_path / "instances_index.json"
    index_path.write_text(json.dumps({"instances": entries}))
    out = tmp_path / "manifest.json"
    command = [
        sys.executable, "scripts/run_smco_evo_ultrahighdim_extension.py",
        "manifest", "--campaign", "e3f", "--selection", str(selection_path),
        "--instances-index", str(index_path), "--out", str(out), "--dry-run",
    ]
    completed = subprocess.run(command, cwd=Path(__file__).parents[1],
                               text=True, capture_output=True, check=True)
    summary = json.loads(completed.stdout)
    assert summary["n_tasks"] == 420
    assert summary["errors"] == []
    assert not out.exists()


def test_instance_index_composer_verifies_artifacts_and_rewrites_repo_relative_path(
    tmp_path, monkeypatch,
):
    import numpy as np
    from smco.highdim_instances import generate_instance, write_instance_artifacts

    script = Path(__file__).parents[1] / "scripts" / "compose_smco_evo_ultrahighdim_instances.py"
    spec = importlib.util.spec_from_file_location("compose_ultra_instances", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = generate_instance(
        "Levy", 8, 0, stage="extension_confirmatory", seed=4,
    )
    artifact = tmp_path / "generated" / "instances" / "levy"
    metadata = write_instance_artifacts(instance, np.zeros((8, 8)), artifact)
    source = tmp_path / "generated" / "instances_index.json"
    source.write_text(json.dumps({"instances": [{
        "function": "Levy", "dimension": 8, "instance_id": 0,
        "stage": "extension_confirmatory", "artifact_dir": "instances/levy",
        "transform_sha256": metadata["transform_sha256"],
        "file_hashes": metadata["file_hashes"],
    }]}))
    monkeypatch.setattr(module, "_expected", lambda campaign: {("Levy", 8, 0)})
    composed = module.compose_instance_index("e3f", [source], repo_root=tmp_path)
    assert composed["n_instances"] == 1
    assert composed["instances"][0]["artifact_dir"] == "generated/instances/levy"

    starts = artifact / "starts.csv.gz"
    starts.write_bytes(starts.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        module.compose_instance_index("e3f", [source], repo_root=tmp_path)
