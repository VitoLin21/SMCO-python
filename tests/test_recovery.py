"""Tests for the one-shot fast-die recovery tool (user option C, 2026-08-04).

The tool marks a fast-die a001 (exactly 1 heartbeat, stale, dead PID, no
worker, still open) as stalled so the normal 1da43e5 dispatch resume will
supersede it. It must NOT touch a001 evidence, must refuse everything that
fails the 5-condition gate, and must not run a concurrent dispatcher.
"""

import json

from smco.ultrahighdim_extension import AttemptLedger, _hash_document


def _write_heartbeat(run_dir, attempt_id, *, captured, fe, pid, n=0):
    hb_dir = run_dir / "attempts" / attempt_id / "heartbeats"
    hb_dir.mkdir(parents=True, exist_ok=True)
    hb = {
        "kind": "heartbeat", "run_id": run_dir.name, "attempt_id": attempt_id,
        "captured_unix_sec": captured, "fe_used": fe,
        "process_resources": {"pid": pid},
    }
    hb["sidecar_sha256"] = _hash_document(hb, "sidecar_sha256")
    (hb_dir / f"{n}.json").write_text(json.dumps(hb))
    return hb


def test_fast_die_attempt_recovered_to_stalled_with_decision_file(tmp_path):
    from smco.recovery import recover_fast_die_attempt
    import os
    run_dir = tmp_path / "rX"
    ledger = AttemptLedger(run_dir / "attempt_ledger.json", run_id="rX")
    a1 = ledger.start(machine_id="n", git_commit="g", environment_hash="e")
    _write_heartbeat(run_dir, a1["attempt_id"], captured=1000.0, fe=None, pid=999999)
    # now = 2201 -> age 1201s >= 1200 stale; pid 999999 dead; 1 heartbeat
    decision = recover_fast_die_attempt(tmp_path, "rX", now_unix_sec=2201.0)
    assert decision["action"] == "finish_stalled"
    assert decision["failure_reason"] == "early_worker_lost_before_second_heartbeat"
    atts = ledger.attempts()
    assert atts[-1]["finish"]["status"] == "stalled"
    assert (run_dir / "recovery_decision.json").exists()
    # a001 evidence preserved (heartbeat still on disk)
    assert (run_dir / "attempts" / a1["attempt_id"] / "heartbeats" / "0.json").exists()


def test_fast_die_refused_with_two_heartbeats(tmp_path):
    from smco.recovery import recover_fast_die_attempt
    run_dir = tmp_path / "r2"
    ledger = AttemptLedger(run_dir / "attempt_ledger.json", run_id="r2")
    a1 = ledger.start(machine_id="n", git_commit="g", environment_hash="e")
    _write_heartbeat(run_dir, a1["attempt_id"], captured=1000.0, fe=None, pid=999999, n=0)
    _write_heartbeat(run_dir, a1["attempt_id"], captured=1100.0, fe=5, pid=999999, n=1)
    decision = recover_fast_die_attempt(tmp_path, "r2", now_unix_sec=2201.0)
    assert decision["action"] is None
    assert "exactly 1 heartbeat" in decision["reason"]
    # a001 NOT finished
    assert ledger.attempts()[-1].get("finish") is None


def test_fast_die_refused_when_pid_still_alive(tmp_path):
    from smco.recovery import recover_fast_die_attempt
    import os
    run_dir = tmp_path / "r3"
    ledger = AttemptLedger(run_dir / "attempt_ledger.json", run_id="r3")
    a1 = ledger.start(machine_id="n", git_commit="g", environment_hash="e")
    _write_heartbeat(run_dir, a1["attempt_id"], captured=1000.0, fe=None, pid=os.getpid())
    decision = recover_fast_die_attempt(tmp_path, "r3", now_unix_sec=2201.0)
    assert decision["action"] is None
    assert "alive" in decision["reason"]


def test_fast_die_refused_when_heartbeat_not_stale(tmp_path):
    from smco.recovery import recover_fast_die_attempt
    run_dir = tmp_path / "r4"
    ledger = AttemptLedger(run_dir / "attempt_ledger.json", run_id="r4")
    a1 = ledger.start(machine_id="n", git_commit="g", environment_hash="e")
    _write_heartbeat(run_dir, a1["attempt_id"], captured=2100.0, fe=None, pid=999999)
    decision = recover_fast_die_attempt(tmp_path, "r4", now_unix_sec=2201.0)  # age 101s < 1200
    assert decision["action"] is None
    assert "not stale" in decision["reason"]


def test_fast_die_refused_when_already_finished(tmp_path):
    from smco.recovery import recover_fast_die_attempt
    run_dir = tmp_path / "r5"
    ledger = AttemptLedger(run_dir / "attempt_ledger.json", run_id="r5")
    a1 = ledger.start(machine_id="n", git_commit="g", environment_hash="e")
    _write_heartbeat(run_dir, a1["attempt_id"], captured=1000.0, fe=None, pid=999999)
    ledger.finish(a1["attempt_id"], status="infra_failure", failure_reason="x")
    decision = recover_fast_die_attempt(tmp_path, "r5", now_unix_sec=2201.0)
    assert decision["action"] is None
    assert "already finished" in decision["reason"]


def test_fast_die_refused_when_worker_still_in_ps(tmp_path, monkeypatch):
    from smco.recovery import recover_fast_die_attempt
    run_dir = tmp_path / "r6"
    ledger = AttemptLedger(run_dir / "attempt_ledger.json", run_id="r6")
    a1 = ledger.start(machine_id="n", git_commit="g", environment_hash="e")
    _write_heartbeat(run_dir, a1["attempt_id"], captured=1000.0, fe=None, pid=999999)
    monkeypatch.setattr("subprocess.run", lambda *a, **k: type("R", (), {"stdout": "123 python worker --task r6.task.json"})())
    decision = recover_fast_die_attempt(tmp_path, "r6", now_unix_sec=2201.0)
    assert decision["action"] is None
    assert "worker still in ps" in decision["reason"]
