"""One-shot recovery tool for fast-die E7 attempts (user option C, 2026-08-04).

Marks a fast-die a001 (exactly 1 hash-valid heartbeat, stale >= 20 min, dead
PID, no worker in ps, still open with no finish event) as stalled so the
normal 1da43e5 dispatch resume will supersede it via the standard retryable
chain. Does NOT modify ultrahighdim_extension.py, does NOT run a concurrent
dispatcher, and preserves all a001 evidence plus a recovery_decision.json.

This is intentionally separate: active dispatchers keep running unchanged,
and the actual compute retry happens only after the existing dispatcher exits
and a 1da43e5 resume picks up the now-retryable run.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from .ultrahighdim_extension import AttemptLedger, _hash_document

FAILURE_REASON = "early_worker_lost_before_second_heartbeat"


def _heartbeat_hash_valid(hb) -> bool:
    return hb.get("sidecar_sha256") == _hash_document(hb, "sidecar_sha256")


def _pid_dead(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except (TypeError, ValueError, OSError):
        return False
    return False


def _worker_in_ps(run_id) -> bool:
    try:
        ps = subprocess.run(
            ["ps", "-eo", "pid,cmd"], capture_output=True, text=True, check=False,
        ).stdout
    except OSError:
        return False
    return run_id in ps


def recover_fast_die_attempt(
    evidence_root, run_id: str, *, now_unix_sec: float | None = None,
    stale_after_sec: float = 1200.0, dry_run: bool = False,
) -> dict:
    """Apply the 5-condition fast-die gate to one run_id. If it passes, append
    attempt_finished(status="stalled") to the existing ledger (a001 preserved)
    and write recovery_decision.json. Returns a decision record either way."""
    now = time.time() if now_unix_sec is None else float(now_unix_sec)
    run_dir = Path(evidence_root) / run_id
    decision = {
        "run_id": run_id, "now_unix_sec": now, "conditions": {},
        "action": None, "tool": "recover_fast_die_attempt",
    }
    ledger = AttemptLedger(run_dir / "attempt_ledger.json", run_id=run_id)
    if ledger.validate():
        decision["reason"] = "ledger invalid"
        return decision
    attempts = ledger.attempts()
    if not attempts:
        decision["reason"] = "no attempts"
        return decision
    latest = attempts[-1]
    if latest.get("finish") is not None:
        decision["reason"] = "latest attempt already finished"
        return decision
    aid = latest["attempt_id"]
    hb_dir = run_dir / "attempts" / aid / "heartbeats"
    hbs = sorted(hb_dir.glob("*.json")) if hb_dir.exists() else []
    decision["conditions"]["n_heartbeats"] = len(hbs)
    if len(hbs) != 1:
        decision["reason"] = f"need exactly 1 heartbeat, got {len(hbs)}"
        return decision
    hb = json.loads(hbs[-1].read_text())
    decision["conditions"]["heartbeat_hash_valid"] = _heartbeat_hash_valid(hb)
    if not decision["conditions"]["heartbeat_hash_valid"]:
        decision["reason"] = "heartbeat hash invalid"
        return decision
    captured = float(hb.get("captured_unix_sec", 0))
    decision["conditions"]["age_sec"] = now - captured
    decision["conditions"]["stale"] = (now - captured) >= stale_after_sec
    if not decision["conditions"]["stale"]:
        decision["reason"] = "heartbeat not stale"
        return decision
    pid = (hb.get("process_resources") or {}).get("pid")
    decision["conditions"]["pid"] = pid
    decision["conditions"]["pid_dead"] = _pid_dead(pid) if pid is not None else True
    if not decision["conditions"]["pid_dead"]:
        decision["reason"] = "worker PID still alive"
        return decision
    decision["conditions"]["no_worker_in_ps"] = not _worker_in_ps(run_id)
    if not decision["conditions"]["no_worker_in_ps"]:
        decision["reason"] = "worker still in ps"
        return decision
    # all 5 conditions pass
    decision["action"] = "finish_stalled"
    decision["attempt_id"] = aid
    decision["failure_reason"] = FAILURE_REASON
    if dry_run:
        decision["dry_run"] = True
        return decision
    ledger.finish(aid, status="stalled", failure_reason=FAILURE_REASON)
    (run_dir / "recovery_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True)
    )
    return decision
