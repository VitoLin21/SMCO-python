"""Tests for worker observability diagnostics (P1 part 2, 2026-08-04 review).

Next-round run diagnostics so an external interruption is distinguishable
from an algorithm failure: worker_pid/parent_pid/command_sha256/started_unix/
exit_code/termination_signal/interruption_kind. Must NOT change existing
success-outcome schema semantics and must NOT mislabel an external
interruption as algorithm_failure.
"""

from smco.ultrahighdim_extension import (
    _classify_interruption, _worker_diagnostics, _signal_name,
)


def test_classify_interruption_distinguishes_four_kinds():
    assert _classify_interruption(0) == "normal_exit"
    assert _classify_interruption(1) == "nonzero_exit"
    assert _classify_interruption(42) == "nonzero_exit"
    assert _classify_interruption(-15) == "signal"   # SIGTERM
    assert _classify_interruption(-2) == "signal"    # SIGINT
    assert _classify_interruption(-1) == "signal"    # SIGHUP
    assert _classify_interruption(None) == "unknown"


def test_signal_name_maps_known_signals_and_falls_back():
    assert _signal_name(-15) == "SIGTERM"
    assert _signal_name(-2) == "SIGINT"
    assert _signal_name(-1) == "SIGHUP"
    assert _signal_name(-9) == "SIGKILL"
    assert _signal_name(0) is None
    assert _signal_name(None) is None


def test_worker_diagnostics_captures_sigterm_interruption():
    d = _worker_diagnostics(
        worker_pid=123, parent_pid=45, command=["python", "-c", "x"],
        started_unix_sec=1000.0, returncode=-15, recorded_unix_sec=2000.0,
    )
    assert d["worker_pid"] == 123
    assert d["parent_pid"] == 45
    assert d["worker_exit_code"] == -15
    assert d["worker_termination_signal"] == "SIGTERM"
    assert d["interruption_kind"] == "signal"
    assert d["worker_started_unix_sec"] == 1000.0
    assert d["recorded_unix_sec"] == 2000.0
    assert len(d["worker_command_sha256"]) == 64


def test_worker_diagnostics_normal_exit_has_no_signal():
    d = _worker_diagnostics(worker_pid=1, parent_pid=2, command=["x"], started_unix_sec=1000.0, returncode=0, recorded_unix_sec=2000.0)
    assert d["worker_exit_code"] == 0
    assert d["worker_termination_signal"] is None
    assert d["interruption_kind"] == "normal_exit"


def test_worker_diagnostics_nonzero_exit_not_mislabeled_signal():
    d = _worker_diagnostics(worker_pid=1, parent_pid=2, command=["x"], started_unix_sec=1000.0, returncode=3, recorded_unix_sec=2000.0)
    assert d["interruption_kind"] == "nonzero_exit"
    assert d["worker_termination_signal"] is None
