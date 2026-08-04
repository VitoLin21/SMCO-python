"""Tests for provenance defaults, confirmatory fail-fast, and worker propagation.

Covers reviewer P1a+/P1b: --confirmatory batch runners fail fast on incomplete
provenance; factorial/baseline worker commands propagate git/env/machine so
workers on rsync'd (non-git) trees emit non-empty provenance.
"""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

from smco.provenance import require_confirmatory_provenance


def _load_script(rel):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, Path(rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --- require_confirmatory_provenance (fail-fast) ---

def test_require_confirmatory_accepts_full_sha():
    require_confirmatory_provenance("host", "0" * 40, "env")


def test_require_confirmatory_rejects_empty_fields():
    with pytest.raises(SystemExit):
        require_confirmatory_provenance("", "0" * 40, "env")
    with pytest.raises(SystemExit):
        require_confirmatory_provenance("host", "", "env")
    with pytest.raises(SystemExit):
        require_confirmatory_provenance("host", "0" * 40, "")


def test_require_confirmatory_rejects_non_sha_git_commit():
    with pytest.raises(SystemExit):
        require_confirmatory_provenance("host", "short", "env")
    with pytest.raises(SystemExit):
        require_confirmatory_provenance("host", "g" * 39, "env")


# --- factorial _worker_command propagates provenance to the worker subprocess ---

def test_factorial_worker_command_python_propagates_provenance():
    f = _load_script("scripts/run_smco_evo_highdim_factorial.py")
    cmd = f._worker_command({"language": "python"}, "/tmp/t.json", "/inst", "/res", "/log",
                            machine_id="m1", git_commit="g" * 40, environment_hash="eh")
    assert "--git-commit" in cmd and ("g" * 40) in cmd
    assert "--environment-hash" in cmd and "eh" in cmd
    assert "--machine-id" in cmd and "m1" in cmd


def test_factorial_worker_command_r_propagates_git_commit():
    # R worker consumes --git-commit explicitly; --machine-id/--environment-hash
    # are passed on the command line but R records the actual host (Sys.info)
    # and R environment (R.version + package versions) — see the comment in
    # run_smco_evo_highdim_r.R. The dispatcher overrides only git_commit for R.
    f = _load_script("scripts/run_smco_evo_highdim_factorial.py")
    cmd = f._worker_command({"language": "r"}, "/tmp/t.json", "/inst", "/res", "/log",
                            machine_id="m1", git_commit="g" * 40, environment_hash="eh")
    assert cmd[0] == "Rscript"
    assert "--git-commit" in cmd and ("g" * 40) in cmd


# --- baseline _dispatch_baseline signature carries provenance ---

def test_baseline_dispatch_signature_carries_provenance():
    b = _load_script("scripts/run_smco_evo_highdim_baselines.py")
    params = inspect.signature(b._dispatch_baseline).parameters
    assert "machine_id" in params
    assert "git_commit" in params
    assert "environment_hash" in params
