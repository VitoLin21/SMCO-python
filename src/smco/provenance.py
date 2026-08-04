"""Default provenance fields (git_commit / environment_hash / machine_id).

Shared by the SMCO and baseline workers so every emitted outcome carries an
auditable source. The merge audit ``provenance_complete`` (P1a) requires all
three to be non-empty for E1 winner freezing and E2-E6 confirmatory results.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import socket
import subprocess


def default_git_commit() -> str:
    """Current git HEAD, or "" if not in a git repo (e.g. rsync'd worker tree)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def default_environment_hash() -> str:
    """Stable hash of the python/numpy/platform runtime."""
    import numpy as np

    payload = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def default_machine_id() -> str:
    return socket.gethostname()


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def require_confirmatory_provenance(machine_id, git_commit, environment_hash):
    """Fail fast BEFORE dispatch: a confirmatory run must carry complete provenance
    so the merge ``provenance_complete`` audit passes every row after a full run.
    Raises ``SystemExit`` with guidance — do not waste a fleet run on rows the
    audit will reject. ``git_commit`` must be a full 40-hex SHA (so the result
    points at an exact commit, not a branch tip or empty default).
    """
    missing = [n for n, v in [("machine_id", machine_id),
                              ("git_commit", git_commit),
                              ("environment_hash", environment_hash)] if not v]
    if missing:
        raise SystemExit(
            f"confirmatory run requires non-empty {missing}; pass "
            f"--git-commit <40-hex-SHA> --environment-hash <h> --machine-id <h>"
        )
    if not _SHA_RE.match(git_commit or ""):
        raise SystemExit(
            f"confirmatory run requires a full 40-hex git_commit SHA, got "
            f"{git_commit!r}; pass --git-commit $(git rev-parse HEAD)"
        )


__all__ = [
    "default_git_commit", "default_environment_hash", "default_machine_id",
    "require_confirmatory_provenance",
]
