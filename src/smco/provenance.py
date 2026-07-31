"""Default provenance fields (git_commit / environment_hash / machine_id).

Shared by the SMCO and baseline workers so every emitted outcome carries an
auditable source. The merge audit ``provenance_complete`` (P1a) requires all
three to be non-empty for E1 winner freezing and E2-E6 confirmatory results.
"""
from __future__ import annotations

import hashlib
import json
import platform
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


__all__ = ["default_git_commit", "default_environment_hash", "default_machine_id"]
