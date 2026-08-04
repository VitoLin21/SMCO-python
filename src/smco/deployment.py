"""Deployment manifest: bind a frozen git_commit to source-tree hashes.

Remote E7 nodes have no `.git`, so `--git-commit` passed to dispatch is a
claim, not a verifiable fact. A deployment manifest captures the SHA-256 of
each algorithm-core source file at a given commit, so a recovered evidence
directory can later prove which exact source produced it. This does NOT
retroactively verify trees that were never snapshotted; it only lets us bind
new/future deployments and detect drift (2026-08-04 review, P1).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Iterable

# The algorithm execution core: a change here can alter optimizer behaviour and
# MUST NOT be silently mixed across the campaign. ultrahighdim_extension.py is
# scheduling/recovery, listed for completeness but the audit distinguishes it.
ALGORITHM_CORE_FILES = (
    "src/smco/e7_algorithm_adapters.py",
    "src/smco/baseline_worker.py",
    "src/smco/highdim_worker.py",
    "src/smco/highdim_instances.py",
    "src/smco/test_functions.py",
)
SCHEDULING_FILES = (
    "src/smco/ultrahighdim_extension.py",
)


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _combined_sha256(entries: Iterable[str]) -> str:
    return hashlib.sha256("".join(entries).encode("utf-8")).hexdigest()


def git_commit(repo_root) -> str:
    """Return the full HEAD commit of `repo_root`, or '' if not a git repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
    except (FileNotFoundError, OSError):
        return ""
    return out.stdout.strip()


def compute_source_manifest(
    repo_root, *, algorithm_core_files=ALGORITHM_CORE_FILES,
    scheduling_files=SCHEDULING_FILES, git_commit: str | None = None,
) -> dict:
    """Snapshot the source-tree identity at `repo_root`.

    `algorithm_core_sha256` covers optimizer/worker/objective files only; a
    change here means results are NOT comparable across the boundary.
    `scheduling_sha256` covers dispatch/recovery and may change without
    invalidating algorithm results. `git_commit` is injected for testability.
    """
    repo = Path(repo_root)
    core = {rel: file_sha256(repo / rel) for rel in algorithm_core_files}
    sched = {rel: file_sha256(repo / rel) for rel in scheduling_files}
    return {
        "git_commit": git_commit if git_commit is not None else git_commit(repo_root),
        "algorithm_core": core,
        "algorithm_core_sha256": _combined_sha256(core.values()),
        "scheduling": sched,
        "scheduling_sha256": _combined_sha256(sched.values()),
    }


def verify_source_manifest(manifest: dict, repo_root, **kwargs) -> dict:
    """Compare a saved manifest against the live tree. Returns a report with
    `algorithm_core_match` (results-comparability gate) and `git_commit_match`."""
    actual = compute_source_manifest(repo_root, **kwargs)
    return {
        "algorithm_core_match":
            actual["algorithm_core_sha256"] == manifest.get("algorithm_core_sha256"),
        "scheduling_match":
            actual["scheduling_sha256"] == manifest.get("scheduling_sha256"),
        "git_commit_match":
            actual["git_commit"] == manifest.get("git_commit"),
        "actual_git_commit": actual["git_commit"],
        "manifest_git_commit": manifest.get("git_commit"),
    }
