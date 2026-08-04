"""Tests for the deployment manifest (P1, 2026-08-04 review)."""

from smco.deployment import compute_source_manifest, verify_source_manifest


def test_source_manifest_binds_core_distinguishes_scheduling_and_detects_drift(tmp_path):
    """The manifest must (a) snapshot algorithm-core and scheduling hashes
    separately, (b) verify a live tree against a saved manifest, (c) flag an
    algorithm-core change as a results-comparability break, while (d) a
    scheduling-only change keeps results comparable. This is exactly the
    aa7 (algorithm core) vs 1da (scheduling-recovery) distinction the E7
    audit must report."""
    core = ["a.py", "b.py"]
    sched = ["c.py"]
    for rel in (*core, *sched):
        (tmp_path / rel).write_text("v1")
    m1 = compute_source_manifest(
        tmp_path, algorithm_core_files=core, scheduling_files=sched, commit="g1",
    )
    assert m1["algorithm_core_sha256"]
    assert m1["scheduling_sha256"] != m1["algorithm_core_sha256"]
    assert m1["git_commit"] == "g1"

    rep = verify_source_manifest(
        m1, tmp_path, algorithm_core_files=core, scheduling_files=sched, commit="g1",
    )
    assert rep["algorithm_core_match"] and rep["scheduling_match"] and rep["git_commit_match"]

    # algorithm-core drift -> results NOT comparable
    (tmp_path / "a.py").write_text("v2")
    rep_core = verify_source_manifest(
        m1, tmp_path, algorithm_core_files=core, scheduling_files=sched, commit="g1",
    )
    assert not rep_core["algorithm_core_match"]

    # scheduling-only change -> algorithm core still matches (results comparable)
    (tmp_path / "a.py").write_text("v1")
    (tmp_path / "c.py").write_text("v2")
    rep_sched = verify_source_manifest(
        m1, tmp_path, algorithm_core_files=core, scheduling_files=sched, commit="g1",
    )
    assert rep_sched["algorithm_core_match"]
    assert not rep_sched["scheduling_match"]


def test_compute_source_manifest_resolves_git_commit_when_not_injected(tmp_path):
    """Regression: a parameter named `git_commit` shadowed the module function,
    so omitting it raised TypeError (called None). Omitting `commit` must
    resolve the repo's HEAD (empty string for a non-git tmp_path, not crash)."""
    (tmp_path / "a.py").write_text("v1")
    m = compute_source_manifest(tmp_path, algorithm_core_files=["a.py"], scheduling_files=[])
    assert m["git_commit"] == ""  # tmp_path is not a git repo
    assert m["algorithm_core_sha256"]
