"""Cross-language instance parity: R loader vs Python loader (Task 6 scenario 7).

The R loader (vendor/SMCO_R/main/highdim_instances.R) reads the same csv.gz
artifacts produced by Python and must return identical objective values at any
point. This runs ``Rscript`` on the base-R loader (no jsonlite needed: metadata
scalars are passed in), so it executes wherever Rscript exists.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from smco.experiment_manifests import build_algorithm_config, build_task
from smco.highdim_instances import generate_instance, write_instance_artifacts

pytestmark = pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not available")

REPO = Path(__file__).resolve().parent.parent
R_LOADER = REPO / "vendor" / "SMCO_R" / "main" / "highdim_instances.R"
R_WORKER = REPO / "scripts" / "run_smco_evo_highdim_r.R"


def _r_has(pkgs):
    expr = "cat(" + " && ".join(f"requireNamespace('{p}', quietly=TRUE)" for p in pkgs) + ")"
    res = subprocess.run(["Rscript", "-e", expr], capture_output=True, text=True)
    return "TRUE" in res.stdout


def _eval_in_r(artifact_dir, function_name, dim, asym, scale, opt_value, points):
    points_file = artifact_dir / "_parity_points.csv"
    values_file = artifact_dir / "_parity_values.txt"
    np.savetxt(points_file, np.asarray(points, dtype=float), delimiter=",")
    script = f'''
    source("{R_LOADER}")
    inst <- load_highdim_instance("{artifact_dir}", "{function_name}", {dim}, {asym}, {scale}, {opt_value})
    pts <- as.matrix(read.csv("{points_file}", header = FALSE))
    vals <- apply(pts, 1, inst$objective)
    writeLines(formatC(vals, digits = 17, format = "g"), "{values_file}")
    '''
    result = subprocess.run(
        ["Rscript", "-e", script], capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Rscript failed:\n{result.stderr}"
    return [float(line) for line in values_file.read_text().splitlines()]


@pytest.mark.parametrize(
    "function_name, dim",
    [
        ("Rastrigin", 4),
        ("Ackley", 8),
        ("Griewank", 12),
        ("Zakharov", 240),  # block rotation (d > FULL_ROTATION_DIM)
        ("Rosenbrock", 4),  # non-zero base optimum (ones)
        ("Levy", 8),
        ("Schwefel226", 8),
        ("HighConditionedEllipsoid", 12),
    ],
)
def test_r_objective_matches_python(tmp_path, function_name, dim):
    inst = generate_instance(function_name, dim, 0, seed=7)
    rng = np.random.default_rng(3)
    span = inst.bounds_upper - inst.bounds_lower
    points = [inst.known_optimum_x] + [
        inst.bounds_lower + rng.uniform(size=dim) * span for _ in range(5)
    ]
    py_values = [inst.objective(p) for p in points]

    artifact_dir = tmp_path / "inst"
    write_instance_artifacts(inst, np.zeros((2, dim)), artifact_dir)
    meta = json.loads((artifact_dir / "metadata.json").read_text())

    r_values = _eval_in_r(
        artifact_dir,
        function_name,
        dim,
        meta["asymmetry_strength"],
        meta["objective_scale"],
        float(meta["known_optimum_value"]),
        points,
    )
    assert len(r_values) == len(py_values)
    for py_v, r_v in zip(py_values, r_values):
        assert np.isclose(py_v, r_v, atol=1e-9, rtol=1e-9), (
            f"{function_name} d={dim}: python={py_v!r} R={r_v!r}"
        )


def test_r_worker_runs_base_task_end_to_end(tmp_path):
    if not _r_has(["jsonlite", "qrng"]):
        pytest.skip("jsonlite/qrng not installed")
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    rng = np.random.default_rng(5)
    span = inst.bounds_upper - inst.bounds_lower
    starts = inst.bounds_lower + rng.uniform(size=(4, 4)) * span
    art_dir = tmp_path / "inst" / "instances" / "dev_i0"
    meta = write_instance_artifacts(inst, starts, art_dir)
    cfg = build_algorithm_config(
        "r", "smco", False, "none",
        evolution_strategy="none", evolution_points=(),
        elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=4,
    )
    task = build_task(
        "e0_contract", "contract", "Rastrigin", 4, 0, 0,
        config=cfg, fe_budget=200, checkpoints=(50, 100, 200), seed=12345,
        instance_artifact_dir="instances/dev_i0",
        instance_hash=meta["transform_sha256"],
        start_points_hash=meta["file_hashes"]["starts"],
    )
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task))
    res = subprocess.run(
        ["Rscript", str(R_WORKER), "--task", str(task_path),
         "--instance-root", str(tmp_path / "inst"),
         "--result-dir", str(tmp_path / "raw"), "--log-dir", str(tmp_path / "logs")],
        capture_output=True, text=True, timeout=180,
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "raw" / f"{task['run_id']}.json").read_text())
    assert payload["status"] == "success"
    assert payload["fe_used"] <= 200
    assert payload["best_value"] >= -1e-9  # minimisation
    assert set(payload["target_hit_fe"]) == {"1e-1", "1e-2", "1e-3", "1e-5"}
