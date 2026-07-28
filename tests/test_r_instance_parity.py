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

from smco.highdim_instances import generate_instance, write_instance_artifacts

pytestmark = pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not available")

REPO = Path(__file__).resolve().parent.parent
R_LOADER = REPO / "vendor" / "SMCO_R" / "main" / "highdim_instances.R"


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
