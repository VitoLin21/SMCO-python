"""Tests for high-dimensional reproducible instances (Task 6).

Covers the eight mandatory scenarios in the implementation plan plus basic
correctness of the shift / asymmetry / permutation / block-rotation transform.

The transform is built so the base optimum is a fixed point: every linear or
piecewise piece (shift, block rotation, permutation, T_asym) leaves the optimum
value untouched, so the instance ``known_optimum_value`` equals the base
function optimum and ``known_optimum_x`` equals the shift vector.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from smco.highdim_instances import (
    DEFAULT_BLOCK_SIZE,
    FULL_ROTATION_DIM,
    HighDimInstance,
    TransformSpec,
    generate_instance,
    instance_seed,
    load_instance,
    load_starts,
    known_functions,
    write_instance_artifacts,
)

# E1 is frozen to its first four functions.  E3-F/E7 add the remaining four
# entries below without modifying that historical grid.
INSTANCE_FUNCTIONS = [
    "Rastrigin", "Ackley", "Griewank", "Zakharov",
    "Rosenbrock", "Levy", "Schwefel226", "HighConditionedEllipsoid",
]
# Minimization global optimum of each base function (transform preserves it).
BASE_OPTIMUM = {
    "Rastrigin": 0.0,
    "Ackley": 0.0,
    "Griewank": 0.0,
    "Zakharov": 0.0,
    "Rosenbrock": 0.0,
    "Levy": 0.0,
    "Schwefel226": 0.0,
    "HighConditionedEllipsoid": 0.0,
}


@pytest.mark.parametrize("function_name", INSTANCE_FUNCTIONS)
def test_known_optimum_within_bounds(function_name):
    # Scenario 1: the known optimum lies inside the search domain.
    inst = generate_instance(function_name, dimension=8, instance_id=0, seed=1)
    lo, hi = inst.bounds_lower, inst.bounds_upper
    x = inst.known_optimum_x
    assert np.all(x >= lo - 1e-9)
    assert np.all(x <= hi + 1e-9)


@pytest.mark.parametrize("function_name", INSTANCE_FUNCTIONS)
def test_transformed_optimum_reaches_known_value(function_name):
    # Scenario 2: after the transform the optimum still attains the known value.
    inst = generate_instance(function_name, dimension=8, instance_id=0, seed=1)
    val = inst.objective(inst.known_optimum_x)
    assert np.isclose(val, inst.known_optimum_value, atol=1e-8)
    assert np.isclose(inst.known_optimum_value, BASE_OPTIMUM[function_name], atol=1e-9)


def test_objective_at_optimum_beats_random_point():
    inst = generate_instance("Rastrigin", dimension=10, instance_id=2, seed=7)
    rng = np.random.default_rng(0)
    span = inst.bounds_upper - inst.bounds_lower
    x_rand = inst.bounds_lower + rng.uniform(size=10) * span
    # Minimization: the optimum must be no worse than a random feasible point.
    assert inst.objective(inst.known_optimum_x) <= inst.objective(x_rand) + 1e-9


@pytest.mark.parametrize("function_name", INSTANCE_FUNCTIONS)
def test_all_highdim_functions_are_finite_at_boundaries_and_random_point(function_name):
    inst = generate_instance(function_name, dimension=240, instance_id=1, seed=15)
    rng = np.random.default_rng(16)
    random_point = inst.bounds_lower + rng.uniform(size=inst.dimension) * (
        inst.bounds_upper - inst.bounds_lower
    )

    for point in (inst.bounds_lower, inst.bounds_upper, random_point):
        assert np.isfinite(inst.objective(point))


def test_highdim_registry_matches_the_eight_function_extension_grid():
    assert known_functions() == tuple(INSTANCE_FUNCTIONS)


def test_same_seed_same_transform_hash():
    # Scenario 3: identical parameters reproduce the identical transform hash.
    a = generate_instance("Ackley", dimension=12, instance_id=3, seed=42)
    b = generate_instance("Ackley", dimension=12, instance_id=3, seed=42)
    assert isinstance(a, HighDimInstance)
    assert a.transform_spec.sha256() == b.transform_spec.sha256()


def test_different_instance_id_different_transform():
    # Scenario 4: different instance ids yield different transforms. Use the
    # default instance_seed derivation (which depends on instance_id) so the
    # two instances actually differ.
    a = generate_instance("Griewank", dimension=12, instance_id=0)
    b = generate_instance("Griewank", dimension=12, instance_id=1)
    assert a.transform_spec.sha256() != b.transform_spec.sha256()
    assert not np.allclose(a.known_optimum_x, b.known_optimum_x)


def test_block_rotation_is_orthogonal():
    # Scenario 5: each block-rotation block is (near) orthogonal.
    dim = FULL_ROTATION_DIM + DEFAULT_BLOCK_SIZE  # forces block mode
    inst = generate_instance("Zakharov", dimension=dim, instance_id=0, seed=3)
    assert inst.transform_spec.rotation_mode == "block"
    for block in inst.transform_spec.rotation_blocks:
        prod = block @ block.T
        assert np.allclose(prod, np.eye(block.shape[0]), atol=1e-9)


def test_full_rotation_is_orthogonal():
    # d <= FULL_ROTATION_DIM uses a single full orthogonal rotation.
    inst = generate_instance("Rastrigin", dimension=16, instance_id=0, seed=5)
    assert inst.transform_spec.rotation_mode == "full"
    blocks = inst.transform_spec.rotation_blocks
    assert len(blocks) == 1
    r = blocks[0]
    assert np.allclose(r @ r.T, np.eye(r.shape[0]), atol=1e-9)


def test_inverse_transform_round_trip():
    # Scenario 6: apply_inverse(apply_forward(z)) == z for mixed-sign vectors.
    inst = generate_instance("Ackley", dimension=20, instance_id=1, seed=9)
    spec = inst.transform_spec
    assert isinstance(spec, TransformSpec)
    rng = np.random.default_rng(123)
    for _ in range(5):
        z = rng.uniform(-2.0, 2.0, size=spec.dimension)
        z_back = spec.apply_inverse(spec.apply_forward(z))
        assert np.allclose(z_back, z, atol=1e-9)


def test_artifact_load_reproduces_objective(tmp_path):
    # Scenario 7: an instance rebuilt from artifacts matches the original
    # objective at arbitrary points (this is what the R worker must reproduce
    # by reading the same language-neutral csv.gz files).
    inst = generate_instance("Rastrigin", dimension=24, instance_id=4, seed=11)
    rng = np.random.default_rng(0)
    span = inst.bounds_upper - inst.bounds_lower
    starts = inst.bounds_lower + rng.uniform(size=(8, 24)) * span
    meta = write_instance_artifacts(inst, starts, tmp_path)
    assert set(meta["file_hashes"]).issuperset(
        {"shift", "permutation", "rotation_blocks"}
    )
    loaded = load_instance(tmp_path)
    rng2 = np.random.default_rng(99)
    x = inst.bounds_lower + rng2.uniform(size=24) * span
    assert np.isclose(loaded.objective(x), inst.objective(x), atol=1e-9)
    assert np.isclose(
        loaded.objective(inst.known_optimum_x), inst.known_optimum_value, atol=1e-8
    )
    assert np.allclose(load_starts(tmp_path), starts)


def test_d10000_does_not_allocate_dense_matrix_and_is_finite_at_domain_points():
    # Scenario 8: d=10000 must use fixed-size blocks rather than a dense
    # matrix, and all prescribed evaluation points remain finite.
    inst = generate_instance("HighConditionedEllipsoid", dimension=10_000, instance_id=0, seed=2)
    blocks = inst.transform_spec.rotation_blocks
    total = sum(b.size for b in blocks)
    assert total < 10_000 * 10_000
    assert total * 8 < 10 * 1024 * 1024
    assert all(b.shape[0] <= DEFAULT_BLOCK_SIZE for b in blocks)
    midpoint = (inst.bounds_lower + inst.bounds_upper) / 2.0
    random_point = inst.bounds_lower + np.random.default_rng(3).uniform(size=inst.dimension) * (
        inst.bounds_upper - inst.bounds_lower
    )
    for point in (inst.known_optimum_x, inst.bounds_lower, inst.bounds_upper, midpoint, random_point):
        assert np.isfinite(inst.objective(point))
    assert inst.objective(inst.known_optimum_x) == 0.0


@pytest.mark.parametrize("function_name", ["Levy", "Schwefel226", "HighConditionedEllipsoid"])
def test_new_functions_replay_hash_and_transformed_optimum(function_name):
    a = generate_instance(function_name, dimension=240, instance_id=2, stage="e3f")
    b = generate_instance(function_name, dimension=240, instance_id=2, stage="e3f")

    assert a.transform_spec.rotation_mode == "block"
    assert a.transform_spec.sha256() == b.transform_spec.sha256()
    assert a.objective(a.known_optimum_x) == 0.0


def test_schwefel_domain_extension_prevents_false_below_optimum_value():
    inst = generate_instance(
        "Schwefel226", dimension=200, instance_id=0,
        stage="extension_confirmatory",
    )
    z = np.zeros(inst.dimension)
    z[0] = 717.0659474261937 - 420.9687462275036
    x = inst.known_optimum_x + inst.transform_spec.apply_forward(z)
    assert np.all(x >= inst.bounds_lower)
    assert np.all(x <= inst.bounds_upper)
    assert inst.objective(x) >= inst.known_optimum_value


def test_block_size_is_part_of_the_replay_hash():
    a = generate_instance("Levy", dimension=240, instance_id=2, seed=4, block_size=40)
    b = generate_instance("Levy", dimension=240, instance_id=2, seed=4, block_size=60)

    assert a.transform_spec.rotation_mode == b.transform_spec.rotation_mode == "block"
    assert a.transform_spec.block_size == 40
    assert b.transform_spec.block_size == 60
    assert a.transform_spec.sha256() != b.transform_spec.sha256()


def test_dev_and_confirmatory_namespaces_do_not_overlap():
    # Development and confirmatory namespaces must not share transforms even
    # at the same instance id. Default seeding depends on stage, so the two
    # differ without an explicit seed.
    dev = generate_instance("Rastrigin", dimension=200, instance_id=0, stage="development")
    conf = generate_instance("Rastrigin", dimension=200, instance_id=0, stage="confirmatory")
    assert dev.transform_spec.sha256() != conf.transform_spec.sha256()


def test_instance_seed_is_stable_and_stage_separated():
    assert instance_seed("Rastrigin", 200, 0, "development") == instance_seed(
        "Rastrigin", 200, 0, "development"
    )
    assert instance_seed("Rastrigin", 200, 0, "development") != instance_seed(
        "Rastrigin", 200, 0, "confirmatory"
    )


_MANIFEST_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "generate_smco_evo_manifests.py"
)


def _load_manifest_module():
    spec = importlib.util.spec_from_file_location(
        "smco_evo_manifests_cli", _MANIFEST_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_instance_set_writes_artifacts_and_index(tmp_path):
    # The Task 6 CLI exposes build_instance_set: it materialises per-instance
    # artifacts, an instances_index.json, and is reproducible.
    mod = _load_manifest_module()
    index = mod.build_instance_set(
        functions=["Rastrigin", "Ackley"],
        dims=[6, 12],
        n_instances=2,
        stage="development",
        out_dir=tmp_path,
        n_starts=4,
    )
    assert index["stage"] == "development"
    assert len(index["instances"]) == 2 * 2 * 2
    for entry in index["instances"]:
        art = tmp_path / entry["artifact_dir"]
        assert (art / "metadata.json").exists()
        loaded = load_instance(art)
        assert np.isclose(
            loaded.objective(loaded.known_optimum_x),
            loaded.known_optimum_value,
            atol=1e-8,
        )
        assert np.asarray(load_starts(art)).shape == (4, loaded.dimension)
    # Reproducibility: rebuilding yields identical transform hashes.
    index2 = mod.build_instance_set(
        functions=["Rastrigin", "Ackley"],
        dims=[6, 12],
        n_instances=2,
        stage="development",
        out_dir=tmp_path / "again",
        n_starts=4,
    )
    key = lambda e: (e["function"], e["dimension"], e["instance_id"])
    h1 = {key(e): e["transform_sha256"] for e in index["instances"]}
    h2 = {key(e): e["transform_sha256"] for e in index2["instances"]}
    assert h1 == h2
    assert (tmp_path / "instances_index.json").exists()


def test_write_extra_starts_artifacts(tmp_path):
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    starts8 = np.tile(inst.bounds_lower, (8, 1))
    extra16 = np.tile(inst.bounds_upper, (16, 1))
    meta = write_instance_artifacts(inst, starts8, tmp_path, extra_starts={16: extra16})
    assert meta["extra_starts"]["16"]["file"] == "starts_n16.csv.gz"
    assert meta["extra_starts"]["16"]["n_starts"] == 16
    assert (tmp_path / "starts_n16.csv.gz").exists()
    # default n8 path unchanged
    assert meta["file_hashes"]["starts"] and meta["n_starts"] == 8


def test_load_starts_by_n(tmp_path):
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    s8 = np.zeros((8, 4))
    s16 = np.full((16, 4), 7.0)
    write_instance_artifacts(inst, s8, tmp_path, extra_starts={16: s16})
    assert load_starts(tmp_path).shape == (8, 4)        # default n8
    assert load_starts(tmp_path, 8).shape == (8, 4)     # explicit n8
    assert load_starts(tmp_path, 16).shape == (16, 4)   # extra tier


def test_load_starts_missing_tier_raises(tmp_path):
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    write_instance_artifacts(inst, np.zeros((8, 4)), tmp_path)
    with pytest.raises(FileNotFoundError):
        load_starts(tmp_path, 99)


def test_build_instance_set_extra_starts(tmp_path):
    import math
    mod = _load_manifest_module()
    idx = mod.build_instance_set(
        ["Rastrigin"], [16], 1, stage="development", out_dir=tmp_path,
        n_starts=8, extra_n_starts=("16", "sqrt"))
    entry = idx["instances"][0]
    sqrt_n = int(math.ceil(math.sqrt(16)))  # =4
    assert "16" in entry["extra_starts"]            # explicit 16
    assert str(sqrt_n) in entry["extra_starts"]     # sqrt tier resolved to 4
    art = tmp_path / entry["artifact_dir"]
    assert load_starts(art, 16).shape == (16, 16)
    assert load_starts(art, sqrt_n).shape == (sqrt_n, 16)
