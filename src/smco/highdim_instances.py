"""Reproducible high-dimensional test instances for the SMCO-EVO paper (Task 6).

Each instance wraps a scalable base function with a composition of transforms
that all keep the base optimum a fixed point, so the instance optimum *value* is
always the base optimum value and the instance optimum *location* is the shift
vector. The transform order (base -> instance) is::

    x = x_opt + R . perm . T_asym(z)                       (forward)
    z = T_asym^{-1} . perm^{-1} . R^T (x - x_opt)          (inverse, evaluate)

where ``R`` is orthogonal (full for ``d <= FULL_ROTATION_DIM``, block-diagonal
above so we never materialise a dense ``d x d`` matrix), ``perm`` is a
coordinate permutation and ``T_asym`` is the COCO left/right asymmetric power
transform (identity at the origin, hence optimum-preserving).

Artifacts are language-neutral gzipped CSV plus a JSON ``metadata.json`` with
the SHA-256 of each payload, so the R worker (Task 8) can consume the exact
same transform by reading the same files.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paper_contract import canonical_json
from .test_functions import (
    ackley,
    griewank,
    high_conditioned_ellipsoid,
    levy,
    rastrigin,
    rosenbrock,
    SCHWEFEL_226_OPTIMUM_X,
    schwefel_226,
    zakharov,
)

# --- tunable constants (written into every artifact for reproducibility) ---
FULL_ROTATION_DIM = 200
DEFAULT_BLOCK_SIZE = 40
DEFAULT_ASYMMETRY_STRENGTH = 0.2
DEFAULT_BOUND_MARGIN = 0.2
DEFAULT_OBJECTIVE_SCALE = 1.0
GENERATOR_VERSION = "1"

# name -> (raw minimization fn, lower, upper, optimum kind, optimum value)
# optimum kind "zeros"/"ones" selects the base optimum location.
_BASE_REGISTRY: dict[str, tuple] = {
    "Rastrigin": (rastrigin, -5.12, 5.12, "zeros", 0.0),
    "Ackley": (ackley, -32.768, 32.768, "zeros", 0.0),
    "Griewank": (griewank, -600.0, 600.0, "zeros", 0.0),
    "Zakharov": (zakharov, -5.0, 10.0, "zeros", 0.0),
    "Rosenbrock": (rosenbrock, -5.0, 10.0, "ones", 0.0),
    "Levy": (levy, -10.0, 10.0, "ones", 0.0),
    "Schwefel226": (
        schwefel_226, -500.0, 500.0, "schwefel", 0.0,
    ),
    "HighConditionedEllipsoid": (
        high_conditioned_ellipsoid, -5.0, 5.0, "zeros", 0.0,
    ),
}


def known_functions() -> tuple[str, ...]:
    return tuple(_BASE_REGISTRY.keys())


def base_optimum_x(name: str, dim: int) -> np.ndarray:
    kind = _BASE_REGISTRY[name][3]
    if kind == "zeros":
        return np.zeros(dim)
    if kind == "ones":
        return np.ones(dim)
    if kind == "schwefel":
        return np.full(dim, SCHWEFEL_226_OPTIMUM_X)
    raise ValueError(f"unknown base optimum kind: {kind!r}")


def _base_raw(name: str):
    return _BASE_REGISTRY[name][0]


def _frepr(value: float) -> str:
    """Full-precision stable string used inside hashes."""
    return repr(float(value))


def _freeze_array(arr: np.ndarray, expected_dim: int | None = None) -> np.ndarray:
    a = np.array(arr, dtype=float, copy=True)
    if expected_dim is not None and a.shape != (expected_dim,):
        raise ValueError(f"expected a ({expected_dim},) vector, got shape {a.shape}")
    a.setflags(write=False)
    return a


def _block_ranges(dim: int, block_size: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < dim:
        end = min(start + block_size, dim)
        ranges.append((start, end))
        start = end
    return ranges


def _orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Reproducible orthogonal matrix via QR with a canonical sign convention."""
    a = rng.standard_normal((n, n))
    q, r = np.linalg.qr(a)
    # Flip columns so the diagonal of r is positive -> unique Q for a given seed.
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def instance_seed(
    function_name: str, dimension: int, instance_id: int, stage: str
) -> int:
    """Stable 32-bit seed derived from the run key (independent of run order)."""
    key = f"{stage}:{function_name}:{dimension}:{instance_id}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


@dataclass(frozen=True)
class TransformSpec:
    dimension: int
    shift: np.ndarray
    permutation: np.ndarray
    rotation_blocks: tuple[np.ndarray, ...]
    rotation_mode: str
    block_size: int
    asymmetry_strength: float
    objective_scale: float

    def __post_init__(self) -> None:
        d = int(self.dimension)
        if self.rotation_mode not in ("full", "block"):
            raise ValueError(f"unknown rotation_mode: {self.rotation_mode!r}")
        object.__setattr__(self, "shift", _freeze_array(self.shift, d))
        object.__setattr__(
            self, "permutation", np.array(self.permutation, dtype=np.int64)
        )
        blocks = tuple(_freeze_array(b) for b in self.rotation_blocks)
        object.__setattr__(self, "rotation_blocks", blocks)
        object.__setattr__(self, "dimension", d)

    # --- transform pieces ---
    def _gamma(self) -> np.ndarray:
        d = self.dimension
        if d <= 1:
            return np.zeros(d)
        return self.asymmetry_strength * np.arange(d) / (d - 1)

    def _t_asym_forward(self, z: np.ndarray) -> np.ndarray:
        out = z.copy()
        pos = z > 0
        if pos.any():
            g = self._gamma()
            out[pos] = z[pos] ** (1.0 + g[pos])
        return out

    def _t_asym_inverse(self, w: np.ndarray) -> np.ndarray:
        out = w.copy()
        pos = w > 0
        if pos.any():
            g = self._gamma()
            out[pos] = w[pos] ** (1.0 / (1.0 + g[pos]))
        return out

    def _ranges(self) -> list[tuple[int, int]]:
        if self.rotation_mode == "full":
            return [(0, self.dimension)]
        return _block_ranges(self.dimension, self.block_size)

    def _rotate_forward(self, z: np.ndarray) -> np.ndarray:
        out = np.empty_like(z)
        for (start, end), block in zip(self._ranges(), self.rotation_blocks):
            out[start:end] = block @ z[start:end]
        return out

    def _rotate_inverse(self, z: np.ndarray) -> np.ndarray:
        out = np.empty_like(z)
        for (start, end), block in zip(self._ranges(), self.rotation_blocks):
            out[start:end] = block.T @ z[start:end]
        return out

    # --- public forward/inverse on the offset vector (centered at x_opt) ---
    def apply_forward(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        t = self._t_asym_forward(z)
        tp = t[self.permutation]
        return self._rotate_forward(tp)

    def apply_inverse(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        r = self._rotate_inverse(z)
        # ``r == t[permutation]`` after undoing the rotation.  Scatter it
        # back instead of computing ``argsort(permutation)`` for every
        # objective call: this preserves the inverse exactly and is O(d).
        ri = np.empty_like(r)
        ri[self.permutation] = r
        return self._t_asym_inverse(ri)

    def sha256(self) -> str:
        payload = {
            "dimension": self.dimension,
            "rotation_mode": self.rotation_mode,
            "block_size": self.block_size,
            "asymmetry_strength": _frepr(self.asymmetry_strength),
            "objective_scale": _frepr(self.objective_scale),
            "shift": [_frepr(v) for v in np.asarray(self.shift).ravel()],
            "permutation": [int(v) for v in np.asarray(self.permutation).ravel()],
            "rotation": [
                {
                    "shape": list(b.shape),
                    "data": [_frepr(v) for v in np.asarray(b).ravel()],
                }
                for b in self.rotation_blocks
            ],
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HighDimInstance:
    function_name: str
    dimension: int
    instance_id: int
    stage: str
    bounds_lower: np.ndarray
    bounds_upper: np.ndarray
    known_optimum_value: float
    known_optimum_x: np.ndarray
    transform_spec: TransformSpec
    generator_version: str = GENERATOR_VERSION

    def __post_init__(self) -> None:
        d = int(self.dimension)
        object.__setattr__(self, "dimension", d)
        object.__setattr__(self, "bounds_lower", _freeze_array(self.bounds_lower, d))
        object.__setattr__(self, "bounds_upper", _freeze_array(self.bounds_upper, d))
        object.__setattr__(self, "known_optimum_x", _freeze_array(self.known_optimum_x, d))

    def base_objective(self, y: np.ndarray) -> float:
        """Raw base function in minimization sense (no transform)."""
        return float(_base_raw(self.function_name)(np.asarray(y, dtype=float)))

    def objective(self, x: np.ndarray) -> float:
        """Instance objective (minimization); optimum-preserving transform."""
        x = np.asarray(x, dtype=float)
        offset = x - self.known_optimum_x
        y = base_optimum_x(self.function_name, self.dimension) + self.transform_spec.apply_inverse(offset)
        return float(self.transform_spec.objective_scale * self.base_objective(y))


def _build_rotation_blocks(
    dim: int, block_size: int, rng: np.random.Generator
) -> tuple[str, int, tuple[np.ndarray, ...]]:
    if dim <= FULL_ROTATION_DIM:
        return "full", dim, (_orthogonal(dim, rng),)
    blocks = tuple(
        _orthogonal(end - start, rng) for start, end in _block_ranges(dim, block_size)
    )
    return "block", block_size, blocks


def generate_instance(
    function_name: str,
    dimension: int,
    instance_id: int,
    *,
    stage: str = "development",
    block_size: int = DEFAULT_BLOCK_SIZE,
    asymmetry_strength: float = DEFAULT_ASYMMETRY_STRENGTH,
    objective_scale: float = DEFAULT_OBJECTIVE_SCALE,
    bound_margin: float = DEFAULT_BOUND_MARGIN,
    seed: int | None = None,
) -> HighDimInstance:
    """Construct one reproducible high-dimensional instance.

    When ``seed`` is ``None`` the seed is derived from the run key via
    :func:`instance_seed`, so the same ``(function, dim, instance_id, stage)``
    always reproduces the same transform.
    """
    if function_name not in _BASE_REGISTRY:
        raise ValueError(f"unknown function: {function_name!r}")
    _, lower, upper, _, base_value = _BASE_REGISTRY[function_name]
    dimension = int(dimension)
    if dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if seed is None:
        seed = instance_seed(function_name, dimension, instance_id, stage)
    rng = np.random.default_rng(seed)

    bounds_lower = np.full(dimension, lower, dtype=float)
    bounds_upper = np.full(dimension, upper, dtype=float)
    span = bounds_upper - bounds_lower
    x_opt = (
        bounds_lower
        + bound_margin * span
        + rng.uniform(size=dimension) * (1.0 - 2.0 * bound_margin) * span
    )
    permutation = rng.permutation(dimension)
    rotation_mode, blk, blocks = _build_rotation_blocks(dimension, block_size, rng)

    spec = TransformSpec(
        dimension=dimension,
        shift=x_opt,
        permutation=permutation,
        rotation_blocks=blocks,
        rotation_mode=rotation_mode,
        block_size=blk,
        asymmetry_strength=asymmetry_strength,
        objective_scale=objective_scale,
    )
    return HighDimInstance(
        function_name=function_name,
        dimension=dimension,
        instance_id=instance_id,
        stage=stage,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        known_optimum_value=float(objective_scale * base_value),
        known_optimum_x=x_opt,
        transform_spec=spec,
    )


# --- artifact IO (language-neutral csv.gz + metadata.json) ---
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_matrix_gz(path: Path, matrix: np.ndarray, fmt: str = "%.17g") -> None:
    matrix = np.asarray(matrix)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    with gzip.open(path, "wt") as handle:
        np.savetxt(handle, matrix, delimiter=",", fmt=fmt)


def _read_matrix_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rt") as handle:
        return np.loadtxt(handle, delimiter=",")


def write_instance_artifacts(
    instance: HighDimInstance, starts: np.ndarray, out_dir: str | Path,
    *, extra_starts: dict | None = None,
) -> dict:
    """Persist an instance + shared starts; return metadata with file hashes.

    ``extra_starts`` maps n_starts tiers (e.g. ``{16: matrix}``) to additional
    start matrices written as ``starts_n{N}.csv.gz`` with their own hashes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = instance.transform_spec

    shift_path = out_dir / "shift.csv.gz"
    _write_matrix_gz(shift_path, spec.shift)

    perm_path = out_dir / "permutation.csv.gz"
    _write_matrix_gz(perm_path, spec.permutation, fmt="%d")

    # Rotation in long form: block_start, block_size, local_row, local_col, value.
    rot_rows: list[tuple[int, int, int, int, float]] = []
    for (start, _), block in zip(spec._ranges(), spec.rotation_blocks):
        block = np.asarray(block)
        local_size = block.shape[0]
        for row in range(block.shape[0]):
            for col in range(block.shape[1]):
                rot_rows.append((start, local_size, row, col, float(block[row, col])))
    rot_path = out_dir / "rotation_blocks.csv.gz"
    with gzip.open(rot_path, "wt") as handle:
        np.savetxt(
            handle,
            np.asarray(rot_rows, dtype=float).reshape(-1, 5),
            delimiter=",",
            fmt=("%d", "%d", "%d", "%d", "%.17g"),
        )

    starts = np.asarray(starts, dtype=float)
    starts_path = out_dir / "starts.csv.gz"
    _write_matrix_gz(starts_path, starts)

    extra_starts_meta: dict = {}
    if extra_starts:
        for n_starts_tier, tier_starts in extra_starts.items():
            n_starts_tier = int(n_starts_tier)
            tier_starts = np.asarray(tier_starts, dtype=float)
            if tier_starts.shape[0] != n_starts_tier:
                raise ValueError(
                    f"extra_starts[{n_starts_tier}] has {tier_starts.shape[0]} rows"
                )
            if tier_starts.shape[1] != instance.dimension:
                raise ValueError(
                    f"extra_starts[{n_starts_tier}] has {tier_starts.shape[1]} cols, "
                    f"expected dimension {instance.dimension}"
                )
            tier_path = out_dir / f"starts_n{n_starts_tier}.csv.gz"
            _write_matrix_gz(tier_path, tier_starts)
            extra_starts_meta[str(n_starts_tier)] = {
                "file": tier_path.name,
                "hash": _sha256_file(tier_path),
                "n_starts": n_starts_tier,
            }

    metadata = {
        "generator_version": instance.generator_version,
        "function_name": instance.function_name,
        "dimension": instance.dimension,
        "instance_id": instance.instance_id,
        "stage": instance.stage,
        "block_size": spec.block_size,
        "rotation_mode": spec.rotation_mode,
        "asymmetry_strength": spec.asymmetry_strength,
        "objective_scale": spec.objective_scale,
        "full_rotation_dim": FULL_ROTATION_DIM,
        "bounds_lower": [_frepr(v) for v in instance.bounds_lower],
        "bounds_upper": [_frepr(v) for v in instance.bounds_upper],
        "known_optimum_value": _frepr(instance.known_optimum_value),
        "known_optimum_x": [_frepr(v) for v in instance.known_optimum_x],
        "n_starts": int(starts.shape[0]),
        "file_hashes": {
            "shift": _sha256_file(shift_path),
            "permutation": _sha256_file(perm_path),
            "rotation_blocks": _sha256_file(rot_path),
            "starts": _sha256_file(starts_path),
        },
        "transform_sha256": spec.sha256(),
        "extra_starts": extra_starts_meta,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    return metadata


def load_instance(artifact_dir: str | Path) -> HighDimInstance:
    """Rebuild an instance from its artifacts, verifying the transform hash."""
    artifact_dir = Path(artifact_dir)
    metadata = json.loads((artifact_dir / "metadata.json").read_text())

    shift = _read_matrix_gz(artifact_dir / "shift.csv.gz").ravel()
    permutation = _read_matrix_gz(artifact_dir / "permutation.csv.gz").ravel().astype(np.int64)
    rot = _read_matrix_gz(artifact_dir / "rotation_blocks.csv.gz").reshape(-1, 5)

    blocks: list[np.ndarray] = []
    starts_col = rot[:, 0].astype(int)
    for start in sorted(set(starts_col.tolist())):
        rows = rot[starts_col == start]
        local_size = int(rows[0, 1])
        block = np.zeros((local_size, local_size))
        for entry in rows:
            _, _, row, col, value = entry
            block[int(row), int(col)] = value
        blocks.append(block)

    spec = TransformSpec(
        dimension=int(metadata["dimension"]),
        shift=shift,
        permutation=permutation,
        rotation_blocks=tuple(blocks),
        rotation_mode=metadata["rotation_mode"],
        block_size=int(metadata["block_size"]),
        asymmetry_strength=float(metadata["asymmetry_strength"]),
        objective_scale=float(metadata["objective_scale"]),
    )
    if spec.sha256() != metadata.get("transform_sha256"):
        raise ValueError(
            "transform_sha256 mismatch: artifact corrupted or generated by a "
            "different highdim_instances version"
        )

    return HighDimInstance(
        function_name=metadata["function_name"],
        dimension=int(metadata["dimension"]),
        instance_id=int(metadata["instance_id"]),
        stage=metadata["stage"],
        bounds_lower=np.asarray(metadata["bounds_lower"], dtype=float),
        bounds_upper=np.asarray(metadata["bounds_upper"], dtype=float),
        known_optimum_value=float(metadata["known_optimum_value"]),
        known_optimum_x=np.asarray(metadata["known_optimum_x"], dtype=float),
        transform_spec=spec,
        generator_version=metadata.get("generator_version", GENERATOR_VERSION),
    )


def starts_filename(artifact_dir: str | Path, n_starts: int = 8) -> str:
    """Resolve the starts artifact filename for the requested n_starts tier.

    The default tier (metadata ``n_starts``) lives in ``starts.csv.gz``; every
    other tier lives in ``starts_n{N}.csv.gz``.
    """
    artifact_dir = Path(artifact_dir)
    default_n = 8
    metadata_path = artifact_dir / "metadata.json"
    if metadata_path.exists():
        try:
            default_n = int(json.loads(metadata_path.read_text()).get("n_starts", 8))
        except Exception:
            default_n = 8
    if int(n_starts) == default_n:
        return "starts.csv.gz"
    return f"starts_n{int(n_starts)}.csv.gz"


def load_starts(artifact_dir: str | Path, n_starts: int | None = None) -> np.ndarray:
    """Load the starts matrix for a tier.

    ``n_starts=None`` → the default tier (``starts.csv.gz``); a specific n →
    ``starts_n{N}.csv.gz`` (or ``starts.csv.gz`` if it is the default tier).
    """
    artifact_dir = Path(artifact_dir)
    if n_starts is None:
        path = artifact_dir / "starts.csv.gz"
    else:
        path = artifact_dir / starts_filename(artifact_dir, n_starts)
    if not path.exists():
        raise FileNotFoundError(
            f"no starts artifact for n_starts={n_starts} in {artifact_dir}"
        )
    return _read_matrix_gz(path)


__all__ = [
    "FULL_ROTATION_DIM",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_ASYMMETRY_STRENGTH",
    "DEFAULT_BOUND_MARGIN",
    "DEFAULT_OBJECTIVE_SCALE",
    "GENERATOR_VERSION",
    "TransformSpec",
    "HighDimInstance",
    "known_functions",
    "base_optimum_x",
    "instance_seed",
    "generate_instance",
    "write_instance_artifacts",
    "load_instance",
    "load_starts",
    "starts_filename",
]
