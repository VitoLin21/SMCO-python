"""SMCO-EVO high-dim paper result contract (single Python source of truth).

Mirrors ``docs/smco-evo-result-contract.md``. The R side
(``vendor/SMCO_R/main/contract.R``) must reproduce these enums and hash recipes
byte-for-byte so that Python, R and the merge script agree on ``run_id`` and
``configuration_hash`` for the same task.

Keep this module dependency-free (stdlib only) so it can be imported from
runners, tests and the merge tool without pulling NumPy.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "1"

# --- enum tuples (authoritative; mirror in contract.R and CSV headers) ---
LANGUAGES = ("python", "r")
STATE_SEMANTICS = ("state_preserving", "restart")
FAMILIES = ("smco", "smco_refine", "smco_boost_refine")
STRATEGIES = ("rand1bin", "current-to-best1bin", "best1bin", "sobol")
OBJECTIVE_SENSES = ("minimize", "maximize")
STATUSES = ("success", "algorithm_failure", "infra_failure", "timeout")
TERMINATION_REASONS = (
    "iteration_limit",
    "evaluation_budget",
    "boundary_budget",
    "convergence",
    "clip_stopped",
    "error",
)
STAGES = (
    "e0_contract",
    "e1_development",
    "e1b_baseline_selection",
    "e2_factorial_highdim",
    "e3_baselines_highdim",
    "e4_bbob_largescale",
    "e5_lowdim_check",
    "e6_ablations",
)
SUITES = ("synthetic_highdim", "bbob_largescale", "bbob", "contract")

# Function-evaluation event names; must match evaluation.py / evaluation_budget.R.
EVENTS = (
    "initialization",
    "finite_difference",
    "iterate",
    "replacement_initialization",
    "restart_initialization",
    "refine",
    "boost",
    "clip_recheck",
)

NONE_TOKEN = "none"

# --- algorithm_id translation tables ---
_FAMILY_TO_TOKEN = {
    "smco": "SMCO",
    "smco_refine": "SMCO-REFINE",
    "smco_boost_refine": "SMCO-BOOST-REFINE",
}
_TOKEN_TO_FAMILY = {v: k for k, v in _FAMILY_TO_TOKEN.items()}
_LANG_TO_CODE = {"python": "PY", "r": "R"}
_CODE_TO_LANG = {v: k for k, v in _LANG_TO_CODE.items()}
_SEM_TO_SLOT = {"state_preserving": "SP", "restart": "RS"}
_SLOT_TO_SEM = {v: k for k, v in _SEM_TO_SLOT.items()}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value == "true":
            return True
        if value == "false":
            return False
    raise ValueError(f"cannot interpret evolutionary value: {value!r}")


def build_algorithm_id(
    language: str,
    family: str,
    evolutionary: Any,
    state_semantics: str | None = None,
) -> str:
    """Build the canonical algorithm_id per contract section 2."""
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of {LANGUAGES}, got {language!r}")
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    lang = _LANG_TO_CODE[language]
    fam = _FAMILY_TO_TOKEN[family]
    if _as_bool(evolutionary):
        if state_semantics not in STATE_SEMANTICS:
            raise ValueError(
                f"state_semantics required for EVO, got {state_semantics!r}"
            )
        return f"{lang}-{_SEM_TO_SLOT[state_semantics]}-{fam}-EVO"
    return f"{lang}-BASE-{fam}"


def parse_algorithm_id(algorithm_id: str) -> dict[str, Any]:
    """Reverse of :func:`build_algorithm_id`; used for provenance audit."""
    s = algorithm_id
    evolutionary = False
    if s.endswith("-EVO"):
        evolutionary = True
        s = s[: -len("-EVO")]
    parts = s.split("-")
    if len(parts) < 3:
        raise ValueError(f"malformed algorithm_id: {algorithm_id!r}")
    lang_code, slot = parts[0], parts[1]
    fam_token = "-".join(parts[2:])
    if lang_code not in _CODE_TO_LANG:
        raise ValueError(f"unknown language code in {algorithm_id!r}")
    if evolutionary:
        if slot not in _SLOT_TO_SEM:
            raise ValueError(f"EVO algorithm_id missing SP/RS slot: {algorithm_id!r}")
        state_semantics = _SLOT_TO_SEM[slot]
    else:
        if slot != "BASE":
            raise ValueError(f"non-EVO algorithm_id must use BASE slot: {algorithm_id!r}")
        state_semantics = NONE_TOKEN
    if fam_token not in _TOKEN_TO_FAMILY:
        raise ValueError(f"unknown family token in {algorithm_id!r}")
    return {
        "language": _CODE_TO_LANG[lang_code],
        "family": _TOKEN_TO_FAMILY[fam_token],
        "evolutionary": evolutionary,
        "state_semantics": state_semantics,
    }


# --- hashing helpers ---
def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no spaces, UTF-8 preserved."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalize_scalar(value: Any) -> str:
    """Normalize a task/config scalar to a hash-stable string."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return repr(value)
    return str(value)


_RUN_ID_KEYS = (
    "stage",
    "suite",
    "function",
    "dimension",
    "instance",
    "replication",
    "algorithm_id",
    "evolution_strategy",
    "seed",
    "fe_budget",
    "n_starts",
    "configuration_hash",
)


def compute_run_id(task: Mapping[str, Any]) -> str:
    """``run_id = 'r' + sha256(canonical_json(task_subset))[:16]`` (contract 3)."""
    canonical = {k: _normalize_scalar(task.get(k)) for k in _RUN_ID_KEYS}
    digest = hashlib.sha256(canonical_json(canonical).encode("utf-8")).hexdigest()
    return "r" + digest[:16]


def format_cfg_float(value: Any) -> str:
    """Two-decimal string used inside configuration_hash (contract 4)."""
    return f"{float(value):.2f}"


def compute_configuration_hash(config: Mapping[str, Any]) -> str:
    """``sha256(canonical_json(config))[:16]``; floats must be pre-stringified."""
    digest = hashlib.sha256(canonical_json(dict(config)).encode("utf-8")).hexdigest()
    return digest[:16]


# --- CSV column order (contract section 5) ---
RESULT_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "manifest_id",
    "stage",
    "suite",
    "function",
    "dimension",
    "instance",
    "replication",
    "seed",
    "language",
    "state_semantics",
    "family",
    "evolutionary",
    "evolution_strategy",
    "algorithm_id",
    "n_starts",
    "fe_budget",
    "fe_used",
    "checkpoint_fe",
    "best_value",
    "known_optimum",
    "normalized_gap",
    "objective_sense",
    "target_hit_fe_1e-1",
    "target_hit_fe_1e-2",
    "target_hit_fe_1e-3",
    "target_hit_fe_1e-5",
    "wall_time_sec",
    "peak_memory_mb",
    "status",
    "failure_reason",
    "is_confirmatory",
    "supersedes_run_id",
    "machine_id",
    "git_commit",
    "environment_hash",
    "start_points_hash",
    "instance_hash",
    "configuration_hash",
    "run_id",
    "termination_reason",
    "fe_counts_by_event",
)


def validate_result_row(row: Mapping[str, Any]) -> list[str]:
    """Return a list of contract violations for a result row (empty == ok)."""
    errors: list[str] = []
    for col in RESULT_COLUMNS:
        if col not in row:
            errors.append(f"missing column: {col}")
    if errors:
        return errors
    if row["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if row["language"] not in LANGUAGES:
        errors.append(f"language not in {LANGUAGES}")
    if row["family"] not in FAMILIES:
        errors.append(f"family not in {FAMILIES}")
    if row["evolutionary"] not in ("true", "false"):
        errors.append("evolutionary must be 'true'/'false' string")
    if row["status"] not in STATUSES:
        errors.append(f"status not in {STATUSES}")
    if row["objective_sense"] not in OBJECTIVE_SENSES:
        errors.append(f"objective_sense not in {OBJECTIVE_SENSES}")
    if row["stage"] not in STAGES:
        errors.append(f"stage not in {STAGES}")
    if row["evolutionary"] == "true":
        if row["state_semantics"] not in STATE_SEMANTICS:
            errors.append("EVO row needs state_semantics")
        if row["evolution_strategy"] not in STRATEGIES:
            errors.append(f"evolution_strategy not in {STRATEGIES}")
    else:
        if row["state_semantics"] != NONE_TOKEN:
            errors.append("base row state_semantics must be 'none'")
        if row["evolution_strategy"] != NONE_TOKEN:
            errors.append("base row evolution_strategy must be 'none'")
    try:
        rebuilt = build_algorithm_id(
            row["language"],
            row["family"],
            row["evolutionary"],
            None if row["state_semantics"] == NONE_TOKEN else row["state_semantics"],
        )
    except ValueError as exc:
        errors.append(f"algorithm_id cannot be rebuilt: {exc}")
    else:
        if rebuilt != row["algorithm_id"]:
            errors.append(
                f"algorithm_id mismatch: {row['algorithm_id']!r} vs {rebuilt!r}"
            )
    return errors


__all__ = [
    "SCHEMA_VERSION",
    "LANGUAGES",
    "STATE_SEMANTICS",
    "FAMILIES",
    "STRATEGIES",
    "OBJECTIVE_SENSES",
    "STATUSES",
    "TERMINATION_REASONS",
    "STAGES",
    "SUITES",
    "EVENTS",
    "NONE_TOKEN",
    "RESULT_COLUMNS",
    "build_algorithm_id",
    "parse_algorithm_id",
    "canonical_json",
    "compute_run_id",
    "compute_configuration_hash",
    "format_cfg_float",
    "validate_result_row",
]
