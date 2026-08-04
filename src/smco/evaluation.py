"""Exact objective-evaluation budget for SMCO (Task 1 of the paper campaign).

``EvaluationContext`` is the single counting and hard-budget authority. It is
threaded through the optimizer internals as an optional ``ctx`` argument:

* ``ctx is None``  -> every evaluation goes through the raw objective ``f``,
  so behaviour is byte-for-byte identical to the pre-budget optimizer.
* ``ctx`` is set    -> every objective call is counted, attributed to an event,
  checked against a hard ``max_evals`` cap, and folded into a best-so-far trace
  plus target-hit FE records.

Atomic multi-evaluation steps (a center-difference iteration is ``2d`` partial
evaluations plus one ``iterate`` evaluation) are pre-checked with
:meth:`can_evaluate` *before* they start, so a budget exhaustion never leaves a
half-finished coordinate update. The final ``fe_used`` therefore never exceeds
``fe_budget``.

Event names mirror ``paper_contract.EVENTS`` and the R side
(``evaluation_budget.R``); do not rename them in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from .paper_contract import EVENTS

Objective = Callable[[np.ndarray], float]

# Default minimization-gap targets recorded as target-hit FE (contract 6.1).
DEFAULT_GAP_TARGETS: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-5)

# Canonical CSV suffixes for the default targets (paper_contract.RESULT_COLUMNS).
_GAP_LABEL_BY_VALUE: dict[float, str] = {
    1e-1: "1e-1",
    1e-2: "1e-2",
    1e-3: "1e-3",
    1e-5: "1e-5",
}


def _gap_target_label(value: float) -> str:
    """Canonical CSV suffix for a gap target; matches RESULT_COLUMNS naming."""
    return _GAP_LABEL_BY_VALUE.get(value, f"{value:g}")


class EvaluationBudgetExceeded(RuntimeError):
    """Raised when an evaluation is attempted beyond the hard ``max_evals`` cap.

    Callers should pre-check with :meth:`EvaluationContext.can_evaluate` so that
    a clean ``termination_reason="evaluation_budget"`` stop happens instead of a
    mid-step raise. This exception is a defensive backstop.
    """


@dataclass(frozen=True)
class EvaluationRecord:
    fe_used: int
    event: str
    value: float
    best_value: float


def _is_better(candidate: float, current: float | None, maximize: bool) -> bool:
    if current is None:
        return True
    return candidate > current if maximize else candidate < current


class EvaluationContext:
    """Counts objective evaluations and enforces a hard FE budget.

    The context is shared across all trajectories of one run (multi-start and
    EVO share one global FE pool). Two derived views are supported:

    * :meth:`scoped`  -> a view that re-tags every evaluation to one event
      (used for the refine phase / boosted branch). It shares the parent's
      mutable counter, best, targets and trace.
    * :meth:`split`   -> an independent context with its own counter and cap
      (used to give the regular and boosted EVO branches each 50% of the run
      budget per experiment-plan section 4.1).
    """

    def __init__(
        self,
        f: Objective,
        *,
        max_evals: int | None = None,
        objective_sense: str = "maximize",
        known_optimum: float | None = None,
        gap_targets: Iterable[float] = DEFAULT_GAP_TARGETS,
        record_trace: bool = False,
        record_evaluations: bool = False,
    ) -> None:
        self._f = f
        self._max_evals = max_evals
        if objective_sense not in ("maximize", "minimize"):
            raise ValueError("objective_sense must be 'maximize' or 'minimize'")
        self._sense = objective_sense
        self._maximize = objective_sense == "maximize"
        self._known_optimum = known_optimum
        self._gap_targets = tuple(gap_targets)
        self._target_labels = tuple(_gap_target_label(t) for t in self._gap_targets)
        self._record_trace = bool(record_trace)
        self._record_evaluations = bool(record_evaluations)
        # Owned mutable state (scoped views route through ``_parent``).
        self._evaluations = 0
        self._counts: dict[str, int] = {event: 0 for event in EVENTS}
        self._best_value: float | None = None
        self._best_point: np.ndarray | None = None
        self._target_hit: dict[str, int | None] = {lbl: None for lbl in self._target_labels}
        self._trace: list[EvaluationRecord] = []
        self._termination_reason: str | None = None
        self._event_override: str | None = None
        self._parent: EvaluationContext | None = None

    # ------------------------------------------------------------------
    # Properties / accessors (route through parent for scoped views).
    # ------------------------------------------------------------------
    @property
    def max_evals(self) -> int | None:
        return self._max_evals

    @property
    def objective_sense(self) -> str:
        return self._sense

    @property
    def evaluations(self) -> int:
        return self._parent._evaluations if self._parent is not None else self._evaluations

    @property
    def best_value(self) -> float | None:
        return self._parent._best_value if self._parent is not None else self._best_value

    @property
    def best_point(self) -> np.ndarray | None:
        return self._parent._best_point if self._parent is not None else self._best_point

    @property
    def termination_reason(self) -> str | None:
        return self._parent._termination_reason if self._parent is not None else self._termination_reason

    @termination_reason.setter
    def termination_reason(self, value: str | None) -> None:
        if self._parent is not None:
            self._parent._termination_reason = value
        else:
            self._termination_reason = value

    @property
    def event_override(self) -> str | None:
        return self._event_override

    # ------------------------------------------------------------------
    # Budget control.
    # ------------------------------------------------------------------
    def can_evaluate(self, count: int = 1) -> bool:
        """Return whether ``count`` more evaluations fit inside the cap."""
        if count < 0:
            raise ValueError("count must be non-negative")
        cap = self._max_evals
        if cap is None:
            return True
        return self.evaluations + count <= cap

    def require(self, count: int = 1) -> None:
        """Pre-check an atomic step; raise if it cannot fit."""
        if not self.can_evaluate(count):
            self.termination_reason = "evaluation_budget"
            raise EvaluationBudgetExceeded(
                f"need {count} evaluations but only {self._remaining()} remain"
            )

    def _remaining(self) -> int:
        cap = self._max_evals
        if cap is None:
            return -1
        return max(0, cap - self.evaluations)

    # ------------------------------------------------------------------
    # The single evaluation entry point.
    # ------------------------------------------------------------------
    def evaluate(self, x: np.ndarray, *, event: str = "iterate") -> float:
        """Evaluate the objective at ``x``, count it, and update best/trace.

        The hard cap is enforced here as a backstop so ``fe_used`` can never
        exceed ``fe_budget`` even if a caller forgets to pre-check.
        """
        if event not in EVENTS:
            raise ValueError(f"unknown evaluation event: {event!r}")
        effective_event = self._event_override or event
        cap = self._max_evals
        used = self.evaluations
        if cap is not None and used >= cap:
            self.termination_reason = "evaluation_budget"
            raise EvaluationBudgetExceeded(
                f"evaluation cap {cap} reached (event={effective_event!r})"
            )
        value = float(self._f(np.asarray(x, dtype=float)))
        self._record(effective_event, value, x)
        return value

    def _record(self, event: str, value: float, x: np.ndarray) -> None:
        """Mutate the owning context's state (shared with scoped views)."""
        owner = self._parent if self._parent is not None else self
        owner._evaluations += 1
        owner._counts[event] = owner._counts.get(event, 0) + 1
        if _is_better(value, owner._best_value, owner._maximize):
            owner._best_value = float(value)
            owner._best_point = np.array(x, dtype=float, copy=True)
        # Target-hit FE (absolute gap to known optimum; sense-agnostic distance).
        if owner._known_optimum is not None:
            gap = abs(value - owner._known_optimum)
            for target, label in zip(owner._gap_targets, owner._target_labels):
                if owner._target_hit[label] is None and gap <= target:
                    owner._target_hit[label] = owner._evaluations
        if owner._record_trace or owner._record_evaluations:
            owner._trace.append(
                EvaluationRecord(
                    fe_used=owner._evaluations,
                    event=event,
                    value=float(value),
                    best_value=float(owner._best_value),
                )
            )

    # ------------------------------------------------------------------
    # Derived views.
    # ------------------------------------------------------------------
    def scoped(self, event: str) -> "EvaluationContext":
        """Share the parent's counter/state but tag every eval as ``event``."""
        if event not in EVENTS:
            raise ValueError(f"unknown evaluation event: {event!r}")
        view = EvaluationContext(
            self._f,
            max_evals=self._max_evals,
            objective_sense=self._sense,
            known_optimum=self._known_optimum,
            gap_targets=self._gap_targets,
            record_trace=self._record_trace,
            record_evaluations=self._record_evaluations,
        )
        owner = self._parent if self._parent is not None else self
        view._parent = owner
        view._event_override = event
        return view

    def split(
        self,
        *,
        count: int | None = None,
        fraction: float | None = None,
    ) -> "EvaluationContext":
        """Independent context with its own counter and cap (BR 50/50 split).

        Exactly one of ``count`` / ``fraction`` must be given. The child owns its
        own best/trace/targets; the caller aggregates branch summaries.
        """
        if (count is None) == (fraction is None):
            raise ValueError("specify exactly one of count or fraction")
        if self._max_evals is None and (fraction is not None or count is None):
            # No parent cap: child is uncapped too (caller controls budget).
            child_cap = None if count is None else int(count)
        elif count is not None:
            child_cap = int(count)
        else:
            assert fraction is not None
            child_cap = int(self._max_evals * float(fraction))
        child = EvaluationContext(
            self._f,
            max_evals=child_cap,
            objective_sense=self._sense,
            known_optimum=self._known_optimum,
            gap_targets=self._gap_targets,
            record_trace=self._record_trace,
            record_evaluations=self._record_evaluations,
        )
        return child

    # ------------------------------------------------------------------
    # Reporting.
    # ------------------------------------------------------------------
    def evaluation_counts(self) -> dict[str, int]:
        owner = self._parent if self._parent is not None else self
        return {event: int(owner._counts.get(event, 0)) for event in EVENTS}

    def target_hit_evaluations(self) -> dict[str, int | None]:
        owner = self._parent if self._parent is not None else self
        return {f"target_hit_fe_{lbl}": owner._target_hit[lbl] for lbl in owner._target_labels}

    def best_so_far_trace(self) -> list[EvaluationRecord]:
        owner = self._parent if self._parent is not None else self
        return list(owner._trace)

    def summary(self) -> dict[str, Any]:
        return {
            "fe_budget": self._max_evals,
            "fe_used": int(self.evaluations),
            "termination_reason": self.termination_reason,
            "evaluation_counts_by_event": self.evaluation_counts(),
            "best_value": self.best_value,
            "target_hit_evaluations": self.target_hit_evaluations(),
        }


def evaluate_with_context(
    ctx: EvaluationContext | None,
    f: Objective,
    x: np.ndarray,
    *,
    event: str = "iterate",
) -> float:
    """Evaluate ``f(x)`` through ``ctx`` when present, else fall back to ``f``.

    This is the helper used at every objective-evaluation site inside
    ``optimizer.py`` so that the ``ctx is None`` path stays on the raw objective
    (zero behaviour change) while the budget path routes through the context.
    """
    if ctx is None:
        return float(f(x))
    return ctx.evaluate(x, event=event)


__all__ = [
    "DEFAULT_GAP_TARGETS",
    "EvaluationBudgetExceeded",
    "EvaluationContext",
    "EvaluationRecord",
    "evaluate_with_context",
]
