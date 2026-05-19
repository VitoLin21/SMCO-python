# SMCO-HB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first-version `smco_hb(...)` multi-fidelity optimizer that combines SMCO-based configuration proposal with synchronous Hyperband resource allocation and fixed-size multi-worker execution.

**Architecture:** Add a new `smco.multifidelity` module that owns result types, scheduler helpers, history storage, a runner with synchronous parallel execution, and a history-driven SMCO proposer. Keep the current `optimizer.py` APIs unchanged and expose the new entry point through `smco.__init__`.

**Tech Stack:** Python 3.12, `dataclasses`, `numpy`, `concurrent.futures`, existing `smco` optimizer variants, `pytest`

---

### Task 1: Add failing API and scheduler/history tests

**Files:**
- Create: `tests/test_multifidelity.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from smco import smco_hb
from smco.multifidelity import HyperbandBracket, HyperbandTrial, build_hyperband_brackets


def test_build_hyperband_brackets_produces_expected_budgets():
    brackets = build_hyperband_brackets(min_budget=1, max_budget=9, eta=3)

    assert [bracket.budgets for bracket in brackets] == [
        [9.0],
        [3.0, 9.0],
        [1.0, 3.0, 9.0],
    ]


def test_smco_hb_rejects_non_numeric_config_space():
    with pytest.raises(ValueError, match="continuous numeric"):
        smco_hb(
            suggest_variant="smco",
            config_space={"choices": ["a", "b"]},
            evaluate=lambda config, budget, **_: {"score": 0.0},
            min_budget=1,
            max_budget=3,
            eta=3,
            seed=1,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_multifidelity.py::test_build_hyperband_brackets_produces_expected_budgets -v`
Expected: FAIL with import error because `smco_hb` and `build_hyperband_brackets` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create the module and symbols with the smallest implementation that lets bracket construction exist and config-space validation fail cleanly.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_multifidelity.py::test_build_hyperband_brackets_produces_expected_budgets tests/test_multifidelity.py::test_smco_hb_rejects_non_numeric_config_space -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_multifidelity.py src/smco/multifidelity.py src/smco/__init__.py
git commit -m "feat: scaffold SMCO-HB API and bracket builder"
```

### Task 2: Add failing runner/history/checkpoint tests

**Files:**
- Modify: `tests/test_multifidelity.py`
- Modify: `src/smco/multifidelity.py`

- [ ] **Step 1: Write the failing test**

```python
def test_smco_hb_promotes_configs_and_reuses_checkpoint_when_available():
    seen = []

    def evaluate(config, budget, seed=None, checkpoint=None):
        seen.append((float(config[0]), float(budget), checkpoint))
        return {
            "score": 1.0 - abs(float(config[0]) - 0.25) + 0.01 * float(budget),
            "checkpoint": f"ckpt@{float(config[0]):.3f}:{float(budget):.1f}",
        }

    result = smco_hb(
        suggest_variant="smco",
        config_space=([-1.0], [1.0]),
        evaluate=evaluate,
        min_budget=1,
        max_budget=3,
        eta=3,
        seed=7,
        n_workers=1,
        proposer_options={"pool_size": 3, "iter_max": 20},
    )

    assert result.incumbent_budget == 3.0
    assert any(budget == 3.0 and checkpoint is not None for _, budget, checkpoint in seen)
    assert any(trial.promoted for trial in result.history)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_multifidelity.py::test_smco_hb_promotes_configs_and_reuses_checkpoint_when_available -v`
Expected: FAIL because promotion/history/checkpoint logic is not implemented yet.

- [ ] **Step 3: Write minimal implementation**

Implement trial/history/result dataclasses, rung ranking, checkpoint lookup, and promotion wiring in `smco_hb(...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_multifidelity.py::test_smco_hb_promotes_configs_and_reuses_checkpoint_when_available -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_multifidelity.py src/smco/multifidelity.py
git commit -m "feat: add SMCO-HB history and promotion flow"
```

### Task 3: Add failing synchronous parallel execution test

**Files:**
- Modify: `tests/test_multifidelity.py`
- Modify: `src/smco/multifidelity.py`

- [ ] **Step 1: Write the failing test**

```python
import threading
import time


def test_smco_hb_uses_multiple_workers_within_a_rung():
    active = 0
    max_active = 0
    lock = threading.Lock()

    def evaluate(config, budget, seed=None, checkpoint=None):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {"score": 1.0 - abs(float(config[0]))}

    smco_hb(
        suggest_variant="smco",
        config_space=([-1.0], [1.0]),
        evaluate=evaluate,
        min_budget=1,
        max_budget=1,
        eta=3,
        seed=9,
        n_workers=2,
        proposer_options={"pool_size": 4, "iter_max": 10},
    )

    assert max_active >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_multifidelity.py::test_smco_hb_uses_multiple_workers_within_a_rung -v`
Expected: FAIL because execution is still sequential.

- [ ] **Step 3: Write minimal implementation**

Add a synchronous `ThreadPoolExecutor` path for per-rung trial execution when `n_workers > 1`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_multifidelity.py::test_smco_hb_uses_multiple_workers_within_a_rung -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_multifidelity.py src/smco/multifidelity.py
git commit -m "feat: add synchronous parallel SMCO-HB runner"
```

### Task 4: Add failing proposer/export smoke tests

**Files:**
- Modify: `tests/test_multifidelity.py`
- Modify: `tests/test_optimizer.py`
- Modify: `src/smco/multifidelity.py`
- Modify: `src/smco/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
def test_smco_hb_finds_reasonable_incumbent_on_budgeted_quadratic():
    def evaluate(config, budget, seed=None, checkpoint=None):
        x = float(config[0])
        return {"score": 1.0 - (x - 0.2) ** 2 - 0.2 / float(budget)}

    result = smco_hb(
        suggest_variant="smco_r",
        config_space=([-1.0], [1.0]),
        evaluate=evaluate,
        min_budget=1,
        max_budget=9,
        eta=3,
        seed=3,
        n_workers=2,
        proposer_options={"pool_size": 9, "iter_max": 25},
    )

    assert result.incumbent_budget == 9.0
    assert abs(float(result.incumbent_config[0]) - 0.2) < 0.35
    assert result.metadata["suggest_variant"] == "smco_r"


def test_public_api_exports_smco_hb():
    from smco import smco_hb

    assert callable(smco_hb)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_multifidelity.py::test_smco_hb_finds_reasonable_incumbent_on_budgeted_quadratic tests/test_optimizer.py::test_public_api_exports_smco_hb -v`
Expected: FAIL because proposer quality or export wiring is incomplete.

- [ ] **Step 3: Write minimal implementation**

Add history-driven proposer behavior using the existing SMCO variants over a lightweight surrogate objective, and export `smco_hb` from `smco.__init__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_multifidelity.py tests/test_optimizer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_multifidelity.py tests/test_optimizer.py src/smco/multifidelity.py src/smco/__init__.py
git commit -m "feat: add SMCO-HB proposer and public export"
```

### Task 5: Final verification and documentation sweep

**Files:**
- Modify: `docs/usage.md`
- Modify: `docs/superpowers/specs/2026-05-19-smco-hb-design.md` (only if implementation forced a documented API change)
- Modify: `src/smco/multifidelity.py`
- Modify: `tests/test_multifidelity.py`

- [ ] **Step 1: Update docs for the implemented API**

Add one `smco_hb(...)` usage example and mention first-version constraints:

```python
from smco import smco_hb

run = smco_hb(
    suggest_variant="smco_r",
    config_space=([-5.0, -5.0], [5.0, 5.0]),
    evaluate=my_budgeted_train_fn,
    min_budget=1,
    max_budget=9,
    eta=3,
    n_workers=4,
    seed=123,
)
```

- [ ] **Step 2: Run the focused tests**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_multifidelity.py tests/test_optimizer.py tests/test_benchmark.py -v`
Expected: PASS

- [ ] **Step 3: Run the full test suite**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add docs/usage.md src/smco/multifidelity.py tests/test_multifidelity.py tests/test_optimizer.py
git commit -m "docs: document SMCO-HB API"
```

## Self-Review

- Spec coverage: proposer variants, stateless promotion, optional checkpoint reuse, synchronous multi-worker execution, result object, and continuous-only validation are each covered by Tasks 1-5.
- Placeholder scan: no `TODO`, `TBD`, or undefined task references remain.
- Type consistency: `smco_hb`, `SMCOHBResult`, bracket builder, and history objects use consistent names across tasks.
