# SMCO-HB Multi-Fidelity HPO Design

## Goal

Design a first-version multi-fidelity HPO algorithm family that combines the existing SMCO proposers (`smco`, `smco_r`, `smco_br`) with Hyperband-style resource allocation for training-resource fidelity such as epochs, training steps, or dataset fraction.

The initial deliverable is an `SMCO-HB` framework that:

- keeps SMCO responsible for proposing configurations
- keeps Hyperband / Successive Halving responsible for budget allocation and promotion
- supports real multi-fidelity evaluation through `evaluate(config, budget, ...)`
- supports synchronous multi-worker execution in the first version

## Scope

### In Scope

- A new Hyperband-based high-level API for continuous numeric HPO spaces
- Support for three proposer variants:
  - `SMCO-HB`
  - `SMCO_R-HB`
  - `SMCO_BR-HB`
- Training-resource fidelity:
  - epochs
  - training steps
  - dataset fraction
  - other runner-defined monotone resource budgets
- Standard synchronous Hyperband / Successive Halving brackets and rungs
- Fixed-size multi-worker parallel execution within a rung
- Optional checkpoint reuse at the runner boundary
- Full run history, incumbent tracking, and trajectory reporting

### Out of Scope

- Categorical or conditional configuration spaces
- Multi-objective optimization
- Asynchronous scheduling such as ASHA
- Dynamic resource preemption
- Fidelity-aware surrogate modeling beyond the first-version proposer/history coupling
- Mandatory checkpoint semantics in the scheduler core

## Recommended Architecture

The recommended design is to add a standalone Hyperband scheduler layer on top of the existing SMCO variants rather than embedding rung logic inside `smco`, `smco_r`, or `smco_br`.

This keeps responsibilities separate:

- SMCO variants propose configurations
- Hyperband decides which budgets to allocate and which configurations to promote
- the runner executes training
- the history layer records all outcomes

This mirrors the recommended SMAC-style separation: scheduler-level promotion is configuration-and-budget based, while any checkpoint reuse remains an optional runner capability rather than a core scheduler requirement.

## Core Design Decisions

### 1. Fidelity Semantics

The first version uses training-resource fidelity, not synthetic function-budget fidelity.

The canonical meaning of `budget` is one of:

- number of epochs
- number of training steps
- fraction of training data
- any equivalent monotone training resource exposed by the evaluation function

The scheduler treats `budget` as an opaque ordered resource level. It does not interpret domain meaning beyond ordering and bracket construction.

### 2. Promotion Semantics

The scheduler uses stateless promotion semantics:

- promotion decisions are based on `(config, budget, score)` records
- the scheduler promotes configurations, not training sessions
- the core algorithm does not require persisted training state

Checkpoint reuse is allowed but optional:

- the runner may return a checkpoint handle
- the history may store checkpoint references by `(config_id, budget)`
- higher-budget trials may receive that reference back through the runner interface

If checkpoint restore fails, the runner may fall back to training from scratch at the higher budget.

### 3. SMCO Role

SMCO acts as a global configuration proposer, not as a per-bracket optimizer and not as a rung-level ranking model.

Recommended behavior:

- the full `SMCO-HB` run shares one global history
- whenever the scheduler needs fresh configurations for a rung, it asks the proposer for new candidates
- the proposer uses accumulated history to bias future proposals

Rejected alternatives:

- running an independent SMCO instance per bracket
- using SMCO to override observed rung rankings

## Public API

The first version should expose a new high-level entry point in a new module such as `src/smco/multifidelity.py`.

Suggested API:

```python
run = smco_hb(
    suggest_variant="smco_r",
    config_space=...,
    evaluate=...,
    min_budget=1,
    max_budget=27,
    eta=3,
    seed=123,
    n_workers=4,
    proposer_options={...},
    runner_options={...},
)
```

### API Parameters

- `suggest_variant`
  - one of `smco`, `smco_r`, `smco_br`
  - selects the proposer behavior
- `config_space`
  - first version supports continuous numeric domains only
- `evaluate`
  - user-supplied callable with budget-aware evaluation
- `min_budget`, `max_budget`, `eta`
  - standard Hyperband controls
- `seed`
  - top-level reproducibility control
- `n_workers`
  - fixed worker-pool size for synchronous parallel trial execution
- `proposer_options`
  - proposer-specific controls
- `runner_options`
  - execution-layer controls such as retry, timeout, or artifact handling

### Evaluation Interface

Suggested runner-facing contract:

```python
result = evaluate(config, budget, seed=None, checkpoint=None)
```

Expected normalized fields after runner wrapping:

- `score`
- `cost`
- `status`
- `checkpoint` (optional)
- `metadata` (optional)

Scores are stored under a maximize convention. Minimization targets should be normalized at the facade or runner boundary.

### Result Object

The first version should define a dedicated result type rather than reusing `SMCOResult`.

Suggested fields for `SMCOHBResult`:

- `incumbent_config`
- `incumbent_score`
- `incumbent_budget`
- `history`
- `brackets`
- `trajectory`
- `metadata`

This object represents a multi-budget scheduling process and therefore should remain distinct from the current single-run SMCO result model.

## Internal Components

### `SMCOProposer`

Responsibilities:

- wrap `smco`, `smco_r`, or `smco_br` as configuration proposal logic
- consume global history summaries
- emit fresh candidate configurations

Non-responsibilities:

- bracket management
- promotion decisions
- worker scheduling

### `HyperbandScheduler`

Responsibilities:

- construct brackets and rung budgets from `min_budget`, `max_budget`, and `eta`
- determine how many fresh configurations are needed
- select top-performing configurations for promotion

Non-responsibilities:

- generating configurations
- interpreting checkpoints
- executing trials

### `ObjectiveRunner`

Responsibilities:

- call the user evaluation function
- normalize outputs
- catch exceptions and convert them into standardized trial results
- optionally pass checkpoint references through

### `RunHistory`

Responsibilities:

- store every trial outcome
- provide rung-level ranking queries
- provide highest-budget observed score per configuration
- provide checkpoint lookup by `(config_id, budget)` when present

### `SMCOHBFacade`

Responsibilities:

- validate public API inputs
- connect proposer, scheduler, runner, and history
- orchestrate the full run
- return `SMCOHBResult`

## Data Flow

Recommended execution flow:

1. The scheduler determines the current bracket and rung requirements.
2. If new configurations are needed, the facade asks the proposer for candidates.
3. The runner executes the requested trials at the current rung budget.
4. Trial outcomes are written into run history.
5. The scheduler queries history to select configurations for promotion.
6. The next rung executes after the current rung finishes.
7. After all brackets finish, the facade selects the incumbent from the highest completed budget evidence and returns the final result.

## Parallel Execution

The first version includes synchronous multi-worker execution.

### Included

- `n_workers > 1`
- fixed-size worker pool
- concurrent execution of multiple trials in the same rung
- barrier synchronization at rung boundaries

### Excluded

- asynchronous promotion
- ASHA-style scheduling
- immediate cross-rung refill when a worker becomes idle
- dynamic preemption or migration of in-flight trials

This choice improves wall-clock performance while preserving standard Hyperband semantics and keeping testability and reproducibility manageable.

## Error Handling and Failure Semantics

Each normalized trial result should be classified into one of these statuses:

- `success`
- `failed`

The first version does not require a separate user-facing `pruned` status.

### Failure Rules

- failed trials cannot be promoted
- `NaN` and infinite scores are treated as failure
- one worker failure does not cancel sibling trials in the same rung
- if too many trials fail, the actual promotion count may be smaller than the theoretical top-k count

### Checkpoint Failure Rules

- checkpoint restoration failure may fall back to from-scratch higher-budget execution
- this policy belongs to the runner, not the scheduler
- the scheduler only observes success or failure at the requested budget

## Testing Strategy

The first version should be validated at four layers.

### 1. Scheduler Unit Tests

Validate:

- bracket construction
- rung budgets
- per-rung configuration counts
- promotion counts under standard Hyperband rules

### 2. History and Promotion Unit Tests

Validate:

- rung ranking correctness
- failed-trial filtering
- same-configuration multi-budget queries
- incumbent extraction from highest completed budgets

### 3. Runner and Facade Integration Tests

Use a synthetic budget-aware evaluation function to validate:

- `budget` is passed through correctly
- synchronous multi-worker execution respects rung barriers
- checkpoint references are optionally passed through
- score normalization is consistent

### 4. End-to-End Smoke Tests

Use a cheap toy HPO task where higher budget reduces noise or improves signal fidelity and validate:

- `SMCO-HB` completes successfully
- `SMCO_R-HB` completes successfully
- `SMCO_BR-HB` completes successfully
- final incumbent selection is reasonable and reproducible

## Success Criteria

The first version is considered complete when:

- all three proposer variants can run under the Hyperband wrapper
- budget allocation follows standard synchronous Hyperband / Successive Halving rules
- the user evaluation function receives real training-resource budgets
- rung-level synchronous parallel execution works with a fixed worker pool
- full history and incumbent reporting are available
- unit tests and smoke tests pass

## File and Module Direction

Suggested initial file layout:

```text
src/smco/multifidelity.py
src/smco/multifidelity_scheduler.py
src/smco/multifidelity_history.py
src/smco/multifidelity_runner.py
src/smco/multifidelity_results.py
tests/test_multifidelity.py
```

Exact file splitting may be adjusted during implementation, but proposer, scheduler, runner, history, and result responsibilities should remain clearly separated.

## Implementation Notes

- Preserve the current SMCO optimizer APIs unchanged.
- Add the multi-fidelity layer as a new high-level capability rather than retrofitting rung logic into `optimizer.py`.
- Maintain maximize-normalized internal score handling.
- Keep the first version continuous-only to match the current repository strengths.
- Prefer deterministic seed plumbing across proposer, scheduler, and runner layers.
