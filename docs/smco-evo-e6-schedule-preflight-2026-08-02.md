# E6.3 schedule — P1 preflight findings (2026-08-02)

Per `docs/smco-evo-highdim-fleet-formal-experiment-runbook-2026-08-02.md` §4.1,
before any E6.3 dispatch the manifest must be de-duplicated (216→189), the
instance semantics decided (§4.1.3), and the 61 old raw proven reusable (§4.1.4).
This runbook gate **FAILS** — the `schedule` manifest is not the artefact the
runbook assumed. No dispatch (P2) may run until the manifest-level issues below
are resolved by the paper plan.

## 1. De-duplication: PASS

`result/e6-2026-07-31/schedule/e6_ablations__synthetic_highdim.json` has 216
tasks, 189 unique `run_id`, 27 duplicate run-ids covering 54 tasks. **All 27
duplicate run-ids have byte-identical task content** (0 inconsistent) — so a
first-occurrence de-dup to 189 is safe and deterministic (§4.1.1 satisfied).

## 2. Instance semantics gate (§4.1.3): FAIL — and worse than expected

The manifest is **not** a `development` 200/500/1000 schedule ablation:

- **dimensions: 1000, 3000, 5000** (72 tasks each) — heavy high-dim, not
  200/500/1000. A d=3000/5000 rerun is ~22–30 h **per task** (per fleet memory),
  i.e. days of compute, not minutes.
- **functions: Ackley, Rastrigin, Rosenbrock** (3, not 4) × **instances 0,1,2**
  (3, not 5).
- **`evolution_points` and `elimination_rate` are `None` for ALL 216 tasks** —
  this is not a schedule (evolution_points × elimination_rate) ablation at all;
  it is mis-named. What varies is `evolution_strategy` (rand1bin/best1bin/...).
- **168/216 tasks have `instance_artifact_dir = None`** (all d=3000/5000 + 24
  d=1000); the 48 that name a dir point at `instances/development_*` which **do
  not exist anywhere on disk**, and no `instances_index_confirmatory*.json` (or
  development index) exists.

So Option A ("reuse frozen development instances, rerun 189") is infeasible as
stated: the referenced instances are gone / never generated, and most tasks are
multi-day high-dim compute. The runbook §4.1.3 instance gate fails hard.

## 3. Reuse audit (§4.1.4): FAIL

`reuse_decision.csv` (written under `result/e6-2026-08-02/e6_schedule_dedup/`)
shows **only 8/61 old raw match the current manifest**; 53 have `run_id`s not in
the manifest. The 61 raw all carry a valid 40-hex `git_commit`
(`4e7df7a4...`, "fix(e4e5): cocoex instances comma join") — i.e. a **different,
older generation** produced before `f73585a` (A-08: run_id now includes
checkpoints) and contain strategy-variant tasks not present in the schedule
manifest. The "reuse 61 / rerun 128" upper bound is invalid; nothing here is
cleanly reusable.

## 4. Decision needed (paper plan)

The `schedule` manifest is internally inconsistent with its raw and is not the
intended ablation. Options for the paper plan to choose:

1. **Drop / re-scope E6.3** — exclude this manifest from the paper (it is already
   `deferred` in the canonical index) and define the real schedule ablation
   fresh (correct params/dims/instances) if needed.
2. **Regenerate from a correct spec** — define the intended schedule-ablation
   grid (evolution_points × elimination_rate at the planned dims/instances),
   generate + freeze instances, build a new manifest, rerun. Substantial scope.

Until one is chosen, E6.3 stays blocked. The comparative analysis (E3) and the
E6 start-count + strategy evidence already stand on their own.
