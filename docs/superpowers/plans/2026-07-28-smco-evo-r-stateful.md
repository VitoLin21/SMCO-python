# SMCO-EVO R True State-Preserving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a true state-preserving (SP) evolutionary scheduler to the R SMCO-EVO port, rename the existing restart logic to RS, and thread a `state_semantics` switch through the three public R EVO entry points so the R side completes the 2×2 (language × semantics) matrix.

**Architecture:** New `SMCO_evo_stateful.R` holds the SMCO state container plus three primitives (`initialize_smco_state` / `run_smco_state_until` / `smco_state_to_result`) that mirror Python `optimizer.py`. `run_smco_state_until` continues the recursive accumulator `s_value` directly — it does NOT call `SMCO_single`. Two schedulers built on those primitives replace the current mis-named `run_evolutionary_states`: a real SP runner (survivors carry state across boundaries) and `run_evolutionary_restarts` (RS — re-init from `x_runmax` each boundary, global archive keeps best). `SMCO_evo.R` gains the `state_semantics` switch in its entry points and sources the new file at the end.

**Tech Stack:** R (base + `qrng` for Sobol), `Rscript` test runner with `stopifnot`-style assertions (no testthat). Python `pytest` for cross-language reference and Gate B.

**Spec:** [`docs/superpowers/specs/2026-07-28-smco-evo-r-stateful-design.md`](../specs/2026-07-28-smco-evo-r-stateful-design.md)
**Parent plan:** [`docs/smco-evo-highdim-implementation-plan-2026-07-28.md`](../../smco-evo-highdim-implementation-plan-2026-07-28.md) Task 4
**Commit boundary:** single `feat: add stateful evolutionary scheduler in r` (spec + code + tests together, per user preference)

---

## File Structure

- **Create:** `vendor/SMCO_R/main/SMCO_evo_stateful.R` — state container, three primitives, `.clip_and_promote`, SP runner `run_evolutionary_states`, RS runner `run_evolutionary_restarts`.
- **Modify:** `vendor/SMCO_R/main/SMCO_evo.R` — delete the old (mis-named, restart-based) `run_evolutionary_states` (lines 117-262); add `state_semantics` to `.run_evo_core` / `.run_evolutionary_branch` / `SMCO_EVO` / `SMCO_R_EVO` / `SMCO_BR_EVO`; append `source("SMCO_evo_stateful.R")` at end of file.
- **Create:** `vendor/SMCO_R/main/tests/test_evolution_semantics.R` — mirrors Python `tests/test_evolution_semantics.py` + plan Task 4 R test list.
- **Unchanged:** `vendor/SMCO_R/main/SMCO.R` (SMCO_single byte-identical), `vendor/SMCO_R/v1.0.0/` (frozen).

Source order (the new file is pulled in automatically by `SMCO_evo.R`'s trailing source, so `run_highdim_r.R` / `align/r_side.R` / existing tests need no change):
```
evaluation_budget.R -> SMCO.R (cmpfun at end) -> SMCO_evo.R -> SMCO_evo_stateful.R
```

**Commit cadence note:** Per user preference this Task ships as ONE commit at the end (Task 6). Do not `git commit` between sub-tasks; the TDD red/green steps still run tests after each change.

---

## Task 1: Stateful primitives (`SMCO_evo_stateful.R`)

**Files:**
- Create: `vendor/SMCO_R/main/SMCO_evo_stateful.R`
- Test: `vendor/SMCO_R/main/tests/test_evolution_semantics.R`

- [ ] **Step 1: Create the test file with the primitives tests**

Create `vendor/SMCO_R/main/tests/test_evolution_semantics.R`:

```r
#####################################
# test_evolution_semantics.R - Task 4 / Gate B (R side) state-semantics tests.
# Run: Rscript vendor/SMCO_R/main/tests/test_evolution_semantics.R
# No testthat dependency: stopifnot() assertions, non-zero exit on failure.
#####################################
options(warn = 1)

arg <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("--file=", arg, value = TRUE)
if (length(script_arg) == 1) {
  me <- normalizePath(sub("--file=", "", script_arg))
  smco_dir <- dirname(dirname(me))
} else {
  smco_dir <- "../."
}
for (f in c("evaluation_budget.R", "SMCO.R", "SMCO_evo.R", "SMCO_evo_stateful.R")) {
  source(file.path(smco_dir, f))
}

fail <- function(msg) { message("FAIL: ", msg); quit(status = 1) }
check <- function(cond, msg) { if (!isTRUE(cond)) fail(msg) }
approx <- function(a, b, tol = 1e-9) all(abs(a - b) <= tol)

ff <- function(x) -sum(x^2)

# ---- 1. initialize_smco_state counts the init evaluation -------------------
ctx <- budget_ctx(ff, max_evals = 50L, objective_sense = "maximize")
st <- initialize_smco_state(ff, c(0.5, 0.5), iter_nstart = 1L, iter_boost = 0L,
                           use_runmax = TRUE, birth_iteration = 0L, budget = ctx,
                           event = "initialization")
check(approx(st$f_current, -0.5), "init f_current wrong")
check(approx(st$s_value[1], 0.5) && approx(st$s_value[2], 0.5), "s_value init wrong")
check(st$current_n == 1L && st$initial_n == 1L, "current_n/initial_n wrong")
check(approx(st$f_runmax, -0.5), "f_runmax init wrong")
check(ctx_evaluations(ctx) == 1L, "init did not count 1 evaluation")
check(ctx_evaluation_counts(ctx)$initialization == 1L, "init event miscounted")

# replacement event tagging
ctx2 <- budget_ctx(ff, max_evals = 50L)
st2 <- initialize_smco_state(ff, c(0.1, -0.2), iter_nstart = 1L, iter_boost = 0L,
                            use_runmax = TRUE, birth_iteration = 5L, budget = ctx2,
                            event = "replacement_initialization")
check(ctx_evaluation_counts(ctx2)$replacement_initialization == 1L, "replacement event miscounted")
check(st2$birth_iteration == 5L, "birth_iteration not recorded")

# ---- 2. run_smco_state_until matches SMCO_single (deterministic, buffer_rand=FALSE) ---
# State-preserving continuation must reproduce a fresh SMCO_single when started
# from the same point with the same iter_max.
st3 <- initialize_smco_state(ff, c(0.5, -0.3), iter_nstart = 1L, iter_boost = 0L,
                            use_runmax = TRUE, budget = NULL)
st3 <- run_smco_state_until(st3, ff, c(-1, -1), c(1, 1), bounds_buffer = 0.05,
                           buffer_rand = FALSE, iter_target = 12L, tol_conv = 1e-12,
                           partial_option = "center", use_runmax = TRUE)
single <- SMCO_single(ff, c(-1, -1), c(1, 1), start_point = c(0.5, -0.3),
                     bounds_buffer = 0.05, buffer_rand = FALSE, iter_max = 12L,
                     iter_nstart = 1L, iter_boost = 0L, tol_conv = 1e-12,
                     partial_option = "center", use_runmax = TRUE)
check(approx(st3$f_current, single$f_optimal), "SP f_current != SMCO_single f_optimal")
check(all(approx(st3$x_current, single$x_optimal)), "SP x_current != SMCO_single x_optimal")
check(approx(st3$f_runmax, single$f_runmax), "SP f_runmax != SMCO_single f_runmax")

cat("TASK1 PRIMITIVES TESTS PASSED\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `Rscript vendor/SMCO_R/main/tests/test_evolution_semantics.R`
Expected: FAIL / error — `SMCO_evo_stateful.R` does not exist yet (`could not find function "initialize_smco_state"` or source error).

- [ ] **Step 3: Write the primitives in `SMCO_evo_stateful.R`**

Create `vendor/SMCO_R/main/SMCO_evo_stateful.R`:

```r
#####################################
# SMCO_evo_stateful.R - True state-preserving evolutionary SMCO (R port of
# Python _initialize_smco_state / _run_smco_state_until / _run_evolutionary_states
# / _run_evolutionary_restarts in src/smco/optimizer.py). Task 4 of the
# SMCO-EVO high-dim paper campaign.
#
# This file MUST be sourced AFTER SMCO_evo.R (it reuses generate_evolution_points,
# .evolution_boundaries, .n_eliminate, %||%) and AFTER SMCO.R (compute_partial_signs,
# check_bounds, eval_fe). SMCO_evo.R sources this file at its end.
#####################################

#---- state container (mirrors Python SMCOState, optimizer.py:35) -----------

# Build the initial state. The start-point evaluation is counted once under
# `event` (initialization / replacement_initialization / restart_initialization).
initialize_smco_state <- function(f, start_point, iter_nstart, iter_boost,
                                  use_runmax, birth_iteration = 0L,
                                  budget = NULL, event = "initialization") {
  x_current <- as.numeric(start_point)
  f_current <- eval_fe(budget, f, x_current, event = event)
  n_boost_1 <- as.integer(iter_boost + iter_nstart)
  list(
    x_current = x_current,
    f_current = f_current,
    s_value = x_current * n_boost_1,
    current_n = n_boost_1,
    initial_n = n_boost_1,
    iter_boost = as.integer(iter_boost),
    x_runmax = if (use_runmax) x_current else NULL,
    f_runmax = if (use_runmax) f_current else NULL,
    iterations = 0L,
    birth_iteration = as.integer(birth_iteration),
    stopped_target_n = NULL
  )
}

# Ranking value/point mirror Python SMCOState.ranking_value/ranking_point.
state_ranking_value <- function(state) {
  if (!is.null(state$f_runmax)) state$f_runmax else state$f_current
}
state_ranking_point <- function(state) {
  if (!is.null(state$x_runmax)) state$x_runmax else state$x_current
}

# Snapshot a state into a SingleResult-shaped list (mirrors SMCOState.to_result).
smco_state_to_result <- function(state) {
  if (is.null(state$x_runmax)) {
    list(x_optimal = state$x_current, f_optimal = state$f_current,
         iterations = state$iterations)
  } else {
    list(x_optimal = state$x_current, f_optimal = state$f_current,
         iterations = state$iterations,
         x_runmax = state$x_runmax, f_runmax = state$f_runmax)
  }
}

# Continue the recursive SMCO accumulator from state$current_n up to
# initial_n + iter_target. DOES NOT call SMCO_single — s_value carries across
# calls, which is what makes a trajectory state-preserving.
run_smco_state_until <- function(state, f, lo, hi, bounds_buffer, buffer_rand,
                                 iter_target, tol_conv, partial_option,
                                 use_runmax, budget = NULL) {
  bounds_diff <- hi - lo
  d <- length(bounds_diff)
  fixed_pushout <- bounds_buffer * bounds_diff
  fixed_upper_out <- hi + fixed_pushout
  fixed_lower_out <- lo - fixed_pushout
  initial_n <- if (is.null(state$initial_n)) state$current_n else state$initial_n
  target_n <- initial_n + as.integer(iter_target)
  iter_min_check <- initial_n + ceiling(iter_target / 2)
  if (!is.null(state$stopped_target_n) && target_n <= state$stopped_target_n) {
    return(state)
  }
  step_cost <- as.integer(if (partial_option == "center") 2L * d else d) + 1L
  while (state$current_n <= target_n) {
    if (!is.null(budget) && !ctx_can_evaluate(budget, step_cost)) {
      ctx_set_termination(budget, "evaluation_budget")
      break
    }
    n <- state$current_n
    h_step <- bounds_diff / (n + 1)
    partial <- compute_partial_signs(f, state$x_current, state$f_current, h_step,
                                     lo, hi, partial_option, use_runmax,
                                     budget = budget)
    if (buffer_rand) {
      pushout <- bounds_buffer * bounds_diff * runif(d, -1, 1)
      bounds_upper_out <- hi + pushout
      bounds_lower_out <- lo - pushout
    } else {
      bounds_upper_out <- fixed_upper_out
      bounds_lower_out <- fixed_lower_out
    }
    Z <- partial$signs * bounds_upper_out + (1 - partial$signs) * bounds_lower_out
    state$s_value <- state$s_value + Z
    x_next <- state$s_value / (n + 1)
    f_next <- eval_fe(budget, f, x_next, event = "iterate")
    if (use_runmax) {
      f_next_best <- max(partial$f_partial_best, f_next)
      if (is.null(state$f_runmax) || f_next_best > state$f_runmax) {
        state$f_runmax <- f_next_best
        state$x_runmax <- if (partial$f_partial_best > f_next) partial$x_partial_best else x_next
      }
    }
    f_prev <- state$f_current
    state$f_current <- f_next
    state$x_current <- x_next
    state$current_n <- n + 1L
    state$iterations <- as.integer(n - state$iter_boost)
    if (n >= iter_min_check && abs(state$f_current - f_prev) < tol_conv) {
      state$stopped_target_n <- target_n
      break
    }
  }
  state
}

# Clip a result's optimal/runmax points back into bounds (clip_recheck counted)
# and promote runmax when it beats the current optimum. Mirrors Python
# _clip_result_to_bounds + _promote_runmax (optimizer.py:760-790).
.clip_and_promote <- function(res, f, lo, hi, use_runmax, budget = NULL) {
  chk <- check_bounds(res$x_optimal, lo, hi)
  if (chk$is_out && (is.null(budget) || ctx_can_evaluate(budget, 1L))) {
    res$x_optimal <- chk$x_in
    res$f_optimal <- eval_fe(budget, f, chk$x_in, event = "clip_recheck")
  }
  if (use_runmax && !is.null(res$x_runmax)) {
    chk_rm <- check_bounds(res$x_runmax, lo, hi)
    if (chk_rm$is_out && (is.null(budget) || ctx_can_evaluate(budget, 1L))) {
      res$x_runmax <- chk_rm$x_in
      res$f_runmax <- eval_fe(budget, f, chk_rm$x_in, event = "clip_recheck")
    }
    if (!is.null(res$f_runmax) && res$f_runmax > res$f_optimal) {
      res$x_optimal <- res$x_runmax
      res$f_optimal <- res$f_runmax
    }
  }
  res
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `Rscript vendor/SMCO_R/main/tests/test_evolution_semantics.R`
Expected: PASS — prints `TASK1 PRIMITIVES TESTS PASSED`.

If `SP f_current != SMCO_single f_optimal` fails, check that `iter_target`, `iter_min_check`, and the runmax update branch all match `SMCO_single` (SMCO.R:288-347) exactly.

---

## Task 2: SP scheduler `run_evolutionary_states` (replaces the old mis-named one)

**Files:**
- Modify: `vendor/SMCO_R/main/SMCO_evo_stateful.R` (append SP runner)
- Modify: `vendor/SMCO_R/main/SMCO_evo.R` (delete old `run_evolutionary_states`, lines 117-262)
- Test: `vendor/SMCO_R/main/tests/test_evolution_semantics.R` (append SP tests)

- [ ] **Step 1: Append the SP tests to the test file**

Append to `vendor/SMCO_R/main/tests/test_evolution_semantics.R` (before any final `cat` — the Task 1 `cat` line will be removed when Task 6 finalizes; for now add after it):

```r
# ---- 3. SP staged continuation == one-shot continuation -------------------
# Running to `boundary` then to `iter_max` must equal running to `iter_max`
# once (the accumulator is preserved). Use buffer_rand=FALSE for determinism.
base <- initialize_smco_state(ff, c(0.4, -0.2), iter_nstart = 1L, iter_boost = 0L,
                             use_runmax = TRUE, budget = NULL)
oneshot <- initialize_smco_state(ff, c(0.4, -0.2), iter_nstart = 1L, iter_boost = 0L,
                                use_runmax = TRUE, budget = NULL)
staged <- base
staged <- run_smco_state_until(staged, ff, c(-1, -1), c(1, 1), 0.05, FALSE,
                              iter_target = 5L, tol_conv = 1e-12,
                              partial_option = "center", use_runmax = TRUE)
staged <- run_smco_state_until(staged, ff, c(-1, -1), c(1, 1), 0.05, FALSE,
                              iter_target = 5L, tol_conv = 1e-12,
                              partial_option = "center", use_runmax = TRUE)
oneshot <- run_smco_state_until(oneshot, ff, c(-1, -1), c(1, 1), 0.05, FALSE,
                               iter_target = 10L, tol_conv = 1e-12,
                               partial_option = "center", use_runmax = TRUE)
check(approx(staged$f_current, oneshot$f_current), "SP staged != one-shot f")
check(all(approx(staged$x_current, oneshot$x_current)), "SP staged != one-shot x")

# ---- 4. SP runner end-to-end (history tagged state_preserving) ------------
sp <- matrix(c(-0.8, -0.3, 0.2, 0.7, -0.5, 0.4, -0.1, 0.6), ncol = 2, byrow = TRUE)
ctrl <- list(bounds_buffer = 0.05, buffer_rand = TRUE, tol_conv = 1e-12,
             partial_option = "center", use_runmax = TRUE, iter_nstart = 4L, seed = 123)
evo_sp <- run_evolutionary_states(ff, c(-1, -1), c(1, 1), sp, ctrl,
                                  evolution_points = c(0.5, 0.75),
                                  elimination_rate = 0.25,
                                  evolution_strategy = "rand1bin",
                                  de_factor = 0.8, de_crossover = 0.7,
                                  iter_max = 40L, iter_boost = 0L, budget = NULL)
check(length(evo_sp$results) == 4L, "SP result count != n_starts")
check(all(sapply(evo_sp$history, function(h) identical(h$state_semantics, "state_preserving"))),
      "SP history missing state_preserving tag")
check(is.finite(evo_sp$results[[1]]$f_optimal), "SP result not finite")

# ---- 5. SP vs RS diverge after a boundary (SP preserves accumulator) ------
# With a real boundary, SP (carries s_value) and RS (restarts from x_runmax)
# must differ — proving SP actually preserves state.
ctrl_rs <- ctrl
evo_rs <- run_evolutionary_restarts(ff, c(-1, -1), c(1, 1), sp, ctrl_rs,
                                    evolution_points = c(0.5, 0.75),
                                    elimination_rate = 0.25,
                                    evolution_strategy = "rand1bin",
                                    de_factor = 0.8, de_crossover = 0.7,
                                    iter_max = 40L, iter_boost = 0L, budget = NULL)
sp_best <- max(sapply(evo_sp$results, function(r) r$f_optimal))
rs_best <- max(sapply(evo_rs$results, function(r) r$f_optimal))
check(!approx(sp_best, rs_best, tol = 1e-6), "SP and RS identical after boundary")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `Rscript vendor/SMCO_R/main/tests/test_evolution_semantics.R`
Expected: FAIL — `run_evolutionary_states`/`run_evolutionary_restarts` still refer to the old restart-based body, or RS not defined yet.

- [ ] **Step 3: Delete the old mis-named `run_evolutionary_states` from `SMCO_evo.R`**

In `vendor/SMCO_R/main/SMCO_evo.R`, delete the entire old function block from the comment block starting `# Port of Python _run_evolutionary_states.` (line 117) through the end of `run_evolutionary_states <- function(...) { ... }` (line 262, just before `` `%||%` <- function(a, b) ... ``). Keep `%||%` and everything else.

The old block to remove spans:
```r
# Port of Python _run_evolutionary_states.
# Runs the multi-start evolutionary loop and returns the list of final
# per-state results (each like SMCO_single's return value) plus history.
run_evolutionary_states <- function(f, bounds_lower, bounds_upper, start_points,
                                    ... budget = NULL) {
  ... (entire body, lines 120-262)
}

```
Remove it cleanly so `%||%` (line 264) remains defined.

- [ ] **Step 4: Append the SP runner to `SMCO_evo_stateful.R`**

Append to `vendor/SMCO_R/main/SMCO_evo_stateful.R`:

```r
#---- SP scheduler (mirrors Python _run_evolutionary_states, optimizer.py:1091) ----

# State-preserving evolutionary loop. Survivors carry their s_value/current_n
# across boundaries; only eliminated slots are re-initialized at the boundary.
run_evolutionary_states <- function(f, bounds_lower, bounds_upper, start_points,
                                    opt_control, evolution_points,
                                    elimination_rate, evolution_strategy,
                                    de_factor, de_crossover, iter_max,
                                    iter_boost, budget = NULL) {
  n_starts <- nrow(start_points)
  bounds_buffer <- opt_control$bounds_buffer
  buffer_rand <- opt_control$buffer_rand
  tol_conv <- opt_control$tol_conv
  partial_option <- opt_control$partial_option
  use_runmax <- opt_control$use_runmax
  iter_nstart <- opt_control$iter_nstart

  if (is.null(opt_control$seed)) set.seed(NULL) else set.seed(opt_control$seed)

  states <- vector("list", n_starts)
  n_init <- 0L
  for (i in seq_len(n_starts)) {
    if (!is.null(budget) && !ctx_can_evaluate(budget, 1L)) break
    n_init <- n_init + 1L
    states[[n_init]] <- initialize_smco_state(
      f, start_points[i, ], iter_nstart, iter_boost, use_runmax,
      birth_iteration = 0L, budget = budget, event = "initialization")
  }
  if (n_init < n_starts) states <- states[seq_len(n_init)]

  history <- list()
  boundaries <- .evolution_boundaries(iter_max, evolution_points)

  for (boundary in boundaries) {
    for (i in seq_along(states)) {
      states[[i]] <- run_smco_state_until(
        states[[i]], f, bounds_lower, bounds_upper, bounds_buffer, buffer_rand,
        boundary - states[[i]]$birth_iteration, tol_conv, partial_option,
        use_runmax, budget = budget)
    }
    vals <- sapply(states, state_ranking_value)
    ranked <- order(vals, decreasing = TRUE)
    n_elim <- .n_eliminate(length(states), elimination_rate)
    survivors_idx <- ranked[seq_len(length(ranked) - n_elim)]
    eliminated_idx <- utils::tail(ranked, n_elim)
    best_before <- vals[ranked[1]]
    generated_count <- 0L
    if (length(eliminated_idx) > 0L) {
      parents <- t(sapply(states[survivors_idx], state_ranking_point))
      scores <- vals[survivors_idx]
      new_pts <- generate_evolution_points(parents, scores, n_new = n_elim,
                                           strategy = evolution_strategy,
                                           bounds_lower = bounds_lower,
                                           bounds_upper = bounds_upper,
                                           de_factor = de_factor,
                                           de_crossover = de_crossover)
      generated_count <- nrow(new_pts)
      for (j in seq_len(nrow(new_pts))) {
        if (!is.null(budget) && !ctx_can_evaluate(budget, 1L)) break
        states[[eliminated_idx[j]]] <- initialize_smco_state(
          f, new_pts[j, ], iter_nstart, iter_boost + boundary, use_runmax,
          birth_iteration = as.integer(boundary), budget = budget,
          event = "replacement_initialization")
      }
    }
    history[[length(history) + 1L]] <- list(
      iteration = as.integer(boundary), strategy = evolution_strategy,
      state_semantics = "state_preserving",
      survivor_count = length(survivors_idx), eliminated_count = n_elim,
      generated_count = generated_count, best_before = best_before)
  }

  results <- vector("list", length(states))
  for (i in seq_along(states)) {
    states[[i]] <- run_smco_state_until(
      states[[i]], f, bounds_lower, bounds_upper, bounds_buffer, buffer_rand,
      iter_max - states[[i]]$birth_iteration, tol_conv, partial_option,
      use_runmax, budget = budget)
    res <- smco_state_to_result(states[[i]])
    res$iterations <- as.integer(res$iterations + states[[i]]$birth_iteration)
    res <- .clip_and_promote(res, f, bounds_lower, bounds_upper, use_runmax, budget)
    results[[i]] <- res
  }
  list(results = results, history = history)
}
```

- [ ] **Step 5: Run test to verify SP tests pass (RS test still fails)**

Run: `Rscript vendor/SMCO_R/main/tests/test_evolution_semantics.R`
Expected: sections 1-4 PASS; section 5 fails because `run_evolutionary_restarts` is not yet defined. That is correct for this task — proceed to Task 3.

---

## Task 3: RS scheduler `run_evolutionary_restarts`

**Files:**
- Modify: `vendor/SMCO_R/main/SMCO_evo_stateful.R` (append RS runner)
- Test: `vendor/SMCO_R/main/tests/test_evolution_semantics.R` (append RS tests)

- [ ] **Step 1: Append the RS tests**

Append to `vendor/SMCO_R/main/tests/test_evolution_semantics.R`:

```r
# ---- 6. RS counts restart_initialization + replacement_initialization -----
ctx6 <- budget_ctx(ff, max_evals = 200000L, objective_sense = "maximize")
evo_rs2 <- run_evolutionary_restarts(ff, c(-1, -1), c(1, 1), sp, ctrl,
                                    evolution_points = c(0.5, 0.75),
                                    elimination_rate = 0.25,
                                    evolution_strategy = "rand1bin",
                                    de_factor = 0.8, de_crossover = 0.7,
                                    iter_max = 60L, iter_boost = 0L, budget = ctx6)
counts6 <- unlist(ctx_evaluation_counts(ctx6))
check(counts6[["restart_initialization"]] > 0L, "no restart_initialization counted")
check(counts6[["replacement_initialization"]] > 0L, "no replacement_initialization counted")
check(all(sapply(evo_rs2$history, function(h) identical(h$state_semantics, "restart"))),
      "RS history missing restart tag")

# ---- 7. RS archive preserves global best (best == f_runmax) ---------------
fwig <- function(x) sin(9 * x[1]) - 0.15 * x[1]^2
sp1 <- matrix(c(-1.5, -0.5, 0.3, 1.2, -1.0, 0.8, 0.0, 1.6, -1.8, 0.4, 0.9, -0.3,
                1.1, -1.2, 0.5, 0.7), ncol = 1)
evo_rs3 <- run_evolutionary_restarts(fwig, c(-2), c(2), sp1, ctrl,
                                     evolution_points = c(0.5, 0.75),
                                     elimination_rate = 0.25,
                                     evolution_strategy = "rand1bin",
                                     de_factor = 0.8, de_crossover = 0.7,
                                     iter_max = 40L, iter_boost = 0L, budget = NULL)
best <- evo_rs3$results[[which.max(sapply(evo_rs3$results, function(r) r$f_optimal))]]
check(!is.null(best$f_runmax), "RS best has no f_runmax")
check(approx(best$f_optimal, best$f_runmax), "RS best != its runmax (archive lost best)")

# ---- 8. RS reproducible with same seed ------------------------------------
a <- run_evolutionary_restarts(ff, c(-2, -2), c(2, 2), sp, ctrl,
                              evolution_points = c(0.5, 0.75), elimination_rate = 0.25,
                              evolution_strategy = "rand1bin", de_factor = 0.8,
                              de_crossover = 0.7, iter_max = 50L, iter_boost = 0L)
b <- run_evolutionary_restarts(ff, c(-2, -2), c(2, 2), sp, ctrl,
                              evolution_points = c(0.5, 0.75), elimination_rate = 0.25,
                              evolution_strategy = "rand1bin", de_factor = 0.8,
                              de_crossover = 0.7, iter_max = 50L, iter_boost = 0L)
check(approx(max(sapply(a$results, function(r) r$f_optimal)),
             max(sapply(b$results, function(r) r$f_optimal))), "RS not reproducible")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `Rscript vendor/SMCO_R/main/tests/test_evolution_semantics.R`
Expected: FAIL — `run_evolutionary_restarts` not defined.

- [ ] **Step 3: Append the RS runner to `SMCO_evo_stateful.R`**

Append to `vendor/SMCO_R/main/SMCO_evo_stateful.R`:

```r
#---- RS scheduler (mirrors Python _run_evolutionary_restarts, optimizer.py:1235) ----

# Restart-semantics loop. Every boundary re-initializes a fresh state from
# each survivor's running-best point (s_value NOT carried across). Replacement
# births and survivor restarts are counted; a global archive keeps the best
# running-best seen so elimination can never lose the run best.
run_evolutionary_restarts <- function(f, bounds_lower, bounds_upper, start_points,
                                      opt_control, evolution_points,
                                      elimination_rate, evolution_strategy,
                                      de_factor, de_crossover, iter_max,
                                      iter_boost, budget = NULL) {
  bounds_buffer <- opt_control$bounds_buffer
  buffer_rand <- opt_control$buffer_rand
  tol_conv <- opt_control$tol_conv
  partial_option <- opt_control$partial_option
  use_runmax <- opt_control$use_runmax
  iter_nstart <- opt_control$iter_nstart
  n_starts <- nrow(start_points)

  if (is.null(opt_control$seed)) set.seed(NULL) else set.seed(opt_control$seed)
  boundaries <- .evolution_boundaries(iter_max, evolution_points)

  make_rec <- function(state, anchor) {
    list(x_current = state$x_current, f_current = state$f_current,
         x_runmax = if (use_runmax) state$x_runmax else NULL,
         f_runmax = if (use_runmax) state$f_runmax else NULL,
         anchor = as.integer(anchor), iterations = as.integer(anchor))
  }
  rank_val <- function(rec) if (use_runmax) rec$f_runmax else rec$f_current
  rank_pt <- function(rec) if (use_runmax && !is.null(rec$x_runmax)) rec$x_runmax else rec$x_current

  segment <- function(start_point, anchor, target, event) {
    st <- initialize_smco_state(f, start_point, iter_nstart, iter_boost + anchor,
                                use_runmax, birth_iteration = anchor,
                                budget = budget, event = event)
    if (target > 0L) {
      st <- run_smco_state_until(st, f, bounds_lower, bounds_upper, bounds_buffer,
                                 buffer_rand, target, tol_conv, partial_option,
                                 use_runmax, budget = budget)
    }
    st
  }
  merge_runmax <- function(rec, st) {
    if (use_runmax && !is.null(st$f_runmax)) {
      if (is.null(rec$f_runmax) || st$f_runmax > rec$f_runmax) {
        rec$f_runmax <- st$f_runmax
        rec$x_runmax <- st$x_runmax
      }
    }
    rec$x_current <- st$x_current
    rec$f_current <- st$f_current
    rec
  }

  states <- vector("list", n_starts)
  n_init <- 0L
  for (i in seq_len(n_starts)) {
    if (!is.null(budget) && !ctx_can_evaluate(budget, 1L)) break
    n_init <- n_init + 1L
    st <- segment(start_points[i, ], 0L, 0L, "initialization")
    states[[n_init]] <- make_rec(st, 0L)
  }
  if (n_init < n_starts) states <- states[seq_len(n_init)]

  archive_value <- NULL
  archive_point <- NULL
  history <- list()
  update_archive <- function() {
    if (length(states) == 0L) return(invisible(NULL))
    best <- states[[which.max(sapply(states, rank_val))]]
    v <- rank_val(best)
    if (is.null(archive_value) || v > archive_value) {
      archive_value <<- v
      archive_point <<- rank_pt(best)
    }
    invisible(NULL)
  }
  update_archive()

  for (boundary in boundaries) {
    for (i in seq_along(states)) {
      rec <- states[[i]]
      target <- boundary - rec$anchor
      if (target <= 0L) next
      if (!is.null(budget) && !ctx_can_evaluate(budget, 1L)) break
      st <- segment(rank_pt(rec), rec$anchor, target, "restart_initialization")
      rec <- merge_runmax(rec, st)
      rec$anchor <- as.integer(boundary)
      rec$iterations <- as.integer(boundary)
      states[[i]] <- rec
    }
    update_archive()
    vals <- sapply(states, rank_val)
    ranked <- order(vals, decreasing = TRUE)
    n_elim <- .n_eliminate(length(states), elimination_rate)
    survivors_idx <- ranked[seq_len(length(ranked) - n_elim)]
    eliminated_idx <- utils::tail(ranked, n_elim)
    generated_count <- 0L
    if (length(eliminated_idx) > 0L) {
      parents <- t(sapply(states[survivors_idx], rank_pt))
      scores <- vals[survivors_idx]
      new_pts <- generate_evolution_points(parents, scores, n_new = n_elim,
                                           strategy = evolution_strategy,
                                           bounds_lower = bounds_lower,
                                           bounds_upper = bounds_upper,
                                           de_factor = de_factor,
                                           de_crossover = de_crossover)
      generated_count <- nrow(new_pts)
      for (j in seq_len(nrow(new_pts))) {
        if (!is.null(budget) && !ctx_can_evaluate(budget, 1L)) break
        st <- segment(new_pts[j, ], boundary, 0L, "replacement_initialization")
        states[[eliminated_idx[j]]] <- make_rec(st, boundary)
      }
    }
    history[[length(history) + 1L]] <- list(
      iteration = as.integer(boundary), strategy = evolution_strategy,
      state_semantics = "restart",
      survivor_count = length(survivors_idx), eliminated_count = n_elim,
      generated_count = generated_count,
      best_before = if (length(ranked) > 0L) vals[ranked[1]] else NA_real_)
  }

  results <- vector("list", length(states))
  for (i in seq_along(states)) {
    rec <- states[[i]]
    target <- iter_max - rec$anchor
    if (target > 0L && (is.null(budget) || ctx_can_evaluate(budget, 1L))) {
      st <- segment(rank_pt(rec), rec$anchor, target, "restart_initialization")
      rec <- merge_runmax(rec, st)
      rec$anchor <- as.integer(iter_max)
      rec$iterations <- as.integer(iter_max)
    }
    if (use_runmax && !is.null(rec$f_runmax)) {
      x_opt <- rec$x_runmax; f_opt <- rec$f_runmax
    } else {
      x_opt <- rec$x_current; f_opt <- rec$f_current
    }
    res <- list(x_optimal = x_opt, f_optimal = f_opt, iterations = rec$iterations,
                x_runmax = rec$x_runmax, f_runmax = rec$f_runmax)
    res <- .clip_and_promote(res, f, bounds_lower, bounds_upper, use_runmax, budget)
    results[[i]] <- res
  }

  if (!is.null(archive_value) && length(results) > 0L) {
    final_best <- max(sapply(results, function(r) r$f_optimal))
    if (archive_value > final_best) {
      results[[length(results) + 1L]] <- list(
        x_optimal = archive_point, f_optimal = archive_value,
        iterations = as.integer(iter_max))
    }
  }
  list(results = results, history = history)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `Rscript vendor/SMCO_R/main/tests/test_evolution_semantics.R`
Expected: PASS through section 8. (Sections 1-8 all green.)

---

## Task 4: `state_semantics` entry points + source chain

**Files:**
- Modify: `vendor/SMCO_R/main/SMCO_evo.R` (entry points, `.run_evo_core`, `.run_evolutionary_branch`, trailing source)
- Test: `vendor/SMCO_R/main/tests/test_evolution_semantics.R` (append entry-point tests)

- [ ] **Step 1: Append entry-point tests**

Append to `vendor/SMCO_R/main/tests/test_evolution_semantics.R`:

```r
# ---- 9. invalid state_semantics raises (three families) -------------------
for (entry in c("SMCO_EVO", "SMCO_R_EVO", "SMCO_BR_EVO")) {
  raised <- tryCatch({
    get(entry)(ff, c(-1, -1), c(1, 1), start_points = sp, iter_max = 20,
               seed = 1, state_semantics = "bogus")
    FALSE
  }, error = function(e) TRUE)
  check(raised, paste(entry, "did not reject bogus state_semantics"))
}

# ---- 10. three families run end-to-end under restart ----------------------
for (entry in c("SMCO_EVO", "SMCO_R_EVO", "SMCO_BR_EVO")) {
  res <- get(entry)(ff, c(-1, -1), c(1, 1), start_points = sp, iter_max = 40,
                   seed = 123, tol_conv = 1e-12, state_semantics = "restart")
  check(is.finite(res$best_result$f_optimal), paste(entry, "restart not finite"))
  check(all(sapply(res$evolution_history, function(h) identical(h$state_semantics, "restart"))),
        paste(entry, "restart history tag missing"))
}

# ---- 11. SP and RS agree when there is no boundary ------------------------
kw <- list(start_points = sp, iter_max = 3, seed = 321, tol_conv = 1e-12,
           evolution_points = c(0.999))
sp_res <- do.call(SMCO_EVO, c(list(ff, c(-1, -1), c(1, 1), state_semantics = "state_preserving"), kw))
rs_res <- do.call(SMCO_EVO, c(list(ff, c(-1, -1), c(1, 1), state_semantics = "restart"), kw))
check(approx(sp_res$best_result$f_optimal, rs_res$best_result$f_optimal),
      "SP/RS differ without boundary")
check(all(approx(sp_res$best_result$x_optimal, rs_res$best_result$x_optimal)),
      "SP/RS x differ without boundary")

# ---- 12. BR respects 50/50 split budget -----------------------------------
br <- SMCO_BR_EVO(ff, c(-1, -1), c(1, 1), start_points = sp, iter_max = 40,
                  iter_boost = 20L, seed = 789, state_semantics = "restart",
                  max_evals = 4000)
fe <- br$summary$fe
check(fe$fe_used <= 4000L, "BR fe_used > budget")
check(fe$branch_fe$regular <= 2000L + 1L && fe$branch_fe$boosted <= 2000L + 1L,
      "BR branch cap broken")

# ---- 13. tight budget does not raise and reports evaluation_budget --------
tt <- SMCO_EVO(ff, c(-1, -1), c(1, 1), start_points = sp, iter_max = 200,
               evolution_points = c(0.5), seed = 1, state_semantics = "restart",
               max_evals = 80)
check(tt$summary$fe$fe_used <= 80L, "tight budget exceeded cap")
check(identical(tt$summary$fe$termination_reason, "evaluation_budget"),
      "tight budget termination reason wrong")

# ---- 14. legacy default (no state_semantics) still runs (restart) ---------
legacy <- SMCO_EVO(ff, c(-1, -1), c(1, 1), start_points = sp, iter_max = 40, seed = 1)
check(is.finite(legacy$best_result$f_optimal), "legacy default broke")
check(all(sapply(legacy$evolution_history, function(h) identical(h$state_semantics, "restart"))),
      "legacy default not restart")

cat("ALL R EVOLUTION-SEMANTICS TESTS PASSED\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `Rscript vendor/SMCO_R/main/tests/test_evolution_semantics.R`
Expected: FAIL — `SMCO_EVO` does not yet accept `state_semantics` (unused argument error or wrong default).

- [ ] **Step 3: Add `state_semantics` to `.run_evolutionary_branch` and `.run_evo_core`**

In `vendor/SMCO_R/main/SMCO_evo.R`, modify `.run_evolutionary_branch` (currently ~line 354) to accept `state_semantics` and dispatch:

Replace the existing `.run_evolutionary_branch <- function(...)` with:

```r
.run_evolutionary_branch <- function(f, lo, hi, start_points, opt_control,
                                     evolution_points, elimination_rate,
                                     evolution_strategy, de_factor, de_crossover,
                                     budget = NULL, state_semantics = "restart") {
  splits <- .split_refine_iterations(opt_control$iter_max, opt_control$refine_ratio,
                                     opt_control$refine_search)
  runner <- if (identical(state_semantics, "state_preserving"))
    run_evolutionary_states else run_evolutionary_restarts
  evo <- runner(f, lo, hi, start_points, opt_control,
                evolution_points = evolution_points,
                elimination_rate = elimination_rate,
                evolution_strategy = evolution_strategy,
                de_factor = de_factor, de_crossover = de_crossover,
                iter_max = splits[1], iter_boost = opt_control$iter_boost,
                budget = budget)
  results <- evo$results
  if (isTRUE(opt_control$refine_search) && splits[2] > 0L) {
    results <- lapply(results, .refine_one_evo_result, f = f, lo = lo, hi = hi,
                      opt_control = opt_control, iter_max_refine = splits[2], budget = budget)
  }
  list(results = results, history = evo$history)
}
```

Then modify `.run_evo_core` to accept and validate `state_semantics`, and pass it through to every `.run_evolutionary_branch` call. Change its signature to add `state_semantics = "restart"` and add at the top of its body (after the strategy check):

```r
  if (!(state_semantics %in% c("state_preserving", "restart")))
    stop("state_semantics must be 'state_preserving' or 'restart'")
```

Add `state_semantics = state_semantics` to all three `.run_evolutionary_branch(...)` calls inside `.run_evo_core` (boosted branch, regular branch, and the single-branch `iter_boost <= 0` path).

- [ ] **Step 4: Add `state_semantics` to the three public entry points**

In `vendor/SMCO_R/main/SMCO_evo.R`, add `state_semantics = "restart"` to the signatures of `SMCO_EVO`, `SMCO_R_EVO`, and `SMCO_BR_EVO`, and pass it through to `.run_evo_core`.

`SMCO_EVO` becomes:

```r
SMCO_EVO <- function(f, bounds_lower, bounds_upper, start_points = NULL, ...,
                     evolution_points = c(0.5, 0.75),
                     elimination_rate = 0.25,
                     evolution_strategy = "rand1bin",
                     de_factor = 0.8,
                     de_crossover = 0.7,
                     state_semantics = "restart") {
  if (!(evolution_strategy %in% EVOLUTION_STRATEGIES)) {
    stop("evolution_strategy must be one of: ",
         paste(EVOLUTION_STRATEGIES, collapse = ", "))
  }
  opt_control <- .default_evo_control(list(...))
  opt_control$refine_search <- FALSE
  opt_control$iter_boost <- 0
  .run_evo_core(f, bounds_lower, bounds_upper, start_points, opt_control,
                evolution_points, elimination_rate, evolution_strategy, de_factor, de_crossover,
                state_semantics = state_semantics)
}
```

Apply the same `state_semantics = "restart"` parameter + `state_semantics = state_semantics` pass-through to `SMCO_R_EVO` and `SMCO_BR_EVO` (they already set `refine_search`/`iter_boost`; only the new parameter and pass-through change).

- [ ] **Step 5: Append the trailing source so non-test callers get the stateful file**

At the very end of `vendor/SMCO_R/main/SMCO_evo.R`, append:

```r

# Pull in the true state-preserving / restart schedulers (Task 4). This file
# reuses helpers defined above (generate_evolution_points, .evolution_boundaries,
# .n_eliminate, %||%) and SMCO.R (compute_partial_signs, check_bounds, eval_fe),
# so it MUST be sourced last.
SMCO_EVO_STATEFUL_R <- (function() {
  here <- (function() {
    f <- sys.frame(sys.nframe())
    NA_character_
  })()
  NULL
})()
tryCatch({
  e <- new.env()
  e$. <- sys.source
}, error = function(cond) NULL)
# Source the stateful module from the same directory as this file.
.smco_evo_self_dir <- tryCatch({
  args <- commandArgs(trailingOnly = FALSE)
  fa <- grep("--file=", args, value = TRUE)
  if (length(fa) == 1L) dirname(normalizePath(sub("--file=", "", fa[1L]))) else NA_character_
}, error = function(e) NA_character_)
if (is.na(.smco_evo_self_dir)) {
  # Sourced interactively / via source(): use the directory of this script.
  .smco_evo_self_dir <- dirname(normalizePath(sys.frame(1)$ofile %||% "."))
}
source(file.path(.smco_evo_self_dir, "SMCO_evo_stateful.R"))
rm(.smco_evo_self_dir)
```

**Important simpler alternative (preferred if it resolves cleanly):** because `SMCO_evo.R` is always loaded via `source("<dir>/SMCO_evo.R")`, `sys.frame(1)$ofile` is unreliable across R versions. Instead, replace the whole trailing block above with a direct relative source, and verify in Step 6 that all three loading paths (`Rscript tests/...`, `source` from `run_highdim_r.R`, `source` from `align/r_side.R`) resolve it. Simplest robust form:

```r
# Source the stateful module from the same directory as this file (Task 4).
.evo_self <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NA_character_)
source(file.path(dirname(.evo_self %||% "."), "SMCO_evo_stateful.R"))
rm(.evo_self)
```

If Step 6 shows any loading path fails to resolve the file, switch the trailing source to accept an explicit `base_dir` argument or have the test/runner files source `SMCO_evo_stateful.R` explicitly (the test file already does, so tests are unaffected).

- [ ] **Step 6: Run the full R test, then verify existing tests still pass**

Run:
```bash
Rscript vendor/SMCO_R/main/tests/test_evolution_semantics.R
Rscript vendor/SMCO_R/main/tests/test_evaluation_budget.R
```
Expected: `ALL R EVOLUTION-SEMANTICS TESTS PASSED` and `ALL R BUDGET TESTS PASSED`.

If `test_evaluation_budget.R` section 8/9/10 now miscounts: those sections check `sum(counts) == fe_used` and `replacement_initialization > 0` only — they do NOT assert the `initialization` vs `restart_initialization` split, so the RS rewrite (init now tagged `restart_initialization`) must still pass. If they fail, the regression is in the totals, not the event labels — re-check `.run_evo_core` budget plumbing.

Also verify `SMCO_evo.R`'s trailing source resolves when loaded the way `run_highdim_r.R` loads it:
```bash
Rscript -e 'source("vendor/SMCO_R/main/evaluation_budget.R"); source("vendor/SMCO_R/main/SMCO.R"); source("vendor/SMCO_R/main/SMCO_evo.R"); cat("loaded; SP=", exists("run_evolutionary_states", mode="function"), " RS=", exists("run_evolutionary_restarts", mode="function"), "\n")'
```
Expected: `loaded; SP= TRUE  RS= TRUE`.

---

## Task 5: Cross-language numerical agreement (R-SP vs Python-SP)

**Files:**
- Modify: `vendor/SMCO_R/main/tests/test_evolution_semantics.R` (append cross-language check)

- [ ] **Step 1: Generate the Python reference values**

Run a one-off Python snippet to capture Python-SP reference values for a small deterministic setup, and record them as constants in the R test:

```bash
.venv/bin/python -c "
import numpy as np
from smco import smco_evo
f = lambda x: -float(np.sum((x - 0.1) ** 2))
starts = np.array([[-0.8,-0.3],[0.2,0.7],[-0.5,0.4],[-0.1,0.6]])
r = smco_evo(f, [-1,-1],[1,1], start_points=starts, n_starts=4, iter_max=40,
             evolution_points=(0.5,0.75), elimination_rate=0.25,
             evolution_strategy='rand1bin', de_factor=0.8, de_crossover=0.7,
             seed=123, tol_conv=1e-12, buffer_rand=False, bounds_buffer=0.05,
             partial_option='center', use_runmax=True, state_semantics='state_preserving')
print('f_optimal=', repr(float(r.best_result.f_optimal)))
print('x_optimal=', repr([float(v) for v in r.best_result.x_optimal]))
print('fe_used=', r.summary.get('fe') and r.summary['fe'].get('fe_used'))
"
```

Record the printed `f_optimal` / `x_optimal` into the R test constants below. (Python version: capture `python --version` and note it in the test comment.)

- [ ] **Step 2: Append the cross-language assertion**

Append to `vendor/SMCO_R/main/tests/test_evolution_semantics.R` (substitute the captured values for the placeholders in `PY_SP_F` / `PY_SP_X`):

```r
# ---- 15. R-SP agrees with Python-SP on a small deterministic trajectory ----
# Reference captured from Python smco_evo(state_semantics='state_preserving')
# with the SAME starts/params (buffer_rand=FALSE so both are deterministic).
# Regenerate with: scripts reference in plan Task 5 Step 1.
PY_SP_F <- -0.0   # REPLACE with captured f_optimal
PY_SP_X <- c(0.1, 0.1)  # REPLACE with captured x_optimal
pyx_sp <- matrix(c(-0.8,-0.3, 0.2,0.7, -0.5,0.4, -0.1,0.6), ncol = 2, byrow = TRUE)
ctrl_sp <- list(bounds_buffer = 0.05, buffer_rand = FALSE, tol_conv = 1e-12,
                partial_option = "center", use_runmax = TRUE, iter_nstart = 4L, seed = 123)
rsp <- run_evolutionary_states(ff, c(-1,-1), c(1,1), pyx_sp, ctrl_sp,
                              evolution_points = c(0.5, 0.75), elimination_rate = 0.25,
                              evolution_strategy = "rand1bin", de_factor = 0.8,
                              de_crossover = 0.7, iter_max = 40L, iter_boost = 0L)
rsp_best <- rsp$results[[which.max(sapply(rsp$results, function(r) r$f_optimal))]]
check(approx(rsp_best$f_optimal, PY_SP_F, tol = 1e-6),
      paste("R-SP f_optimal", rsp_best$f_optimal, "!= Python", PY_SP_F))
check(all(approx(rsp_best$x_optimal, PY_SP_X, tol = 1e-6)),
      "R-SP x_optimal != Python")
```

- [ ] **Step 3: Run the cross-language test**

Run: `Rscript vendor/SMCO_R/main/tests/test_evolution_semantics.R`
Expected: PASS through section 15.

If it fails on the values: first confirm `buffer_rand=FALSE` on both sides and identical `iter_nstart`/`iter_max`/`evolution_points`. A divergence after the first boundary indicates the SP accumulator continuation or the replacement `birth_iteration` differs from Python — diff `run_evolutionary_states` against `_run_evolutionary_states` (optimizer.py:1091).

Note: `ff <- function(x) -sum(x^2)` differs from the Python lambda `(x-0.1)^2`. For the cross-language check BOTH sides must use the SAME objective. Either (a) change the R `ff` in this section to `ff_cl <- function(x) -sum((x-0.1)^2)` and use it, or (b) regenerate the Python reference with `-sum(x^2)`. Pick (a) and keep the Python reference as captured above.

---

## Task 6: Gate B verification + single commit

- [ ] **Step 1: Run the full Gate B command set**

```bash
.venv/bin/python -m pytest tests/test_evolution_semantics.py -v
Rscript vendor/SMCO_R/main/tests/test_evolution_semantics.R
Rscript vendor/SMCO_R/main/tests/test_evaluation_budget.R
.venv/bin/python -m pytest -q
```
Expected: all green. `test_evolution_semantics.py` is the existing Python suite (unchanged by this Task); the R scripts are the new + existing R suites; the final `-q` is the full Python regression.

- [ ] **Step 2: Verify no frozen/protected files were touched**

```bash
git status --short
git diff --name-only | grep -E "vendor/SMCO_R/v1\.0\.0/|result/rerun-2026-07-20/|result/highdim-full-rerun-2026-07-20/|result/r-highdim-rerun-2026-07-24/" && echo "PROTECTED FILE TOUCHED — abort" || echo "protected files clean"
```
Expected changed files only: `vendor/SMCO_R/main/SMCO_evo_stateful.R` (new), `vendor/SMCO_R/main/SMCO_evo.R` (modified), `vendor/SMCO_R/main/tests/test_evolution_semantics.R` (new), plus the spec/plan docs. No protected paths.

- [ ] **Step 3: Commit everything as one feat**

```bash
git add vendor/SMCO_R/main/SMCO_evo_stateful.R \
        vendor/SMCO_R/main/SMCO_evo.R \
        vendor/SMCO_R/main/tests/test_evolution_semantics.R \
        docs/superpowers/specs/2026-07-28-smco-evo-r-stateful-design.md \
        docs/superpowers/plans/2026-07-28-smco-evo-r-stateful.md
git commit -m "feat: add stateful evolutionary scheduler in r"
```

- [ ] **Step 4: Update campaign memory**

Update `/amax/math/.claude/projects/-amax-math-code-SMCO/memory/smco-evo-highdim-paper-campaign.md`:
- Mark Task 4 done, Gate B passed (record the commit hash).
- Note the key facts for future agents: R EVO now supports `state_semantics` (default `restart` for legacy compat; SP via `initialize_smco_state`/`run_smco_state_until`); RS runner rewritten on the stateful primitives (no longer uses `SMCO_single`, so init is tagged `restart_initialization`); next is Task 5 (cross-language trace).

---

## Self-Review

**1. Spec coverage:**
- §2.1 `SMCO_evo_stateful.R` + three primitives → Task 1 ✓
- §2.2 rename old logic to RS + archive + restart_initialization + history tag → Task 3 ✓
- §2.3 real SP `run_evolutionary_states` (survivor state carries) → Task 2 ✓
- §2.4 `state_semantics` on three entry points → Task 4 ✓
- §2.5 `test_evolution_semantics.R` covering plan list → Tasks 1-5 ✓
- §2.6 Gate B → Task 6 ✓
- §4.1 (no SMCO_single refactor) → respected; SMCO.R untouched ✓
- §4.2 (default restart) → Task 4 entry points + test 14 ✓
- §4.3 (state as list) → Task 1 ✓
- §5 source chain → Task 4 Step 5 ✓
- §10 cross-language agreement → Task 5 ✓

**2. Placeholder scan:** Task 5 `PY_SP_F`/`PY_SP_X` are intentionally capture-at-runtime constants (the plan tells the engineer exactly how to generate them); flagged as REPLACE, not left vague. No other TBD/TODO. The trailing-source block in Task 4 Step 5 offers a primary and a simpler fallback with a verification step — not a placeholder, a guarded decision.

**3. Type/name consistency:** `initialize_smco_state` / `run_smco_state_until` / `smco_state_to_result` / `state_ranking_value` / `state_ranking_point` / `.clip_and_promote` / `run_evolutionary_states` / `run_evolutionary_restarts` — used identically across Tasks 1-5. State fields `s_value`/`current_n`/`initial_n`/`birth_iteration`/`stopped_target_n` match between `initialize_smco_state`, `run_smco_state_until`, and both runners. `state_semantics` threaded through entry → core → branch → runner consistently.

## Implementation Notes (executed 2026-07-28)

Deviations from the step code above, discovered during TDD execution and applied to the shipped files. Gate B green (Python 228 + R both suites).

1. **Test helper `approx`**: `isTRUE(abs(a-b)<=tol)` → `all(abs(a-b)<=tol)`. `isTRUE` returns FALSE for length>1 logicals, so vector (x_optimal) comparisons silently failed even when values matched.
2. **SP/RS `parents` construction**: `t(sapply(states[idx], rank_pt))` → `do.call(rbind, lapply(states[idx], rank_pt))`. In 1D, `sapply` returns a vector (not a matrix), so `t()` gave a 1×n matrix → `nrow(parents)=1 < 4` → Sobol fallback → `qrng` error (qrng absent on this host). `do.call(rbind, ...)` yields n×d for d=1 and d=2 alike (mirrors Python `np.vstack`).
3. **Test `sp` matrix**: 4 → 8 starts. `rand1bin` needs ≥4 survivors; with 4 starts elimination left 3 survivors (<4) → Sobol fallback. 8 starts → 6 survivors. `ctrl$iter_nstart` 4 → 8; section-4 count assertion 4 → 8.
4. **Task 2 section 3 (staged == one-shot)**: the second `run_smco_state_until` call uses `iter_target = 10` (same absolute target as the one-shot), NOT 5. `iter_target` is absolute (`target_n = initial_n + iter_target`); the SP scheduler passes `iter_target = boundary - birth`, so a continuation call must name the global target, not an increment.
5. **Task 5 cross-language config**: use `buffer_rand=FALSE` + `evolution_points=(0.999,)` (no boundary). Python `np.random` ≠ R RNG, so any boundary-bearing config diverges on DE replacements; with no boundary the SP runner is N independent `SMCO_single` trajectories and the comparison is exact (Gate A already proved SMCO_single cross-language parity). Objective unified to `-sum(x^2)` (reuse R `ff`). Captured reference: `f=-0.0001020408163265301`, `x=[0.0071428571428571175, -0.007142857142857133]`.
6. **Trailing source (Task 4 Step 5)**: implemented as a `local({})` block that walks `sys.frame()` for the `ofile` carried by `source()` (probe confirmed R 4.3.2 sets `ofile` on frame 1 of the sourced file), with `--file=` fallback for `Rscript` direct execution. The simpler `sys.frame(1)$ofile` form in the plan is unreliable across call depths; the loop form is what shipped.
