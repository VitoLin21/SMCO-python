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
      parents <- do.call(rbind, lapply(states[survivors_idx], state_ranking_point))
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
      parents <- do.call(rbind, lapply(states[survivors_idx], rank_pt))
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
