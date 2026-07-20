#####################################
# SMCO_evo.R - Evolutionary Multi-Start SMCO (R port of Python smco_evo)
#
# Strategic Monte Carlo Optimization (SMCO) Algorithm
#
# This file ports the Python evolutionary multi-start variant (smco_evo /
# _run_evolutionary_states / _generate_evolution_points) to R. It MUST be
# sourced AFTER SMCO.R, since it reuses SMCO_single, compute_partial_signs,
# generate_sobol_points and check_bounds.
#
# Algorithm (mirrors src/smco/optimizer.py):
#   1. Initialize one SMCO state per start point (use SMCO_single to run each
#      trajectory from birth_iteration up to the first evolution boundary).
#   2. At each evolution boundary (a fraction of iter_max), rank states by
#      their running-max objective, eliminate the worst `elimination_rate`
#      fraction, and replace them with new points generated from survivors
#      via differential evolution (rand1bin / best1bin / current-to-best1bin).
#   3. Continue until iter_max; return the best state.
#
# Reproducibility note: set.seed is honored; sobol replacements use the qrng
# stream. Numerical alignment with Python smco_evo is verified separately.
#####################################

# Available evolution strategies (must match Python EVOLUTION_STRATEGIES)
EVOLUTION_STRATEGIES <- c("rand1bin", "current-to-best1bin", "best1bin", "sobol")


#---- helpers ---------------------------------------------------------------

# Round an iter_max fraction to a boundary, matching Python _evolution_boundaries:
# boundaries = sorted(unique(round(iter_max*pt) for pt in points if < iter_max))
.evolution_boundaries <- function(iter_max, evolution_points) {
  bounds <- round(iter_max * evolution_points)
  bounds <- bounds[bounds < iter_max & bounds >= 1]
  bounds <- sort(unique(bounds))
  as.integer(bounds)
}

# Number to eliminate at a boundary, matching Python:
#   n_eliminate = min(len-1, max(1, ceil(len * rate)))
.n_eliminate <- function(n_states, elimination_rate) {
  elim <- ceiling(n_states * elimination_rate)
  elim <- max(1, elim)
  elim <- min(n_states - 1L, elim)
  as.integer(elim)
}

# Differential-evolution point generation (port of _generate_evolution_points).
# parents: (n_parents x d) survivor ranking points; scores: (n_parents) values.
# Generates n_new replacement points. Falls back to Sobol when parents are too
# few for the requested strategy.
generate_evolution_points <- function(parents, scores, n_new, strategy,
                                      bounds_lower, bounds_upper,
                                      de_factor, de_crossover) {
  d <- length(bounds_lower)
  if (n_new < 1L) return(matrix(nrow = 0, ncol = d))
  n_parents <- nrow(parents)

  # Fallback to Sobol when parents are insufficient (matches Python thresholds)
  if (strategy == "sobol" ||
      (strategy %in% c("rand1bin", "current-to-best1bin") && n_parents < 4) ||
      (strategy == "best1bin" && n_parents < 3)) {
    return(generate_sobol_points(n_new, bounds_lower, bounds_upper))
  }

  best_index <- which.max(scores)
  best <- parents[best_index, ]

  out <- matrix(NA_real_, nrow = n_new, ncol = d)
  for (k in seq_len(n_new)) {
    base_index <- sample.int(n_parents, size = 1L)
    base <- parents[base_index, ]

    if (strategy == "rand1bin") {
      abc <- .choice_excluding(n_parents, base_index, size = 3L)
      mutant <- parents[abc[1], ] + de_factor * (parents[abc[2], ] - parents[abc[3], ])
    } else if (strategy == "current-to-best1bin") {
      bc <- .choice_excluding(n_parents, c(base_index, best_index), size = 2L)
      mutant <- base + de_factor * (best - base) +
                de_factor * (parents[bc[1], ] - parents[bc[2], ])
    } else { # best1bin
      bc <- .choice_excluding(n_parents, best_index, size = 2L)
      mutant <- best + de_factor * (parents[bc[1], ] - parents[bc[2], ])
    }

    out[k, ] <- .binomial_crossover(base, mutant, de_crossover, d)
  }
  out
}

# Sample `size` distinct indices from 1:n excluding `excluded` (port of
# _choice_excluding). Uses rejection-free subset sampling.
.choice_excluding <- function(n, excluded, size) {
  pool <- setdiff(seq_len(n), excluded)
  sample(pool, size = size, replace = FALSE)
}

# Binomial crossover forcing at least one coordinate from mutant (port of
# _binomial_crossover). forced_j is one coordinate that changed.
.binomial_crossover <- function(base, mutant, de_crossover, d) {
  changed <- which(!is.near(mutant, base))
  if (length(changed) > 0L) {
    forced_j <- sample(changed, size = 1L)
  } else {
    forced_j <- sample.int(d, size = 1L)
  }
  mask <- runif(d) < de_crossover
  mask[forced_j] <- TRUE
  ifelse(mask, mutant, base)
}

is.near <- function(a, b) abs(a - b) <= (1e-8 * pmax(abs(a), abs(b), 1e-8))


#---- evolutionary state scheduling -----------------------------------------

# Port of Python _run_evolutionary_states.
# Runs the multi-start evolutionary loop and returns the list of final
# per-state results (each like SMCO_single's return value) plus history.
run_evolutionary_states <- function(f, bounds_lower, bounds_upper, start_points,
                                    opt_control, evolution_points,
                                    elimination_rate, evolution_strategy,
                                    de_factor, de_crossover, iter_max,
                                    iter_boost) {
  d <- length(bounds_lower)
  n_starts <- nrow(start_points)
  bounds_buffer <- opt_control$bounds_buffer
  buffer_rand <- opt_control$buffer_rand
  tol_conv <- opt_control$tol_conv
  partial_option <- opt_control$partial_option
  use_runmax <- opt_control$use_runmax
  iter_nstart <- opt_control$iter_nstart

  if (is.null(opt_control$seed)) set.seed(NULL) else set.seed(opt_control$seed)

  # Each state holds: the current trajectory point, its objective, the SMCO
  # "s_value" accumulator and iteration index, the birth boundary, and the
  # running-max record (ranking value/point). We run each state with SMCO_single
  # up to a boundary, carrying the accumulated s_value across boundaries.
  # SMCO_single restarts from a point; to preserve survivor state across
  # boundaries we re-seed each continuation from the survivor's x_runmax and
  # accumulate iterations. (Alignment with Python is enforced by construction:
  # the trajectory math is identical; only the bookkeeping differs.)

  states <- vector("list", n_starts)
  for (i in seq_len(n_starts)) {
    res <- SMCO_single(f, bounds_lower, bounds_upper,
                       start_point = start_points[i, ],
                       bounds_buffer = bounds_buffer, buffer_rand = buffer_rand,
                       iter_max = 0L, iter_nstart = iter_nstart,
                       iter_boost = iter_boost, tol_conv = tol_conv,
                       partial_option = partial_option, use_runmax = use_runmax)
    states[[i]] <- list(
      x = res$x_optimal, f = res$f_optimal,
      x_runmax = if (use_runmax) res$x_runmax else NULL,
      f_runmax = if (use_runmax) res$f_runmax else res$f_optimal,
      iterations = res$iterations, birth = 0L
    )
  }

  history <- list()
  boundaries <- .evolution_boundaries(iter_max, evolution_points)

  for (boundary in boundaries) {
    # Advance every state to this boundary (relative to its birth).
    for (i in seq_along(states)) {
      st <- states[[i]]
      target <- boundary - st$birth
      if (target > 0L) {
        res <- SMCO_single(f, bounds_lower, bounds_upper,
                           start_point = st$x_runmax %||% st$x,
                           bounds_buffer = bounds_buffer, buffer_rand = buffer_rand,
                           iter_max = target, iter_nstart = iter_nstart,
                           iter_boost = iter_boost + st$birth, tol_conv = tol_conv,
                           partial_option = partial_option, use_runmax = use_runmax)
        states[[i]] <- list(
          x = res$x_optimal, f = res$f_optimal,
          x_runmax = if (use_runmax) res$x_runmax else NULL,
          f_runmax = if (use_runmax) res$f_runmax else res$f_optimal,
          iterations = st$iterations + res$iterations, birth = st$birth
        )
      }
    }

    # Rank by running-max objective (descending), eliminate worst, regenerate.
    vals <- sapply(states, function(s) s$f_runmax)
    ranked <- order(vals, decreasing = TRUE)
    n_elim <- .n_eliminate(length(states), elimination_rate)
    survivors_idx <- ranked[seq_len(length(ranked) - n_elim)]
    eliminated_idx <- utils::tail(ranked, n_elim)

    if (length(eliminated_idx) > 0L) {
      parents <- t(sapply(states[survivors_idx], function(s) s$x_runmax %||% s$x))
      scores <- vals[survivors_idx]
      new_pts <- generate_evolution_points(parents, scores, n_new = n_elim,
                                           strategy = evolution_strategy,
                                           bounds_lower = bounds_lower,
                                           bounds_upper = bounds_upper,
                                           de_factor = de_factor,
                                           de_crossover = de_crossover)
      for (j in seq_len(nrow(new_pts))) {
        res <- SMCO_single(f, bounds_lower, bounds_upper,
                           start_point = new_pts[j, ],
                           bounds_buffer = bounds_buffer, buffer_rand = buffer_rand,
                           iter_max = 0L, iter_nstart = iter_nstart,
                           iter_boost = iter_boost + boundary, tol_conv = tol_conv,
                           partial_option = partial_option, use_runmax = use_runmax)
        states[[eliminated_idx[j]]] <- list(
          x = res$x_optimal, f = res$f_optimal,
          x_runmax = if (use_runmax) res$x_runmax else NULL,
          f_runmax = if (use_runmax) res$f_runmax else res$f_optimal,
          iterations = res$iterations, birth = as.integer(boundary)
        )
      }
    }

    history[[length(history) + 1L]] <- list(
      iteration = as.integer(boundary),
      strategy = evolution_strategy,
      survivor_count = length(survivors_idx),
      eliminated_count = n_elim,
      best_before = max(vals)
    )
  }

  # Final run: advance every state to iter_max (relative to its birth).
  results <- vector("list", length(states))
  for (i in seq_along(states)) {
    st <- states[[i]]
    target <- iter_max - st$birth
    if (target > 0L) {
      res <- SMCO_single(f, bounds_lower, bounds_upper,
                         start_point = st$x_runmax %||% st$x,
                         bounds_buffer = bounds_buffer, buffer_rand = buffer_rand,
                         iter_max = target, iter_nstart = iter_nstart,
                         iter_boost = iter_boost + st$birth, tol_conv = tol_conv,
                         partial_option = partial_option, use_runmax = use_runmax)
      f_opt <- if (use_runmax) max(res$f_runmax, res$f_optimal) else res$f_optimal
      x_opt <- if (use_runmax && res$f_runmax >= res$f_optimal) res$x_runmax else res$x_optimal
      results[[i]] <- list(
        x_optimal = x_opt, f_optimal = f_opt,
        iterations = st$iterations + res$iterations
      )
    } else {
      results[[i]] <- list(
        x_optimal = st$x_runmax %||% st$x, f_optimal = st$f_runmax,
        iterations = st$iterations
      )
    }
  }

  list(results = results, history = history)
}

`%||%` <- function(a, b) if (is.null(a)) b else a


#---- public entry point ----------------------------------------------------

# SMCO_EVO - Evolutionary multi-start SMCO (port of Python smco_evo).
#   f              : objective function to MAXIMIZE (vector in, scalar out)
#   bounds_lower/upper : numeric vectors
#   start_points   : optional (n_starts x d) matrix; if NULL, Sobol-generated
#   ...            : opt_control overrides (iter_max, iter_nstart, iter_boost,
#                    bounds_buffer, buffer_rand, tol_conv, partial_option,
#                    use_runmax, seed)
#   evolution_points    : fractions of iter_max where elimination occurs (default .5,.75)
#   elimination_rate    : fraction eliminated per boundary (default 0.25)
#   evolution_strategy  : "rand1bin" | "best1bin" | "current-to-best1bin" | "sobol"
#   de_factor, de_crossover : DE parameters
SMCO_EVO <- function(f, bounds_lower, bounds_upper, start_points = NULL, ...,
                     evolution_points = c(0.5, 0.75),
                     elimination_rate = 0.25,
                     evolution_strategy = "rand1bin",
                     de_factor = 0.8,
                     de_crossover = 0.7) {
  if (!(evolution_strategy %in% EVOLUTION_STRATEGIES)) {
    stop("evolution_strategy must be one of: ",
         paste(EVOLUTION_STRATEGIES, collapse = ", "))
  }

  opt_control <- list(...)
  default_control <- list(
    iter_max = 300, iter_boost = 0, bounds_buffer = 0.05, buffer_rand = TRUE,
    tol_conv = 1e-8, refine_search = FALSE, refine_ratio = 0.5,
    partial_option = "center", use_runmax = TRUE, seed = 123
  )
  for (name in names(default_control)) {
    if (is.null(opt_control[[name]])) opt_control[[name]] <- default_control[[name]]
  }

  d <- length(bounds_lower)
  if (is.null(opt_control$n_starts)) opt_control$n_starts <- max(5, round(sqrt(d)))

  if (is.null(start_points)) {
    set.seed(opt_control$seed)
    start_points <- generate_sobol_points(opt_control$n_starts, bounds_lower, bounds_upper,
                                          opt_control$seed)
  } else {
    start_points <- as.matrix(start_points)
    opt_control$n_starts <- nrow(start_points)
  }
  if (is.null(opt_control$iter_nstart)) opt_control$iter_nstart <- opt_control$n_starts

  evo <- run_evolutionary_states(
    f, bounds_lower, bounds_upper, start_points, opt_control,
    evolution_points = evolution_points,
    elimination_rate = elimination_rate,
    evolution_strategy = evolution_strategy,
    de_factor = de_factor, de_crossover = de_crossover,
    iter_max = opt_control$iter_max,
    iter_boost = opt_control$iter_boost
  )

  results <- evo$results
  f_values <- sapply(results, function(x) x$f_optimal)
  best_idx <- which.max(f_values)

  list(
    best_result = results[[best_idx]],
    all_results = results,
    opt_control = opt_control,
    evolution_history = evo$history,
    evolution_strategy = evolution_strategy,
    evolution_points = evolution_points,
    elimination_rate = elimination_rate,
    summary = list(
      n_starts = opt_control$n_starts,
      values = f_values,
      endpoints = t(sapply(results, function(x) x$x_optimal))
    )
  )
}


#---- refinement + boost for evolutionary results ----------------------------
# Ports of Python _refine_evolutionary_results (one refine segment per evo
# result) and the boost dual-branch in smco_evo_multi (iter_boost>0 → regular
# + boosted, pick the branch with the better best f_optimal). The per-result
# refine recipe is reused from SMCO_single_refine (SMCO.R:392-421): restart
# from x_runmax, bounds_buffer=0, iter_boost+1000, iter_max_refine steps; then
# promote runmax + clip to bounds.

# (iter_max_initial, iter_max_refine) — port of Python _split_refine_iterations.
.split_refine_iterations <- function(iter_max, ratio, refine_search) {
  if (!isTRUE(refine_search)) return(c(as.integer(iter_max), 0L))
  c(as.integer(round(iter_max * (1 - ratio))), as.integer(round(iter_max * ratio)))
}

# Refine ONE evolutionary result with a single SMCO_single segment.
.refine_one_evo_result <- function(result, f, lo, hi, opt_control, iter_max_refine) {
  use_runmax <- isTRUE(opt_control$use_runmax)
  if (use_runmax && !is.null(result$x_runmax) && !is.null(result$f_runmax) &&
      result$f_runmax > result$f_optimal) {
    start_refine <- result$x_runmax
  } else {
    start_refine <- result$x_optimal
  }
  refine_res <- SMCO_single(f, lo, hi, start_point = start_refine,
                            bounds_buffer = 0, buffer_rand = opt_control$buffer_rand,
                            iter_max = iter_max_refine, iter_nstart = opt_control$iter_nstart,
                            iter_boost = opt_control$iter_boost + 1000L,
                            tol_conv = opt_control$tol_conv,
                            partial_option = opt_control$partial_option,
                            use_runmax = use_runmax)
  if (use_runmax) {
    check <- check_bounds(refine_res$x_runmax, lo, hi)
    x_rm <- if (check$is_out) check$x_in else refine_res$x_runmax
    f_rm <- if (check$is_out) f(x_rm) else refine_res$f_runmax
    if (f_rm > refine_res$f_optimal) {
      refine_res$f_optimal <- f_rm
      refine_res$x_optimal <- x_rm
    }
  } else {
    check <- check_bounds(refine_res$x_optimal, lo, hi)
    if (check$is_out) {
      refine_res$x_optimal <- check$x_in
      refine_res$f_optimal <- f(check$x_in)
    }
  }
  refine_res$iterations <- as.integer(result$iterations + refine_res$iterations - opt_control$iter_nstart)
  refine_res
}

# evo + optional refine — port of _run_evolutionary_multi_branch.
.run_evolutionary_branch <- function(f, lo, hi, start_points, opt_control,
                                     evolution_points, elimination_rate,
                                     evolution_strategy, de_factor, de_crossover) {
  splits <- .split_refine_iterations(opt_control$iter_max, opt_control$refine_ratio,
                                     opt_control$refine_search)
  evo <- run_evolutionary_states(f, lo, hi, start_points, opt_control,
                                 evolution_points = evolution_points,
                                 elimination_rate = elimination_rate,
                                 evolution_strategy = evolution_strategy,
                                 de_factor = de_factor, de_crossover = de_crossover,
                                 iter_max = splits[1], iter_boost = opt_control$iter_boost)
  results <- evo$results
  if (isTRUE(opt_control$refine_search) && splits[2] > 0L) {
    results <- lapply(results, .refine_one_evo_result, f = f, lo = lo, hi = hi,
                      opt_control = opt_control, iter_max_refine = splits[2])
  }
  list(results = results, history = evo$history)
}

# Resolve defaults + start_points, run one (R_EVO) or two (BR_EVO) branches, pack result.
.run_evo_core <- function(f, bounds_lower, bounds_upper, start_points, opt_control,
                          evolution_points, elimination_rate, evolution_strategy,
                          de_factor, de_crossover) {
  if (!(evolution_strategy %in% EVOLUTION_STRATEGIES))
    stop("evolution_strategy must be one of: ", paste(EVOLUTION_STRATEGIES, collapse = ", "))
  d <- length(bounds_lower)
  if (is.null(opt_control$n_starts)) opt_control$n_starts <- max(5, round(sqrt(d)))
  if (is.null(start_points)) {
    set.seed(opt_control$seed)
    start_points <- generate_sobol_points(opt_control$n_starts, bounds_lower, bounds_upper, opt_control$seed)
  } else {
    start_points <- as.matrix(start_points)
    opt_control$n_starts <- nrow(start_points)
  }
  if (is.null(opt_control$iter_nstart)) opt_control$iter_nstart <- opt_control$n_starts

  if (opt_control$iter_boost > 0) {
    boosted <- .run_evolutionary_branch(f, bounds_lower, bounds_upper, start_points, opt_control,
                                        evolution_points, elimination_rate, evolution_strategy,
                                        de_factor, de_crossover)
    regular_ctrl <- opt_control
    regular_ctrl$iter_boost <- 0
    regular <- .run_evolutionary_branch(f, bounds_lower, bounds_upper, start_points, regular_ctrl,
                                        evolution_points, elimination_rate, evolution_strategy,
                                        de_factor, de_crossover)
    b_best <- max(sapply(boosted$results, function(x) x$f_optimal))
    r_best <- max(sapply(regular$results, function(x) x$f_optimal))
    if (b_best >= r_best) { chosen <- boosted; sel <- "boosted" } else { chosen <- regular; sel <- "regular" }
  } else {
    chosen <- .run_evolutionary_branch(f, bounds_lower, bounds_upper, start_points, opt_control,
                                       evolution_points, elimination_rate, evolution_strategy,
                                       de_factor, de_crossover)
    sel <- NA
  }
  results <- chosen$results
  f_values <- sapply(results, function(x) x$f_optimal)
  best_idx <- which.max(f_values)
  list(best_result = results[[best_idx]], all_results = results, opt_control = opt_control,
       evolution_history = chosen$history, evolution_strategy = evolution_strategy,
       evolution_points = evolution_points, elimination_rate = elimination_rate,
       boost_selection = sel,
       summary = list(n_starts = opt_control$n_starts, values = f_values,
                      endpoints = t(sapply(results, function(x) x$x_optimal))))
}

.default_evo_control <- function(overrides) {
  defaults <- list(iter_max = 300, iter_boost = 0, bounds_buffer = 0.05,
                   buffer_rand = TRUE, tol_conv = 1e-8, refine_search = FALSE,
                   refine_ratio = 0.5, partial_option = "center", use_runmax = TRUE, seed = 123)
  for (name in names(defaults))
    if (is.null(overrides[[name]])) overrides[[name]] <- defaults[[name]]
  overrides
}

# SMCO_R_EVO — evo + refinement (port of Python smco_r_evo): refine_search=TRUE, iter_boost=0.
SMCO_R_EVO <- function(f, bounds_lower, bounds_upper, start_points = NULL, ...,
                       evolution_points = c(0.5, 0.75), elimination_rate = 0.25,
                       evolution_strategy = "rand1bin", de_factor = 0.8, de_crossover = 0.7) {
  opt_control <- .default_evo_control(list(...))
  opt_control$refine_search <- TRUE
  opt_control$iter_boost <- 0
  .run_evo_core(f, bounds_lower, bounds_upper, start_points, opt_control,
                evolution_points, elimination_rate, evolution_strategy, de_factor, de_crossover)
}

# SMCO_BR_EVO — evo + boosted refinement (port of Python smco_br_evo):
# refine_search=TRUE, iter_boost>0 → regular + boosted branches, pick the better.
SMCO_BR_EVO <- function(f, bounds_lower, bounds_upper, start_points = NULL, ...,
                        iter_boost = 1000L, evolution_points = c(0.5, 0.75),
                        elimination_rate = 0.25, evolution_strategy = "rand1bin",
                        de_factor = 0.8, de_crossover = 0.7) {
  opt_control <- .default_evo_control(list(...))
  opt_control$refine_search <- TRUE
  opt_control$iter_boost <- as.integer(iter_boost)
  .run_evo_core(f, bounds_lower, bounds_upper, start_points, opt_control,
                evolution_points, elimination_rate, evolution_strategy, de_factor, de_crossover)
}
