# Strategic Monte Carlo Optimization (SMCO) Algorithm
# Version 1.1.0 - Quality Enhancements
######################################

# "Optimization via Strategic Law of Large Numbers"
# By: Xiaohong Chen, Zengjing Chen, Wayne Yuan Gao, Xiaodong Yan, and Guodong Zhang
#
# Citation: X. Chen, Z. Chen, W.Y. Gao, X. Yan, & G. Zhang, 
# Optimization via the strategic law of large numbers, 
# Proc. Natl. Acad. Sci. U.S.A. 123 (4) e2519845123,
# https://doi.org/10.1073/pnas.2519845123 (2026).
#
# GitHub Repository Maintained by: Wayne Yuan Gao
# Version: 1.1.0
# Release Date: January 25, 2026
######################################

# This Rscript is a self-contained R Implementation of the SMCO Algorithm
#
# ALGORITHM VARIANTS
# ==================
#
# 1. SMCO (Basic) - Single-phase optimization
#    - Best for: Quick exploration, simple problems
#    - Call: SMCO(f, bounds_lower, bounds_upper)
#
# 2. SMCO_R (Refinement) - Two-phase: exploration + local refinement
#    - Best for: Problems where local optimization matters
#    - Call: SMCO_R(f, bounds_lower, bounds_upper)
#
# 3. SMCO_BR (Boosted Refinement) - Regular + boosted searches
#    - Best for: Challenging multimodal problems, high accuracy
#    - Call: SMCO_BR(f, bounds_lower, bounds_upper)
#
# SMCO_multi() provides full control over all parameters.
#
# FUNCTIONS IN THIS FILE
# ======================
#
# Convenience wrapper functions:
# (1) "SMCO": basic algorithm (refine_search=FALSE, iter_boost=0)
# (2) "SMCO_R": with refinement (refine_search=TRUE, iter_boost=0)
# (3) "SMCO_BR": boosted refinement (refine_search=TRUE, iter_boost>0)
#
# Core optimizer functions:
# (4) "SMCO_multi": runs SMCO from multiple starting points (full parameter control)
# (5) "SMCO_single": executes the basic iterations of the SMCO algorithm
# (6) "SMCO_single_refine": calls SMCO_single to run refined search (with options)
# (7) "SMCO_single_boost": calls SMCO_refine to run additional boosted search (with options)
#
# Helper functions:
# (8) "generate_sobol_points": generate multiple Sobol starting points
# (9) "check_bounds": check and enforce bound constraints
# (10) "compute_partial_signs": compute partial finite differences used in the SMCO algorithm
# (11) "validate_smco_inputs": validate inputs to SMCO_multi
#
# Example usage at the end of Rscript

#####################################
# Input validation function
validate_smco_inputs <- function(f, bounds_lower, bounds_upper, start_points = NULL, opt_control = NULL) {
  errors <- character(0)


  # Check f is a function
  if (!is.function(f)) {
    errors <- c(errors, "'f' must be a function")
  }

  # Check bounds are numeric vectors
  if (!is.numeric(bounds_lower)) {
    errors <- c(errors, "'bounds_lower' must be a numeric vector")
  }
  if (!is.numeric(bounds_upper)) {
    errors <- c(errors, "'bounds_upper' must be a numeric vector")
  }

  # Check bounds have same length
  if (length(bounds_lower) != length(bounds_upper)) {
    errors <- c(errors, "'bounds_lower' and 'bounds_upper' must have the same length")
  }

  # Check bounds_lower < bounds_upper element-wise
  if (is.numeric(bounds_lower) && is.numeric(bounds_upper) &&
      length(bounds_lower) == length(bounds_upper)) {
    if (any(bounds_lower >= bounds_upper)) {
      bad_dims <- which(bounds_lower >= bounds_upper)
      errors <- c(errors, paste0("'bounds_lower' must be strictly less than 'bounds_upper' (violated at dimension(s): ",
                                 paste(bad_dims, collapse = ", "), ")"))
    }
  }

  # Check start_points if provided
  if (!is.null(start_points)) {
    start_points <- as.matrix(start_points)
    if (ncol(start_points) != length(bounds_lower)) {
      errors <- c(errors, paste0("'start_points' must have ", length(bounds_lower),
                                 " columns (one per dimension), but has ", ncol(start_points)))
    }
    if (nrow(start_points) < 1) {
      errors <- c(errors, "'start_points' must have at least one row")
    }
  }

  # Check opt_control parameters if provided
  if (!is.null(opt_control)) {
    if (!is.null(opt_control$n_starts) && (!is.numeric(opt_control$n_starts) || opt_control$n_starts < 1)) {
      errors <- c(errors, "'opt_control$n_starts' must be a positive integer")
    }
    if (!is.null(opt_control$iter_max) && (!is.numeric(opt_control$iter_max) || opt_control$iter_max < 1)) {
      errors <- c(errors, "'opt_control$iter_max' must be a positive integer")
    }
    if (!is.null(opt_control$bounds_buffer) && (!is.numeric(opt_control$bounds_buffer) || opt_control$bounds_buffer < 0)) {
      errors <- c(errors, "'opt_control$bounds_buffer' must be a non-negative number")
    }
    if (!is.null(opt_control$refine_ratio) && (!is.numeric(opt_control$refine_ratio) ||
        opt_control$refine_ratio < 0 || opt_control$refine_ratio > 1)) {
      errors <- c(errors, "'opt_control$refine_ratio' must be between 0 and 1")
    }
    if (!is.null(opt_control$partial_option) && !(opt_control$partial_option %in% c("center", "forward"))) {
      errors <- c(errors, "'opt_control$partial_option' must be 'center' or 'forward'")
    }
  }

  if (length(errors) > 0) {
    stop(paste("Input validation failed:\n  - ", paste(errors, collapse = "\n  - "), sep = ""))
  }

  invisible(TRUE)
}

#####################################
# Generate diverse Sobol starting points (vectorized)
generate_sobol_points <- function(n_starts, bounds_lower, bounds_upper, seed = NULL) {
  if (!requireNamespace("qrng", quietly = TRUE)) {
    stop("Package 'qrng' is required for Sobol point generation. Install it with: install.packages('qrng')")
  }
  if (!is.null(seed)) set.seed(seed)

  d <- length(bounds_lower)
  sobol_points <- qrng::sobol(n_starts, d = d)
  # Vectorized scaling: uses R's column-wise recycling via t() double transpose
  bounds_range <- bounds_upper - bounds_lower
  points <- t(t(sobol_points) * bounds_range + bounds_lower)
  return(points)
}

# Helper function to check bounds
check_bounds <- function(x, bounds_lower, bounds_upper) {
  is_out <- (sum((x > bounds_upper) + (x < bounds_lower)) > 0)
  x_in <- if (is_out) pmax(pmin(x, bounds_upper), bounds_lower) else x
  return(list(is_out = is_out, x_in = x_in))
}

# Helper function to:
# (1) Compute discrete partial derivative signs; and
# (2) Return max function evaluation in addition
# Performance optimized: in-place modification, metadata-based runmax tracking
compute_partial_signs <- function(f, x, fx, h_step, bounds_lower, bounds_upper,
                                  partial_option = "center", use_runmax = TRUE) {
  d <- length(x)

  if (is.null(h_step)) {
    h_step <- rep(1e-8, d)
  }

  partial_signs <- numeric(d)
  f_partial_best <- fx
  # Track best point via metadata to avoid vector copies
  best_j <- 0L
  best_perturbed_value <- NA_real_

  if (partial_option == "center") { # Do two-sided perturbation
    # Modify x in-place rather than creating copies
    for(j in 1:d) {
      x_j_orig <- x[j]  # Save original value

      # Evaluate plus perturbation
      x[j] <- min(x_j_orig + h_step[j], bounds_upper[j])
      f_plus <- f(x)

      # Evaluate minus perturbation
      x[j] <- max(x_j_orig - h_step[j], bounds_lower[j])
      f_minus <- f(x)

      sign_f_diff <- (f_plus > f_minus)
      partial_signs[j] <- sign_f_diff

      # Track running maximum using metadata (no vector copy)
      if (use_runmax) {
        if (sign_f_diff && (f_plus > f_partial_best)) {
          f_partial_best <- f_plus
          best_j <- j
          best_perturbed_value <- min(x_j_orig + h_step[j], bounds_upper[j])
        } else if ((!sign_f_diff) && (f_minus > f_partial_best)) {
          f_partial_best <- f_minus
          best_j <- j
          best_perturbed_value <- max(x_j_orig - h_step[j], bounds_lower[j])
        }
      }

      # Restore original value for next iteration
      x[j] <- x_j_orig
    }
  } else { # if partial_option = "forward", do one-sided perturbation
    for(j in 1:d) {
      x_j_orig <- x[j]  # Save original value

      if (x_j_orig >= bounds_upper[j]) {
        x[j] <- max(x_j_orig - h_step[j], bounds_lower[j])
        f_perturb <- f(x)
        sign_f_diff <- (fx > f_perturb)
      } else {
        x[j] <- min(x_j_orig + h_step[j], bounds_upper[j])
        f_perturb <- f(x)
        sign_f_diff <- (f_perturb > fx)
      }

      partial_signs[j] <- sign_f_diff

      # Track running maximum using metadata (no vector copy)
      if (use_runmax && (f_perturb > f_partial_best)) {
        f_partial_best <- f_perturb
        best_j <- j
        best_perturbed_value <- x[j]
      }

      # Restore original value for next iteration
      x[j] <- x_j_orig
    }
  }

  # Reconstruct x_partial_best from metadata (single copy at end)
  if (best_j > 0L) {
    x_partial_best <- x
    x_partial_best[best_j] <- best_perturbed_value
  } else {
    x_partial_best <- x
  }

  return(list(signs = partial_signs, f_partial_best = f_partial_best, x_partial_best = x_partial_best))
}

# Optimized single-start SMCO function
# Performance optimized: pre-computed h_steps, cached bounds, pre-computed convergence threshold
SMCO_single <- function(f, bounds_lower, bounds_upper, start_point,
                        bounds_buffer, buffer_rand, iter_max, iter_nstart, iter_boost,
                        tol_conv, partial_option, use_runmax)
{
  d <- length(bounds_lower)

  # Initialize
  x_current <- start_point
  f_current <- f(x_current)
  f_runmax <- f_current
  x_runmax <- x_current

  # Initialize loop counters and running sum S
  bounds_diff <- bounds_upper - bounds_lower
  n_boost_1 <- iter_boost + iter_nstart
  n_boost_max <- iter_boost + iter_nstart + iter_max

  S <- start_point * n_boost_1

  # Pre-compute h_step values to avoid repeated division in loop
  # For memory efficiency, only pre-compute if iter_max < 1000
  n_iters <- iter_max + 1
  if (n_iters < 1000) {
    # Pre-compute all h_steps: matrix where column i corresponds to n = n_boost_1 + i - 1
    n_values <- n_boost_1:n_boost_max
    h_steps_matrix <- outer(bounds_diff, 1 / (n_values + 1))
  } else {
    h_steps_matrix <- NULL
  }

  # Pre-compute fixed bounds when buffer_rand = FALSE
  if (!buffer_rand) {
    bound_out_fixed <- bounds_buffer * bounds_diff
    bounds_upper_out_fixed <- bounds_upper + bound_out_fixed
    bounds_lower_out_fixed <- bounds_lower - bound_out_fixed
  }

  # Pre-compute convergence check threshold
  iter_min_check <- n_boost_1 + ceiling(iter_max / 2)

  # Main optimization loop
  for (n in n_boost_1:n_boost_max) {

    # Get h_step from pre-computed matrix or compute on-demand
    if (!is.null(h_steps_matrix)) {
      h_step <- h_steps_matrix[, n - n_boost_1 + 1]
    } else {
      h_step <- bounds_diff / (n + 1)
    }

    partial_sign_results <- compute_partial_signs(f, x_current, f_current, h_step,
                                                  bounds_lower, bounds_upper,
                                                  partial_option, use_runmax)

    # Use pre-computed bounds or compute with random deviation
    if (buffer_rand) {
      bound_out <- bounds_buffer * bounds_diff * runif(d, -1, 1)
      bounds_upper_out <- bounds_upper + bound_out
      bounds_lower_out <- bounds_lower - bound_out
    } else {
      bounds_upper_out <- bounds_upper_out_fixed
      bounds_lower_out <- bounds_lower_out_fixed
    }

    # Update sum S and point x
    Z <- partial_sign_results$signs * bounds_upper_out + (1 - partial_sign_results$signs) * bounds_lower_out
    S <- S + Z
    x_next <- S / (n + 1)
    f_next <- f(x_next)

    if (use_runmax) {
      # Update the running max record
      f_partial_best <- partial_sign_results$f_partial_best
      f_next_best <- max(f_partial_best, f_next)
      if (f_next_best > f_runmax) {
        f_runmax <- f_next_best
        x_runmax <- if (f_partial_best > f_next) partial_sign_results$x_partial_best else x_next
      }
    }
    # Update x_current and f_current
    f_prev <- f_current
    f_current <- f_next
    x_current <- x_next

    # Check convergence after at least 50% of allocated iterations (using pre-computed threshold)
    if ((n >= iter_min_check) && (abs(f_current - f_prev) < tol_conv)) {
      break
    }
  }

  if (use_runmax) {
    return(list(
      x_optimal = x_current,
      f_optimal = f_current,
      iterations = n - iter_boost,
      f_runmax = f_runmax,
      x_runmax = x_runmax
    ))} else {
    return(list(
      x_optimal = x_current,
      f_optimal = f_current,
      iterations = n - iter_boost
    ))
    }
}

# Single-start with refinement search
SMCO_single_refine <- function(f, bounds_lower, bounds_upper, start_point,
                               bounds_buffer, buffer_rand, iter_max, iter_nstart, iter_boost, tol_conv,
                               refine_search, refine_ratio, partial_option, use_runmax)
{
  # Run initial search
  if (refine_search == FALSE) { refine_ratio <- 0 }
  iter_max_initial <- round(iter_max * (1 - refine_ratio), digits = 0)

  result <- SMCO_single(f, bounds_lower, bounds_upper, start_point = start_point,
                        bounds_buffer = bounds_buffer, buffer_rand = buffer_rand,
                        iter_max = iter_max_initial, iter_nstart = iter_nstart, iter_boost = iter_boost,
                        tol_conv = tol_conv, partial_option = partial_option,
                        use_runmax = use_runmax)

  # Handle boundary constraints
  check_optimal <- check_bounds(result$x_optimal, bounds_lower, bounds_upper)
  if (check_optimal$is_out) {
    x_optimal_in <- check_optimal$x_in
    f_optimal_in <- f(x_optimal_in)
    result$x_optimal <- x_optimal_in
    result$f_optimal <- f_optimal_in
  } else {
    x_optimal_in <- result$x_optimal
    f_optimal_in <- result$f_optimal
  }

  if (use_runmax) {
    check_runmax <- check_bounds(result$x_runmax, bounds_lower, bounds_upper)
    if (check_runmax$is_out) {
      x_runmax_in <- check_runmax$x_in
      f_runmax_in <- f(x_runmax_in)
      result$x_runmax <- x_runmax_in
      result$f_runmax <- f_runmax_in
    } else {
      x_runmax_in <- result$x_runmax
      f_runmax_in <- result$f_runmax
    }
  }

  # Run refined search
  if (refine_search) {
    iter_max_refine <- round(iter_max * refine_ratio, digits = 0)

    start_point_refine <- if (use_runmax && (f_runmax_in > f_optimal_in)) x_runmax_in else x_optimal_in
    iter_boost_refine <- iter_boost + 1000

    refine_result <- SMCO_single(f, bounds_lower, bounds_upper,
                                 start_point = start_point_refine,
                                 bounds_buffer = 0, buffer_rand = buffer_rand,
                                 iter_max = iter_max_refine, iter_nstart = iter_nstart, iter_boost = iter_boost_refine,
                                 tol_conv = tol_conv, partial_option = partial_option,
                                 use_runmax = use_runmax)

    if (use_runmax) {
      # Handle boundary constraints for refined result
      f_refine_runmax <- refine_result$f_runmax
      x_refine_runmax <- refine_result$x_runmax
      check_refine_runmax <- check_bounds(x_refine_runmax, bounds_lower, bounds_upper)

      if (check_refine_runmax$is_out) {
        x_refine_runmax <- check_refine_runmax$x_in
        f_refine_runmax <- f(x_refine_runmax)
      }

      if (f_refine_runmax > refine_result$f_optimal) {
        refine_result$f_optimal <- f_refine_runmax
        refine_result$x_optimal <- x_refine_runmax
      }
    }

  } else {

    # If no refine_search, use the original results with bounds checking
    refine_result <- result



    if (use_runmax) {
      refine_result$f_runmax <- f_runmax_in
      refine_result$x_runmax <- x_runmax_in

      if (f_runmax_in > f_optimal_in) {
        refine_result$f_optimal <- f_runmax_in
        refine_result$x_optimal <- x_runmax_in
      } else {
        refine_result$f_optimal <- f_optimal_in
        refine_result$x_optimal <- x_optimal_in
      }
    }
  }

  return(refine_result)
}

# Single-start with/without additional boosted search
SMCO_single_boost <- function(f, bounds_lower, bounds_upper, start_point,
                              bounds_buffer, buffer_rand, iter_max, iter_nstart, iter_boost, tol_conv,
                              refine_search, refine_ratio, partial_option, use_runmax)
{

  # Run regular search with non-boosted starting index n = 1
  refine_result <- SMCO_single_refine(f, bounds_lower, bounds_upper, start_point = start_point,
                                      bounds_buffer = bounds_buffer, buffer_rand = buffer_rand,
                                      iter_max = iter_max, iter_nstart = iter_nstart, iter_boost = 0,
                                      tol_conv = tol_conv, refine_search = refine_search, refine_ratio = refine_ratio,
                                      partial_option = partial_option, use_runmax = use_runmax)

  # Run additional boosted search with "boosted" starting index n = iter_boost
  if (iter_boost > 0) {
    refine_result_boost <- SMCO_single_refine(f, bounds_lower, bounds_upper, start_point = start_point,
                                              bounds_buffer = bounds_buffer, buffer_rand = buffer_rand,
                                              iter_max = iter_max, iter_nstart = iter_nstart,
                                              iter_boost = iter_boost,
                                              tol_conv = tol_conv, refine_search = refine_search, refine_ratio = refine_ratio,
                                              partial_option = partial_option, use_runmax = use_runmax)

    if (refine_result_boost$f_optimal > refine_result$f_optimal) {
      refine_result <- refine_result_boost
    }
  }

  return(refine_result)
}

# Optimized multi-start function
SMCO_multi <- function(f, bounds_lower, bounds_upper, start_points = NULL,
                       opt_control = list(
                         n_starts = NULL,  # defaults to max(5, sqrt(dim)) if not provided
                         iter_max = 500,
                         # iter_nstart: defaults to n_starts if not provided
                         iter_boost = 0,
                         bounds_buffer = 0.05,
                         buffer_rand = FALSE,
                         tol_conv = 1e-8,
                         refine_search = TRUE,
                         refine_ratio = 0.5,
                         partial_option = "center",
                         use_runmax = TRUE,
                         use_parallel = FALSE,
                         verbose = FALSE,
                         seed = 123))
{
  # Validate inputs
  validate_smco_inputs(f, bounds_lower, bounds_upper, start_points, opt_control)

  # Set defaults for any missing opt_control parameters
  # NOTE: n_starts and iter_nstart are intentionally omitted - they default dynamically (see below)
  default_control <- list(
    iter_max = 500,
    iter_boost = 0,
    bounds_buffer = 0.05,
    buffer_rand = FALSE,
    tol_conv = 1e-8,
    refine_search = TRUE,
    refine_ratio = 0.5,
    partial_option = "center",
    use_runmax = TRUE,
    use_parallel = FALSE,
    verbose = FALSE,
    seed = 123
  )
  for (name in names(default_control)) {
    if (is.null(opt_control[[name]])) {
      opt_control[[name]] <- default_control[[name]]
    }
  }

  # Set default n_starts based on dimension if not provided
  if (is.null(opt_control$n_starts)) {
    d <- length(bounds_lower)
    opt_control$n_starts <- max(5, round(sqrt(d)))
  }

  # Generate diverse starting points if not provided
  if (is.null(start_points)) {
    if (!is.null(opt_control$seed)) {
      set.seed(opt_control$seed)
    }
    n_starts <- opt_control$n_starts
    start_points <- generate_sobol_points(n_starts, bounds_lower, bounds_upper, opt_control$seed)
  } else {
    start_points <- as.matrix(start_points)
    n_starts <- nrow(start_points)
  }

  if (is.null(opt_control$iter_nstart)) {
    opt_control$iter_nstart <- n_starts
  }

  verbose <- opt_control$verbose

  # Prepare function to process each starting point
  process_start_point <- function(start_point) {
    SMCO_single_boost(f = f, bounds_lower, bounds_upper, start_point = start_point,
                      bounds_buffer = opt_control$bounds_buffer, buffer_rand = opt_control$buffer_rand,
                      iter_max = opt_control$iter_max, iter_nstart = opt_control$iter_nstart,
                      iter_boost = opt_control$iter_boost,
                      tol_conv = opt_control$tol_conv, refine_search = opt_control$refine_search,
                      refine_ratio = opt_control$refine_ratio, partial_option = opt_control$partial_option,
                      use_runmax = opt_control$use_runmax)
  }

  if (opt_control$use_parallel) {
    # Parallel execution with optimized cluster setup
    n_cores <- min(parallel::detectCores() - 1, n_starts)
    if (verbose) {
      cat("Starting parallel optimization with", n_cores, "cores and", n_starts, "starting points\n")
    }
    cl <- parallel::makeCluster(n_cores)

    # Export necessary functions to cluster
    parallel::clusterExport(cl, c("SMCO_single_boost", "SMCO_single_refine", "SMCO_single",
                                  "check_bounds", "compute_partial_signs"),
                            envir = environment())
    # Export the objective function from the current environment
    parallel::clusterExport(cl, "f", envir = environment())

    # Run optimization in parallel
    results <- parallel::parApply(cl, start_points, 1, process_start_point)

    parallel::stopCluster(cl)

    if (verbose) {
      cat("Parallel optimization completed\n")
    }

  } else {
    # Sequential execution
    if (verbose) {
      cat("Starting sequential optimization with", n_starts, "starting points\n")
    }

    if (verbose) {
      results <- vector("list", n_starts)
      for (i in 1:n_starts) {
        results[[i]] <- process_start_point(start_points[i, ])
        if (i %% max(1, floor(n_starts / 10)) == 0 || i == n_starts) {
          cat(sprintf("Progress: %d/%d starting points completed (%.0f%%)\n",
                      i, n_starts, 100 * i / n_starts))
        }
      }
    } else {
      results <- apply(start_points, 1, process_start_point)
    }
  }

  # Find best result
  f_values <- sapply(results, function(x) x$f_optimal)
  best_idx <- which.max(f_values)
  best_result <- results[[best_idx]]

  # Compile summary statistics
  all_endpoints <- t(sapply(results, function(x) x$x_optimal))
  all_iterations <- sapply(results, function(x) x$iterations)

  if (verbose) {
    cat("\nOptimization complete.\n")
    cat("Best f value:", best_result$f_optimal, "\n")
    cat("Best x:", paste(round(best_result$x_optimal, 6), collapse = ", "), "\n")
  }

  return(list(
    best_result = best_result,
    all_results = results,
    opt_control = opt_control,
    summary = list(
      n_starts = n_starts,
      mean_iterations = mean(all_iterations),
      std_values = sd(f_values),
      endpoints = all_endpoints,
      values = f_values
    )
    ))
}

#####################################
# Convenience wrapper functions for SMCO variants
#####################################

# SMCO - Basic algorithm (refine_search=FALSE, iter_boost=0)
SMCO <- function(f, bounds_lower, bounds_upper, start_points = NULL, ...) {
  user_control <- list(...)
  user_control$refine_search <- FALSE
  user_control$iter_boost <- 0
  SMCO_multi(f, bounds_lower, bounds_upper, start_points, opt_control = user_control)
}

# SMCO_R - With refinement (refine_search=TRUE, iter_boost=0)
SMCO_R <- function(f, bounds_lower, bounds_upper, start_points = NULL, ...) {
  user_control <- list(...)
  user_control$refine_search <- TRUE
  user_control$iter_boost <- 0
  if (is.null(user_control$refine_ratio)) user_control$refine_ratio <- 0.5
  SMCO_multi(f, bounds_lower, bounds_upper, start_points, opt_control = user_control)
}

# SMCO_BR - Boosted refinement (refine_search=TRUE, iter_boost>0)
SMCO_BR <- function(f, bounds_lower, bounds_upper, start_points = NULL,
                    iter_boost = 1000, ...) {
  user_control <- list(...)
  user_control$refine_search <- TRUE
  user_control$iter_boost <- iter_boost
  if (is.null(user_control$refine_ratio)) user_control$refine_ratio <- 0.5
  SMCO_multi(f, bounds_lower, bounds_upper, start_points, opt_control = user_control)
}

# Load the compiler package
library(compiler)

# Compile the core computation functions
validate_smco_inputs <- cmpfun(validate_smco_inputs)
generate_sobol_points <- cmpfun(generate_sobol_points)
compute_partial_signs <- cmpfun(compute_partial_signs)
check_bounds <- cmpfun(check_bounds)
SMCO_single <- cmpfun(SMCO_single)
SMCO_single_refine <- cmpfun(SMCO_single_refine)
SMCO_single_boost <- cmpfun(SMCO_single_boost)
SMCO_multi <- cmpfun(SMCO_multi)
SMCO <- cmpfun(SMCO)
SMCO_R <- cmpfun(SMCO_R)
SMCO_BR <- cmpfun(SMCO_BR)


# Example usage
# run_example <- function()
if (FALSE) {
  # Define the objective function
  f <- function(x) {
    return(-sum(x^2))
  }

  # Define the bounds
  bounds_lower <- c(-5, -5)
  bounds_upper <- c(5, 5)

  # Run the optimization (using defaults: n_starts = max(5, sqrt(dim)), iter_max = 500)
  time_start <- Sys.time()
  result <- SMCO_multi(f, bounds_lower, bounds_upper,
                       opt_control = list(
                         verbose = TRUE,
                         seed = 123))
  time_end <- Sys.time()
  time_run <- difftime(time_end, time_start, units = "secs")

  # Print the best result
  print(time_run)
  print(result$best_result$f_optimal)

}
