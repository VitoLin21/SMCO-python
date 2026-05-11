# SMCO Benchmark Script
# Version: 1.1.0
# Release Date: January 25, 2026
#
# Strategic Monte Carlo Optimization (SMCO) Algorithm
# By: Xiaohong Chen, Zengjing Chen, Wayne Yuan Gao, Xiaodong Yan, and Guodong Zhang
#
# Citation: X. Chen, Z. Chen, W.Y. Gao, X. Yan, & G. Zhang, Optimization via the strategic law
#   of large numbers, Proc. Natl. Acad. Sci. U.S.A. 123 (4) e2519845123,
#   https://doi.org/10.1073/pnas.2519845123 (2026).
#
# This script benchmarks all three SMCO variants on standard test functions:
#   - SMCO:    Base algorithm (fastest, good for initial exploration)
#   - SMCO_R:  With refinement phase (balanced speed/accuracy)
#   - SMCO_BR: With boosted refinement (most accurate, slower)
#
# Users can configure the test function, dimension, and shared SMCO parameters below.

#####################################
# USER CONFIGURATION
#####################################

# Test function selection
# Available functions (see testfuncs.R for details):
#   Scalable (any dimension): "Rastrigin", "Ackley", "Griewank", "Rosenbrock",
#                             "Michalewicz", "DixonPrice", "Zakharov", "Qing",
#                             "SqNorm", "-SqNorm", "ANN"
#   Fixed 2D only: "Dropwave", "Shubert", "McCormick", "Bukin6", "Damavandi",
#                  "Mishra06", "EggHolder", "Six-Hump Camel"

name_config <- "Rastrigin"  # Test function name
dim_config <- 50            # Dimension (ignored for fixed 2D functions)

# Benchmark settings
n_replications <- 10        # Number of Monte Carlo replications
random_seed <- 123          # Random seed for reproducibility

# Shared SMCO configuration (variant-specific settings are defined internally)
SMCO_options <- list(
  n_starts = max(5, round(sqrt(dim_config))),  # Number of starting points
  iter_max = 500,           # Maximum iterations per start
  bounds_buffer = 0.05,     # Bounds extension factor
  buffer_rand = TRUE,       # Randomize bounds extension
  tol_conv = 1e-6,          # Convergence tolerance
  partial_option = "center",# "center" (two-sided) or "forward" (one-sided)
  use_runmax = TRUE,        # Track running maximum
  use_parallel = FALSE,     # Enable parallel execution
  verbose = FALSE           # Print progress
)

# ANN-specific configuration (only used if name_config = "ANN")
ANN_options <- list(
  input_dim = 3,            # Number of input features
  hidden_dim = 5,           # Number of hidden neurons
  n = 500,                  # Number of training samples
  use_truth = FALSE,        # Use noiseless target (TRUE) or noisy (FALSE)
  noise_sigma = 1           # Noise standard deviation
)

#####################################
# END USER CONFIGURATION
#####################################

cat("==============================================\n")
cat("SMCO Benchmark - All Variants\n")
cat("==============================================\n\n")

# Load required packages
if (!requireNamespace("qrng", quietly = TRUE)) {
  stop("Package 'qrng' is required. Install with: install.packages('qrng')")
}
library(qrng)

# Source SMCO algorithm and test functions
source("SMCO.R")
source("testfuncs.R")

# Define SMCO variants
# Each variant specifies its unique settings; shared settings come from SMCO_options
variants <- list(
  SMCO = list(
    refine_search = FALSE,
    iter_boost = 0
  ),
  SMCO_R = list(
    refine_search = TRUE,
    refine_ratio = 0.5,
    iter_boost = 0
  ),
  SMCO_BR = list(
    refine_search = TRUE,
    refine_ratio = 0.5,
    iter_boost = 1000
  )
)

# Helper function: generate uniform starting points
generate_unif_starts <- function(n_starts, bounds_lower, bounds_upper) {
  d <- length(bounds_lower)
  result <- matrix(NA, nrow = n_starts, ncol = d)
  for (j in 1:d) {
    result[, j] <- bounds_lower[j] + runif(n_starts) * (bounds_upper[j] - bounds_lower[j])
  }
  return(result)
}

# Get test function configuration
if (name_config == "ANN") {
  test_config <- assign_config(name = name_config, d = dim_config, opt_config = ANN_options)
} else {
  test_config <- assign_config(name = name_config, d = dim_config, opt_config = NULL)
}

bounds_lower <- test_config$bounds_lower
bounds_upper <- test_config$bounds_upper
test_func <- test_config$f
test_func_minus <- function(x) (-test_func(x))

# Print configuration
cat("Configuration:\n")
cat("  Test function:", name_config, "\n")
cat("  Dimension:", length(bounds_lower), "\n")
cat("  Bounds: [", bounds_lower[1], ",", bounds_upper[1], "]^d\n")
cat("  Replications:", n_replications, "\n")
cat("  n_starts:", SMCO_options$n_starts, "\n")
cat("  iter_max:", SMCO_options$iter_max, "\n")
cat("  True minimum:", ifelse(is.na(test_config$f_min), "unknown", test_config$f_min), "\n")
cat("  True maximum:", ifelse(is.na(test_config$f_max), "unknown", test_config$f_max), "\n")
cat("\nVariants to test: SMCO, SMCO_R, SMCO_BR\n\n")

# Run benchmark for a specific variant and direction
run_benchmark <- function(variant_name, variant_settings, to_maximize) {
  direction <- if (to_maximize) "max" else "min"

  if (to_maximize) {
    test_func_opt <- test_func
    true_opt <- test_config$f_max
  } else {
    test_func_opt <- test_func_minus
    true_opt <- test_config$f_min
  }

  # Merge shared options with variant-specific settings
  opt_control <- SMCO_options
  for (key in names(variant_settings)) {
    opt_control[[key]] <- variant_settings[[key]]
  }

  time_results <- numeric(n_replications)
  fopt_results <- numeric(n_replications)

  set.seed(random_seed)

  for (i in 1:n_replications) {
    start_points <- generate_unif_starts(opt_control$n_starts, bounds_lower, bounds_upper)

    start_time <- Sys.time()
    results <- SMCO_multi(f = test_func_opt,
                          bounds_lower, bounds_upper,
                          start_points = start_points,
                          opt_control = opt_control)
    end_time <- Sys.time()

    time_results[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
    fopt <- results$best_result$f_optimal
    fopt_results[i] <- ifelse(to_maximize, fopt, -fopt)
  }

  # Calculate statistics
  time_avg <- mean(time_results)
  time_sd <- sd(time_results)
  fopt_mean <- mean(fopt_results)
  fopt_sd <- sd(fopt_results)

  if (!is.na(true_opt)) {
    AE <- abs(fopt_results - true_opt)
    rMSE <- sqrt(mean(AE^2))
  } else {
    rMSE <- NA
  }

  return(list(
    variant = variant_name,
    direction = direction,
    true_opt = true_opt,
    time_avg = time_avg,
    time_sd = time_sd,
    fopt_mean = fopt_mean,
    fopt_sd = fopt_sd,
    rMSE = rMSE
  ))
}

# Run all benchmarks
cat("Running benchmarks...\n")
all_results <- list()

for (variant_name in names(variants)) {
  cat("  Testing", variant_name, "...")
  for (to_maximize in c(FALSE, TRUE)) {
    result <- run_benchmark(variant_name, variants[[variant_name]], to_maximize)
    key <- paste0(variant_name, "_", result$direction)
    all_results[[key]] <- result
  }
  cat(" done\n")
}

# Print summary tables
cat("\n==============================================\n")
cat("RESULTS SUMMARY\n")
cat("==============================================\n")

print_summary_table <- function(direction_label, direction_key, true_opt) {
  cat("\n", direction_label, "\n", sep = "")
  cat("True optimum:", ifelse(is.na(true_opt), "unknown", round(true_opt, 6)), "\n\n")

  # Table header
  cat(sprintf("%-10s %10s %14s %12s\n", "Variant", "Time (s)", "f_opt (mean)", "rMSE"))
  cat(sprintf("%-10s %10s %14s %12s\n", "-------", "--------", "------------", "----"))

  for (variant_name in names(variants)) {
    key <- paste0(variant_name, "_", direction_key)
    res <- all_results[[key]]

    time_str <- sprintf("%.2f", res$time_avg)
    fopt_str <- sprintf("%.6f", res$fopt_mean)
    rmse_str <- if (is.na(res$rMSE)) "N/A" else sprintf("%.6f", res$rMSE)

    cat(sprintf("%-10s %10s %14s %12s\n", variant_name, time_str, fopt_str, rmse_str))
  }
}

# Get true optima from any result
true_min <- all_results[[paste0(names(variants)[1], "_min")]]$true_opt
true_max <- all_results[[paste0(names(variants)[1], "_max")]]$true_opt

print_summary_table("MINIMIZATION RESULTS", "min", true_min)
print_summary_table("MAXIMIZATION RESULTS", "max", true_max)

# Print detailed results
cat("\n==============================================\n")
cat("DETAILED RESULTS\n")
cat("==============================================\n")

for (direction in c("min", "max")) {
  direction_label <- if (direction == "min") "MINIMIZATION" else "MAXIMIZATION"
  cat("\n", direction_label, "\n", sep = "")

  for (variant_name in names(variants)) {
    key <- paste0(variant_name, "_", direction)
    res <- all_results[[key]]

    cat("  ", variant_name, ":\n", sep = "")
    cat("    Found (mean/sd): ", sprintf("%.6f / %.6f", res$fopt_mean, res$fopt_sd), "\n")
    if (!is.na(res$rMSE)) {
      cat("    rMSE:            ", sprintf("%.6f", res$rMSE), "\n")
    }
    cat("    Time (mean/sd):  ", sprintf("%.3fs / %.3fs", res$time_avg, res$time_sd), "\n")
  }
}

cat("\n==============================================\n")
cat("Benchmark complete.\n")
cat("\nVariant guide:\n")
cat("  SMCO:    Base algorithm - fastest, good for initial exploration\n")
cat("  SMCO_R:  With refinement - balanced speed and accuracy\n")
cat("  SMCO_BR: Boosted refinement - highest accuracy, more compute\n")
