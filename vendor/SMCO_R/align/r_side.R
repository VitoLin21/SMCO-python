#!/usr/bin/env Rscript
# R-side alignment: read the same fixed starts, run SMCO_EVO, print result.
# Usage: Rscript r_side.R Rastrigin 10 8 starts.csv
.libPaths(c("~/Rlibs", .libPaths()))
args <- commandArgs(trailingOnly = TRUE)
func <- if (length(args) >= 1) args[1] else "Rastrigin"
dim <- as.integer(if (length(args) >= 2) args[2] else 10)
n_starts <- as.integer(if (length(args) >= 3) args[3] else 8)
starts_csv <- if (length(args) >= 4) args[4] else "starts.csv"

vendor <- Sys.getenv("SMCO_R_DIR", "~/code/SMCO/vendor/SMCO_R/main")
source(file.path(vendor, "SMCO.R"))
source(file.path(vendor, "SMCO_evo.R"))
source(file.path(vendor, "testfuncs.R"))

cfg <- assign_config(func, dim, NULL)

starts <- as.matrix(read.csv(starts_csv, header = FALSE))
f <- function(x) -cfg$f(x)  # minimize -> maximize

res <- SMCO_EVO(f, cfg$bounds_lower, cfg$bounds_upper, start_points = starts,
                iter_max = 200, bounds_buffer = 0.05, buffer_rand = TRUE,
                tol_conv = 1e-8, evolution_points = c(0.5, 0.75),
                elimination_rate = 0.25, evolution_strategy = "rand1bin",
                de_factor = 0.8, de_crossover = 0.7, seed = 20260615)

cat(sprintf("R_RESULT f_optimal=%.17g\n", res$best_result$f_optimal))
cat(sprintf("R_RESULT iters=%d\n", res$best_result$iterations))
cat(sprintf("R_RESULT x_norm=%.17g\n", sqrt(sum(res$best_result$x_optimal^2))))
cat(sprintf("R_RESULT n_history=%d\n", length(res$evolution_history)))
