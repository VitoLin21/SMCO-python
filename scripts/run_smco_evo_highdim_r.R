#!/usr/bin/env Rscript
# run_smco_evo_highdim_r.R - R single-task high-dim worker (Task 8).
#
# Mirrors scripts/run_smco_evo_highdim_factorial.py: reads one canonical task
# JSON, loads its instance artifact, runs the R SMCO variant named by the task,
# and atomically writes a result payload to <result-dir>/<run_id>.json.
#
# Usage:
#   Rscript scripts/run_smco_evo_highdim_r.R --task task.json \
#     --instance-root DIR --result-dir DIR [--log-dir DIR]
#
# Requires jsonlite (task/metadata JSON) and the SMCO_R suite. The instance
# loader (highdim_instances.R) is cross-validated against Python by
# tests/test_r_instance_parity.py; the full R worker end-to-end (jsonlite +
# SMCO budget + result parity) must be exercised on a node with jsonlite
# (e.g. 10.16.144.215 / 10.25.40.251) before confirmatory runs.

# Force single-thread BLAS before sourcing SMCO.
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1",
           MKL_NUM_THREADS = "1", NUMEXPR_NUM_THREADS = "1")

# --- locate repo root from this script's path (Rscript --file= or source()) ---
.self_path <- (function() {
  fa <- grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(fa)) return(normalizePath(sub("--file=", "", fa[1])))
  for (i in seq_len(sys.nframe())) {
    of <- sys.frame(i)$ofile
    if (!is.null(of)) return(normalizePath(of))
  }
  NA_character_
})()
.repo_root <- dirname(dirname(.self_path))
.vendor <- file.path(.repo_root, "vendor", "SMCO_R", "main")
# Source order (per campaign memory): evaluation_budget.R -> SMCO.R ->
# SMCO_evo.R (which auto-sources SMCO_evo_stateful.R from its own directory).
source(file.path(.vendor, "evaluation_budget.R"))
source(file.path(.vendor, "SMCO.R"))
source(file.path(.vendor, "SMCO_evo.R"))
source(file.path(.vendor, "highdim_instances.R"))

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("jsonlite is required by the R worker; install it (or run on 215/251)")
}

# --- minimal flag parser ---
flag_value <- function(name, args, default = NULL) {
  idx <- which(args == paste0("--", name))
  if (length(idx) == 0L || idx == length(args)) return(default)
  args[idx + 1L]
}

.args <- commandArgs(trailingOnly = TRUE)
.task_path <- flag_value("task", .args)
.instance_root <- flag_value("instance-root", .args)
.result_dir <- flag_value("result-dir", .args)
.log_dir <- flag_value("log-dir", .args,
                       default = file.path(dirname(normalizePath(.result_dir)), "logs"))

if (is.null(.task_path) || is.null(.instance_root) || is.null(.result_dir)) {
  stop("--task, --instance-root and --result-dir are required")
}

.dir.create <- function(d) dir.create(d, recursive = TRUE, showWarnings = FALSE)
.dir.create(.result_dir); .dir.create(.log_dir)

.task <- jsonlite::fromJSON(.task_path)
.run_id <- .task$run_id
.log_con <- file(file.path(.log_dir, paste0(.run_id, ".log")), open = "w")
.say <- function(msg) { cat(msg, "\n"); writeLines(msg, .log_con); flush(.log_con) }

.say(sprintf("[r-worker] start run_id=%s algo=%s func=%s d=%s fe_budget=%s",
             .run_id, .task$algorithm_id, .task[["function"]], .task$dimension, .task$fe_budget))

.payload <- NULL
.t0 <- proc.time()
tryCatch({
  .inst_dir <- file.path(.instance_root, .task$instance_artifact_dir)
  .meta <- jsonlite::fromJSON(file.path(.inst_dir, "metadata.json"))

  # instance_hash check (metadata stored the transform sha-256).
  if (!is.null(.task$instance_hash) && nzchar(.task$instance_hash) &&
      !identical(as.character(.meta$transform_sha256), as.character(.task$instance_hash))) {
    stop("instance_hash mismatch: task=", .task$instance_hash,
         " artifact=", .meta$transform_sha256)
  }

  .default_n <- 8L
  if (!is.null(.meta$n_starts)) .default_n <- as.integer(.meta$n_starts)
  .inst <- load_highdim_instance(
    .inst_dir, .task[["function"]], as.integer(.task$dimension),
    as.numeric(.meta$asymmetry_strength), as.numeric(.meta$objective_scale),
    as.numeric(.meta$known_optimum_value),
    n_starts = as.integer(.task$n_starts), default_n_starts = .default_n
  )
  .starts <- .inst$starts
  .dim <- .inst$dimension
  .known_optimum <- .inst$known_optimum_value

  # observer: SMCO maximises g = -objective; record minimisation best-so-far.
  .obs <- new.env(parent = emptyenv())
  .obs$fe <- 0L; .obs$best_min <- Inf
  .obs$trace_fe <- integer(); .obs$trace_val <- numeric()
  .fobj <- function(x) {
    v <- .inst$objective(x)
    .obs$fe <- .obs$fe + 1L
    if (v < .obs$best_min) {
      .obs$best_min <- v
      .obs$trace_fe <- c(.obs$trace_fe, .obs$fe)
      .obs$trace_val <- c(.obs$trace_val, .obs$best_min)
    }
    -v
  }

  .cfg <- .task$algorithm_config
  # task$seed is a 32-bit run-key hash (can exceed R's integer max); R's RNG
  # needs a seed < 2^31. Python/R use independent RNG streams (plan 4.4), so we
  # only need determinism, hence the modulo into R's integer range.
  .seed <- as.integer(as.numeric(.task$seed) %% 2147483647)
  .fe_budget <- as.integer(.task$fe_budget)
  # A-01: split the global FE budget across n_starts so every start advances
  # before the first evolution boundary and boundaries land near 50%/75% of B
  # (mirrors Python global_stage_iter_max; max_evals stays the hard stop).
  .n_starts <- nrow(.starts)
  .iter_max <- max(1L, as.integer(.fe_budget %/% (.n_starts * (2L * .dim + 1L))))
  .ctrl <- list(iter_max = .iter_max, max_evals = .fe_budget,
                objective_sense = "maximize", known_optimum = -.known_optimum,
                seed = .seed, bounds_buffer = 0.05)

  .parts <- strsplit(.task$algorithm_id, "-")[[1]]
  .is_evo <- endsWith(.task$algorithm_id, "-EVO")
  .fam_token <- if (.is_evo) paste(.parts[3:(length(.parts) - 1L)], collapse = "-") else
                                  paste(.parts[3:length(.parts)], collapse = "-")
  .family <- switch(.fam_token, "SMCO" = "smco", "SMCO-REFINE" = "smco_refine",
                    "SMCO-BOOST-REFINE" = "smco_boost_refine")
  .state_sem <- if (.is_evo) switch(.parts[2], "SP" = "state_preserving", "RS" = "restart") else NA_character_

  if (!.is_evo) {
    if (.family == "smco_refine") { .ctrl$refine_search <- TRUE; .ctrl$refine_ratio <- 0.5 }
    else if (.family == "smco_boost_refine") {
      .ctrl$refine_search <- TRUE; .ctrl$iter_boost <- 1000L; .ctrl$refine_ratio <- 0.5
    }
    .r <- SMCO_multi(.fobj, .inst$bounds_lower, .inst$bounds_upper,
                     start_points = .starts, opt_control = .ctrl)
  } else {
    .common <- list(.fobj, .inst$bounds_lower, .inst$bounds_upper)
    names(.common) <- c("f", "bounds_lower", "bounds_upper")
    .common$start_points <- .starts
    .common$evolution_points <- as.numeric(.cfg$evolution_points)
    .common$elimination_rate <- as.numeric(.cfg$elimination_rate)
    .common$evolution_strategy <- .cfg$evolution_strategy
    .common$de_factor <- as.numeric(.cfg$de_factor)
    .common$de_crossover <- as.numeric(.cfg$de_crossover)
    .common$state_semantics <- .state_sem
    .fn <- switch(.family, smco = SMCO_EVO, smco_refine = SMCO_R_EVO, smco_boost_refine = SMCO_BR_EVO)
    .r <- do.call(.fn, c(.common, .ctrl))
  }

  .fe_summary <- if (!is.null(.r$summary$fe)) .r$summary$fe else list()
  .fe_used <- .obs$fe
  .best_min <- if (length(.obs$trace_val)) .obs$best_min else NA_real_
  .initial_ref <- median(.inst$objective(.starts))
  .norm_gap <- max(.best_min - .known_optimum, 1e-12) / max(.initial_ref - .known_optimum, 1e-12)

  # Targets are RELATIVE to the normalized gap (contract 6 / plan 6.1):
  # best <= f* + target * (initial_reference - f*).
  .gap_span <- .initial_ref - .known_optimum
  .target_hit <- list()
  for (.lbl in c("1e-1", "1e-2", "1e-3", "1e-5")) {
    .target <- .known_optimum + as.numeric(.lbl) * .gap_span
    .fe_hit <- NA
    if (length(.obs$trace_val)) {
      .idx <- which(.obs$trace_val <= .target)[1]
      if (!is.na(.idx)) .fe_hit <- .obs$trace_fe[.idx]
    }
    .target_hit[[.lbl]] <- .fe_hit
  }

  .anytime <- lapply(.task$checkpoints, function(cp) {
    cp <- as.integer(cp)
    best <- .best_min
    if (length(.obs$trace_val)) {
      sel <- .obs$trace_fe <= cp
      if (any(sel)) best <- .obs$trace_val[max(which(sel))]
    }
    list(checkpoint_fe = cp, fe_used = min(cp, .fe_used), best_value = best,
         normalized_gap = max(best - .known_optimum, 1e-12) / max(.initial_ref - .known_optimum, 1e-12))
  })

  .payload <- list(
    run_id = .run_id,
    status = "success",
    failure_reason = "none",
    fe_used = .fe_used,
    fe_budget = .fe_budget,
    best_value = .best_min,
    known_optimum = .known_optimum,
    normalized_gap = .norm_gap,
    objective_sense = "minimize",
    target_hit_fe = .target_hit,
    anytime = .anytime,
    best_so_far_trace = mapply(c, .obs$trace_fe, .obs$trace_val, SIMPLIFY = FALSE),
    termination_reason = .fe_summary$termination_reason %||% "evaluation_budget",
    fe_counts_by_event = as.list(.fe_summary$evaluation_counts_by_event %||% list()),
    wall_time_sec = as.numeric((proc.time() - .t0)["elapsed"]),
    peak_memory_mb = NA_real_,  # R has no portable ru_maxrss; serialised as null
    machine_id = Sys.info()[["nodename"]],
    git_commit = "",
    environment_hash = paste0("R-", R.version$major, ".", R.version$minor),
    task = .task,
    algorithm_id = .task$algorithm_id,
    supersedes_run_id = "none",
    evolution_history = .r$evolution_history %||% list()
  )
}, error = function(e) {
  .say(sprintf("[r-worker] INFRA_FAILURE %s: %s", class(e)[1], conditionMessage(e)))
  .payload <<- list(run_id = .run_id, status = "infra_failure",
                    failure_reason = paste0(class(e)[1], ": ", conditionMessage(e)))
})

# atomic write
.tmp <- file.path(.result_dir, paste0(.run_id, ".json.tmp"))
jsonlite::write_json(.payload, .tmp, auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null")
file.rename(.tmp, file.path(.result_dir, paste0(.run_id, ".json")))
.say(sprintf("[r-worker] done status=%s", .payload$status))
close(.log_con)
