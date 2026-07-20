#!/usr/bin/env Rscript
# Experiment 3b: high-dimensional R reproduction, matching the Python baseline.
#
# Mirrors scripts/run_highdim_full_comparison.py (Python scipy-based) so the
# output CSV is directly comparable row-for-row (same schema, same dims/funcs/
# reps/iter_max/strategy). The difference: here the "comparison" algorithms are
# the REAL R packages (GenSA/DEoptim/GA/pso + a basin-hopping SA), not scipy.
#
# CONFIG (strict match to Python all_results.csv):
#   dims   {1000,3000,5000}
#   funcs  {Rastrigin,Ackley,Rosenbrock}
#   reps   {5,2,2}
#   iter_max = 300, strategy = rand1bin (for SMCO_EVO)
#   n_starts = max(3, ceil(sqrt(dim)))   # 1000->32, 3000->55, 5000->71
#
# SIGN CONVENTION (matches Python baseline's ACTUAL -- reversed -- direction;
# see docs/direction-bug-2026-06-15.md): the Python baseline passes raw(x) to
# the maximizer, so it pushes x toward the boundary. To be row-comparable to
# all_results.csv we do the same here: maximize raw(x). fopt recorded is the
# maximizer's objective = raw(x*) (NOT the true minimum). This keeps R and
# Python on the same footing for cross-language comparison; the absolute sign
# caveat is documented separately.
#
# Output: <outdir>/r_highdim_results.csv with columns
#   strategy, algo, func, dim, rep, fopt, time, iterations, seed
# Resumable: existing (algo,func,dim,rep) keys are skipped.

.libPaths(c("~/Rlibs", .libPaths()))

# filelock: fd-level advisory locks. Verified zero-loss / zero-corruption under
# 300-way fork concurrency (system2("flock",...) name-form loses ~1/200 rows).
suppressMessages(library(filelock))

args <- commandArgs(trailingOnly = TRUE)

# Collect all values following a flag, up to the next "--xxx" flag. Supports
# both space-separated (--dims 1000 3000) and comma-separated (--dims 1000,3000).
flag_values <- function(flag, args) {
  if (!(flag %in% args)) return(NULL)
  i <- which(args == flag) + 1L
  vals <- c()
  while (i <= length(args) && !startsWith(args[i], "--")) {
    vals <- c(vals, strsplit(args[i], "[, ]+")[[1]])
    i <- i + 1L
  }
  vals[nzchar(vals)]
}

outdir   <- { v <- flag_values("--outdir", args); if (is.null(v)) "result/r-highdim-2026-06-15" else v[1] }
dims_arg <- { v <- flag_values("--dims", args); if (is.null(v)) c(1000,3000,5000) else as.integer(v) }
funcs_arg<- { v <- flag_values("--funcs", args); if (is.null(v)) c("Rastrigin","Ackley","Rosenbrock") else v }
workers  <- { v <- flag_values("--workers", args); if (is.null(v)) 1L else as.integer(v[1]) }
quick    <- "--quick" %in% args
algos_arg<- flag_values("--algos", args)   # NULL = 全部 SMCO+CMP; 否则只跑指定 algos
seed_base<- 21L

vendor <- normalizePath(".", mustWork = NA)
if (!file.exists(file.path(vendor, "SMCO.R"))) {
  vendor <- "~/code/SMCO/vendor/SMCO_R/main"
}
source(file.path(vendor, "SMCO.R"))
source(file.path(vendor, "SMCO_evo.R"))
source(file.path(vendor, "testfuncs.R"))
for (p in c("qrng","GenSA","DEoptim","GA","pso")) suppressMessages(library(p, character.only = TRUE))

ITER_MAX   <- 300
REPS       <- list("1000"=5L, "3000"=2L, "5000"=2L)
N_STARTS   <- function(dim) max(3L, as.integer(ceiling(sqrt(dim))))

FIELDS <- c("strategy","algo","func","dim","rep","fopt","time","iterations","seed")

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
csv_path <- file.path(outdir, "r_highdim_results.csv")
csv_lock_path <- paste0(csv_path, ".lock")   # filelock advisory lock file

# Unbuffered log: open the run.log as a separate append-mode connection and flush
# after every line. R's stdout under nohup redirection is BLOCK-buffered, so the
# old cat(...) + flush.console() left run.log stale for the whole run (looked
# hung even while computing). lcat() writes+flushes immediately so progress is
# observable live. Fork workers each open their own fd (O_APPEND), so concurrent
# line writes stay atomic (line < PIPE_BUF).
log_path <- file.path(outdir, "run.log")
lcat <- function(...) {
  con <- file(log_path, open = "a")
  on.exit(close(con), add = TRUE)
  cat(..., file = con)
  flush.connection(con)
}

# ---- resumable I/O --------------------------------------------------------
existing_keys <- function() {
  if (!file.exists(csv_path)) return(list())
  d <- read.csv(csv_path, stringsAsFactors = FALSE)
  if (nrow(d) == 0) return(list())
  keys <- vector("list", nrow(d))
  for (i in seq_len(nrow(d)))
    keys[[i]] <- paste(d$algo[i], d$func[i], d$dim[i], d$rep[i], sep = "|")
  keys
}
done <- existing_keys()

append_row <- function(row) {
  has_header <- file.exists(csv_path) && file.info(csv_path)$size > 0
  con <- file(csv_path, open = if (has_header) "a" else "w")
  on.exit(close(con))
  if (!has_header) {
    writeLines(paste(FIELDS, collapse = ","), con)
  }
  writeLines(paste(vapply(FIELDS, function(f) {
    v <- row[[f]]
    if (is.na(v)) "NA" else as.character(v)
  }, ""), collapse = ","), con)
}

# Concurrency-safe append for fork (mclapply) workers using filelock (fd-level
# advisory locks; verified zero-loss under 300-way fork concurrency). Each worker
# takes an exclusive lock, writes the header on first call, appends its row, then
# releases. The CSV is therefore LIVE under parallelism -- every finished algo is
# on disk immediately (unlike the old batch-flush that left it empty until every
# task returned). flock()'d open/close/flush all happen INSIDE the critical
# section, so no interleave corruption is possible.
append_row_locked <- function(row) {
  lk <- filelock::lock(csv_lock_path, exclusive = TRUE, timeout = 60000)
  on.exit(filelock::unlock(lk), add = TRUE)
  need_hdr <- !file.exists(csv_path) || file.info(csv_path)$size == 0
  line <- paste(vapply(FIELDS, function(f) {
    v <- row[[f]]; if (is.na(v)) "NA" else as.character(v)
  }, ""), collapse = ",")
  con <- file(csv_path, open = if (need_hdr) "w" else "a")
  on.exit(close(con), add = TRUE)
  if (need_hdr) writeLines(paste(FIELDS, collapse = ","), con)
  writeLines(line, con)
  flush(con)
}

# ---- algorithm runners ----------------------------------------------------
# maximize raw(x): f_obj returns raw(x) for the maximizer; record raw(x*).
run_algo <- function(algo, cfg, starts, seed) {
  d   <- length(cfg$bounds_lower)
  lo  <- cfg$bounds_lower; hi  <- cfg$bounds_upper
  raw <- cfg$f                       # raw test function (e.g. rastrigin)
  fobj<- function(x) raw(x)          # maximize raw -> matches Python baseline direction

  t0 <- proc.time()[["elapsed"]]
  iters <- NA_integer_; fopt <- NA_real_

  if (algo == "SMCO") {
    r <- SMCO_multi(fobj, lo, hi, start_points = starts,
                    opt_control = list(iter_max=ITER_MAX, bounds_buffer=0.05,
                                       buffer_rand=TRUE, tol_conv=1e-8,
                                       partial_option="center", use_runmax=TRUE,
                                       seed=seed, refine_search=FALSE, iter_boost=0))
    fopt <- r$best_result$f_optimal; iters <- r$best_result$iterations
  } else if (algo == "SMCO_R") {
    r <- SMCO_multi(fobj, lo, hi, start_points = starts,
                    opt_control = list(iter_max=ITER_MAX, bounds_buffer=0.05,
                                       buffer_rand=TRUE, tol_conv=1e-8,
                                       partial_option="center", use_runmax=TRUE,
                                       seed=seed, refine_search=TRUE, refine_ratio=0.5,
                                       iter_boost=0))
    fopt <- r$best_result$f_optimal; iters <- r$best_result$iterations
  } else if (algo == "SMCO_BR") {
    r <- SMCO_multi(fobj, lo, hi, start_points = starts,
                    opt_control = list(iter_max=as.integer(ceiling(ITER_MAX/2)),
                                       bounds_buffer=0.05, buffer_rand=TRUE,
                                       tol_conv=1e-8, partial_option="center",
                                       use_runmax=TRUE, seed=seed, refine_search=TRUE,
                                       refine_ratio=0.5, iter_boost=1000))
    fopt <- r$best_result$f_optimal; iters <- r$best_result$iterations
  } else if (algo == "SMCO_EVO") {
    r <- SMCO_EVO(fobj, lo, hi, start_points = starts, seed = seed,
                  iter_max = ITER_MAX, bounds_buffer = 0.05, buffer_rand = TRUE,
                  tol_conv = 1e-8, evolution_points = c(0.5, 0.75),
                  elimination_rate = 0.5, evolution_strategy = "rand1bin",
                  de_factor = 0.8, de_crossover = 0.7)
    fopt <- r$best_result$f_optimal; iters <- r$best_result$iterations
  } else if (algo == "SMCO_R_EVO") {
    r <- SMCO_R_EVO(fobj, lo, hi, start_points = starts, seed = seed,
                    iter_max = ITER_MAX, bounds_buffer = 0.05, buffer_rand = TRUE,
                    tol_conv = 1e-8, evolution_points = c(0.5, 0.75),
                    elimination_rate = 0.5, evolution_strategy = "rand1bin",
                    de_factor = 0.8, de_crossover = 0.7)
    fopt <- r$best_result$f_optimal; iters <- r$best_result$iterations
  } else if (algo == "SMCO_BR_EVO") {
    r <- SMCO_BR_EVO(fobj, lo, hi, start_points = starts, seed = seed,
                     iter_max = as.integer(ceiling(ITER_MAX/2)), bounds_buffer = 0.05,
                     buffer_rand = TRUE, tol_conv = 1e-8, evolution_points = c(0.5, 0.75),
                     elimination_rate = 0.5, evolution_strategy = "rand1bin",
                     de_factor = 0.8, de_crossover = 0.7, iter_boost = 1000)
    fopt <- r$best_result$f_optimal; iters <- r$best_result$iterations
  } else if (algo == "DEoptim") {
    lo_mat <- matrix(rep(lo, each = 1), ncol = d); hi_mat <- matrix(rep(hi, each = 1), ncol = d)
    set.seed(seed)
    r <- DEoptim(function(x) -fobj(x), lower = lo, upper = hi,
                 DEoptim.control(itermax = ITER_MAX, NP = 15L, trace = FALSE,
                                 strategy = 2, CR = 0.7, F = 0.8))
    fopt <- fobj(r$optim$bestmem); iters <- r$optim$iter
  } else if (algo == "GA") {
    set.seed(seed)
    r <- ga(type = "real-valued", fitness = fobj, lower = lo, upper = hi,
            maxiter = ITER_MAX, popSize = 50L, run = ITER_MAX, seed = seed,
            monitor = FALSE, pmutation = 0.1)
    xbest <- r@solution[1, ]; fopt <- fobj(xbest); iters <- as.integer(r@iter)
  } else if (algo == "PSO") {
    # psoptim MINIMIZES its fn, so to maximize raw(x) pass -fobj.
    set.seed(seed)
    r <- psoptim(rep(NA, d), function(x) -fobj(x), lower = lo, upper = hi,
                 control = list(maxit = ITER_MAX, s = 40L, trace = 0))
    fopt <- fobj(r$par); iters <- ITER_MAX
  } else if (algo == "GenSA") {
    set.seed(seed)
    r <- GenSA(par = starts[1, ], fn = function(x) -fobj(x), lower = lo, upper = hi,
               control = list(maxit = ITER_MAX, smooth = FALSE, verbose = FALSE))
    fopt <- fobj(r$par); iters <- as.integer(sum(unlist(r$counts)))
  } else if (algo == "SA") {
    # basin-hopping-style via optim + multiple restarts (no R basinhopping pkg);
    # approximate with BFGS from best start, bounded by clipping.
    set.seed(seed)
    best <- list(f = -Inf, par = starts[1, ])
    for (k in seq_len(nrow(starts))) {
      r <- tryCatch(optim(starts[k, ], function(x) -fobj(x), method = "L-BFGS-B",
                          lower = lo, upper = hi, control = list(maxit = ITER_MAX)),
                    error = function(e) NULL)
      if (!is.null(r) && -r$value > best$f) best <- list(f = -r$value, par = r$par)
    }
    fopt <- fobj(best$par); iters <- ITER_MAX
  }
  elapsed <- proc.time()[["elapsed"]] - t0
  list(fopt = fopt, time = elapsed, iterations = as.integer(iters))
}

# ---- main loop ------------------------------------------------------------
if (is.null(algos_arg)) {
  SMCO_ALGOS <- c("SMCO","SMCO_R","SMCO_BR","SMCO_EVO","SMCO_R_EVO","SMCO_BR_EVO")
  CMP_ALGOS  <- c("DEoptim","GA","PSO","GenSA","SA")
} else {
  SMCO_ALGOS <- algos_arg
  CMP_ALGOS  <- character(0)
}

lcat("outdir:", normalizePath(outdir), "\n", sep = " ")
lcat("dims:", paste(dims_arg, collapse=" "), " funcs:", paste(funcs_arg, collapse=" "), "\n")
lcat("workers:", workers, " resumable done keys:", length(done), "\n")

# Build the task list: one task per (dim, func, rep) that still has missing
# algos. Each task runs its 9 algos serially and returns its result rows.
tasks <- list()
for (dim in dims_arg) {
  reps <- if (quick) 1L else REPS[[as.character(dim)]]
  if (is.null(reps)) reps <- 2L
  for (func in funcs_arg) {
    for (rep in seq_len(reps) - 1L) {
      todo <- c()
      for (algo in c(SMCO_ALGOS, CMP_ALGOS)) {
        key <- paste(algo, func, dim, rep, sep = "|")
        if (!(key %in% done)) todo <- c(todo, algo)
      }
      if (length(todo) > 0) {
        tasks[[length(tasks)+1]] <- list(dim=dim, func=func, rep=rep, algos=todo)
      }
    }
  }
}
lcat("tasks to run:", length(tasks), "\n")

run_task <- function(tk) {
  dim <- tk$dim; func <- tk$func; rep <- tk$rep
  cfg <- assign_config(func, dim, NULL)
  nst <- N_STARTS(dim)
  set.seed(seed_base + rep * 1000003L + dim)
  starts <- cfg$bounds_lower + matrix(runif(dim*nst), nst) *
            (cfg$bounds_upper - cfg$bounds_lower)
  aseed <- as.integer(seed_base + rep*1000003L + dim)
  for (i in seq_along(tk$algos)) {
    algo <- tk$algos[i]
    lcat(sprintf("[%dD %s rep%d] %s ...\n", dim, func, rep, algo))
    res <- tryCatch(run_algo(algo, cfg, starts, aseed),
                    error = function(e) {lcat("ERR:", conditionMessage(e),"\n");
                                         list(fopt=NA_real_, time=0, iterations=NA_integer_)})
    row <- list(strategy="rand1bin", algo=algo, func=func, dim=dim, rep=rep,
                fopt=res$fopt, time=res$time, iterations=res$iterations, seed=aseed)
    append_row_locked(row)   # 即时落盘：每个 algo 完成立即 filelock-append，CSV 实时可见
    lcat(sprintf("[%dD %s rep%d] %s fopt=%.4f t=%.1fs\n", dim, func, rep, algo, res$fopt, res$time))
  }
  invisible(NULL)
}

# Run tasks. Each algo is flock-appended to the CSV *inside* run_task the moment
# it finishes, so the CSV is live and resumable even under mclapply parallelism
# (this fixes the old batch-flush pattern that left the CSV empty until every
# task returned). With workers>1 use mclapply (fork); the return value is unused.
if (length(tasks) > 0) {
  if (workers > 1 && .Platform$OS.type == "unix") {
    parallel::mclapply(tasks, run_task, mc.cores = workers, mc.preschedule = FALSE)
  } else {
    for (tk in tasks) run_task(tk)
  }
}
lcat("DONE ->", csv_path, "\n")

