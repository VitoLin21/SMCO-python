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

# ---- 3. SP staged continuation == one-shot continuation -------------------
# iter_target is an ABSOLUTE target (target_n = initial_n + iter_target); a
# second call must name the same final target as the one-shot, not an increment
# (this is exactly how the SP scheduler passes iter_target = boundary - birth).
base <- initialize_smco_state(ff, c(0.4, -0.2), iter_nstart = 1L, iter_boost = 0L,
                             use_runmax = TRUE, budget = NULL)
oneshot <- initialize_smco_state(ff, c(0.4, -0.2), iter_nstart = 1L, iter_boost = 0L,
                                use_runmax = TRUE, budget = NULL)
staged <- base
staged <- run_smco_state_until(staged, ff, c(-1, -1), c(1, 1), 0.05, FALSE,
                              iter_target = 5L, tol_conv = 1e-12,
                              partial_option = "center", use_runmax = TRUE)
staged <- run_smco_state_until(staged, ff, c(-1, -1), c(1, 1), 0.05, FALSE,
                              iter_target = 10L, tol_conv = 1e-12,
                              partial_option = "center", use_runmax = TRUE)
oneshot <- run_smco_state_until(oneshot, ff, c(-1, -1), c(1, 1), 0.05, FALSE,
                               iter_target = 10L, tol_conv = 1e-12,
                               partial_option = "center", use_runmax = TRUE)
check(approx(staged$f_current, oneshot$f_current), "SP staged != one-shot f")
check(all(approx(staged$x_current, oneshot$x_current)), "SP staged != one-shot x")

# ---- 4. SP runner end-to-end (history tagged state_preserving) ------------
sp <- matrix(c(-0.8, -0.3, 0.2, 0.7, -0.5, 0.4, -0.1, 0.6,
               0.5, -0.6, -0.4, 0.3, 0.1, 0.8, -0.7, -0.2), ncol = 2, byrow = TRUE)
ctrl <- list(bounds_buffer = 0.05, buffer_rand = TRUE, tol_conv = 1e-12,
             partial_option = "center", use_runmax = TRUE, iter_nstart = 8L, seed = 123)
evo_sp <- run_evolutionary_states(ff, c(-1, -1), c(1, 1), sp, ctrl,
                                  evolution_points = c(0.5, 0.75),
                                  elimination_rate = 0.25,
                                  evolution_strategy = "rand1bin",
                                  de_factor = 0.8, de_crossover = 0.7,
                                  iter_max = 40L, iter_boost = 0L, budget = NULL)
check(length(evo_sp$results) == 8L, "SP result count != n_starts")
check(all(sapply(evo_sp$history, function(h) identical(h$state_semantics, "state_preserving"))),
      "SP history missing state_preserving tag")
check(is.finite(evo_sp$results[[1]]$f_optimal), "SP result not finite")

# ---- 5. SP vs RS diverge after a boundary (SP preserves accumulator) ------
# With a real boundary, SP (carries s_value) and RS (restarts from x_runmax)
# must differ — proving SP actually preserves state.
evo_rs <- run_evolutionary_restarts(ff, c(-1, -1), c(1, 1), sp, ctrl,
                                    evolution_points = c(0.5, 0.75),
                                    elimination_rate = 0.25,
                                    evolution_strategy = "rand1bin",
                                    de_factor = 0.8, de_crossover = 0.7,
                                    iter_max = 40L, iter_boost = 0L, budget = NULL)
sp_best <- max(sapply(evo_sp$results, function(r) r$f_optimal))
rs_best <- max(sapply(evo_rs$results, function(r) r$f_optimal))
check(!approx(sp_best, rs_best, tol = 1e-6), "SP and RS identical after boundary")

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

# ---- 15. R-SP agrees with Python-SP on a small deterministic trajectory ----
# Reference captured from Python smco_evo(state_semantics='state_preserving',
# buffer_rand=FALSE, evolution_points=(0.999,) -> no boundary, so the SP runner
# degenerates to N independent SMCO_single trajectories and the cross-language
# comparison is exact (no RNG divergence). Python np.random != R RNG, so a
# boundary-bearing config would diverge on DE replacements and is out of scope.
PY_SP_F <- -0.0001020408163265301
PY_SP_X <- c(0.0071428571428571175, -0.007142857142857133)
ctrl_sp <- list(bounds_buffer = 0.05, buffer_rand = FALSE, tol_conv = 1e-12,
                partial_option = "center", use_runmax = TRUE, iter_nstart = 8L, seed = 123)
pyx_sp <- matrix(c(-0.8,-0.3, 0.2,0.7, -0.5,0.4, -0.1,0.6,
                   0.5,-0.6, -0.4,0.3, 0.1,0.8, -0.7,-0.2), ncol = 2, byrow = TRUE)
rsp <- run_evolutionary_states(ff, c(-1,-1), c(1,1), pyx_sp, ctrl_sp,
                              evolution_points = c(0.999), elimination_rate = 0.25,
                              evolution_strategy = "rand1bin", de_factor = 0.8,
                              de_crossover = 0.7, iter_max = 20L, iter_boost = 0L)
rsp_best <- rsp$results[[which.max(sapply(rsp$results, function(r) r$f_optimal))]]
check(approx(rsp_best$f_optimal, PY_SP_F, tol = 1e-6),
      paste("R-SP f_optimal", rsp_best$f_optimal, "!= Python", PY_SP_F))
check(all(approx(rsp_best$x_optimal, PY_SP_X, tol = 1e-6)),
      "R-SP x_optimal != Python")

cat("ALL R EVOLUTION-SEMANTICS TESTS PASSED\n")
