#####################################
# test_evaluation_budget.R - Task 2 / Gate A (R side) FE-budget regression.
# Run: Rscript vendor/SMCO_R/main/tests/test_evaluation_budget.R
# No testthat dependency: stopifnot() assertions, non-zero exit on failure.
#####################################
options(warn = 1)

# Resolve the SMCO_R/main directory from this script's path
# (vendor/SMCO_R/main/tests/test_evaluation_budget.R -> vendor/SMCO_R/main).
arg <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("--file=", arg, value = TRUE)
if (length(script_arg) == 1) {
  me <- normalizePath(sub("--file=", "", script_arg))
  smco_dir <- dirname(dirname(me))
} else {
  smco_dir <- "../."  # fallback when sourced interactively
}
for (f in c("evaluation_budget.R", "SMCO.R", "SMCO_evo.R")) {
  source(file.path(smco_dir, f))
}

fail <- function(msg) { message("FAIL: ", msg); quit(status = 1) }
check <- function(cond, msg) { if (!isTRUE(cond)) fail(msg) }

# ---- 1. eval_fe NULL is raw; budget counts --------------------------------
ff <- function(x) -sum(x^2)
check(abs(eval_fe(NULL, ff, c(1, 2)) + 5) < 1e-9, "raw path mismatch")
ctx <- budget_ctx(ff, max_evals = 10L, objective_sense = "maximize", known_optimum = 0)
for (i in 1:3) eval_fe(ctx, ff, c(i), event = "iterate")
check(ctx_evaluations(ctx) == 3L, "count != 3")
check(isTRUE(abs(ctx$best_value - (-1)) < 1e-9), "best_value mismatch")
check(ctx_can_evaluate(ctx, 7L) && !ctx_can_evaluate(ctx, 8L), "cap precheck wrong")

# ---- 2. hard guard ---------------------------------------------------------
ctx2 <- budget_ctx(ff, max_evals = 2L)
eval_fe(ctx2, ff, c(1), event = "iterate")
eval_fe(ctx2, ff, c(1), event = "iterate")
raised <- tryCatch({ eval_fe(ctx2, ff, c(1), event = "iterate"); FALSE },
                   error = function(e) TRUE)
check(raised, "hard guard did not raise")
check(identical(ctx_termination_reason(ctx2), "evaluation_budget"), "termination not set")

# ---- 3. scoped re-tag shares counter --------------------------------------
ctx3 <- budget_ctx(ff, max_evals = 20L)
sc <- ctx_scoped(ctx3, "refine")
eval_fe(sc, ff, c(2), event = "iterate")
check(ctx_evaluations(ctx3) == 1L, "scoped did not share counter")
check(ctx_evaluation_counts(sc)$refine == 1L, "scoped event miscounted")
check(ctx_evaluation_counts(sc)$iterate == 0L, "scoped leaked natural event")

# ---- 4. split independent counters ----------------------------------------
parent <- budget_ctx(ff, max_evals = 100L)
reg <- ctx_split(parent, fraction = 0.5); boo <- ctx_split(parent, fraction = 0.5)
check(reg$max_evals == 50L && boo$max_evals == 50L, "split caps wrong")
eval_fe(reg, ff, c(1), event = "iterate")
check(ctx_evaluations(reg) == 1L && ctx_evaluations(parent) == 0L, "split not independent")

# ---- 5. SMCO_single exact step accounting (d=2 center, step = 2*2+1 = 5) ---
sp_state <- SMCO_single(ff, c(-1, -1), c(1, 1), start_point = c(0.5, 0.5),
                        bounds_buffer = 0.05, buffer_rand = FALSE, iter_max = 0L,
                        iter_nstart = 1L, iter_boost = 0L, tol_conv = 1e-12,
                        partial_option = "center", use_runmax = TRUE)
# Sanity: SMCO_single still returns a valid result without a budget.
check(is.finite(sp_state$f_optimal), "SMCO_single legacy broke")

ctx5 <- budget_ctx(ff, max_evals = 6L)
res5 <- SMCO_single(ff, c(-1, -1), c(1, 1), start_point = c(0.5, 0.5),
                    bounds_buffer = 0.05, buffer_rand = FALSE, iter_max = 10L,
                    iter_nstart = 1L, iter_boost = 0L, tol_conv = 1e-12,
                    partial_option = "center", use_runmax = TRUE, budget = ctx5)
# init(1) + exactly one center-difference iteration (5) = 6; next step can't fit.
check(ctx_evaluations(ctx5) == 6L, paste("expected 6 evals, got", ctx_evaluations(ctx5)))
check(identical(ctx_termination_reason(ctx5), "evaluation_budget"), "termination reason")

# ---- 6. SMCO multi-start with budget --------------------------------------
sp <- matrix(c(-0.8, -0.3, 0.2, 0.7, -0.5, 0.4, -0.1, 0.6), ncol = 2, byrow = TRUE)
r6 <- SMCO(ff, c(-1, -1), c(1, 1), start_points = sp, iter_max = 200, seed = 1,
           max_evals = 400, known_optimum = 0)
fe6 <- r6$summary$fe
check(!is.null(fe6) && fe6$fe_used <= 400L, "SMCO fe_used > budget")
counts6 <- unlist(fe6$evaluation_counts_by_event)
check(sum(counts6) == fe6$fe_used, "SMCO event sum != fe_used")

# ---- 7. legacy path unchanged (no fe in summary) --------------------------
r7 <- SMCO(ff, c(-1, -1), c(1, 1), start_points = sp, iter_max = 30, seed = 1)
check(!("fe" %in% names(r7$summary)), "legacy summary carries fe")
check(is.finite(r7$best_result$f_optimal), "legacy best not finite")

# ---- 8. SMCO_EVO with budget counts replacement_initialization ------------
set.seed(7); sp8 <- matrix(runif(8 * 2, -1, 1), ncol = 2)
r8 <- SMCO_EVO(ff, c(-1, -1), c(1, 1), start_points = sp8, iter_max = 40,
               evolution_points = c(0.5), seed = 1, max_evals = 2000, known_optimum = 0)
fe8 <- r8$summary$fe
counts8 <- unlist(fe8$evaluation_counts_by_event)
check(fe8$fe_used <= 2000L, "EVO fe_used > budget")
check(sum(counts8) == fe8$fe_used, "EVO event sum != fe_used")
check(counts8[["replacement_initialization"]] > 0L, "no replacement_initialization counted")

# ---- 9. SMCO_BR_EVO 50/50 split -------------------------------------------
r9 <- SMCO_BR_EVO(ff, c(-1, -1), c(1, 1), start_points = sp8, iter_max = 40,
                  iter_boost = 20L, evolution_points = c(0.5), seed = 1, max_evals = 4000)
fe9 <- r9$summary$fe
check(fe9$fe_used <= 4000L, "BR fe_used > budget")
check(fe9$branch_fe$regular <= 2000L && fe9$branch_fe$boosted <= 2000L, "BR branch cap broken")
check(fe9$fe_used == fe9$branch_fe$regular + fe9$branch_fe$boosted, "BR branch sum mismatch")

# ---- 10. tight budget does not raise and stays under the cap ---------------
r10 <- SMCO_EVO(ff, c(-1, -1), c(1, 1), start_points = sp8, iter_max = 200,
                evolution_points = c(0.5), seed = 1, max_evals = 50)
check(r10$summary$fe$fe_used <= 50L, "tight budget exceeded cap")

cat("ALL R BUDGET TESTS PASSED\n")
