#####################################
# evaluation_budget.R - Exact FE budget for SMCO (R side, mirrors Python
# src/smco/evaluation.py). Task 2 of the SMCO-EVO high-dim paper campaign.
#
# An R environment holds the mutable counter/state (lists would be copied on
# write). Thread a `budget` argument through the optimizer internals; when
# `budget` is NULL every evaluation goes through the raw objective `f` (legacy
# behaviour, byte-for-byte). When set, every objective call is counted,
# attributed to an event, hard-capped at `max_evals`, and folded into a
# best-so-far record plus target-hit FE.
#
# Event names MUST match Python paper_contract.EVENTS exactly so that the
# cross-language trace contract (Task 5) and the merge script agree.
#####################################

# Evaluation events (must match Python paper_contract.EVENTS).
EVENTS <- c("initialization", "finite_difference", "iterate",
            "replacement_initialization", "restart_initialization",
            "refine", "boost", "clip_recheck")

# Default minimization-gap targets recorded as target-hit FE.
DEFAULT_GAP_TARGETS <- c(0.1, 0.01, 0.001, 1e-5)

# Canonical CSV suffixes for the default targets (paper_contract.RESULT_COLUMNS).
.GAP_LABEL_BY_VALUE <- list("1e-1" = 0.1, "1e-2" = 0.01, "1e-3" = 0.001, "1e-5" = 1e-5)

.gap_target_label_suffix <- function(v) {
  for (nm in names(.GAP_LABEL_BY_VALUE)) {
    if (isTRUE(all.equal(.GAP_LABEL_BY_VALUE[[nm]], v))) return(nm)
  }
  format(v, scientific = FALSE)
}

# Character payload of the budget-exceeded condition so callers can detect it.
BUDGET_EXCEEDED_MSG <- "smco_evaluation_budget_exceeded"

# Construct a budget context (an environment holding mutable counter/state).
budget_ctx <- function(f, max_evals = NULL, objective_sense = "maximize",
                       known_optimum = NULL,
                       gap_targets = DEFAULT_GAP_TARGETS,
                       record_trace = FALSE) {
  if (!(objective_sense %in% c("maximize", "minimize")))
    stop("objective_sense must be 'maximize' or 'minimize'")
  ctx <- new.env(parent = emptyenv())
  ctx$f <- f
  ctx$max_evals <- if (is.null(max_evals)) NULL else as.integer(max_evals)
  ctx$maximize <- identical(objective_sense, "maximize")
  ctx$objective_sense <- objective_sense
  ctx$known_optimum <- if (is.null(known_optimum)) NULL else as.numeric(known_optimum)
  ctx$gap_targets <- as.numeric(gap_targets)
  ctx$target_labels <- paste0("target_hit_fe_",
                              vapply(ctx$gap_targets, .gap_target_label_suffix, character(1)))
  ctx$record_trace <- isTRUE(record_trace)
  ctx$evaluations <- 0L
  ctx$counts <- setNames(integer(length(EVENTS)), EVENTS)
  ctx$best_value <- NULL
  ctx$best_point <- NULL
  ctx$target_hit <- setNames(as.list(rep(as.numeric(NA), length(ctx$gap_targets))),
                             ctx$target_labels)
  ctx$termination_reason <- NULL
  ctx$event_override <- NULL
  ctx$parent <- NULL
  class(ctx) <- "smco_budget_ctx"
  ctx
}

# All state lives on the owning context (the root); scoped views route through
# `parent`.
.ctx_owner <- function(ctx) if (is.null(ctx$parent)) ctx else ctx$parent

ctx_evaluations <- function(ctx) .ctx_owner(ctx)$evaluations

ctx_max_evals <- function(ctx) ctx$max_evals

ctx_can_evaluate <- function(ctx, count = 1L) {
  if (count < 0) stop("count must be non-negative")
  cap <- ctx$max_evals
  if (is.null(cap)) return(TRUE)
  ctx_evaluations(ctx) + count <= cap
}

# Pre-check an atomic step; raises a condition on failure so callers can either
# stop a step before starting it (preferred, via ctx_can_evaluate) or let the
# hard guard in evaluate_with_budget catch a stray call.
ctx_require <- function(ctx, count = 1L) {
  if (!ctx_can_evaluate(ctx, count)) {
    owner <- .ctx_owner(ctx)
    owner$termination_reason <- "evaluation_budget"
    stop(BUDGET_EXCEEDED_MSG, call. = FALSE)
  }
  invisible(TRUE)
}

ctx_termination_reason <- function(ctx) .ctx_owner(ctx)$termination_reason

ctx_set_termination <- function(ctx, reason) {
  owner <- .ctx_owner(ctx)
  owner$termination_reason <- reason
  invisible(NULL)
}

# The single evaluation entry point. Hard cap enforced as a backstop.
evaluate_with_budget <- function(ctx, f, x, event = "iterate") {
  if (!(event %in% EVENTS)) stop("unknown evaluation event: ", event)
  eff <- if (is.null(ctx$event_override)) event else ctx$event_override
  owner <- .ctx_owner(ctx)
  cap <- ctx$max_evals
  if (!is.null(cap) && owner$evaluations >= cap) {
    owner$termination_reason <- "evaluation_budget"
    stop(BUDGET_EXCEEDED_MSG, call. = FALSE)
  }
  val <- as.numeric(f(as.numeric(x)))
  owner$evaluations <- owner$evaluations + 1L
  owner$counts[eff] <- owner$counts[eff] + 1L
  better <- is.null(owner$best_value) ||
    (if (owner$maximize) val > owner$best_value else val < owner$best_value)
  if (better) {
    owner$best_value <- val
    owner$best_point <- as.numeric(x)
  }
  if (!is.null(owner$known_optimum)) {
    gap <- abs(val - owner$known_optimum)
    for (i in seq_along(owner$gap_targets)) {
      lbl <- owner$target_labels[i]
      if (is.na(owner$target_hit[[lbl]]) && gap <= owner$gap_targets[i]) {
        owner$target_hit[[lbl]] <- owner$evaluations
      }
    }
  }
  val
}

# Site helper: budget NULL -> raw f (legacy); else route through the context.
# This is the function called at every objective-evaluation site in SMCO.R /
# SMCO_evo.R, mirroring Python evaluate_with_context.
eval_fe <- function(budget, f, x, event = "iterate") {
  if (is.null(budget)) return(as.numeric(f(as.numeric(x))))
  evaluate_with_budget(budget, f, x, event = event)
}

# A scoped view shares the parent's counter/state but tags every evaluation
# with one event (refine phase / boosted branch).
ctx_scoped <- function(ctx, event) {
  if (!(event %in% EVENTS)) stop("unknown evaluation event: ", event)
  view <- budget_ctx(ctx$f, max_evals = ctx$max_evals,
                     objective_sense = ctx$objective_sense,
                     known_optimum = ctx$known_optimum,
                     gap_targets = ctx$gap_targets,
                     record_trace = ctx$record_trace)
  view$parent <- .ctx_owner(ctx)
  view$event_override <- event
  view
}

# An independent context with its own counter and cap (BR regular/boosted split).
ctx_split <- function(ctx, count = NULL, fraction = NULL) {
  if (xor(is.null(count), is.null(fraction))) {
    # exactly one given — ok
  } else {
    stop("specify exactly one of count or fraction")
  }
  if (is.null(ctx$max_evals) && is.null(count)) {
    child_cap <- NULL
  } else if (!is.null(count)) {
    child_cap <- as.integer(count)
  } else {
    child_cap <- as.integer(ctx$max_evals * fraction)
  }
  budget_ctx(ctx$f, max_evals = child_cap,
             objective_sense = ctx$objective_sense,
             known_optimum = ctx$known_optimum,
             gap_targets = ctx$gap_targets,
             record_trace = ctx$record_trace)
}

# Per-event counts as a plain list (mirrors Python summary shape).
ctx_evaluation_counts <- function(ctx) as.list(.ctx_owner(ctx)$counts)

ctx_target_hit <- function(ctx) as.list(.ctx_owner(ctx)$target_hit)

ctx_summary <- function(ctx) {
  owner <- .ctx_owner(ctx)
  list(
    fe_budget = owner$max_evals,
    fe_used = owner$evaluations,
    termination_reason = owner$termination_reason,
    evaluation_counts_by_event = as.list(owner$counts),
    best_value = owner$best_value,
    target_hit_evaluations = as.list(owner$target_hit)
  )
}

# Build a context from an opt_control list (mirrors Python _maybe_build_context);
# pops the FE keys so they do not leak into opt_control. Returns list(budget, opt_control)
# or NULL when no max_evals is requested (legacy path).
maybe_build_budget <- function(f, opt_control) {
  max_evals <- opt_control$max_evals
  objective_sense <- if (is.null(opt_control$objective_sense)) "maximize" else opt_control$objective_sense
  known_optimum <- opt_control$known_optimum
  gap_targets <- if (is.null(opt_control$gap_targets)) DEFAULT_GAP_TARGETS else opt_control$gap_targets
  record <- isTRUE(opt_control$record_evaluations) || isTRUE(opt_control$record_trace)
  # Pop every FE key so opt_control stays a clean algorithm config.
  for (k in c("max_evals", "record_evaluations", "record_trace", "objective_sense",
              "known_optimum", "gap_targets")) {
    opt_control[[k]] <- NULL
  }
  if (is.null(max_evals)) return(NULL)
  list(budget = budget_ctx(
    f, max_evals = as.integer(max_evals),
    objective_sense = objective_sense,
    known_optimum = known_optimum, gap_targets = gap_targets,
    record_trace = record
  ), opt_control = opt_control)
}
