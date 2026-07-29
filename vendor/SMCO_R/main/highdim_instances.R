# highdim_instances.R - R loader for the Task 6 high-dim instance artifacts.
#
# Mirrors src/smco/highdim_instances.py: it reads the language-neutral csv.gz
# artifacts (shift / permutation / block-rotation / starts) and rebuilds the
# optimum-preserving transform
#
#     objective(x) = scale * base_f(base_opt_x + T^{-1}(x - x_opt))
#     T^{-1}       = T_asym^{-1} . perm^{-1} . R^T
#
# so that R and Python produce identical objective values at any point (Task 6
# scenario 7, verified end-to-end by tests/test_r_instance_parity.py).
#
# Metadata scalars are passed as arguments (or read via jsonlite from
# metadata.json when available) so this module runs on base R alone, which lets
# the cross-language parity test run on machines without jsonlite installed.

# --- base functions (minimisation; identical formulas to test_functions.py) ---
base_objective <- list(
  rastrigin = function(x) 10 * length(x) + sum(x^2 - 10 * cos(2 * pi * x)),
  ackley = function(x) {
    -20 * exp(-0.2 * sqrt(mean(x^2))) - exp(mean(cos(2 * pi * x))) + 20 + exp(1)
  },
  griewank = function(x) {
    i <- seq_along(x)
    1 + sum(x^2) / 4000 - prod(cos(x / sqrt(i)))
  },
  zakharov = function(x) {
    i <- seq_along(x)
    L <- sum(0.5 * i * x)
    sum(x^2) + L^2 + L^4
  },
  rosenbrock = function(x) {
    n <- length(x)
    sum(100 * (x[2:n] - x[1:(n - 1)]^2)^2 + (1 - x[1:(n - 1)])^2)
  }
)

# base bounds and optimum location (mirror Python _BASE_REGISTRY).
.base_bounds <- list(
  rastrigin = c(-5.12, 5.12),
  ackley = c(-32.768, 32.768),
  griewank = c(-600.0, 600.0),
  zakharov = c(-5.0, 10.0),
  rosenbrock = c(-5.0, 10.0)
)

.norm_key <- function(name) {
  key <- tolower(gsub("[^a-zA-Z]", "", name))
  # collapse rosenbrock/rastrigin/etc; Python uses lowercased alnum-only key.
  key
}

base_optimum_x <- function(name, dim) {
  if (.norm_key(name) == "rosenbrock") return(rep(1.0, dim))
  rep(0.0, dim)
}

.read_gz_num <- function(path) {
  as.numeric(unlist(read.csv(gzfile(path), header = FALSE)))
}

.read_gz_int <- function(path) {
  as.integer(unlist(read.csv(gzfile(path), header = FALSE)))
}

.read_gz_matrix <- function(path) {
  as.matrix(read.csv(gzfile(path), header = FALSE))
}

# COCO T_asym inverse (power transform; identity at 0, hence optimum-preserving).
t_asym_inverse <- function(z, beta, d) {
  if (d <= 1) return(z)
  gamma <- beta * (seq_len(d) - 1) / (d - 1)
  pos <- z > 0
  z[pos] <- z[pos]^(1 / (1 + gamma[pos]))
  z
}

# Load a Task 6 instance artifact and return a list with $objective(x) plus the
# provenance fields needed by the R worker. Metadata scalars are arguments so
# the loader works without jsonlite; use load_highdim_instance_meta() to read
# them from metadata.json when jsonlite is available.
load_highdim_instance <- function(artifact_dir, function_name, dimension,
                                  asymmetry_strength, objective_scale,
                                  known_optimum_value, n_starts = 8L,
                                  default_n_starts = 8L) {
  dim <- as.integer(dimension)
  shift <- .read_gz_num(file.path(artifact_dir, "shift.csv.gz"))
  perm <- .read_gz_int(file.path(artifact_dir, "permutation.csv.gz"))
  rot <- read.csv(gzfile(file.path(artifact_dir, "rotation_blocks.csv.gz")),
                  header = FALSE)
  # Default tier (default_n_starts) lives in starts.csv.gz; other tiers in
  # starts_n{N}.csv.gz. The caller passes default_n_starts (from metadata) so
  # this loader stays base-R (no jsonlite dependency).
  starts_file <- if (as.integer(n_starts) == as.integer(default_n_starts))
    "starts.csv.gz" else paste0("starts_n", as.integer(n_starts), ".csv.gz")
  starts_path <- file.path(artifact_dir, starts_file)
  if (!file.exists(starts_path))
    stop("no starts artifact for n_starts=", n_starts, " in ", artifact_dir)
  starts <- .read_gz_matrix(starts_path)

  # Rebuild block-diagonal rotation. rot columns are
  # block_start, block_size, local_row, local_col, value (row/col are 0-based).
  block_start <- as.integer(rot[[1]])
  block_size <- as.integer(rot[[2]])
  row0 <- as.integer(rot[[3]])
  col0 <- as.integer(rot[[4]])
  val <- rot[[5]]
  starts_unique <- sort(unique(block_start))
  blocks <- vector("list", length(starts_unique))
  block_ranges <- vector("list", length(starts_unique))
  for (k in seq_along(starts_unique)) {
    bs <- starts_unique[k]
    sel <- block_start == bs
    msize <- block_size[sel][1]
    M <- matrix(0.0, msize, msize)
    M[cbind(row0[sel] + 1L, col0[sel] + 1L)] <- val[sel]
    blocks[[k]] <- M
    block_ranges[[k]] <- c(bs, bs + msize)  # 0-based half-open [start, end)
  }

  base_x <- base_optimum_x(function_name, dim)
  fkey <- .norm_key(function_name)
  base_f <- base_objective[[fkey]]
  if (is.null(base_f)) stop("unknown base function: ", function_name)
  inv_perm <- order(perm)  # matches Python np.argsort(permutation) for inverse permute

  rotate_inverse <- function(z) {
    out <- numeric(length(z))
    for (k in seq_along(blocks)) {
      rng <- block_ranges[[k]]
      s <- rng[1] + 1L
      e <- rng[2]
      out[s:e] <- drop(t(blocks[[k]]) %*% z[s:e])
    }
    out
  }

  objective <- function(x) {
    z <- as.numeric(x) - shift
    z <- rotate_inverse(z)
    z <- z[inv_perm]
    z <- t_asym_inverse(z, asymmetry_strength, dim)
    y <- base_x + z
    objective_scale * base_f(y)
  }

  bnds <- .base_bounds[[fkey]]
  list(
    function_name = function_name,
    dimension = dim,
    bounds_lower = rep(bnds[1], dim),
    bounds_upper = rep(bnds[2], dim),
    known_optimum_value = known_optimum_value,
    known_optimum_x = shift,
    starts = starts,
    objective = objective
  )
}

# Convenience: read metadata scalars from metadata.json via jsonlite (fleet).
# Returns NULL (and signals via stop) if jsonlite is unavailable.
load_instance_metadata <- function(artifact_dir) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite required to read metadata.json; pass scalars to load_highdim_instance() instead")
  }
  jsonlite::fromJSON(file.path(artifact_dir, "metadata.json"))
}
