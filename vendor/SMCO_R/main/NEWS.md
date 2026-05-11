# NEWS

## SMCO v1.1.0 (January 25, 2026)

### Code Quality Improvements

This release focuses on code quality, usability, and better defaults.

#### Changes in `compute_partial_signs()`
- In-place vector modification for cleaner code structure
- Metadata-based runmax tracking

#### Changes in `SMCO_single()`
- Pre-computed h_step lookup table
- Cached fixed bounds when `buffer_rand = FALSE`
- Pre-computed convergence threshold

#### Changes in `generate_sobol_points()`
- Vectorized scaling using R's column-wise recycling

### Performance

Performance is equivalent between v1.0.0 and v1.1.0 (within ~1% measurement noise). The code improvements focus on readability and maintainability rather than speed.

### Correctness Verification

All accuracy metrics (rMSE, AE50, AE95, AE99) are identical between v1.0.0 and v1.1.0, confirming that the optimizations do not affect numerical results.

### Repository Streamlining

- **Merged `testfunc_NN1L.R` into `testfuncs.R`**: All test functions (deterministic and neural network) are now in a single file
- **Added `benchmark.R`**: New user-configurable benchmark script for testing SMCO
- **Removed comparison algorithms**: `GD.R`, `SignGD.R`, `SPSA.R`, and `RunComparison.R` are no longer included (available in v1.0.0 archive)

### New API: Convenience Wrapper Functions

Three new wrapper functions provide simplified access to SMCO algorithm variants:

| Function | Description | Key Settings |
|----------|-------------|--------------|
| `SMCO()` | Basic single-phase optimization | `refine_search=FALSE`, `iter_boost=0` |
| `SMCO_R()` | Two-phase with local refinement | `refine_search=TRUE`, `iter_boost=0` |
| `SMCO_BR()` | Boosted refinement for maximum accuracy | `refine_search=TRUE`, `iter_boost=1000` |

**Usage example:**
```r
source("SMCO.R")
f <- function(x) -sum(x^2)
bounds_lower <- c(-5, -5)
bounds_upper <- c(5, 5)

result <- SMCO(f, bounds_lower, bounds_upper)      # Basic - fastest
result <- SMCO_R(f, bounds_lower, bounds_upper)    # Refinement - balanced
result <- SMCO_BR(f, bounds_lower, bounds_upper)   # Boosted - most accurate
```

All wrapper functions accept additional parameters via `...` (e.g., `n_starts`, `iter_max`, `verbose`) which are passed to `SMCO_multi()`.

### Documentation Updates

- **README.md**: Added Algorithm Variants section, expanded API Reference with wrapper function documentation, added Performance Comparison section
- **SMCO.R header**: Updated to document the three algorithm variants and all available functions
- **Added `benchmark_version_comparison.R`**: New script for comparing v1.0.0 vs v1.1.0 performance

---

## SMCO v1.0.0 (July 16, 2025)

Initial public release of the SMCO algorithm.

### Features
- Core SMCO algorithm with multi-start optimization
- Refinement search phase for local optimization
- Boosted search with configurable starting index
- Support for parallel execution
- Comprehensive benchmarking framework
- Deterministic and neural network-based test functions
- Comparison implementations: Gradient Descent, Sign Gradient Descent, SPSA
