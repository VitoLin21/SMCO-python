# Repository Guidelines

## Project Structure & Module Organization
Core Python packages live under `src/`:

- `src/smco/`: main SMCO optimizers, benchmarks, results helpers, and `multifidelity.py` for `smco_hb`.
- `src/comparison/`: comparison algorithms and experiment runners.
- `tests/`: pytest suites such as `test_optimizer.py`, `test_comparison.py`, and `test_multifidelity.py`.
- `docs/`: usage notes, benchmark guides, and design/plan records.
- `scripts/`: reporting and visualization utilities.
- `vendor/SMCO_R/`: upstream R reference code for comparison only; do not modify unless syncing references.
- `result/` and `figures/`: generated experiment outputs and plots.

Important result-producing scripts for the current benchmark workflow:

- `scripts/run_paper_tables_with_evo.py`: run SMCO-EVO on paper Table 1/2/3 settings and write integrated CSV/JSON outputs.
- `scripts/run_synced_r_benchmarks_with_evo.py`: run the Python-synced R benchmark set with SMCO-EVO and write per-benchmark reports plus `summary.csv`.
- `scripts/package_smco_evo_results.py`: collect the current SMCO-EVO artifacts into `result/smco-evo/` and regenerate the final report.
- `scripts/run_evo_strategy_sweep_serial.py`: fill missing strategy-sweep cases serially when long-running `table2 d=200` tasks need controlled retry.

Current consolidated deliverable:

- `result/smco-evo/`: final packaged folder for this repo's SMCO-EVO benchmark campaign.
- `result/smco-evo/report.md`: detailed report covering direct SMCO vs SMCO-EVO comparison, paper tables, R-synced benchmarks, and four evolutionary strategies.
- `result/smco-evo/analysis/`: generated summary CSVs used by the report.

## Build, Test, and Development Commands
- `uv venv .venv --python 3.12`: create the recommended local environment.
- `uv pip install -e ".[test]"`: install the package in editable mode with test dependencies.
- `.venv/bin/python -m pytest -q`: run the full test suite.
- `.venv/bin/python -m pytest tests/test_multifidelity.py -v`: run a focused area while changing `smco_hb`.
- `.venv/bin/python -m pytest tests/test_comparison.py -v`: validate comparison-layer changes.

Common benchmark/report commands:

- `.venv/bin/python scripts/run_paper_tables_with_evo.py --table12-evo --base-result-dir result/comparison --evo-result-dir result/comparison_evo --integrated-dir result/paper_tables_integrated_2026-05-27`
- `.venv/bin/python scripts/run_paper_tables_with_evo.py --table3 --table3-evo-only --table3-dir result/table3_2026-05-27 --cache-dir result/table3_cache`
- `.venv/bin/python scripts/run_synced_r_benchmarks_with_evo.py --algo-set smco --out-dir result/r-synced-benchmarks-2026-05-28`
- `.venv/bin/python scripts/package_smco_evo_results.py`

Prefer the local virtualenv interpreter shown above; the repo already configures `pytest` with `pythonpath = ["src"]`.

## Coding Style & Naming Conventions
Use 4-space indentation and follow existing Python conventions: `snake_case` for functions/modules, `PascalCase` for dataclasses, and clear keyword-heavy parameter names such as `bounds_lower` and `known_best_value`. Keep edits ASCII unless a file already uses Chinese text. This repository does not currently enforce `ruff`, `black`, or `mypy`, so match the surrounding style and keep imports, dataclasses, and NumPy usage consistent with neighboring files.

## Testing Guidelines
Write tests with `pytest` and place them in `tests/test_<area>.py`. Add targeted regression tests for behavior changes before or alongside implementation, especially for scheduler logic, validation errors, and comparison semantics. Run the narrowest relevant test file first, then finish with `.venv/bin/python -m pytest -q` before opening a PR.

When changing reporting or packaging scripts, also verify the generated artifacts you touched. For SMCO-EVO work that usually means checking:

- `result/smco-evo/report.md`
- `result/smco-evo/analysis/strategy_sweep_summary.csv`
- `result/smco-evo/analysis/paper_pairwise_summary.csv`
- `result/smco-evo/analysis/r_synced_pairwise_summary.csv`

## Commit & Pull Request Guidelines
Recent history uses concise prefixes such as `feat:`, `fix:`, `docs:`, and combined scopes like `fix+results:`. Keep commit subjects imperative and specific, for example `feat: add SMCO-HB multifidelity optimizer`. PRs should summarize the user-visible change, list validation commands run, and link any related issue or experiment note. Include updated docs and generated artifacts only when they are part of the requested deliverable.
