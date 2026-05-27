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

## Build, Test, and Development Commands
- `uv venv .venv --python 3.12`: create the recommended local environment.
- `uv pip install -e ".[test]"`: install the package in editable mode with test dependencies.
- `.venv/bin/python -m pytest -q`: run the full test suite.
- `.venv/bin/python -m pytest tests/test_multifidelity.py -v`: run a focused area while changing `smco_hb`.
- `.venv/bin/python -m pytest tests/test_comparison.py -v`: validate comparison-layer changes.

Prefer the local virtualenv interpreter shown above; the repo already configures `pytest` with `pythonpath = ["src"]`.

## Coding Style & Naming Conventions
Use 4-space indentation and follow existing Python conventions: `snake_case` for functions/modules, `PascalCase` for dataclasses, and clear keyword-heavy parameter names such as `bounds_lower` and `known_best_value`. Keep edits ASCII unless a file already uses Chinese text. This repository does not currently enforce `ruff`, `black`, or `mypy`, so match the surrounding style and keep imports, dataclasses, and NumPy usage consistent with neighboring files.

## Testing Guidelines
Write tests with `pytest` and place them in `tests/test_<area>.py`. Add targeted regression tests for behavior changes before or alongside implementation, especially for scheduler logic, validation errors, and comparison semantics. Run the narrowest relevant test file first, then finish with `.venv/bin/python -m pytest -q` before opening a PR.

## Commit & Pull Request Guidelines
Recent history uses concise prefixes such as `feat:`, `fix:`, `docs:`, and combined scopes like `fix+results:`. Keep commit subjects imperative and specific, for example `feat: add SMCO-HB multifidelity optimizer`. PRs should summarize the user-visible change, list validation commands run, and link any related issue or experiment note. Include updated docs and generated artifacts only when they are part of the requested deliverable.
