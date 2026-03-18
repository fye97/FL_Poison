# Changelog

All notable repository-level changes are documented here.

## 2026-03-18

### Added

- Introduced the `flpoison/` namespace package as the canonical home for application code.
- Added package entrypoints so the project can be run with `python -m flpoison` and installed with console scripts.
- Added this changelog to track structural and release-facing updates.

### Changed

- Reorganized runtime code into clearer layers:
  - `flpoison/fl` for training orchestration, algorithms, models, and runtime objects
  - `flpoison/cli` for command-line parsing and batch launchers
  - `flpoison/utils` for config, logging, plotting, and performance helpers
  - `flpoison/aggregators`, `flpoison/attackers`, and `flpoison/datapreprocessor` as domain packages
- Removed the redundant root `main.py` and `batchrun.py` wrappers.
- Standardized runtime entrypoints on `python -m flpoison`, installed console scripts, and the `exps/` launchers.
- Refreshed the README to document installation, package layout, and the new entrypoints.
- Updated packaging metadata so setuptools discovers `flpoison*` packages cleanly.

### Fixed

- Optional dependencies no longer break package import for unrelated experiments:
  - `FLAME` / `DeepSight` tolerate missing `hdbscan` until used
  - `EdgeCase` tolerates missing `rarfile` until used
- Trigger image resolution now works after the package move and remains backward-compatible with older relative paths.
- Repository-root path resolution was corrected after moving config helpers under `flpoison/utils`.

### Validation

- `pytest -q tests` passed (`25 passed`)
- `python -m flpoison --help` passed
