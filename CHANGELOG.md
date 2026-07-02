# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-07-02

### Added
- Implemented a Data-Oriented Design (DOD) Memory Arena in `simulator.py`, utilizing pre-allocated NumPy matrices (`local_states`, `coe_states`, `mu_array`) for future high-performance vectorized physics integration.
- Developed a robust Topological Sort algorithm (`_topological_sort`) to dynamically resolve orbital hierarchies, handle disconnected graphs, evaluated Scoped Roots, and defer Vessel Allocations for future Sphere of Influence (SoI) transition events.
- Added 6 explicit Classical Orbital Elements (COEs) columns ($p, e, i, \Omega, \omega, \theta$) to the database schema for $O(1)$ deserialization and instant SQL querying capabilities.
- Added root-anchoring logic to explicitly zero-out COEs for any body acting as the absolute spatial origin point.

### Changed
- Stripped object-oriented physics state from `body.py`. `BaseBody` is now a lightweight, immutable `BodyHandle` dataclass acting strictly as integer pointers to the DOD matrices.
- Flattened the SQLAlchemy database architecture, elevating `physics_models` to the master `BaseBodyORM` and formally introducing `VirtualNodeORM` and `VessalORM` via Single-Table Polymorphic Inheritance.
- Renamed `types.py` to `custom_types.py` to resolve Python standard library shadowing conflicts.

---

## [1.2.0] - 2026-06-15

### Added
- Integrated `mypy` strict static type-checking into GitHub Actions CI/CD pipeline.
- Implemented a dynamic Model Registry (`register.py`) using the Decorator/Factory pattern (`@register_model`) to instantiate physics classes from database configuration strings.

### Changed
- Migrated the core math and physics engine to strict static typing.
- Replaced `NewType` wrappers with standard Type Aliases (e.g., `Radians = float`) to maintain semantic function signatures without mathematical casting bloat.
- Hardened `frames.py` and `utilities.py` against `None` type traps and NumPy C-struct (`floating[Any]`).

---

## [1.1.0] - 2026-06-02

### Added
- Implemented an `SQLAlchemy` declarative ORM (`database.py`) to manage celestial body invariants.
- Utilized Single-Table Polymorphic Inheritance to distinguish between `CelestialBodyORM` and `StellarObjectORM` within a unified database structure.
- Added a JSON column for dynamic `physics_models` registry (e.g., atmospheric drag coefficients).
- Refactored `BaseBody` initialization to automatically hydrate physical invariants (like $\mu$ and radius) from the SQLite database if not manually overridden.
- Extracted physical constants to a dedicated `constants.py` module to prevent circular dependencies.
- Updated `pyproject.toml` with `dev` optional dependencies to support local Jupyter Kernel execution.

---

## [1.0.0] - 2026-05-16

### Added
- Migrated code to a standardized `src/` layout to ensure package isolation.
- Implemented `pyproject.toml` with `setuptools` backend for package installation management.
- Developed a automated test suite using `pytest`, featuring random vector fuzzing and exact boundary condition factories.
- Configured GitHub Actions pipeline (`ci.yml`) to test code across Python 3.10, 3.11, and 3.12.

### Fixed
- Patched an equatorial coordinate breakdown within `coe_to_rv` where undefined nodes caused inverted position signs. Rebuilt transformation arrays to fall back to the True Longitude of Periapsis (`omega_true`).

---

## [0.3.0] - 2026-05-13
### Added
- Introduced `body.py`, `propagators.py`, and `simulator.py` to establish the baseline simulation control loop.
- Added `visualisations.ipynb` to process engine telemetry and plot orbital profiles.

---

## [0.2.0] - 2026-04-26
### Added
- Refactored monolithic code into domain-specific files (`utilities.py` and `frames.py`).
- Isolated orbital math into class-level `@staticmethod` blocks to provide a stateless, reusable utility layer.

---

## [0.1.0] - 2026-03-02
### Added
- Created `theory.ipynb` outlining core mathematical derivations for two-body Keplerian motion.
- Drafted functional prototype for state vector reconstruction (`coe_to_rv.py`).

---

## [0.0.1] - 2025-10-03
### Added
- Initialized repository, configured remote tracking with SSH keys, and established baseline project structure.