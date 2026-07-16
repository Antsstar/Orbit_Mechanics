# OrbitalEngine
[![Orbital Engine CI](https://github.com/Antsstar/Orbit_Mechanics/actions/workflows/ci.yml/badge.svg)](https://github.com/Antsstar/Orbit_Mechanics/actions/workflows/ci.yml)

A 2-body orbital mechanics sandbox built to handle state-vector conversions, trajectory propagation, and orbital classifications. Developed to serve as the baseline physics utility for a planned satellite network constellation emulator.

## Core Functionality

- **Barycentric ECS Engine:** A strict Data-Oriented Design (DOD) memory arena capable of effortlessly hot-swapping between isolated 2-Body Point-Mass physics and complex N-Body Barycentric wobbles via a unified vectorized pipeline.
- **State Space Transformations**: Bi-directional conversion between Cartesian state vectors (r, v) and Classical Orbital Elements (COE).
- **Singularity Handling**: Analytical fallbacks to handle coordinate breakdowns inherent to circular, equatorial, polar, and parabolic geometries.
- **Invariant Validation**: Verification of state transitions using conservation of specific mechanical energy and angular momentum.
- **Dynamic Topology Sorting:** Automated Breadth-First Search (BFS) graph stratification allowing for cyclic-dependency breaking and perfect $O(1)$ cascaded coordinate rendering.
- **Strategy Pattern Integration:** Pure separation of concerns allowing the core Simulation manager to hot-swap between Analytical Keplerian propagators and High-Fidelity numerical integrators.
- **Package Structure**: Structured package utilizing a standard `src/` directory layout for clean downstream imports.


---

## Directory Architecture

The repository isolates core engine logic from exploratory analysis and testing configurations:

```text
orbital_engine/
├── .github/workflows/   # GitHub Actions multi-version CI configuration
├── notebooks/           # Reference derivations and data visualizations
├── src/                 # Source directory
│   └── orbital_engine/  # Core package namespace
│       ├── __init__.py  
│       ├── body.py      # DOD BodyHandle Dataclass
│       ├── constants.py # Classic unit conversions or gravitational constants.
│       ├── custom_types.py# Type Aliases (e.g., Kilometers, Radians)
│       ├── database.py  # SQLAlchemy ORM and Setup Wizard
│       ├── exceptions.py# Custom exceptions like NonConvergenceError
│       ├── frames.py    # Coordinate and state space transformations
│       ├── propagators.py# Analytical Keplerian and Barker solvers
│       ├── registry.py  # String to Function handle mapper for usable models.
│       ├── simulator.py # Simulation control loops, DOD Memory Arena & Matrix Allocation.
│       └── utilities.py # Orbital anomaly and math utilities
│       └── data/        # Database
│           └── planets.db
├── tests/               # Pytest suite
├── CHANGELOG.md/        
├── README.md/           
└── pyproject.toml       # Package metadata and installation configuration
```

---

## Installation and Setup

### 1. Environment Isolation
Isolate the workspace using Conda or your preferred virtual environment manager:

```bash
conda create -n orbital_env python=3.11 -y
conda activate orbital_env
```

### 2. Editable Development Installation
Install the package in editable mode along with development and testing dependencies:

```bash
pip install -e ".[dev, test]"
```

Using the `-e` flag links the source code to your environment, allowing modifications to propagate instantly when importing the engine inside the `notebooks/` directory.

---

## Code Verification Examples

Click to expand the examples below to see standard execution routines for the engine.

<details>
<summary><b>State Vector to Classical Orbital Elements (rv_to_coe)</b></summary>

```python
import numpy as np
import orbital_engine.frames as rf

mu = 398600.4418
r = np.array([7000.0, 0.0, 0.0])
v = np.array([0.0, np.sqrt(mu / 7000.0), 0.0])

coe = rf.rv_to_coe(r, v, mu)
print(coe)
```
</details>

<details>
<summary><b>Classical Orbital Elements to State Vector (coe_to_rv)</b></summary>

```python
import orbital_engine.frames as rf

# Resolves coordinate singularities natively via omega_true fallback logic
r_out, v_out = rf.coe_to_rv(coe, mu)
print(f"Position Vector: {r_out}")
print(f"Velocity Vector: {v_out}")
```
</details>

---

## Testing and Verification

To prevent floating-point drift and logic regressions, the engine relies on an automated test suite executed via `pytest`.

```bash
# Run the test suite from the project root
pytest
```

To run the strict type-checker:
```bash
# Run mypy from project root
mypy src/ --strict
```

### Test Strategy
* **Property-Based Fuzzing**: Generates 100+ randomized, valid state vectors to verify that round-trip conversions (`rv_to_coe` to `coe_to_rv`) resolve back to original inputs within fixed limits.
* **Boundary Validation**: Generates mathematically exact geometric edge cases (e.g., forcing orthogonal position and velocity vectors via cross-products) to ensure stability at classification limits.
* **Continuous Integration**: GitHub Actions runs the test matrix automatically on every push, validating compatibility across Python 3.10, 3.11, and 3.12.

---

## Planned Extensions

- **Analytical Perturbation Theories**: Implement variation of parameters formulations utilizing Gauss Variational Equations (GVE) and Lagrange Planetary Equations (LPE) to track secular element drift.
- **High-Fidelity Numerical Propagation**: Integrate specialized differential equation solvers, specifically implementing Cowell’s formulation for direct integration and Encke’s method for tracking deviations from a reference osculating orbit.
- **Symplectic & Geometric Integrators**: Develop structure-preserving symplectic integration algorithms to minimize artificial energy drift during long-duration chaotic n-body simulations.
- **Environmental Perturbation Forces**: Incorporate non-Keplerian acceleration models, including atmospheric drag profiles, high-degree geopotential harmonics (e.g., J2), and solar radiation pressure.
- **Trajectory Maneuver Loops**: Implement impulsive delta-v burns and continuous low-thrust propulsion profiles to support constellation deployment and station-keeping control logic.