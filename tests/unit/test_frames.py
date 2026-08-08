"""
Unit tests for the Cartesian <-> Classical Orbital Element transformations.

Each case is parametrised so a failure names the geometry that broke rather than a line number, and
every random draw uses an explicit seed so a failure can be reproduced exactly.

Randomness note: these use `np.random.default_rng(seed)` rather than `np.random.seed(...)`. The
latter mutates NumPy's global state, which leaks between tests and makes results depend on
execution order. A Generator is local to the test that made it.
"""
from __future__ import annotations

import numpy as np
import pytest

from orbital_engine.frames import ReferenceFrames as rf

MU_EARTH = 398600.4418  # km^3/s^2


# ==================================================================================================
# Known geometries - exact analytic answers
# ==================================================================================================

def _circular_equatorial() -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    r = np.array([0.0, 12000.0, 0.0])
    v = np.array([-np.sqrt(MU_EARTH / r[1]), 0.0, 0.0])
    # e and i are both undefined here; theta absorbs RAAN and argument of periapsis.
    return r, v, {0: (np.linalg.norm(r) * np.linalg.norm(v)) ** 2 / MU_EARTH,
                  1: 0.0, 2: 0.0, 5: np.pi / 2}


def _parabolic_polar() -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    r = np.array([0.0, 0.0, 16000.0])
    v = np.array([0.0, np.sqrt(2.0 * MU_EARTH / r[2]), 0.0])  # exactly escape speed -> e = 1
    return r, v, {1: 1.0, 2: np.pi / 2, 3: 3.0 * np.pi / 2, 4: np.pi / 2, 5: 0.0}


def _elliptic_inclined() -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    r = np.array([7000.0, 0.0, 0.0])
    s = np.sqrt(0.75 * MU_EARTH / r[0])
    v = np.array([0.0, -s, s])
    return r, v, {0: (np.linalg.norm(r) * np.linalg.norm(np.array([0.0, -s, s]))) ** 2 / MU_EARTH,
                  1: 0.5, 2: 3.0 * np.pi / 4, 3: 0.0, 4: 0.0, 5: 0.0}


def _hyperbolic_inclined() -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    r = np.array([10000.0, 0.0, 0.0])
    s = np.sqrt(1.5 * MU_EARTH / r[0])
    v = np.array([0.0, s, s])
    return r, v, {0: (np.linalg.norm(r) * np.linalg.norm(np.array([0.0, s, s]))) ** 2 / MU_EARTH,
                  1: 2.0, 2: np.pi / 4, 3: 0.0, 4: 0.0, 5: 0.0}


@pytest.mark.parametrize(
    "case, tol",
    [
        pytest.param(_circular_equatorial, 1e-9, id="circular-equatorial"),
        pytest.param(_parabolic_polar, 1e-9, id="parabolic-polar"),
        pytest.param(_elliptic_inclined, 1e-6, id="elliptic-inclined-e0.5"),
        pytest.param(_hyperbolic_inclined, 1e-6, id="hyperbolic-inclined-e2.0"),
    ],
)
def test_orbit_classifications(case, tol):
    """Each geometry has a closed-form answer; the singular elements must fall back correctly."""
    r, v, expected = case()
    coe, success = rf.rv_to_coe(r, v, MU_EARTH)

    assert bool(np.all(success)), "rv_to_coe rejected a valid orbit"

    for column, want in expected.items():
        assert coe[column] == pytest.approx(want, abs=tol), f"COE column {column}"


# ==================================================================================================
# Round-trip property: rv -> coe -> rv must return the original state
# ==================================================================================================

def test_roundtrip_over_random_valid_orbits():
    """
    Property test over 100 randomly generated orbits.

    The mask here excludes near-radial trajectories, where angular momentum vanishes and the element
    set is genuinely undefined - those are *expected* to fail conversion and are not the subject of
    this test. Everything else must survive the round trip.
    """
    rng = np.random.default_rng(20260808)

    r_dir = rng.random((100, 3)) - 0.5
    r_dir /= np.linalg.norm(r_dir, axis=-1, keepdims=True)
    r_in = r_dir * rng.uniform(6500.0, 50000.0, 100)[..., np.newaxis]

    v_dir = rng.random((100, 3)) - 0.5
    v_dir /= np.linalg.norm(v_dir, axis=-1, keepdims=True)
    v_in = v_dir * rng.uniform(1.0, 15.0, 100)[..., np.newaxis]

    degenerate = np.linalg.norm(np.cross(r_in, v_in), axis=-1) < 1e-3
    testable = ~degenerate

    # Without this guard the assertions below would pass vacuously if the filter ever selected
    # nothing - np.all() on an empty array is True.
    assert testable.sum() > 90, f"expected ~100 usable orbits, got {testable.sum()}"

    coe, success = rf.rv_to_coe(r_in, v_in, MU_EARTH * np.ones(100, dtype=np.float64))
    r_out, v_out, _ = rf.coe_to_rv(coe, MU_EARTH)

    checked = testable & success
    assert checked.sum() > 90, f"conversion succeeded for only {checked.sum()} orbits"

    # Relative tolerance: positions run to 5e4 km, so a fixed absolute bound would be far stricter
    # on small orbits than large ones.
    r_err = np.linalg.norm(r_out[checked] - r_in[checked], axis=-1) / np.linalg.norm(r_in[checked], axis=-1)
    v_err = np.linalg.norm(v_out[checked] - v_in[checked], axis=-1) / np.linalg.norm(v_in[checked], axis=-1)

    assert np.max(r_err) < 1e-9, f"worst position round-trip error {np.max(r_err):.3e}"
    assert np.max(v_err) < 1e-9, f"worst velocity round-trip error {np.max(v_err):.3e}"


# ==================================================================================================
# Round-trip at the coordinate singularities
# ==================================================================================================

TRIALS = list(range(25))


@pytest.mark.parametrize("trial", TRIALS)
def test_roundtrip_equatorial(trial):
    """Inclination zero: the node vector is undefined and argument of periapsis absorbs RAAN."""
    rng = np.random.default_rng(trial)
    r_mag = rng.uniform(6500.0, 42000.0)

    r = np.array([r_mag, 0.0, 0.0])
    v = float(rng.choice([-1.0, 1.0])) * np.array([0.0, rng.uniform(3.0, 7.0), 0.0])

    coe, _ = rf.rv_to_coe(r, v, MU_EARTH)
    assert coe[2] == pytest.approx(0.0, abs=1e-12) or coe[2] == pytest.approx(np.pi, abs=1e-12)

    r_out, v_out, _ = rf.coe_to_rv(coe, MU_EARTH)
    assert r_out == pytest.approx(r, abs=1e-4)
    assert v_out == pytest.approx(v, abs=1e-4)


@pytest.mark.parametrize("trial", TRIALS)
def test_roundtrip_circular(trial):
    """Eccentricity zero: the eccentricity vector is undefined and theta absorbs argument of periapsis."""
    rng = np.random.default_rng(1000 + trial)
    r_mag = rng.uniform(6500.0, 42000.0)

    r_dir = rng.standard_normal(3)
    r_dir /= np.linalg.norm(r_dir)
    r = r_dir * r_mag

    v_dir = np.cross(r, rng.standard_normal(3))  # cross product forces exact orthogonality
    v_dir /= np.linalg.norm(v_dir)
    v = v_dir * np.sqrt(MU_EARTH / r_mag)        # exact circular speed

    coe, _ = rf.rv_to_coe(r, v, MU_EARTH)
    assert coe[1] == pytest.approx(0.0, abs=1e-6)

    r_out, v_out, _ = rf.coe_to_rv(coe, MU_EARTH)
    assert r_out == pytest.approx(r, abs=1e-4)
    assert v_out == pytest.approx(v, abs=1e-4)


@pytest.mark.parametrize("trial", TRIALS)
def test_roundtrip_polar(trial):
    """Inclination pi/2: the orbit normal lies in the equatorial plane."""
    rng = np.random.default_rng(2000 + trial)
    r_mag = rng.uniform(6500.0, 42000.0)

    r = np.array([r_mag, 0.0, 0.0])
    v = np.array([0.0, 0.0, rng.uniform(3.0, 7.0)])

    coe, _ = rf.rv_to_coe(r, v, MU_EARTH)
    assert coe[2] == pytest.approx(np.pi / 2, abs=1e-12)

    r_out, v_out, _ = rf.coe_to_rv(coe, MU_EARTH)
    assert r_out == pytest.approx(r, abs=1e-4)
    assert v_out == pytest.approx(v, abs=1e-4)


@pytest.mark.parametrize("trial", TRIALS)
def test_roundtrip_parabolic(trial):
    """Eccentricity exactly 1: semi-major axis is infinite, which is why column 0 stores p."""
    rng = np.random.default_rng(3000 + trial)
    r_mag = rng.uniform(6500.0, 42000.0)

    r_dir = rng.standard_normal(3)
    r_dir /= np.linalg.norm(r_dir)
    r = r_dir * r_mag

    v_dir = np.cross(r, rng.standard_normal(3))
    v_dir /= np.linalg.norm(v_dir)
    v = v_dir * np.sqrt(2.0 * MU_EARTH / r_mag)  # exact escape speed

    coe, _ = rf.rv_to_coe(r, v, MU_EARTH)
    assert coe[1] == pytest.approx(1.0, abs=1e-6)

    r_out, v_out, _ = rf.coe_to_rv(coe, MU_EARTH)
    assert r_out == pytest.approx(r, abs=1e-3)
    assert v_out == pytest.approx(v, abs=1e-3)
