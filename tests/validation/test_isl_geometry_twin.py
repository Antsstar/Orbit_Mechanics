"""
The compiled per-pair link geometry (`kernels.isl_pair_geometry`) against its NumPy definition
(`isl_scale._geometry_numpy`), and the scan built on each.

Without cones every operation is a correctly rounded add, multiply, divide or sqrt in the reference's
order, so the twin is held **bit-identical**. The cone margin goes through `arccos`/`sin`, which NumPy
and libm may round differently by an ulp, so a coned margin is held to 1e-12 of its scale. Both paths
run with numba present (machine code) and absent (the kernel interpreted).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from orbital_engine import isl, isl_scale, kernels
from orbital_engine.isl_scale import _Model, _geometry, _geometry_numpy, isl_contact_table

RNG_SEED = 20261010
EARTH_R, MOON_R = 6378.137, 1737.4


def _cloud(rng: np.random.Generator, n: int = 400) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Satellite-like positions on shells from LEO to beyond GEO, and velocities of the right order."""
    r = rng.uniform(6800.0, 60000.0, n)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    return np.ascontiguousarray(d * r[:, None]), np.ascontiguousarray(rng.normal(scale=3.0, size=(n, 3)))


def _pairs(rng: np.random.Generator, pos: NDArray[np.float64]
           ) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Random pairs plus the hard ones: coincident, radially stacked (tau* clamps to 0 and 1), perpendicular
    (tau* = 0 to rounding) and a foot of the perpendicular a hair either side of each clamp."""
    n = pos.shape[0]
    ia = rng.integers(0, n, 3000)
    ib = rng.integers(0, n, 3000)
    ia[:50] = ib[:50]                                               # coincident points
    p = pos.copy()
    extra = []
    for k in range(60):
        a = pos[k]
        u = a / np.linalg.norm(a)
        perp = np.cross(u, rng.normal(size=3))
        perp /= np.linalg.norm(perp)
        for eps in (0.0, 1e-13, -1e-13, 1e-9, -1e-9):
            extra.append(a + 1500.0 * perp * (1.0 + eps))            # foot at the a end
            extra.append(a * (1.0 + 0.3 * (1.0 + eps)))              # radially stacked
            extra.append(-a * (1.0 + eps))                           # through the centre, foot interior
            # foot at the b end: b such that -(a . d)/|d|^2 = 1 exactly in the limit
            extra.append(a + (-a * 0.5) * (1.0 + eps) + 800.0 * perp * 0.0)
    ex = np.asarray(extra)
    p = np.ascontiguousarray(np.vstack([p, ex]))
    ia2 = np.concatenate([ia, np.repeat(np.arange(60), 20)]).astype(np.int64)
    ib2 = np.concatenate([ib, n + np.arange(ex.shape[0])]).astype(np.int64)
    return p, ia2, ib2


def _both(pos, vel, ia, ib, model, occ=None, bore=None, half=None):  # type: ignore[no-untyped-def]
    ref = _geometry_numpy(pos, vel, ia, ib, model, occ, bore, half)
    twin = _geometry(pos, vel, ia, ib, model, occ, bore, half, True)
    return ref, twin


def _model(n_occ: int, max_range: Optional[float]) -> _Model:
    return _Model((EARTH_R, MOON_R)[:n_occ], (100.0, 50.0)[:n_occ], max_range, None)


@pytest.mark.parametrize("max_range", [None, 5000.0, 40000.0])
@pytest.mark.parametrize("n_occ", [1, 2])
def test_twin_is_bit_identical_without_cones(n_occ: int, max_range: Optional[float]) -> None:
    rng = np.random.default_rng(RNG_SEED)
    vel_src = _cloud(rng)
    pos, ia, ib = _pairs(rng, vel_src[0])
    vel = np.ascontiguousarray(rng.normal(scale=3.0, size=pos.shape))
    occ = np.array([[384400.0, 1000.0, -2000.0]]) if n_occ == 2 else None
    # An occulter near the cloud, so the second clearance is the binding one for some pairs.
    if occ is not None:
        occ = np.array([[30000.0, 20000.0, -10000.0]])
    (m, r, q), (m2, r2, q2) = _both(pos, vel, ia, ib, _model(n_occ, max_range), occ)
    assert np.array_equal(m, m2) and np.array_equal(r, r2) and np.array_equal(q, q2)
    assert (m > 0).any() and (m < 0).any()                           # both sides of the edge are exercised


@pytest.mark.parametrize("coned", ["a", "b", "both"])
def test_twin_with_cones_matches_to_1e12(coned: str) -> None:
    rng = np.random.default_rng(RNG_SEED + 1)
    cloud, _ = _cloud(rng)
    pos, ia, ib = _pairs(rng, cloud)
    vel = np.ascontiguousarray(rng.normal(scale=3.0, size=pos.shape))
    n = pos.shape[0]
    bore = rng.normal(size=(n, 3))
    bore /= np.linalg.norm(bore, axis=1)[:, None]
    half = np.full(n, np.nan)
    half_all = np.radians(rng.uniform(5.0, 80.0, n))
    sel = np.zeros(n, bool)
    if coned in ("a", "both"):
        sel[np.unique(ia)] = True
    if coned in ("b", "both"):
        sel[np.unique(ib)] = True
    half[sel] = half_all[sel]
    model = _model(2, 20000.0)
    occ = np.array([[30000.0, 20000.0, -10000.0]])
    (m, r, q), (m2, r2, q2) = _both(pos, vel, ia, ib, model, occ, bore, half)
    assert np.array_equal(r, r2) and np.array_equal(q, q2)           # range and rate have no cone term
    scale = np.maximum(np.abs(m), r)
    worst = float(np.max(np.abs(m - m2) / np.where(scale > 0, scale, 1.0)))
    assert worst <= 1e-12
    # The cone term was live: without it the margins differ for many pairs.
    no_cone = _geometry_numpy(pos, vel, ia, ib, model, occ)[0]
    assert (no_cone != m).sum() > 100


def test_negative_control_detects_a_dropped_occulter_and_a_perturbed_radius() -> None:
    """The comparison is sharp: a dropped second occulter or a relative-1e-9 radius error is caught."""
    rng = np.random.default_rng(RNG_SEED + 2)
    cloud, _ = _cloud(rng)
    pos, ia, ib = _pairs(rng, cloud)
    vel = np.ascontiguousarray(rng.normal(scale=3.0, size=pos.shape))
    occ = np.array([[30000.0, 20000.0, -10000.0]])
    ref = _geometry_numpy(pos, vel, ia, ib, _model(2, 20000.0), occ)
    assert np.array_equal(ref[0], _geometry(pos, vel, ia, ib, _model(2, 20000.0), occ, None, None, True)[0])
    dropped = _geometry(pos, vel, ia, ib, _model(1, 20000.0), None, None, None, True)[0]
    assert not np.array_equal(ref[0], dropped)
    skew = _Model((EARTH_R * (1.0 + 1e-9), MOON_R), (100.0, 50.0), 20000.0, None)
    assert not np.array_equal(ref[0], _geometry(pos, vel, ia, ib, skew, occ, None, None, True)[0])
    no_range = _geometry(pos, vel, ia, ib, _model(2, None), occ, None, None, True)[0]
    assert not np.array_equal(ref[0], no_range)


def _small_constellation() -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(7)
    n, steps = 24, 40
    a = rng.uniform(7000.0, 8200.0, n)
    inc = np.radians(rng.uniform(30.0, 98.0, n))
    raan = rng.uniform(0.0, 2 * np.pi, n)
    ph = rng.uniform(0.0, 2 * np.pi, n)
    t = np.arange(steps) * 60.0
    w = np.sqrt(398600.4418 / a**3)
    u = ph[None, :] + w[None, :] * t[:, None]
    x, y = a * np.cos(u), a * np.sin(u)
    ci, si, cr, sr = np.cos(inc), np.sin(inc), np.cos(raan), np.sin(raan)
    pos = np.stack([cr * x - sr * ci * y, sr * x + cr * ci * y, si * y], axis=-1)
    vx, vy = -a * w * np.sin(u), a * w * np.cos(u)
    vel = np.stack([cr * vx - sr * ci * vy, sr * vx + cr * ci * vy, si * vy], axis=-1)
    return pos, vel, t


def test_compiled_scan_equals_numpy_scan_and_dense_reference() -> None:
    pos, vel, t = _small_constellation()
    spec = isl.IslSpec("Earth", EARTH_R, 100.0, 6000.0)
    names = [f"s{k}" for k in range(pos.shape[1])]
    a = isl_contact_table(pos, vel, t, spec, names=names, compiled=False)
    b = isl_contact_table(pos, vel, t, spec, names=names, compiled=True)
    assert len(a) > 0
    assert a.to_contacts() == b.to_contacts()                        # same windows, edges bit for bit
    dense = isl.isl_contacts(pos, vel, t, spec)
    assert b.to_contacts() == dense


def test_default_follows_numba_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    pos, vel, t = _small_constellation()
    ia, ib = np.array([0, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64)
    calls = []
    real = kernels.isl_pair_geometry

    def spy(*args):  # type: ignore[no-untyped-def]
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(kernels, "isl_pair_geometry", spy)
    model = _model(1, None)
    monkeypatch.setattr(kernels, "NUMBA_AVAILABLE", False)
    _geometry(pos[0], vel[0], ia, ib, model)
    assert not calls
    monkeypatch.setattr(kernels, "NUMBA_AVAILABLE", True)
    _geometry(pos[0], vel[0], ia, ib, model)
    assert calls
    assert isl_scale._geometry is _geometry
