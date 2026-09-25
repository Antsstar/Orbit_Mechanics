"""
Validation of the NRLMSIS 2.0 density law (`msis_bridge.py`, `atmosphere.DENSITY_MODEL_MSIS`).

Skipped without `pymsis`; the wiring that must hold without it is `test_msis_wiring.py`. Every
tolerance is derived in this docstring or at its constant before it was measured, and the measurement
follows. Solar activity: the ECSS-E-ST-10-04C / ISO 14222 long-term levels as recalled (**from memory,
unverified**) - low F10.7 = F10.7a = 65, Ap = 0; moderate 140/140/15; high 250/250/45
(`msis_bridge.SOLAR_ACTIVITY_*`).

a. The wrap is faithful
-----------------------
The engine evaluates a profile - MSIS's average at 601 nodes, log-linear between them. Against a
**direct** `pymsis` evaluation of the same average at arbitrary altitudes, built here from its own
node construction (not `msis_bridge.quadrature_nodes`), the error of log-linear interpolation at `h`
in `[h_k, h_{k+1}]` is `(1/2) |f''(xi)| (h - h_k)(h_{k+1} - h)`, `f = ln rho`, i.e. at most
`|f''| dh^2 / 8` at a midpoint. `f''` is taken from MSIS itself, by second differences at a quarter of
the node spacing over the test point's own interval; the assertion allows 1.25x that (the
second-difference estimate of the interval's maximum) plus 3e-7 for `pymsis`'s float32 output
(2^-24 = 6e-8 on each of three averaged values). Worst bounds: 1.0e-3 below 150 km, 1.2e-4 at
150-200, 1e-4 at 200-250, 5e-5 above.

Above 100 km the leading-order term is not only a bound but the error itself: `f''` changes across
one 1-2 km interval by roughly `dh / H` <= 10 %, so wherever the term is well clear of the float32
floor (> 3e-5, where float32 noise in the second-difference curvature is under 2 %) the error must be
**0.9 to 1.1 times** it. That is what makes the test sharp - an off-by-one band is ~8x the term.
(Below 100 km it is only a bound: the scale height is 6-8 km, and `f''` was measured to change by
65 % across the 5-6 km interval, which the first draft of this test missed. So did its 1e-5 floor for
the equality: at 897 km, `f''` = 2e-5 /km^2 and float32 noise pulled the ratio to 0.909.)
**Measured**: all 138 points (3 activities) inside the bound, worst `|error| / bound` 0.87; the
largest error 6.5e-4 at 120.5 km; on the 34 resolved points `|error| / term` = **0.950 to 0.999**.

The quadrature of the average is a *design* choice, not a derivable error, so it is asserted as a
budget: against a refined quadrature (48 days, 16 latitudes) the profile's average must agree to
`QUADRATURE_BUDGET = 2e-4`, below the interpolation bound at 150-250 km. **Measured 4.7e-5** (low
activity; 2.5e-5 moderate, 2.0e-5 high).

b. A physical signature of MSIS itself
--------------------------------------
Guards on the call - column, units - that a unit slip cannot pass:

- **Column and units, exactly.** MSIS's mass density is the sum of `n_s m_s` over its own species
  columns (m^-3) with its own masses (N2 28.0134, O2 31.9988, O 15.9994, He 4, H 1, Ar 39.948,
  N 14.0067, anomalous O 15.9994; the NO column is all-NaN under 2.0). The wrap's averaged `rho` must
  equal the same average of that sum to float32 rounding plus the CODATA revision of the atomic mass
  unit between MSIS's constant and the one used here (each ~1.2e-7): `SPECIES_REL_TOL = 1e-6`. Any
  other column is off by orders of magnitude; a 1e3 unit slip by 1e3. **Measured 5.4e-8.**
- **Sea level** against USSA-76's defining 1.225 kg/m^3. The global-mean surface state is close to the
  standard atmosphere's 288 K / 1013 hPa, so a few percent: `SEA_LEVEL_REL_TOL = 0.03`. **Measured
  -0.96 %** (1.2132).
- **400 km, moderate activity**, inside the literature's solar-cycle range at that height,
  1e-12 .. 1e-11 kg/m^3. **Measured 3.38e-12.**
- **Solar high / low at 400 km.** The literature statement is "an order of magnitude" over a solar
  cycle (e.g. Emmert 2015, *Thermospheric mass density: a review*, Adv. Space Res. 56; from memory),
  with the ratio depending on how extreme the chosen endpoints are; ECSS's low level is the deep
  minimum with Ap = 0. Asserted inside `[5, 50]` - which a temperature column (~2x) or a helium column
  (falls with activity) fails. **Measured 24.1.**
- Monotonic decrease with altitude, 0..1500 km, every profile.

What the average discards, measured (moderate): the equatorial equinox **diurnal** max/min over local
time is **2.30** at 400 km (peak 14:30, trough 04:30 local time) and **1.68** at 292 km (the
literature's "~factor 2"; asserted in [1.5, 3.5] and [1.2, 2.5]); the global mean's **seasonal**
max/min over the year is **1.63** at 400 km and 1.44 at 292 km (the semiannual variation; asserted in
[1.2, 2.2]); a 51.6 deg orbit's latitude sampling differs from area weighting by **+0.19 %** at
292 km, +0.15 % at 400 km (asserted under 1 %).

c. Cross-check of the from-memory table
---------------------------------------
**A plausibility check against an independent model, not verification of the table against its
text.** `atmosphere.py`'s layered table (CIRA-72 based, "mean" activity per Vallado) over MSIS at
moderate activity (140/140/15, the ECSS moderate level - the table does not state its indices),
150..1000 km. Derived envelope: the table's reference condition is unstated, and MSIS's sensitivity
at 400 km is `d ln rho / d F10.7` = 0.011 per sfu (from the moderate and high profiles), so a +-30 sfu
reading of "mean" is +-33 %; independent empirical models at the same indices differ by 15-30 %. So
**a factor of 1.5** either way, `TABLE_ENVELOPE = 1.5`. A mis-transcribed exponent (x10) or a band
misplaced by one row is far outside; a last-digit slip is not visible here (continuity catches that,
`test_atmosphere.py`).

**Measured**, every 10 km: 1.24 at 150 km, a maximum **1.278 at 160 km**, 1.11-1.17 through 200-400
km (1.130 at 290, 1.161 at 350), crossing 1.0 at ~550 km, 0.93 at 600, 0.84 at 700, a minimum
**0.793 at 800 km**, 0.84 at 1000. So the table is denser than moderate-activity MSIS by 11-28 %
below 540 km and thinner by up to 21 % above - it reads as a denser lower thermosphere with a
steeper upper profile. Above 700 km the ratio carries a **saw-tooth of 2-4 % per band** with local
minima exactly at the table's base altitudes (700, 800, 900 km): a constant scale height per band
against MSIS's smoothly growing one, whose size is the fit's own error `|dH/dh| dh^2 / (8 H^2)` - 3.1 %
for the 800-900 km band from the table's own `H` jump. No band stands out from its neighbours beyond
that: the largest local feature below 500 km is a 4 % rise from 300 to 350 km and fall back by
400 km, and the 350-400 km band is the only thermospheric band whose `H` is *smaller* than the one
below it (53.30 against 53.63 km) - consistent with a quirk of the fit. It is not a transcription
slip of one number: continuity closes both boundaries of that band to 1e-4, which a single wrong
`rho` or `H` could not do.

d. Orbit dynamics
-----------------
`test_atmosphere.py`'s decay scenario and budget, on the MSIS law: circular equatorial, 355 km, a
non-rotating atmosphere, `B = 0.4 m^2/kg`, Cowell + point mass + drag, 3 days at 30 s. Two satellites
in **one** arena - moderate and low activity - so the kernel's per-triple grouping runs with two
profiles at once. Each is compared with a scalar RK4 integration of `da/dt = -rho(a - R) B' sqrt(mu a)`
whose density is the profile evaluated by `np.interp` on `ln rho` - the same nodes, an independent
lookup. Budget, re-derived for each:

Two RK4 terms are common to both, and both push toward *more* decay: the Kepler energy drift
`(n dt)^6 / 36` per step, 2.4e-3 km in `a` over 8640 steps at `n dt = 0.0339`; and RK4's truncation
of the drag term itself, `docs/architecture.md`'s station-keeping finding (3e-3 of the drag rate at
60 s, falling as `h^4`): **1.9e-4 of the decay** at 30 s.

- moderate: `rho(355) = 7.48e-12`, `H` = 54.6 km. The osculating endpoint scatter
  `|adot| (a e / H) / n` (`e ~ rho B' a` = 6e-5 at the end) is ~8e-5 of the decay, RK4's Kepler drift
  3e-5, its drag term 1.9e-4: ~3e-4, and `DECAY_REL_TOL = 2e-3` is `test_atmosphere.py`'s budget.
  **Measured -76.02 km against -76.00 km, 3.2e-4.**
- low: `rho(355) = 1.38e-12`, 8.2 km of decay. Scatter ~2e-6 (`e` = 4e-6); the Kepler drift is now
  2.4e-3 / 8.2 = 2.9e-4 and the drag term 1.9e-4: **4.8e-4**, `LOW_DECAY_REL_TOL = 1e-3`. **Measured
  -8.2202 km against -8.2154 km, 5.9e-4.** (The first draft of this budget had only the Kepler drift,
  1.8e-4 at a guessed 15 km of decay; the measured 5.9e-4 disagreed, and the drag term - already
  recorded in this repo - closes it to 20 %.)

The moderate satellite's decay also exceeds the closed form `Delta a = H ln(1 - k t / H)` at the
starting profile's scale height (-72.61 km), because `H` shrinks below 355 km - by 4.7 %, asserted
within 15 %; and the two satellites' first-orbit decay ratio, 0.1841, matches the density ratio at
355 km, 0.1851, to 0.55 % (asserted 1 %: over the first orbit the moderate satellite descends 0.85 km
and the low one 0.16 km, so on average they sample 0.8 % and 0.2 % more than the starting density -
a 0.6 % shift of the ratio).

Negative controls
-----------------
Each mutation was applied to the real source, this module plus `test_msis_wiring.py` and
`test_msis_delta_v.py` run (48 tests), and the file restored with `git checkout --`:

- *units slip at the wrap* (`rho * 1e3` in `msis_mean_density`) - **18 fail**: every direct
  comparison, the species sum, sea level and 400 km, the table envelope, all three decay tests (the
  orbit re-enters), and every Delta-v test.
- *wrong output column* (`Variable.N2`, a number density in m^-3) - **19 fail**: the same set plus the
  solar high/low ratio.
- *altitude in metres* (`alts * 1e3` into `pymsis`) - **18 fail**.
- *NRLMSISE-00 instead of 2.0* (`version=0`) - **10 fail**: the direct comparisons, the quadrature
  budget, the species sum (00 carries its own masses and anomalous-oxygen convention), the discards,
  and the table/single-band Delta-v predictions. Not the magnitude checks: 00 is only 1-20 % away.
- *Gauss-Legendre nodes with equal weights* - **10 fail**, the direct comparisons and Delta-v.
- *MSIS rows dispatched to the table* (`use_msis = selector >= 2.5` in `drag.py`; also running
  `test_atmosphere.py` and `test_drag.py`, 76 tests) - **9 fail**: the kernel closed form, the three
  decay tests, the cache-miss test, four Delta-v tests. All of `test_atmosphere.py` and `test_drag.py`
  pass - the legacy laws share nothing with the MSIS path.

One mutation **cannot** fail, and is recorded as an equivalent mutant rather than a gap: dropping
`version=MSIS_VERSION` lets `pymsis` default to 2.1, whose mass density is **bitwise identical** to
2.0's (2.1 adds NO, which is not part of the total) - measured on `pymsis` 0.13.0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

pymsis = pytest.importorskip("pymsis")

from orbital_engine import drag, gravity, msis_bridge, scenarios  # noqa: E402
from orbital_engine.atmosphere import (  # noqa: E402
    DENSITY_MODEL_LAYERED, DENSITY_MODEL_MSIS, layered_density,
)
from orbital_engine.custom_types import PropagatorType  # noqa: E402
from orbital_engine.database import Base  # noqa: E402
from orbital_engine.drag import DRAG_MODEL, DRAG_PARAM_NAMES, EARTH_OMEGA  # noqa: E402
from orbital_engine.geopotential import EARTH_R_EQ  # noqa: E402
from orbital_engine.msis_bridge import (  # noqa: E402
    MSIS_ALTITUDE_GRID_KM, SOLAR_ACTIVITY_HIGH, SOLAR_ACTIVITY_LOW, SOLAR_ACTIVITY_MODERATE,
    MsisProfile, msis_coefficients, msis_density, msis_mean_density, msis_profile,
)
from orbital_engine.simulator import Simulation  # noqa: E402
from orbital_engine.sweep import ForceModelSpec, ModelConfig, apply_config  # noqa: E402

ArrF = NDArray[np.float64]
Activity = Tuple[float, float, float]
MU = scenarios.MU_EARTH
LOW: Activity = (65.0, 65.0, 0.0)
MODERATE: Activity = (140.0, 140.0, 15.0)
HIGH: Activity = (250.0, 250.0, 45.0)

# Tolerances - derivations in the module docstring.
FLOAT32_FLOOR = 3e-7
CURVATURE_MARGIN = 1.25
QUADRATURE_BUDGET = 2e-4
SPECIES_REL_TOL = 1e-6
SEA_LEVEL_REL_TOL = 0.03
TABLE_ENVELOPE = 1.5
DECAY_REL_TOL = 2e-3
LOW_DECAY_REL_TOL = 1e-3

AMU_KG = 1.66053906660e-27      # CODATA 2018
MSIS_SPECIES_AMU = {             # MSIS 2.0's own species masses, in pymsis.Variable order 1..8
    1: 28.0134, 2: 31.9988, 3: 15.9994, 4: 4.0, 5: 1.0, 6: 39.948, 7: 14.0067, 8: 15.9994,
}


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _profile(activity: Activity) -> MsisProfile:
    return msis_profile(*activity)


def _engine_rho(activity: Activity, h: ArrF) -> ArrF:
    """The step-time path, `msis_bridge.msis_density`, exactly as the kernel calls it."""
    _profile(activity)
    act = np.tile(np.asarray(activity, dtype=np.float64), (h.size, 1))
    return msis_density(h, act, np.ones(h.size, dtype=bool))


# --------------------------------------------------------------------------------------------------
# Independent direct evaluation: pymsis called here, with this module's own node construction.
# --------------------------------------------------------------------------------------------------

def _direct(alts: ArrF, activity: Activity, n_days: int = 24, n_lat: int = 8,
            column: int = 0) -> NDArray[np.float64]:
    """Raw pymsis grid `(days, lons, lats, alts, 11)` on the documented average's nodes, float64."""
    day = np.floor((np.arange(n_days) + 0.5) * 365.0 / n_days).astype(int)
    dates = np.array([np.datetime64("2001-01-01T00:00") + np.timedelta64(int(d), "D") for d in day])
    lons = np.array([22.5 + 45.0 * j for j in range(8)])
    x, _ = np.polynomial.legendre.leggauss(n_lat)
    lats = np.degrees(np.arcsin(x))
    f107, f107a, ap = activity
    out = pymsis.calculate(dates, lons, lats, np.asarray(alts, dtype=np.float64),
                           [f107] * n_days, [f107a] * n_days, [[ap] * 7] * n_days, version=2.0)
    grid: NDArray[np.float64] = np.asarray(out, dtype=np.float64)
    return grid


def _average(grid: ArrF, n_lat: int = 8) -> ArrF:
    _, w = np.polynomial.legendre.leggauss(n_lat)
    days, lons = grid.shape[0], grid.shape[1]
    total = np.zeros(grid.shape[3])
    for i in range(n_lat):
        total += w[i] / 2.0 * grid[:, :, i, :].sum(axis=(0, 1))
    mean: ArrF = total / (days * lons)
    return mean


def _direct_mean(alts: ArrF, activity: Activity, n_days: int = 24, n_lat: int = 8) -> ArrF:
    return _average(_direct(alts, activity, n_days, n_lat)[..., 0], n_lat)


# ==================================================================================================
# a. The wrap is faithful
# ==================================================================================================

@pytest.mark.parametrize("activity", [LOW, MODERATE, HIGH], ids=["low", "moderate", "high"])
def test_profile_matches_a_direct_msis_average_within_the_interpolation_bound(activity: Activity) -> None:
    rng = np.random.default_rng(7)
    points = np.concatenate([rng.uniform(0.0, 1000.0, 36),
                             [0.5, 99.5, 120.5, 150.5, 175.5, 201.0, 225.0, 293.0, 401.0, 799.0]])
    k = np.searchsorted(MSIS_ALTITUDE_GRID_KM, points, side="right") - 1
    lo, hi = MSIS_ALTITUDE_GRID_KM[k], MSIS_ALTITUDE_GRID_KM[k + 1]
    step = (hi - lo) / 4.0
    # Second differences over [lo - step, hi + step] at quarter-spacing, per point, for f''.
    sub = lo[:, None] + step[:, None] * np.arange(-1, 6)[None, :]
    samples = _direct_mean(np.concatenate([points, sub.ravel()]), activity)
    direct, ln_sub = samples[:points.size], np.log(samples[points.size:]).reshape(sub.shape)
    curvature = np.max(np.abs(np.diff(ln_sub, 2, axis=1)), axis=1) / step ** 2

    engine = _engine_rho(activity, points)
    err = np.abs(engine / direct - 1.0)
    term = 0.5 * curvature * (points - lo) * (hi - points)
    bound = CURVATURE_MARGIN * term + FLOAT32_FLOOR
    worst = int(np.argmax(err / bound))
    assert np.all(err <= bound), (
        f"h = {points[worst]:.3f} km: |error| {err[worst]:.3e} against bound {bound[worst]:.3e}")
    # Not just a bound: where the term clears the float32 floor, it *is* the error, to the ~5 %
    # variation of f'' across one interval. An off-by-one band would read ~8.
    resolved = (term > 3e-5) & (points > 100.0)
    assert resolved.sum() >= 8
    ratio = err[resolved] / term[resolved]
    assert np.all((0.9 < ratio) & (ratio < 1.1)), (points[resolved], ratio)


def test_the_profile_is_the_wraps_average_at_its_own_nodes() -> None:
    """At a node the profile *is* the average - no interpolation. Against this module's own direct
    average, to summation-order rounding."""
    nodes = MSIS_ALTITUDE_GRID_KM[::37]
    profile = _profile(MODERATE)
    assert np.allclose(profile.density_kg_m3[::37], _direct_mean(nodes, MODERATE), rtol=1e-12, atol=0)
    assert np.allclose(_engine_rho(MODERATE, nodes), profile.density_kg_m3[::37], rtol=1e-14, atol=0)


def test_the_quadrature_is_inside_its_design_budget() -> None:
    alts = np.array([150.0, 200.0, 292.0, 400.0, 500.0, 700.0, 1000.0])
    for activity in (LOW, MODERATE, HIGH):
        refined = _direct_mean(alts, activity, n_days=48, n_lat=16)
        rel = np.abs(msis_mean_density(alts, *activity) / refined - 1.0)
        assert np.all(rel < QUADRATURE_BUDGET), (activity, rel)


def test_the_profile_is_a_pure_function_of_its_indices() -> None:
    """Memoisation is only legitimate if a rebuild is bitwise the same profile."""
    cached = _profile(MODERATE)
    rebuilt = msis_bridge._build_profile(*MODERATE)
    assert rebuilt is not cached
    assert np.array_equal(rebuilt.density_kg_m3, cached.density_kg_m3)
    assert np.array_equal(rebuilt.scale_height_km, cached.scale_height_km)
    assert msis_profile(140, 140, 15) is cached, "int and float spellings of a triple are one key"
    assert not cached.density_kg_m3.flags.writeable


# ==================================================================================================
# b. A physical signature of MSIS itself
# ==================================================================================================

def test_mass_density_column_equals_the_sum_over_msis_species() -> None:
    alts = np.array([0.0, 50.0, 100.0, 150.0, 292.0, 400.0, 600.0, 1000.0])
    grid = _direct(alts, MODERATE)
    species = sum(np.nan_to_num(grid[..., col]) * mass for col, mass in MSIS_SPECIES_AMU.items())
    expected = _average(species * AMU_KG)
    rel = np.abs(msis_mean_density(alts, *MODERATE) / expected - 1.0)
    assert np.all(rel < SPECIES_REL_TOL), rel


def test_sea_level_and_400_km_have_the_right_magnitude() -> None:
    rho = msis_mean_density(np.array([0.0, 400.0]), *MODERATE)
    assert abs(rho[0] / 1.225 - 1.0) < SEA_LEVEL_REL_TOL, rho[0]
    assert 1e-12 < rho[1] < 1e-11, rho[1]


def test_solar_high_over_low_at_400_km_is_of_order_ten() -> None:
    ratio = float(_engine_rho(HIGH, np.array([400.0]))[0] / _engine_rho(LOW, np.array([400.0]))[0])
    assert 5.0 < ratio < 50.0, ratio
    # Activity must order the densities at every thermospheric altitude, not just at 400 km.
    h = np.arange(150.0, 1000.1, 50.0)
    low, mod, high = (_engine_rho(a, h) for a in (LOW, MODERATE, HIGH))
    assert np.all(low < mod) and np.all(mod < high)


@pytest.mark.parametrize("activity", [LOW, MODERATE, HIGH], ids=["low", "moderate", "high"])
def test_density_decreases_monotonically_with_altitude(activity: Activity) -> None:
    profile = _profile(activity)
    assert np.all(np.diff(profile.density_kg_m3) < 0.0) and np.all(profile.scale_height_km > 0.0)
    h = np.arange(-10.0, 1500.0, 0.37)
    rho = _engine_rho(activity, h)
    assert np.all(np.diff(rho) < 0.0) and np.all(np.isfinite(rho))


def test_what_the_average_discards_has_its_stated_size() -> None:
    """The diurnal bulge, the season and the latitude sampling - measured, so the docstrings' numbers
    are pinned rather than quoted."""
    lons = np.arange(0.0, 360.0, 7.5)
    equinox = np.datetime64("2001-03-21T00:00")
    diurnal = {}
    for h in (292.0, 400.0):
        rho = np.asarray(pymsis.calculate(equinox, lons, [0.0], [h], [140.0], [140.0], [[15.0] * 7],
                                          version=2.0))[..., 0].ravel()
        diurnal[h] = float(rho.max() / rho.min())
    assert 1.5 < diurnal[400.0] < 3.5, diurnal
    assert 1.2 < diurnal[292.0] < 2.5, diurnal

    grid = _direct(np.array([400.0]), MODERATE)[..., 0]
    _, w = np.polynomial.legendre.leggauss(8)
    per_day = np.einsum("dol,l->d", grid[..., 0], w / 2.0) / grid.shape[1]
    assert 1.2 < float(per_day.max() / per_day.min()) < 2.2

    inc = math.radians(51.6)
    u = (np.arange(64) + 0.5) * 2.0 * math.pi / 64
    lats = np.degrees(np.arcsin(math.sin(inc) * np.sin(u)))
    day = np.floor((np.arange(24) + 0.5) * 365.0 / 24).astype(int)
    dates = np.array([np.datetime64("2001-01-01T00:00") + np.timedelta64(int(d), "D") for d in day])
    orbit = np.asarray(pymsis.calculate(dates, np.arange(22.5, 360.0, 45.0), lats, [292.0],
                                        [140.0] * 24, [140.0] * 24, [[15.0] * 7] * 24,
                                        version=2.0))[..., 0].mean()
    assert abs(orbit / float(_engine_rho(MODERATE, np.array([292.0]))[0]) - 1.0) < 0.01


# ==================================================================================================
# c. The from-memory table against MSIS
# ==================================================================================================

def test_the_layered_table_is_within_a_derived_envelope_of_moderate_msis() -> None:
    """A plausibility check against an independent model - **not** verification of the table against
    Vallado's text. See section c of the module docstring for the envelope and the measured profile."""
    h = np.arange(150.0, 1000.1, 10.0)
    ratio = layered_density(h) / _engine_rho(MODERATE, h)
    assert np.all(ratio < TABLE_ENVELOPE) and np.all(ratio > 1.0 / TABLE_ENVELOPE), (
        dict(zip(h[np.argsort(ratio)[[0, -1]]], np.sort(ratio)[[0, -1]])))
    # The comparison has to be able to see a table problem: an exponent slip in any band (x10) or the
    # whole table read one row off (the next band's density) leaves the envelope somewhere.
    shifted = layered_density(h + 50.0) / _engine_rho(MODERATE, h)
    assert np.any((shifted > TABLE_ENVELOPE) | (shifted < 1.0 / TABLE_ENVELOPE))


# ==================================================================================================
# The kernel and the sweep surface
# ==================================================================================================

def test_closed_form_msis_acceleration_in_the_kernel() -> None:
    """The MSIS law *inside* `drag_kernel`: selector 2.0, the row's own triple, log-linear between the
    two bracketing nodes (written out as `rho_k^(1-f) rho_{k+1}^f`), the 1/2 and the 1e3. Rounding
    on ~20 operations: 1e-13."""
    h = 420.7
    profile = _profile(HIGH)
    k = int(np.searchsorted(profile.altitude_km, h, side="right") - 1)
    f = (h - profile.altitude_km[k]) / (profile.altitude_km[k + 1] - profile.altitude_km[k])
    rho = profile.density_kg_m3[k] ** (1.0 - f) * profile.density_kg_m3[k + 1] ** f
    r = (EARTH_R_EQ + h) * np.array([2.0, 2.0, 1.0]) / 3.0
    v = np.array([-5.0, 4.0, 3.0])
    expected = -0.5 * rho * 0.02 * math.sqrt(50.0) * v * 1e3

    state = np.zeros((3, 6))
    state[1:, :3] = r
    state[1:, 3:] = v
    params = np.zeros((3, len(DRAG_PARAM_NAMES)))
    params[1] = [0.02, 0.0, 0.0, 0.0, EARTH_R_EQ, 0.0, DENSITY_MODEL_MSIS, *HIGH]
    params[2] = [0.02, 0.0, 0.0, 0.0, EARTH_R_EQ, 0.0, DENSITY_MODEL_MSIS, *MODERATE]
    out = np.zeros((3, 3))
    with np.errstate(all="raise"):
        drag.drag_kernel(np.array([1, 2], dtype=np.int64), 0.0, state, np.zeros(3),
                         np.zeros(3, dtype=np.int32), params, out)
    assert np.all(np.abs(out[1] / expected - 1.0) < 1e-13)
    ratio = out[1, 0] / out[2, 0]
    assert ratio == pytest.approx(float(_engine_rho(HIGH, np.array([h]))[0]
                                        / _engine_rho(MODERATE, np.array([h]))[0]), rel=1e-13)


def test_msis_rows_leave_every_other_row_bit_identical() -> None:
    """Legacy rows' accelerations with MSIS rows (two different triples) in the same arena are bitwise
    what they are without them; `test_msis_wiring.py` pins the latter to the pre-MSIS kernel."""
    n = 60
    rng = np.random.default_rng(3)
    state = np.zeros((n, 6))
    state[1:, 0] = EARTH_R_EQ + rng.uniform(150.0, 900.0, n - 1)
    state[1:, 3:] = rng.normal(scale=4.0, size=(n - 1, 3))
    params = np.zeros((n, len(DRAG_PARAM_NAMES)))
    params[:, 0] = 0.03
    params[:, 4] = EARTH_R_EQ
    params[:, 5] = EARTH_OMEGA
    law = np.arange(n) % 4
    params[law == 0, 1:4] = [1e-12, 400.0, 60.0]
    params[law == 1, 6] = DENSITY_MODEL_LAYERED
    params[law == 2, 6:10] = [DENSITY_MODEL_MSIS, *LOW]
    params[law == 3, 6:10] = [DENSITY_MODEL_MSIS, *HIGH]
    _profile(LOW), _profile(HIGH)
    everyone = np.arange(1, n, dtype=np.int64)
    legacy = everyone[law[1:] < 2]
    together, alone = np.zeros((n, 3)), np.zeros((n, 3))
    drag.drag_kernel(everyone, 0.0, state, np.zeros(n), np.zeros(n, dtype=np.int32), params, together)
    drag.drag_kernel(legacy, 0.0, state, np.zeros(n), np.zeros(n, dtype=np.int32), params, alone)
    assert np.array_equal(together[legacy], alone[legacy])
    assert np.all(together[everyone[law[1:] >= 2]] != 0.0)


def test_solar_activity_is_a_sweep_axis_in_coefficients_alone() -> None:
    """`ForceModelSpec` coefficients select the law and its indices; a later call may move the indices
    of an MSIS row without restating the selector, and the new profile is evaluated at that call."""
    config = ModelConfig("msis moderate", PropagatorType.COWELL, 30.0, force_models=(
        ForceModelSpec(gravity.POINT_MASS_MODEL),
        ForceModelSpec(DRAG_MODEL, {"ballistic_coeff": 0.02, "r_ref": EARTH_R_EQ,
                                    "omega": EARTH_OMEGA,
                                    **msis_coefficients(SOLAR_ACTIVITY_MODERATE)}),
    ))
    sim = scenarios.earth_constellation(_session(), n_sats=2, n_planes=2)
    sim.record_history = False
    idx = apply_config(sim, config)
    params = sim.force_model_params[DRAG_MODEL]
    assert np.all(params[idx, 6:10] == [DENSITY_MODEL_MSIS, *MODERATE])
    before = sim.global_states[idx].copy()
    sim.step(30.0)
    assert np.all(np.isfinite(sim.global_states[idx])) and not np.array_equal(
        sim.global_states[idx], before)

    sim.enable_force_model(DRAG_MODEL, idx.tolist(), **SOLAR_ACTIVITY_HIGH)
    assert np.all(params[idx, 6:10] == [DENSITY_MODEL_MSIS, *HIGH])
    sim.step(30.0)
    assert SOLAR_ACTIVITY_LOW["ap"] == 0.0, "ap = 0 is a legitimate index, not a missing one"


def test_configuring_msis_never_touches_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """A triple no other test uses, so its profile is genuinely computed here, through the real
    configuration path, with every route to the network made to raise: pymsis's index lookup, its
    download, `urllib`, and socket connections."""
    import socket
    import urllib.request

    import pymsis.msis
    import pymsis.utils

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network or space-weather lookup attempted")

    monkeypatch.setattr(pymsis.msis, "get_f107_ap", refuse)
    monkeypatch.setattr(pymsis.utils, "get_f107_ap", refuse)
    monkeypatch.setattr(pymsis.utils, "download_f107_ap", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)

    fresh = {"f107": 97.5, "f107a": 101.25, "ap": 6.5}
    assert (fresh["f107"], fresh["f107a"], fresh["ap"]) not in msis_bridge._PROFILES
    sim = scenarios.earth_constellation(_session(), n_sats=1, n_planes=1)
    sat = [i for n, i in sim.name_to_index.items() if n.startswith("SAT-")]
    sim.enable_force_model(DRAG_MODEL, sat, ballistic_coeff=0.02, r_ref=EARTH_R_EQ,
                           density_model=DENSITY_MODEL_MSIS, **fresh)
    assert (fresh["f107"], fresh["f107a"], fresh["ap"]) in msis_bridge._PROFILES


# ==================================================================================================
# d. Orbit dynamics
# ==================================================================================================

H_START_KM = 355.0
A0_KM = EARTH_R_EQ + H_START_KM
B = 0.4
DT = 30.0
N_STEPS = int(round(3.0 * 86400.0 / DT))


@dataclass(frozen=True)
class DecayRun:
    endpoints: ArrF          # (2, 2, 6): [start, end] x [moderate, low], relative to Earth
    first_orbit: ArrF        # (2,) decay of a over the first orbit, km


def _semi_major_axis(y: ArrF) -> ArrF:
    r = np.linalg.norm(y[..., :3], axis=-1)
    v2 = np.einsum("...i,...i->...", y[..., 3:], y[..., 3:])
    a: ArrF = -MU / (2.0 * (0.5 * v2 - MU / r))
    return a


@pytest.fixture(scope="module")
def decay() -> Iterator[DecayRun]:
    sim = scenarios.earth_constellation(_session(), n_sats=2, n_planes=2,
                                        altitude_km=A0_KM - scenarios.EARTH_RADIUS,
                                        inclination_deg=0.0)
    sim.record_history = False
    earth = sim.name_to_index["Earth"]
    sats = sorted(i for n, i in sim.name_to_index.items() if n.startswith("SAT-"))
    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model(gravity.POINT_MASS_MODEL, bodies=sats)
    common = dict(ballistic_coeff=B, r_ref=EARTH_R_EQ, omega=0.0)
    sim.enable_force_model(DRAG_MODEL, sats[0], **common, **msis_coefficients(SOLAR_ACTIVITY_MODERATE))
    sim.enable_force_model(DRAG_MODEL, sats[1], **common, **msis_coefficients(SOLAR_ACTIVITY_LOW))

    endpoints = np.empty((2, 2, 6))
    endpoints[0] = sim.global_states[sats] - sim.global_states[earth]
    period = 2.0 * math.pi * math.sqrt(A0_KM ** 3 / MU)
    first_orbit = np.zeros(2)
    for n in range(N_STEPS):
        sim.step(DT)
        if n + 1 == int(round(period / DT)):
            first_orbit = (_semi_major_axis(sim.global_states[sats] - sim.global_states[earth])
                           - _semi_major_axis(endpoints[0]))
    endpoints[1] = sim.global_states[sats] - sim.global_states[earth]
    yield DecayRun(endpoints=endpoints, first_orbit=first_orbit)


def _mean_decay(a0: float, t_end: float, profile: MsisProfile, m: int = 4000) -> float:
    ln_rho = np.log(profile.density_kg_m3)

    def rate(a: float) -> float:
        rho = math.exp(float(np.interp(a - EARTH_R_EQ, profile.altitude_km, ln_rho)))
        return -rho * B * 1e3 * math.sqrt(MU * a)

    h, a = t_end / m, a0
    for _ in range(m):
        k1 = rate(a)
        k2 = rate(a + 0.5 * h * k1)
        k3 = rate(a + 0.5 * h * k2)
        k4 = rate(a + h * k3)
        a += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return a - a0


@pytest.mark.parametrize("column, activity, tol", [(0, MODERATE, DECAY_REL_TOL),
                                                   (1, LOW, LOW_DECAY_REL_TOL)],
                         ids=["moderate", "low"])
def test_decay_matches_the_orbit_averaged_rate_on_the_msis_profile(
    decay: DecayRun, column: int, activity: Activity, tol: float,
) -> None:
    a = _semi_major_axis(decay.endpoints)
    measured = float(a[1, column] - a[0, column])
    predicted = _mean_decay(float(a[0, column]), N_STEPS * DT, _profile(activity))
    err = abs(measured / predicted - 1.0)
    assert err < tol, f"Delta a {measured:.4f} km vs mean ODE {predicted:.4f} km: {err:.2e}"


def test_decay_has_the_size_the_densities_predict(decay: DecayRun) -> None:
    a = _semi_major_axis(decay.endpoints)
    moderate = float(a[1, 0] - a[0, 0])
    rho0 = float(_engine_rho(MODERATE, np.array([H_START_KM]))[0])
    h_scale = float(_profile(MODERATE).scale_height_km[
        np.searchsorted(MSIS_ALTITUDE_GRID_KM, H_START_KM, side="right") - 1])
    kt = rho0 * B * 1e3 * math.sqrt(MU * A0_KM) * N_STEPS * DT
    closed_form = h_scale * math.log(1.0 - kt / h_scale)
    assert moderate < closed_form < 0.0, "H shrinks below 355 km, so the decay must exceed this"
    assert abs(moderate / closed_form - 1.0) < 0.15, (moderate, closed_form)

    density_ratio = float(_engine_rho(LOW, np.array([H_START_KM]))[0]) / rho0
    assert decay.first_orbit[1] / decay.first_orbit[0] == pytest.approx(density_ratio, rel=0.01)


def test_profiles_differing_only_in_ap_are_distinct_and_each_exact_at_a_node() -> None:
    """
    The memo is keyed on the whole `(f107, f107a, ap)` triple. Every preset differs in F10.7, so in
    review a memo that matched on `(f107, f107a)` alone passed all 48 MSIS tests - and a sweep
    comparing quiet against storm-time conditions at the same F10.7 would then silently reuse the first
    profile for both. Here two triples differ **only** in Ap.

    Expected: at a grid node (400 km; 2 km spacing above 200 km) the log-linear evaluation is exact, so
    each profile equals a direct `msis_mean_density` of its own triple to float64 rounding of the same
    float32 MSIS output (1e-12 relative is generous). And the two must differ: geomagnetic heating at
    Ap = 50 against Ap = 0 raises 400 km density by tens of percent at moderate F10.7 (the
    quiet/storm contrast is the reason Ap is an input at all); assert more than 10 %, which a units or
    column error would not mimic and a shared profile (ratio exactly 1) cannot pass.
    """
    node = np.array([400.0])
    assert node[0] in msis_bridge.MSIS_ALTITUDE_GRID_KM
    quiet = (140.0, 140.0, 0.0)
    storm = (140.0, 140.0, 50.0)
    msis_bridge.msis_profile(*quiet)
    msis_bridge.msis_profile(*storm)
    rho_quiet = float(msis_bridge.msis_density(node, np.array([quiet]))[0])
    rho_storm = float(msis_bridge.msis_density(node, np.array([storm]))[0])
    direct_quiet = float(msis_bridge.msis_mean_density(node, *quiet)[0])
    direct_storm = float(msis_bridge.msis_mean_density(node, *storm)[0])
    assert rho_quiet == pytest.approx(direct_quiet, rel=1e-12)
    assert rho_storm == pytest.approx(direct_storm, rel=1e-12)
    assert rho_storm / rho_quiet > 1.10, (rho_quiet, rho_storm)
