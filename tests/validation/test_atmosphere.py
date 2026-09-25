"""
Validation of the layered atmosphere (`atmosphere.py`) and of `drag.py`'s density-law selector.

`tests/validation/test_drag.py` already pins the drag *acceleration* - the 1/2, the unit conversion,
the sign of `w x r` - against a hand-computed SI value and against orbit-averaged decay. It does that
under the single-exponential law, and it still does, unchanged. This module validates the second law
and the switch between them. Every tolerance below is derived before it was measured; the measured
value follows each derivation in the comment.

1. Continuity of the table
--------------------------
A piecewise-exponential *fit* is continuous, so for every interior boundary

    rho_k * exp(-(h_{k+1} - h_k) / H_k)  ==  rho_{k+1}

That is a property of the table alone, not of the lookup, and it is the check that says whether the
28 rows were transcribed correctly. The tolerance follows from the table's own printed precision:
`rho` to 4 significant figures and `H` to 4 or 5, so the worst-case rounding residual at a boundary is

    d(rho_k)/rho_k + d(rho_{k+1})/rho_{k+1} + (dh / H_k) * d(H_k)/H_k

The `(dh / H_k)` factor is the amplification of a scale-height error across a band; it is between 0.55
and 1.9 for every band above 25 km, so the bound evaluates to at most **8.4e-4**, and
`CONTINUITY_REL_TOL = 1e-3` is that bound rounded up. **Measured maximum: 9.6e-5** (boundary 70/80 km),
an order of magnitude inside the bound, which is what randomly-signed rounding errors should look like.

**One boundary does not pass that bound, and it is reported rather than accommodated.** The 0-25 km
band closes with a residual of 1.36e-3 against a bound of 7.7e-4 - 1.8 times too large. Either one of
that band's three entries (most likely `H = 7.249` km, whose 3.45 band-widths give the table's largest
amplification) is mis-transcribed at the last digit, or Vallado's fit is genuinely discontinuous there,
which is plausible: one exponential across 25 km of troposphere and stratosphere is the crudest band
in the table and the fitter had no reason to tie it to the next. It cannot be resolved without the
text, so `test_the_lowest_band_closes_worse_than_the_tables_precision_allows` asserts it *as a known
anomaly* with its own tolerance and its own name. It is irrelevant to orbital decay - nothing orbits
at 25 km - and every boundary from 30 km up is inside 1e-4.

2. Known values
---------------
`layered_density` at a band's own base altitude must return that band's density **exactly**: the
exponent is `0.0` and `exp(0.0)` is exactly `1.0`, so there is no tolerance to derive. That pins the
table's 28 densities against the arrays and pins the band lookup at the one altitude per band where an
off-by-one is invisible to the continuity check.

The evaluation *between* base altitudes is checked against an independently written scalar reference
(a linear scan, not `np.searchsorted`) on a 250 m grid from -50 km to 1200 km, refined to 1 nm either
side of every band boundary. Agreement must be to floating-point rounding, `1e-15` relative: the two
compute the same three-operation expression on the same three floats, differing only in how they find
the band. This is the check that catches an off-by-one band index or a flipped exponent sign, neither
of which the continuity check can see.

**An off-by-one is a surprisingly small error locally**, and that is why it needs a grid rather than a
spot check. Because the table is *continuous*, evaluating band `k-1`'s exponential inside band `k`
agrees exactly at the shared boundary and only drifts apart across the band, by the ratio of the two
scale heights. The worst case anywhere in the table is about 25 % (band 200-250's `H` extrapolated to
295 km), and it is 2 to 3 % around 420 km. A single hand-computed state would catch it only because
the tolerance there is 1e-13; the grid catches it structurally.

The one genuinely *external* check available without the text is sea level, `1.225 kg/m^3`, which is
the US Standard Atmosphere's defining value, and 100 km, where USSA-76 gives `5.604e-7 kg/m^3` against
the table's `5.297e-7` - 5.5 % low, because the table is a fit, not a sampling. That comparison is
asserted at a factor of 1.5, which is a sanity check on the exponent of the number and nothing more.
Above 150 km no external check is meaningful at any tolerance: real thermospheric density varies by
more than an order of magnitude over the solar cycle and the table is a single static mean.

3. Against the single band
--------------------------
Where a single band is matched to the table - same `rho0` at the same `h0` - the two laws agree to
rounding at `h0` and nowhere else. This is a **comparison** in `CLAUDE.md`'s sense: the divergence is
the result, not a bug.

*Below the match* the direction is unambiguous: every table scale height between 355 km and the ground
is smaller than 60 km, so the table is denser, increasingly so with depth - 1.07x at 320 km, rising to
5.2x at 165 km.

*Above the match it reverses, and then reverses again*, which the first draft of this module got
wrong. The table's scale heights just above 355 km are 53.3 and 58.5 km, still under 60, so the table
is initially **thinner** - 0.90x at 480 km. But they keep growing (63.8 km at 500, 88.7 at 700, 181 at
900), and once the table's accumulated `integral dh/H` falls below `h/60` the ratio crosses back
through 1, once, between 600 and 700 km, reaching 2.9x at 830 km. A single exponential band cannot
reproduce that shape at all, which is the whole argument for the table.

4. What the choice of density law costs in predicted decay
----------------------------------------------------------
The headline comparison. Two identical satellites, seeded circular at 355 km over an equatorial,
non-rotating atmosphere (`omega = 0`, so the co-rotation factor `test_drag.py` already validates plays
no part here), `B = 0.4 m^2/kg`, propagated under Cowell + `point_mass_gravity` + drag for 3 days.
One carries the layered law; the other carries a single band matched to the table at 355 km with
`H = 60 km`, the "one number for LEO" choice a user actually makes. A third carries no drag at all and
measures the integrator's own drift.

*Derived sign.* The table's scale heights over the altitudes traversed are 53.3 km (350-400), 53.6 km
(300-350), 45.5 km (250-300) and 37.1 km (200-250) - all **smaller** than 60. A smaller scale height
means density rises faster on the way down, so the layered law is denser than the matched single band
at every altitude below 355 km, and by a ratio that grows monotonically. The layered satellite must
therefore decay **more**. If the measured difference came out the other way, the band lookup would be
reading the table upside down.

*Derived size.* With `da/dt = -rho(a) B' sqrt(mu a)` and a single scale height `H`, holding
`sqrt(mu a)` constant integrates in closed form:

    Delta a = H ln(1 - k t / H),         k = rho(a0) B' sqrt(mu a0)

Here `k = 1.796e-4 km/s` and `t = 259200 s`, so `k t = 46.55 km`. With `H = 60` that gives
**-89.7 km**; with the table's scale height at the *starting* band, `H = 53.3`, it gives **-110.2 km**.
The layered profile keeps steepening below 300 km, so its true decay must exceed 110 km. The expected
difference is therefore **about -21 km, and somewhat larger in magnitude** - call it 20 to 30 km,
20-35 % of the single-band decay.

*Measured:* single band -89.196 km against the closed form's -89.7 km (0.6 %, the residual being the
`sqrt(mu a)` variation the closed form freezes); layered -116.252 km; **difference -27.056 km, the
layered law predicting 30.3 % more decay over three days.**

*Error budget, for `DECAY_REL_TOL`.* Each satellite's measured `Delta a` is also compared with a
scalar RK4 integration of the mean equation using its own density law, which is what makes this a
validation and not a snapshot. The budget follows `test_drag.py`'s, re-evaluated at this scenario's
much stronger drag (11 km per orbit at the end, against 0.2 km there):

- *Osculating vs mean.* Drag drives an eccentricity `e ~ rho B' a`, which at the final altitude is
  2.6e-4, so `a e = 1.71 km` and `a e / H = 0.046`. The osculating `a` therefore carries an
  oscillation of amplitude `|da/dt| (a e / H) / n` = 0.081 km, and the endpoint samples it at whatever
  phase it lands on: **7.0e-4** of the 116 km decay. This dominates.
- *Mean-rate second order.* The orbit-averaged rate for a slightly eccentric orbit gains a factor
  `I_0(a e / H) ~ 1 + (a e / H)^2 / 4`, which is 5.3e-4 at the very end and far less earlier; weighted
  over the decay, under **1e-4**.
- *RK4 drift.* `(n dt)^6 / 36` per step at `n dt = 0.0346` is 4.8e-11, or 4.1e-7 over 8640 steps,
  which is 2.8e-3 km in `a`: **2.4e-5** of the decay. The constant is 1/36 on a circular Kepler orbit,
  not the harmonic oscillator's 1/72 (measured 1/36.0 at `n dt` = 0.035 and 0.069; see
  `docs/engineering-log.md`). The drag-free control measures it directly.
- *Mean-ODE quadrature.* Converged: 2000 and 32000 RK4 steps agree to 1e-10 relative, including across
  the band kinks, so it contributes nothing.

Budget 7.0e-4 + 1e-4 = 8e-4; doubled for headroom, `DECAY_REL_TOL = 2e-3`. **Measured 1.8e-4
(layered) and 1.9e-4 (single band)**, both a factor of 4 inside the derived dominant term, consistent
with the endpoint landing at a favourable phase rather than the worst one.

Negative controls
-----------------
Each mutation below was applied to the real `src/orbital_engine/atmosphere.py`, this module and
`test_drag.py` were run, and the file was restored from git.

- *off-by-one band index* (`searchsorted(..., side="right") - 1` -> `- 2`) - **7 failures**: the
  scalar-reference lookup, the base-altitude values, monotonicity, the masked-row check, the
  matched-single-band comparison, the hand-computed closed form, and the decay comparison.
- *sign flip in the band exponent* (`np.exp(-exponent)` -> `np.exp(exponent)`) - **6 failures**: the
  scalar-reference lookup, monotonicity, the matched-single-band comparison, the closed form, and
  both decay tests (the orbit reaches `exp` overflow and the state goes non-finite).

Three things that did *not* fail are worth recording, because they say what each check is for:

- **All ten of `test_drag.py`'s tests pass under both mutants.** The single-exponential path shares no
  code with the layered one, which is the backwards-compatibility claim, demonstrated rather than
  asserted.
- **The continuity check survives both.** It reads `BASE_*` directly and never calls the lookup. It
  tests the 28 transcribed rows; the scalar reference tests the lookup. Two different bugs.
- **`test_the_two_laws_agree_where_matched_and_diverge_away_from_it` survived the off-by-one in its
  first draft**, because it sampled 300/250/200/150 km - all *base* altitudes, where the table's own
  continuity makes reading one band away give exactly the right answer. It now samples 320/275/225/165
  and 480/650/830 km instead, and fails. That near miss is the reason the grid check exists at all.
- `test_decay_under_each_density_law_matches_its_own_orbit_averaged_rate` does not fail under the
  off-by-one, and cannot: its mean ODE calls `layered_density` too, so both sides move together. It
  validates the *propagated* decay against the *orbit-averaged* decay for a given density law - it is
  not, and is not meant to be, an independent check of the density law itself. That is what the
  scalar reference and the hand-computed closed form are for.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import atmosphere, drag, gravity, registry, scenarios
from orbital_engine.atmosphere import (
    BASE_ALTITUDE_KM, BASE_DENSITY_KG_M3, DENSITY_MODEL_EXPONENTIAL, DENSITY_MODEL_LAYERED,
    SCALE_HEIGHT_KM, layered_density,
)
from orbital_engine.custom_types import PropagatorType
from orbital_engine.database import Base
from orbital_engine.drag import DRAG_MODEL, DRAG_PARAM_NAMES, EARTH_OMEGA
from orbital_engine.geopotential import EARTH_R_EQ
from orbital_engine.simulator import Simulation
from orbital_engine.sweep import ForceModelSpec, ModelConfig, apply_config

ArrF = NDArray[np.float64]

MU = scenarios.MU_EARTH
N_BANDS = BASE_ALTITUDE_KM.size

# --------------------------------------------------------------------------------------------------
# Tolerances, derived (see the module docstring).
# --------------------------------------------------------------------------------------------------

# Worst-case residual from 4-significant-figure rounding of rho and H, rounded up. Measured max 9.6e-5
# over the boundaries at 30 km and above.
CONTINUITY_REL_TOL = 1e-3
# The 0-25 km boundary, whose residual exceeds its own 7.7e-4 bound. Pinned, not accommodated: see the
# module docstring. An off-by-one or a flipped exponent gives O(1) here, so this still discriminates.
LOWEST_BAND_ANOMALY_REL = 3e-3
# Vectorised lookup against a scalar linear scan: the same three floats through the same expression,
# so only floating-point rounding separates them.
LOOKUP_REL_TOL = 1e-15
# USSA-76 at 100 km is 5.604e-7 kg/m^3 from memory; the table's fit reads 5.297e-7, 5.5% low. A factor
# of 1.5 checks the exponent and nothing finer - see the module docstring on why no tighter external
# check exists above 150 km.
EXTERNAL_SANITY_FACTOR = 1.5
# Decay against the mean ODE. Derived 8e-4 (osculating endpoint scatter dominating), doubled.
# Measured 1.8e-4 layered, 1.9e-4 single band.
DECAY_REL_TOL = 2e-3
# The drag-free control's |Delta a| as a fraction of the drag decay. Derived 1.2e-5, measured 2.3e-5;
# 1e-4 is 4x the measurement and 8x the derivation.
INTEGRATOR_DRIFT_REL_TOL = 1e-4

# --------------------------------------------------------------------------------------------------
# Decay-comparison scenario. See section 4 of the module docstring.
# --------------------------------------------------------------------------------------------------

H_START_KM = 355.0                                  # spherical altitude above EARTH_R_EQ
A0_KM = EARTH_R_EQ + H_START_KM                     # 6733.137 km
BALLISTIC_COEFF = 0.4                               # m^2/kg, a high area-to-mass drag sail
SINGLE_BAND_SCALE_HEIGHT_KM = 60.0                  # the "one number for LEO" choice
RHO_START = float(layered_density(np.array([H_START_KM]))[0])   # 8.6657e-12 kg/m^3
DT = 30.0
N_STEPS = int(round(3.0 * 86400.0 / DT))            # 3 days, 8640 steps
PERIOD = 2.0 * math.pi * math.sqrt(A0_KM ** 3 / MU)

_COMMON_COEFFS: Dict[str, float] = dict(
    ballistic_coeff=BALLISTIC_COEFF, r_ref=EARTH_R_EQ, omega=0.0,
)
_SINGLE_BAND_COEFFS: Dict[str, float] = dict(
    rho0=RHO_START, h0=H_START_KM, scale_height=SINGLE_BAND_SCALE_HEIGHT_KM,
    density_model=DENSITY_MODEL_EXPONENTIAL, **_COMMON_COEFFS,
)
_LAYERED_COEFFS: Dict[str, float] = dict(
    density_model=DENSITY_MODEL_LAYERED, **_COMMON_COEFFS,
)


def _session() -> Session:
    """An isolated in-memory database, built the way `tests/conftest.py` builds one."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                          poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _sat_slots(sim: Simulation) -> List[int]:
    return sorted(i for n, i in sim.name_to_index.items() if n.startswith("SAT-"))


# --------------------------------------------------------------------------------------------------
# Independent physics used by the assertions. None of it calls `atmosphere.layered_density`.
# --------------------------------------------------------------------------------------------------

def _scalar_layered_density(altitude_km: float) -> float:
    """
    The piecewise law written as a linear scan over the table - the band lookup done the obvious,
    slow way, with no `np.searchsorted` and no index arithmetic to get wrong. It reads the same three
    arrays, so it is a reference for the *lookup and evaluation*, not for the table's values; those
    are what the continuity check is for.
    """
    band = 0
    for j in range(N_BANDS):
        if altitude_km >= float(BASE_ALTITUDE_KM[j]):
            band = j
    return float(BASE_DENSITY_KG_M3[band]) * math.exp(
        -(altitude_km - float(BASE_ALTITUDE_KM[band])) / float(SCALE_HEIGHT_KM[band]))


def _single_band_density(altitude_km: float) -> float:
    return RHO_START * math.exp(-(altitude_km - H_START_KM) / SINGLE_BAND_SCALE_HEIGHT_KM)


def _semi_major_axis(y: ArrF) -> ArrF:
    r = np.linalg.norm(y[..., :3], axis=-1)
    v2 = np.einsum("...i,...i->...", y[..., 3:], y[..., 3:])
    a: ArrF = -MU / (2.0 * (0.5 * v2 - MU / r))
    return a


def _mean_decay(a0: float, t_end: float, density: Callable[[float], float], m: int = 4000) -> float:
    """
    Integrate `da/dt = -rho(a - r_ref) B' sqrt(mu a)` with scalar RK4. `B'` carries the 1e3 that turns
    `rho B` from 1/m into 1/km. The derivation of this rate is `test_drag.py`'s module docstring; the
    only thing this module changes is which `rho` goes into it.
    """
    def rate(a: float) -> float:
        rho = density(a - EARTH_R_EQ)
        return -rho * BALLISTIC_COEFF * 1e3 * math.sqrt(MU * a)

    h = t_end / m
    a = a0
    for _ in range(m):
        k1 = rate(a)
        k2 = rate(a + 0.5 * h * k1)
        k3 = rate(a + 0.5 * h * k2)
        k4 = rate(a + h * k3)
        a += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return a - a0


# --------------------------------------------------------------------------------------------------
# The propagation run, shared by the decay tests.
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DecayRun:
    # (2, 3, 6): [start, end] x [layered, single band, drag-free control], relative to Earth.
    endpoints: ArrF
    fused_ok: bool


@pytest.fixture(scope="module")
def decay() -> Iterator[DecayRun]:
    sim = scenarios.earth_constellation(
        _session(), n_sats=3, n_planes=3,
        altitude_km=A0_KM - scenarios.EARTH_RADIUS, inclination_deg=0.0,
    )
    sim.record_history = False
    earth = sim.name_to_index["Earth"]
    sats = _sat_slots(sim)

    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model(gravity.POINT_MASS_MODEL, bodies=sats)
    sim.enable_force_model(DRAG_MODEL, bodies=sats[0], **_LAYERED_COEFFS)
    sim.enable_force_model(DRAG_MODEL, bodies=sats[1], **_SINGLE_BAND_COEFFS)

    endpoints = np.empty((2, 3, 6), dtype=np.float64)
    endpoints[0] = sim.global_states[sats] - sim.global_states[earth]
    for _ in range(N_STEPS):
        sim.step(DT)
    endpoints[1] = sim.global_states[sats] - sim.global_states[earth]
    yield DecayRun(endpoints=endpoints, fused_ok=sim._cowell_fused_ok)


# ==================================================================================================
# 1. The table itself
# ==================================================================================================

def test_density_is_continuous_across_every_band_boundary() -> None:
    """Each band, extrapolated to the next band's base altitude, must reproduce that band's tabulated
    density. This reads the arrays directly and never uses the band lookup, so it tests the 28
    transcribed rows and nothing else. See section 1 of the module docstring for the tolerance."""
    dh = np.diff(BASE_ALTITUDE_KM)
    from_below = BASE_DENSITY_KG_M3[:-1] * np.exp(-dh / SCALE_HEIGHT_KM[:-1])
    residual = np.abs(from_below - BASE_DENSITY_KG_M3[1:]) / BASE_DENSITY_KG_M3[1:]

    assert np.all(np.diff(BASE_ALTITUDE_KM) > 0.0), "base altitudes must be strictly increasing"
    assert np.all(SCALE_HEIGHT_KM > 0.0), "every band needs a positive scale height"
    assert BASE_ALTITUDE_KM.size == 28, "Vallado Table 8-4 has 28 bands"

    worst = int(np.argmax(residual[1:])) + 1
    assert np.all(residual[1:] < CONTINUITY_REL_TOL), (
        f"boundary at {BASE_ALTITUDE_KM[worst + 1]} km closes to {residual[worst]:.2e}, "
        f"tolerance {CONTINUITY_REL_TOL:.1e}")


def test_the_lowest_band_closes_worse_than_the_tables_precision_allows() -> None:
    """A known, reported anomaly, not an accommodation: the 0-25 km boundary closes to 1.4e-3 against
    a rounding bound of 7.7e-4. Most likely one last digit of `H = 7.249 km` (the band with the
    table's largest `dh/H`, 3.45), or a genuinely discontinuous fit across the troposphere. It is
    asserted here so it cannot drift, and named so it cannot be mistaken for a passing check."""
    residual = abs(BASE_DENSITY_KG_M3[0] * math.exp(-25.0 / SCALE_HEIGHT_KM[0])
                   - BASE_DENSITY_KG_M3[1]) / BASE_DENSITY_KG_M3[1]
    assert 5e-4 < residual < LOWEST_BAND_ANOMALY_REL, (
        f"the 0-25 km boundary closes to {residual:.2e}; if this has become small, the band was "
        f"corrected and this test should be folded back into the continuity check")


def test_density_at_a_band_base_altitude_is_that_bands_tabulated_value() -> None:
    """Exact equality, with no tolerance to derive: at a base altitude the exponent is `0.0` and
    `exp(0.0)` is exactly `1.0`. Pins all 28 tabulated densities and the band lookup at the one
    altitude per band where continuity cannot see an off-by-one."""
    assert np.array_equal(layered_density(BASE_ALTITUDE_KM), BASE_DENSITY_KG_M3)
    assert float(layered_density(np.array([0.0]))[0]) == 1.225, "sea level, USSA-76's defining value"


def test_table_density_at_100_km_is_within_a_factor_of_the_published_atmosphere() -> None:
    """The one external check available without the text. USSA-76 gives 5.604e-7 kg/m^3 at 100 km
    (quoted from memory); a piecewise-exponential *fit* is not obliged to match it and this one reads
    5.5% low. Asserted at a factor of 1.5, which checks the exponent of the number and nothing finer.
    No equivalent check is asserted above 150 km: real thermospheric density swings by more than an
    order of magnitude over the solar cycle, so any tolerance wide enough to be true would be too wide
    to fail."""
    published_ussa76 = 5.604e-7
    table = float(layered_density(np.array([100.0]))[0])
    assert published_ussa76 / EXTERNAL_SANITY_FACTOR < table < published_ussa76 * EXTERNAL_SANITY_FACTOR
    assert abs(table - published_ussa76) / published_ussa76 > 0.01, (
        "the table is a fit, not a sampling of USSA-76; if it now matches exactly, the source changed")


# ==================================================================================================
# 2. The lookup
# ==================================================================================================

def test_vectorised_band_lookup_matches_an_independent_scalar_reference() -> None:
    """`np.searchsorted(..., side='right') - 1` against a linear scan, on a 250 m grid from -50 km to
    1200 km, refined to 1 nm either side of every band boundary, plus the extrapolated tails at both
    ends. This is the check an off-by-one band index or a flipped exponent sign cannot survive, and
    the one the continuity check is blind to."""
    dense = np.concatenate([BASE_ALTITUDE_KM + d for d in (-1.0, -1e-9, 0.0, 1e-9, 1e-3, 1.0)])
    grid = np.sort(np.concatenate([np.arange(-50.0, 1200.0, 0.25), dense]))
    vectorised = layered_density(grid)
    reference = np.array([_scalar_layered_density(float(h)) for h in grid])

    rel = np.abs(vectorised - reference) / reference
    assert np.all(rel < LOOKUP_REL_TOL), (
        f"worst relative disagreement {rel.max():.2e} at h = {grid[int(np.argmax(rel))]:.2f} km")
    assert vectorised.shape == grid.shape


def test_density_decreases_monotonically_and_extrapolates_at_both_ends() -> None:
    """Density falls with altitude everywhere, including across every band boundary, and the law is
    total: below the bottom band it extrapolates upward without raising, above the top band downward.
    A flipped exponent sign or a scrambled band order breaks monotonicity immediately."""
    grid = np.arange(-50.0, 1500.0, 0.5)
    rho = layered_density(grid)
    assert np.all(np.diff(rho) < 0.0), "density must fall with altitude at every step of the grid"
    assert np.all(np.isfinite(rho)) and np.all(rho > 0.0)

    assert float(layered_density(np.array([-10.0]))[0]) > 1.225        # denser than sea level
    assert float(layered_density(np.array([1500.0]))[0]) < 3.019e-15   # thinner than the top row
    # The tails follow the terminal bands' own scale heights, not some other rule.
    assert float(layered_density(np.array([-10.0]))[0]) == pytest.approx(
        1.225 * math.exp(10.0 / 7.249), rel=1e-14)
    assert float(layered_density(np.array([1500.0]))[0]) == pytest.approx(
        3.019e-15 * math.exp(-500.0 / 268.0), rel=1e-14)


def test_an_invalid_mask_zeroes_rows_without_evaluating_the_exponential() -> None:
    """The guard `drag.py` relies on. A body at zero separation from its parent reads
    `altitude = -r_ref = -6378 km`, whose band-0 exponent is +880 and would overflow `exp`. With that
    row masked out, the result is exactly `0.0` and nothing is evaluated - asserted under
    `np.errstate(all='raise')`, which turns an overflow into a test failure rather than a warning."""
    h = np.array([-EARTH_R_EQ, 400.0, -EARTH_R_EQ, 90.0])
    valid = np.array([False, True, False, True])
    with np.errstate(all="raise"):
        rho = layered_density(h, valid)
    assert rho[0] == 0.0 and rho[2] == 0.0
    assert rho[1] == pytest.approx(_scalar_layered_density(400.0), rel=1e-15)
    assert rho[3] == pytest.approx(_scalar_layered_density(90.0), rel=1e-15)


# ==================================================================================================
# 3. The two laws against each other, and the kernel's selector
# ==================================================================================================

def test_the_two_laws_agree_where_matched_and_diverge_away_from_it() -> None:
    """A **comparison**, in `CLAUDE.md`'s sense, not a verification. The single band is matched to the
    table at 355 km, so the two agree there to rounding; everywhere else they must not, and the
    divergence is the result.

    Below the match the direction is one-way: every table scale height under 355 km is smaller than
    60, so the table is denser and increasingly so. Above it the ratio dips below 1 and then climbs
    back through it, because the table's scale heights start at 53.3 km and grow past 60 by 500 km.
    See section 3 of the module docstring - the reversal is real, and the first draft of this test
    asserted a monotone divergence that does not exist."""
    matched = float(layered_density(np.array([H_START_KM]))[0])
    assert matched == pytest.approx(_single_band_density(H_START_KM), rel=1e-15)

    # Deliberately *off* the band base altitudes. At a base altitude the table is continuous, so
    # reading one band away reproduces exactly the right density and an off-by-one lookup is
    # invisible - the first draft sampled 300/250/200/150 and the off-by-one mutant sailed through.
    below = np.array([320.0, 275.0, 225.0, 165.0])
    ratio_below = layered_density(below) / np.array([_single_band_density(float(h)) for h in below])
    assert np.all(ratio_below > 1.0) and np.all(np.diff(ratio_below) > 0.0), (
        f"the table must be denser below the match, increasingly so: {ratio_below}")
    # Size of the divergence, so the test fails if the two laws quietly become the same law.
    assert ratio_below[0] == pytest.approx(1.0724, abs=0.002), "1.072x at 320 km"
    assert ratio_below[-1] == pytest.approx(5.17, abs=0.05), "5.2x at 165 km"

    above = np.array([480.0, 650.0, 830.0])
    ratio_above = layered_density(above) / np.array([_single_band_density(float(h)) for h in above])
    assert ratio_above[0] == pytest.approx(0.897, abs=0.004), (
        f"just above the match the table is thinner (H = 58.5, 60.8 km < 60): {ratio_above}")
    assert ratio_above[2] == pytest.approx(2.91, abs=0.03), (
        f"by 830 km the table's 89-125 km scale heights make it far denser: {ratio_above}")
    # The crossing is between 600 and 700 km, and it is a crossing, not a tangency.
    fine = np.arange(400.5, 1000.0, 1.0)
    ratio_fine = layered_density(fine) / np.array([_single_band_density(float(h)) for h in fine])
    crossings = np.flatnonzero(np.diff(np.signbit(ratio_fine - 1.0)))
    assert crossings.size == 1 and 600.0 < fine[crossings[0]] < 700.0, (
        f"expected exactly one crossing between 600 and 700 km, got {fine[crossings].tolist()}")


# Closed form for the layered law inside the kernel, hand-computed in SI and sharing no line of code
# with it. The parent-relative state is r = (6378.137 + 420) km * (2/3, 2/3, 1/3), so h = 420 km, which
# sits in the 400-450 band: rho = 3.725e-12 * exp(-20 / 58.515) kg/m^3. The velocity is
# v = (-5, 4, 3) km/s with omega = 0, so |v| = sqrt(50) km/s and
# a = -0.5 * rho * B * |v| * v in SI (m/s^2), divided by 1e3 for km/s^2.
_CF_ALTITUDE = 420.0
_CF_R_KM = (EARTH_R_EQ + _CF_ALTITUDE) * np.array([2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0])
_CF_V = np.array([-5.0, 4.0, 3.0])
_CF_B = 0.02


def test_closed_form_layered_acceleration_matches_a_hand_computed_si_value() -> None:
    """Pins the layered law *inside the kernel*: the band it picks, the 1/2, and the 1e3 unit
    conversion, in one state. Computed in SI from the band's three numbers written out literally, so a
    unit slip (1e3) or a dropped 1/2 (2x) shows up immediately. The tolerance is rounding on ~15
    operations through two independent expressions, about 30 eps = 7e-15; 1e-13 is 14x that.

    A wrong *band* is a much smaller error than either, because the table is continuous: reading the
    400-450 band from one row away changes the density here by only 1.9 to 3.3 %. That is still 2e11 times
    this test's tolerance, so it is caught - but the structural check for an off-by-one is
    `test_vectorised_band_lookup_matches_an_independent_scalar_reference`, not this one."""
    rho_si = 3.725e-12 * math.exp(-(_CF_ALTITUDE - 400.0) / 58.515)
    speed = math.sqrt(50.0)
    # In SI, a[m/s^2] = -0.5 rho[kg/m^3] B[m^2/kg] |v|[m/s] v[m/s]. Writing |v| and v in km/s
    # multiplies by 1e6, and converting m/s^2 to km/s^2 divides by 1e3, so the net factor on the
    # km/s^2 answer is 1e3 - which is `drag._PER_M_TO_PER_KM`, arrived at here independently.
    expected = -0.5 * rho_si * _CF_B * speed * _CF_V * 1e3

    state = np.zeros((2, 6))
    state[1, :3] = _CF_R_KM
    state[1, 3:] = _CF_V
    params = np.zeros((2, len(DRAG_PARAM_NAMES)))
    params[1, :7] = [_CF_B, 0.0, 0.0, 0.0, EARTH_R_EQ, 0.0, DENSITY_MODEL_LAYERED]
    out = np.zeros((2, 3))
    with np.errstate(all="raise"):
        drag.drag_kernel(np.array([1], dtype=np.int64), 0.0, state, np.array([MU, 0.0]),
                         np.array([0, 0], dtype=np.int32), params, out)

    rel = np.abs(out[1] - expected) / np.abs(expected)
    assert np.all(rel < 1e-13), f"relative error {rel} against the SI hand computation"
    assert np.array_equal(out[0], np.zeros(3))
    # Sensitivity: reading a neighbouring band here is a 1.9% (450-500) or 3.3% (350-400) error -
    # small, but 2e11 times the tolerance above, so this state does discriminate between bands even
    # though it is not the test designed to.
    for wrong_band_rho in (9.518e-12 * math.exp(-(_CF_ALTITUDE - 350.0) / 53.298),
                           1.585e-12 * math.exp(-(_CF_ALTITUDE - 450.0) / 60.828)):
        assert abs(wrong_band_rho - rho_si) / rho_si > 0.015


def test_the_selector_dispatches_per_body_and_an_unwritten_row_stays_on_the_single_band() -> None:
    """One kernel call over an arena holding both laws at once, plus the backwards-compatibility
    guarantee: a row whose `density_model` was never written is all-zero, which is
    `DENSITY_MODEL_EXPONENTIAL`, so every configuration that pre-dates the layered law keeps its
    previous behaviour. A row with neither a scale height nor the layered selector stays a silent
    no-op, exactly as before."""
    state = np.zeros((5, 6))
    state[1:, :3] = [EARTH_R_EQ + 400.0, 0.0, 0.0]
    state[1:, 3:] = [0.0, 7.0, 0.0]
    params = np.zeros((5, len(DRAG_PARAM_NAMES)))
    params[1, :7] = [0.02, 1e-12, 400.0, 60.0, EARTH_R_EQ, 0.0, DENSITY_MODEL_EXPONENTIAL]
    params[2, :7] = [0.02, 0.0, 0.0, 0.0, EARTH_R_EQ, 0.0, DENSITY_MODEL_LAYERED]
    params[3, :7] = [0.02, 0.0, 0.0, 0.0, EARTH_R_EQ, 0.0, DENSITY_MODEL_EXPONENTIAL]  # H = 0: no-op
    params[4, :7] = [0.02, 1e-12, 400.0, 60.0, EARTH_R_EQ, 0.0, 0.0]                   # unwritten selector
    parents = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    out = np.zeros((5, 3))
    with np.errstate(all="raise"):
        drag.drag_kernel(np.array([1, 2, 3, 4], dtype=np.int64), 0.0, state, np.zeros(5),
                         parents, params, out)

    k = -0.5 * 0.02 * 1e3 * 7.0 * 7.0
    assert out[1, 1] == pytest.approx(k * 1e-12, rel=1e-14)
    assert out[2, 1] == pytest.approx(k * _scalar_layered_density(400.0), rel=1e-14)
    assert np.array_equal(out[3], np.zeros(3)), "H = 0 under the single band is still a silent no-op"
    assert out[4, 1] == out[1, 1], "an unwritten selector must mean the single exponential"
    # The two laws must actually differ here, or this test proves nothing about the selector.
    assert abs(out[2, 1] / out[1, 1] - 1.0) > 1.0


def test_enable_force_model_rejects_an_unknown_density_law() -> None:
    """`validate_coefficients` refuses anything but the known selectors, before any bit or coefficient
    is written. Without it, `density_model=3.0` would silently pick the nearest law (MSIS, since 2.0
    was taken by it) - a config typo that changes the physics and raises nothing."""
    sim = scenarios.earth_constellation(_session(), n_sats=2, n_planes=2)
    sats = _sat_slots(sim)
    mask_before = sim.force_model_mask.copy()

    with pytest.raises(ValueError, match="density_model"):
        sim.enable_force_model(DRAG_MODEL, sats, density_model=3.0, **_COMMON_COEFFS)
    assert np.array_equal(sim.force_model_mask, mask_before)
    assert DRAG_MODEL not in sim.force_model_params

    sim.enable_force_model(DRAG_MODEL, sats, **_LAYERED_COEFFS)
    bit = np.uint64(1) << np.uint64(registry.get_force_model(DRAG_MODEL).bit)
    assert np.all(sim.force_model_mask[sats] & bit)
    params = sim.force_model_params[DRAG_MODEL]
    assert np.all(params[sats, DRAG_PARAM_NAMES.index("density_model")] == DENSITY_MODEL_LAYERED)


def test_the_layered_law_still_disqualifies_the_fused_compiled_cowell_kernel() -> None:
    """The density law is a coefficient, not a mask bit, so it cannot change the fused plan - and
    must not. `kernels.cowell_rk4_step` fuses `point_mass_gravity`, `j2` and `zonal` only; a drag bit of
    either flavour has to send the whole Cowell set down the NumPy path, or the compiled path would
    silently drop drag altogether."""
    sim = scenarios.earth_constellation(_session(), n_sats=2, n_planes=2)
    sats = _sat_slots(sim)
    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model(gravity.POINT_MASS_MODEL, bodies=sats)
    assert sim._cowell_fused_ok
    sim.enable_force_model(DRAG_MODEL, bodies=sats[0], **_LAYERED_COEFFS)
    assert not sim._cowell_fused_ok


@pytest.mark.parametrize(
    "label, selector, extra",
    [("exponential", DENSITY_MODEL_EXPONENTIAL,
      dict(rho0=2e-11, h0=550.0, scale_height=60.0)),
     ("layered", DENSITY_MODEL_LAYERED, {})],
)
def test_a_model_config_can_select_the_density_law_as_plain_sweep_data(
    label: str, selector: float, extra: Dict[str, float],
) -> None:
    """The whole reason the law is a coefficient and not a second registered force model: a sweep
    expresses the choice as one more key in a `ForceModelSpec`'s `coefficients`, with no new
    machinery, no second mask bit, and no way for a body to end up carrying both laws at once. This
    is item 3 of `CLAUDE.md`'s per-feature contract - the model is immediately sweepable - checked
    through `sweep.apply_config` rather than asserted in prose."""
    config = ModelConfig(
        name=f"cowell + drag ({label})", propagator=PropagatorType.COWELL, dt=30.0,
        force_models=(
            ForceModelSpec(gravity.POINT_MASS_MODEL),
            ForceModelSpec(DRAG_MODEL, dict(density_model=selector, **_COMMON_COEFFS, **extra)),
        ),
    )
    sim = scenarios.earth_constellation(_session(), n_sats=4, n_planes=2)
    sim.record_history = False
    idx = apply_config(sim, config)

    assert idx.size == 4
    column = DRAG_PARAM_NAMES.index("density_model")
    assert np.all(sim.force_model_params[DRAG_MODEL][idx, column] == selector)
    assert not sim._cowell_fused_ok, "drag of either flavour must keep the fused plan disabled"

    before = sim.global_states[idx].copy()
    sim.step(config.dt)
    assert np.all(np.isfinite(sim.global_states[idx]))
    assert not np.array_equal(sim.global_states[idx], before)


def test_drags_citation_and_parameter_layout_carry_the_density_law() -> None:
    model = registry.get_force_model(DRAG_MODEL)
    assert model.param_names[6] == "density_model"
    assert model.validate_coefficients is not None
    assert "Table 8-4" in model.citation
    assert atmosphere.DENSITY_MODEL_EXPONENTIAL == 0.0, "the all-zero default must be the old law"


# ==================================================================================================
# 4. What the choice costs in predicted decay
# ==================================================================================================

def test_decay_under_each_density_law_matches_its_own_orbit_averaged_rate(decay: DecayRun) -> None:
    """Each satellite's measured `Delta a` against a scalar RK4 integration of
    `da/dt = -rho(a) B' sqrt(mu a)` using that satellite's own density law. This is what makes the
    layered law *validated* rather than merely *different*: the same band lookup, driven through 8640
    RK4 steps and three band boundaries, has to reproduce an independently integrated mean equation.
    The budget is in section 4 of the module docstring."""
    assert not decay.fused_ok, "drag must have disqualified the fused compiled kernel"
    a = _semi_major_axis(decay.endpoints)
    assert np.allclose(a[0], A0_KM, rtol=1e-12)
    t_end = N_STEPS * DT

    for column, label, density in (
        (0, "layered", lambda h: float(layered_density(np.array([h]))[0])),
        (1, "single band", _single_band_density),
    ):
        measured = float(a[1, column] - a[0, column])
        predicted = _mean_decay(float(a[0, column]), t_end, density)
        err = abs(measured - predicted) / abs(predicted)
        assert err < DECAY_REL_TOL, (
            f"{label}: Delta a = {measured:.6f} km against the mean ODE's {predicted:.6f} km, "
            f"relative error {err:.3e} (tolerance {DECAY_REL_TOL:.1e})")

    control = abs(float(a[1, 2] - a[0, 2])) / abs(float(a[1, 0] - a[0, 0]))
    assert control < INTEGRATOR_DRIFT_REL_TOL, f"drag-free RK4 drift is {control:.3e} of the decay"


def test_the_layered_law_predicts_measurably_more_decay_than_a_matched_single_band(
    decay: DecayRun,
) -> None:
    """**The point of the module.** Two satellites identical but for the density law, 3 days from
    355 km. The sign is derived before it is measured: every scale height the table uses between 355
    and 239 km is smaller than the single band's 60 km, so the table is denser all the way down and
    must decay more. The size is bracketed by the closed form `Delta a = H ln(1 - k t / H)`, which
    gives -89.7 km at `H = 60` and -110.2 km at the starting band's `H = 53.3`, so at least 20 km of
    difference and more, because the profile keeps steepening below 300 km.

    Measured: -116.25 km against -89.20 km, a difference of **-27.06 km**, or 30.3 % more decay.
    That number is this module's deliverable - it is what choosing a single exponential band costs a
    three-day LEO decay prediction."""
    a = _semi_major_axis(decay.endpoints)
    layered_delta = float(a[1, 0] - a[0, 0])
    single_delta = float(a[1, 1] - a[0, 1])
    difference = layered_delta - single_delta

    assert layered_delta < single_delta < 0.0, (
        f"the layered law must decay more, not less: {layered_delta:.3f} vs {single_delta:.3f} km")

    # The closed form, derived in the module docstring: Delta a = H ln(1 - k t / H), which is
    # negative because k t < H. It freezes sqrt(mu a), so it is good to ~1%.
    k = RHO_START * BALLISTIC_COEFF * 1e3 * math.sqrt(MU * A0_KM)
    closed_form_single = SINGLE_BAND_SCALE_HEIGHT_KM * math.log(
        1.0 - k * (N_STEPS * DT) / SINGLE_BAND_SCALE_HEIGHT_KM)
    closed_form_start_band = 53.298 * math.log(1.0 - k * (N_STEPS * DT) / 53.298)
    assert closed_form_single == pytest.approx(-89.7, abs=0.2)
    assert closed_form_start_band == pytest.approx(-110.2, abs=0.3)
    assert abs(single_delta - closed_form_single) / abs(closed_form_single) < 0.02, (
        "the single band must track its own closed form to the ~1% the frozen sqrt(mu a) allows")
    assert layered_delta < closed_form_start_band, (
        "the layered decay must exceed the closed form at the starting band's scale height, because "
        "every band below 350 km is steeper than that one")

    # Derived 20 to 30 km of difference (20-35% of the single-band decay); measured -27.06 km, 30.3%.
    assert -35.0 < difference < -20.0, f"difference between the laws is {difference:.3f} km"
    fraction = difference / single_delta
    assert 0.20 < fraction < 0.35, f"the layered law predicts {100 * fraction:.1f}% more decay"

    # Three band boundaries actually crossed, or the comparison is not exercising the layering.
    h_end = float(a[1, 0]) - EARTH_R_EQ
    crossed = BASE_ALTITUDE_KM[(BASE_ALTITUDE_KM > h_end) & (BASE_ALTITUDE_KM <= H_START_KM)]
    assert crossed.size >= 3, f"only crossed {crossed.tolist()} between {h_end:.1f} and {H_START_KM} km"


def test_a_co_rotating_layered_atmosphere_still_scales_drag_by_the_corotation_factor() -> None:
    """The density law and the co-rotation term are independent: swapping `rho(h)` must not disturb
    `v_rel = v - w x r`. On a prograde equatorial circular orbit the drag magnitude scales by
    `(1 - omega r / v)^2` under either law. At r = 6733.137 km, omega r = 0.49099 km/s and
    v = 7.6947 km/s, so omega r / v = 0.063815 and the factor is 0.87645. `test_drag.py` validates
    the factor itself under the single band; this checks the layered law inherits it exactly."""
    r = A0_KM
    v = math.sqrt(MU / r)
    factor = (1.0 - EARTH_OMEGA * r / v) ** 2
    assert factor == pytest.approx(0.87645, abs=1e-5)

    state = np.zeros((3, 6))
    state[1:, :3] = [r, 0.0, 0.0]
    state[1:, 3:] = [0.0, v, 0.0]
    params = np.zeros((3, len(DRAG_PARAM_NAMES)))
    params[1, :7] = [0.02, 0.0, 0.0, 0.0, EARTH_R_EQ, 0.0, DENSITY_MODEL_LAYERED]
    params[2, :7] = [0.02, 0.0, 0.0, 0.0, EARTH_R_EQ, EARTH_OMEGA, DENSITY_MODEL_LAYERED]
    out = np.zeros((3, 3))
    drag.drag_kernel(np.array([1, 2], dtype=np.int64), 0.0, state, np.zeros(3),
                     np.array([0, 0, 0], dtype=np.int32), params, out)
    assert out[2, 1] / out[1, 1] == pytest.approx(factor, rel=1e-12)
