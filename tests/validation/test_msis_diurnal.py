"""
Validation of the diurnal NRLMSIS 2.0 law (`msis_diurnal.py`, `atmosphere.DENSITY_MODEL_MSIS_DIURNAL`).

Skipped without `pymsis` (and `scipy`, for the independent interpolator); the wiring that must hold
without them is `test_msis_diurnal_wiring.py`, and the Sun is `test_solar_ephemeris.py`. Epoch
**2024-03-20T00:00 UT** (`epoch_days` 8844.5, the March equinox day), ECSS moderate activity
140/140/15 throughout. Every tolerance below is derived before it was measured; measurements follow.

a. The wrap is faithful
-----------------------
- **Nodes.** At a node the table *is* the zonal mean: rebuilt here independently (its own UT loop,
  `lon = 15 (s - UT)`, fly-through `pymsis` calls), equal to float64 summation rounding of identical
  float32 MSIS values: `rtol = 1e-12`. **Measured 0.0 (bitwise).**
- **Off-node interpolation.** Trilinear in `ln rho`; the error at `(h, phi, s)` is to leading order
  `+ (1/2) sum_i f_ii (x_i - x_i0)(x_i1 - x_i)` (linear interpolation over-reads a convex function),
  with each `f_ii` from direct MSIS second differences across the point's own cell. The residual after
  subtracting that term is the curvature's variation across the cell: <= 10 % in altitude above 100 km,
  ~20 % in local time (harmonics up to 3/day over half a 30 min cell) and up to ~35 % in latitude
  (structure up to degree ~8 over 2.5 deg). So `|error - term| <= 0.4 sum |term_i| + 2e-6` (float32
  noise in three averaged values and in the curvature estimates), and `|error| <= 1.4 sum |term_i|
  + 2e-6`. **Measured:** see `test_off_node_interpolation_is_the_derived_leading_term`.
- **End to end, through the kernel's own geometry.** A satellite placed by an independent construction
  (IAU 1982 GMST and MSIS's `s = UT + lon / 15`) at a node's altitude, latitude and local time, at
  `t = 0` and after 5.25 days, must read the direct zonal mean there. The only mismatch is the 0.15 s
  of time between that construction and the engine's mean-Sun local time, times a largest
  `|d ln rho / d s|` of ~0.2 /h: 1e-5. `END_TO_END_TOL = 2e-5`. The apparent-Sun definition would be
  off by the equation of time (-7.5 min at this epoch, ~1 % at a steep point); a 12 h slip by 2x.

b. Physics signatures of the bulge
----------------------------------
Jacchia (1970, SAO Special Report 313; from memory) places the bulge maximum at ~14 h local solar time
near the subsolar latitude and the minimum at ~03-04 h in the opposite hemisphere; the averaged law's
own record (`msis_bridge.py`) is a **2.30x** day/night ratio at 400 km on the equator at equinox,
measured at one UT (longitude = local time). Asserted: at equinox, 400 km, peak local time in
[13.5, 15.5] h within 10 deg of the equator, minimum in [2.5, 6] h; the equatorial max/min within
**6 %** of 2.30 (the zonal mean against one longitude: the UT/longitude residual is 1.3 % rms, 4 % max
per point, so up to ~5 % on a ratio of two points). At the June and December solstices the peak moves
into the summer hemisphere (5-30 deg, sign of the Sun's declination) and the minimum into the winter
one. **Measured:** equinox peak 14.5 h at 0 deg, minimum 4.0 h at -35 deg, equatorial ratio 2.341
(+1.8 %); June peak +15 deg / 14.5 h, minimum -45 deg / 05 h; December peak -20 deg, minimum +45 deg.

c. Consistency with the averaged law
------------------------------------
Averaging the diurnal construction over latitude (Simpson on the 5 deg nodes, `cos phi` weights), local
time (48 nodes) and the averaged law's own 24 days must reproduce the global mean:
- **Sharp**, against the averaged law's quadrature (8 Gauss-Legendre latitudes, 8 longitudes, 24 days)
  evaluated here with the same 6 UT samples: the only differences are the two latitude rules, each
  measured against 32-point Gauss-Legendre at <= 1.2e-6 (Simpson) and <= 6.1e-6 (GL-8) - local-time
  sampling is exact below harmonic 8 in both. `SHARP_TOL = 2e-5`.
- **Against `msis_bridge.msis_profile` itself**, which samples 00:00 UT only: the difference is that
  profile's UT bias, the global mean's UT dependence, measured <= 0.16 / 0.08 / 0.23 % on one day at
  292 / 400 / 500 km - averaged over 24 days it can only be smaller. `PROFILE_TOL = 3e-3`.
**Measured:** see the two tests.

d. Orbit-averaged density by orbit type
---------------------------------------
Predicted from the table alone (an independent `scipy` interpolator on its `ln rho` nodes, an analytic
circular orbit, weighted by the co-rotation factor `|v_rel| (v_rel . v)` that drag actually integrates),
against the averaged law, at 400 km (292 km in brackets):

    dawn-dusk SSO (LTAN 18 h)     1.1221 (1.0753)
    noon-midnight SSO (LTAN 12 h) 1.2039 (1.1524)
    51.6 deg, node at 00 h        1.2125 (1.1492)
    this day's global mean        1.1928 (1.1432)   <- the season alone

so relative to the day's own global mean the dawn-dusk plane sees **0.941**, noon-midnight 1.009, the
51.6 deg plane 1.017: the local-time effect is -6 % / +1 %, and at this epoch the season (+19 %) is
larger than either. Confirmed by Cowell (point mass + drag, J2 zeroed so the plane and circularity are
exactly the prediction's; 1 day at 30 s, B = 0.05): each satellite's decay against the prediction's
integral of `da/dt = -(a^2/mu) B' rho |v_rel| (v_rel . v)` along its own time-resolved track (the Sun
moving, the partial last orbit included), with RK4's Kepler drift `-a (n dt)^6 / 36` per step added
(8.2e-4 km, 9e-4 of the decay). Remaining terms: RK4's truncation of drag, 1.9e-4 of the decay
(`test_msis.py`), and the prediction's own RK4 at 10 s (<1e-6): `DECAY_REL_TOL = 2e-3`, as
`test_msis.py`. **Measured:** see `test_cowell_decay_confirms_the_orbit_averaged_prediction`.

e. The headline, in Delta-v
---------------------------
`run_sweep(..., station_keeping=, delta_v_baseline="msis averaged")` on
`scenarios.sun_synchronous_satellites` (dawn-dusk and noon-midnight, 297.0 km osculating seed = mean
~292.0 km, 96.64 deg, Cowell + pm + J2, so each plane precesses with the Sun), the `msis_sweep.py`
band [291, 293.5] km, B = 0.05, dt = 30 s, 20 h. Predicted before running by
`test_msis_delta_v.py`'s cycle model with each law's co-rotation-weighted orbit-averaged density:
**dawn-dusk +0.0750, noon-midnight +0.1516** (see `PREDICTED`; committed before the first sweep ran). Tolerance: that module's 1e-2 on
`(1 + error)` (one cycle's endpoint term) plus 3e-3 for what the circular-orbit prediction omits under
J2 - the 7.7 km peak-to-peak osculating altitude swing correlating with the table's latitude structure
(`<dh dln rho>/H` with `dh` 3.9 km, `H` 45 km and a latitude swing of `ln rho` <~0.1: <= 0.4 %) and
the same swing's convexity, common to both laws: `RATIO_TOL = 1.3e-2`. The node rate itself is checked
first: sun-synchronous to the mean-altitude offset (`dRAAN/dt` goes as `a^-7/2`; the one-day mean orbit
sits 5.0 km below the seed: +0.26 %) within 0.3 %.

The frozen season
-----------------
From MSIS directly: the global mean at 400 km moves by +1.4 % from the epoch's day to ten days later
(asserted in [+0.5, +2.5] %) - the error of freezing the table at the epoch over a 10-day run.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

pymsis = pytest.importorskip("pymsis")
scipy_interpolate = pytest.importorskip("scipy.interpolate")

from orbital_engine import drag, gravity, msis_bridge, msis_diurnal, scenarios  # noqa: E402
from orbital_engine import solar_ephemeris as se  # noqa: E402
from orbital_engine.atmosphere import (  # noqa: E402
    DENSITY_MODEL_LAYERED, DENSITY_MODEL_MSIS, DENSITY_MODEL_MSIS_DIURNAL,
)
from orbital_engine.custom_types import PropagatorType  # noqa: E402
from orbital_engine.database import Base  # noqa: E402
from orbital_engine.drag import DRAG_MODEL, DRAG_PARAM_NAMES, EARTH_OMEGA  # noqa: E402
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ, J2_MODEL  # noqa: E402
from orbital_engine.msis_bridge import (  # noqa: E402
    MSIS_ALTITUDE_GRID_KM, SOLAR_ACTIVITY_MODERATE, msis_coefficients,
)
from orbital_engine.msis_diurnal import (  # noqa: E402
    DIURNAL_LATITUDE_GRID_DEG, DIURNAL_LST_GRID_HOURS, MsisDiurnalTable, msis_diurnal_coefficients,
    msis_diurnal_table,
)
from orbital_engine.simulator import Simulation  # noqa: E402
from orbital_engine.stationkeeping import StationKeepingSpec  # noqa: E402
from orbital_engine.sweep import ForceModelSpec, ModelConfig, SweepResult, run_sweep  # noqa: E402

ArrF = NDArray[np.float64]
MU = scenarios.MU_EARTH
MOD = (140.0, 140.0, 15.0)
EPOCH = se.epoch_days("2024-03-20T00:00")
DATE = np.datetime64("2024-03-20")
UT_HOURS = (0.0, 4.0, 8.0, 12.0, 16.0, 20.0)

NODE_RTOL = 1e-12
FLOOR = 2e-6
END_TO_END_TOL = 2e-5
SHARP_TOL = 2e-5
PROFILE_TOL = 3e-3
DECAY_REL_TOL = 2e-3
RATIO_TOL = 1.3e-2


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


@pytest.fixture(scope="module")
def table() -> MsisDiurnalTable:
    return msis_diurnal_table(*MOD, EPOCH)


# --------------------------------------------------------------------------------------------------
# Independent direct evaluation: pymsis called here, fly-through mode, this module's own UT loop.
# --------------------------------------------------------------------------------------------------

def direct_zonal_mean(h: ArrF, lat: ArrF, lst: ArrF, date: np.datetime64 = DATE,
                      activity: Tuple[float, float, float] = MOD) -> ArrF:
    """Zonal mean at fixed local time for matching 1-D arrays of points."""
    n = h.size
    total = np.zeros(n)
    for ut in UT_HOURS:
        when = np.full(n, np.datetime64(date, "s") + np.timedelta64(int(ut * 3600), "s"))
        lon = np.mod(15.0 * (lst - ut), 360.0)
        out = np.asarray(pymsis.calculate(when, lon, lat, h, np.full(n, activity[0]),
                                          np.full(n, activity[1]), np.full((n, 7), activity[2]),
                                          version=2.0))
        if n == 1:
            out = out.reshape(1, 11)
        assert out.shape == (n, 11), out.shape
        total += out[:, 0].astype(np.float64)
    mean: ArrF = total / len(UT_HOURS)
    return mean


def gmst_deg(days: float) -> float:
    t = days / 36525.0
    return (280.46061837 + 360.98564736629 * days + 0.000387933 * t * t - t ** 3 / 38710000.0) % 360.0


def oracle_position(h: ArrF, lat_deg: ArrF, lst_h: ArrF, days: float) -> ArrF:
    ut_h = ((days + 0.5) % 1.0) * 24.0
    alpha = np.radians(gmst_deg(days) + 15.0 * (lst_h - ut_h))
    lat = np.radians(lat_deg)
    r = EARTH_R_EQ + h
    out: ArrF = np.stack([r * np.cos(lat) * np.cos(alpha), r * np.cos(lat) * np.sin(alpha),
                          r * np.sin(lat)], axis=1)
    return out


# ==================================================================================================
# a. The wrap is faithful
# ==================================================================================================

def test_table_nodes_are_the_direct_zonal_mean(table: MsisDiurnalTable) -> None:
    h, lat, lst = np.meshgrid(MSIS_ALTITUDE_GRID_KM[::60], DIURNAL_LATITUDE_GRID_DEG[::4],
                              DIURNAL_LST_GRID_HOURS[::5], indexing="ij")
    direct = direct_zonal_mean(h.ravel(), lat.ravel(), lst.ravel()).reshape(h.shape)
    engine = np.exp(table.ln_density[::60, ::4, ::5])
    assert np.allclose(engine, direct, rtol=NODE_RTOL, atol=0.0)
    assert table.date == "2024-03-20"


def test_off_node_interpolation_is_the_derived_leading_term(table: MsisDiurnalTable) -> None:
    rng = np.random.default_rng(2024)
    n = 40
    h = rng.uniform(150.0, 950.0, n)
    lat = rng.uniform(-87.0, 87.0, n)
    lst = rng.uniform(0.0, 24.0, n)
    alt = table.altitude_km
    k = np.searchsorted(alt, h, side="right") - 1
    h0, h1 = alt[k], alt[k + 1]
    lat0 = np.floor((lat + 90.0) / 5.0) * 5.0 - 90.0
    s0 = np.floor(lst / 0.5) * 0.5
    axes = [(h, h0, h0 + (h1 - h0)), (lat, lat0, lat0 + 5.0), (lst, s0, s0 + 0.5)]
    # f at the point, and at the three points of each axis's cell (others held at the point).
    columns = [h.copy(), lat.copy(), lst.copy()]
    probes: List[Tuple[ArrF, ArrF, ArrF]] = [(h, lat, lst)]
    for i, (_, lo, hi) in enumerate(axes):
        for x in (lo, 0.5 * (lo + hi), hi):
            c = [col.copy() for col in columns]
            c[i] = x
            probes.append((c[0], c[1], c[2]))
    stacked = [np.concatenate([p[j] for p in probes]) for j in range(3)]
    ln_f = np.log(direct_zonal_mean(*stacked)).reshape(len(probes), n)
    direct = ln_f[0]
    terms = np.zeros((3, n))
    for i, (x, lo, hi) in enumerate(axes):
        f0, fm, f1 = ln_f[1 + 3 * i], ln_f[2 + 3 * i], ln_f[3 + 3 * i]
        curvature = 4.0 * (f0 - 2.0 * fm + f1) / (hi - lo) ** 2
        terms[i] = 0.5 * curvature * (x - lo) * (hi - x)
    error = np.log(table.density_at(h, lat, lst)) - direct
    predicted = terms.sum(axis=0)
    scale = np.abs(terms).sum(axis=0)
    residual = np.abs(error - predicted)
    assert np.all(residual <= 0.4 * scale + FLOOR), np.max(residual / (scale + FLOOR))
    assert np.all(np.abs(error) <= 1.4 * scale + FLOOR)
    assert np.max(np.abs(error)) > 1e-4, "the test must see interpolation error at all"


@pytest.mark.parametrize("t_days", [0.0, 5.25])
def test_end_to_end_the_kernel_geometry_lands_on_msis(table: MsisDiurnalTable, t_days: float) -> None:
    """Oracle-placed satellites at nodes, through `msis_diurnal_density` exactly as `drag_kernel` calls
    it, against the direct zonal mean at that node."""
    rng = np.random.default_rng(int(t_days * 100) + 1)
    n = 60
    h = MSIS_ALTITUDE_GRID_KM[rng.integers(250, 600, n)]
    lat = DIURNAL_LATITUDE_GRID_DEG[rng.integers(1, 36, n)]
    lst = DIURNAL_LST_GRID_HOURS[rng.integers(0, 48, n)]
    r = oracle_position(h, lat, lst, EPOCH + t_days)
    coeff = np.tile([*MOD, EPOCH], (n, 1))
    engine = msis_diurnal.msis_diurnal_density(h, r, t_days * 86400.0, coeff)
    direct = direct_zonal_mean(h, lat, lst)
    assert np.max(np.abs(engine / direct - 1.0)) < END_TO_END_TOL


def test_the_table_is_a_pure_function_of_its_key(table: MsisDiurnalTable) -> None:
    """Epochs on one UT day share the table object; a cached lookup never evaluates."""
    assert msis_diurnal_table(140, 140, 15, EPOCH + 0.49) is table
    assert msis_diurnal.cached_msis_diurnal_table(*MOD, EPOCH) is table
    assert not table.ln_density.flags.writeable
    with pytest.raises(LookupError):
        msis_diurnal.cached_msis_diurnal_table(*MOD, EPOCH + 0.51)   # the next UT day: never built
    assert msis_diurnal.table_date(EPOCH + 0.49) == np.datetime64("2024-03-20")
    assert msis_diurnal.table_date(EPOCH + 1.0 - 1e-9) == np.datetime64("2024-03-20")
    assert msis_diurnal.table_date(EPOCH + 1.0) == np.datetime64("2024-03-21")


# ==================================================================================================
# b. Physics signatures of the bulge
# ==================================================================================================

def _extremes(rho: ArrF) -> Tuple[float, float, float, float]:
    """(peak lat, peak lst, min lat, min lst) of a (lat, lst) field on the table's nodes."""
    j, i = np.unravel_index(int(np.argmax(rho)), rho.shape)
    jm, im = np.unravel_index(int(np.argmin(rho)), rho.shape)
    return (float(DIURNAL_LATITUDE_GRID_DEG[j]), float(DIURNAL_LST_GRID_HOURS[i]),
            float(DIURNAL_LATITUDE_GRID_DEG[jm]), float(DIURNAL_LST_GRID_HOURS[im]))


def test_equinox_bulge_peaks_early_afternoon_at_the_equator(table: MsisDiurnalTable) -> None:
    k = int(np.searchsorted(MSIS_ALTITUDE_GRID_KM, 400.0))
    rho = np.exp(table.ln_density[k])
    peak_lat, peak_lst, min_lat, min_lst = _extremes(rho)
    assert 13.5 <= peak_lst <= 15.5 and abs(peak_lat) <= 10.0, (peak_lat, peak_lst)
    assert 2.5 <= min_lst <= 6.0, (min_lat, min_lst)
    equator = rho[int(np.argmin(np.abs(DIURNAL_LATITUDE_GRID_DEG)))]
    assert abs(float(equator.max() / equator.min()) / 2.30 - 1.0) < 0.06


@pytest.mark.parametrize("date, sign", [("2024-06-20", 1.0), ("2024-12-21", -1.0)])
def test_solstice_bulge_moves_into_the_summer_hemisphere(date: str, sign: float) -> None:
    rho = msis_diurnal.msis_zonal_mean_density(np.array([400.0]), DIURNAL_LATITUDE_GRID_DEG,
                                               DIURNAL_LST_GRID_HOURS, *MOD, np.datetime64(date))[0]
    peak_lat, peak_lst, min_lat, _ = _extremes(rho)
    assert 5.0 <= sign * peak_lat <= 30.0, peak_lat
    assert 13.0 <= peak_lst <= 16.0, peak_lst
    assert sign * min_lat < -5.0, min_lat
    dec = math.degrees(float(se.solar_position(se.epoch_days(date + "T12:00")).declination_rad))
    assert math.copysign(1.0, dec) == sign


def test_the_frozen_season_drifts_as_stated() -> None:
    lat, lst = DIURNAL_LATITUDE_GRID_DEG, DIURNAL_LST_GRID_HOURS
    w = np.cos(np.radians(lat)) * _simpson_weights(lat.size)
    w /= w.sum()

    def global_mean(date: np.datetime64) -> float:
        z = msis_diurnal.msis_zonal_mean_density(np.array([400.0]), lat, lst, *MOD, date)[0]
        return float(w @ z.mean(axis=1))

    drift = global_mean(DATE + np.timedelta64(10, "D")) / global_mean(DATE) - 1.0
    assert 0.005 < drift < 0.025, drift


# ==================================================================================================
# c. Consistency with the averaged law
# ==================================================================================================

def _simpson_weights(n: int) -> ArrF:
    s = np.ones(n)
    s[1:-1:2] = 4.0
    s[2:-1:2] = 2.0
    return s


def _diurnal_global_mean(alts: ArrF) -> ArrF:
    dates, _, _, _ = msis_bridge.quadrature_nodes()
    lat = DIURNAL_LATITUDE_GRID_DEG
    w = np.cos(np.radians(lat)) * _simpson_weights(lat.size)
    w /= w.sum()
    total = np.zeros(alts.size)
    for d in dates:
        z = msis_diurnal.msis_zonal_mean_density(alts, lat, DIURNAL_LST_GRID_HOURS, *MOD,
                                                 np.datetime64(d, "D"))
        total += np.einsum("alt,l->a", z, w) / DIURNAL_LST_GRID_HOURS.size
    out: ArrF = total / dates.size
    return out


@pytest.fixture(scope="module")
def consistency() -> Dict[str, ArrF]:
    alts = np.array([150.0, 292.0, 400.0, 600.0, 900.0])
    diurnal = _diurnal_global_mean(alts)
    # The averaged law's own quadrature, with the diurnal law's 6 UT samples.
    dates, lons, lats, weights = msis_bridge.quadrature_nodes()
    total = np.zeros(alts.size)
    for ut in UT_HOURS:
        when = dates.astype("datetime64[s]") + np.timedelta64(int(ut * 3600), "s")
        raw = np.asarray(pymsis.calculate(when, lons, lats, alts, np.full(dates.size, MOD[0]),
                                          np.full(dates.size, MOD[1]), np.full((dates.size, 7), MOD[2]),
                                          version=2.0))[..., 0].astype(np.float64)
        total += np.einsum("dolh,l->h", raw, weights) / (dates.size * lons.size)
    ut_averaged = total / len(UT_HOURS)
    msis_bridge.msis_profile(*MOD)
    profile = msis_bridge.msis_density(alts, np.tile(MOD, (alts.size, 1)))
    return {"alts": alts, "diurnal": diurnal, "ut_averaged": ut_averaged, "profile": profile}


def test_diurnal_average_reproduces_the_averaged_quadrature(consistency: Dict[str, ArrF]) -> None:
    rel = np.abs(consistency["diurnal"] / consistency["ut_averaged"] - 1.0)
    assert np.all(rel < SHARP_TOL), rel


def test_diurnal_average_reproduces_the_averaged_law(consistency: Dict[str, ArrF]) -> None:
    rel = np.abs(consistency["diurnal"] / consistency["profile"] - 1.0)
    assert np.all(rel < PROFILE_TOL), rel


# ==================================================================================================
# Kernel: bit-identity with MSIS rows, and the boundary
# ==================================================================================================

def test_every_other_law_is_bit_identical_with_diurnal_rows_present(table: MsisDiurnalTable) -> None:
    n = 80
    rng = np.random.default_rng(8)
    state = np.zeros((n, 6))
    state[1:, 0] = EARTH_R_EQ + rng.uniform(150.0, 900.0, n - 1)
    state[1:, 2] = rng.uniform(-3000.0, 3000.0, n - 1)
    state[1:, 3:] = rng.normal(scale=4.0, size=(n - 1, 3))
    params = np.zeros((n, len(DRAG_PARAM_NAMES)))
    params[:, 0] = 0.03
    params[:, 4] = EARTH_R_EQ
    params[:, 5] = EARTH_OMEGA
    law = np.arange(n) % 4
    params[law == 0, 1:4] = [1e-12, 400.0, 60.0]
    params[law == 1, 6] = DENSITY_MODEL_LAYERED
    params[law == 2, 6:10] = [DENSITY_MODEL_MSIS, *MOD]
    params[law == 3, 6:11] = [DENSITY_MODEL_MSIS_DIURNAL, *MOD, EPOCH]
    msis_bridge.msis_profile(*MOD)
    everyone = np.arange(1, n, dtype=np.int64)
    others = everyone[law[1:] < 3]
    together, alone = np.zeros((n, 3)), np.zeros((n, 3))
    zeros, parents = np.zeros(n), np.zeros(n, dtype=np.int32)
    drag.drag_kernel(everyone, 777.0, state, zeros, parents, params, together)
    drag.drag_kernel(others, 777.0, state, zeros, parents, params[:, :10].copy(), alone)
    assert np.array_equal(together[others], alone[others])
    assert np.all(together[everyone[law[1:] == 3]] != 0.0)


def test_configuring_the_diurnal_law_never_touches_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key no other test uses, so its table is genuinely built here through `enable_force_model`,
    with every route to the network made to raise."""
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
    fresh = (97.5, 101.25, 6.5)
    epoch = EPOCH + 100.0
    assert (*fresh, epoch) not in msis_diurnal._BY_EPOCH
    sim = scenarios.earth_constellation(_session(), n_sats=1, n_planes=1)
    sat = [i for name, i in sim.name_to_index.items() if name.startswith("SAT-")]
    sim.enable_force_model(DRAG_MODEL, sat, ballistic_coeff=0.02, r_ref=EARTH_R_EQ,
                           **msis_diurnal_coefficients(dict(zip(("f107", "f107a", "ap"), fresh)), epoch))
    assert (*fresh, epoch) in msis_diurnal._BY_EPOCH


# ==================================================================================================
# d. Orbit-averaged density by orbit type
# ==================================================================================================

class Predictor:
    """The table through an independent interpolator, and the analytic circular orbit."""

    def __init__(self, table: MsisDiurnalTable) -> None:
        lst = np.append(table.lst_hours, 24.0)
        f = np.concatenate([table.ln_density, table.ln_density[:, :, :1]], axis=2)
        self.rgi = scipy_interpolate.RegularGridInterpolator(
            (table.altitude_km, table.latitude_deg, lst), f)
        profile = msis_bridge.msis_profile(*MOD)
        self.profile_h, self.profile_ln = profile.altitude_km, np.log(profile.density_kg_m3)

    def rho(self, law: str, r: ArrF, days: float | ArrF) -> ArrF:
        h = np.linalg.norm(r, axis=-1) - EARTH_R_EQ
        if law == "averaged":
            out: ArrF = np.exp(np.interp(h, self.profile_h, self.profile_ln))
            return out
        lat = np.degrees(np.arctan2(r[..., 2], np.hypot(r[..., 0], r[..., 1])))
        sun = 280.460 + 0.9856474 * np.asarray(days)            # the AA mean longitude, restated
        lst = np.mod(12.0 + (np.degrees(np.arctan2(r[..., 1], r[..., 0])) - sun) / 15.0, 24.0)
        out = np.exp(self.rgi(np.stack([h, lat, lst], axis=-1)))
        return out


def circular_state(a: float, inc_deg: float, raan_deg: float, u: ArrF) -> Tuple[ArrF, ArrF]:
    i, node = math.radians(inc_deg), math.radians(raan_deg)
    cu, su = np.cos(u), np.sin(u)
    p = np.stack([cu, su * math.cos(i), su * math.sin(i)], axis=-1)
    q = np.stack([-su, cu * math.cos(i), cu * math.sin(i)], axis=-1)
    rot = np.array([[math.cos(node), -math.sin(node), 0.0], [math.sin(node), math.cos(node), 0.0],
                    [0.0, 0.0, 1.0]])
    r: ArrF = a * p @ rot.T
    v: ArrF = math.sqrt(MU / a) * q @ rot.T
    return r, v


def corotation_weight(r: ArrF, v: ArrF) -> ArrF:
    v_rel = v - np.cross(np.array([0.0, 0.0, EARTH_OMEGA]), r)
    out: ArrF = np.linalg.norm(v_rel, axis=-1) * np.einsum("...i,...i->...", v_rel, v)
    return out


def orbit_average_ratio(pred: Predictor, h: float, inc: float, node_lt_h: float) -> float:
    sun = float(se.sun_mean_longitude_deg(EPOCH))
    u = (np.arange(720) + 0.5) * 2.0 * math.pi / 720
    r, v = circular_state(EARTH_R_EQ + h, inc, sun + 15.0 * (node_lt_h - 12.0), u)
    w = corotation_weight(r, v)
    return float(np.mean(pred.rho("diurnal", r, EPOCH) * w) / np.mean(pred.rho("averaged", r, EPOCH) * w))


def predicted_decay(pred: Predictor, law: str, a0: float, inc: float, raan_deg: float,
                    horizon_s: float, b: float, step_s: float = 10.0) -> float:
    """RK4 on (a, u) along the analytic circular track, `u0 = 0` at t = 0, the Sun moving."""
    def rates(t: float, y: ArrF) -> ArrF:
        a, u = float(y[0]), float(y[1])
        r, v = circular_state(a, inc, raan_deg, np.array([u]))
        rho = pred.rho(law, r, EPOCH + t / 86400.0)
        adot = -(a * a / MU) * b * 1e3 * float(rho[0] * corotation_weight(r, v)[0])
        return np.array([adot, math.sqrt(MU / a ** 3)])

    y = np.array([a0, 0.0])
    t = 0.0
    for _ in range(int(round(horizon_s / step_s))):
        k1 = rates(t, y)
        k2 = rates(t + step_s / 2, y + step_s / 2 * k1)
        k3 = rates(t + step_s / 2, y + step_s / 2 * k2)
        k4 = rates(t + step_s, y + step_s * k3)
        y = y + step_s / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += step_s
    return float(y[0] - a0)


@pytest.fixture(scope="module")
def predictor(table: MsisDiurnalTable) -> Predictor:
    return Predictor(table)


ORBIT_RATIOS = {  # (h, inclination, node local time) -> predicted diurnal / averaged
    "dawn-dusk 400": (400.0, None, 18.0, 1.1221), "noon-midnight 400": (400.0, None, 12.0, 1.2039),
    "51.6 node 00h 400": (400.0, 51.6, 0.0, 1.2125), "dawn-dusk 292": (292.0, None, 18.0, 1.0753),
    "noon-midnight 292": (292.0, None, 12.0, 1.1524), "51.6 node 00h 292": (292.0, 51.6, 0.0, 1.1492),
}


@pytest.mark.parametrize("name", list(ORBIT_RATIOS))
def test_orbit_averaged_ratios_are_the_stated_predictions(predictor: Predictor, name: str) -> None:
    h, inc, lt, stated = ORBIT_RATIOS[name]
    inc_deg = scenarios.sun_synchronous_inclination_deg(EARTH_R_EQ + h) if inc is None else inc
    assert orbit_average_ratio(predictor, h, inc_deg, lt) == pytest.approx(stated, abs=5e-4)


DECAY_H_KM = 400.0
DECAY_B = 0.05
DECAY_DT = 30.0
DECAY_HORIZON_S = 86400.0


def _decay_arena(kind: str) -> Tuple[Simulation, List[int], List[str], List[Tuple[float, float]]]:
    """(sim, sats, laws, [(inclination, raan)]) - J2 zeroed, drag per law."""
    sun = float(se.sun_mean_longitude_deg(EPOCH))
    if kind == "sso":
        sim = scenarios.sun_synchronous_satellites(_session(), epoch_days=EPOCH,
                                                   ltan_hours=(18.0, 12.0, 18.0), altitude_km=DECAY_H_KM)
        sats = [sim.name_to_index[scenarios.sso_satellite_name(k)] for k in range(3)]
        inc = scenarios.sun_synchronous_inclination_deg(EARTH_R_EQ + DECAY_H_KM)
        geometry = [(inc, sun + 90.0), (inc, sun), (inc, sun + 90.0)]
        laws = ["diurnal", "diurnal", "averaged"]
    else:
        raan = (sun - 180.0) % 360.0
        sim = scenarios.station_keeping_satellites(_session(), n_sats=2, altitude_km=DECAY_H_KM,
                                                   inclination_deg=51.6, raan_deg=raan)
        sats = [sim.name_to_index[f"SK-SAT-{k:02d}"] for k in range(2)]
        geometry = [(51.6, raan), (51.6, raan)]
        laws = ["diurnal", "averaged"]
    sim.record_history = False
    sim.enable_force_model(J2_MODEL, sats, j2=0.0, r_eq=EARTH_R_EQ)
    common = dict(ballistic_coeff=DECAY_B, r_ref=EARTH_R_EQ, omega=EARTH_OMEGA)
    for s, law in zip(sats, laws):
        coefficients = (msis_diurnal_coefficients(SOLAR_ACTIVITY_MODERATE, EPOCH) if law == "diurnal"
                        else msis_coefficients(SOLAR_ACTIVITY_MODERATE))
        sim.enable_force_model(DRAG_MODEL, [s], **common, **coefficients)
    return sim, sats, laws, geometry


def _semi_major_axis(y: ArrF) -> ArrF:
    r = np.linalg.norm(y[..., :3], axis=-1)
    v2 = np.einsum("...i,...i->...", y[..., 3:], y[..., 3:])
    a: ArrF = -MU / (2.0 * (0.5 * v2 - MU / r))
    return a


@pytest.fixture(scope="module")
def decay_runs(predictor: Predictor, table: MsisDiurnalTable) -> Dict[str, Tuple[ArrF, ArrF]]:
    """kind -> (measured Delta a per sat, predicted Delta a per sat)."""
    out: Dict[str, Tuple[ArrF, ArrF]] = {}
    for kind in ("sso", "inclined"):
        sim, sats, laws, geometry = _decay_arena(kind)
        earth = sim.name_to_index["Earth"]
        a0 = _semi_major_axis(sim.global_states[sats] - sim.global_states[earth])
        n_steps = int(round(DECAY_HORIZON_S / DECAY_DT))
        for _ in range(n_steps):
            sim.step(DECAY_DT)
        measured = _semi_major_axis(sim.global_states[sats] - sim.global_states[earth]) - a0
        predicted = np.empty(len(sats))
        for k, (law, (inc, raan)) in enumerate(zip(laws, geometry)):
            a_start = EARTH_R_EQ + DECAY_H_KM
            n_dt = math.sqrt(MU / a_start ** 3) * DECAY_DT
            kepler_drift = -a_start * n_dt ** 6 / 36.0 * n_steps
            predicted[k] = predicted_decay(predictor, law, a_start, inc, raan, DECAY_HORIZON_S,
                                           DECAY_B) + kepler_drift
        out[kind] = (measured, predicted)
    return out


@pytest.mark.parametrize("kind", ["sso", "inclined"])
def test_cowell_decay_confirms_the_orbit_averaged_prediction(
    decay_runs: Dict[str, Tuple[ArrF, ArrF]], kind: str,
) -> None:
    measured, predicted = decay_runs[kind]
    rel = np.abs(measured / predicted - 1.0)
    assert np.all(rel < DECAY_REL_TOL), (kind, measured, predicted, rel)
    # The ratio each diurnal satellite sees against its averaged twin (last column).
    ratios = measured[:-1] / measured[-1]
    assert np.all(np.abs(ratios / (predicted[:-1] / predicted[-1]) - 1.0) < DECAY_REL_TOL)


# ==================================================================================================
# e. The headline, in Delta-v
# ==================================================================================================

SSO_SEED_KM = 297.0
SK_B = 0.05
SK_DT = 30.0
SK_HORIZON_S = 20.0 * 3600.0
LOWER_KM, UPPER_KM = 291.0, 293.5
CENTRE_KM = 0.5 * (LOWER_KM + UPPER_KM)
VARIANCE_KM2 = 15.6
SPEC = StationKeepingSpec(LOWER_KM, UPPER_KM)
LTANS = (18.0, 12.0)
PREDICTED = {"dawn-dusk": 0.0750, "noon-midnight": 0.1516}


def build_sso() -> Simulation:
    return scenarios.sun_synchronous_satellites(_session(), epoch_days=EPOCH, ltan_hours=LTANS,
                                                altitude_km=SSO_SEED_KM)


def sweep_configs() -> List[ModelConfig]:
    def spec(coefficients: Dict[str, float]) -> ForceModelSpec:
        return ForceModelSpec(DRAG_MODEL, {"ballistic_coeff": SK_B, "r_ref": EARTH_R_EQ,
                                           "omega": EARTH_OMEGA, **coefficients})
    return [
        ModelConfig("msis averaged", PropagatorType.COWELL, SK_DT,
                    force_models=(spec(msis_coefficients(SOLAR_ACTIVITY_MODERATE)),)),
        ModelConfig("msis diurnal", PropagatorType.COWELL, SK_DT,
                    force_models=(spec(msis_diurnal_coefficients(SOLAR_ACTIVITY_MODERATE, EPOCH)),)),
    ]


def _hdot(pred: Predictor, law: str, h: ArrF, node_lt_h: float) -> ArrF:
    """`test_msis_delta_v.py`'s orbit-averaged decay, the density taken along the SSO's own track."""
    sun = float(se.sun_mean_longitude_deg(EPOCH))
    u = (np.arange(360) + 0.5) * 2.0 * math.pi / 360
    out = np.empty(h.size)
    inc = scenarios.sun_synchronous_inclination_deg(EARTH_R_EQ + CENTRE_KM)
    for j, hj in enumerate(h):
        a = EARTH_R_EQ + hj
        r, v = circular_state(a, inc, sun + 15.0 * (node_lt_h - 12.0), u)
        w = corotation_weight(r, v)
        rho = pred.rho(law, r, EPOCH)
        k = int(np.searchsorted(pred.profile_h, hj, side="right") - 1)
        scale = (pred.profile_h[k + 1] - pred.profile_h[k]) / (pred.profile_ln[k] - pred.profile_ln[k + 1])
        kappa = 1.0 + VARIANCE_KM2 / (2.0 * scale ** 2)
        out[j] = -(a * a / MU) * SK_B * 1e3 * float(np.mean(rho * w)) * kappa
    return out


def t_cycle(pred: Predictor, law: str, node_lt_h: float) -> float:
    h = np.linspace(LOWER_KM, UPPER_KM, 201)
    tau = math.pi * math.sqrt((EARTH_R_EQ + CENTRE_KM) ** 3 / MU)
    delta = abs(float(_hdot(pred, law, np.array([CENTRE_KM]), node_lt_h)[0])) * tau
    top = h <= UPPER_KM - delta
    y = 1.0 / np.abs(_hdot(pred, law, h[top], node_lt_h))
    return tau + float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(h[top])))


def test_the_headline_prediction_is_what_the_docstring_states(predictor: Predictor) -> None:
    for name, lt in zip(("dawn-dusk", "noon-midnight"), LTANS):
        predicted = t_cycle(predictor, "averaged", lt) / t_cycle(predictor, "diurnal", lt) - 1.0
        assert predicted == pytest.approx(PREDICTED[name], abs=5e-4), (name, predicted)


def test_sso_planes_precess_with_the_mean_sun() -> None:
    sim = build_sso()
    sim.record_history = False
    sats = [sim.name_to_index[scenarios.sso_satellite_name(k)] for k in range(len(LTANS))]
    earth = sim.name_to_index["Earth"]
    times, nodes = [], []
    for n in range(2880):
        sim.step(30.0)
        rel = sim.global_states[sats] - sim.global_states[earth]
        h = np.cross(rel[:, :3], rel[:, 3:])
        times.append((n + 1) * 30.0)
        nodes.append(np.arctan2(h[:, 0], -h[:, 1]))           # RAAN from the angular momentum
    rate = np.polyfit(np.array(times), np.unwrap(np.array(nodes), axis=0), 1)[0]
    sun_rate = 2.0 * math.pi / (scenarios.TROPICAL_YEAR_DAYS * 86400.0)
    assert np.all(np.abs(rate / sun_rate - 1.0 - 0.0026) < 0.003), rate / sun_rate - 1.0


@pytest.fixture(scope="module")
def headline() -> List[SweepResult]:
    return run_sweep(build_sso, sweep_configs(), SK_HORIZON_S, timing_batches=1, timing_warmup=0,
                     oblateness={"Earth": (EARTH_J2, EARTH_R_EQ)},
                     station_keeping=SPEC, delta_v_baseline="msis averaged")


def test_headline_diurnal_budget_matches_the_prediction(headline: List[SweepResult]) -> None:
    base, diurnal = (r.delta_v for r in headline)
    assert base is not None and diurnal is not None
    for k, name in enumerate(("dawn-dusk", "noon-midnight")):
        assert base.bodies[k].n_raises >= 2 and diurnal.bodies[k].n_raises >= 2
        measured = diurnal.bodies[k].steady_rate_m_s_per_day / base.bodies[k].steady_rate_m_s_per_day - 1.0
        assert (1.0 + measured) / (1.0 + PREDICTED[name]) - 1.0 == pytest.approx(0.0, abs=RATIO_TOL), (
            name, measured, PREDICTED[name])
