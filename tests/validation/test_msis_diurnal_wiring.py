"""
`DENSITY_MODEL_MSIS_DIURNAL` wiring that must hold **without `pymsis`** - the refusals, the boundary,
the fused-plan exclusion, bit-identity of every other row, and the step-time geometry (latitude, local
solar time, time dependence). The MSIS physics is `test_msis_diurnal.py`.

**A synthetic table stands in for MSIS here.** `msis_diurnal`'s memo is keyed on
`(f107, f107a, ap, epoch_days)`; a table injected under a key no real configuration uses lets the whole
configuration and kernel path run with `pymsis` unimportable, and - more usefully - with a density
whose value at every node is known in closed form:

    ln rho = ln(1e-12) - (h - 400) / 60 + 0.004 phi_deg + 0.4 cos(2 pi (s - 14 h) / 24 h)

a bulge peaking at 14 h, latitude-asymmetric (so a latitude read from the wrong axis, or with the
wrong sign, cannot pass), altitude-dependent. At a table node trilinear interpolation is exact, so a
satellite placed - by an **independent** construction - at a node's altitude, latitude and local
time must read `exp(ln rho)` to rounding (`NODE_REL_TOL = 1e-12`; ~10 operations on numbers of order
30, plus the 1e-16-relative position reconstruction). The independent construction is the IAU 1982
GMST (written out in `test_solar_ephemeris.py`) and NRLMSIS's own definition of local time,
`s = UT + lon / 15`: `alpha = GMST(t) + 15 (s - UT(t))`. It agrees with the engine's
`12 h + (alpha - L) / 15` to 0.15 s of time, which moves `ln rho` by at most
`0.4 (2 pi / 24 h) 0.15 s` = 4.4e-6 - so the node test uses **4.4e-6** off the pure-rounding floor:
`GEOMETRY_REL_TOL = 1e-5`. A 12 h local-time error is a factor `e^0.8` = 2.2; a latitude sign error at
+45 deg is `e^0.36`; a wrong sign on the Sun's motion is 8 min of local time after one day (a factor
`1 + 0.4 * 2 pi * 8 / 1440` = 1.014), all far outside.
"""
from __future__ import annotations

import math
import sys
from typing import Iterator, List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import drag, gravity, msis_diurnal, scenarios, solar_ephemeris as se
from orbital_engine.atmosphere import (
    DENSITY_MODEL_EXPONENTIAL, DENSITY_MODEL_LAYERED, DENSITY_MODEL_MSIS, DENSITY_MODEL_MSIS_DIURNAL,
    DENSITY_MODELS,
)
from orbital_engine.custom_types import PropagatorType
from orbital_engine.database import Base
from orbital_engine.drag import DRAG_MODEL, DRAG_PARAM_NAMES, EARTH_OMEGA
from orbital_engine.geopotential import EARTH_R_EQ
from orbital_engine.msis_diurnal import (
    DIURNAL_LATITUDE_GRID_DEG, DIURNAL_LST_GRID_HOURS, MsisDiurnalTable,
)
from orbital_engine.msis_bridge import MSIS_ALTITUDE_GRID_KM
from orbital_engine.simulator import Simulation

ArrF = NDArray[np.float64]
EPOCH = se.epoch_days("2024-03-20T00:00")
SYNTH = {"f107": 111.125, "f107a": 109.875, "ap": 3.25}     # never a real configuration
NEVER_EVALUATED = {"f107": 171.25, "f107a": 163.5, "ap": 11.75, "epoch_days": 9000.25}
GEOMETRY_REL_TOL = 1e-5
NODE_REL_TOL = 1e-12
_COMMON = {"ballistic_coeff": 0.02, "r_ref": EARTH_R_EQ, "omega": EARTH_OMEGA}


def synthetic_ln_rho(h: ArrF, lat: ArrF, lst: ArrF) -> ArrF:
    out: ArrF = (math.log(1e-12) - (h - 400.0) / 60.0 + 0.004 * lat
                 + 0.4 * np.cos(2.0 * math.pi * (lst - 14.0) / 24.0))
    return out


@pytest.fixture
def synthetic(monkeypatch: pytest.MonkeyPatch) -> Iterator[MsisDiurnalTable]:
    """The synthetic table, injected into the memo under `SYNTH` + `EPOCH` for one test."""
    h, lat, lst = np.meshgrid(MSIS_ALTITUDE_GRID_KM, DIURNAL_LATITUDE_GRID_DEG, DIURNAL_LST_GRID_HOURS,
                              indexing="ij")
    table = MsisDiurnalTable(
        f107=SYNTH["f107"], f107a=SYNTH["f107a"], ap=SYNTH["ap"], date="2024-03-20",
        altitude_km=MSIS_ALTITUDE_GRID_KM, latitude_deg=DIURNAL_LATITUDE_GRID_DEG,
        lst_hours=DIURNAL_LST_GRID_HOURS, ln_density=np.ascontiguousarray(synthetic_ln_rho(h, lat, lst)),
    )
    key = (SYNTH["f107"], SYNTH["f107a"], SYNTH["ap"], EPOCH)
    monkeypatch.setitem(msis_diurnal._BY_EPOCH, key, table)
    yield table


@pytest.fixture
def no_pymsis(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setitem(sys.modules, "pymsis", None)
    yield


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _sim() -> Tuple[Simulation, List[int]]:
    sim = scenarios.earth_constellation(_session(), n_sats=3, n_planes=3)
    sats = sorted(i for n, i in sim.name_to_index.items() if n.startswith("SAT-"))
    return sim, sats


def oracle_position(h: ArrF, lat_deg: ArrF, lst_h: ArrF, days: float) -> ArrF:
    """Independent: MSIS's `s = UT + lon/15` and IAU 1982 GMST give the inertial right ascension."""
    t = days / 36525.0
    gmst = (280.46061837 + 360.98564736629 * days + 0.000387933 * t * t - t ** 3 / 38710000.0) % 360.0
    ut_h = ((days + 0.5) % 1.0) * 24.0
    alpha = np.radians(gmst + 15.0 * (lst_h - ut_h))
    lat = np.radians(lat_deg)
    r = EARTH_R_EQ + h
    out: ArrF = np.stack([r * np.cos(lat) * np.cos(alpha), r * np.cos(lat) * np.sin(alpha),
                          r * np.sin(lat)], axis=1)
    return out


def _coefficients(n: int) -> ArrF:
    c = np.empty((n, 4))
    c[:] = [SYNTH["f107"], SYNTH["f107a"], SYNTH["ap"], EPOCH]
    return c


# ==================================================================================================
# Step-time geometry, exact at nodes
# ==================================================================================================

@pytest.mark.parametrize("t_days", [0.0, 0.37, 3.3, 9.9])
def test_density_at_oracle_nodes_is_the_tabulated_value(synthetic: MsisDiurnalTable, t_days: float,
                                                        no_pymsis: None) -> None:
    rng = np.random.default_rng(5)
    n = 64
    h = MSIS_ALTITUDE_GRID_KM[rng.integers(150, 600, n)]
    lat = DIURNAL_LATITUDE_GRID_DEG[rng.integers(0, 37, n)]
    lst = DIURNAL_LST_GRID_HOURS[rng.integers(0, 48, n)]
    r = oracle_position(h, lat, lst, EPOCH + t_days)
    rho = msis_diurnal.msis_diurnal_density(h, r, t_days * 86400.0, _coefficients(n))
    expected = np.exp(synthetic_ln_rho(h, lat, lst))
    assert np.max(np.abs(rho / expected - 1.0)) < GEOMETRY_REL_TOL


def test_an_inertially_fixed_point_moves_in_local_time_only_with_the_sun(
    synthetic: MsisDiurnalTable,
) -> None:
    """Local time is inertial geometry: a point fixed in inertial space keeps its local time except for
    the mean Sun's 0.9856 deg/day - after one day it reads 3.94 min *earlier*. Not 24 h, and not the
    Earth's rotation. Exact at the shifted node up to the linear-in-LST interpolation, which is evaluated
    here in closed form from the synthetic table's own cosine."""
    lst0 = np.array([10.0])
    r = oracle_position(np.array([400.0]), np.array([20.0]), lst0, EPOCH)
    one_day = msis_diurnal.msis_diurnal_density(np.array([400.0]), r, 86400.0, _coefficients(1))
    shift_h = 0.9856474 / 15.0
    s = (lst0[0] - shift_h) / 0.5
    i0 = math.floor(s)
    w = s - i0
    ln_expected = ((1.0 - w) * synthetic_ln_rho(np.array([400.0]), np.array([20.0]), np.array([i0 * 0.5]))
                   + w * synthetic_ln_rho(np.array([400.0]), np.array([20.0]), np.array([(i0 + 1) * 0.5])))
    assert float(one_day[0]) == pytest.approx(float(np.exp(ln_expected[0])), rel=GEOMETRY_REL_TOL)


def test_trilinear_interpolation_is_exact_on_a_multilinear_field(synthetic: MsisDiurnalTable) -> None:
    """Between nodes the law is linear in each of `h` and latitude (the synthetic field is linear in
    both), so off-node values in those two directions are exact to rounding."""
    rng = np.random.default_rng(9)
    h = rng.uniform(210.0, 990.0, 50)
    lat = rng.uniform(-90.0, 90.0, 50)
    lst = DIURNAL_LST_GRID_HOURS[rng.integers(0, 48, 50)]
    rho = synthetic.density_at(h, lat, lst)
    assert np.max(np.abs(np.log(rho) - synthetic_ln_rho(h, lat, lst))) < 1e-12
    # Latitude is clamped to the table, never extrapolated: the poles are nodes.
    assert np.all(np.isfinite(synthetic.density_at(np.full(2, 400.0), np.array([-90.0, 90.0]),
                                                   np.array([23.99, 0.0]))))


def test_local_time_wraps_between_2330_and_0000(synthetic: MsisDiurnalTable) -> None:
    lst = np.array([23.75])
    rho = synthetic.density_at(np.array([400.0]), np.array([0.0]), lst)
    ln_expected = 0.5 * (synthetic_ln_rho(np.array([400.0]), np.array([0.0]), np.array([23.5]))
                         + synthetic_ln_rho(np.array([400.0]), np.array([0.0]), np.array([0.0])))
    assert float(np.log(rho[0])) == pytest.approx(float(ln_expected[0]), abs=1e-12)


# ==================================================================================================
# The kernel: bit-identity of every other law, the closed form, and the boundary
# ==================================================================================================

def _mixed_arena(n: int, width: int) -> Tuple[ArrF, NDArray[np.int32], ArrF, NDArray[np.int64]]:
    rng = np.random.default_rng(42)
    state = np.zeros((n, 6))
    radius = EARTH_R_EQ + rng.uniform(150.0, 900.0, n)
    d = rng.normal(size=(n, 3))
    state[:, :3] = radius[:, None] * d / np.linalg.norm(d, axis=1)[:, None]
    state[:, 3:] = rng.normal(scale=4.0, size=(n, 3))
    state[0] = 0.0
    params = np.zeros((n, width))
    params[:, 0] = 0.03
    params[:, 4] = EARTH_R_EQ
    params[:, 5] = EARTH_OMEGA
    law = np.arange(n) % 3
    params[law == 0, 1:4] = [1e-12, 400.0, 60.0]
    params[law == 1, 6] = DENSITY_MODEL_LAYERED
    if width > 10:
        params[law == 2, 6:11] = [DENSITY_MODEL_MSIS_DIURNAL, SYNTH["f107"], SYNTH["f107a"],
                                  SYNTH["ap"], EPOCH]
    return state, np.zeros(n, dtype=np.int32), params, law


def test_legacy_rows_are_bit_identical_with_diurnal_rows_present(synthetic: MsisDiurnalTable) -> None:
    """Rows on laws 0/1 give bitwise the same acceleration whether diurnal rows share the call or not,
    and whether the parameter array is the current 11 columns or the 10 before `epoch_days` (the pre-
    diurnal kernel's layout). `test_msis_wiring.py` pins the same rows to the frozen pre-MSIS kernel;
    `test_msis_diurnal.py` does it with MSIS rows. A diurnal row reads its time: the result moves
    with `t`, which a steady law's never does."""
    n = 90
    state, parents, params, law = _mixed_arena(n, len(DRAG_PARAM_NAMES))
    everyone = np.arange(n, dtype=np.int64)
    legacy = everyone[law < 2]
    together, alone, narrow = (np.zeros((n, 3)) for _ in range(3))
    with np.errstate(all="raise"):
        drag.drag_kernel(everyone, 1234.5, state, np.zeros(n), parents, params, together)
        drag.drag_kernel(legacy, 1234.5, state, np.zeros(n), parents, params, alone)
        drag.drag_kernel(legacy, 1234.5, state, np.zeros(n), parents, params[:, :10].copy(), narrow)
    assert np.array_equal(together[legacy], alone[legacy])
    assert np.array_equal(narrow[legacy], alone[legacy])
    diurnal_rows = everyone[(law == 2) & (np.arange(n) > 0)]
    assert np.all(together[diurnal_rows] != 0.0)
    later = np.zeros((n, 3))
    drag.drag_kernel(everyone, 1234.5 + 6 * 3600.0, state, np.zeros(n), parents, params, later)
    assert np.array_equal(later[legacy], together[legacy])
    assert not np.array_equal(later[diurnal_rows], together[diurnal_rows])


def test_closed_form_acceleration_on_the_diurnal_law(synthetic: MsisDiurnalTable) -> None:
    """`-(1/2) rho B |v_rel| v_rel * 1e3` with `rho` at an oracle node; co-rotation off. 1e-5 from
    the geometry tolerance above."""
    h, lat, lst = np.array([420.0]), np.array([35.0]), np.array([15.5])
    r = oracle_position(h, lat, lst, EPOCH)[0]
    v = np.array([-5.0, 4.0, 3.0])
    rho = float(np.exp(synthetic_ln_rho(h, lat, lst))[0])
    expected = -0.5 * rho * 0.02 * math.sqrt(50.0) * v * 1e3
    state = np.zeros((2, 6))
    state[1] = [*r, *v]
    params = np.zeros((2, len(DRAG_PARAM_NAMES)))
    params[1, :11] = [0.02, 0.0, 0.0, 0.0, EARTH_R_EQ, 0.0, DENSITY_MODEL_MSIS_DIURNAL,
                      SYNTH["f107"], SYNTH["f107a"], SYNTH["ap"], EPOCH]
    out = np.zeros((2, 3))
    drag.drag_kernel(np.array([1], dtype=np.int64), 0.0, state, np.zeros(2),
                     np.zeros(2, dtype=np.int32), params, out)
    assert np.max(np.abs(out[1] / expected - 1.0)) < GEOMETRY_REL_TOL


def test_the_kernel_raises_on_an_unevaluated_table_and_never_imports_pymsis(no_pymsis: None) -> None:
    state = np.zeros((2, 6))
    state[1, :3] = [EARTH_R_EQ + 400.0, 0.0, 0.0]
    state[1, 3:] = [0.0, 7.6, 0.0]
    params = np.zeros((2, len(DRAG_PARAM_NAMES)))
    params[1, :11] = [0.02, 0.0, 0.0, 0.0, EARTH_R_EQ, 0.0, DENSITY_MODEL_MSIS_DIURNAL,
                      *NEVER_EVALUATED.values()]
    with pytest.raises(LookupError, match="configuration time"):
        drag.drag_kernel(np.array([1], dtype=np.int64), 0.0, state, np.zeros(2),
                         np.zeros(2, dtype=np.int32), params, np.zeros((2, 3)))


# ==================================================================================================
# Refusals - before any bit or coefficient is written
# ==================================================================================================

def test_the_selector_values() -> None:
    assert DENSITY_MODEL_MSIS_DIURNAL == 3.0 and DENSITY_MODEL_MSIS == 2.0
    assert DENSITY_MODELS == (DENSITY_MODEL_EXPONENTIAL, DENSITY_MODEL_LAYERED, DENSITY_MODEL_MSIS,
                              DENSITY_MODEL_MSIS_DIURNAL)
    assert DRAG_PARAM_NAMES[-1] == "epoch_days" and len(DRAG_PARAM_NAMES) == 11


@pytest.mark.parametrize("drop", ["f107", "f107a", "ap", "epoch_days"])
def test_diurnal_refuses_to_be_selected_without_all_four(drop: str) -> None:
    sim, sats = _sim()
    given = {k: v for k, v in NEVER_EVALUATED.items() if k != drop}
    with pytest.raises(ValueError, match="needs f107, f107a, ap and epoch_days"):
        sim.enable_force_model(DRAG_MODEL, sats, density_model=DENSITY_MODEL_MSIS_DIURNAL, **given,
                               **_COMMON)
    assert DRAG_MODEL not in sim.force_model_params


@pytest.mark.parametrize("law", [DENSITY_MODEL_EXPONENTIAL, DENSITY_MODEL_LAYERED, None])
def test_an_epoch_on_a_non_diurnal_row_is_refused(law: float | None) -> None:
    sim, sats = _sim()
    selector = {} if law is None else {"density_model": law}
    with pytest.raises(ValueError, match="epoch_days is read only under"):
        sim.enable_force_model(DRAG_MODEL, sats, epoch_days=EPOCH, **selector, **_COMMON)


@pytest.mark.parametrize("epoch", [float("nan"), float("inf"), -20000.0, 18300.0])
def test_bad_epochs_are_refused_before_pymsis(epoch: float, no_pymsis: None) -> None:
    sim, sats = _sim()
    mask = sim.force_model_mask.copy()
    with pytest.raises(ValueError, match="epoch_days"):
        sim.enable_force_model(DRAG_MODEL, sats, density_model=DENSITY_MODEL_MSIS_DIURNAL,
                               **{**NEVER_EVALUATED, "epoch_days": epoch}, **_COMMON)
    assert np.array_equal(sim.force_model_mask, mask)


def test_selecting_diurnal_without_pymsis_fails_at_configuration(no_pymsis: None) -> None:
    sim, sats = _sim()
    mask = sim.force_model_mask.copy()
    with pytest.raises(ImportError, match=r"orbital_engine\[msis\]"):
        sim.enable_force_model(DRAG_MODEL, sats, density_model=DENSITY_MODEL_MSIS_DIURNAL,
                               **NEVER_EVALUATED, **_COMMON)
    assert np.array_equal(sim.force_model_mask, mask)
    assert DRAG_MODEL not in sim.force_model_params


# ==================================================================================================
# The fused compiled plan
# ==================================================================================================

def test_a_diurnal_row_is_foreign_to_the_fused_cowell_plan(synthetic: MsisDiurnalTable) -> None:
    """The fused twin decodes 3.0 as the averaged MSIS law and has no `t`, so a Cowell body on the
    diurnal law must send the Cowell set down the NumPy path - and switching it back to a fused law
    must re-qualify the plan."""
    sim, sats = _sim()
    sim.record_history = False
    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model(gravity.POINT_MASS_MODEL, sats)
    sim.enable_force_model(DRAG_MODEL, sats, density_model=DENSITY_MODEL_LAYERED, **_COMMON)
    assert sim._cowell_fused_ok
    sim.enable_force_model(DRAG_MODEL, sats[:1], density_model=DENSITY_MODEL_MSIS_DIURNAL,
                           **SYNTH, epoch_days=EPOCH, **_COMMON)
    assert not sim._cowell_fused_ok
    assert np.all(sim._cowell_drag_table_of[sats] == -1), "a diurnal row must never get a profile row"
    before = sim.global_states[sats].copy()
    sim.step(30.0)
    assert np.all(np.isfinite(sim.global_states[sats])) and not np.array_equal(
        sim.global_states[sats], before)
    sim.enable_force_model(DRAG_MODEL, sats[:1], density_model=DENSITY_MODEL_LAYERED, **_COMMON)
    assert sim._cowell_fused_ok
