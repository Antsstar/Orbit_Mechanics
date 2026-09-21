"""
Validation of `sgp4_bridge.py`: TLE -> SGP4 -> Cartesian, and SGP4 as a sweep tier.

**1. Published vectors (the headline).** Vallado, Crawford, Hujsak & Kelso, *Revisiting Spacetrack
Report #3*, AIAA 2006-6753, publishes 33 verification element sets (`SGP4-VER.TLE`) and the C++
reference output for them (`tcppver.out`). Both ship **inside the installed `sgp4` package** (next to
`sgp4/__init__.py`) and are read from there with `pkgutil.get_data` - nothing is transcribed here.
Every printed row is reproduced *through this repository's bridge* (`sgp4_tsince`, which runs the
same `SatrecArray` path the sweep does):

- `tcppver.out` prints position to 1e-8 km and velocity to 1e-9 km/s. If the bridge reproduces the
  reference exactly, what is left is that print rounding: **|dr| <= 5e-9 km, |dv| <= 5e-10 km/s**,
  and - because rounding error is uniform - a **mean |dr| of 2.5e-9 km** and mean |dv| 2.5e-10 km/s.
  Measured over 1788 position components in 31 cases: max 5.002e-9 km, mean 2.515e-9 km; velocity
  max 4.997e-10, mean 2.491e-10 km/s. The mean is asserted to +/-6 standard errors (3.4e-11 km each),
  so a systematic disagreement of ~1e-10 km would fail here even though it hides inside the max.
- **Named exception:** the last case (SL-12 R/B, 20413) is evaluated 1.84e6 minutes (3.5 yr) past
  epoch through perigee on a deep-space resonant orbit, and disagrees by up to **1.17e-7 km**. This is
  not the bridge: the package's own `Satrec.sgp4_tsince` gives the same value to 6.4e-10 km. It is
  held to the package's own verification tolerance, 2e-7 km (`sgp4/tests.py`). The bridge used to
  put the whole time offset on the Julian-date fraction and measured 1.59e-7 there; carrying whole
  days separately (`_julian_offsets`) is what made it equal to the package's native path.
- **Error outcomes are published too.** The reference stops printing a satellite when SGP4 flags it.
  Seven cases do so (codes 1, 1, 6, 6, 4, 3, 6 - the list in `sgp4/tests.py`), and the bridge must
  report exactly that code at the next grid time.

**2. Interchange safety.** Seed the engine from SGP4's Cartesian state, integrate Cowell + point
mass + J2, and compare with SGP4. They are different models, so they must differ - by exactly the
difference of their local expansions at epoch. Derived before measuring, the two terms are:

- SGP4 is first order in J2, so its velocity is consistent with the derivative of its position only
  to O(J2^2): `|v_sgp4 - dr/dt| ~ J2^2 v = 1.2e-6 * 7.66 = 9e-6 km/s` (measured 1.06e-5; it is the
  same with B* = 0, so it is not drag). This makes the difference *linear* in t.
- Its implied acceleration differs from point mass + J2 by J3/J4 (`~J3 (R/r)^3 g ~ 3e-8 km/s^2`)
  plus O(J2^2 g) = 1e-8 (measured 4.5e-8 km/s^2).

Together `9e-6 * 300 + 0.5 * 4e-8 * 300^2 ~ 4.5e-3 km` after 5 minutes; measured 3.70e-3 km. The
test does better than the order of magnitude: it builds `(rdot - v) t + (1/2)(rddot - a_engine) t^2`
from raw `sgp4.api.Satrec` finite differences (no bridge code) and requires the engine to land on it
with a residual under 2 % at 60 s (measured 0.6 %) that grows as **t^3** (measured ratio 8.1 for
t = 120 vs 60 s). A units slip in the seed (m vs km, km/min vs km/s) or a seed taken at the wrong
epoch adds a term that is zeroth or first order in t and orders of magnitude larger.

**3. The anti-pattern, quantified.** Feeding the TLE's *mean* elements to `coe_to_rv` - which
`CLAUDE.md` forbids and this module never does - misses SGP4's own epoch state by the first-order
J2 short-period terms. Their scale is `J2 R^2 / a = 6.48 km` for the ISS, times O(1) coefficients.
Measured: **7.64 km** in position (R +6.84, S -2.50, W +2.30 km) and 5.0 m/s in velocity, against
**9.1e-13 km** for the Cartesian seed. Asserted between 1 km and 3 J2 R^2 / a.

**4. SGP4 as a sweep tier.** `run_sweep(..., external=[sgp4_tier(...)])` scores SGP4 against the same
truth as the engine tiers. At 300 s its error must equal the directly computed SGP4-minus-truth
distance (same numbers, no scoring bug) and sit in the band derived in (2).
"""
from __future__ import annotations

import math
import os
import pkgutil
import subprocess
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np
import pytest
from sqlalchemy.orm import Session

pytest.importorskip("sgp4", reason="SGP4 validation requires the [test] or [sgp4] extra")
pytest.importorskip("scipy", reason="reference integration requires the [test] or [reference] extra")

from sgp4.api import Satrec  # noqa: E402  (after importorskip, deliberately)

from orbital_engine import scenarios, sgp4_bridge  # noqa: E402
from orbital_engine.access import AccessSpec, GroundStation  # noqa: E402
from orbital_engine.custom_types import PropagatorType  # noqa: E402
from orbital_engine.drag import EARTH_OMEGA  # noqa: E402
from orbital_engine.frames import ReferenceFrames  # noqa: E402
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ  # noqa: E402
from orbital_engine.reference import TRUTH_ATOL, TRUTH_RTOL, j2_field, reference_for  # noqa: E402
from orbital_engine.sweep import ForceModelSpec, ModelConfig, run_sweep  # noqa: E402
from orbital_engine.sgp4_bridge import TLE, sgp4_tsince  # noqa: E402

# --------------------------------------------------------------------------------------------------
# Published data, read from the installed package
# --------------------------------------------------------------------------------------------------

PRINT_ROUNDING_KM = 5e-9        # tcppver.out prints %16.8f km
PRINT_ROUNDING_KM_S = 5e-10     # and %12.9f km/s
LONG_CASE_TOL_KM = 2e-7         # sgp4's own tcppver tolerance, for the one named 3.5-yr case
# SGP4 error codes at which the reference output stops, in file order (sgp4/tests.py lists them).
PUBLISHED_ERROR_CODES = [1, 1, 6, 6, 4, 3, 6]


def _published_cases() -> List[Tuple[str, str, int, np.ndarray]]:
    """(line1, line2 incl. the start/stop/step columns, satnum, rows[tsince, r, v]) per case."""
    tle_data = pkgutil.get_data("sgp4", "SGP4-VER.TLE")
    out_data = pkgutil.get_data("sgp4", "tcppver.out")
    assert tle_data is not None and out_data is not None, "sgp4 no longer ships Vallado's vectors"
    lines = iter(tle_data.decode("ascii").splitlines())
    pairs = [(l1, next(lines)) for l1 in lines if l1.startswith("1")]

    blocks: List[Tuple[int, List[List[float]]]] = []
    for line in out_data.decode("ascii").replace("\r", "").splitlines():
        if line.endswith("xx"):
            blocks.append((int(line.split()[0]), []))
        elif line.strip():
            blocks[-1][1].append([float(x) for x in line.split()[:7]])

    assert len(pairs) == len(blocks) == 33
    return [(l1, l2, satnum, np.array(rows)) for (l1, l2), (satnum, rows) in zip(pairs, blocks)]


def _tle(l1: str, l2: str, satnum: int) -> TLE:
    return TLE(str(satnum), l1[:69], l2[:69])


def _is_long_case(rows: np.ndarray) -> bool:
    return bool(rows[0, 0] == 0.0 and rows.shape[0] > 1 and rows[1, 0] > 1.0e6)


def test_published_vectors_reproduce_to_print_rounding() -> None:
    dr_all: List[np.ndarray] = []
    dv_all: List[np.ndarray] = []
    n_cases = 0
    for l1, l2, satnum, rows in _published_cases():
        if satnum == 33334:
            continue    # fails at epoch; its only printed row is testcpp's repeat of the previous case
        out = sgp4_tsince(_tle(l1, l2, satnum), rows[:, 0])
        assert np.all(out.errors[:, 0] == 0), f"{satnum}: SGP4 error on a row the reference printed"
        dr = out.states[:, 0, :3] - rows[:, 1:4]
        dv = out.states[:, 0, 3:] - rows[:, 4:7]
        if _is_long_case(rows):
            assert np.abs(dr).max() < LONG_CASE_TOL_KM, f"{satnum}: {np.abs(dr).max():.3e} km"
            assert np.abs(dv).max() <= 2 * PRINT_ROUNDING_KM_S
            continue
        n_cases += 1
        dr_all.append(dr.ravel())
        dv_all.append(dv.ravel())

    dr = np.abs(np.concatenate(dr_all))
    dv = np.abs(np.concatenate(dv_all))
    assert n_cases == 31
    assert dr.size == 1788
    # Per component: print rounding, with 2x margin for platform floating-point differences.
    assert dr.max() <= 2 * PRINT_ROUNDING_KM, dr.max()
    assert dv.max() <= 2 * PRINT_ROUNDING_KM_S, dv.max()
    # Uniform rounding => mean |error| = half the bound; +/- 6 standard errors of that mean.
    se_r = PRINT_ROUNDING_KM / math.sqrt(3.0) / math.sqrt(dr.size)
    se_v = PRINT_ROUNDING_KM_S / math.sqrt(3.0) / math.sqrt(dv.size)
    assert abs(dr.mean() - PRINT_ROUNDING_KM / 2) < 6 * se_r, dr.mean()
    assert abs(dv.mean() - PRINT_ROUNDING_KM_S / 2) < 6 * se_v, dv.mean()


def test_long_case_residual_is_the_package_not_the_bridge() -> None:
    """The one case outside print rounding is reproduced identically by sgp4's own tsince path."""
    (l1, l2, satnum, rows), = [c for c in _published_cases() if _is_long_case(c[3])]
    out = sgp4_tsince(_tle(l1, l2, satnum), rows[:, 0])
    sat = Satrec.twoline2rv(l1[:69], l2[:69])
    native = np.array([sat.sgp4_tsince(t)[1] for t in rows[:, 0]])
    assert np.abs(out.states[:, 0, :3] - native).max() < 1e-8
    assert np.abs(out.states[:, 0, :3] - rows[:, 1:4]).max() > 5 * PRINT_ROUNDING_KM  # still named


def test_published_error_outcomes() -> None:
    codes: List[int] = []
    for l1, l2, satnum, rows in _published_cases():
        tle = _tle(l1, l2, satnum)
        start, stop, step = (float(x) for x in l2[69:].split())
        if satnum == 33334:
            codes.append(int(sgp4_tsince(tle, np.array([0.0])).errors[0, 0]))
            continue
        last = rows[-1, 0]
        if last < stop - 1e-9:
            out = sgp4_tsince(tle, np.array([last + step]))
            codes.append(int(out.errors[0, 0]))
            assert np.all(np.isnan(out.states[0, 0])), "a flagged state must not be silently filled"
    assert codes == PUBLISHED_ERROR_CODES


# --------------------------------------------------------------------------------------------------
# The embedded TLE and the Cartesian seed
# --------------------------------------------------------------------------------------------------

def _checksum(line: str) -> int:
    return sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68]) % 10


def test_embedded_iss_tle_checksums() -> None:
    for line in (sgp4_bridge.ISS_TLE.line1, sgp4_bridge.ISS_TLE.line2):
        assert len(line) == 69
        assert _checksum(line) == int(line[68])


def test_scenario_seeds_sgp4_cartesian_state(db_session: Session) -> None:
    """rv_to_coe then coe_to_rv of SGP4's own state: round-off only (measured 9.1e-13 km)."""
    tle = sgp4_bridge.ISS_TLE
    r0, v0 = sgp4_bridge.teme_seed([tle], sgp4_bridge.tle_epoch(tle))
    sat = Satrec.twoline2rv(tle.line1, tle.line2)
    _, r_raw, v_raw = sat.sgp4_tsince(0.0)
    assert np.abs(r0[0] - r_raw).max() == 0.0 and np.abs(v0[0] - v_raw).max() == 0.0

    sim = scenarios.tle_satellites(db_session)
    k, e = sim.name_to_index[tle.name], sim.name_to_index["Earth"]
    rel = sim.global_states[k] - sim.global_states[e]
    assert np.abs(rel[:3] - r0[0]).max() < 1e-9
    assert np.abs(rel[3:] - v0[0]).max() < 1e-12


# --------------------------------------------------------------------------------------------------
# Interchange safety: Cowell + J2 from SGP4's Cartesian state
# --------------------------------------------------------------------------------------------------

def _raw_sgp4_position(tle: TLE) -> Callable[[float], np.ndarray]:
    sat = Satrec.twoline2rv(tle.line1, tle.line2)

    def at(t_s: float) -> np.ndarray:
        err, r, _ = sat.sgp4_tsince(t_s / 60.0)
        assert err == 0
        return np.array(r)
    return at


def test_cowell_from_cartesian_seed_matches_local_expansion(db_session: Session) -> None:
    tle = sgp4_bridge.ISS_TLE
    pos = _raw_sgp4_position(tle)
    sat = Satrec.twoline2rv(tle.line1, tle.line2)
    v0 = np.array(sat.sgp4_tsince(0.0)[2])

    # Fourth-order central differences of SGP4's *position*, h = 1 s: truncation ~h^4 r^(5) ~ 1e-15.
    h = 1.0
    R: Dict[int, np.ndarray] = {m: pos(m * h) for m in (-2, -1, 0, 1, 2)}
    rdot = (-R[2] + 8 * R[1] - 8 * R[-1] + R[-2]) / (12 * h)
    rddot = (-R[2] + 16 * R[1] - 30 * R[0] + 16 * R[-1] - R[-2]) / (12 * h * h)
    r0 = R[0]
    a_engine = (-scenarios.MU_EARTH * r0 / np.linalg.norm(r0) ** 3
                + scenarios.MU_EARTH * j2_field(r0[None, :], EARTH_J2, EARTH_R_EQ)[0])
    dv0, da0 = rdot - v0, rddot - a_engine
    # The two derived magnitudes from the module docstring.
    assert 3e-6 < np.linalg.norm(dv0) < 3e-5, np.linalg.norm(dv0)
    assert 1e-8 < np.linalg.norm(da0) < 1e-7, np.linalg.norm(da0)

    sim = scenarios.tle_satellites(db_session)
    k, e = sim.name_to_index[tle.name], sim.name_to_index["Earth"]
    sim.set_propagator([k], PropagatorType.COWELL)
    sim.enable_force_model("point_mass_gravity", [k])
    sim.enable_force_model("j2", [k], j2=EARTH_J2, r_eq=EARTH_R_EQ)

    residual: Dict[int, float] = {}
    dt = 10.0   # RK4 at 10 s: ~5e-9 km over 5 min, far below everything asserted here
    for n in range(1, 31):
        sim.step(dt)
        t = n * dt
        if n in (6, 12, 30):
            diff = pos(t) - (sim.global_states[k, :3] - sim.global_states[e, :3])
            predicted = dv0 * t + 0.5 * da0 * t * t
            residual[int(t)] = float(np.linalg.norm(diff - predicted))
            if t == 60:
                assert residual[60] < 0.02 * np.linalg.norm(predicted)
            if t == 300:
                # The derived order of magnitude, 4.5e-3 km; measured 3.70e-3 km.
                assert 1e-3 < np.linalg.norm(diff) < 1e-2, np.linalg.norm(diff)
    # The remainder is third order: doubling t multiplies it by 8.
    assert 6.0 < residual[120] / residual[60] < 10.0, residual


def test_mean_elements_into_coe_to_rv_is_wrong_by_kilometres() -> None:
    """
    DOCUMENTED ANTI-PATTERN - not supported API. This is what `CLAUDE.md` forbids, done once here so
    the cost has a number: the TLE's mean elements read as osculating elements.
    """
    tle = sgp4_bridge.ISS_TLE
    sat = Satrec.twoline2rv(tle.line1, tle.line2)
    _, r_sgp4, v_sgp4 = sat.sgp4_tsince(0.0)

    mu = scenarios.MU_EARTH
    n = sat.no_kozai / 60.0                      # rad/min -> rad/s
    a = (mu / n ** 2) ** (1.0 / 3.0)
    e, m = sat.ecco, sat.mo
    ecc_anom = m
    for _ in range(30):
        ecc_anom -= (ecc_anom - e * math.sin(ecc_anom) - m) / (1.0 - e * math.cos(ecc_anom))
    nu = 2.0 * math.atan2(math.sqrt(1 + e) * math.sin(ecc_anom / 2), math.sqrt(1 - e) * math.cos(ecc_anom / 2))
    coe = np.array([[a * (1 - e * e), e, sat.inclo, sat.nodeo, sat.argpo, nu]])
    r_bad, v_bad, ok = ReferenceFrames.coe_to_rv(coe, mu)
    assert ok.all()

    miss_km = float(np.linalg.norm(r_bad[0] - np.array(r_sgp4)))
    scale_km = EARTH_J2 * EARTH_R_EQ ** 2 / a     # 6.48 km: the first-order short-period scale
    assert 1.0 < miss_km < 3.0 * scale_km, miss_km          # measured 7.64 km
    assert float(np.linalg.norm(v_bad[0] - np.array(v_sgp4))) > 1e-3   # measured 5.0e-3 km/s


# --------------------------------------------------------------------------------------------------
# SGP4 as a sweep tier
# --------------------------------------------------------------------------------------------------

def test_sgp4_tier_is_scored_against_the_common_truth(db_session_factory: Callable[[], Session]) -> None:
    tle = sgp4_bridge.ISS_TLE
    epoch = sgp4_bridge.tle_epoch(tle)
    horizon = 300.0
    j2 = {"j2": EARTH_J2, "r_eq": EARTH_R_EQ}
    oblateness = {"Earth": (EARTH_J2, EARTH_R_EQ)}
    cowell = ModelConfig(
        "Cowell+J2", PropagatorType.COWELL, dt=10.0,
        force_models=(ForceModelSpec("point_mass_gravity"), ForceModelSpec("j2", j2)),
    )
    build = lambda: scenarios.tle_satellites(db_session_factory())  # noqa: E731
    results = run_sweep(
        build, [cowell], horizon, oblateness=oblateness, timing_batches=1, timing_warmup=0,
        external=[sgp4_bridge.sgp4_tier([tle], epoch, dt=10.0)],
    )
    assert [r.config_name for r in results] == ["Cowell+J2", "SGP4"]
    sgp4_result = results[1]
    assert sgp4_result.n_bodies == 1

    truth = reference_for(build(), np.array([0.0, horizon]), rtol=TRUTH_RTOL, atol=TRUTH_ATOL,
                          oblateness=oblateness)
    truth_rel = truth.position_of(tle.name)[-1] - truth.position_of("Earth")[-1]
    direct = float(np.linalg.norm(_raw_sgp4_position(tle)(horizon) - truth_rel))
    assert abs(sgp4_result.error.median_km - direct) < 1e-9
    assert 1e-3 < sgp4_result.error.median_km < 1e-2     # the interchange band; measured 3.70e-3
    # The engine tier is a verification against J2 truth, three orders tighter.
    assert results[0].error.median_km < 1e-5


def test_sgp4_tier_access_metrics_share_the_truth_grid(db_session_factory: Callable[[], Session]) -> None:
    """
    One orbit, one station under the ground track. SGP4 differs from J2 truth by ~0.1 km after an
    orbit (measured 0.100 km, almost all along-track), i.e. ~0.1 / 7.66 = 0.013 s of timing, so the
    windows must match one-for-one with sub-0.1 s shifts. A wrong central body, time origin or frame
    handling in the external access path would move them by seconds to minutes or lose the pass.
    """
    tle = sgp4_bridge.ISS_TLE
    epoch = sgp4_bridge.tle_epoch(tle)
    horizon = 5580.0
    r0, _ = sgp4_bridge.teme_seed([tle], epoch)
    theta0 = sgp4_bridge.greenwich_angle(tle)
    # Put the station directly under the satellite a quarter-orbit in, in body-fixed coordinates.
    r_q = sgp4_bridge.sgp4_states([tle], epoch, np.array([1200.0])).states[0, 0, :3]
    lon = math.atan2(r_q[1], r_q[0]) - (theta0 + EARTH_OMEGA * 1200.0)
    lat = math.asin(r_q[2] / np.linalg.norm(r_q))
    spec = AccessSpec(
        stations=[GroundStation("sub-track", lat, lon)], central_body="Earth", omega=EARTH_OMEGA,
        body_radius_km=scenarios.EARTH_RADIUS, mask_angle_rad=math.radians(5.0), sample_dt_s=20.0,
        theta0=theta0,
    )
    build = lambda: scenarios.tle_satellites(db_session_factory())  # noqa: E731
    results = run_sweep(
        build, [], horizon, oblateness={"Earth": (EARTH_J2, EARTH_R_EQ)}, timing_batches=1,
        timing_warmup=0, access=spec, external=[sgp4_bridge.sgp4_tier([tle], epoch, dt=20.0)],
    )
    metrics = results[0].access
    assert metrics is not None
    assert metrics.n_truth_windows >= 1
    assert metrics.n_matched == metrics.n_truth_windows == metrics.n_model_windows
    assert metrics.passes_lost == metrics.passes_gained == 0
    assert metrics.rise.max_abs_s < 0.1 and metrics.set.max_abs_s < 0.1
    assert float(np.linalg.norm(r0)) > 6500.0


def test_sgp4_tier_refuses_a_flagged_span() -> None:
    """Vallado's case 33333 is flagged (error 4) at 25 min; the tier must raise, not score NaN."""
    case = [c for c in _published_cases() if c[2] == 33333][0]
    tle = _tle(case[0], case[1], 33333)
    tier = sgp4_bridge.sgp4_tier([tle], sgp4_bridge.tle_epoch(tle), dt=60.0)
    assert np.isfinite(tier.positions(np.array([0.0, 600.0]))).all()
    with pytest.raises(ValueError, match="SGP4 reported errors"):
        tier.positions(np.array([0.0, 1500.0]))


# --------------------------------------------------------------------------------------------------
# Packaging: the engine imports without sgp4
# --------------------------------------------------------------------------------------------------

def test_engine_imports_without_sgp4() -> None:
    code = (
        "import sys; sys.modules['sgp4'] = None; sys.modules['sgp4.api'] = None\n"
        "import orbital_engine, orbital_engine.scenarios, orbital_engine.sweep, orbital_engine.sgp4_bridge as b\n"
        "try:\n"
        "    b.tle_epoch(b.ISS_TLE)\n"
        "except ImportError:\n"
        "    print('lazy-ok')\n"
    )
    env = dict(os.environ)
    src = os.path.dirname(os.path.dirname(os.path.abspath(sgp4_bridge.__file__)))
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "lazy-ok"
