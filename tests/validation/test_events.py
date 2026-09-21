"""
Validation of event-driven step splitting (`events.py`), and of the shadow event it exists for.

Every check states the number it expects **before** measuring it, per `CLAUDE.md`'s item 5. The
order is:

1. **The event function.** `srp.umbra_clearance` is the cylindrical terminator written as a signed
   scalar; its sign must agree with `srp.shadow_factor`'s `nu`, exactly, for every body outside the
   occulter. And the branch latch must pin `nu` to exactly 0 or exactly 1 regardless of geometry.
2. **The root find in isolation.** `events.locate_crossing` against an analytic scalar, checking the
   bracket, its width, the one-sidedness of the returned far endpoint, and the iteration count -
   which is the *cost* of the whole feature and the thing that regresses silently.
3. **Crossing location.** Against an independently computed terminator time on an orbit that is
   analytic end to end, so the comparison has no integrator in it.
4. **The headline: RK4's order comes back.** The convergence ladder that reads 1.46 unsplit must
   read the fourth-order band split, and the error must come down to the no-shadow control's.
5. **The residual is the tolerance.** The error left after splitting must scale with `tol_s` and sit
   under the derived `n Delta_a tol_s T_rem`.
6. **No regression.** A step with no crossing is **bit-identical** to the same step with no events
   registered; a scheduled impulse still lands exactly.
7. **Bookkeeping.** Several crossings in one step, the per-arena split, the split cap, and
   configuration errors.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import events, scenarios
from orbital_engine.custom_types import COEIndex, PropagatorType
from orbital_engine.database import Base
from orbital_engine.frames import ReferenceFrames
from orbital_engine.geopotential import EARTH_R_EQ
from orbital_engine.gravity import POINT_MASS_MODEL
from orbital_engine.simulator import Simulation
from orbital_engine.srp import (
    LATCH_AUTO, LATCH_DARK, LATCH_LIT, SHADOW_MODEL_CONICAL, SHADOW_MODEL_CYLINDRICAL,
    SOLAR_PRESSURE_1AU, SRP_MODEL, SUN_RADIUS, shadow_factor, umbra_clearance,
)

ArrF = NDArray[np.float64]

CR = 1.3            # a representative radiation-pressure coefficient
AREA_MASS = 0.2     # m^2/kg, matching tests/validation/test_srp.py's high-A/m object

#: The acceleration jump across the terminator, km/s^2. `a = C_r P (A/m) 1e-3` at 1 AU, which is
#: where `scenarios.eclipsed_satellite` puts its light source. Every error budget below is this
#: number times a duration.
DELTA_A = CR * SOLAR_PRESSURE_1AU * AREA_MASS * 1e-3


def _session() -> Session:
    """An isolated in-memory database, built the way `tests/conftest.py` builds one."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _build(
    *, r_occ: float = EARTH_R_EQ, shadow: float = SHADOW_MODEL_CYLINDRICAL, split: bool = False,
    tol_s: float = events.DEFAULT_EVENT_TOL_S, n_sats: int = 1, cowell: bool = True,
) -> Tuple[Simulation, NDArray[np.int64], int]:
    """`eclipsed_satellite` with `"srp"` enabled and, optionally, the shadow event registered."""
    sim = scenarios.eclipsed_satellite(_session(), n_sats=n_sats)
    sats = np.array([sim.name_to_index[f"ECLIPSE-SAT-{k:02d}"] for k in range(n_sats)],
                    dtype=np.int64)
    light = sim.name_to_index[scenarios.LIGHT_SOURCE_NAME]
    if cowell:
        sim.set_propagator(sats, PropagatorType.COWELL)
        sim.enable_force_model(POINT_MASS_MODEL, sats)
    sim.enable_force_model(
        SRP_MODEL, sats, cr=CR, area_mass=AREA_MASS, p_srp=SOLAR_PRESSURE_1AU,
        source=float(light), r_occ=r_occ, r_source=SUN_RADIUS, shadow_model=shadow)
    if split:
        sim.add_event(events.shadow_event(sim, tol_s=tol_s))
    return sim, sats, sim.name_to_index["Earth"]


def _run(dt: float, total: float, **kwargs: object) -> Tuple[ArrF, Simulation]:
    """Earth-relative end state of ECLIPSE-SAT-00 after `total` seconds at step `dt`."""
    sim, sats, earth = _build(**kwargs)                          # type: ignore[arg-type]
    for _ in range(int(round(total / dt))):
        sim.step(dt)
    return sim.global_states[sats[0], :3] - sim.global_states[earth, :3], sim


# ==================================================================================================
# 1. The event function and the branch latch
# ==================================================================================================

def test_umbra_clearance_agrees_in_sign_with_the_cylindrical_shadow_factor() -> None:
    """
    `umbra_clearance < 0` must mean exactly what `shadow_factor`'s cylindrical `nu == 0` means.

    The two are separate code paths over the same geometry - one a boolean membership test
    (`along < 0 and perp < r_occ`), the other a continuous signed distance
    (`hypot(perp, max(along, 0)) - r_occ`) - and the whole of event splitting rests on their
    agreeing. This samples a spherical shell of radius `1.5 r_occ` (every point of which is outside
    the occulter, the condition under which the two are equivalent - see `umbra_clearance`) on a
    deterministic lattice, and requires **exact** agreement: no tolerance, because the question is
    a sign, not a value.
    """
    r_occ, d_source = EARTH_R_EQ, scenarios.LIGHT_SOURCE_DISTANCE_KM
    n = 40
    u = np.linspace(-1.0, 1.0, n)
    phi = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    uu, pp = np.meshgrid(u, phi, indexing="ij")
    radius = 1.5 * r_occ
    pos = np.stack([
        radius * uu.ravel(),
        radius * np.sqrt(1.0 - uu.ravel() ** 2) * np.cos(pp.ravel()),
        radius * np.sqrt(1.0 - uu.ravel() ** 2) * np.sin(pp.ravel()),
    ], axis=1)

    source = np.array([d_source, 0.0, 0.0])
    to_source = source[None, :] - pos
    to_occulter = -pos                                          # occulter at the origin
    ro = np.full(pos.shape[0], r_occ)

    g = umbra_clearance(to_source, to_occulter, ro)
    nu = shadow_factor(to_source, to_occulter, np.full(pos.shape[0], SUN_RADIUS), ro,
                       np.zeros(pos.shape[0], dtype=np.bool_))

    assert np.array_equal(g < 0.0, nu == 0.0), (
        "the signed clearance and the boolean umbra test disagree on "
        f"{int(np.count_nonzero((g < 0.0) != (nu == 0.0)))} of {g.size} sample points")
    # Not a degenerate assertion: both branches must actually occur on this shell.
    assert 0 < int(np.count_nonzero(nu == 0.0)) < g.size


def test_no_occulter_can_never_report_a_crossing() -> None:
    """`r_occ <= 0` is `srp.py`'s "no shadow" convention. `umbra_clearance` must then be
    non-negative *everywhere* by construction (`hypot(...) >= 0 >= r_occ`), so the event can never
    fire and a heliocentric body costs nothing."""
    pos = np.array([[1.0, 2.0, 3.0], [-7000.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    g = umbra_clearance(np.array([[1e8, 0.0, 0.0]]) - pos, -pos, np.zeros(3))
    assert np.all(g >= 0.0), g


def test_the_latch_pins_the_branch_exactly() -> None:
    """
    A latched row returns exactly `1.0` or exactly `0.0` whatever the geometry says, and
    `LATCH_AUTO` is bit-identical to no latch at all.

    This is the property that makes a cut sub-step integrate one smooth branch at all four RK4
    stages. Exact equality, not `approx`: the kernel multiplies by `nu`, so a latched-lit row must
    contribute the full unshadowed acceleration and a latched-dark row exactly none.
    """
    r_occ = EARTH_R_EQ
    deep_umbra = np.array([[-7000.0, 0.0, 0.0]])                 # behind the occulter, on the axis
    full_sun = np.array([[7000.0, 0.0, 0.0]])                    # sunward
    source = np.array([[scenarios.LIGHT_SOURCE_DISTANCE_KM, 0.0, 0.0]])
    one, ro = np.ones(1), np.full(1, r_occ)
    conical = np.zeros(1, dtype=np.bool_)

    for pos, geometric in ((deep_umbra, 0.0), (full_sun, 1.0)):
        ts, to = source - pos, -pos
        assert shadow_factor(ts, to, one * SUN_RADIUS, ro, conical)[0] == geometric
        assert shadow_factor(ts, to, one * SUN_RADIUS, ro, conical,
                             np.full(1, LATCH_AUTO))[0] == geometric
        assert shadow_factor(ts, to, one * SUN_RADIUS, ro, conical, np.full(1, LATCH_LIT))[0] == 1.0
        assert shadow_factor(ts, to, one * SUN_RADIUS, ro, conical, np.full(1, LATCH_DARK))[0] == 0.0


# ==================================================================================================
# 2. The root find in isolation
# ==================================================================================================

def test_locate_crossing_brackets_the_root_and_returns_the_far_side() -> None:
    """
    `locate_crossing` on an analytic scalar, with the three properties `Simulation` relies on.

    `h(t) = sin(t) - 1/2` on `[0, 1]` has its only root at `asin(1/2) = pi/6 = 0.5235987756`. The
    contract is: the returned bracket straddles that root, is no wider than `tol_s`, and its far
    endpoint is the one with `h >= 0` - so the *error is one-sided*, `0 <= hi - root <= tol_s`,
    which is what lets `Simulation` step past the crossing rather than onto it.

    The iteration count is asserted too, and it is not incidental. Each call to `h_at` is, in
    `Simulation`, a full trial propagation of the arena - that is the entire cost of this feature.
    Plain bisection needs `log2(1 / 1e-9) = 30`; Illinois-modified false position needs about 10 on
    a function this smooth. A cap of 20 catches a regression to bisection (which is exactly what an
    over-strong interior safeguard causes, and did) while leaving real headroom.
    """
    root = math.asin(0.5)
    tol = 1.0e-9
    calls: List[float] = []

    def h_at(t: float) -> float:
        calls.append(t)
        return math.sin(t) - 0.5

    lo, hi, n = events.locate_crossing(h_at, 0.0, 1.0, h_at(0.0), h_at(1.0), tol)
    n_probe = len(calls) - 2                                     # the two endpoint calls above

    assert lo <= root <= hi, (lo, root, hi)
    assert hi - lo <= tol, f"bracket is {hi - lo:.3e} s wide against a tolerance of {tol:.3e}"
    assert h_at(hi) >= 0.0 > h_at(lo), "the far endpoint must be the one past the root"
    assert 0.0 <= hi - root <= tol, f"the error must be one-sided: hi - root = {hi - root:.3e}"
    assert n == n_probe
    assert n <= 20, (
        f"{n} evaluations to bracket a smooth root to {tol:.0e}; bisection alone would need "
        f"{math.ceil(math.log2(1.0 / tol))}, so this has stopped converging superlinearly")


def test_locate_crossing_raises_rather_than_returning_an_unconverged_root() -> None:
    """A pathological event function must raise, not quietly return a bracket wider than asked for.
    A silently loose bracket is a silently mistimed discontinuity."""
    with pytest.raises(RuntimeError, match="did not converge"):
        events.locate_crossing(
            lambda t: -1.0 if t < 0.5 else 1.0, 0.0, 1.0, -1.0, 1.0, 1e-12, max_iter=5)


def test_crossing_indices_filters_by_direction_and_ignores_a_zero_start() -> None:
    """
    The `direction` filter, `scipy.integrate.solve_ivp`'s convention: `+1` negative-to-positive,
    `-1` positive-to-negative, `0` both. Plus the two guards that stop a split rediscovering itself:
    a component starting at exactly `0.0` never fires, and a non-finite component never fires.
    """
    g0 = np.array([-1.0, +1.0, -1.0, +1.0, 0.0, np.nan, -1.0])
    g1 = np.array([+1.0, -1.0, +1.0, -1.0, -1.0, 1.0, -1.0])
    both = np.zeros(7)

    assert events.crossing_indices(g0, g1, both).tolist() == [0, 1, 2, 3]
    assert events.crossing_indices(g0, g1, np.full(7, +1.0)).tolist() == [0, 2]
    assert events.crossing_indices(g0, g1, np.full(7, -1.0)).tolist() == [1, 3]


def test_reduce_to_scalar_puts_its_first_zero_at_the_earliest_crossing() -> None:
    """
    The vector-to-scalar reduction, which is what lets one root find serve any number of crossing
    bodies. `H(tau) = max_j -sign(g_j(0)) g_j(tau)` is negative while *every* body is still on its
    starting side and non-negative once *any* has crossed, so its first zero is `min_j tau_j`.

    Two bodies crossing at `tau = 0.3` and `tau = 0.7`, both from positive to negative: `H` must be
    negative at 0.2, non-negative at 0.4, and the *earlier* crossing must be the one that sets it.
    """
    sign0 = np.array([1.0, 1.0])

    def g(tau: float) -> ArrF:
        return np.array([0.3 - tau, 0.7 - tau])

    assert events.reduce_to_scalar(g(0.2), sign0) < 0.0
    assert events.reduce_to_scalar(g(0.4), sign0) >= 0.0
    assert events.reduce_to_scalar(g(0.8), sign0) >= 0.0
    assert events.reduce_to_scalar(g(0.3), sign0) == pytest.approx(0.0, abs=1e-15)


# ==================================================================================================
# 3. Crossing location against an independent computation
# ==================================================================================================

def _analytic_terminator(sim: Simulation, sat: int, bracket: Tuple[float, float]) -> float:
    """
    The terminator crossing time from the *elements*, with no integrator and no `events.py`.

    Both the satellite and the light source are Keplerian and **circular** here (`e == 0`, asserted),
    so each one's Earth-relative position at an arbitrary time is `coe_to_rv` of the same elements
    with the true anomaly advanced linearly, `theta(t) = theta_now + n (t - t_now)`. No Kepler
    solver, no `Simulation.step`, no `events.locate_crossing`, no trial propagation: the only shared
    code is `frames.coe_to_rv` and the event function itself, neither of which is what is under
    test. Bisected to 1e-12 s, six orders below the tolerance being checked.
    """
    light = sim.name_to_index[scenarios.LIGHT_SOURCE_NAME]
    r_occ = float(sim.force_model_params[SRP_MODEL][sat, 4])
    t_now = float(sim.t)

    def position(slot: int, t: float) -> ArrF:
        coe = sim.coe_states[slot].copy()
        e = float(coe[COEIndex.E])
        # `_rehydrate_coes` leaves a seeded-circular orbit at e ~ 1e-16 rather than exactly 0. The
        # true anomaly then leads the mean by `2 e sin M <= 2.2e-16 rad`, which is `1.5e-12 km` of
        # arc at 7000 km, or `5e-13 s` - six orders below the tolerance under test.
        assert e < 1.0e-12, f"slot {slot} is not circular (e={e!r}); the linear advance needs it"
        mu = float(sim.mu_array[slot] + sim.mu_array[sim.parent_indices[slot]])
        a = float(coe[COEIndex.P])                               # circular, so p == a
        coe[COEIndex.THETA] += math.sqrt(mu / a ** 3) * (t - t_now)
        r, _v, ok = ReferenceFrames.coe_to_rv(coe[None, :], np.array([mu]))
        assert bool(ok[0])
        return np.asarray(r[0])

    def clearance(t: float) -> float:
        # Earth heads the arena root and never moves, so `coe_to_rv` of a parent-relative element
        # set *is* the Earth-relative position, and the occulter sits at the origin of that frame.
        r_sat = position(sat, t)
        r_src = position(light, t)
        return float(umbra_clearance(
            (r_src - r_sat)[None, :], (-r_sat)[None, :], np.array([r_occ]))[0])

    lo, hi = bracket
    f_lo = clearance(lo)
    assert f_lo * clearance(hi) < 0.0, "the supplied bracket does not straddle a crossing"
    while hi - lo > 1.0e-12:
        mid = 0.5 * (lo + hi)
        if clearance(mid) * f_lo > 0.0:
            lo, f_lo = mid, clearance(mid)
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_located_crossing_matches_the_analytic_terminator_to_the_tolerance() -> None:
    """
    The located crossing must equal an independently computed one to `tol_s`, one-sided.

    The satellite is left **Keplerian**, so its trajectory is a closed form and the comparison
    contains no integrator error at all - what is being measured is the root find and nothing else.
    `"srp"` is still enabled (a massless Keplerian body feels no acceleration from it, but the event
    is defined from the same coefficients), so the event fires exactly as it would for a Cowell body.

    Derived expectation. `locate_crossing` returns the bracket's far endpoint and the bracket is no
    wider than `tol_s`, so the contract is `|located - true| <= tol_s`. In practice it is far better
    than that: Illinois converges the *iterate* long before the bracket *width* reaches `tol_s`, so
    the far endpoint sits essentially on the root and what is left is the difference between two
    analytic propagations of the same circular orbit - **1e-10 s**, which is also why the sign of
    the residual is not assertable here. The one-sidedness the splitter relies on is a property of
    `locate_crossing` against its own function and is asserted there, in
    `test_locate_crossing_brackets_the_root_and_returns_the_far_side`.

    So: `|error| <= tol_s` is the contract, and `|error| < 1e-8 s` is the measurement - 100x tighter
    - which is what would actually catch a mislocated crossing.
    """
    tol = 1.0e-6
    sim, sats, _ = _build(split=True, tol_s=tol, cowell=False)
    for _ in range(880):
        sim.step(10.0)

    assert len(sim.event_epochs) >= 2, sim.event_epochs
    entry, exit_ = sim.event_epochs[0], sim.event_epochs[1]

    true_entry = _analytic_terminator(sim, int(sats[0]), (entry - 60.0, entry + 1.0))
    true_exit = _analytic_terminator(sim, int(sats[0]), (exit_ - 60.0, exit_ + 1.0))

    for located, truth, label in ((entry, true_entry, "entry"), (exit_, true_exit, "exit")):
        err = abs(located - truth)
        assert err <= tol, (
            f"{label}: located {located!r} against the analytic {truth!r}, error {err:.3e} s, "
            f"outside the stated tolerance of {tol:.0e} s")
        assert err < 1.0e-8, (
            f"{label}: error {err:.3e} s. Both trajectories here are analytic, so the root find "
            f"should agree with the closed form to the Kepler rounding (~1e-10 s), not merely to "
            f"tol_s. Something has moved the crossing.")

    # And the eclipse itself must be the right length: ~35 % of the orbit at 550 km.
    period = 2.0 * math.pi * math.sqrt(
        (scenarios.EARTH_RADIUS + 550.0) ** 3 / scenarios.MU_EARTH)
    assert 0.25 < (true_exit - true_entry) / period < 0.40, (true_exit - true_entry, period)


# ==================================================================================================
# 4. The headline: RK4's order comes back
# ==================================================================================================

CONV_TOTAL = 8800.0                      # s; ~1.5 orbits, three terminator crossings
CONV_STEPS = (10.0, 5.0, 2.5, 1.25)
CONV_REF_DT = 0.3125                     # 2 halvings below the finest, so its own error is ~1/16


@pytest.fixture(scope="module")
def convergence() -> Dict[str, List[float]]:
    """
    Absolute end-state error against a fine reference, for each configuration.

    Errors against a common reference rather than successive differences: on a first-order sequence
    the successive differences are dominated by whichever crossing happened to be mistimed most and
    read as noise, which is why the existing `test_srp.py` ladder can only assert an inequality.

    Two references, not three. `"cylinder"` and `"split"` are two discretisations of the *same*
    discontinuous problem and converge to the same trajectory, so they share the split run at
    `CONV_REF_DT` - which is the accurate one of the pair, and is what makes the unsplit ladder's
    first-order behaviour visible rather than self-referential. `"none"` is a different problem (no
    shadow at all) and gets its own.
    """
    ref_shadow, _ = _run(CONV_REF_DT, CONV_TOTAL, r_occ=EARTH_R_EQ, split=True)
    ref_none, _ = _run(CONV_REF_DT, CONV_TOTAL, r_occ=0.0, split=False)

    out: Dict[str, List[float]] = {}
    for key, ref, kwargs in (
        ("none", ref_none, dict(r_occ=0.0, split=False)),
        ("cylinder", ref_shadow, dict(r_occ=EARTH_R_EQ, split=False)),
        ("split", ref_shadow, dict(r_occ=EARTH_R_EQ, split=True)),
    ):
        out[key] = [float(np.linalg.norm(_run(dt, CONV_TOTAL, **kwargs)[0] - ref))  # type: ignore[arg-type]
                    for dt in CONV_STEPS]
    return out


def test_event_splitting_restores_rk4s_convergence_order(
    convergence: Dict[str, List[float]],
) -> None:
    """
    The measurement this whole module exists for, and the reverse of
    `test_srp.py::test_the_cylindrical_terminator_costs_rk4_its_convergence_order`.

    Derivation, stated before measuring.

    *Unsplit.* A step straddling the terminator mistimes the switch by up to `h`, so each crossing
    contributes a velocity error of order `Delta_a h / 2` which then grows linearly in the time
    remaining. Halving `h` halves it: the step-halving error ratio tends to **2**, not 16.

    *Split.* No step straddles the discontinuity; every step integrates one smooth branch (the
    latch is what guarantees that at all four stages - see `Simulation._advance_with_events`). RK4's
    order theorem then applies unchanged, so the ratio must return to **16** and the error must fall
    to what the same orbit costs with no shadow at all. What is left over is the crossing-time
    tolerance, `n Delta_a tol_s T_rem`, which at `tol_s = 1e-6 s` is `2 x 1.19e-9 x 1e-6 x 2200`
    = `5e-12 km` - two orders below RK4's own truncation at the finest step on this ladder, and so
    invisible here by design. It is measured directly in
    `test_the_residual_after_splitting_is_the_crossing_tolerance`.

    The assertions are therefore:

    - every split ratio is in the fourth-order band, `> 8` (half of 16, the usual allowance for a
      ladder that has not fully reached the asymptotic regime);
    - the unsplit ratio at the finest pair has fallen **below 4**, i.e. is not fourth order;
    - the split error at the finest step is within 1.5x of the no-shadow control's - the
      discontinuity has stopped costing anything;
    - and the split error at the finest step is at least 20x below the unsplit one, which is the
      engineering claim.
    """
    def ratios(key: str) -> List[float]:
        e = convergence[key]
        return [e[i] / e[i + 1] for i in range(len(e) - 1)]

    split_r, none_r, cyl_r = ratios("split"), ratios("none"), ratios("cylinder")
    split_e, none_e, cyl_e = (convergence[k][-1] for k in ("split", "none", "cylinder"))

    assert min(none_r) > 8.0, f"the control itself is not converging at fourth order: {none_r}"
    assert cyl_r[-1] < 4.0, (
        f"the unsplit cylinder converged at ratio {cyl_r[-1]:.2f} on the finest halving; the "
        f"discontinuity is supposed to hold it near 2, so this measurement has lost its subject")
    assert min(split_r) > 8.0, (
        f"split step-halving ratios {['%.2f' % r for r in split_r]} are not in the fourth-order "
        f"band; a first-order error halves, a fourth-order one falls 16x")
    assert split_e < 1.5 * none_e, (
        f"split error {split_e:.3e} km against the no-shadow control's {none_e:.3e} km: the "
        f"discontinuity should cost essentially nothing once the step is cut at it")
    assert split_e < cyl_e / 20.0, (
        f"splitting improved the finest-step error only {cyl_e / split_e:.1f}x "
        f"({cyl_e:.3e} -> {split_e:.3e} km)")


# ==================================================================================================
# 5. The residual is the tolerance
# ==================================================================================================

def test_the_residual_after_splitting_is_the_crossing_tolerance() -> None:
    """
    What is left after splitting is the crossing-time tolerance, and it must be bounded by the
    derived budget.

    Derivation. A crossing located `eps` seconds late applies the wrong branch of the acceleration
    for `eps`, so the velocity is wrong by `Delta_a eps`, which grows into position error linearly
    in the time remaining: `sum_j Delta_a eps T_rem,j` over the crossings. Here there are two
    (entry at ~1796 s, exit at ~3934 s) and over a 4400 s horizon the remaining times are 2604 s and
    466 s, summing to `T_sum = 3070 s`, with `Delta_a = 1.19e-9 km/s^2`.

    `eps` is not exactly `tol_s`. The arena is left at the bracket's far endpoint - at most `tol_s`
    past the crossing - plus one postcondition nudge of `tol_s` for each time the split had to be
    pushed clear of the surface (`Simulation.event_nudges`). So
    `eps <= (1 + MAX_CROSSING_NUDGES) tol_s`, and the *measured* coefficient
    `k = err / (Delta_a tol_s T_sum)` is asserted to stay under 4 - one bracket width plus one or
    two nudges. It comes out at **2.0 to 2.7**, which is the honest statement of where the residual
    comes from.

    Measured against a `tol_s = 1e-9` run, three decades tighter than the loosest tested.

    **The residual stops tracking `tol_s` above about 1e-4**, and that is a real property rather
    than a convenience: Illinois converges the *iterate* far inside the bracket long before the
    bracket *width* reaches a loose `tol_s`, so at `tol_s = 1e-2` and `1e-3` the split lands in the
    same place and the residual is identical (1.42e-9 km both times). The upper bound is therefore
    checked at every tolerance, while the linear scaling is checked over the decades where the
    residual is actually tolerance-limited.
    """
    total, dt = 4400.0, 10.0
    reference, _ = _run(dt, total, split=True, tol_s=1.0e-9)

    t_rem_sum = (total - 1796.0) + (total - 3934.0)              # 3070 s
    tolerances = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6)

    measured: Dict[float, float] = {}
    for tol in tolerances:
        x, sim = _run(dt, total, split=True, tol_s=tol)
        err = float(np.linalg.norm(x - reference))
        measured[tol] = err
        assert sim.event_splits == 2, (tol, sim.event_splits)
        assert sim.event_nudges <= sim.event_splits, (
            f"tol_s={tol:.0e}: {sim.event_nudges} nudges for {sim.event_splits} crossings; more "
            f"than one apiece means the root find's trial trajectory and the split's disagree "
            f"by more than the tolerance, which would make the residual bound below meaningless")
        k = err / (DELTA_A * tol * t_rem_sum)
        assert k <= 4.0, (
            f"tol_s={tol:.0e}: residual {err:.3e} km is {k:.2f} x Delta_a tol_s T_sum. The wrong "
            f"branch should be applied for one bracket width plus at most a nudge or two")

    # And it must actually track the tolerance where it is tolerance-limited: two decades of tol_s
    # must buy between 30x and 300x, against the 100x a linear residual predicts. A residual that
    # did not move would mean the split is not where the root find says it is.
    shrink = measured[1.0e-4] / measured[1.0e-6]
    assert 30.0 < shrink < 300.0, (shrink, measured)

    # The number that matters: at the default tolerance the residual is two orders below RK4's own
    # truncation at the finest step of the convergence ladder, which is why it is invisible there.
    assert measured[1.0e-6] < 2.0e-11, measured


# ==================================================================================================
# 6. No regression
# ==================================================================================================

def test_a_step_with_no_crossing_is_bit_identical_to_one_with_no_events() -> None:
    """
    The assertion that protects every other test in the suite.

    Over `[0, 1700]` s the satellite never reaches the terminator (entry is at ~1796 s), so the
    event is registered, evaluated twice per step, and never fires. The arena must come out **bit
    for bit** identical to the same run with no event registered at all - not `approx`, not to
    1e-12. That is achievable by construction because the detection is two pure reads around the
    *same single* `_advance` call, and the speculative advance is kept rather than repeated.
    """
    plain, _ = _run(10.0, 1700.0, split=False)
    with_event_sim, sats, earth = _build(split=True)
    for _ in range(170):
        with_event_sim.step(10.0)
    split = with_event_sim.global_states[sats[0], :3] - with_event_sim.global_states[earth, :3]

    assert with_event_sim.event_splits == 0, with_event_sim.event_epochs
    assert np.array_equal(plain, split), (
        f"a step with no crossing is not bit-identical: {plain!r} vs {split!r}, "
        f"difference {np.abs(plain - split)!r}")


def test_a_scheduled_impulse_still_splits_exactly_with_events_registered() -> None:
    """
    Events and scheduled manoeuvres compose: each manoeuvre sub-step is itself scanned for
    crossings, and a manoeuvre epoch inside a crossing-free window must land exactly where it did
    before - bit for bit, for the same reason as the test above.

    The impulse is at 1000 s, an epoch that is *not* a multiple of the 300 s step, so the step is
    genuinely cut at it; and the whole window is before the first terminator crossing at ~1796 s.
    """
    burn = np.array([0.0, 0.01, 0.0])                            # 10 m/s prograde
    ends: List[ArrF] = []
    for split in (False, True):
        sim, sats, earth = _build(split=split)
        sim.schedule_delta_v(sats, burn, epoch_s=1000.0)
        for _ in range(5):
            sim.step(300.0)
        assert sim.pending_manoeuvres == ()
        assert sim.event_splits == 0
        ends.append(sim.global_states[sats[0], :3] - sim.global_states[earth, :3])

    assert np.array_equal(ends[0], ends[1]), (ends, np.abs(ends[0] - ends[1]))


# ==================================================================================================
# 7. Bookkeeping: several crossings, the per-arena split, and configuration
# ==================================================================================================

def test_two_crossings_inside_one_step_are_both_resolved() -> None:
    """
    One step, two crossings, both resolved: the remainder of a split step is re-scanned.

    Two satellites half an orbit apart. The trailing one is seeded inside the umbra and leaves it at
    ~1069 s; the leading one enters at ~1796 s. A single 2400 s step therefore contains both, of
    different bodies. A splitter that cut once and integrated the rest in one go would resolve the
    first and run straight through the second - and `event_splits` would read 1.

    The located epochs are checked against a fine (10 s) run, to within the 1 s by which the coarse
    step's own trajectory error can move a terminator.
    """
    fine, _, _ = _build(split=True, cowell=False, n_sats=2)
    for _ in range(240):
        fine.step(10.0)
    assert fine.event_splits == 2, fine.event_epochs

    coarse, _, _ = _build(split=True, cowell=False, n_sats=2)
    coarse.step(2400.0)
    assert coarse.event_splits == 2, (
        f"one 2400 s step containing two crossings resolved {coarse.event_splits} of them; the "
        f"remainder of a split step is not being re-scanned")
    for coarse_t, fine_t in zip(coarse.event_epochs, fine.event_epochs):
        assert abs(coarse_t - fine_t) < 1.0, (coarse.event_epochs, fine.event_epochs)


def test_an_even_number_of_crossings_of_one_body_in_one_step_is_invisible() -> None:
    """
    The documented blind spot, asserted rather than left in prose so that nobody "fixes" it by
    accident and nobody relies on it not existing.

    Detection compares the event function at the two ends of an interval. A body that crosses an
    *even* number of times inside one interval shows the same sign at both ends and fires nothing -
    the same limitation `scipy.integrate.solve_ivp` has, and for the same reason: closing it needs
    interior sampling, and here every interior sample costs a full trial propagation of the arena.

    A single 8000 s step spans one satellite's eclipse entry (~1796 s), its exit (~3934 s) and its
    next entry (~7526 s): three crossings, so the *first* is found, and the remainder then holds an
    even two and is silent. One split, not three. The defence is the step size - an eclipse lasts
    thousands of seconds and `dt` is seconds to minutes - and that is a statement about how the
    engine is used, not a property of this code.
    """
    sim, _, _ = _build(split=True, cowell=False)
    sim.step(8000.0)
    assert sim.event_splits == 1, (
        f"a single step spanning three crossings resolved {sim.event_splits}; if this now reads 3 "
        f"the detector has gained interior sampling and events.py's limitations section is stale")


def test_the_split_is_per_arena_and_analytic_bodies_are_unharmed_by_it() -> None:
    """
    Detection is per body; the split is per arena. Both halves are asserted here.

    *Per body*: with four satellites phased evenly round one orbit, each crosses the terminator on
    its own schedule. Over a window of slightly more than one orbital period every satellite enters
    and leaves the umbra exactly once whatever its phase, so the split count must be **eight** - two
    per body, four times the one-satellite count. An implementation that split once per *step* in
    which anything crossed, rather than once per crossing, would read four.

    *Per arena*: every body takes the sub-steps, including ones that are nowhere near their own
    event surface. For an **analytic** body that costs exactly nothing, because a closed-form
    advance of `h1` then `h2` is the advance of `h1 + h2` - the same argument `manoeuvres.py` makes
    for impulse splitting. The light source is Keplerian and sits 1 AU out, so it is the sharpest
    available probe: its position after a heavily split run must match an unsplit one to the Kepler
    solver's own convergence, which on a `1.5e8 km` orbit is `~1e-5 km` relative to `1e-10`.
    """
    n_steps = 600                                                # 6000 s, ~1.05 orbits at 550 km
    one, _, _ = _build(split=True, n_sats=1)
    for _ in range(n_steps):
        one.step(10.0)

    four, _, _ = _build(split=True, n_sats=4)
    for _ in range(n_steps):
        four.step(10.0)

    assert one.event_splits == 2, one.event_epochs
    assert four.event_splits == 8, (four.event_splits, four.event_epochs)

    plain, _, _ = _build(split=False, n_sats=4)
    for _ in range(n_steps):
        plain.step(10.0)

    light = four.name_to_index[scenarios.LIGHT_SOURCE_NAME]
    drift = float(np.linalg.norm(four.global_states[light, :3] - plain.global_states[light, :3]))
    assert drift < 1.0e-4, (
        f"splitting moved a Keplerian body by {drift:.3e} km; an analytic advance of h1 then h2 is "
        f"the advance of h1 + h2, so this should be the Kepler solver's rounding and nothing more")


def test_the_cost_per_crossing_is_the_root_finds_iteration_count() -> None:
    """
    The price of the feature, stated as a number so that a regression in it is visible.

    Each root-find iteration is one trial propagation of the whole arena - `_advance` plus a
    snapshot restore - so `event_evaluations / event_splits` is the multiplier a crossing step costs
    over a plain one. Illinois-modified false position on a near-linear geometric event needs about
    6 to 10 from a 10 s step at the default `1e-6 s` tolerance; plain bisection would need 24, and
    that is exactly what an over-strong interior safeguard produced before it was measured. The cap
    of 16 sits between the two.

    A step with **no** crossing costs two extra evaluations of the event function - a handful of dot
    products over the event bodies - and the snapshot copies, and nothing else: no extra propagation
    at all, which is what `event_evaluations` staying at 0 over a crossing-free window shows.
    """
    quiet, _, _ = _build(split=True)
    for _ in range(170):
        quiet.step(10.0)
    assert quiet.event_evaluations == 0, (
        "a crossing-free window must cost no trial propagations at all")

    sim, _, _ = _build(split=True)
    for _ in range(440):
        sim.step(10.0)
    assert sim.event_splits == 2, sim.event_epochs
    per_crossing = sim.event_evaluations / sim.event_splits
    assert per_crossing <= 16.0, (
        f"{per_crossing:.1f} trial propagations per crossing; bisection to 1e-6 s from a 10 s step "
        f"needs {math.ceil(math.log2(10.0 / 1e-6))}, so the root find has stopped being superlinear")


def test_exceeding_max_event_splits_raises_rather_than_dropping_crossings() -> None:
    """
    Silently dropping the remaining crossings would restore, invisibly, the first-order error this
    machinery removes. So it raises.

    The two-satellite step from `test_two_crossings_inside_one_step_are_both_resolved`, which
    resolves two crossings, against a cap of one. That test is the control: the same call succeeds
    with the default cap, so what fails here is the cap and nothing else.
    """
    sim, _, _ = _build(split=True, cowell=False, n_sats=2)
    sim.max_event_splits = 1
    with pytest.raises(ValueError, match="max_event_splits"):
        sim.step(2400.0)


def test_shadow_event_and_add_event_reject_configurations_that_could_never_fire() -> None:
    """A conical shadow is continuous and has no discontinuity to split at; an `r_occ <= 0` body has
    no occulter; a body without `"srp"` has no shadow. All three leave `shadow_event` with an empty
    body list, which raises rather than registering an event that can never fire."""
    conical, _, _ = _build(shadow=SHADOW_MODEL_CONICAL)
    with pytest.raises(ValueError, match="no body qualifies"):
        events.shadow_event(conical)

    no_occulter, _, _ = _build(r_occ=0.0)
    with pytest.raises(ValueError, match="no body qualifies"):
        events.shadow_event(no_occulter)

    sim, sats, _ = _build()
    good = events.shadow_event(sim)
    assert good.bodies.tolist() == sats.tolist()
    assert good.direction == 0                                   # entry and exit are both sharp

    for bad, match in (
        (events.Event("x", good.function, np.empty(0, dtype=np.int64)), "names no bodies"),
        (events.Event("x", good.function, np.array([10 ** 6])), "outside the arena"),
        (events.Event("x", good.function, sats, direction=2), "direction"),
        (events.Event("x", good.function, sats, tol_s=0.0), "tol_s"),
    ):
        with pytest.raises(ValueError, match=match):
            sim.add_event(bad)

    sim.add_event(good)
    assert sim.registered_events == (good,)
    sim.clear_events()
    assert sim.registered_events == () and sim.event_epochs == ()


def test_the_shadow_latch_is_engine_owned_and_not_a_coefficient() -> None:
    """`shadow_latch` is written and released inside one step by `_advance_with_events`. A value
    pinned at configuration time would freeze the shadow for the whole run - a plausible-looking
    configuration producing a silently wrong orbit - so `enable_force_model` refuses it."""
    sim, sats, _ = _build()
    with pytest.raises(ValueError, match="engine-owned state"):
        sim.enable_force_model(SRP_MODEL, sats, shadow_latch=LATCH_DARK)


def test_the_latch_is_released_before_step_returns() -> None:
    """The latch is per sub-step, never per run. After a step containing a crossing, every row must
    be back at `LATCH_AUTO` - otherwise the *next* step would integrate a frozen shadow and nothing
    would raise."""
    sim, sats, _ = _build(split=True)
    for _ in range(440):
        sim.step(10.0)
    assert sim.event_splits >= 1
    latch = sim.force_model_params[SRP_MODEL][sats, 7]
    assert np.array_equal(latch, np.full(sats.size, LATCH_AUTO)), latch
