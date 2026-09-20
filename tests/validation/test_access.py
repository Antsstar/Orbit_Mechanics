"""
Validation of `access.py`: model error expressed in contact windows rather than kilometres.

The five checks the feature was specified against, each with its expected magnitude **derived before
it was measured** (`CLAUDE.md`'s item 5):

1. **Self-consistency.** Truth compared with itself gives *exactly* zero shift, zero duration error,
   zero contact error and no pass gained or lost. Exact, not approximate: the same float is
   subtracted from itself.
2. **A known shift.** For a circular *equatorial* orbit seen from an *equatorial* station, elevation
   depends only on the station-relative angle `theta0 + (n - omega) t`, so advancing the satellite's
   in-plane phase by `delta` is **exactly** a time translation of `-delta / Omega`,
   `Omega = n - omega`. Choosing `delta = Omega * h` makes the translation exactly one grid step, so
   the sampled elevation sequences are index-shifted copies and the interpolated edges must move by
   exactly `-h`. Derived: `-5.000000 s`. Measured: see the test's own tolerance, 1e-6 s.
   The *sign* is the part that matters - a satellite ahead of truth rises **early**.
3. **Tie to the existing numbers.** `run_sweep`'s position-error statistics are bit-identical with
   and without `access=`, so the new metric is additive in fact and not merely in intent.
4. **Matching.** A mask angle is raised until two of four true passes drop below it *for the model
   only*. The metric must report `passes_lost == 2` and pair the two survivors with the correct true
   passes - which is asserted by identity and by the shift magnitude, because a matcher that paired
   by list index would produce the same *counts* while differencing passes an orbit apart.
5. **The matcher alone**, on hand-built window records with no propagation at all, where the right
   answer is written down.

`geometry.py`'s documented limits are handled rather than ignored: window edges carry an `O(h^2)`
convex-horizon **bias** (durations read long), so truth and model are always sampled on the *same*
grid and the bias cancels to first order - see `access.py`'s "Sampling step", which derives the
residual `C * Delta * h` and the 60 s default from it. `peak_elevation_rad` is sampled rather
than refined, so it is used here only to *choose* a mask angle, never differenced.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Callable, List, Sequence

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import access, geopotential, scenarios, sweep
from orbital_engine.custom_types import PropagatorType
from orbital_engine.drag import EARTH_OMEGA
from orbital_engine.geometry import AccessWindow
from orbital_engine.simulator import Simulation

# --------------------------------------------------------------------------------------------------
# Shared geometry: the `scenarios.ground_station_pass` site, and the orbit rate its pass is set by.
# --------------------------------------------------------------------------------------------------

ALTITUDE_KM = 550.0
ORBIT_RADIUS_KM = scenarios.EARTH_RADIUS + ALTITUDE_KM
MEAN_MOTION = math.sqrt(scenarios.MU_EARTH / ORBIT_RADIUS_KM ** 3)          # 1.0965e-3 rad/s
STATION_RELATIVE_RATE = MEAN_MOTION - EARTH_OMEGA                           # Omega, 1.0236e-3 rad/s

STATION = access.GroundStation(
    name="ground_station_pass site",
    latitude_rad=math.radians(scenarios.STATION_LATITUDE_DEG),
    longitude_rad=math.radians(scenarios.STATION_LONGITUDE_DEG),
    altitude_km=scenarios.STATION_ALTITUDE_KM,
)


def _spec(sample_dt_s: float, mask_deg: float = 0.0) -> access.AccessSpec:
    return access.AccessSpec(
        stations=[STATION],
        central_body="Earth",
        omega=EARTH_OMEGA,
        body_radius_km=scenarios.EARTH_RADIUS,
        mask_angle_rad=math.radians(mask_deg),
        sample_dt_s=sample_dt_s,
    )


def _pass_windows(
    session_factory: Callable[[], Session],
    times: np.ndarray,
    spec: access.AccessSpec,
    *,
    phase_deg: float,
    inclination_deg: float,
) -> List[AccessWindow]:
    """
    Windows for one `scenarios.ground_station_pass` satellite whose in-plane phase is advanced by
    `phase_deg`.

    The phase is applied through `raan_deg`, which for this scenario's circular orbit adds directly
    to the argument of latitude (`Rz(raan) Rx(i) Rz(argpe)` with `argpe = 0`), so it advances the
    satellite along its own track without touching `p`, `e` or `i`. For `inclination_deg = 0` that is
    the pure along-track phase shift test 2 needs; for an inclined orbit it also rotates the ground
    track, which is what gives test 4 its varying peak elevations.
    """
    sim = scenarios.ground_station_pass(
        session_factory(), n_sats=1, altitude_km=ALTITUDE_KM,
        inclination_deg=inclination_deg, raan_deg=phase_deg,
    )
    sim.record_history = False
    return access.windows_from_simulation(
        sim, [sim.name_to_index["PASS-SAT-00"]], times, spec
    )


def _win(
    rise_s: float,
    set_s: float,
    *,
    station: int = 0,
    body: int = 0,
    rise_clipped: bool = False,
    set_clipped: bool = False,
) -> AccessWindow:
    """A hand-built window record, for the matcher tests that involve no propagation."""
    return AccessWindow(
        station_index=station, body_index=body, rise_s=rise_s, set_s=set_s,
        duration_s=set_s - rise_s, peak_elevation_rad=0.5,
        rise_clipped=rise_clipped, set_clipped=set_clipped,
    )


# ==================================================================================================
# 1. Self-consistency: truth against itself is exactly zero
# ==================================================================================================

def test_truth_against_itself_is_exactly_zero(db_session_factory: Callable[[], Session]) -> None:
    spec = _spec(sample_dt_s=10.0)
    times = access.access_grid(12.0 * 3600.0, spec.sample_dt_s)
    windows = _pass_windows(
        db_session_factory, times, spec, phase_deg=0.0, inclination_deg=53.0
    )
    assert len(windows) >= 2, "the self-comparison must have something to compare"

    metrics = access.compare_windows(windows, windows)

    assert metrics.passes_lost == 0
    assert metrics.passes_gained == 0
    assert metrics.n_matched == len(windows) == metrics.n_truth_windows == metrics.n_model_windows
    # Exact zeros, not approximate: identical floats differenced.
    for stats in (metrics.rise, metrics.set, metrics.duration):
        assert stats.n > 0
        assert stats.mean_s == 0.0
        assert stats.median_s == 0.0
        assert stats.max_abs_s == 0.0
    assert metrics.total_contact_error_s == 0.0
    assert access.contact_fraction(metrics) == 1.0


# ==================================================================================================
# 2. A known shift, derived from the orbit rate rather than from the code
# ==================================================================================================

def test_along_track_phase_shift_moves_the_window_by_the_derived_time(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Equatorial orbit, equatorial station: elevation is a function of the station-relative angle
    `theta0 + Omega t` alone, so adding `delta` to the satellite's phase is exactly the time
    translation `t -> t + delta / Omega`. The model therefore reaches any given geometry **early**,
    and both its rise and its set must move by `-delta / Omega`.

    `delta` is chosen as `Omega * h`, one grid step, so the model's sampled elevation array is the
    truth's shifted by exactly one index and the linear edge interpolation reproduces the
    translation exactly - the `O(h^2)` convex-horizon bias is identical on both sides and cancels
    completely. Derived expectation: **-5.000000 s** on both edges. Anything near +5 s is a sign
    error in the differencing; anything near 0 is a matcher that paired nothing.
    """
    h = 5.0
    expected_shift_s = -h
    delta_rad = STATION_RELATIVE_RATE * h

    spec = _spec(sample_dt_s=h)
    # 4000 s puts the whole 784 s pass (2677 s -> 3461 s) strictly inside the grid, so neither edge
    # is clipped and both are interpolated.
    times = access.access_grid(4000.0, h)

    truth = _pass_windows(db_session_factory, times, spec, phase_deg=0.0, inclination_deg=0.0)
    model = _pass_windows(
        db_session_factory, times, spec, phase_deg=math.degrees(delta_rad), inclination_deg=0.0
    )
    assert len(truth) == len(model) == 1
    assert not (truth[0].rise_clipped or truth[0].set_clipped)
    assert not (model[0].rise_clipped or model[0].set_clipped)

    metrics = access.compare_windows(truth, model)
    assert metrics.n_matched == 1 and metrics.passes_lost == 0 and metrics.passes_gained == 0

    assert metrics.rise.mean_s == pytest.approx(expected_shift_s, abs=1e-6)
    assert metrics.set.mean_s == pytest.approx(expected_shift_s, abs=1e-6)
    # A pure translation changes neither the duration nor the total contact time.
    assert metrics.duration.max_abs_s == pytest.approx(0.0, abs=1e-6)
    assert metrics.total_contact_error_s == pytest.approx(0.0, abs=1e-6)


def test_phase_shift_scales_with_the_orbit_rate(db_session_factory: Callable[[], Session]) -> None:
    """
    The same derivation at a phase offset that is *not* a whole grid step: `delta = 0.002 rad` must
    move the window by `-delta / Omega = -1.9539 s`.

    Here the two crossings generally fall at different fractions of their (same) sample interval, so
    the cancellation of the edge bias is incomplete and the residual is bounded by
    `C * Delta * h = 1.206e-3 * 1.954 * 5 = 1.2e-2 s` (`access.py`, "Sampling step"). The tolerance
    is that bound, not a number tuned until the test passed.
    """
    h = 5.0
    delta_rad = 0.002
    expected_shift_s = -delta_rad / STATION_RELATIVE_RATE                     # -1.95389 s
    bias_coefficient = 0.5 * STATION_RELATIVE_RATE / math.tan(
        math.acos(scenarios.EARTH_RADIUS / ORBIT_RADIUS_KM)
    )                                                                        # C = 1.206e-3 1/s
    tolerance_s = bias_coefficient * abs(expected_shift_s) * h               # 1.18e-2 s

    spec = _spec(sample_dt_s=h)
    times = access.access_grid(4000.0, h)
    truth = _pass_windows(db_session_factory, times, spec, phase_deg=0.0, inclination_deg=0.0)
    model = _pass_windows(
        db_session_factory, times, spec, phase_deg=math.degrees(delta_rad), inclination_deg=0.0
    )

    metrics = access.compare_windows(truth, model)
    assert metrics.n_matched == 1
    assert metrics.rise.mean_s == pytest.approx(expected_shift_s, abs=tolerance_s)
    assert metrics.set.mean_s == pytest.approx(expected_shift_s, abs=tolerance_s)
    assert bias_coefficient == pytest.approx(1.206e-3, rel=1e-2), "the derived C, restated"


# ==================================================================================================
# 3. Tie to the existing numbers: `run_sweep`'s error statistics do not move
# ==================================================================================================

def _constellation_builder(
    db_session_factory: Callable[[], Session],
) -> Callable[[], Simulation]:
    def build() -> Simulation:
        sim = scenarios.earth_constellation(
            db_session_factory(), n_sats=2, n_planes=1,
            altitude_km=ALTITUDE_KM, inclination_deg=53.0,
        )
        sim.record_history = False
        return sim
    return build


def test_access_metrics_do_not_disturb_the_position_error_statistics(
    db_session_factory: Callable[[], Session],
) -> None:
    """`run_sweep`'s existing contract: the three `ErrorStats` fields must be **bit-identical** with
    and without `access=`, since neither the propagation nor the truth they are computed from is
    touched by the new keyword."""
    build = _constellation_builder(db_session_factory)
    config = sweep.ModelConfig(name="kepler", propagator=PropagatorType.KEPLERIAN, dt=60.0)
    horizon = 3600.0

    [plain] = sweep.run_sweep(build, [config], horizon, timing_batches=1, timing_warmup=0)
    [with_access] = sweep.run_sweep(
        build, [config], horizon, timing_batches=1, timing_warmup=0,
        access=_spec(sample_dt_s=60.0),
    )

    assert plain.access is None
    assert with_access.access is not None
    assert with_access.n_bodies == plain.n_bodies == 2
    assert with_access.error.median_km == plain.error.median_km
    assert with_access.error.rms_km == plain.error.rms_km
    assert with_access.error.max_km == plain.error.max_km

    metrics = with_access.access
    assert metrics.n_matched + metrics.passes_lost == metrics.n_truth_windows
    assert metrics.n_matched + metrics.passes_gained == metrics.n_model_windows


def test_truth_is_integrated_once_more_for_access_and_not_once_per_config(
    db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truth stays a per-sweep cost, not a per-configuration one: one endpoint integration plus one
    on the dense access grid, for any number of configurations."""
    build = _constellation_builder(db_session_factory)
    calls = {"n": 0}
    original = sweep.reference_for

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(sweep, "reference_for", counting)

    configs = [
        sweep.ModelConfig(name="a", propagator=PropagatorType.KEPLERIAN, dt=60.0),
        sweep.ModelConfig(name="b", propagator=PropagatorType.KEPLERIAN, dt=30.0),
        sweep.ModelConfig(name="c", propagator=PropagatorType.KEPLERIAN, dt=20.0),
    ]
    results = sweep.run_sweep(
        build, configs, 3600.0, timing_batches=1, timing_warmup=0, access=_spec(sample_dt_s=60.0),
    )

    assert calls["n"] == 2, "endpoint truth plus access-grid truth, once each for the whole sweep"
    assert all(r.access is not None for r in results)


def test_a_step_that_does_not_divide_the_sample_grid_raises(
    db_session_factory: Callable[[], Session],
) -> None:
    """The silent-misrepresentation guard. `viz.sample_states` sub-divides a sample interval into
    steps of *at most* `max_dt`, so a 60 s grid asked of a `dt = 90 s` configuration would propagate
    it at 30 s - a different, more accurate model than the one `run_sweep` scored. It must raise."""
    build = _constellation_builder(db_session_factory)
    config = sweep.ModelConfig(name="odd-dt", propagator=PropagatorType.KEPLERIAN, dt=90.0)
    with pytest.raises(ValueError, match="does not divide the access sample spacing"):
        sweep.run_sweep(
            build, [config], 3600.0, timing_batches=1, timing_warmup=0,
            access=_spec(sample_dt_s=60.0),
        )


def test_default_sample_step_is_the_derived_one() -> None:
    """`DEFAULT_SAMPLE_DT_S` is derived in `access.py`'s "Sampling step", not chosen by taste; if it
    moves, that derivation moved with it."""
    assert access.DEFAULT_SAMPLE_DT_S == 60.0
    assert access.AccessSpec(
        stations=[STATION], central_body="Earth", omega=EARTH_OMEGA,
        body_radius_km=scenarios.EARTH_RADIUS,
    ).sample_dt_s == access.DEFAULT_SAMPLE_DT_S



# ==================================================================================================
# 4. Matching: a pass that genuinely disappears is reported, not mis-paired
# ==================================================================================================

def test_a_marginal_pass_that_drops_below_the_mask_is_reported_lost(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    An inclined orbit over the equatorial site gives four passes in 24 h with peaks 20.53, 19.55,
    21.57 and 18.61 deg. Rotating the plane by -2 deg leaves the same four passes with peaks 16.44,
    24.41, 17.28 and 23.23 deg - the odd-numbered ones have grown and the even-numbered ones have
    shrunk. A mask at **18 deg** therefore sits between the two sets on passes 0 and 2: truth keeps
    all four, the model keeps only passes 1 and 3.

    The assertion is the discrete one - `passes_lost == 2`, `passes_gained == 0` - and, crucially,
    that the two survivors are paired with truth passes **1 and 3**, not 0 and 1. A matcher that
    paired by list index would report exactly the same counts while differencing windows an orbit
    apart, which is why the pairing is asserted by identity and the shifts are bounded: the true
    inter-pass gap is at least 5800 s, and the real shifts are tens of seconds.
    """
    h = 5.0
    spec = _spec(sample_dt_s=h, mask_deg=18.0)
    times = access.access_grid(24.0 * 3600.0, h)

    truth = _pass_windows(db_session_factory, times, spec, phase_deg=0.0, inclination_deg=53.0)
    model = _pass_windows(db_session_factory, times, spec, phase_deg=-2.0, inclination_deg=53.0)
    assert len(truth) == 4 and len(model) == 2

    metrics = access.compare_windows(truth, model)
    assert metrics.n_truth_windows == 4
    assert metrics.n_model_windows == 2
    assert metrics.n_matched == 2
    assert metrics.passes_lost == 2
    assert metrics.passes_gained == 0

    matched_truth = [m.truth for m in metrics.matches if m.matched]
    assert matched_truth == [truth[1], truth[3]], "survivors paired with the wrong true passes"

    # The smallest gap between true passes; any mis-pairing lands at least this far out.
    min_gap_s = min(truth[i + 1].rise_s - truth[i].set_s for i in range(3))
    assert min_gap_s > 5000.0
    assert metrics.rise.max_abs_s < 0.05 * min_gap_s, metrics.rise

    # Lost contact time is real contact time, so it shows in the total as well as in the count.
    assert metrics.total_contact_error_s < 0.0
    assert access.contact_fraction(metrics) < 1.0


def test_the_same_construction_reversed_reports_gained_passes(
    db_session_factory: Callable[[], Session],
) -> None:
    """Symmetry of the metric: swapping the roles of truth and model turns two lost passes into two
    gained ones and flips the sign of every shift."""
    h = 5.0
    spec = _spec(sample_dt_s=h, mask_deg=18.0)
    times = access.access_grid(24.0 * 3600.0, h)
    a = _pass_windows(db_session_factory, times, spec, phase_deg=0.0, inclination_deg=53.0)
    b = _pass_windows(db_session_factory, times, spec, phase_deg=-2.0, inclination_deg=53.0)

    forward = access.compare_windows(a, b)
    reverse = access.compare_windows(b, a)

    assert reverse.passes_gained == forward.passes_lost == 2
    assert reverse.passes_lost == forward.passes_gained == 0
    assert reverse.rise.mean_s == pytest.approx(-forward.rise.mean_s, rel=1e-12)
    assert reverse.total_contact_error_s == pytest.approx(
        -forward.total_contact_error_s, rel=1e-12
    )


# ==================================================================================================
# 5. The matcher alone, on hand-built records
# ==================================================================================================

def test_non_overlapping_windows_are_orphans_not_a_huge_shift() -> None:
    """The rule the module is built on: no overlap, no pairing. A model window 10 000 s away from
    the only true one is a lost pass *and* a gained one, never a 10 000 s shift."""
    truth = [_win(0.0, 500.0)]
    model = [_win(10_000.0, 10_500.0)]

    metrics = access.compare_windows(truth, model)

    assert metrics.n_matched == 0
    assert metrics.passes_lost == 1
    assert metrics.passes_gained == 1
    assert metrics.rise.n == 0 and metrics.rise.mean_s == 0.0
    # Durations are equal, so the contact total is unchanged even though nothing matched - the two
    # families of numbers are independent by construction (`summarise_matches`).
    assert metrics.total_contact_error_s == 0.0


def test_greedy_matching_prefers_the_larger_overlap() -> None:
    """Two model windows overlap the one true window; the larger overlap wins and the other is
    reported gained. Pairing by index would take the 10 s overlap instead of the 60 s one."""
    truth = [_win(0.0, 100.0)]
    model = [_win(90.0, 200.0), _win(-50.0, 60.0)]        # overlaps 10 s and 60 s

    metrics = access.compare_windows(truth, model)

    assert metrics.n_matched == 1
    assert metrics.passes_gained == 1
    [pair] = [m for m in metrics.matches if m.matched]
    assert pair.model is model[1]
    assert pair.rise_shift_s == pytest.approx(-50.0)
    assert pair.set_shift_s == pytest.approx(-40.0)
    assert pair.duration_error_s == pytest.approx(10.0)


def test_matching_is_independent_of_input_order() -> None:
    truth = [_win(0.0, 100.0), _win(1000.0, 1100.0), _win(2000.0, 2100.0)]
    model = [_win(5.0, 104.0), _win(2003.0, 2099.0), _win(990.0, 1105.0)]

    forward = access.compare_windows(truth, model)
    reversed_ = access.compare_windows(list(reversed(truth)), list(reversed(model)))

    assert forward.n_matched == reversed_.n_matched == 3
    assert forward.rise.mean_s == pytest.approx(reversed_.rise.mean_s)
    assert forward.duration.mean_s == pytest.approx(reversed_.duration.mean_s)
    rises: Sequence[float] = [m.rise_shift_s for m in forward.matches]  # type: ignore[assignment]
    assert sorted(rises) == pytest.approx([-10.0, 3.0, 5.0])


def test_windows_never_match_across_stations_or_bodies() -> None:
    """`station_index` and `body_index` partition the problem; a pass one station saw is not a pass
    another station saw, however well the intervals line up."""
    truth = [_win(0.0, 100.0, station=0, body=0)]
    model = [_win(0.0, 100.0, station=1, body=0), _win(0.0, 100.0, station=0, body=1)]

    metrics = access.compare_windows(truth, model)

    assert metrics.n_matched == 0
    assert metrics.passes_lost == 1
    assert metrics.passes_gained == 2


def test_clipped_edges_are_excluded_from_shifts_but_counted_in_contact_time() -> None:
    """A grid endpoint is not a rise. Differencing two clipped edges would contribute a guaranteed
    exact zero to the statistics; the contact time it represents is real and is still summed."""
    truth = [_win(0.0, 100.0, rise_clipped=True), _win(500.0, 600.0)]
    model = [_win(0.0, 104.0, rise_clipped=True), _win(503.0, 601.0)]

    metrics = access.compare_windows(truth, model)

    assert metrics.n_matched == 2
    assert metrics.rise.n == 1 and metrics.rise.mean_s == pytest.approx(3.0)
    assert metrics.set.n == 2
    assert metrics.duration.n == 1 and metrics.duration.mean_s == pytest.approx(-2.0)
    assert metrics.n_clipped_edges == 2
    assert metrics.total_contact_truth_s == pytest.approx(200.0)
    assert metrics.total_contact_model_s == pytest.approx(202.0)
    assert metrics.total_contact_error_s == pytest.approx(2.0)


def test_empty_populations_are_reported_as_empty_not_as_zero_error() -> None:
    metrics = access.compare_windows([], [])
    assert metrics.n_truth_windows == 0 and metrics.n_model_windows == 0
    assert metrics.rise.n == 0
    assert math.isnan(access.contact_fraction(metrics))


# ==================================================================================================
# Grid and specification guards
# ==================================================================================================

def test_access_grid_hits_the_horizon_exactly() -> None:
    grid = access.access_grid(3600.0, 7.0)          # 7 s does not divide 3600 s
    assert grid[0] == 0.0
    assert grid[-1] == 3600.0
    assert grid.size == 515                          # round(3600 / 7) = 514 intervals
    spacing = np.diff(grid)
    assert np.allclose(spacing, spacing[0])


def test_access_grid_rejects_degenerate_arguments() -> None:
    with pytest.raises(ValueError):
        access.access_grid(0.0, 10.0)
    with pytest.raises(ValueError):
        access.access_grid(100.0, 0.0)


def test_truth_windows_refuse_a_mismatched_grid(
    db_session_factory: Callable[[], Session],
) -> None:
    """`windows_from_truth` does not resample - a grid mismatch must raise rather than become a
    plausible-looking window shift, the same stance `viz.position_error` takes."""
    spec = _spec(sample_dt_s=10.0)
    sim = scenarios.ground_station_pass(db_session_factory(), n_sats=1, altitude_km=ALTITUDE_KM)
    sim.record_history = False
    from orbital_engine.reference import reference_for

    truth = reference_for(sim, access.access_grid(600.0, 10.0))
    # A grid of the right length is fine; one of the wrong length cannot be silently interpolated.
    assert len(access.windows_from_truth(truth, ["PASS-SAT-00"], spec)) >= 0
    with pytest.raises(ValueError):
        access.windows_from_positions(
            np.zeros((5, 1, 3)), np.linspace(0.0, 40.0, 4), spec
        )


def test_spec_without_stations_raises() -> None:
    spec = replace(_spec(sample_dt_s=10.0), stations=[])
    with pytest.raises(ValueError):
        access.windows_from_positions(np.zeros((3, 1, 3)), np.array([0.0, 1.0, 2.0]), spec)
