"""
The figure gallery: every PNG in `docs/figures/` except `frontier.png`, which `frontier_plot.py`
owns.

Run with:

    <env>/python.exe benchmarks/figures.py            # all figures
    <env>/python.exe benchmarks/figures.py ground drag # a subset, by key

This script and `frontier_plot.py` are the only places in the project that import matplotlib;
`orbital_engine.viz` prepares every array drawn here and has no plotting dependency (see its module
docstring). Backend is `Agg`, so nothing tries to open a window.

---

**1. `ground_tracks.png` - caption.** Ground tracks of a 12-satellite, 3-plane Walker constellation
at 550 km / 53 deg over 24 h, propagated with Cowell + point-mass gravity + J2 at a 60 s step and
projected onto a co-rotating Earth (omega = 7.292115e-5 rad/s about the frame +z, the same axis
`geopotential.py` and `drag.py` assume). Drawing all twelve at equal weight fills the band and reads
as noise, so the constellation is faint coverage, one satellite is picked out, and its first two
revolutions are highlighted: the arrow between their equator crossings is the per-revolution drift.
Earth rotation alone predicts 23.94 deg west per 95.5 min revolution; the measured node-to-node shift
is 24.19 deg, and the extra 0.25 deg is J2 - the node regresses westward too, and the draconitic
period is not the Keplerian one. With `j2` off the engine gives -23.94089563 deg against a predicted
-23.94089577 deg, which `tests/validation/test_viz.py` asserts. Latitude is bounded by the
inclination, so nothing reaches beyond +/-53 deg: an uncovered polar cap that no scalar error metric
would show.

**2. `error_growth.png` - caption.** Median position error over 12 satellites against a DOP853 + J2
truth over 24 h, for four model tiers, with the min-to-max spread over satellites as a band. This is
the time axis the frontier scatter collapses - it plots only each curve's final point. Final medians:
Kepler 607.61 km, secular J2 osculating-seeded 431.31 km, secular J2 mean-seeded 4.04 km, Cowell +
j2 at a 60 s step 1.88 km. The first three reproduce `frontier_plot.py`'s own numbers exactly. The
shape is the point the scatter cannot make: the two growing tiers accumulate error without bound,
while the mean-seeded curve is *flat* - its 4 km is a bounded short-period oscillation an averaged
theory cannot represent, not a drift. Cowell's curve is RK4 truncation, and grows too.

**3. `drag_decay.png` - caption.** Altitude against time for two otherwise identical 550 km / 0 deg
satellites over 20 orbits (31.8 h), one with `drag.py`'s exponential-atmosphere model enabled and one
without. Coefficients are `tests/validation/test_drag.py`'s: B = 2.2*4/260 m^2/kg from the scenario's
own vessel, rho0 = 2e-11 kg/m^3 at 550 km - roughly the real density near 300 km, chosen so the decay
is large against RK4 drift. Measured drop 3.6586 km against the orbit-averaged closed form 3.6600 km,
a ratio of 0.9996. That closed form is derived in the code below from two factors this figure exists
to make visible: the prograde co-rotation factor f = 0.8714, and the density feedback as the orbit
descends, which alone raises the mean rate 3.1% above the initial tangent -0.1775 km/orbit. The
control holds altitude to 1e-04 km. Both run Cowell; the drag one falls to the NumPy path, because
the fused compiled twin carries only point-mass gravity and J2.

**5. `atmosphere.png` - caption.** The two density laws `drag.py` can be configured with, drawn
against altitude on a log density axis, with their ratio below. The single band is matched to the
table at 355 km with H = 60 km - the "one number for LEO" choice a user actually makes. Below the
match the table is denser and increasingly so (1.07x at 320 km, 5.2x at 165 km), because every one of
its scale heights down there is under 60 km. Above the match the relationship *reverses and then
reverses again*: the table is thinner at 480 km (its H is still 58-61 km) and 2.9x denser by 830 km
(its H has grown to 125 km), crossing back through 1 between 600 and 700 km. No single exponential
reproduces that shape at any choice of H, which is the whole argument for the table. The shaded band
marks the altitudes the decay comparison in `tests/validation/test_atmosphere.py` traverses: over 3
days from 355 km at B = 0.4 m^2/kg the table predicts 116.25 km of decay against the single band's
89.20 km, a difference of 27.06 km, or 30 % more. The 28 band boundaries are the ticks on the ratio
panel - the kinks in the curve are real and are what `np.searchsorted` is selecting between.

**6. `access_windows.png` - caption.** The same 12-satellite constellation and the same four tiers as
`error_growth.png`, but the error is in **contact windows** rather than kilometres: rise-time shift
against a DOP853 + J2 truth for every pass over three stations (Kiruna, Wallops, Santiago) at a 5 deg
mask over 24 h - 189 true passes. Svalbard is deliberately absent: a 53 deg orbit at 550 km has a
23.0 deg horizon half-angle and a sub-point bounded by +/-53 deg, so a site above ~76 deg never sees
it at all. Both panels report the *rise* shift; set and duration behave the same way and are in
`access.AccessMetrics`.

Mean |rise shift| / max |rise shift| / lost / gained: Kepler 31.65 s / 154.21 s / 5 / 4; secular J2
osculating-seeded 30.82 s / 115.82 s / 1 / 2; secular J2 mean-seeded 1.60 s / 8.10 s / 1 / 0;
Cowell + j2 at 60 s **0.063 s / 0.223 s / 0 / 0**. Against a 1 s threshold - roughly the acquisition
pad a real schedule already carries - **Cowell + J2 is the first tier that clears it**, and the only
one that neither invents nor loses a pass.

Three things the kilometre metric cannot say. First, the **discrete failures**: Kepler does not merely
mistime its passes, it deletes five that happen and predicts four that do not, and those nine are
scheduling decisions rather than error bars. Second, **ranking changes**: mean-seeded secular J2 is
100x better than Kepler in kilometres (4.04 km against 607.61 km in `error_growth.png`) but only 20x
better in mean window shift, and it still drops a marginal pass - an averaged theory reproduces the
along-track position far better than it reproduces the *elevation profile* near the horizon, which is
where a marginal pass lives. Third, **the two secular-J2 seedings are indistinguishable in the scatter
above ~10 s** while being two orders of magnitude apart at the horizon: the osculating-seeded tier's
shift is a coherent drift, and the top panel shows it growing from seconds to a minute over the day
whereas the mean-seeded tier's is a bounded short-period wobble that never trends.

Cowell's 0.223 s maximum is the sanity check on the whole measurement. Its position error after 24 h
is 1.88 km (`error_growth.png`), and a 1.88 km along-track displacement moves a pass by
`(1.88 / 6921) / Omega = 0.265 s` at `Omega = n - omega = 1.024e-3 rad/s`. Measured 0.223 s, i.e.
the along-track share of a 1.88 km error, as it should be. The residual edge-interpolation bias at
this 60 s grid is bounded by `C Delta h = 1.6e-2 s` (`access.py`), so the number is signal, not
sampling.

**4. `hierarchy.png` - caption.** `sun_earth_moon` over 60 days, drawn in the frame that makes the
hierarchy visible: relative to the Earth-Moon barycentre. Neither body's `parent_indices` parent is
the other - both are measured about the barycentre, which is itself the body carrying the
heliocentric ellipse. The Earth also *heads* that system, so its own COE row is deliberately zeroed
(`_rehydrate_coes`): its motion is not an orbit but the reflex kick, r = 4697 km, entirely inside its
own 6371 km surface, which is why the right panel is an 81x zoom. The measured radius ratio is the
mass ratio to machine precision - 0.01230463 against mu_Moon/mu_Earth = 0.01230463, a relative
difference of 8e-15 - which is the barycentric model's defining invariant, drawn rather than
asserted.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import access, scenarios, sweep, viz
from orbital_engine.atmosphere import (
    BASE_ALTITUDE_KM, DENSITY_MODEL_EXPONENTIAL, DENSITY_MODEL_LAYERED, layered_density,
)
from orbital_engine.custom_types import PropagatorType
from orbital_engine.database import Base
from orbital_engine.drag import DRAG_MODEL, EARTH_OMEGA
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ, J2_MODEL
from orbital_engine.gravity import POINT_MASS_MODEL
from orbital_engine.kernels import NUMBA_AVAILABLE
from orbital_engine.reference import TRUTH_ATOL, TRUTH_RTOL, reference_for
from orbital_engine.simulator import Simulation

ArrF = NDArray[np.float64]

FIGURE_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"

# Shared colours, so a tier reads the same across this gallery and `frontier_plot.py`.
TIER_COLOURS = {
    "Kepler": "tab:red",
    "Secular J2 (osculating-seeded)": "tab:orange",
    "Secular J2 (mean-seeded)": "tab:green",
    "Cowell + j2 (dt=60 s)": "tab:blue",
}


def fresh_session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _sat_slots(sim: Simulation) -> List[int]:
    return sorted(i for n, i in sim.name_to_index.items() if n.startswith("SAT-"))


def _save(fig: "plt.Figure", name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def _break_at_antimeridian(lon_deg: ArrF, lat_deg: ArrF) -> tuple[ArrF, ArrF]:
    """
    Insert NaN wherever the longitude wraps, so a line plot does not draw a horizontal streak across
    the whole map. A jump of more than 180 deg between consecutive samples can only be the wrap:
    the satellite moves under 4 deg of longitude per 60 s sample at this altitude.
    """
    lon = lon_deg.astype(np.float64).copy()
    lat = lat_deg.astype(np.float64).copy()
    wrap = np.abs(np.diff(lon)) > 180.0
    lon[:-1][wrap] = np.nan
    lat[:-1][wrap] = np.nan
    return lon, lat


def _ascending_node_longitudes(lon_deg: ArrF, lat_deg: ArrF) -> List[float]:
    """
    Longitude of each northbound equator crossing, linearly interpolated between the bracketing
    samples. Used only to annotate the measured per-revolution drift on the ground-track figure, so
    linear interpolation over a 60 s sample is ample - the latitude moves under 4 deg in that time.
    """
    out: List[float] = []
    rising = np.flatnonzero((lat_deg[:-1] < 0.0) & (lat_deg[1:] >= 0.0))
    for k in rising:
        if abs(lon_deg[k + 1] - lon_deg[k]) > 180.0:      # a wrap, not a crossing worth annotating
            continue
        frac = -lat_deg[k] / (lat_deg[k + 1] - lat_deg[k])
        out.append(float(lon_deg[k] + frac * (lon_deg[k + 1] - lon_deg[k])))
    return out


# ==================================================================================================
# 1. Ground tracks
# ==================================================================================================

GROUND_N_SATS = 12
GROUND_N_PLANES = 3
GROUND_ALT_KM = 550.0
GROUND_INC_DEG = 53.0
GROUND_HORIZON_S = 86400.0
GROUND_DT_S = 60.0


def figure_ground_tracks() -> None:
    sim = scenarios.earth_constellation(
        fresh_session(), n_sats=GROUND_N_SATS, n_planes=GROUND_N_PLANES,
        altitude_km=GROUND_ALT_KM, inclination_deg=GROUND_INC_DEG,
    )
    sim.record_history = False
    slots = _sat_slots(sim)
    earth = sim.name_to_index["Earth"]
    for slot in slots:
        sim.set_propagator(slot, PropagatorType.COWELL)
        sim.enable_force_model(POINT_MASS_MODEL, bodies=slot)
        sim.enable_force_model(J2_MODEL, bodies=slot, j2=EARTH_J2, r_eq=EARTH_R_EQ)

    times = np.arange(0.0, GROUND_HORIZON_S + 0.5 * GROUND_DT_S, GROUND_DT_S)
    states = viz.sample_states(sim, slots, times, relative_to=earth, max_dt=GROUND_DT_S)
    track = viz.ground_track(
        states[..., :3], times, omega=EARTH_OMEGA, body_radius_km=EARTH_R_EQ
    )

    period_s = 2.0 * math.pi * math.sqrt(
        (scenarios.EARTH_RADIUS + GROUND_ALT_KM) ** 3 / scenarios.MU_EARTH
    )
    print(f"  period {period_s:.1f} s, westward drift per orbit "
          f"{math.degrees(EARTH_OMEGA * period_s):.2f} deg")
    print(f"  latitude range {track.latitude_deg.min():+.2f} .. {track.latitude_deg.max():+.2f} deg "
          f"(inclination {GROUND_INC_DEG:.1f} deg)")
    print(f"  mean altitude {track.altitude_km.mean():.2f} km, "
          f"peak-to-peak {np.ptp(track.altitude_km):.3f} km")
    nodes_all = _ascending_node_longitudes(track.longitude_deg[:, 0], track.latitude_deg[:, 0])
    if len(nodes_all) >= 2:
        print(f"  measured node-to-node shift {nodes_all[1] - nodes_all[0]:+.2f} deg "
              f"(predicted {-math.degrees(EARTH_OMEGA * period_s):+.2f} deg, J2 nodal regression "
              f"included in the measurement)")

    # Drawing all twelve 24 h tracks at equal weight fills the whole band and reads as noise: fifteen
    # revolutions x twelve satellites is 180 crossing curves. The constellation is therefore drawn as
    # faint coverage, one satellite is picked out for the shape of a single day, and its first two
    # revolutions are highlighted so the per-orbit westward shift is a thing you can see and measure
    # off the axis rather than a claim in the caption.
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    for k in range(len(slots)):
        lon, lat = _break_at_antimeridian(track.longitude_deg[:, k], track.latitude_deg[:, k])
        ax.plot(lon, lat, lw=0.5, alpha=0.30, color="0.55", zorder=1,
                label="all 12 satellites, 24 h" if k == 0 else None)

    lon0, lat0 = _break_at_antimeridian(track.longitude_deg[:, 0], track.latitude_deg[:, 0])
    ax.plot(lon0, lat0, lw=0.9, alpha=0.9, color="tab:blue", zorder=2,
            label="SAT-00-000, 24 h (15.1 revolutions)")

    n_per_rev = int(round(period_s / GROUND_DT_S))
    for rev, colour in ((0, "tab:red"), (1, "tab:orange")):
        sl = slice(rev * n_per_rev, (rev + 1) * n_per_rev + 1)
        lon_r, lat_r = _break_at_antimeridian(
            track.longitude_deg[sl, 0], track.latitude_deg[sl, 0]
        )
        ax.plot(lon_r, lat_r, lw=2.4, color=colour, zorder=3, label=f"revolution {rev + 1}")

    # The two highlighted revolutions cross the equator northbound one orbit apart; the gap between
    # those two longitudes is the measured drift, annotated at the crossing itself.
    nodes = _ascending_node_longitudes(track.longitude_deg[:, 0], track.latitude_deg[:, 0])
    if len(nodes) >= 2:
        ax.annotate(
            "", xy=(nodes[1], 0.0), xytext=(nodes[0], 0.0),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.4), zorder=5,
        )
        ax.text(0.5 * (nodes[0] + nodes[1]), 4.0,
                f"{abs(nodes[1] - nodes[0]):.2f} deg west per revolution",
                ha="center", va="bottom", fontsize=9, zorder=5,
                bbox=dict(fc="white", ec="0.7", alpha=0.9, pad=2.0))

    ax.axhline(GROUND_INC_DEG, color="0.3", ls="--", lw=0.9, zorder=4)
    ax.axhline(-GROUND_INC_DEG, color="0.3", ls="--", lw=0.9, zorder=4)
    ax.text(-178, GROUND_INC_DEG + 3, f"latitude bound = inclination = {GROUND_INC_DEG:.0f} deg",
            ha="left", va="bottom", fontsize=9, color="0.25")
    ax.text(-178, 72, "no coverage above 53 deg: a property of the constellation, not of the engine",
            ha="left", va="center", fontsize=8.5, color="0.35", style="italic")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.set_xlabel("geocentric longitude (deg east)")
    ax.set_ylabel("geocentric latitude (deg)")
    ax.set_title(
        f"Ground tracks: earth_constellation, {GROUND_N_SATS} sats / {GROUND_N_PLANES} planes, "
        f"{GROUND_ALT_KM:.0f} km / {GROUND_INC_DEG:.0f} deg, 24 h\n"
        f"Cowell + point_mass_gravity + j2, dt = {GROUND_DT_S:.0f} s, co-rotating Earth"
    )
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8, ncol=3, framealpha=0.9)
    measured_shift = (nodes_all[1] - nodes_all[0]) if len(nodes_all) >= 2 else float("nan")
    fig.text(
        0.01, 0.005,
        f"Earth rotation alone predicts {-math.degrees(EARTH_OMEGA * period_s):.2f} deg per "
        f"{period_s / 60:.1f} min revolution; the measured node-to-node shift is "
        f"{measured_shift:.2f} deg.\nThe extra {abs(measured_shift) - math.degrees(EARTH_OMEGA * period_s):.2f} "
        f"deg is J2: the orbit plane's node regresses westward too, and the draconitic period is not "
        f"the Keplerian one. Turn j2 off and the two agree.",
        fontsize=8, va="bottom",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "ground_tracks.png")


# ==================================================================================================
# 2. Error growth over time
# ==================================================================================================

ERROR_N_SATS = 12
ERROR_HORIZON_S = 86400.0
ERROR_SAMPLES = 97                    # every 900 s
ERROR_COWELL_DT_S = 60.0


def _error_scenario() -> Simulation:
    sim = scenarios.earth_constellation(
        fresh_session(), n_sats=ERROR_N_SATS, n_planes=1, altitude_km=550.0, inclination_deg=53.0
    )
    sim.record_history = False
    return sim


def figure_error_growth() -> None:
    times = np.linspace(0.0, ERROR_HORIZON_S, ERROR_SAMPLES)
    truth = reference_for(
        _error_scenario(), times, rtol=TRUTH_RTOL, atol=TRUTH_ATOL,
        oblateness={"Earth": (EARTH_J2, EARTH_R_EQ)},
    )

    # (propagator, mean_seed, force models, max_dt) per tier - the same data `sweep.ModelConfig`
    # carries, held here as a literal because this script draws tiers rather than sweeping them.
    tiers: Dict[str, Callable[[Simulation, List[int]], None]] = {}

    def _kepler(sim: Simulation, slots: List[int]) -> None:
        return None  # Keplerian is the default propagator; nothing to configure.

    def _secular(mean_seed: bool) -> Callable[[Simulation, List[int]], None]:
        def _apply(sim: Simulation, slots: List[int]) -> None:
            sim.set_propagator(slots, PropagatorType.SECULAR_J2, mean_seed,
                               j2=EARTH_J2, r_eq=EARTH_R_EQ)
        return _apply

    def _cowell(sim: Simulation, slots: List[int]) -> None:
        sim.set_propagator(slots, PropagatorType.COWELL)
        sim.enable_force_model(POINT_MASS_MODEL, bodies=slots)
        sim.enable_force_model(J2_MODEL, bodies=slots, j2=EARTH_J2, r_eq=EARTH_R_EQ)

    tiers["Kepler"] = _kepler
    tiers["Secular J2 (osculating-seeded)"] = _secular(False)
    tiers["Secular J2 (mean-seeded)"] = _secular(True)
    tiers["Cowell + j2 (dt=60 s)"] = _cowell

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    for name, configure in tiers.items():
        sim = _error_scenario()
        slots = _sat_slots(sim)
        configure(sim, slots)
        slot_to_name = {slot: n for n, slot in sim.name_to_index.items()}
        names = [slot_to_name[s] for s in slots]

        max_dt = ERROR_COWELL_DT_S if name.startswith("Cowell") else None
        states = viz.sample_states(sim, slots, times, max_dt=max_dt)
        curve = viz.error_curve(states[..., :3], truth, names)

        print(f"  {name:<34} final median {curve.median_km[-1]:>10.4f} km   "
              f"max {curve.max_km[-1]:>10.4f} km")
        ax.plot(times / 3600.0, curve.median_km, lw=1.8, color=TIER_COLOURS[name], label=name)
        ax.fill_between(times / 3600.0, curve.per_body_km.min(axis=1), curve.max_km,
                        color=TIER_COLOURS[name], alpha=0.12, lw=0)

    ax.set_yscale("log")
    ax.set_xlim(0.0, ERROR_HORIZON_S / 3600.0)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("position error vs DOP853 + J2 truth (km, log scale)")
    ax.set_title(
        f"Error growth by model tier: earth_constellation, 550 km / 53 deg, "
        f"{ERROR_N_SATS} sats, 24 h\nline = median over satellites, band = min-to-max spread"
    )
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.text(
        0.01, 0.005,
        "The frontier plot reports only each curve's final point. The band shows why the statistic is "
        "taken over bodies:\ninitial phase alone spreads a single satellite's error across a decade.",
        fontsize=8, va="bottom",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "error_growth.png")


# ==================================================================================================
# 3. Drag decay
# ==================================================================================================

DRAG_ALT_KM = 550.0
DRAG_A0_KM = scenarios.EARTH_RADIUS + DRAG_ALT_KM
DRAG_COEFFS = dict(
    ballistic_coeff=2.2 * 4.0 / 260.0,      # C_d A / m for the scenario's own vessel
    rho0=2e-11,
    h0=DRAG_A0_KM - EARTH_R_EQ,             # so rho(a0) == rho0
    scale_height=60.0,
    r_ref=EARTH_R_EQ,
)
DRAG_DT_S = 20.0
DRAG_N_ORBITS = 20


def figure_drag_decay() -> None:
    sim = scenarios.earth_constellation(
        fresh_session(), n_sats=2, n_planes=1, altitude_km=DRAG_ALT_KM, inclination_deg=0.0
    )
    sim.record_history = False
    slots = _sat_slots(sim)
    earth = sim.name_to_index["Earth"]
    for slot in slots:
        sim.set_propagator(slot, PropagatorType.COWELL)
        sim.enable_force_model(POINT_MASS_MODEL, bodies=slot)
    # Only the first satellite feels the atmosphere; the second is the control.
    sim.enable_force_model(DRAG_MODEL, bodies=slots[0], omega=EARTH_OMEGA, **DRAG_COEFFS)

    period_s = 2.0 * math.pi * math.sqrt(DRAG_A0_KM ** 3 / scenarios.MU_EARTH)
    horizon = DRAG_N_ORBITS * period_s
    times = np.arange(0.0, horizon + 0.5 * DRAG_DT_S, DRAG_DT_S)
    states = viz.sample_states(sim, slots, times, relative_to=earth, max_dt=DRAG_DT_S)
    altitude = viz.altitude_series(states[..., :3], body_radius_km=EARTH_R_EQ)

    # Orbit-averaged decay for a circular orbit in an exponential atmosphere, derived here
    # independently of the kernel (`tests/validation/test_drag.py` derives the same two factors):
    #
    #   da/dN = -2 pi rho B' a^2 * f,   B' = rho * B converted from 1/m to 1/km (the kernel's x1e3),
    #
    # with `f = (1 - omega a / v)^2` the co-rotation factor for a *prograde equatorial* orbit, where
    # the atmosphere's velocity is along-track. Holding `a^2` fixed but letting the density respond to
    # the descent, `rho = rho0 exp((a0 - a)/H)`, integrates in closed form to
    #
    #   a(N) = a0 + H ln(1 - c N / H),   c = 2 pi rho0 B' a0^2 f,
    #
    # and that density feedback is not a nicety: over 20 orbits it raises the mean rate 3.1% above
    # the initial-tangent `c`, which is exactly the gap a straight line would have shown as error.
    v_circ = math.sqrt(scenarios.MU_EARTH / DRAG_A0_KM)
    corotation_f = (1.0 - EARTH_OMEGA * DRAG_A0_KM / v_circ) ** 2
    rate0_per_orbit = -2.0 * math.pi * (
        DRAG_COEFFS["rho0"] * DRAG_COEFFS["ballistic_coeff"] * 1e3
    ) * DRAG_A0_KM ** 2 * corotation_f
    scale_h = DRAG_COEFFS["scale_height"]
    orbits = times / period_s
    predicted_alt = altitude[0, 0] + scale_h * np.log(1.0 + rate0_per_orbit * orbits / scale_h)

    measured_total = float(altitude[-1, 0] - altitude[0, 0])
    predicted_total = float(predicted_alt[-1] - predicted_alt[0])
    control_drift = float(altitude[-1, 1] - altitude[0, 1])
    print(f"  period {period_s:.1f} s, horizon {horizon / 3600:.2f} h, "
          f"fused compiled path: {sim._cowell_fused_ok}")
    print(f"  co-rotation factor f = {corotation_f:.4f}, initial rate {rate0_per_orbit:+.4f} km/orbit")
    print(f"  total drop {measured_total:+.4f} km over {DRAG_N_ORBITS} orbits; closed form "
          f"{predicted_total:+.4f} km (ratio {measured_total / predicted_total:.5f})")
    print(f"  mean rate {measured_total / DRAG_N_ORBITS:+.4f} km/orbit; control drift "
          f"{control_drift:+.2e} km")

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    hours = times / 3600.0
    ax.plot(hours, altitude[:, 1], lw=1.4, color="tab:green", label="no drag (control)")
    ax.plot(hours, altitude[:, 0], lw=1.4, color="tab:blue", label="drag enabled")
    ax.plot(hours, predicted_alt, lw=1.2, ls="--", color="0.35",
            label=f"orbit-averaged closed form, $c$ = {rate0_per_orbit:.4f} km/orbit")

    ax.set_xlim(0.0, hours[-1])
    ax.set_xlabel("time (h)")
    ax.set_ylabel("geocentric altitude (km)")
    ax.set_title(
        f"Drag decay: two 550 km / 0 deg satellites, {DRAG_N_ORBITS} orbits "
        f"({horizon / 3600:.1f} h), Cowell dt = {DRAG_DT_S:.0f} s\n"
        f"exponential atmosphere, B = {DRAG_COEFFS['ballistic_coeff']:.4f} m$^2$/kg, "
        f"$\\rho_0$ = {DRAG_COEFFS['rho0']:.0e} kg/m$^3$, H = {DRAG_COEFFS['scale_height']:.0f} km"
    )
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9)
    fig.text(
        0.01, 0.005,
        f"Measured drop {measured_total:.4f} km against the independently derived orbit-averaged closed "
        f"form {predicted_total:.4f} km, a ratio of {measured_total / predicted_total:.4f}.\n"
        f"The closed form carries the prograde co-rotation factor f = {corotation_f:.4f} and the density "
        f"feedback as the orbit descends; the control holds altitude to {abs(control_drift):.0e} km.",
        fontsize=8, va="bottom",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "drag_decay.png")


# ==================================================================================================
# 4. The hierarchy, drawn
# ==================================================================================================

HIER_HORIZON_S = 60.0 * 86400.0
HIER_SAMPLES = 2401


def figure_hierarchy() -> None:
    sim = scenarios.sun_earth_moon(fresh_session())
    sim.record_history = False
    earth = sim.name_to_index["Earth"]
    moon = sim.name_to_index["Moon"]
    bary = sim.body_sys_map[moon]

    times = np.linspace(0.0, HIER_HORIZON_S, HIER_SAMPLES)
    # Barycentre-relative: the frame in which the hierarchy is visible at all. Both bodies orbit the
    # same point, and the Earth's share of that motion is the reflex kick the head carries.
    states = viz.sample_states(sim, [moon, earth], times, relative_to=int(bary))
    moon_xy, earth_xy = states[:, 0, :3], states[:, 1, :3]

    moon_r = np.linalg.norm(moon_xy, axis=1)
    reflex_r = np.linalg.norm(earth_xy, axis=1)
    ratio = float(np.mean(reflex_r / moon_r))
    mass_ratio = scenarios.MU_MOON / scenarios.MU_EARTH
    print(f"  Moon barycentric range {moon_r.min():.0f} .. {moon_r.max():.0f} km")
    print(f"  Earth reflex radius {reflex_r.min():.1f} .. {reflex_r.max():.1f} km "
          f"(Earth's own radius {scenarios.EARTH_RADIUS:.0f} km, so the wobble stays inside it)")
    print(f"  radius ratio {ratio:.6f} against the mass ratio mu_Moon/mu_Earth = {mass_ratio:.6f} "
          f"(rel. diff {abs(ratio - mass_ratio) / mass_ratio:.2e})")
    print(f"  Earth COE eccentricity row = {sim.coe_states[earth, 1]:.1f} (zeroed: it is a system head)")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.8))

    ax_l.plot(moon_xy[:, 0], moon_xy[:, 1], lw=0.9, color="tab:blue",
              label="Moon about the barycentre")
    ax_l.plot(earth_xy[:, 0], earth_xy[:, 1], lw=1.4, color="tab:green",
              label="Earth about the same barycentre")
    ax_l.plot([earth_xy[0, 0], moon_xy[0, 0]], [earth_xy[0, 1], moon_xy[0, 1]],
              lw=1.0, ls=":", color="0.35", zorder=3,
              label="Earth-Moon line at t = 0: always opposed")
    ax_l.scatter([0.0], [0.0], s=55, marker="x", color="tab:orange", zorder=4,
                 label="Earth-Moon barycentre (the Keplerian parent of both)")
    ax_l.set_aspect("equal")
    ax_l.set_xlabel("x (km, relative to the Earth-Moon barycentre)")
    ax_l.set_ylabel("y (km)")
    ax_l.text(0.02, 0.03,
              f"the Earth's own path is {moon_r.mean() / reflex_r.mean():.0f}x smaller than the Moon's\n"
              f"and vanishes into the marker at this scale - see right",
              transform=ax_l.transAxes, fontsize=8, color="0.35", style="italic")
    ax_l.set_title("Both bodies orbit the barycentre, 60 days")
    ax_l.grid(True, ls=":", alpha=0.5)
    ax_l.legend(loc="upper right", fontsize=8)

    ax_r.plot(earth_xy[:, 0], earth_xy[:, 1], lw=1.4, color="tab:green",
              label="Earth about the barycentre")
    ax_r.add_patch(plt.Circle((0.0, 0.0), scenarios.EARTH_RADIUS, fill=False, ls="--", lw=1.1,
                              color="0.4"))
    ax_r.scatter([0.0], [0.0], s=55, marker="x", color="tab:orange", zorder=4,
                 label="Earth-Moon barycentre")
    ax_r.set_aspect("equal")
    ax_r.set_xlabel("x (km, relative to the barycentre)")
    ax_r.set_ylabel("y (km)")
    ax_r.set_title(
        f"Zoom, x{moon_r.mean() / reflex_r.mean():.0f}: the reflex kick, r = {reflex_r.mean():.0f} km\n"
        f"(dashed circle: the Earth's own {scenarios.EARTH_RADIUS:.0f} km surface)"
    )
    ax_r.grid(True, ls=":", alpha=0.5)
    ax_r.legend(loc="upper right", fontsize=8)

    fig.suptitle("Why the two parent graphs diverge: sun_earth_moon, 60 days", fontsize=13)
    fig.text(
        0.01, 0.005,
        f"Neither body's Keplerian parent (parent_indices) is the other: both are measured about the "
        f"barycentre, which is itself the body carrying the heliocentric ellipse.\n"
        f"The Earth also *heads* that system, so its own COE row is deliberately zeroed - its motion is "
        f"not an orbit but the reflex kick on the right, r = {reflex_r.mean():.0f} km,\n"
        f"entirely inside its own surface. The radius ratio is the mass ratio: "
        f"{ratio:.6f} against mu_Moon/mu_Earth = {mass_ratio:.6f}.",
        fontsize=8, va="bottom",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.96))
    _save(fig, "hierarchy.png")


# ==================================================================================================
# 5. atmosphere.png
# ==================================================================================================

# The decay comparison's scenario, mirrored from `tests/validation/test_atmosphere.py` so the figure
# and the test describe the same choice. Nothing here re-derives physics; the test owns that.
ATM_MATCH_KM = 355.0
ATM_SINGLE_H_KM = 60.0
ATM_DECAY_LOW_KM = 238.75     # where the layered satellite ends after 3 days
ATM_DECAY_LAYERED_KM = -116.25
ATM_DECAY_SINGLE_KM = -89.20


def figure_atmosphere() -> None:
    # From 100 km up: below that the two laws differ by factors of 1e9, which would flatten the
    # ratio panel into a straight line, and nothing orbits there anyway.
    altitude = np.arange(100.0, 1000.01, 0.5)
    table = layered_density(altitude)
    rho_match = float(layered_density(np.array([ATM_MATCH_KM]))[0])
    single: ArrF = rho_match * np.exp(-(altitude - ATM_MATCH_KM) / ATM_SINGLE_H_KM)
    ratio = table / single

    # Two crossings: the match itself, and the one above it where the table overtakes the single
    # band again as its scale heights grow past 60 km. The second is the interesting one.
    crossings = altitude[np.flatnonzero(np.diff(np.signbit(ratio - 1.0)))]
    reversal = float(crossings[-1])
    print(f"  matched at {ATM_MATCH_KM:.0f} km: rho = {rho_match:.4e} kg/m^3, "
          f"agreement {abs(ratio[np.argmin(abs(altitude - ATM_MATCH_KM))] - 1.0):.2e}")
    print(f"  ratio crossings through 1: {np.round(crossings, 1).tolist()} km")
    print(f"  ratio at 165 / 320 / 480 / 830 km: "
          f"{[round(float(ratio[np.argmin(abs(altitude - h))]), 3) for h in (165, 320, 480, 830)]}")

    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(9.0, 7.4), sharex=True, gridspec_kw={"height_ratios": [2.4, 1.0]})

    for axis in (ax, ax_r):
        axis.axvspan(ATM_DECAY_LOW_KM, ATM_MATCH_KM, color="0.85", zorder=0)
    ax.semilogy(altitude, table, lw=1.5, color="tab:blue",
                label="layered: Vallado Table 8-4, 28 bands")
    ax.semilogy(altitude, single, lw=1.5, color="tab:red", ls="--",
                label=f"single band, matched at {ATM_MATCH_KM:.0f} km, H = {ATM_SINGLE_H_KM:.0f} km")
    ax.scatter([ATM_MATCH_KM], [rho_match], s=45, zorder=4, color="k", label="the matched altitude")
    ax.set_ylabel("density (kg/m$^3$)")
    ax.set_title("Two density laws the `drag` model can be configured with")
    ax.grid(True, ls=":", alpha=0.5, which="both")
    ax.legend(loc="upper right", fontsize=9)

    ax_r.plot(altitude, ratio, lw=1.5, color="tab:purple")
    ax_r.axhline(1.0, lw=1.0, color="0.4", ls="--")
    ax_r.scatter(crossings, np.ones_like(crossings), s=35, zorder=4, color="tab:purple")
    for base in BASE_ALTITUDE_KM[BASE_ALTITUDE_KM <= 1000.0]:
        ax_r.axvline(float(base), lw=0.6, color="0.75", zorder=0)
    ax_r.set_yscale("log")
    ax_r.set_xlabel("altitude above 6378.137 km (km)")
    ax_r.set_ylabel("layered / single band")
    ax_r.set_xlim(100.0, 1000.0)
    ax_r.set_ylim(0.5, 20.0)
    ax_r.grid(True, ls=":", alpha=0.5, which="both")

    fig.text(
        0.01, 0.005,
        f"Shaded: the altitudes the 3-day decay comparison in "
        f"tests/validation/test_atmosphere.py traverses.\n"
        f"It measures what the choice costs: "
        f"{ATM_DECAY_LAYERED_KM:.2f} km of decay under the table\n"
        f"against {ATM_DECAY_SINGLE_KM:.2f} km under the single band, a difference of "
        f"{ATM_DECAY_LAYERED_KM - ATM_DECAY_SINGLE_KM:.2f} km "
        f"({100 * (ATM_DECAY_LAYERED_KM / ATM_DECAY_SINGLE_KM - 1):.0f} % more). Thin vertical lines "
        f"are the 28 band boundaries.\n"
        f"Above the match the ratio crosses 1 again at {reversal:.0f} km: below the match every "
        f"table scale height is under 60 km,\n"
        f"above it they grow past 60 km - so a single band is wrong in both directions, and in "
        f"opposite senses.",
        fontsize=8, va="bottom",
    )
    fig.tight_layout(rect=(0, 0.085, 1, 1.0))
    _save(fig, "atmosphere.png")



# ==================================================================================================
# 6. Access-window error by model tier
# ==================================================================================================

ACCESS_HORIZON_S = 86400.0
ACCESS_SAMPLE_DT_S = 60.0             # must be a multiple of every tier's dt - see access.py
ACCESS_DT_S = 60.0                    # the same step for every tier, so the constraint holds for all
ACCESS_MASK_DEG = 5.0

# Three sites a 53 deg orbit actually reaches. A 550 km satellite's horizon half-angle is 23.0 deg
# and its sub-point never leaves +/-53 deg, so a station north of about 76 deg would see nothing at
# all - Svalbard is the wrong site for this constellation and is deliberately not in the list.
ACCESS_STATIONS = [
    access.GroundStation("Kiruna", math.radians(67.86), math.radians(20.96), 0.40),
    access.GroundStation("Wallops", math.radians(37.94), math.radians(-75.46), 0.01),
    access.GroundStation("Santiago", math.radians(-33.15), math.radians(-70.67), 0.73),
]

# The threshold the headline is stated against. A ground-station schedule already carries a few
# seconds of acquisition pad, so a model whose windows land within ~1 s of truth changes no
# decision; one out by tens of seconds eats the pad, and one that loses a pass changes the plan.
ACCESS_THRESHOLD_S = 1.0


def _access_configs() -> Dict[str, sweep.ModelConfig]:
    j2 = {"j2": EARTH_J2, "r_eq": EARTH_R_EQ}
    return {
        "Kepler": sweep.ModelConfig(
            name="Kepler", propagator=PropagatorType.KEPLERIAN, dt=ACCESS_DT_S),
        "Secular J2 (osculating-seeded)": sweep.ModelConfig(
            name="Secular J2 (osculating-seeded)", propagator=PropagatorType.SECULAR_J2,
            dt=ACCESS_DT_S, propagator_coefficients=j2),
        "Secular J2 (mean-seeded)": sweep.ModelConfig(
            name="Secular J2 (mean-seeded)", propagator=PropagatorType.SECULAR_J2,
            dt=ACCESS_DT_S, mean_seed=True, propagator_coefficients=j2),
        "Cowell + j2 (dt=60 s)": sweep.ModelConfig(
            name="Cowell + j2 (dt=60 s)", propagator=PropagatorType.COWELL, dt=ACCESS_DT_S,
            force_models=(sweep.ForceModelSpec(POINT_MASS_MODEL),
                          sweep.ForceModelSpec(J2_MODEL, j2))),
    }


def figure_access_windows() -> None:
    spec = access.AccessSpec(
        stations=ACCESS_STATIONS,
        central_body="Earth",
        omega=EARTH_OMEGA,
        body_radius_km=scenarios.EARTH_RADIUS,
        mask_angle_rad=math.radians(ACCESS_MASK_DEG),
        sample_dt_s=ACCESS_SAMPLE_DT_S,
    )
    grid = access.access_grid(ACCESS_HORIZON_S, spec.sample_dt_s)
    truth = reference_for(
        _error_scenario(), grid, rtol=TRUTH_RTOL, atol=TRUTH_ATOL,
        oblateness={"Earth": (EARTH_J2, EARTH_R_EQ)},
    )

    metrics: Dict[str, access.AccessMetrics] = {}
    for name, config in _access_configs().items():
        metrics[name] = sweep.access_metrics_for(
            _error_scenario, config, ACCESS_HORIZON_S, spec, truth
        )
        print("  " + access.format_metrics(name, metrics[name]))

    fig, (ax_scatter, ax_bar) = plt.subplots(
        2, 1, figsize=(10.0, 8.5), gridspec_kw={"height_ratios": [2.0, 1.0]}
    )

    for name, m in metrics.items():
        times_h = [pair.truth.rise_s / 3600.0 for pair in m.matches
                   if pair.rise_shift_s is not None and pair.truth is not None]
        shifts = [pair.rise_shift_s for pair in m.matches if pair.rise_shift_s is not None]
        ax_scatter.scatter(times_h, shifts, s=9, alpha=0.55, color=TIER_COLOURS[name], label=name)
    ax_scatter.axhspan(-ACCESS_THRESHOLD_S, ACCESS_THRESHOLD_S, color="0.85", zorder=0)
    ax_scatter.axhline(0.0, color="0.4", lw=0.8)
    ax_scatter.set_yscale("symlog", linthresh=1.0)
    ax_scatter.set_xlim(0.0, ACCESS_HORIZON_S / 3600.0)
    ax_scatter.set_xlabel("time of the true rise (h)")
    ax_scatter.set_ylabel("rise-time shift vs truth (s)\npositive = model rises late")
    ax_scatter.set_title(
        f"Contact-window error by model tier: {ERROR_N_SATS} sats at 550 km / 53 deg, "
        f"{len(ACCESS_STATIONS)} stations, {ACCESS_MASK_DEG:.0f} deg mask, 24 h\n"
        f"shaded band = +/-{ACCESS_THRESHOLD_S:.0f} s, the acquisition pad a schedule already carries"
    )
    ax_scatter.grid(True, which="both", ls=":", alpha=0.5)
    ax_scatter.legend(loc="upper left", fontsize=8, markerscale=1.8)

    names = list(metrics)
    y = np.arange(len(names), dtype=np.float64)
    mean_abs = np.array([metrics[n].rise.mean_abs_s for n in names])
    max_abs = np.array([metrics[n].rise.max_abs_s for n in names])
    ax_bar.barh(y, max_abs, height=0.62, color=[TIER_COLOURS[n] for n in names], alpha=0.30,
                label="max |rise shift|")
    ax_bar.barh(y, mean_abs, height=0.62, color=[TIER_COLOURS[n] for n in names],
                label="mean |rise shift|")
    ax_bar.axvline(ACCESS_THRESHOLD_S, color="k", ls="--", lw=1.2)
    ax_bar.set_xscale("log")
    # Room on the right for the per-tier annotations, which are the discrete half of the story.
    ax_bar.set_xlim(0.5 * float(mean_abs.min()), 60.0 * float(max_abs.max()))
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(names, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("|rise-time shift| (s, log scale)  -  dashed line is the 1 s threshold")
    ax_bar.grid(True, axis="x", which="both", ls=":", alpha=0.5)
    ax_bar.legend(loc="lower right", fontsize=8)
    for k, n in enumerate(names):
        m = metrics[n]
        ax_bar.text(
            max_abs[k] * 1.3, y[k],
            f"lost {m.passes_lost} / gained {m.passes_gained},  "
            f"contact {m.total_contact_error_s:+.0f} s of {m.total_contact_truth_s:.0f} s",
            va="center", fontsize=8,
        )

    fig.text(
        0.01, 0.005,
        "Windows are matched to truth's by time overlap; one with no overlapping counterpart is "
        "counted lost or gained, never given a shift.\nTruth and model share a single 60 s grid, so "
        "the convex-horizon interpolation bias is common-mode and cancels (access.py).",
        fontsize=8, va="bottom",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    _save(fig, "access_windows.png")


# ==================================================================================================

FIGURES: Dict[str, Callable[[], None]] = {
    "ground": figure_ground_tracks,
    "error": figure_error_growth,
    "drag": figure_drag_decay,
    "hierarchy": figure_hierarchy,
    "atmosphere": figure_atmosphere,
    "access": figure_access_windows,
}


def main(argv: List[str]) -> int:
    keys = argv[1:] or list(FIGURES)
    unknown = [k for k in keys if k not in FIGURES]
    if unknown:
        print(f"unknown figure key(s) {unknown}; have {list(FIGURES)}")
        return 2
    print(f"numba available: {NUMBA_AVAILABLE}")
    for key in keys:
        print(f"\n[{key}]")
        FIGURES[key]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
