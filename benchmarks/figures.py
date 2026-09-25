"""
The figure gallery: every PNG in `docs/figures/` except `frontier.png`, which `frontier_plot.py`
owns.

Run with:

    <env>/python.exe benchmarks/figures.py            # all figures
    <env>/python.exe benchmarks/figures.py ground drag # a subset, by key
    <env>/python.exe benchmarks/figures.py solar       # solar_activity.png only (a few minutes; pymsis)

Keys: ground, error, drag, hierarchy, atmosphere, access, stationkeeping, solar.

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
the fused compiled twin carries only point-mass gravity, J2 and the J3..J6 zonals.

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
150x better than Kepler in kilometres (4.04 km against 607.61 km in `error_growth.png`) but only 20x
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

**7. `station_keeping.png` - caption.** One satellite, four atmospheres: four co-located 51.6 deg
Cowell + J2 satellites (B = 0.05 m^2/kg) held in a [291.0, 293.5] km *mean*-altitude band for 10 days
by `stationkeeping.py`'s dead-band controller at dt = 30 s, with two-impulse raises. The faint trace is
the osculating altitude, swinging ~12 km peak to peak under J2 - the reason the controller keys on the
one-period mean (solid), which is blank for ~2.5 orbits after each raise while it re-establishes it.
Under the table: 24 raises every 10.1 h, 34.78 m/s, a steady 3.436 m/s/day. Under the single band
matched at 355 km with H = 60 km (`atmosphere.png`'s): 20 raises every 11.8 h, 28.98 m/s, 2.951
m/s/day - **14.1 % under-budgeted**, because at 292 km the table's scale height is 45.5 km. The same
single band anchored at the band centre: 3.431 m/s/day, **0.14 %** off. The drag-free control never
burns. The steady rates are validated in `tests/validation/test_stationkeeping.py` against the
orbit-averaged decay converted at `(n/2) da`, to 4e-4.

**8. `solar_activity.png` - caption.** What the solar-activity assumption costs, against what the
atmosphere-model choice costs, in station-keeping Delta-v. One 51.6 deg satellite (B = 0.05 m^2/kg)
held in the [291, 293.5] km mean-altitude band of `station_keeping.png` for 6 days, Cowell + J2 + drag
at dt = 30 s, two-impulse raises. The numbers are not computed here: `msis_sweep.run_msis_sweep()`
runs `run_sweep(..., station_keeping=, delta_v_baseline="msis moderate")` on `msis_sweep.py`'s own
five configurations, and this figure draws its `DeltaVMetrics` (steady rate, first raise excluded).
Left, m/s/day and the signed error against the baseline: NRLMSIS 2.0 at moderate activity (F10.7 =
F10.7a = 140, Ap = 15) **3.047, 13 raises**; quiet Sun (65, 0) 0.855, **-71.9 %**, 4 raises; active Sun
(250, 45) 7.142, **+134.4 %**, 29 raises; Vallado's 28-band table 3.438, +12.8 %, 14 raises; the single
band matched to the table at 355 km with H = 60 km 2.952, -3.1 %, 12 raises - `CLAUDE.md`'s recorded
headline to every printed digit, and each within 6e-4 of the prediction in `msis_sweep.py`'s docstring.
The solar swings are **5.6x and 10.5x** the largest atmosphere-model error (12.8 %), which is the
"5-10x" of the headline; the shaded band is the whole spread of the model choice, 2.95 to 3.44
m/s/day, and the two static laws straddle moderate MSIS. Right, the mechanism: the three MSIS mean
profiles (read through the kernel's own evaluator) and the two static laws against altitude. At 292 km,
relative to moderate MSIS: quiet 0.280, active 2.341, table 1.128, single band 0.969 - the density
ratios of `docs/architecture.md`'s NRLMSIS table. The profiles are averages over latitude, local time
and season, so the diurnal bulge (2.30x day/night at 400 km) is absent by construction; the ECSS
presets are from memory, unverified.

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

# This repository's own `src` first, as `msis_sweep.py` and `zonal_sweep.py` do: the package is an
# editable install of the *main* checkout, so run from a git worktree this script would otherwise
# draw the main checkout's engine.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from orbital_engine import access, scenarios, sweep, viz  # noqa: E402
from orbital_engine.atmosphere import (  # noqa: E402
    BASE_ALTITUDE_KM, DENSITY_MODEL_EXPONENTIAL, DENSITY_MODEL_LAYERED, layered_density,
)
from orbital_engine.custom_types import PropagatorType  # noqa: E402
from orbital_engine.database import Base  # noqa: E402
from orbital_engine.drag import DRAG_MODEL, EARTH_OMEGA  # noqa: E402
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ, J2_MODEL  # noqa: E402
from orbital_engine.gravity import POINT_MASS_MODEL  # noqa: E402
from orbital_engine.kernels import NUMBA_AVAILABLE  # noqa: E402
from orbital_engine.reference import TRUTH_ATOL, TRUTH_RTOL, reference_for  # noqa: E402
from orbital_engine.simulator import Simulation  # noqa: E402

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
# 7. station_keeping.png - what the atmosphere model costs in Delta-v
# ==================================================================================================

# The satellite and band of `tests/validation/test_stationkeeping.py`, over a longer span. The test
# owns the derivation and the budget; this script only runs it longer and draws it.
SK_SEED_KM = 298.85                    # osculating; J2 puts the one-period mean at 292.0 km
SK_INC_DEG = 51.6
SK_B = 0.05                            # m^2/kg
SK_LOWER_KM = 291.0
SK_UPPER_KM = 293.5
SK_DT_S = 30.0                         # 60 s leaves a drag-dependent RK4 bias of 3e-3; see the test
SK_DAYS = 10.0
SK_MATCH_KM = 355.0                    # `test_atmosphere.py`'s single band: matched at 355 km, H = 60
SK_SINGLE_H_KM = 60.0
SK_PANELS = [
    ("layered: Vallado Table 8-4", "tab:blue"),
    (f"single band matched at {SK_MATCH_KM:.0f} km, H = {SK_SINGLE_H_KM:.0f} km", "tab:red"),
    (f"single band matched at the band centre, H = {SK_SINGLE_H_KM:.0f} km", "tab:orange"),
    ("no drag (control)", "tab:green"),
]


def figure_station_keeping() -> None:
    from orbital_engine.stationkeeping import StationKeepingSpec, run_station_keeping

    sim = scenarios.station_keeping_satellites(
        fresh_session(), n_sats=4, altitude_km=SK_SEED_KM, inclination_deg=SK_INC_DEG)
    sim.record_history = False
    slots = [sim.name_to_index[f"SK-SAT-{k:02d}"] for k in range(4)]
    common = dict(ballistic_coeff=SK_B, r_ref=EARTH_R_EQ, omega=EARTH_OMEGA)
    centre = 0.5 * (SK_LOWER_KM + SK_UPPER_KM)
    rho = lambda h: float(layered_density(np.array([h]))[0])        # noqa: E731
    sim.enable_force_model(DRAG_MODEL, slots[0], density_model=DENSITY_MODEL_LAYERED, **common)
    sim.enable_force_model(DRAG_MODEL, slots[1], density_model=DENSITY_MODEL_EXPONENTIAL,
                           rho0=rho(SK_MATCH_KM), h0=SK_MATCH_KM, scale_height=SK_SINGLE_H_KM, **common)
    sim.enable_force_model(DRAG_MODEL, slots[2], density_model=DENSITY_MODEL_EXPONENTIAL,
                           rho0=rho(centre), h0=centre, scale_height=SK_SINGLE_H_KM, **common)

    spec = StationKeepingSpec(SK_LOWER_KM, SK_UPPER_KM)
    out = run_station_keeping(sim, slots, spec, SK_DAYS * 86400.0, SK_DT_S)
    days = out.times_s / 86400.0

    lines = []
    for k, s in enumerate(out.summary):
        rate = float("nan")
        mine = [b for b in out.burns if b.body == slots[k]]
        if len(mine) >= 2:
            rate = sum(b.dv_km_s for b in mine[1:]) / (mine[-1].epoch_s - mine[0].epoch_s)
        lines.append((s.total_dv_km_s * 1e3, s.n_burns, s.mean_interval_s / 3600.0, rate * 1e3 * 86400.0))
        print(f"  {SK_PANELS[k][0]}: {s.n_burns} burns, total {s.total_dv_km_s * 1e3:.3f} m/s, "
              f"mean interval {s.mean_interval_s / 3600.0:.2f} h, steady rate {rate * 1e3 * 86400.0:.4f} m/s/day")
    base = lines[0][3]
    for k in (1, 2):
        print(f"  {SK_PANELS[k][0]} / layered steady rate: {lines[k][3] / base:.4f} "
              f"({100.0 * (lines[k][3] / base - 1.0):+.2f} %)")

    fig, axes = plt.subplots(4, 1, figsize=(10.0, 10.5), sharex=True)
    for k, ax in enumerate(axes):
        label, colour = SK_PANELS[k]
        ax.axhspan(SK_LOWER_KM, SK_UPPER_KM, color="0.9", zorder=0)
        ax.plot(days[::4], out.osculating_altitude_km[::4, k], lw=0.3, color=colour, alpha=0.35)
        ax.plot(days, out.mean_altitude_km[:, k], lw=1.6, color=colour)
        for b in out.burns:
            if b.body == slots[k]:
                ax.axvline(b.epoch_s / 86400.0, lw=0.7, color="k", alpha=0.6, ymax=0.08)
        total, n, hours, rate = lines[k]
        text = (f"{label}\n{n} raises, total $\\Delta v$ = {total:.2f} m/s" +
                (f", every {hours:.1f} h, {rate:.3f} m/s/day steady" if n >= 2 else ""))
        ax.text(0.005, 0.97, text, transform=ax.transAxes, fontsize=8.5, va="top",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        ax.set_ylim(SK_LOWER_KM - 4.5, SK_UPPER_KM + 8.5)
        ax.set_ylabel("altitude (km)")
        ax.grid(True, ls=":", alpha=0.5)
    axes[-1].set_xlabel("time (days)")
    axes[-1].set_xlim(0.0, SK_DAYS)
    axes[0].set_title(
        f"Station-keeping a 51.6 deg satellite in a [{SK_LOWER_KM:.1f}, {SK_UPPER_KM:.1f}] km "
        f"mean-altitude band, B = {SK_B} m$^2$/kg, {SK_DAYS:.0f} days\n"
        f"Cowell + J2 + drag at dt = {SK_DT_S:.0f} s, two-impulse raises. Faint: osculating altitude. "
        f"Solid: the controller's one-period mean\n(blank while it re-establishes the mean after a "
        f"raise). Ticks: raises.", fontsize=10)
    fig.text(
        0.01, 0.005,
        f"Same satellite, same band, same controller - only the density law differs. Anchored at "
        f"{SK_MATCH_KM:.0f} km, the single band budgets {100 * (1 - lines[1][3] / base):.1f} % less "
        f"Delta-v than the table,\nbecause at 292 km the table's scale height is 45.5 km, not 60. "
        f"Anchored at the band itself, the same single band differs by only "
        f"{100 * abs(lines[2][3] / base - 1):.2f} %:\nwhat the choice costs is where the single band is "
        f"anchored, not how many bands there are. Steady rates are validated in\n"
        f"tests/validation/test_stationkeeping.py against the orbit-averaged decay converted at (n/2) da, "
        f"to 4e-4.",
        fontsize=8, va="bottom",
    )
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    _save(fig, "station_keeping.png")


# ==================================================================================================
# 8. solar_activity.png - the solar-activity assumption against the atmosphere-model choice
# ==================================================================================================

# Colour carries the *question* a tier answers, not the tier: orange = which Sun (one model, three
# activity levels, darker = more active), blue = which atmosphere model. Both hues are the dataviz
# reference palette's slots 1-2, validated against each other (CVD dE 24.7, normal-vision 33.6); the
# light orange of the low-activity curve sits under 3:1 contrast, which is why every bar carries its
# row label and number, and the two outer density curves are labelled directly as well as in the
# legend (the three central ones overlap too closely for direct labels to stay legible).
SA_SOLAR = "#eb6834"
SA_MODEL = "#2a78d6"
SA_SOLAR_SHADES = {"msis low": "#f29a62", "msis moderate": "#eb6834", "msis high": "#9c3410"}
SA_INK = "#0b0b0b"
SA_MUTED = "#52514e"
SA_STATION_KM = 292.0              # the dead band's centre, [291, 293.5] km mean altitude
# Tier name (as `msis_sweep.build_configs` spells it) -> the row label a non-specialist can read.
SA_ROWS = [
    ("msis low", "Quiet Sun\nF10.7 = 65"),
    ("msis moderate", "Moderate Sun\nF10.7 = 140"),
    ("msis high", "Active Sun\nF10.7 = 250"),
    ("layered", "28-band table\n(Vallado)"),
    ("single@355", "Single exponential\n(H = 60 km)"),
]


def figure_solar_activity() -> None:
    # The numbers come from the engine's own sweep, exactly as `msis_sweep.py` prints them: same
    # scenario, same five configurations, same baseline. Nothing here re-derives a Delta-v.
    import msis_sweep
    from orbital_engine.msis_bridge import (
        SOLAR_ACTIVITY_HIGH, SOLAR_ACTIVITY_LOW, SOLAR_ACTIVITY_MODERATE, msis_density, msis_profile,
    )

    results = {r.config_name: r.delta_v for r in msis_sweep.run_msis_sweep()}
    rate: Dict[str, float] = {}
    err: Dict[str, float] = {}
    for name, dv in results.items():
        assert dv is not None
        rate[name] = dv.median_steady_rate_m_s_per_day
        err[name] = dv.rate_error_rel
        print(f"  {name:<15} {dv.bodies[0].n_raises:>3} raises  {rate[name]:.4f} m/s/day  "
              f"rate error {err[name]:+.4f}")
    base = rate[msis_sweep.BASELINE]
    solar_swing = (err["msis low"], err["msis high"])
    model_swing = (min(err["layered"], err["single@355"]), max(err["layered"], err["single@355"]))
    largest_model = max(abs(e) for e in model_swing)
    ratio_low, ratio_high = abs(solar_swing[0]) / largest_model, abs(solar_swing[1]) / largest_model
    print(f"  solar swing {100 * solar_swing[0]:+.1f} % / {100 * solar_swing[1]:+.1f} %, model swing "
          f"{100 * model_swing[0]:+.1f} % / {100 * model_swing[1]:+.1f} %: solar = "
          f"{ratio_low:.1f}x / {ratio_high:.1f}x the largest model error")

    # Density: the profiles the kernel reads (memoised by the sweep's configuration), evaluated
    # through the kernel's own evaluator, and the two static laws exactly as the sweep configured them.
    altitude = np.arange(150.0, 700.01, 1.0)
    activity = {"msis low": SOLAR_ACTIVITY_LOW, "msis moderate": SOLAR_ACTIVITY_MODERATE,
                "msis high": SOLAR_ACTIVITY_HIGH}
    density: Dict[str, ArrF] = {}
    for name, preset in activity.items():
        msis_profile(preset["f107"], preset["f107a"], preset["ap"])
        triple = np.tile([preset["f107"], preset["f107a"], preset["ap"]], (altitude.size, 1))
        density[name] = msis_density(altitude, triple)
    density["layered"] = layered_density(altitude)
    rho355 = float(layered_density(np.array([355.0]))[0])
    density["single@355"] = rho355 * np.exp(-(altitude - 355.0) / 60.0)
    at_station = {n: float(np.interp(SA_STATION_KM, altitude, d)) for n, d in density.items()}
    rel_station = {n: v / at_station["msis moderate"] for n, v in at_station.items()}
    print("  density at 292 km / MSIS moderate: " +
          ", ".join(f"{n} {v:.3f}" for n, v in rel_station.items()))

    rc = {"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12,
          "ytick.labelsize": 12, "axes.edgecolor": "0.55", "axes.labelcolor": SA_INK,
          "xtick.color": SA_MUTED, "ytick.color": SA_INK}
    with plt.rc_context(rc):
        fig, (ax, ax_d) = plt.subplots(
            1, 2, figsize=(15.0, 7.6), gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.28})

        # ---- left: the budget, in m/s per day ------------------------------------------------
        ys = [0.0, 1.0, 2.0, 3.55, 4.55]
        # The whole spread of the atmosphere-model choice (baseline included), as one shaded band:
        # the solar bars run straight through it and out the other side.
        lo = min(rate["layered"], rate["single@355"], base)
        hi = max(rate["layered"], rate["single@355"], base)
        ax.axvspan(lo, hi, color=SA_MODEL, alpha=0.13, lw=0, zorder=0)
        for (name, _), y in zip(SA_ROWS, ys):
            colour = SA_SOLAR if name.startswith("msis") else SA_MODEL
            ax.barh(y, rate[name], height=0.62, color=colour, zorder=2)
            label = (f"{rate[name]:.2f} m/s/day   (baseline)" if name == msis_sweep.BASELINE
                     else f"{rate[name]:.2f} m/s/day   ({100 * err[name]:+.1f} %)")
            # A bar ending inside the shaded band is labelled clear of it; one ending short of the
            # band is labelled at its own end, on a white ground that masks the band and the line.
            x_label = (hi if lo <= rate[name] <= hi else rate[name]) + 0.12
            ax.text(x_label, y, label, va="center", ha="left", fontsize=13, zorder=4,
                    color=SA_INK, fontweight="bold" if name in ("msis low", "msis high") else None,
                    bbox=dict(fc="white", ec="none", pad=1.5))
        ax.axvline(base, color=SA_INK, lw=1.2, ls="--", zorder=3)
        ax.text(base, -0.62, "baseline", ha="center", va="bottom", fontsize=11, color=SA_MUTED)
        ax.text(0.0, -0.95, "WHICH SUN?  one model (NRLMSIS 2.0), three activity levels",
                fontsize=12, fontweight="bold", color=SA_SOLAR, va="bottom", zorder=4,
                bbox=dict(fc="white", ec="none", pad=1.5))
        ax.text(0.0, 2.72, "WHICH ATMOSPHERE MODEL?  static density laws",
                fontsize=12, fontweight="bold", color=SA_MODEL, va="bottom", zorder=4,
                bbox=dict(fc="white", ec="none", pad=1.5))
        ax.text(hi + 0.12, 4.55 + 0.62, "shaded: the whole spread of the model choice",
                fontsize=10.5, color=SA_MODEL, va="bottom", style="italic")
        ax.set_yticks(ys)
        ax.set_yticklabels([label for _, label in SA_ROWS])
        ax.set_ylim(5.3, -1.25)
        ax.set_xlim(0.0, 11.0)
        ax.set_xlabel("station-keeping $\\Delta v$ (m/s per day)")
        ax.grid(True, axis="x", ls=":", alpha=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title("Propellant to hold a satellite at 292 km, 6 days", loc="left", pad=12)

        # ---- right: the mechanism, density against altitude ----------------------------------
        curves = [
            ("msis low", "NRLMSIS, quiet Sun", SA_SOLAR_SHADES["msis low"], "-"),
            ("msis moderate", "NRLMSIS, moderate Sun", SA_SOLAR_SHADES["msis moderate"], "-"),
            ("msis high", "NRLMSIS, active Sun", SA_SOLAR_SHADES["msis high"], "-"),
            ("layered", "28-band table", SA_MODEL, "-"),
            ("single@355", "single exponential", SA_MODEL, "--"),
        ]
        for name, label, colour, ls in curves:
            ax_d.semilogx(density[name], altitude, color=colour, ls=ls, lw=2.2, label=label)
        ax_d.axhline(SA_STATION_KM, color=SA_INK, lw=1.0, ls=":")
        ax_d.text(0.98, SA_STATION_KM + 6, "station, 292 km", transform=ax_d.get_yaxis_transform(),
                  ha="right", va="bottom", fontsize=11, color=SA_INK)
        # Direct labels on the two outer curves, in the empty space outside them at 450 km.
        k = int(np.argmin(abs(altitude - 450.0)))
        ax_d.text(density["msis low"][k] / 1.4, 450.0, "quiet Sun", color=SA_INK,
                  fontsize=12, fontweight="bold", va="center", ha="right")
        ax_d.text(density["msis high"][k] * 1.4, 450.0, "active Sun", color=SA_INK,
                  fontsize=12, fontweight="bold", va="center", ha="left")
        ax_d.set_ylim(150.0, 700.0)
        # A decade of empty space on the left, below the quiet-Sun curve, holds the 292 km box.
        ax_d.set_xlim(1e-15, 5e-9)
        ax_d.set_xlabel("air density (kg/m$^3$, log scale)")
        ax_d.set_ylabel("altitude (km)")
        ax_d.grid(True, which="major", ls=":", alpha=0.6)
        ax_d.spines[["top", "right"]].set_visible(False)
        ax_d.legend(loc="upper right", fontsize=11, frameon=False)
        ax_d.text(
            0.02, 0.03,
            "at 292 km, vs moderate Sun:\n"
            f"quiet {rel_station['msis low']:.2f}x   active {rel_station['msis high']:.2f}x\n"
            f"table {rel_station['layered']:.2f}x   single {rel_station['single@355']:.2f}x",
            transform=ax_d.transAxes, fontsize=11, color=SA_INK, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="0.75"))
        ax_d.set_title("Why: the Sun heats and inflates the air", loc="left", pad=12)

        fig.suptitle(
            f"The solar-activity assumption moves a station-keeping budget "
            f"{ratio_low:.1f}-{ratio_high:.1f}x more than the atmosphere model does\n"
            f"Sun: {100 * solar_swing[0]:+.0f} % to {100 * solar_swing[1]:+.0f} %.   "
            f"Atmosphere model: {100 * model_swing[0]:+.0f} % to {100 * model_swing[1]:+.0f} %.",
            x=0.01, ha="left", fontsize=16, fontweight="bold", color=SA_INK)
        fig.text(
            0.01, 0.012,
            f"One 51.6 deg satellite, B = {msis_sweep.B} m$^2$/kg, held in a [{msis_sweep.SPEC.lower_km}, "
            f"{msis_sweep.SPEC.upper_km}] km mean-altitude band for 6 days by two-impulse raises; "
            f"Cowell + J2 + drag at dt = {msis_sweep.DT_S:.0f} s.\n"
            f"Steady rate, first raise excluded, from run_sweep(..., station_keeping=, "
            f"delta_v_baseline=\"msis moderate\") in benchmarks/msis_sweep.py. NRLMSIS 2.0 via pymsis, "
            f"averaged over latitude, local time and season.\n"
            f"Solar presets ECSS low / moderate / high: F10.7 = F10.7a = 65 / 140 / 250, "
            f"Ap = 0 / 15 / 45 (from memory, unverified). Single exponential: matched to the table at "
            f"355 km, scale height 60 km.",
            fontsize=10, color=SA_MUTED, va="bottom")
        fig.subplots_adjust(left=0.12, right=0.985, top=0.83, bottom=0.2)
        _save(fig, "solar_activity.png")


# ==================================================================================================

FIGURES: Dict[str, Callable[[], None]] = {
    "ground": figure_ground_tracks,
    "error": figure_error_growth,
    "drag": figure_drag_decay,
    "hierarchy": figure_hierarchy,
    "atmosphere": figure_atmosphere,
    "access": figure_access_windows,
    "stationkeeping": figure_station_keeping,
    "solar": figure_solar_activity,
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
