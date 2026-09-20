"""
Reusable scenario builders.

A *scenario* is a database state plus the query that turns it into a `Simulation`. Keeping them here
rather than inside individual test modules means the validation suite, the benchmark harness and
(eventually) the model-comparison sweep all exercise the *same* universes, so a performance number
and a correctness number refer to the same thing.

Every builder takes an open `Session`, seeds it, commits, and returns a built `Simulation`. They are
deterministic: no clock, no RNG, no ambient database. Bodies are seeded from published J2000-era
elements where they represent real objects, so the same scenario can later be diffed against an
ephemeris without being rebuilt.

Units follow the engine convention throughout: km, km/s, radians, seconds, mu in km^3/s^2.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from sqlalchemy.orm import Session

from .custom_types import PropagatorType
from .database import CelestialBodyORM, SystemORM, VesselORM, VirtualBodyORM
from .gravity import POINT_MASS_MODEL
from .simulator import Simulation
from .thrust import THRUST_MODEL

__all__ = [
    "MU_SUN", "MU_EARTH", "MU_MOON",
    "EARTH_P", "EARTH_E", "MOON_P", "MOON_E",
    "VESSEL_DRY_MASS", "VESSEL_FUEL_MASS",
    "STATION_LATITUDE_DEG", "STATION_LONGITUDE_DEG", "STATION_ALTITUDE_KM",
    "two_body", "sun_earth_moon", "earth_constellation", "powered_vessel", "hohmann_pair",
    "ground_station_pass", "eclipsed_satellite",
    "LIGHT_SOURCE_NAME", "LIGHT_SOURCE_DISTANCE_KM",
]

# --------------------------------------------------------------------------------------------------
# Reference constants. Values are IAU/JPL DE440 gravitational parameters and J2000 mean elements,
# expressed as semi-latus rectum p = a(1 - e^2) to match the engine's COE column 0.
# --------------------------------------------------------------------------------------------------
MU_SUN = 1.32712440042e11
MU_EARTH = 3.986004418e5
MU_MOON = 4.9048695e3

EARTH_P = 149556260.0
EARTH_E = 0.0167086
MOON_P = 383241.0
MOON_E = 0.0549

EARTH_RADIUS = 6371.0


def two_body(
    session: Session,
    *,
    mu_primary: float = MU_EARTH,
    mu_secondary: float = 0.0,
    p: float = 11000.0,
    e: float = 0.2,
    i: float = 0.0,
    raan: float = 0.0,
    arg_pe: float = 0.0,
    theta: float = 0.0,
    capacity: int = 16,
) -> Simulation:
    """
    An isolated primary with a single orbiting secondary, and no barycenter.

    This is the case with a closed-form answer, so it is what analytic assertions are written
    against. `mu_secondary` defaults to zero to give the restricted problem exactly - a massless
    secondary means the engine's two-body mass sum reduces to `mu_primary` and the orbit period is
    the textbook one, with no barycentric correction to account for.
    """
    bary = VirtualBodyORM(name="TB Barycenter")
    session.add(bary)
    session.flush()

    system = SystemORM(name="Two Body System", barycenter_id=bary.id)
    session.add(system)
    session.flush()

    primary = CelestialBodyORM(
        name="Primary", mu=mu_primary, system_id=system.id, radius=EARTH_RADIUS,
        p=0.0, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, theta=0.0,
    )
    session.add(primary)
    session.flush()
    system.head_body_id = primary.id

    secondary = CelestialBodyORM(
        name="Secondary", mu=mu_secondary, system_id=system.id, parent_id=primary.id, radius=1.0,
        p=p, e=e, i=i, raan=raan, arg_pe=arg_pe, theta=theta,
    )
    session.add(secondary)
    session.commit()

    return Simulation(
        body_names=["Primary", "Secondary"],
        system_names=["Two Body System"],
        session=session,
        max_capacity=capacity,
    )


def sun_earth_moon(
    session: Session, *, capacity: int = 64, moon_mu: float = MU_MOON, leo_satellite: bool = False,
) -> Simulation:
    """
    Sun heading the Solar System; Earth heading a nested Earth-Moon system whose barycenter is
    itself a member of the Solar System.

    This is the configuration that makes the engine's two graphs diverge, and so the one that
    exercises what is actually novel here: the Moon's COE is measured against Earth
    (`parent_indices`) while its Cartesian state is measured against the Earth-Moon barycenter
    (`body_sys_map`).

    `moon_mu` defaults to the real lunar mass. Passing `0.0` gives a massless Moon: it stops
    contributing to Earth's reflex kick (the same "massless secondary" degenerate limit
    `two_body(mu_secondary=0.0)` exercises), while Earth itself keeps its genuine heliocentric
    acceleration toward the Sun - this is what makes the scenario useful for validating a body whose
    *parent* accelerates, which `two_body`'s always-fixed primary cannot exercise at all.

    `leo_satellite=True` adds a massless vessel, `LEO-SAT`, in a 7000 km orbit about Earth inside the
    Earth-Moon system. With the default massive Moon, its parent (Earth) and its kinematic bubble (the
    Earth-Moon barycentre) sit about 4700 km apart. Every other shipped massless body has a parent that
    coincides with its bubble, which hides any code that confuses `parent_indices` with `body_sys_map`.
    """
    ssb = VirtualBodyORM(name="SSB")
    emb = VirtualBodyORM(name="EMB")
    session.add_all([ssb, emb])
    session.flush()

    solar = SystemORM(name="Solar System", barycenter_id=ssb.id)
    earth_moon = SystemORM(name="Earth-Moon System", barycenter_id=emb.id)
    session.add_all([solar, earth_moon])
    session.flush()

    sun = CelestialBodyORM(
        name="Sun", mu=MU_SUN, system_id=solar.id, radius=696340.0,
        p=0.0, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, theta=0.0,
    )
    session.add(sun)
    session.flush()
    solar.head_body_id = sun.id

    earth = CelestialBodyORM(
        name="Earth", mu=MU_EARTH, system_id=earth_moon.id, parent_id=sun.id, radius=EARTH_RADIUS,
        p=EARTH_P, e=EARTH_E, i=0.0,
        raan=math.radians(-11.26), arg_pe=math.radians(114.2), theta=math.radians(102.34),
    )
    session.add(earth)
    session.flush()
    earth_moon.head_body_id = earth.id

    moon = CelestialBodyORM(
        name="Moon", mu=moon_mu, system_id=earth_moon.id, parent_id=earth.id, radius=1737.4,
        p=MOON_P, e=MOON_E, i=math.radians(5.145),
        raan=math.radians(125.08), arg_pe=math.radians(318.15), theta=math.radians(115.0),
    )
    session.add(moon)

    if leo_satellite:
        session.add(VesselORM(
            name="LEO-SAT", mu=0.0, system_id=earth_moon.id, parent_id=earth.id,
            dry_mass=260.0, fuel_mass=0.0, drag_area=4.0,
            p=7000.0 * (1.0 - 0.001 ** 2), e=0.001, i=math.radians(51.6),
            raan=math.radians(40.0), arg_pe=math.radians(10.0), theta=math.radians(75.0),
        ))

    # The Earth-Moon barycenter is itself a body of the Solar System, orbiting the Sun.
    emb.parent_id = sun.id
    emb.system_id = solar.id
    session.commit()

    return Simulation(
        body_names=["Sun", "Earth", "Moon"] + (["LEO-SAT"] if leo_satellite else []),
        system_names=["Solar System", "Earth-Moon System"],
        session=session,
        max_capacity=capacity,
    )


def earth_constellation(
    session: Session,
    *,
    n_sats: int = 60,
    n_planes: int = 6,
    altitude_km: float = 550.0,
    inclination_deg: float = 53.0,
    capacity: Optional[int] = None,
) -> Simulation:
    """
    A Walker-style constellation of massless vessels about a single Earth.

    Built for scaling measurements: body count is the free parameter, the topology stays flat (one
    tier of siblings under one head), and the vessels carry no mass, so adding satellites changes
    the array lengths without changing the barycentric structure. That isolates *cost per body* from
    cost per topological tier, which a hierarchical scenario like `sun_earth_moon` cannot.

    Geometry is a Walker delta: `n_planes` equally spaced RAANs, satellites phased evenly within
    each plane. Vessels are seeded with `mu = 0` so they perturb neither Earth nor each other.
    """
    if n_planes < 1:
        raise ValueError(f"n_planes must be at least 1, got {n_planes}")
    if n_sats < n_planes:
        raise ValueError(f"n_sats ({n_sats}) must be at least n_planes ({n_planes})")

    bary = VirtualBodyORM(name="Earth Barycenter")
    session.add(bary)
    session.flush()

    system = SystemORM(name="Earth System", barycenter_id=bary.id)
    session.add(system)
    session.flush()

    earth = CelestialBodyORM(
        name="Earth", mu=MU_EARTH, system_id=system.id, radius=EARTH_RADIUS,
        p=0.0, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, theta=0.0,
    )
    session.add(earth)
    session.flush()
    system.head_body_id = earth.id

    radius = EARTH_RADIUS + altitude_km
    inclination = math.radians(inclination_deg)
    per_plane = n_sats // n_planes
    remainder = n_sats % n_planes

    names: List[str] = ["Earth"]
    for plane in range(n_planes):
        raan = 2.0 * math.pi * plane / n_planes
        count = per_plane + (1 if plane < remainder else 0)
        for slot in range(count):
            name = f"SAT-{plane:02d}-{slot:03d}"
            session.add(VesselORM(
                name=name, mu=0.0, system_id=system.id, parent_id=earth.id,
                dry_mass=260.0, fuel_mass=0.0, drag_area=4.0,
                p=radius, e=0.0, i=inclination,          # circular, so p == a == r
                raan=raan, arg_pe=0.0,
                theta=2.0 * math.pi * slot / count,
            ))
            names.append(name)

    session.commit()

    # +8 slots of headroom for the barycenter and any future spawns.
    return Simulation(
        body_names=names,
        system_names=["Earth System"],
        session=session,
        max_capacity=capacity if capacity is not None else len(names) + 8,
    )


# --------------------------------------------------------------------------------------------------
# Powered flight
# --------------------------------------------------------------------------------------------------

VESSEL_DRY_MASS = 260.0      # kg, matching the constellation's vessels
VESSEL_FUEL_MASS = 40.0      # kg


def powered_vessel(
    session: Session,
    *,
    n_vessels: int = 2,
    n_powered: int = 1,
    p_km: float = EARTH_RADIUS + 550.0,
    e: float = 0.0,
    inclination_deg: float = 0.0,
    raan_deg: float = 0.0,
    arg_pe_deg: float = 0.0,
    theta_deg: float = 0.0,
    dry_mass: float = VESSEL_DRY_MASS,
    fuel_mass: float = VESSEL_FUEL_MASS,
    thrust_n: float = 0.0,
    isp_s: float = 300.0,
    direction: tuple[float, float, float] = (0.0, 1.0, 0.0),
    capacity: Optional[int] = None,
) -> Simulation:
    """
    One Earth and `n_vessels` co-located, identical, massless vessels, all integrated with Cowell
    under `point_mass_gravity`; the first `n_powered` of them also carry the `"thrust"` force model.

    The remainder are **thrust-free twins**: same arena, same initial state, same integrator, same
    step size, so differencing a powered vessel against one isolates the thrust term from everything
    else including RK4's own drift. That is what `tests/validation/test_thrust.py` uses to separate a
    burn's `Delta v` from the gravity turn, and what `test_drag.py` uses a drag-free control for.

    Vessels are named `THRUSTER-00`, `THRUSTER-01`, ...; `Earth` heads the system. They are massless
    (`mu = 0`), so co-location is physical rather than a singularity: they neither attract each other
    nor perturb Earth, and `"thrust"` requires masslessness anyway (see `thrust.py`).

    `p_km` is the semi-latus rectum, matching the engine's COE column 0 - for the default `e = 0` it
    is the circular radius. Raising it is how a test gets a long orbital period, which is how the
    geometric contamination of a short burn (the thrust direction turning with the orbit, and the
    gravity gradient across the twins' separation) is driven down: both scale as `(n tau)^2`.

    `mass_kg` is seeded from the vessel's own `dry_mass + fuel_mass`, so the propellant budget is a
    property of the seeded vessel, not of the force-model call. Nothing reads the ORM again after
    build; the mass afterwards lives in `force_model_params["thrust"]` and is advanced by
    `Simulation.step` (see `thrust.py`'s "Where the mass lives").
    """
    if n_vessels < 1:
        raise ValueError(f"n_vessels must be at least 1, got {n_vessels}")
    if not 0 <= n_powered <= n_vessels:
        raise ValueError(f"n_powered must be in [0, {n_vessels}], got {n_powered}")

    bary = VirtualBodyORM(name="Earth Barycenter")
    session.add(bary)
    session.flush()

    system = SystemORM(name="Earth System", barycenter_id=bary.id)
    session.add(system)
    session.flush()

    earth = CelestialBodyORM(
        name="Earth", mu=MU_EARTH, system_id=system.id, radius=EARTH_RADIUS,
        p=0.0, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, theta=0.0,
    )
    session.add(earth)
    session.flush()
    system.head_body_id = earth.id

    names: List[str] = ["Earth"]
    for k in range(n_vessels):
        name = f"THRUSTER-{k:02d}"
        session.add(VesselORM(
            name=name, mu=0.0, system_id=system.id, parent_id=earth.id,
            dry_mass=dry_mass, fuel_mass=fuel_mass, drag_area=4.0,
            p=p_km, e=e, i=math.radians(inclination_deg),
            raan=math.radians(raan_deg), arg_pe=math.radians(arg_pe_deg),
            theta=math.radians(theta_deg),
        ))
        names.append(name)

    session.commit()

    sim = Simulation(
        body_names=names,
        system_names=["Earth System"],
        session=session,
        max_capacity=capacity if capacity is not None else len(names) + 8,
    )

    slots = np.array([sim.name_to_index[n] for n in names[1:]], dtype=np.int64)
    sim.set_propagator(slots, PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, slots)
    if n_powered > 0:
        sim.enable_force_model(
            THRUST_MODEL, slots[:n_powered],
            thrust_n=thrust_n, isp_s=isp_s,
            mass_kg=dry_mass + fuel_mass, dry_mass_kg=dry_mass,
            dir_r=direction[0], dir_s=direction[1], dir_w=direction[2],
        )
    return sim


# --------------------------------------------------------------------------------------------------
# Impulsive manoeuvres
# --------------------------------------------------------------------------------------------------

def hohmann_pair(
    session: Session,
    *,
    r1_km: float = 7000.0,
    inclination_deg: float = 28.5,
    raan_deg: float = 40.0,
    arg_pe_deg: float = 0.0,
    theta_deg: float = 0.0,
    capacity: Optional[int] = None,
) -> Simulation:
    """
    One Earth and four co-located massless vessels on the same circular orbit of radius `r1_km`, two
    propagated analytically and two by Cowell under `point_mass_gravity`.

    Built for `tests/validation/test_manoeuvres.py`, whose headline case is a Hohmann transfer - the
    manoeuvre with exact closed-form answers for both burns and for the transfer time. Carrying *four*
    vessels in one arena gives that transfer two independent realisations and two controls in a single
    run:

    | name | propagator | role |
    |---|---|---|
    | `KEPLER-SAT`  | Keplerian | flies the transfer; its elements are re-derived at each impulse |
    | `KEPLER-TWIN` | Keplerian | **control**: never manoeuvres. A split step must not move it |
    | `COWELL-SAT`  | Cowell + `point_mass_gravity` | flies the same transfer numerically |
    | `COWELL-TWIN` | Cowell + `point_mass_gravity` | **control**, the numerical counterpart |

    The analytic pair answers "is the closed form reproduced", the numerical pair answers "does the
    impulse mean the same thing to an integrator", their difference is bounded by RK4's own truncation,
    and the twins measure what step splitting costs a body that is not manoeuvring - the same
    co-located-twin idiom `powered_vessel` uses for continuous thrust, and for the same reason:
    identical arena, identical step size, so a difference is the manoeuvre and nothing else.

    The orbit is deliberately **inclined and rotated** (`inclination_deg`, `raan_deg` default to
    non-zero). At `i = raan = arg_pe = theta = 0` the RSW basis coincides with the inertial axes, so a
    Delta-v applied in the wrong frame would be numerically identical to one applied correctly and the
    frame convention could not be validated at all. Hohmann's closed form is orientation-independent,
    so tilting the orbit costs the test nothing.

    `p == a == r1_km`, the seed orbit being circular. The vessels are massless, so co-location is
    physical rather than a singularity and the system barycentre sits exactly on Earth.
    """
    bary = VirtualBodyORM(name="Earth Barycenter")
    session.add(bary)
    session.flush()

    system = SystemORM(name="Earth System", barycenter_id=bary.id)
    session.add(system)
    session.flush()

    earth = CelestialBodyORM(
        name="Earth", mu=MU_EARTH, system_id=system.id, radius=EARTH_RADIUS,
        p=0.0, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, theta=0.0,
    )
    session.add(earth)
    session.flush()
    system.head_body_id = earth.id

    names = ["KEPLER-SAT", "KEPLER-TWIN", "COWELL-SAT", "COWELL-TWIN"]
    for name in names:
        session.add(VesselORM(
            name=name, mu=0.0, system_id=system.id, parent_id=earth.id,
            dry_mass=VESSEL_DRY_MASS, fuel_mass=VESSEL_FUEL_MASS, drag_area=4.0,
            p=r1_km, e=0.0, i=math.radians(inclination_deg),
            raan=math.radians(raan_deg), arg_pe=math.radians(arg_pe_deg),
            theta=math.radians(theta_deg),
        ))
    session.commit()

    sim = Simulation(
        body_names=["Earth"] + names,
        system_names=["Earth System"],
        session=session,
        max_capacity=capacity if capacity is not None else len(names) + 8,
    )

    cowell = np.array([sim.name_to_index[n] for n in ("COWELL-SAT", "COWELL-TWIN")], dtype=np.int64)
    sim.set_propagator(cowell, PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, cowell)
    return sim


# --------------------------------------------------------------------------------------------------
# Observation geometry
# --------------------------------------------------------------------------------------------------

STATION_LATITUDE_DEG = 0.0      # geocentric, matching `geometry.py`'s spherical convention
STATION_LONGITUDE_DEG = 0.0     # east-positive; the station is on +x at prime-meridian angle 0
STATION_ALTITUDE_KM = 0.0       # on the sphere of radius EARTH_RADIUS


def ground_station_pass(
    session: Session,
    *,
    n_sats: int = 1,
    altitude_km: float = 550.0,
    inclination_deg: float = 0.0,
    raan_deg: float = 0.0,
    capacity: Optional[int] = None,
) -> Simulation:
    """
    One Earth and `n_sats` massless vessels on a circular orbit, seeded for the **ground-station
    pass** geometry `tests/validation/test_geometry.py` checks against closed forms.

    The station itself is not a body - it is fixed to the rotating central body, so it lives in
    `geometry.py`'s arguments rather than in the arena. `STATION_LATITUDE_DEG`,
    `STATION_LONGITUDE_DEG` and `STATION_ALTITUDE_KM` above are the companion constants: the
    equatorial site on the prime meridian, which at prime-meridian angle `theta = 0` sits on the
    inertial `+x` axis.

    **Why the default is equatorial and phased to 180 deg.** With `inclination_deg = 0` the station
    lies exactly in the orbital plane, which is the only geometry whose pass has a closed form
    simple enough to be an *exact* expectation rather than a numerical one: the satellite reaches
    exactly 90 deg elevation, the horizon crossings sit at central angle `acos(R / r)` from the
    station, and the pass length is that angle over the **station-relative** angular rate `n - omega`
    - the body's own rotation subtracts, and forgetting it shortens the pass by 6.7 % at 550 km
    (784.2 s becomes 732.1 s), silently and plausibly. The vessel is
    seeded at true anomaly 180 deg so that the overhead moment, `pi / (n - omega)`, falls in the
    interior of a grid starting at zero rather than on its edge, which is what lets a window have
    two interpolated ends.

    That symmetry is also a hazard, as `docs/engineering-log.md` records for `hohmann_pair`: an
    equatorial station watching an equatorial orbit has an identically zero SEZ *south* component,
    so it cannot discriminate a transposed south/east axis. `inclination_deg` is therefore a
    parameter, and the test suite runs a second, inclined case against the general central-angle
    relation for exactly that reason.

    Vessels are massless, so the system barycentre sits on Earth and `global_states` are
    Earth-relative to machine precision; they keep the default Keplerian propagator, so the
    trajectory carries no integration error to confuse a geometry tolerance.
    """
    if n_sats < 1:
        raise ValueError(f"n_sats must be at least 1, got {n_sats}")

    bary = VirtualBodyORM(name="Earth Barycenter")
    session.add(bary)
    session.flush()

    system = SystemORM(name="Earth System", barycenter_id=bary.id)
    session.add(system)
    session.flush()

    earth = CelestialBodyORM(
        name="Earth", mu=MU_EARTH, system_id=system.id, radius=EARTH_RADIUS,
        p=0.0, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, theta=0.0,
    )
    session.add(earth)
    session.flush()
    system.head_body_id = earth.id

    radius = EARTH_RADIUS + altitude_km
    names: List[str] = ["Earth"]
    for k in range(n_sats):
        name = f"PASS-SAT-{k:02d}"
        session.add(VesselORM(
            name=name, mu=0.0, system_id=system.id, parent_id=earth.id,
            dry_mass=VESSEL_DRY_MASS, fuel_mass=0.0, drag_area=4.0,
            p=radius, e=0.0, i=math.radians(inclination_deg),      # circular, so p == a == r
            raan=math.radians(raan_deg), arg_pe=0.0,
            theta=math.pi + 2.0 * math.pi * k / n_sats,
        ))
        names.append(name)

    session.commit()

    return Simulation(
        body_names=names,
        system_names=["Earth System"],
        session=session,
        max_capacity=capacity if capacity is not None else len(names) + 8,
    )


# --------------------------------------------------------------------------------------------------
# Eclipse geometry, in an Earth-rooted arena
# --------------------------------------------------------------------------------------------------

#: Name of the massless, effectively static luminous marker `eclipsed_satellite` seeds as the SRP
#: light source. It is deliberately not called "Sun": it carries `mu = 0`, so it is a *position* that
#: light comes from and not a star. That is all `srp.py` ever asks of a source - the model reads the
#: source's row of `global_states` and never its mass.
LIGHT_SOURCE_NAME = "LIGHT"

#: Distance of that marker from Earth, km. One astronomical unit, so `(AU/d)^2` in `srp.py`'s kernel
#: is 1 to a part in 1e8 and the acceleration magnitude is the textbook `C_r P_srp (A/m)`.
LIGHT_SOURCE_DISTANCE_KM = 1.495978707e8


def eclipsed_satellite(
    session: Session,
    *,
    n_sats: int = 1,
    altitude_km: float = 550.0,
    inclination_deg: float = 23.4,
    capacity: Optional[int] = None,
) -> Simulation:
    """
    Earth at the arena root, a distant massless light source, and `n_sats` satellites that pass
    through Earth's umbra once per orbit.

    **Why this exists rather than `sun_earth_moon(leo_satellite=True)`.** That scenario's arena is
    heliocentric: a LEO satellite's `global_states` row is `~1.5e8 km`, and a Cowell body's
    acceleration is computed by differencing it against its parent's, which throws away nine
    significant digits. Over 1.5 LEO orbits that puts a **round-off floor of about `1e-5 km`** on the
    integrated trajectory - measured, with the shadow and SRP switched off entirely - which is the
    same size as RK4's own truncation at `h = 10 s`. RK4's fourth-order convergence is therefore not
    observable at all in that arena, so neither is its *recovery* after a discontinuity is removed
    (`events.py`). Here Earth heads its own system at the root, so a satellite's coordinates are
    `~7000 km`, the floor drops by six orders, and a step-halving ladder reads a clean 16.

    The light source is `LIGHT_SOURCE_NAME`: a massless vessel on a circular orbit at
    `LIGHT_SOURCE_DISTANCE_KM`, which is 1 AU, where its period is `~1.8e10 s` and it moves under half
    a kilometre over a 1.5-orbit run. It is a real Keplerian body rather than a frozen row, so the
    scenario stays time-invariant and needs no special case anywhere.

    Geometry: the source sits on `+x` at `t = 0`, so Earth's umbra trails along `-x`; the satellites
    are circular at `raan = 0`, so their line of nodes runs along `+/-x` and every satellite passes
    through the shadow axis at its descending node. `inclination_deg` defaults to a non-zero,
    non-special value for the same reason `ground_station_pass`'s does - an equatorial orbit is
    symmetric about the shadow axis and a transposed component could hide in it. Satellites start at
    true anomaly 0 (on `+x`, full sun) and are phased evenly, so `n_sats > 1` gives an arena in which
    some bodies are crossing the terminator while others are not - which is what makes the split per
    arena rather than per body observable (`events.py`).

    Nothing is configured: the caller sets the propagator and enables `"srp"`, because the shadow
    model and its coefficients are the thing under test.
    """
    if n_sats < 1:
        raise ValueError(f"n_sats must be at least 1, got {n_sats}")

    bary = VirtualBodyORM(name="Earth Barycenter")
    session.add(bary)
    session.flush()

    system = SystemORM(name="Earth System", barycenter_id=bary.id)
    session.add(system)
    session.flush()

    earth = CelestialBodyORM(
        name="Earth", mu=MU_EARTH, system_id=system.id, radius=EARTH_RADIUS,
        p=0.0, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, theta=0.0,
    )
    session.add(earth)
    session.flush()
    system.head_body_id = earth.id

    session.add(VesselORM(
        name=LIGHT_SOURCE_NAME, mu=0.0, system_id=system.id, parent_id=earth.id,
        dry_mass=1.0, fuel_mass=0.0, drag_area=0.0,
        p=LIGHT_SOURCE_DISTANCE_KM, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, theta=0.0,
    ))

    radius = EARTH_RADIUS + altitude_km
    names: List[str] = ["Earth", LIGHT_SOURCE_NAME]
    for k in range(n_sats):
        name = f"ECLIPSE-SAT-{k:02d}"
        session.add(VesselORM(
            name=name, mu=0.0, system_id=system.id, parent_id=earth.id,
            dry_mass=VESSEL_DRY_MASS, fuel_mass=0.0, drag_area=4.0,
            p=radius, e=0.0, i=math.radians(inclination_deg),      # circular, so p == a == r
            raan=0.0, arg_pe=0.0,
            theta=2.0 * math.pi * k / n_sats,
        ))
        names.append(name)

    session.commit()

    return Simulation(
        body_names=names,
        system_names=["Earth System"],
        session=session,
        max_capacity=capacity if capacity is not None else len(names) + 8,
    )
