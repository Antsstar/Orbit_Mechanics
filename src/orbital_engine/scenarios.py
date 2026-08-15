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

from sqlalchemy.orm import Session

from .database import CelestialBodyORM, SystemORM, VesselORM, VirtualBodyORM
from .simulator import Simulation

__all__ = [
    "MU_SUN", "MU_EARTH", "MU_MOON",
    "EARTH_P", "EARTH_E", "MOON_P", "MOON_E",
    "two_body", "sun_earth_moon", "earth_constellation",
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


def sun_earth_moon(session: Session, *, capacity: int = 64) -> Simulation:
    """
    Sun heading the Solar System; Earth heading a nested Earth-Moon system whose barycenter is
    itself a member of the Solar System.

    This is the configuration that makes the engine's two graphs diverge, and so the one that
    exercises what is actually novel here: the Moon's COE is measured against Earth
    (`parent_indices`) while its Cartesian state is measured against the Earth-Moon barycenter
    (`body_sys_map`).
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
        name="Moon", mu=MU_MOON, system_id=earth_moon.id, parent_id=earth.id, radius=1737.4,
        p=MOON_P, e=MOON_E, i=math.radians(5.145),
        raan=math.radians(125.08), arg_pe=math.radians(318.15), theta=math.radians(115.0),
    )
    session.add(moon)

    # The Earth-Moon barycenter is itself a body of the Solar System, orbiting the Sun.
    emb.parent_id = sun.id
    emb.system_id = solar.id
    session.commit()

    return Simulation(
        body_names=["Sun", "Earth", "Moon"],
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
