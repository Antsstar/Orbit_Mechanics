"""
Validation of the hierarchical barycentric machinery on a Sun-Earth-Moon system.

Two-body Kepler exercises one body about a fixed primary. This exercises the part that is actually
novel in this engine: nested system bubbles, mass-weighted barycenters derived at build time, and
the reflex kick that makes a head body wobble about its own system's centre of mass rather than
sitting still.

The invariants below are exact consequences of the barycenter definition, not approximations, so
they hold to floating-point noise regardless of how long the simulation runs. That makes them a
sharp test: any error in the mass aggregation or the reflex kick shows up immediately as a drifting
centre of mass, and nothing else does.

Several also pin behaviour that CLAUDE.md documents as deliberate but surprising - summed barycenter
masses, zeroed head COEs - so a future refactor that quietly changes them fails here rather than in
someone's analysis six months later.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from orbital_engine.database import CelestialBodyORM, SystemORM, VirtualBodyORM
from orbital_engine.simulator import Simulation

from .invariants import (
    COM_DRIFT_TOL_KM,
    MOMENTUM_DRIFT_TOL_KMS,
    centre_of_mass,
    total_linear_momentum,
    vector_drift,
)

MU_SUN = 1.32712440042e11   # km^3/s^2
MU_EARTH = 3.986004418e5
MU_MOON = 4.9048695e3

EARTH_P = 149556260.0       # semi-latus rectum, km
EARTH_E = 0.0167086
MOON_P = 383241.0
MOON_E = 0.0549

SIM_DURATION_S = 365.25 * 86400.0
SIM_STEP_S = 12.0 * 3600.0


def _build_sun_earth_moon(session) -> Simulation:
    """
    Sun at the head of the Solar System; Earth at the head of a nested Earth-Moon system whose
    barycenter is itself a member of the Solar System.

    This is the configuration that makes the two graphs diverge: the Moon's COE is measured against
    Earth (`parent_indices`), while its Cartesian state is measured against the Earth-Moon
    barycenter (`body_sys_map`).
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
        name="Earth", mu=MU_EARTH, system_id=earth_moon.id, parent_id=sun.id, radius=6371.0,
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
        max_capacity=64,
    )


def _physical_bodies(sim: Simulation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mass, position and velocity of real bodies only - barycenters would double-count."""
    mask = sim.active_mask & ~sim.is_system
    return sim.mu_array[mask], sim.global_states[mask, :3], sim.global_states[mask, 3:]


def _pos(sim: Simulation, name: str) -> np.ndarray:
    return sim.global_states[sim.name_to_index[name], :3].copy()


# ==================================================================================================
# Build-time invariants
# ==================================================================================================

def test_barycenter_mass_is_the_sum_of_its_members(db_session):
    """
    CLAUDE.md documents that after _recalculate_all_barycenters a barycenter's mu_array entry holds
    the summed system mass rather than zero. Aggregation runs bottom-up, so the Solar System
    barycenter must include the Moon via the Earth-Moon barycenter.
    """
    sim = _build_sun_earth_moon(db_session)

    emb_mu = sim.mu_array[sim.name_to_index["EMB"]]
    ssb_mu = sim.mu_array[sim.name_to_index["SSB"]]

    assert emb_mu == pytest.approx(MU_EARTH + MU_MOON, rel=1e-12)
    assert ssb_mu == pytest.approx(MU_SUN + MU_EARTH + MU_MOON, rel=1e-12)


def test_head_bodies_carry_zeroed_coes(db_session):
    """
    A head's motion is a reflex kick about its barycenter, not an orbit, so its COE row is zeroed by
    design. Earth heads the Earth-Moon system, which is why the heliocentric ellipse lives on the
    EMB row instead - surprising enough to be worth pinning.
    """
    sim = _build_sun_earth_moon(db_session)

    earth_coe = sim.coe_states[sim.name_to_index["Earth"]]
    assert np.all(earth_coe == 0.0), f"head body carries non-zero COEs: {earth_coe}"


def test_barycenter_carries_the_heliocentric_orbit(db_session):
    """
    The ellipse the head would have carried moves to the barycenter row - but it is *not* the head's
    seeded ellipse, and expecting it to be is an easy mistake to make.

    The EMB sits mu_Moon/(mu_Earth + mu_Moon) of the lunar separation from Earth, about 4770 km, and
    more importantly moves at a different velocity: Earth circles the EMB once a month at roughly
        v_offset = 2*pi*4770 km / 27.32 d ~ 0.0127 km/s
    against a heliocentric speed of ~29.8 km/s, so dv/v ~ 4.3e-4.

    Since p = h^2/mu and h ~ r*v, the semi-latus rectum should differ by
        dp/p ~ 2 * dv/v ~ 8.5e-4
    with the ~3.2e-5 positional offset a much smaller correction.

    The bounds below are that prediction with a factor of ~2 of headroom, not the observed values
    rounded up. Measured at time of writing: dp/p = 8.73e-4, de = 8.6e-5.
    """
    sim = _build_sun_earth_moon(db_session)
    emb_coe = sim.coe_states[sim.name_to_index["EMB"]]

    # It must carry a real orbit, not a zeroed or degenerate row.
    assert emb_coe[0] > 0.0, "barycenter has no semi-latus rectum"
    assert emb_coe[1] == pytest.approx(EARTH_E, abs=5e-4), "eccentricity is not near Earth's"

    relative_p_shift = abs(emb_coe[0] - EARTH_P) / EARTH_P
    assert relative_p_shift < 2e-3, f"dp/p = {relative_p_shift:.3e} exceeds the predicted ~8.5e-4"

    # Guard the other direction: if the barycenter had collapsed onto Earth the shift would vanish,
    # and the assertion above would pass while the reflex machinery was doing nothing.
    assert relative_p_shift > 1e-4, f"dp/p = {relative_p_shift:.3e} is too small - is the EMB offset?"


# ==================================================================================================
# Conserved quantities over a full orbit
# ==================================================================================================

def test_centre_of_mass_holds_at_the_origin(db_session):
    """An isolated system's centre of mass cannot move. _zero_roots puts it at the origin at t=0."""
    sim = _build_sun_earth_moon(db_session)

    mu, r0, _ = _physical_bodies(sim)
    com_initial = centre_of_mass(mu, r0)
    assert float(np.linalg.norm(com_initial)) < COM_DRIFT_TOL_KM

    sim.run(SIM_DURATION_S, SIM_STEP_S)

    mu, r1, _ = _physical_bodies(sim)
    com_final = centre_of_mass(mu, r1)

    assert vector_drift(com_initial, com_final) < COM_DRIFT_TOL_KM
    assert float(np.linalg.norm(com_final)) < COM_DRIFT_TOL_KM


def test_total_linear_momentum_stays_zero(db_session):
    """
    Built about a stationary barycenter, so the mass-weighted mean velocity is zero and must remain
    so. A reflex kick applied with the wrong sign or normalisation would show here as net drift.
    """
    sim = _build_sun_earth_moon(db_session)

    mu, _, v0 = _physical_bodies(sim)
    p_initial = total_linear_momentum(mu, v0)
    assert float(np.linalg.norm(p_initial)) < MOMENTUM_DRIFT_TOL_KMS

    sim.run(SIM_DURATION_S, SIM_STEP_S)

    mu, _, v1 = _physical_bodies(sim)
    p_final = total_linear_momentum(mu, v1)

    assert vector_drift(p_initial, p_final) < MOMENTUM_DRIFT_TOL_KMS


# ==================================================================================================
# The reflex kick itself
# ==================================================================================================

def test_solar_wobble_satisfies_the_barycentre_condition(db_session):
    """
    The Sun is not fixed: it orbits the Solar System barycenter in response to the Earth-Moon system.

    With the barycenter at the origin the mass moments must cancel exactly,

        mu_Sun * r_Sun  +  mu_EMB * r_EMB  =  0

    which is a definition rather than an approximation and so holds to floating-point noise. A test
    on the wobble's *amplitude* alone would pass even if the Sun were displaced in the wrong
    direction; comparing the vectors catches that.
    """
    sim = _build_sun_earth_moon(db_session)
    sim.run(SIM_DURATION_S, SIM_STEP_S)

    r_sun = _pos(sim, "Sun")
    r_emb = _pos(sim, "EMB")

    mu_emb = sim.mu_array[sim.name_to_index["EMB"]]
    moment = MU_SUN * r_sun + mu_emb * r_emb

    # Normalised against the larger of the two moments so the tolerance is relative, not absolute.
    scale = MU_SUN * float(np.linalg.norm(r_sun))
    assert float(np.linalg.norm(moment)) / scale < 1e-9

    # And the displacement is physically the right size: roughly 450 km, well inside the Sun.
    wobble_km = float(np.linalg.norm(r_sun))
    predicted = (mu_emb / (MU_SUN + mu_emb)) * float(np.linalg.norm(r_sun - r_emb))
    assert wobble_km == pytest.approx(predicted, rel=1e-9)
    assert 100.0 < wobble_km < 1000.0, f"solar wobble {wobble_km:.1f} km is not physically plausible"


def test_lunar_distance_stays_within_its_apsides(db_session):
    """
    The Moon's separation from Earth must stay between periapsis and apoapsis of its seeded orbit.

    This is what catches a reflex kick applied to the wrong body or with the wrong mass ratio: the
    barycentric quantities above would still balance, but the Moon's orbit about Earth would
    quietly inflate or collapse.
    """
    sim = _build_sun_earth_moon(db_session)

    r_periapsis = MOON_P / (1.0 + MOON_E)
    r_apoapsis = MOON_P / (1.0 - MOON_E)

    separations = []
    steps = int(SIM_DURATION_S / SIM_STEP_S)
    for _ in range(steps):
        sim.step(SIM_STEP_S)
        separations.append(float(np.linalg.norm(_pos(sim, "Moon") - _pos(sim, "Earth"))))

    observed_min, observed_max = min(separations), max(separations)

    assert observed_min >= r_periapsis * (1.0 - 1e-9), f"{observed_min:.1f} < periapsis {r_periapsis:.1f}"
    assert observed_max <= r_apoapsis * (1.0 + 1e-9), f"{observed_max:.1f} > apoapsis {r_apoapsis:.1f}"

    # Over a year the Moon completes ~13 orbits, so both apsides should actually be approached -
    # otherwise the bounds above are satisfied trivially by an orbit that barely moves.
    span = observed_max - observed_min
    assert span > 0.9 * (r_apoapsis - r_periapsis), f"lunar radius only spanned {span:.1f} km"
