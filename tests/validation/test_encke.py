"""
Encke's method (`integrators.EnckeIntegrator`, `set_cowell_integrator("encke")`): RK4 on the deviation
from the osculating conic, re-anchored every step.

Expected, derived before measuring:
- **Battin's identity** `-(mu/rho^3)(f(q) r + dr) = -mu r/r^3 + mu r_ref/rho^3` (derived in the class
  docstring) holds to the direct difference's own cancellation error, ~1e-16 |a| / |difference| ~ 4e-11
  relative for a 10 m deviation at 7,000 km; and to round-off for a 100 km deviation, where the direct
  form is accurate. *Measured:* 6.7e-11.
- **Two-body motion is exact** at any step: the deviation is identically zero (q = 0, f(0) = 0, a_p = 0
  to round-off), so one orbit in 8 steps lands on the conic to the Kepler solve's tolerance. *Measured:*
  4.7e-11 km, against RK4's 11,185 km at the same step.
- **Under J2 the error is that of the perturbation**, ~|a_J2| / |a_central| ~ 1e-3 of RK4's at the same
  step, still fourth order. *Measured* (550 km, 6 satellites, 6,400 s): 1.9e-3 km at 160 s against
  RK4's 2.1 km (1,100x), 545x at 10 s; ratio per step doubling 16.6. The 1 km step limit moves from
  80-160 s to 640-1,280 s.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import geopotential, iod, scenarios
from orbital_engine.custom_types import PropagatorType
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ, J2_MODEL
from orbital_engine.gravity import POINT_MASS_MODEL
from orbital_engine.integrators import battin_f, kepler_advance
from orbital_engine.sweep import ForceModelSpec, ModelConfig, run_sweep

MU = scenarios.MU_EARTH
J2 = {"j2": EARTH_J2, "r_eq": EARTH_R_EQ}
OBL = {"Earth": (geopotential.EARTH_J2, geopotential.EARTH_R_EQ)}
FM = (ForceModelSpec("point_mass_gravity"), ForceModelSpec("j2", J2))


def _battin(r: np.ndarray, dr: np.ndarray) -> np.ndarray:
    rref = r - dr
    q = np.einsum("ij,ij->i", dr, dr - 2.0 * r) / np.einsum("ij,ij->i", r, r)
    rho = np.linalg.norm(rref, axis=1)
    out: np.ndarray = -(MU / rho ** 3)[:, None] * (battin_f(q)[:, None] * r + dr)
    return out


def _direct(r: np.ndarray, dr: np.ndarray) -> np.ndarray:
    rref = r - dr
    out: np.ndarray = (-MU * r / np.linalg.norm(r, axis=1)[:, None] ** 3
                       + MU * rref / np.linalg.norm(rref, axis=1)[:, None] ** 3)
    return out


def test_battins_identity() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(size=(5, 3)) * 7000.0
    for scale, tol in ((1e-2, 1e-9), (100.0, 1e-12)):
        dr = rng.normal(size=(5, 3)) * scale
        b, d = _battin(r, dr), _direct(r, dr)
        assert float(np.max(np.abs(b - d)) / np.max(np.abs(b))) < tol


def test_kepler_advance_matches_iod() -> None:
    r0 = np.array([[9000.0, 1000.0, 500.0], [-7000.0, 2000.0, 3000.0]])
    v0 = np.array([[-1.0, 6.5, 1.0], [-2.0, -6.0, 3.0]])
    r, v = kepler_advance(r0, v0, 2345.6, np.full(2, MU))
    for j in range(2):
        rr, vv = iod.kepler_universal(r0[j], v0[j], 2345.6, MU)
        np.testing.assert_allclose(r[j], rr, rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(v[j], vv, rtol=1e-12, atol=1e-11)


def _two_body(session: Session, integrator: str) -> tuple:
    sim = scenarios.two_body(session, p=11000.0, e=0.2)
    k = sim.name_to_index["Secondary"]
    sim.record_history = False
    sim.set_propagator(np.array([k], dtype=np.int64), PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, [k])
    sim.set_cowell_integrator(integrator)
    return sim, k


def test_two_body_is_exact_at_any_step(db_session: Session) -> None:
    sim, k = _two_body(db_session, "encke")
    x0 = (sim.global_states[k] - sim.global_states[sim.parent_indices[k]]).copy()
    period = 2.0 * math.pi * math.sqrt((11000.0 / 0.96) ** 3 / MU)
    for _ in range(8):
        sim.step(period / 8)
    r_true, _ = iod.kepler_universal(x0[:3], x0[3:], period, MU)
    x = sim.global_states[k] - sim.global_states[sim.parent_indices[k]]
    assert float(np.linalg.norm(x[:3] - r_true)) < 1e-8


@pytest.fixture(scope="module")
def build() -> Callable:
    def make():  # type: ignore[no-untyped-def]
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool
        from orbital_engine.database import Base
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        return scenarios.earth_constellation(sessionmaker(bind=engine)(), n_sats=6, n_planes=6)
    return make


def test_j2_error_is_the_perturbations(build: Callable) -> None:
    def errors(dt: float) -> dict:
        cfgs = [ModelConfig("rk4", PropagatorType.COWELL, dt, force_models=FM),
                ModelConfig("encke", PropagatorType.COWELL, dt, force_models=FM, integrator="encke")]
        return {r.config_name: r.error.median_km
                for r in run_sweep(build, cfgs, 6400.0, oblateness=OBL, timing_batches=1, timing_warmup=0)}
    at160, at320 = errors(160.0), errors(320.0)
    assert at160["encke"] < at160["rk4"] / 300.0
    assert 10.0 < at320["encke"] / at160["encke"] < 24.0


def test_refuses_to_step_without_point_mass(db_session: Session) -> None:
    sim = scenarios.two_body(db_session, p=11000.0, e=0.2)
    k = sim.name_to_index["Secondary"]
    sim.set_propagator(np.array([k], dtype=np.int64), PropagatorType.COWELL)
    sim.enable_force_model(J2_MODEL, [k], j2=EARTH_J2, r_eq=EARTH_R_EQ)
    sim.set_cowell_integrator("encke")
    with pytest.raises(ValueError, match="point_mass_gravity"):
        sim.step(60.0)
