"""
Step-cost benchmark for the propagation hot loop.

Run with:

    <env>/python.exe benchmarks/bench_step.py

Reports five things, because they answer five different questions:

1. **Reference against compiled kernel** - the headline. Both compute the same thing, to within the
   tolerance asserted in `tests/validation/test_kernel_equivalence.py`, so the ratio is pure cost.
   Keplerian propagation, then the fused Cowell RK4 step per force-model set (point mass, + j2,
   + J3..J6 zonal), with the full `step()` cost alongside so the kernel's share is visible.
2. **Cost per step by scenario** - what a performance regression would show up in.
3. **Cost per step against body count** - separates per-body cost from fixed per-step overhead. A
   flat line means the engine is dispatch-bound rather than arithmetic-bound.
4. **Component breakdown** - where inside a step the remaining time goes.

History recording is disabled throughout. It appends one dict per body per step, which is the right
behaviour for analysis and pure overhead for a sweep that reads only final state; leaving it on would
measure the recorder rather than the physics.

Nothing here asserts. Thresholds belong in the test suite; this is the instrument, not the gate.
"""
from __future__ import annotations

import sys
from typing import Callable

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import geopotential, kernels, scenarios, zonal
from orbital_engine.benchmark import measure
from orbital_engine.custom_types import PropagatorType
from orbital_engine.database import Base
from orbital_engine.integrators import RK4Integrator
from orbital_engine.propagators import KeplerianPropagator
from orbital_engine.simulator import Simulation

DT = 3600.0

Builder = Callable[[Session], Simulation]

SCENARIOS: list[tuple[str, Builder]] = [
    ("two_body", lambda s: scenarios.two_body(s)),
    ("sun_earth_moon", lambda s: scenarios.sun_earth_moon(s)),
    ("constellation-60", lambda s: scenarios.earth_constellation(s, n_sats=60)),
    ("constellation-600", lambda s: scenarios.earth_constellation(s, n_sats=600)),
]


def fresh_session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def build(builder: Builder) -> Simulation:
    sim = builder(fresh_session())
    sim.record_history = False
    return sim


def reference_call(sim: Simulation) -> Callable[[], None]:
    def _run() -> None:
        KeplerianPropagator.propagate(
            dt=DT, primary_states=sim.coe_states, secondary_states=sim.local_states,
            mu_array=sim.mu_array, parent_indices=sim.parent_indices,
            active_mask=sim.active_mask, is_head=sim.is_head, is_system=sim.is_system,
            body_sys_map=sim.body_sys_map, sys_head_map=sim.sys_head_map,
        )
    return _run


def kernel_call(sim: Simulation) -> Callable[[], None]:
    def _run() -> None:
        kernels.kepler_propagate(
            DT, sim.coe_states, sim.local_states, sim.mu_array, sim.parent_indices,
            sim.body_sys_map, sim.sys_head_map, sim.is_system,
            sim._sib_idx, sim._head_idx, sim._kick, sim._accum,
        )
    return _run


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def bench_propagators() -> None:
    rule("Propagator: NumPy reference against compiled kernel")
    print(f"{'scenario':<20}{'slots':>7}{'reference':>12}{'kernel':>10}{'speedup':>10}{'us/body':>10}")

    for name, builder in SCENARIOS:
        sim = build(builder)
        slots = int(sim.active_mask.sum())
        inner = 50 if slots < 400 else 10
        ref = measure(reference_call(sim), inner=inner).best
        ker = measure(kernel_call(sim), inner=inner).best
        print(f"{name:<20}{slots:>7}{ref:>11.1f}u{ker:>9.2f}u{ref / ker:>9.1f}x{ker / slots:>10.3f}")


COWELL_DT = 60.0

# Cowell force-model sets the fused kernel implements, cheapest first. "zonal" is J3..J6 on top of j2.
COWELL_MODEL_SETS: list[tuple[str, tuple[str, ...]]] = [
    ("pm", ("point_mass_gravity",)),
    ("pm+j2", ("point_mass_gravity", "j2")),
    ("pm+j2+zonal", ("point_mass_gravity", "j2", "zonal")),
]


def build_cowell(n_sats: int, models: tuple[str, ...]) -> tuple[Simulation, np.ndarray]:
    """`n_sats` Cowell satellites at 550 km / 53 deg carrying `models`, EGM96 coefficients."""
    sim = build(lambda s: scenarios.earth_constellation(s, n_sats=n_sats, n_planes=6))
    sats = np.asarray(
        [i for n, i in sim.name_to_index.items() if n.startswith("SAT-")], dtype=np.int64)
    sim.set_propagator(sats, PropagatorType.COWELL)
    coefficients: dict[str, dict[str, float]] = {
        "point_mass_gravity": {},
        "j2": {"j2": geopotential.EARTH_J2, "r_eq": geopotential.EARTH_R_EQ},
        "zonal": {"r_eq": geopotential.EARTH_R_EQ, **zonal.EARTH_ZONALS},
    }
    for model in models:
        sim.enable_force_model(model, sats.tolist(), **coefficients[model])
    return sim, sats


def bench_cowell() -> None:
    rule("Cowell RK4 step: NumPy RK4Integrator against fused kernels.cowell_rk4_step")
    print(f"{'models':<14}{'sats':>6}{'reference':>12}{'kernel':>10}{'speedup':>10}{'us/body':>10}"
          f"{'full step':>11}")

    for n_sats in (12, 60):
        for label, models in COWELL_MODEL_SETS:
            sim, sats = build_cowell(n_sats, models)
            assert sim._cowell_fused_ok, f"{label} must qualify for the fused kernel"
            primaries = sim.parent_indices[sats]
            integrator = RK4Integrator(sim.max_capacity)

            def reference() -> None:
                integrator.step(sim.accelerations, sim.t, sim.global_states, COWELL_DT, sats, primaries)

            def kernel() -> None:
                kernels.cowell_rk4_step(
                    COWELL_DT, sim.global_states, sim.mu_array, sim.parent_indices, sats,
                    sim._cowell_has_point_mass, sim._cowell_has_j2, sim._cowell_j2_params,
                    sim._cowell_has_zonal, sim._cowell_zonal_params, sim._cowell_rel,
                )

            ref = measure(reference, inner=50).best
            ker = measure(kernel, inner=200).best
            full = measure(lambda: sim.step(COWELL_DT), inner=50).best
            print(f"{label:<14}{n_sats:>6}{ref:>11.1f}u{ker:>9.2f}u{ref / ker:>9.1f}x"
                  f"{ker / n_sats:>10.3f}{full:>10.1f}u")


def bench_scenarios() -> None:
    rule("Full step by scenario")
    print(f"{'scenario':<20}{'slots':>7}{'tiers':>7}{'us/step':>10}{'noise':>9}")

    for name, builder in SCENARIOS:
        sim = build(builder)
        slots = int(sim.active_mask.sum())
        m = measure(lambda: sim.step(DT), inner=50 if slots < 400 else 10)
        print(f"{name:<20}{slots:>7}{len(sim.topological_tiers):>7}{m.best:>10.1f}{m.noise_ratio:>8.2f}x")


def bench_scaling() -> None:
    rule("Full step against body count (flat topology)")
    print(f"{'satellites':>12}{'slots':>8}{'us/step':>12}{'us/body':>12}")

    for n_sats in (6, 12, 60, 240, 600, 2400):
        sim = build(lambda s: scenarios.earth_constellation(s, n_sats=n_sats, n_planes=6))
        slots = int(sim.active_mask.sum())
        m = measure(lambda: sim.step(DT), inner=50 if slots < 400 else 10)
        print(f"{n_sats:>12}{slots:>8}{m.best:>12.1f}{m.best / slots:>12.3f}")


def bench_components() -> None:
    for label, builder in (("sun_earth_moon", SCENARIOS[1][1]), ("constellation-600", SCENARIOS[3][1])):
        rule(f"Component breakdown ({label})")
        sim = build(builder)
        slots = int(sim.active_mask.sum())
        inner = 50 if slots < 400 else 10

        propagate = kernel_call(sim) if sim.use_compiled_kernel else reference_call(sim)
        parts: list[tuple[str, Callable[[], object]]] = [
            ("propagate", propagate),
            ("calc_global", sim.calc_global),
            ("_record_state", sim._record_state),
        ]
        total = measure(lambda: sim.step(DT), inner=inner).best

        print(f"{'component':<20}{'us':>10}{'share':>9}   (history off in `full step`)")
        for name, fn in parts:
            us = measure(fn, inner=inner).best
            print(f"{name:<20}{us:>10.2f}{us / total * 100:>8.0f}%")
            if name == "_record_state":
                sim.clear_history()
        print(f"{'full step':<20}{total:>10.2f}{100:>8.0f}%")


def main() -> int:
    print(f"numba available: {kernels.NUMBA_AVAILABLE}")
    bench_propagators()
    bench_cowell()
    bench_scenarios()
    bench_scaling()
    bench_components()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
