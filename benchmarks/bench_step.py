"""
Step-cost benchmark for the propagation hot loop.

Run with:

    <env>/python.exe benchmarks/bench_step.py

Reports three things, because they answer three different questions:

1. **Cost per step by scenario** - the headline number, and what a regression would show up in.
2. **Cost per step against body count** - separates per-body cost from fixed per-step overhead.
   A flat line here means the engine is dispatch-bound, not arithmetic-bound, and that array sizes
   are irrelevant at this scale.
3. **Component breakdown** - where inside the step the time actually goes.

Nothing here asserts. Thresholds belong in the test suite; this is the instrument, not the gate.
"""
from __future__ import annotations

import sys
from typing import Callable

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import scenarios
from orbital_engine.benchmark import measure
from orbital_engine.database import Base
from orbital_engine.propagators import KeplerianPropagator
from orbital_engine.simulator import Simulation

DT = 3600.0


def fresh_session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def stepper(sim: Simulation) -> Callable[[], None]:
    """A step that does not accumulate history, so the buffer cannot grow during timing."""
    def _step() -> None:
        sim.step(DT)
        sim.clear_history()
    return _step


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def bench_scenarios() -> None:
    rule("Cost per step by scenario")
    print(f"{'scenario':<24}{'bodies':>8}{'tiers':>8}{'us/step':>12}{'noise':>9}")

    cases: list[tuple[str, Callable[[Session], Simulation]]] = [
        ("two_body", lambda s: scenarios.two_body(s)),
        ("sun_earth_moon", lambda s: scenarios.sun_earth_moon(s)),
        ("constellation-60", lambda s: scenarios.earth_constellation(s, n_sats=60)),
    ]
    for name, build in cases:
        sim = build(fresh_session())
        m = measure(stepper(sim))
        n = int(sim.active_mask.sum())
        print(f"{name:<24}{n:>8}{len(sim.topological_tiers):>8}{m.best:>12.1f}{m.noise_ratio:>8.2f}x")


def bench_scaling() -> None:
    rule("Cost per step against body count (flat topology)")
    print(f"{'satellites':>12}{'slots':>8}{'us/step':>12}{'us/body':>12}")

    for n_sats in (6, 12, 60, 240, 600):
        sim = scenarios.earth_constellation(fresh_session(), n_sats=n_sats, n_planes=6)
        m = measure(stepper(sim), inner=20)
        slots = int(sim.active_mask.sum())
        print(f"{n_sats:>12}{slots:>8}{m.best:>12.1f}{m.best / slots:>12.3f}")


def bench_components() -> None:
    rule("Component breakdown (sun_earth_moon)")
    sim = scenarios.sun_earth_moon(fresh_session())

    def propagate() -> None:
        KeplerianPropagator.propagate(
            dt=DT, primary_states=sim.coe_states, secondary_states=sim.local_states,
            mu_array=sim.mu_array, parent_indices=sim.parent_indices,
            active_mask=sim.active_mask, is_head=sim.is_head, is_system=sim.is_system,
            body_sys_map=sim.body_sys_map, sys_head_map=sim.sys_head_map,
        )

    parts: list[tuple[str, Callable[[], object]]] = [
        ("propagate", propagate),
        ("calc_global", sim.calc_global),
        ("_record_state", sim._record_state),
        ("full step", stepper(sim)),
    ]
    print(f"{'component':<20}{'us':>10}{'share':>9}")
    results = [(label, measure(fn).best) for label, fn in parts]
    total = dict(results)["full step"]
    for label, us in results:
        print(f"{label:<20}{us:>10.1f}{us / total * 100:>8.0f}%")


def main() -> int:
    bench_scenarios()
    bench_scaling()
    bench_components()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
