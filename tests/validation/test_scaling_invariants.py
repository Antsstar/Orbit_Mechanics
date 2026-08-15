"""
Architectural scaling invariants.

These are performance tests, which the rest of the suite deliberately is not. They are here because
the properties they check are *architectural* rather than micro-optimisations: violating one means a
structural regression - work reintroduced that scales with the wrong quantity - not that a machine
was a few percent slower.

They are written as **ratios between two configurations measured in the same process**, never as
absolute microsecond thresholds. A ratio cancels out machine speed, CI runner variance and
interpreter version, so the bound can be tight in the property while staying loose in the timing.
The bounds below sit two to three orders of magnitude away from the regression they catch: the
capacity test compares a 1024x change in arena size against a 5x allowance in runtime, so noise
would have to be extraordinary to trip it, and a genuine regression would overshoot by ~1000x.

Marked `perf` so they can be deselected with `-m "not perf"` on a contended machine.
"""
from __future__ import annotations

from typing import Callable

import pytest
from sqlalchemy.orm import Session

from orbital_engine import scenarios
from orbital_engine.benchmark import measure

pytestmark = pytest.mark.perf

DT = 3600.0

# A 1024x increase in arena capacity may cost at most 5x in step time. Per-step work that scales
# with capacity rather than with the active set would blow through this by roughly two orders of
# magnitude; ordinary timing noise is nowhere near it.
CAPACITY_RATIO = 1024
CAPACITY_TIME_ALLOWANCE = 5.0

# Doubling body count may cost at most 3x. Linear would be 2x; the headroom covers cache effects
# and the fixed per-step overhead that dominates at small counts.
BODY_COUNT_TIME_ALLOWANCE = 3.0


def test_step_cost_is_independent_of_arena_capacity(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Step cost must track the *active body count*, not `max_capacity`.

    Sizing the arena generously is the normal thing to do when bodies will be spawned later, so a
    per-step cost that scales with capacity is a trap rather than a tuning matter - and it is one
    this engine had. The original propagator allocated and zeroed four `(max_capacity, 3)` scratch
    arrays on every step, so a five-body scenario in a 65536-slot arena cost 2093 us against 479 us
    in a 64-slot arena, having done identical physics.

    The compiled path allocates its scratch once and zeroes only the active head slots.
    """
    small = scenarios.sun_earth_moon(db_session_factory(), capacity=64)
    large = scenarios.sun_earth_moon(db_session_factory(), capacity=64 * CAPACITY_RATIO)

    for sim in (small, large):
        sim.record_history = False
        # Forced explicitly rather than left to auto-detection: the NumPy reference *is* capacity
        # dependent, so on a machine without numba the default would select the path this test is
        # designed to reject and the failure would look like a regression rather than a config.
        sim.use_compiled_kernel = True

    assert small.active_mask.sum() == large.active_mask.sum(), "scenarios differ in body count"

    t_small = measure(lambda: small.step(DT), inner=50).best
    t_large = measure(lambda: large.step(DT), inner=50).best

    ratio = t_large / t_small
    assert ratio < CAPACITY_TIME_ALLOWANCE, (
        f"step cost scales with arena capacity: {CAPACITY_RATIO}x capacity cost {ratio:.1f}x time "
        f"({t_small:.2f} us at 64 slots, {t_large:.2f} us at {64 * CAPACITY_RATIO}). "
        f"Something in the step is sized by max_capacity rather than the active set."
    )


def test_step_cost_grows_no_worse_than_linearly_with_body_count(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Keplerian propagation is independent per body, so cost must be linear in body count.

    Anything super-linear implies an accidental all-pairs interaction - the classic way an N-body
    force loop gets written when only two-body motion was intended. Doubling the constellation is
    allowed to cost up to 3x, against the 2x that linear scaling predicts.
    """
    base = scenarios.earth_constellation(db_session_factory(), n_sats=300, n_planes=6)
    double = scenarios.earth_constellation(db_session_factory(), n_sats=600, n_planes=6)

    for sim in (base, double):
        sim.record_history = False
        sim.use_compiled_kernel = True

    t_base = measure(lambda: base.step(DT), inner=10).best
    t_double = measure(lambda: double.step(DT), inner=10).best

    ratio = t_double / t_base
    assert ratio < BODY_COUNT_TIME_ALLOWANCE, (
        f"step cost grows super-linearly with body count: 2x bodies cost {ratio:.2f}x time "
        f"({t_base:.1f} us at 300 sats, {t_double:.1f} us at 600). "
        f"Suggests an all-pairs loop where per-body work was intended."
    )


def test_history_recording_cost_does_not_dominate_the_step(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Recording a step must cost less than taking one.

    The row-wise recorder built one sixteen-key dict per body per step, which at 600 satellites cost
    1875 us against a 120 us physics step - so a simulation spent 94% of its time describing itself.
    Recording is bookkeeping and must stay cheaper than the work it describes.
    """
    sim = scenarios.earth_constellation(db_session_factory(), n_sats=600, n_planes=6)
    sim.use_compiled_kernel = True

    sim.record_history = False
    physics = measure(lambda: sim.step(DT), inner=10).best

    recording = measure(sim._record_state, inner=10, setup=sim.clear_history).best
    sim.clear_history()

    assert recording < physics, (
        f"history recording ({recording:.1f} us) costs more than the step it records "
        f"({physics:.1f} us)"
    )
