"""
Event-driven step splitting: cut a step at the instant a continuous event function changes sign.

This is the generalisation of the scheduled-epoch split `manoeuvres.py` already has. A scheduled
impulse knows its epoch in advance, so `Simulation.step` can simply cut there. An *event* does not:
its epoch is defined implicitly, as the root of a scalar function of the arena state, and has to be
found during the step. Everything downstream is the same machinery - the step is cut, `_advance` runs
twice, and the clock is pinned to `t + dt` at the end.

Why this exists
---------------
`srp.py`'s cylindrical shadow makes the acceleration discontinuous at the terminator: `nu` steps
between 0 and 1 across a surface of zero thickness, so `a` jumps by the full
`C_r P_srp (A/m) (AU/d)^2`. An RK4 step that straddles that surface has its four stages disagreeing
about which side the body is on, and the switch is effectively mistimed by up to `h`. The resulting
velocity error is `~ Delta_a h / 2` per crossing - **first order in `h`**, not fourth. It is measured
in `tests/validation/test_srp.py`: step-halving difference ratios of 4.23 then 1.96 where a smooth
problem gives 16, and 7x the error of running with no shadow at all.

Smoothing the model (the conical shadow) hides the symptom. The general fix is to **stop the step at
the discontinuity**, so that no step ever straddles it and every step integrates a function that is
smooth over its whole interval. RK4's order then comes back, and what is left is the error of not
knowing the crossing time exactly - see "What this costs" below, which is the number the validation
suite asserts.

The event interface
-------------------
An `Event` is plain frozen data, like `manoeuvres.Manoeuvre`:

    Event(name, function, bodies, direction, tol_s)

`function(sim, bodies) -> (k,)` returns one float per body of `bodies`, in *any* unit: only its sign
and its continuity matter. It must be a **pure read** of the arena - `Simulation` evaluates it at
trial times inside the step and relies on the evaluation leaving no trace, which is also what keeps a
step with no crossing bit-identical to a step with no events registered at all.

- **`direction`** filters which sign changes fire, the same convention `scipy.integrate.solve_ivp`
  uses: `+1` for negative-to-positive only, `-1` for positive-to-negative only, `0` (the default) for
  both. A discontinuity wants `0` - eclipse entry and exit are equally sharp. An apoapsis event would
  be `function = r . v`, `direction = -1`.
- **`tol_s`** is the time tolerance the crossing is located to. See `DEFAULT_EVENT_TOL_S`.

Ready-made events: `shadow_event(sim)`, the cylindrical umbra boundary of every `"srp"` body that has
one. A future altitude-threshold, apoapsis or elevation-mask event is the same three lines - a pure
array function of the state, and a slot list.

Per body, or per arena?
-----------------------
**Detected per body; split per arena, and that is not a compromise.** `function` returns one value
per body and each body's sign change is tracked separately, so a constellation in which one satellite
is entering eclipse and another is leaving it registers two independent crossings. But the *split* is
a property of the clock, not of a body: `Simulation._advance` advances the whole arena by one `dt`,
because `calc_global` and the Cowell/secular-J2 re-base all assume every row is at the same instant.
Per-body step sizes would mean rows at different times inside one arena, which is a different engine.

So when one body crosses and another does not, the step is cut at the crossing body's root and
**every** body takes two sub-steps instead of one. What that costs a body that is not crossing is
exactly what `manoeuvres.py`'s split costs it, and is already derived there: nothing at all for an
analytic (Keplerian or secular-J2) body, since a closed-form advance of `h1` then `h2` is the advance
of `h1 + h2`; and one extra local truncation error `~ r (n h)^5 / 120` for a Cowell body, whose RK4
quadrature nodes move when the interval is cut. That is a one-off per split, not a per-step bias.

The cost model that follows is linear in crossing bodies: `n_sats` satellites each eclipsed twice per
orbit produce `2 n_sats` splits per orbit, and the arena pays for all of them. Independently timing a
hundred satellites would be cheaper per satellite - and would not be this engine.

Locating the crossing
---------------------
The engine has no dense output: there is no way to ask for the state at `t + tau` without advancing
to it. So the root find works by **trial propagation** - snapshot the arena, `_advance(tau)`,
evaluate, restore - and `Simulation` owns the snapshot (`_capture_event_state`). The function handed
to `locate_crossing` is that whole round trip.

Several bodies may cross inside one interval, and the split has to land on the *earliest* of them.
Rather than running one root find per crossing body, the vector is reduced to a single scalar:

    h_j(tau) = -sign(g_j(0)) * g_j(tau)     (negative before body j crosses, positive after)
    H(tau)   = max over crossing j of h_j(tau)

`H(0) < 0`, `H(dt) > 0`, `H` is continuous, and `H(tau) >= 0` exactly when *some* body has already
crossed - so `H`'s first zero is the earliest crossing time, and one scalar root find serves any
number of bodies at one trial propagation per iteration.

The root find is **false position with the Illinois modification**, safeguarded by a bisection step
whenever the interpolant falls too near an endpoint. It keeps a valid bracket at every iteration
(unlike a secant method, which can leave one), and converges superlinearly on the near-linear `g` a
geometric event produces, where plain bisection would need `log2(dt / tol_s)` iterations - 24 for
`dt = 10 s` at the default tolerance. Termination is on **bracket width**, not on `|H|`, because the
quantity that has to be small is a *time*: `|H|` small means nothing without the slope.

The returned time is the bracket's **far** endpoint - the side on which `H >= 0`, i.e. past the
crossing - not its midpoint. Two reasons. It makes the error one-sided and bounded
(`0 <= tau - tau_true <= tol_s`) rather than two-sided, and it guarantees the state at the split has
the *post*-crossing sign, so the re-scan of the remainder cannot rediscover the crossing it just
resolved and stall.

Multiple crossings in one step are handled by re-scanning: after splitting at the earliest root the
remainder is scanned again, up to `Simulation.max_event_splits`. That is what makes a step longer
than an eclipse (`dt = 3600 s` against a ~2100 s LEO eclipse) resolve *both* the entry and the exit.

What this costs, and what it buys
---------------------------------
Per crossing: one extra `g` evaluation pair per step (negligible - it is a handful of dot products
over the event bodies), plus `n_iter` trial propagations, each of which is one `_advance` of the whole
arena, plus the final split advance. At the default tolerance `n_iter` is measured at 10-12 for a LEO
terminator crossing from a 10 s step, so a crossing step costs roughly 12x a plain step, and a step
with no crossing costs the same as today plus the snapshot copies. Both counters are exposed:
`Simulation.event_splits` and `Simulation.event_evaluations`.

What it buys, derived before measuring (`tests/validation/test_events.py` is where this is asserted):

- **RK4's order comes back.** With no step straddling the discontinuity, every step integrates a
  smooth right-hand side, so the step-halving difference ratio returns to the fourth-order band. It
  does not return to a clean 16, and must not be asserted to: the crossing *times* also move with `h`
  (the trial propagations that locate them are themselves `h`-dependent), and RK4's asymptotic regime
  needs the error to be dominated by one term.
- **The residual is the tolerance.** Locating a crossing `eps` seconds late applies the wrong
  acceleration for `eps`, giving a velocity error `Delta_a * eps` which then grows into position error
  linearly in the time remaining, `T_rem`. Over `n` crossings that is `n * Delta_a * eps * T_rem`. For
  `srp.py`'s validation case (`Delta_a = 1.19e-9 km/s^2`, 4 crossings, mean `T_rem = 4400 s`) it is
  `2.1e-5 * eps` km - `2.1e-11 km` at the default tolerance, two orders below RK4's own truncation at
  the *finest* step of the convergence ladder (`~2.3e-9 km` at `h = 1.25 s`). That headroom is the
  whole reason the default is `1e-6` and not `1e-3`.

`DEFAULT_EVENT_TOL_S` is not made smaller than that because it cannot usefully be: the shadow
geometry is differenced out of heliocentric `global_states` (`~1.5e8 km`), so `g` carries `~1e-7 km`
of rounding, which at a terminator closing speed of order `1 km/s` is `~1e-7 s` of time noise. The
default sits one decade above that floor.

Limitations
-----------
- **Endpoint sampling.** A crossing is detected from the sign of `g` at the two ends of an interval.
  A body that crosses an *even* number of times inside one interval shows no sign change and fires
  nothing - the classic event-detection blind spot, shared with `scipy.integrate.solve_ivp`. The
  defence is the step size: an eclipse lasts thousands of seconds. Splitting and re-scanning handles
  any number of crossings as long as no *sub-interval* contains an even number of them.
- **No tangential events.** A `g` that touches zero without changing sign (a grazing eclipse) is
  invisible for the same reason. Locating it would need a minimum of `g` rather than a root.
- **Not free for analytic bodies.** The trial propagations advance the whole arena, not just the
  crossing body, because there is no cheaper state-at-time query.
- **No integration with an adaptive integrator.** See `docs/architecture.md`.

References
----------
- Event location by bracketed root finding on a continuous event function, with a `direction` filter:
  Hairer, Norsett & Wanner, *Solving Ordinary Differential Equations I*, 2nd ed., Sec. II.6
  ("Dense Output, Discontinuities"); the same interface `scipy.integrate.solve_ivp`'s `events=`
  argument exposes. **Section number from memory and unverified against the text.**
- **Stopping the step at a discontinuity, rather than integrating through it,** is the standard
  remedy: a Runge-Kutta method's order derives from a Taylor expansion of the solution over the step,
  which does not exist when the right-hand side is discontinuous inside it. Hairer & Wanner,
  *Solving Ordinary Differential Equations II*, Sec. IV.6, and Gear & Osterby (1984), "Solving
  Ordinary Differential Equations with Discontinuities", ACM TOMS 10(1), 23-44. **Unverified.**
- The Illinois modification of false position: Dowell & Jarratt (1971), "A modified regula falsi
  method for computing the root of an equation", BIT 11, 168-174. **Unverified.** The method is
  re-derived in `locate_crossing`'s own docstring and is elementary either way; the validation case
  for it is `tests/validation/test_events.py`'s analytic terminator time, not the citation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Final, Optional, TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .custom_types import ArrayFloat
from .srp import SRP_MODEL, cylindrical_shadow_bodies, latch_shadow_branch, shadow_clearance

if TYPE_CHECKING:
    from .simulator import Simulation

__all__ = [
    "Event", "EventFunction", "LatchFunction", "DEFAULT_EVENT_TOL_S", "MAX_SPLITS_PER_STEP",
    "MAX_CROSSING_NUDGES", "crossing_indices", "locate_crossing", "reduce_to_scalar", "shadow_event",
]


#: An event function: `(sim, bodies) -> (k,)`, one value per body, whose **sign change** marks the
#: event. Units are free; only the sign and the continuity matter. It must not mutate the arena.
EventFunction: TypeAlias = Callable[["Simulation", NDArray[np.int64]], ArrayFloat]

#: A branch latch: `(sim, bodies, clearance_sign) -> None`, where `clearance_sign` is the sign of the
#: event function at the start of a sub-interval known to contain no crossing, or `None` to release.
#: Only a *discontinuous* model needs one; see `Event.latch` and `srp.latch_shadow_branch`.
LatchFunction: TypeAlias = Callable[
    ["Simulation", NDArray[np.int64], Optional[ArrayFloat]], None]


#: Time tolerance a crossing is located to, seconds. Derived in the module docstring: it puts the
#: residual position error of `srp.py`'s validation case at `2.1e-11 km`, two orders below RK4's own
#: truncation at the finest step of that convergence ladder, and one decade above the `~1e-7 s`
#: rounding floor the heliocentric geometry imposes on the event function itself.
DEFAULT_EVENT_TOL_S: Final[float] = 1.0e-6

#: Default cap on how many times one call to `Simulation.step` may cut its step. Exceeding it raises
#: rather than silently dropping the remaining crossings - a dropped crossing is exactly the
#: plausible-looking, non-raising error this module exists to remove. Override per simulation with
#: `Simulation.max_event_splits`; the usual reason to need more is a `dt` spanning many eclipses.
MAX_SPLITS_PER_STEP: Final[int] = 32

#: How many extra micro-steps of one tolerance each `Simulation._advance_with_events` may take to
#: force the arena strictly past a located crossing. It normally takes none, and never more than one
#: in anything measured here. The need for any is subtle: `locate_crossing`'s bracket is a property
#: of the *trial* trajectory (one advance of length `tau` from the interval start), while the split
#: follows a slightly different one (an advance to `tau_lo`, then the micro-step). Within a tolerance
#: of the surface those two can disagree about which side they are on, and a split that lands on the
#: near side leaves the next sub-interval's first stage reading the pre-crossing branch - the exact
#: first-order error being removed. So the postcondition is *checked*, not assumed, and each nudge
#: costs `Delta_a tol_s`, the same order as the tolerance residual itself.
MAX_CROSSING_NUDGES: Final[int] = 8

#: Hard cap on root-find iterations. Bisection alone reaches `1e-12 s` from a 3600 s step in 52
#: iterations, so exhausting this with a sane tolerance means the event function is pathological
#: (non-finite, or changing sign on every evaluation), which raises rather than returning a
#: quietly-unconverged root.
MAX_ROOT_ITERATIONS: Final[int] = 80



@dataclass(frozen=True, eq=False)
class Event:
    """
    One event: the sign change of `function` over `bodies`, located to `tol_s` seconds.

    Plain frozen data with no reference to a `Simulation`, exactly like `manoeuvres.Manoeuvre`, so a
    configuration is a list of these and nothing else. `eq=False` because the dataclass-generated
    `__eq__` would compare `bodies` elementwise and return an array, not a bool.

    Parameters
    ----------
    name : str
        For error messages and diagnostics only.
    function : EventFunction
        `(sim, bodies) -> (k,)`. A **pure read** of the arena; see the module docstring.
    bodies : (k,) int64
        Arena slots, in the order `function` returns values for.
    direction : int
        `+1` fires only on negative-to-positive, `-1` only on positive-to-negative, `0` on both.
    tol_s : float
        Bracket width the crossing time is converged to, seconds.
    latch : LatchFunction or None
        Only needed by an event that marks a **discontinuity** in the acceleration. Splitting the
        step is not by itself enough for one: RK4's stages sample points *off* the trajectory, by
        `O(h^3)` in position, so a sub-step that ends exactly at the crossing still has a stage
        landing on the far side of the surface - weight 1/6 of a full `Delta_a h`, which is first
        order and is the whole error being removed. The latch pins the model to one branch for the
        duration of a sub-interval known to contain no crossing, which makes that sub-interval's
        right-hand side genuinely smooth. An event that marks something *continuous* (apoapsis, an
        altitude threshold, an elevation mask) needs none and leaves this `None`.
    """

    name: str
    function: EventFunction
    bodies: NDArray[np.int64]
    direction: int = 0
    tol_s: float = DEFAULT_EVENT_TOL_S
    latch: Optional[LatchFunction] = None
    label: str = field(default="", compare=False)


def crossing_indices(
    g0: ArrayFloat, g1: ArrayFloat, direction: ArrayFloat,
) -> NDArray[np.int64]:
    """
    Which components changed sign between the two ends of an interval, as an integer index array.

    Parameters
    ----------
    g0, g1 : (K,)
        Event function values at the start and the end of the interval, concatenated over every
        registered event.
    direction : (K,)
        Per-component direction filter, `+1`, `-1` or `0`, in the same concatenated order.

    Returns an **integer index array** rather than a boolean mask, so the caller's "did anything
    cross?" test is `.size` rather than a reduction over an arena-sized array (`CLAUDE.md`).

    A component whose `g0` is exactly `0.0` never fires. That is what stops a split - which lands the
    arena on the event surface to within `tol_s` - from rediscovering its own crossing on the
    re-scan, in the one case where the far endpoint's value rounds to exactly zero. Non-finite values
    never fire either; an event function that has gone NaN must not silently cut every step in half.
    """
    finite = np.isfinite(g0) & np.isfinite(g1)
    changed = finite & (g0 != 0.0) & ((g0 > 0.0) != (g1 > 0.0))
    rising = g1 > g0
    wanted = (direction == 0.0) | ((direction > 0.0) & rising) | ((direction < 0.0) & ~rising)
    out: NDArray[np.int64] = np.flatnonzero(changed & wanted).astype(np.int64)
    return out


def reduce_to_scalar(g: ArrayFloat, sign0: ArrayFloat) -> float:
    """
    `H(tau) = max_j (-sign0_j * g_j(tau))`: one scalar whose first zero is the earliest crossing.

    `sign0_j` is the sign of component `j` at the *start* of the interval, so `-sign0_j * g_j` is
    negative while body `j` is still on its original side and positive once it has crossed. Taking
    the maximum makes `H >= 0` exactly when at least one body has crossed, which puts `H`'s first
    zero at the earliest crossing time - see the module docstring. `g` and `sign0` are already
    restricted to the crossing components by the caller.
    """
    return float(np.max(-sign0 * g))


def locate_crossing(
    h_at: Callable[[float], float],
    lo: float,
    hi: float,
    h_lo: float,
    h_hi: float,
    tol_s: float,
    max_iter: int = MAX_ROOT_ITERATIONS,
) -> tuple[float, float, int]:
    """
    Locate the root of `h_at` in the bracket `[lo, hi]`, to a bracket width of `tol_s`.

    False position with the Illinois modification. Plain false position keeps one endpoint fixed
    whenever `h` is convex over the bracket, so the bracket width stops shrinking even though the
    iterate converges; Illinois halves the *stagnant* endpoint's function value after two consecutive
    updates on the same side, which drags the interpolant across and collapses the bracket
    superlinearly. A bisection is substituted only when the interpolant does not land *strictly*
    inside the bracket - a floating-point guard, not a convergence one; see the comment at that line
    for what the stronger safeguard that was there first cost in measured iteration count.

    Parameters
    ----------
    h_at : (float) -> float
        The event scalar at an offset from `lo`'s reference. In `Simulation` this is a whole
        snapshot / trial-advance / evaluate / restore round trip, which is why the iteration count is
        the cost that matters and not the arithmetic.
    lo, hi : float
        The bracket, `lo < hi`, with `h_lo < 0 <= h_hi`.
    h_lo, h_hi : float
        `h_at(lo)` and `h_at(hi)`, already known to the caller from the step's own endpoints - which
        is why they are passed rather than re-evaluated.
    tol_s : float
        Termination is on **bracket width**, not on `|h|`: the quantity that must be small is a time.

    Returns
    -------
    (lo, hi, n_evals)
        The converged bracket, `hi - lo <= tol_s`, straddling the crossing: `h(lo) < 0` (nothing has
        crossed yet) and `h(hi) >= 0` (something has). **Both** endpoints are returned, not a single
        root, because the caller needs both sides - see `Simulation._advance_with_events` for why a
        single split point cannot be right for a *discontinuous* right-hand side. `n_evals` counts
        calls to `h_at`.

    Raises `RuntimeError` if `max_iter` is reached with the bracket still wider than `tol_s`; see
    `MAX_ROOT_ITERATIONS` for why that cannot happen for a well-behaved event function.
    """
    n = 0
    side = 0
    while hi - lo > tol_s:
        if n >= max_iter:
            raise RuntimeError(
                f"event root find did not converge: bracket [{lo!r}, {hi!r}] is still "
                f"{hi - lo:.3e} s wide after {n} evaluations, against a tolerance of {tol_s:.3e} s. "
                f"The event function is not behaving like a continuous scalar (non-finite, or "
                f"changing sign on every evaluation). See events.py."
            )
        span = hi - lo
        denom = h_hi - h_lo
        mid = lo + span * (-h_lo / denom) if denom > 0.0 else lo + 0.5 * span
        # Bisect only when the interpolant is not *strictly* inside the bracket - which floating
        # point produces once the bracket is a few ulp wide, or a denormal `denom` at any width.
        # The guard is deliberately this weak. Keeping the trial away from the ends by a *fraction*
        # of the span instead forces a bisection on essentially every iteration, because a
        # converging false position legitimately places its trial arbitrarily close to one end -
        # measured at 23.5 trial propagations per crossing that way, against 8 for plain Illinois.
        if not lo < mid < hi:
            mid = lo + 0.5 * span

        h_mid = h_at(mid)
        n += 1
        if h_mid < 0.0:
            lo, h_lo = mid, h_mid
            if side == -1:
                h_hi *= 0.5
            side = -1
        else:
            hi, h_hi = mid, h_mid
            if side == 1:
                h_lo *= 0.5
            side = 1

    return lo, hi, n


# ==================================================================================================
# Ready-made events
# ==================================================================================================

def shadow_event(
    sim: "Simulation",
    bodies: Optional[NDArray[np.int64]] = None,
    tol_s: float = DEFAULT_EVENT_TOL_S,
) -> Event:
    """
    The cylindrical umbra boundary of every `"srp"` body that casts one: the discontinuity this
    module exists for.

    `bodies` defaults to `srp.cylindrical_shadow_bodies(sim)` - the slots that carry the `"srp"` mask
    bit **and** a positive `r_occ` **and** the cylindrical shadow model. A conical body is
    deliberately excluded: its `nu` is continuous, so there is no discontinuity to stop at, and
    splitting for it would buy nothing and cost trial propagations. A body with `r_occ <= 0` has no
    occulter at all.

    `direction=0`: eclipse entry and exit are the same discontinuity seen from two sides, and both
    cost RK4 its order.

    The event function is `srp.shadow_clearance`, which reads `sim.force_model_params["srp"]` live,
    so re-enabling the model with a different `r_occ` after the event is added is picked up. The
    *body list* is captured here and is not; re-add the event after changing which bodies carry SRP.
    Raises `ValueError` if the resulting list is empty, rather than registering an event that can
    never fire.
    """
    slots = cylindrical_shadow_bodies(sim) if bodies is None else np.asarray(bodies, dtype=np.int64)
    if slots.size == 0:
        raise ValueError(
            f"no body qualifies for a '{SRP_MODEL}' shadow event: a body needs the '{SRP_MODEL}' "
            f"force model enabled, a positive 'r_occ' and the cylindrical shadow model. A conical "
            f"shadow is continuous and has no discontinuity to split at."
        )
    return Event(
        name=f"{SRP_MODEL}.umbra", function=shadow_clearance, bodies=slots, direction=0,
        tol_s=tol_s, latch=latch_shadow_branch,
    )
