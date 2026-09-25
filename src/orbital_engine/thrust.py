"""
Continuous rocket thrust with propellant depletion, registered as the force model `"thrust"`.

This is the first model whose per-body coefficients are not constant: a thrusting vessel burns mass,
so the acceleration it produces grows over a burn even at fixed throttle, and stops dead when the
tanks are empty. It composes additively with `point_mass_gravity`, `j2`, `drag` and `third_body` on a
Cowell body, exactly like every other model.

Model
-----
For body `i` with parent `P = parent_indices[i]`, let `r = state[i,:3] - state[P,:3]` and
`v = state[i,3:] - state[P,3:]`. With `(R, S, W)` the satellite-based RSW basis of `(r, v)`
(`frames.ReferenceFrames.RSW_basis`; R radial-out, W along `r x v`, S = W x R along-track):

    a = (T / m) * (d_R R + d_S S + d_W W) * 1e-3              km/s^2
    m_dot = T / (Isp * g0)                                     kg/s
    m(t + dt) = max(m(t) - m_dot dt, m_dry)

`T` is newtons, `m` kilograms, `Isp` seconds, `g0 = 9.80665 m/s^2` exactly. `(d_R, d_S, d_W)` is the
direction law, per body, in RSW: `(0, 1, 0)` is a prograde (horizontal, along-track) burn,
`(1, 0, 0)` is radial-out, `(0, 0, 1)` is normal. **S is the along-track axis, not the velocity
direction** - the two coincide only where the flight-path angle is zero (circular orbits, apsides);
see `RSW_basis`'s docstring.

Citations
---------
- Thrust acceleration `a = T/m` and mass flow `m_dot = T/(Isp g0)`, and the Tsiolkovsky result
  `Delta v = Isp g0 ln(m0/m1)` that follows by integrating `dv/dt = T/m` along a fixed direction:
  Curtis, *Orbital Mechanics for Engineering Students*, 3rd ed., Sec. 11.2-11.3 (Eq. 11.6 for the
  rocket equation); Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Sec. 6.2
  (Eq. 6-1) for the same and Sec. 8.7 for continuous low-thrust. **Equation numbers are from memory
  and unverified**, as in `drag.py`; the section numbers are more likely right than the equation
  numbers. Nothing here depends on the citation: `a = T/m` is Newton's second law and
  `m_dot = T/(Isp g0)` is the *definition* of specific impulse.
- RSW frame definition: Vallado 4e, Sec. 3.3, implemented and cited in `frames.py`. This module does
  not re-derive the basis; it calls `ReferenceFrames.RSW_to_cart`.
- Orbit-raising rate `da/dt = 2 a_S / n` for a small along-track acceleration on a near-circular
  orbit: the `e -> 0` limit of Gauss's variational equation for `a` (Vallado 4e, Sec. 9.3;
  Battin Sec. 10.3). Derived from energy in `tests/validation/test_thrust.py`'s docstring, which is
  the version that should be checked.

The physics checks that do not rely on the citations are in `tests/validation/test_thrust.py`: a
closed-form acceleration with the RSW rotation recomputed by hand, the rocket equation against a
thrust-free twin, orbit raising against the energy-derived `da/dt`, and exact burnout behaviour.

Units: the single conversion
----------------------------
| param        | column | unit |
|---|---|---|
| `thrust_n`   | 0 | newtons (kg m / s^2) |
| `isp_s`      | 1 | seconds |
| `mass_kg`    | 2 | kilograms, **current total wet mass**; mutated by `deplete_mass` |
| `dry_mass_kg`| 3 | kilograms, the floor `mass_kg` may never cross |
| `dir_r`      | 4 | RSW radial component of the direction law |
| `dir_s`      | 5 | RSW along-track component |
| `dir_w`      | 6 | RSW cross-track component |

`T [N] / m [kg]` is m/s^2 and the engine works in km/s^2, so **the only unit conversion is
`_M_PER_S2_TO_KM_PER_S2 = 1e-3`**, applied once to `T/m`. `m_dot = T/(Isp g0)` is SI throughout and
needs none: kilograms and seconds are already the engine's units for both.

Where the mass lives, and why
-----------------------------
**In column 2 of `force_model_params["thrust"]`, mutated once per step by `Simulation.step`.** The
alternatives were an arena array (`Simulation.mass_array`, alongside `mu_array`) or reading
`VesselORM.dry_mass`/`fuel_mass` live.

- `forces.ForceKernel` gives a kernel exactly one per-body data channel, `params`. A mass that lived
  anywhere else would have to be copied into `params` before every evaluation, or the signature would
  have to grow - and it is shared by every model, none of which needs mass. (Drag does not: its
  `ballistic_coeff` already folds `1/m` in.)
- An arena array would say "mass is state every model may read", which is a larger claim than the
  engine can back. `mu_array` is the gravitational mass and is *not* updated here: a thrusting body
  must be massless (`validate_bodies` below enforces it), so there is nothing to keep consistent.
  When a massive thrusting body is ever wanted, that is the point to promote mass to the arena, and
  the promotion is mechanical.
- The ORM is read at ingest only. `VesselORM.dry_mass + fuel_mass` is the natural *seed* for
  `mass_kg` and `dry_mass` for `dry_mass_kg`, and `scenarios.powered_vessel` does exactly that. A
  step must not touch a stateful third-party object (`CLAUDE.md`), so the ORM row is never consulted
  after build.

The cost of the choice is that `force_model_params["thrust"]` is the one coefficient array that is
**not** idempotent across runs: re-running a scenario needs `mass_kg` re-seeded, the same way an
initial state does. A sweep that calls `enable_force_model("thrust", ..., mass_kg=...)` per
configuration gets that for free.

The depletion itself is `deplete_mass`, called by `Simulation.step` on the cached integer index array
of thrusting slots. It is three vectorised statements over that set, so its cost tracks the thrusting
body count and not `max_capacity` (`tests/validation/test_scaling_invariants.py`'s property).

Design decisions
----------------
**Mass is frozen across the four RK4 stages and updated at the step boundary**, which makes the mass
half of the scheme first-order accurate even though the position half is fourth-order. This is the
same trade `thirdbody.py` makes with its frozen perturber, and it is visible in the numbers: `T/m`
rises through a burn while the whole step is flown at the heavier start-of-step mass, so the
integrated `Delta v` falls **short** of the exact rocket equation by `(dt/2)(a_end - a_start)` - a
left Riemann sum's bias. Measured -8.84e-5 of `Delta v` against a predicted -8.79e-5 for the burn in
`tests/validation/test_thrust.py`, which asserts that bias *as a prediction* rather than tolerating
it. Halve `dt` to halve it.

**The direction vector is used as given, not normalised.** Its norm therefore scales the thrust, so
`(0, 0.5, 0)` is a half-throttle prograde burn and `(0, 0, 0)` is a coast. That is a useful throttle
channel, and normalising would have to special-case the zero vector anyway. Supply a unit vector
unless a throttle is what is meant.

**The direction is body-fixed in RSW and re-evaluated at every stage**, from that stage's candidate
state. So the direction law is exact under RK4 - unlike the mass - and a prograde burn stays prograde
as the orbit turns. An inertially fixed direction law is not expressible; it would be a second model
(or a fourth/fifth/sixth column pair), not a flag on this one.

**Degenerate rows contribute exactly `0.0`**, with no division and no warning:

- *empty tanks* (`mass_kg <= dry_mass_kg`), which is how a burn ends;
- *`thrust_n <= 0`* or *`isp_s <= 0`*, which includes a body whose bit is set with coefficients never
  written - a silent no-op, as with `"j2"` and `"drag"`. Always pass the coefficients.
- *an undefined RSW frame* - zero separation from the parent (a root body), a rectilinear state, or a
  non-finite one. `RSW_basis` returns exactly-zero rows there rather than NaN, so nothing poisons the
  accumulator.

**Massive bodies are refused at configuration time.** `validate_bodies` rejects `mu != 0`, because
`mu_array` is fixed at build and would not track a burning body's own mass. `Simulation.set_propagator`
already refuses Cowell for `mu != 0` bodies for a different reason, so in practice a thrusting body
was going to be massless anyway; this makes the reason explicit at the point of configuration.

**Coefficients are range-checked** by `validate_coefficients`: negative thrust, non-positive `isp_s`
alongside positive thrust, non-positive mass, negative dry mass, and `mass_kg < dry_mass_kg`. The
check sees only the coefficients of the *call it is in*, so splitting one body's setup across two
`enable_force_model` calls defeats the cross-check between `mass_kg` and `dry_mass_kg`. Pass them
together.

Composition
-----------
The `"thrust"` bit is foreign to `Simulation._refresh_cowell_plan`, whose fused compiled kernel
covers `point_mass_gravity`, `j2` and `zonal` only. A Cowell body carrying thrust therefore sends the **whole**
Cowell set down the NumPy `RK4Integrator` path, exactly as `"drag"` and `"third_body"` do
(`tests/validation/test_thrust.py` asserts it). There is deliberately no compiled twin.

Limitations
-----------
- **Burn duration is quantised to `dt`.** Fuel that would run out mid-step is instead spent over that
  whole step: the mass is clamped at `dry_mass_kg`, so no propellant is invented, but the impulse
  delivered in the final step exceeds the physical one by up to `T dt`. Choose a burn time that is an
  integer multiple of `dt` when that matters.
- **No attitude, no slew rate, no thruster geometry.** The direction law changes instantaneously and
  the vehicle is a point. A finite slew would need vehicle state the arena does not carry.
- **`Isp` is constant.** No throttle-dependent or ambient-pressure-dependent efficiency.
- **No thrust-vector jitter, misalignment or cosine losses**, and no coupling into `drag.py`'s
  ballistic coefficient: a vessel that burns half its mass keeps the ballistic coefficient it was
  configured with. Enabling both models on one body is therefore inconsistent at the few-percent
  level once a large fraction of the mass is gone.
"""
from __future__ import annotations

from typing import Final, Mapping, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .custom_types import ScalarSeconds
from .frames import ReferenceFrames
from .registry import register_force_model

if TYPE_CHECKING:
    from .simulator import Simulation

__all__ = [
    "THRUST_MODEL", "THRUST_PARAM_NAMES", "STANDARD_GRAVITY",
    "thrust_kernel", "deplete_mass", "mass_flow_rate",
]

THRUST_MODEL: Final[str] = "thrust"
THRUST_PARAM_NAMES: Final = ("thrust_n", "isp_s", "mass_kg", "dry_mass_kg", "dir_r", "dir_s", "dir_w")
_THRUST_COL: Final[int] = 0
_ISP_COL: Final[int] = 1
_MASS_COL: Final[int] = 2
_DRY_MASS_COL: Final[int] = 3
_DIR_COL: Final[int] = 4          # dir_r, dir_s, dir_w occupy columns 4, 5, 6.

# T [N] / m [kg] is m/s^2; the engine wants km/s^2. This is the model's only unit conversion.
_M_PER_S2_TO_KM_PER_S2: Final[float] = 1.0e-3

# Standard gravity, m/s^2. Exact by definition (CGPM 1901 / BIPM SI brochure) - it is a defined
# constant in the specific-impulse convention, not a measurement of Earth's gravity, so it does not
# change with the gravity model.
STANDARD_GRAVITY: Final[float] = 9.80665


def _reject_massive_bodies(sim: "Simulation", bodies: NDArray[np.int64]) -> None:
    """`validate_bodies` hook for `"thrust"`: refuse bodies carrying gravitational mass."""
    offending = bodies[sim.mu_array[bodies] != 0.0]
    if offending.size > 0:
        raise ValueError(
            f"force model '{THRUST_MODEL}' requires a massless body (mu == 0): a burning body's mass "
            f"changes, and `mu_array` is fixed at build, so its gravity could not track its own fuel "
            f"burn; slot(s) {offending.tolist()} have mu != 0. See thrust.py's module docstring."
        )


def _check_coefficients(
    sim: "Simulation", bodies: NDArray[np.int64], coefficients: Mapping[str, float],
) -> None:
    """
    `validate_coefficients` hook for `"thrust"`: range-check what this call supplies.

    Only the coefficients of the current call are visible, so splitting one body's setup across two
    `enable_force_model` calls defeats the `mass_kg >= dry_mass_kg` cross-check. Pass them together.
    """
    thrust_n = coefficients.get("thrust_n")
    isp_s = coefficients.get("isp_s")
    mass_kg = coefficients.get("mass_kg")
    dry_mass_kg = coefficients.get("dry_mass_kg")

    if thrust_n is not None and thrust_n < 0.0:
        raise ValueError(f"force model '{THRUST_MODEL}': thrust_n must be >= 0, got {thrust_n}")
    if isp_s is not None and isp_s <= 0.0 and (thrust_n is None or thrust_n > 0.0):
        raise ValueError(
            f"force model '{THRUST_MODEL}': isp_s must be > 0 to define a mass flow, got {isp_s}")
    if mass_kg is not None and mass_kg <= 0.0:
        raise ValueError(f"force model '{THRUST_MODEL}': mass_kg must be > 0, got {mass_kg}")
    if dry_mass_kg is not None and dry_mass_kg < 0.0:
        raise ValueError(
            f"force model '{THRUST_MODEL}': dry_mass_kg must be >= 0, got {dry_mass_kg}")
    if mass_kg is not None and dry_mass_kg is not None and mass_kg < dry_mass_kg:
        raise ValueError(
            f"force model '{THRUST_MODEL}': mass_kg ({mass_kg}) is below dry_mass_kg "
            f"({dry_mass_kg}); a vessel cannot start with negative propellant")


@register_force_model(
    THRUST_MODEL,
    param_names=THRUST_PARAM_NAMES,
    validate_bodies=_reject_massive_bodies,
    validate_coefficients=_check_coefficients,
    citation=(
        "a = T/m (Newton II) with m_dot = T/(Isp g0) (the definition of specific impulse); Curtis, "
        "Orbital Mechanics for Engineering Students, 3rd ed., Sec. 11.2-11.3, Eq. 11.6 (Tsiolkovsky); "
        "Vallado, Fundamentals of Astrodynamics and Applications, 4th ed., Sec. 6.2 Eq. 6-1 and "
        "Sec. 8.7 (continuous low thrust). Equation numbers from memory, unverified. RSW direction "
        "law: Vallado 4e Sec. 3.3, via frames.ReferenceFrames.RSW_to_cart."
    ),
)
def thrust_kernel(
    indices: NDArray[np.int64],
    t: ScalarSeconds,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    params: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """
    Add `(T/m) * (d_R R + d_S S + d_W W) * 1e-3` to `out[indices]`, km/s^2, with the RSW basis taken
    from each body's state relative to its Keplerian parent.

    Coefficient units and the single conversion (`_M_PER_S2_TO_KM_PER_S2`) are in the module
    docstring. `t` and `mu_array` are unused: the throttle is constant in time and the direction law
    is purely geometric. Mass is read from `params`, never advanced here - `deplete_mass` does that,
    once per step, so all four RK4 stages of a step see the same mass.
    """
    primaries = parent_indices[indices]
    rel = state[indices] - state[primaries]                     # (k, 6): [r | v] relative to parent

    coeff = params[indices]
    thrust_n = coeff[:, _THRUST_COL]
    mass = coeff[:, _MASS_COL]

    # Rows with empty tanks, no thruster, or an unconfigured (all-zero) coefficient row divide by 1
    # and are multiplied by a zero numerator, so a body burnt out mid-run contributes exactly 0.0.
    live = (thrust_n > 0.0) & (mass > coeff[:, _DRY_MASS_COL]) & (mass > 0.0)
    accel_mag = np.divide(
        thrust_n * _M_PER_S2_TO_KM_PER_S2, mass, out=np.zeros_like(mass), where=live,
    )

    # RSW -> Cartesian. Zero rows come back for an undefined frame (root body, rectilinear or
    # non-finite state), so no such body can push an accumulator off a NaN.
    direction, _ = ReferenceFrames.RSW_to_cart(
        rel[:, :3], rel[:, 3:], coeff[:, _DIR_COL:_DIR_COL + 3],
    )

    out[indices] += accel_mag[:, np.newaxis] * direction


def mass_flow_rate(params: NDArray[np.float64], indices: NDArray[np.int64]) -> NDArray[np.float64]:
    """
    `m_dot = T / (Isp g0)`, kg/s, for the given thrusting slots. Zero where the thruster is off or
    `isp_s <= 0`, so an unconfigured row burns nothing.

    Exposed because the test suite needs the same rate the step uses, and re-deriving it there would
    prove only that the test can multiply.
    """
    thrust_n = params[indices, _THRUST_COL]
    isp_s = params[indices, _ISP_COL]
    live = (thrust_n > 0.0) & (isp_s > 0.0)
    mdot: NDArray[np.float64] = np.zeros_like(thrust_n)
    np.divide(thrust_n, isp_s * STANDARD_GRAVITY, out=mdot, where=live)
    return mdot


def deplete_mass(
    indices: NDArray[np.int64], params: NDArray[np.float64], dt: ScalarSeconds,
) -> None:
    """
    Advance `mass_kg` by one step of propellant burn, in place, for the thrusting slots `indices`.

        m <- max(m - m_dot dt, m_dry)

    Called once per `Simulation.step`, *after* the propagation, so every RK4 stage of a step sees the
    mass the step began with. That makes the mass half of the scheme first-order in `dt`; see the
    module docstring on why, and `tests/validation/test_thrust.py` for the resulting `Delta v` bias
    asserted as a prediction.

    The clamp is what stops a burn: once `m == m_dry` the tanks are empty, `thrust_kernel`'s `live`
    mask is false for that row, and its contribution is exactly `0.0` thereafter. Mass then stops
    changing, because `max(m_dry - m_dot dt, m_dry) == m_dry`.

    Three vectorised statements over `indices` - an integer index array, so the cost is the thrusting
    body count and never `max_capacity`. `indices` must be sorted and duplicate-free (it is: it comes
    from `forces.resolve_force_models`, which builds it with `np.flatnonzero`), so the fancy-indexed
    write below is not a many-to-one scatter.
    """
    mass = params[indices, _MASS_COL]
    burnt = mass - mass_flow_rate(params, indices) * float(dt)
    params[indices, _MASS_COL] = np.maximum(burnt, params[indices, _DRY_MASS_COL])
