"""
Solar radiation pressure, registered as the force model `"srp"`.

The momentum carried by sunlight, pushing a body **away from the Sun**, attenuated by the shadow of
an occulting body. It is the dominant non-gravitational perturbation above roughly 800 km - the
altitude at which it overtakes drag is measured, not assumed, in `tests/validation/test_srp.py` - and
it is what shapes the eccentricity of a GEO satellite over a year.

Model: the cannonball
---------------------
For body `i` with light source `S = int(params[i, source])`, let `u = R_S - R_i` be the vector from
the body to the source and `d = |u|`:

    a = - nu * C_r * P_srp * (A/m) * (AU / d)^2 * u_hat,     u_hat = u / d

- `C_r` is the radiation pressure coefficient: 0 for a transparent body, 1 for a perfect absorber,
  2 for a flat perfect specular reflector facing the Sun. A real spacecraft sits between 1 and ~1.5.
- `A/m` is the cross-sectional area to mass ratio, m^2/kg.
- `P_srp` is the radiation pressure at 1 AU, N/m^2, and `AU` is `constants.AU2KM`. Their pairing is
  what fixes the flux: `P_srp (AU/d)^2` is the pressure at the body.
- `nu` in `[0, 1]` is the shadow factor - see below.

**The minus sign is the physics.** Sunlight pushes outward: `u_hat` points from the body *to* the
Sun, so the acceleration is `-u_hat`. `tests/validation/test_srp.py` checks this as a vector
projection (`a . u_hat / |a| == -1`), never as a magnitude, because a magnitude check cannot see a
sign error - and a sign-flipped SRP produces an entirely plausible-looking orbit.

**Cannonball means the body is a sphere.** Its cross-section does not depend on attitude, and the
reflected momentum is purely anti-sunward. A real spacecraft with flat panels has an acceleration
that also has a component along the panel normal, and an `A` that varies with attitude. Neither is
modelled; `C_r` and `A/m` are constants per body. That is the standard first-cut model and is what
`C_r` exists to absorb.

**There is no indirect term**, unlike `"third_body"`. Cowell integrates `r = R_i - R_P` relative to
the Keplerian parent, so strictly the right-hand side is `a_i,srp - a_P,srp`. The parent is a
celestial body: Earth's own area-to-mass ratio is `pi R^2 / M = 2.1e-11 m^2/kg` against a
spacecraft's 0.01 to 0.1, nine to ten orders of magnitude smaller, so `a_P,srp` is dropped. The
Poynting-Robertson and relativistic corrections (of order `v/c ~ 1e-4` relative) are dropped for the
same reason, as is thermal re-radiation.

Shadow: two models, selected per body
-------------------------------------
`shadow_model` is a per-body coefficient, exactly the way `drag.py` selects its density law - not a
second mask bit, so it is one more key in a `sweep.ForceModelSpec`'s `coefficients`:

| `shadow_model` | geometry | `nu` |
|---|---|---|
| `0.0` (`SHADOW_MODEL_CYLINDRICAL`, the default) | umbra only: a half-infinite cylinder of radius `r_occ` trailing the occulter | exactly `0` or exactly `1` |
| `1.0` (`SHADOW_MODEL_CONICAL`) | the fraction of the source's *apparent disc* the occulter covers | continuous in `[0, 1]` |

An unwritten coefficient row is all zeros, so `0.0` is the cylinder. The **occulting body is the
body's `parent_indices` parent**, and its radius is the `r_occ` coefficient. `r_occ <= 0` means *no
occulter at all*, `nu == 1` everywhere - which is both the right answer for a heliocentric body whose
parent is the Sun and the same silent-no-op convention `drag.py` uses for `scale_height <= 0`. Only
one occulter per body: an Earth satellite is never put into the Moon's shadow here.

**The cylinder is discontinuous at the terminator, and that is the model, not a bug.** `nu` steps
between 0 and 1 across a surface of zero thickness, so the acceleration jumps by the full
`C_r P_srp (A/m) (AU/d)^2`. For a GEO satellite with `C_r = 1.3`, `A/m = 0.02 m^2/kg` that jump is
`1.19e-10 km/s^2`. What it costs an RK4 integrator: on the one step that straddles the terminator,
the four stages disagree about which side of it the body is on, and the switch is effectively
mistimed by up to `h`. The velocity error from one crossing is therefore about `Delta_a h / 2`, of
random sign - **first order in `h`, not fourth** - and it then propagates as a position error growing
linearly in the remaining time. At `h = 60 s` that is `3.6e-9 km/s` per crossing, or about
`3e-4 km` of position drift per day for a satellite eclipsed twice a day. RK4's *own* truncation on
the same orbit is far smaller, so eclipse crossings dominate the error budget of a shadowed run, and
halving `h` only halves it. The conical model removes the discontinuity in `nu` itself (it is
continuous, though its derivative is not), which is the real reason to prefer it over the extra
fidelity of the penumbra. The *general* fix is to stop the step at the terminator instead of smoothing
it: `umbra_clearance` below is this surface written as a signed scalar, `events.shadow_event(sim)`
wraps it, and `Simulation.add_event` turns the splitting on - which restores RK4's order rather than
hiding the symptom. See `events.py`.

The conical model computes the **occulted fraction of the source's disc** from three angles measured
at the body: the apparent radius of the source `alpha = asin(r_source / d_source)`, the apparent
radius of the occulter `beta = asin(r_occ / d_occ)`, and their angular separation `c`. Then

    c >= alpha + beta      -> nu = 1                      (full sun)
    c <= beta - alpha      -> nu = 0                      (umbra: the occulter covers the disc)
    c <= alpha - beta      -> nu = 1 - (beta/alpha)^2     (annular: the occulter is wholly inside it)
    otherwise              -> nu = 1 - A_overlap / (pi alpha^2)   (penumbra)

with the standard circular-segment overlap area

    x = (c^2 + alpha^2 - beta^2) / (2 c),   y = sqrt(alpha^2 - x^2)
    A_overlap = alpha^2 acos(x / alpha) + beta^2 acos((c - x) / beta) - c y

This treats both discs as flat and uniformly bright: solar limb darkening, which would make the
penumbra transition slightly steeper near the edges, is not modelled. Earth's atmosphere - which
refracts sunlight into the geometric umbra and reddens it - is not modelled either. Both are
sub-percent effects on an orbit-averaged SRP budget and are dwarfed by the uncertainty in `C_r`.

References
----------
- Cannonball SRP: Montenbruck, O. & Gill, E., *Satellite Orbits* (2000), Sec. 3.4, Eq. 3.75;
  Vallado, D. A., *Fundamentals of Astrodynamics and Applications*, 4th ed., Sec. 8.6.4, Eq. 8-44.
  **Section and equation numbers are from memory and unverified against the texts.**
- Conical shadow via apparent-disc overlap: Montenbruck & Gill, Sec. 3.4.2, Eq. 3.87 (the occulted
  fraction) - **also unverified**. The overlap area itself is elementary plane geometry and
  `tests/validation/test_srp.py` checks it against a direct numerical quadrature of the disc overlap,
  which shares no code with the formula. That test, not the citation, is what the model rests on.
- `P_srp = S / c`, with `S` the total solar irradiance at 1 AU. `SOLAR_CONSTANT_1AU = 1367.0 W/m^2`
  is Vallado's value and gives `4.5598e-6 N/m^2`, the `4.56e-6` quoted everywhere. The modern IAU 2015
  nominal TSI is `1361 W/m^2`, which gives `4.5399e-6` - **0.44 % lower**. That difference is a real
  fork in the literature, not a rounding error, and it is why `p_srp` is a swept coefficient rather
  than a constant baked into the kernel.

Coefficients
------------
| param | column | unit |
|---|---|---|
| `cr` | 0 | dimensionless, `0 <= C_r <= 2` |
| `area_mass` | 1 | m^2/kg, cross-section over mass |
| `p_srp` | 2 | N/m^2 at 1 AU (`SOLAR_PRESSURE_1AU`); **mandatory** |
| `source` | 3 | arena slot of the light source; **mandatory** |
| `r_occ` | 4 | km, occulting body (the Keplerian parent) radius; `<= 0` disables the shadow |
| `r_source` | 5 | km, light source radius; read by the conical model only |
| `shadow_model` | 6 | selector: 0.0 cylindrical, 1.0 conical |
| `shadow_latch` | 7 | **engine-owned**: branch override written by `events.py`, never by a caller |

`p_srp [N/m^2] * (A/m) [m^2/kg]` is m/s^2, and the engine wants km/s^2. **The only unit conversion is
`_M_TO_KM = 1e-3`**, applied once. `AU / d` is a ratio of kilometres and needs none.

Design decisions
----------------
**The Sun is a named body, exactly as in `thirdbody.py`.** `source` is a float slot index in
`force_model_params["srp"]`, because `forces.ForceKernel` gives a kernel no other channel for
per-body data and float64 represents every arena slot exactly. A slot number is not portable across
builds, so it is never the *sweep* representation: `sweep.ForceModelSpec.body_coefficients` maps the
name `source` to a body name, and `sweep.apply_config` resolves it through `sim.name_to_index`:

    ForceModelSpec("srp", {"cr": 1.3, "area_mass": 0.02, "p_srp": SOLAR_PRESSURE_1AU,
                           "r_occ": EARTH_R_EQ, "r_source": SUN_RADIUS,
                           "shadow_model": SHADOW_MODEL_CONICAL},
                   body_coefficients={"source": "Sun"})

**`source` and `p_srp` are both mandatory** (`_validate_srp`, the registry's `validate_coefficients`
hook, run by `enable_force_model` before any bit is set). An unwritten row would read slot 0 at zero
pressure: a plausible-looking configuration that silently produces exactly no force. That is the
failure mode `CLAUDE.md`'s item 5 exists to catch, so it raises instead. `cr` and `area_mass` are
*not* mandatory - zero there is a legitimate "this body is not pushed by sunlight", and unlike a zero
`p_srp` it is per-body and deliberate.

**The source is read from `state`, like every other row**, so the kernel needs no ephemeris and works
for any luminous body in the arena. The source is *not* required to be massive: SRP does not read
`mu_array` at all.

**Geometry is absolute, not parent-relative.** Every other force model in the engine works from
`state[i] - state[parent]`; SRP needs `state[source] - state[i]`, and the shadow needs the occulter's
absolute row too. `integrators.RK4Integrator` writes `state[primaries] + candidate` into the body's
row at every stage, so `state[i]` really is that stage's absolute candidate position and the geometry
is staged correctly. What is *not* staged is the source's and the parent's rows - see below.

The approximation: the source is frozen within a Cowell step
------------------------------------------------------------
`integrators.RK4Integrator` advances only the Cowell rows; the source's row and the parent's stay at
their start-of-step values across all four stages. This is the same freeze `thirdbody.py` derives in
full, and it has the same consequence in principle: RK4's `sum b_i c_i = 1/2` makes the staged
evaluation reproduce the step-mean of a linearly varying forcing, so freezing at `t_k` applies the
perturbation with a lag of `h/2`, and the global error is **first order in `h`**.

In practice it is negligible here, and the reason is worth stating. The part of the geometry that
varies fastest within a step is the *body's own* position, and that is staged. What the freeze misses
is the Sun's apparent motion as seen from the parent, `n_E = 1.99e-7 rad/s`. Over `h/2` that rotates
`u_hat` by `n_E h / 2` and changes `(AU/d)^2` by twice as much in relative terms, so
`|da/dt| ~ a n_E = 2.4e-17 km/s^3` for the GEO case above. Even taken as fully coherent, one day of
that lag gives `(1/2)(h/2)|da/dt| T^2 = 8.9e-8 h` km - `5e-6 km` at `h = 60 s`, four orders below the
eclipse-crossing error derived above, and below RK4's own truncation. **The freeze does not matter at
LEO or GEO step sizes.** It would matter for a body whose source direction swings quickly, such as a
close planetary flyby, and that case is unmeasured.

The freeze does *not* apply to the shadow's discontinuity: that is a property of the model itself and
survives any amount of staging.

Composition and tiers
---------------------
Reference tier only, with no compiled twin. `Simulation._refresh_cowell_plan` accepts only
`point_mass_gravity`, `j2`, `drag` and `zonal` into the fused `kernels.cowell_rk4_step`, so this model's bit is
foreign and a Cowell set containing an `"srp"` body runs entirely on the NumPy `RK4Integrator` path
(`tests/validation/test_srp.py` asserts it).

Limitations
-----------
- One occulter per body, and it must be the Keplerian parent. Lunar eclipses of an Earth satellite,
  and Earth eclipses of a lunar orbiter (whose parent is the Moon), are not modelled.
- Attitude, flat panels, thermal re-radiation, Earth albedo and infrared re-radiation: none modelled.
  Albedo is typically 10-30 % of direct SRP in LEO and is the largest of these omissions.
- The cylindrical model's discontinuity degrades RK4 to first order on crossing steps **unless the
  step is cut at the terminator**. That fix now exists and is opt-in: `umbra_clearance` below is the
  signed event function, `events.shadow_event(sim)` wraps it, and `Simulation.add_event` turns it on.
  Neither shadow model does event detection by itself - `nu` is still a plain function of the state.
- `C_r` and `A/m` are constant: no articulated arrays, no mass coupling to `thrust.py`'s propellant.
"""
from __future__ import annotations

from typing import Final, Mapping, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .constants import AU2KM, C_SI
from .custom_types import ArrayFloat, ScalarSeconds
from .registry import mask_for, register_force_model

if TYPE_CHECKING:
    from .simulator import Simulation

__all__ = [
    "SRP_MODEL", "SRP_PARAM_NAMES", "srp_kernel", "shadow_factor",
    "umbra_clearance", "shadow_clearance", "cylindrical_shadow_bodies", "latch_shadow_branch",
    "LATCH_AUTO", "LATCH_LIT", "LATCH_DARK",
    "SOLAR_CONSTANT_1AU", "SOLAR_PRESSURE_1AU", "SUN_RADIUS",
    "SHADOW_MODEL_CYLINDRICAL", "SHADOW_MODEL_CONICAL",
]

SRP_MODEL: Final[str] = "srp"
SRP_PARAM_NAMES: Final = (
    "cr", "area_mass", "p_srp", "source", "r_occ", "r_source", "shadow_model", "shadow_latch")
_CR_COL: Final[int] = 0
_AREA_MASS_COL: Final[int] = 1
_P_SRP_COL: Final[int] = 2
_SOURCE_COL: Final[int] = 3
_R_OCC_COL: Final[int] = 4
_R_SOURCE_COL: Final[int] = 5
_SHADOW_MODEL_COL: Final[int] = 6
_SHADOW_LATCH_COL: Final[int] = 7

#: `shadow_latch` values. **Engine-owned state, not a configuration knob** - `_validate_srp` refuses
#: a user-supplied one, and `enable_force_model` leaves the column at `LATCH_AUTO`. The only writer
#: is `latch_shadow_branch`, called by `Simulation._advance_with_events` for the duration of the
#: sub-step that runs up to a located terminator crossing. See that method, and `events.py`, for why
#: a piecewise-smooth right-hand side has to be told which piece it is on.
LATCH_AUTO: Final[float] = 0.0
LATCH_LIT: Final[float] = 1.0
LATCH_DARK: Final[float] = -1.0

# p_srp [N/m^2] * (A/m) [m^2/kg] is m/s^2; the engine wants km/s^2. The model's only unit conversion.
_M_TO_KM: Final[float] = 1.0e-3

#: Total solar irradiance at 1 AU, W/m^2. Vallado 4e's value; the IAU 2015 nominal figure is 1361.0,
#: which is 0.44 % lower. See the module docstring - this is why `p_srp` is a coefficient.
SOLAR_CONSTANT_1AU: Final[float] = 1367.0

#: Solar radiation pressure at 1 AU, N/m^2 = kg m^-1 s^-2. `S / c`, the usual 4.56e-6.
SOLAR_PRESSURE_1AU: Final[float] = SOLAR_CONSTANT_1AU / C_SI

#: Nominal solar radius, km (IAU 2015 nominal value, quoted from memory). Only the conical shadow
#: reads it. `scenarios.sun_earth_moon` seeds the Sun at 696340 km, a 0.09 % difference that moves
#: the penumbra's width by the same fraction.
SUN_RADIUS: Final[float] = 695700.0

#: Half-infinite cylinder of the occulter's radius: `nu` is exactly 0 or exactly 1. The default, and
#: what an unwritten (all-zero) coefficient row already selects.
SHADOW_MODEL_CYLINDRICAL: Final[float] = 0.0
#: Apparent-disc overlap: umbra, penumbra and annular eclipse, `nu` continuous in [0, 1].
SHADOW_MODEL_CONICAL: Final[float] = 1.0

_SHADOW_MODELS: Final = (SHADOW_MODEL_CYLINDRICAL, SHADOW_MODEL_CONICAL)

#: `C_r` outside this range is not a cannonball. 0 is transparent, 1 a perfect absorber, 2 a flat
#: perfect specular reflector held normal to the Sun - the physical maximum for the model's geometry.
_CR_MAX: Final[float] = 2.0


# ==================================================================================================
# Shadow geometry
# ==================================================================================================

def shadow_factor(
    to_source: ArrayFloat,
    to_occulter: ArrayFloat,
    source_radius_km: ArrayFloat,
    occulter_radius_km: ArrayFloat,
    conical: NDArray[np.bool_],
    latch: Optional[ArrayFloat] = None,
) -> ArrayFloat:
    """
    The illuminated fraction `nu` in `[0, 1]` of the source's disc, per row.

    Parameters
    ----------
    to_source, to_occulter : (k, 3)
        Vectors **from the body** to the light source and to the occulting body, km.
    source_radius_km, occulter_radius_km : (k,)
        Radii of the two discs, km. `occulter_radius_km <= 0` means no occulter: that row is lit,
        exactly `1.0`, and neither law is evaluated for it.
    conical : (k,) bool
        `True` selects the apparent-disc overlap (umbra/penumbra/annular), `False` the cylindrical
        umbra. Both laws are evaluated for every row and masked, so a mixed arena costs no branch -
        the same branchless-select idiom `drag.py` uses for its two density laws.
    latch : (k,) or None
        Per-row branch override, `LATCH_AUTO` / `LATCH_LIT` / `LATCH_DARK`. `None` (and `LATCH_AUTO`)
        evaluate the geometry, which is the only behaviour anything outside `events.py` ever sees.
        A latched row returns exactly `1.0` or exactly `0.0` **regardless of where it is**, which is
        what makes a sub-step that has been cut at a terminator integrate one, smooth branch of the
        piecewise right-hand side at *all four* of its RK4 stages rather than whichever branch each
        stage's off-trajectory sample point happens to fall in. Without it, the stage that lands on
        the crossing carries weight 1/6 of a full `Delta_a h`, which is first order and is exactly
        the error event splitting is supposed to remove - measured at `3.38e-6 km` against a derived
        `3.26e-6 km` before the latch existed. See `Simulation._advance_with_events`.

    Returns exactly `0.0` or exactly `1.0` under the cylindrical law, and a continuous value under
    the conical one. See the module docstring for both derivations and for what the cylinder's
    discontinuity costs an RK4 integrator.

    Degenerate rows are lit, with no division evaluated: a zero-length `to_source` or `to_occulter`
    (a body sitting on its own parent, or on the Sun), a coincident source and occulter, and - for
    the conical law only - a source of zero radius, whose apparent disc has no area to occult.
    """
    d_source: ArrayFloat = np.sqrt(np.einsum("ij,ij->i", to_source, to_source))
    d_occ: ArrayFloat = np.sqrt(np.einsum("ij,ij->i", to_occulter, to_occulter))

    axis = to_source - to_occulter                       # occulter -> source
    d_axis: ArrayFloat = np.sqrt(np.einsum("ij,ij->i", axis, axis))

    active = (occulter_radius_km > 0.0) & (d_source > 0.0) & (d_occ > 0.0) & (d_axis > 0.0)
    nu: ArrayFloat = np.ones_like(d_source)

    # --- cylindrical umbra -------------------------------------------------------------------
    # The occulter-centred position of the body is `-to_occulter`. Resolve it along the exact
    # occulter->source axis: behind the occulter (`along < 0`) and within its radius of the axis.
    inv_axis = np.divide(1.0, d_axis, out=np.zeros_like(d_axis), where=active)
    along = -np.einsum("ij,ij->i", to_occulter, axis) * inv_axis
    perp2 = np.maximum(d_occ * d_occ - along * along, 0.0)
    cyl_dark = active & ~conical & (along < 0.0) & (perp2 < occulter_radius_km * occulter_radius_km)
    nu[cyl_dark] = 0.0

    # --- conical: the fraction of the source's apparent disc the occulter covers ---------------
    fine = active & conical & (source_radius_km > 0.0)
    inv_ds = np.divide(1.0, d_source, out=np.zeros_like(d_source), where=fine)
    inv_do = np.divide(1.0, d_occ, out=np.zeros_like(d_occ), where=fine)

    alpha = np.arcsin(np.clip(source_radius_km * inv_ds, -1.0, 1.0))       # apparent source radius
    beta = np.arcsin(np.clip(occulter_radius_km * inv_do, -1.0, 1.0))      # apparent occulter radius
    sep = np.arccos(np.clip(
        np.einsum("ij,ij->i", to_source, to_occulter) * inv_ds * inv_do, -1.0, 1.0))

    total = fine & (sep <= beta - alpha)
    annular = fine & ~total & (sep <= alpha - beta)
    partial = fine & ~total & ~annular & (sep < alpha + beta)

    # Annular: the occulter sits wholly inside the source's disc, blocking (beta/alpha)^2 of its area.
    ratio = np.divide(beta, alpha, out=np.zeros_like(alpha), where=annular)
    # Partial: the circular-segment overlap area of two discs of angular radii alpha and beta whose
    # centres are `sep` apart. `alpha > 0` throughout this region, since `sep < alpha + beta` and
    # `sep > alpha - beta` together force it.
    x = np.divide(sep * sep + alpha * alpha - beta * beta, 2.0 * sep,
                  out=np.zeros_like(sep), where=partial & (sep > 0.0))
    y = np.sqrt(np.maximum(alpha * alpha - x * x, 0.0))
    inv_alpha = np.divide(1.0, alpha, out=np.zeros_like(alpha), where=partial)
    inv_beta = np.divide(1.0, beta, out=np.zeros_like(beta), where=partial)
    overlap = (
        alpha * alpha * np.arccos(np.clip(x * inv_alpha, -1.0, 1.0))
        + beta * beta * np.arccos(np.clip((sep - x) * inv_beta, -1.0, 1.0))
        - sep * y
    )
    lit_partial = 1.0 - overlap * inv_alpha * inv_alpha / np.pi

    nu[total] = 0.0
    nu[annular] = 1.0 - ratio[annular] * ratio[annular]
    nu[partial] = np.clip(lit_partial[partial], 0.0, 1.0)

    # The branch override, applied last so it wins over the geometry. `LATCH_AUTO` (and `None`)
    # leave every row exactly as computed above, so this is a no-op for every caller but `events.py`.
    if latch is not None:
        nu[latch > 0.5] = 1.0
        nu[latch < -0.5] = 0.0
    return nu


# ==================================================================================================
# The umbra boundary as an event function
# ==================================================================================================

def umbra_clearance(
    to_source: ArrayFloat, to_occulter: ArrayFloat, occulter_radius_km: ArrayFloat,
) -> ArrayFloat:
    """
    Signed clearance from the cylindrical umbra's surface, km: **negative inside the umbra**.

    This is the event function `events.py` locates the root of, and it is the same geometry
    `shadow_factor`'s cylindrical branch tests - written as a continuous signed scalar rather than as
    a boolean, so that a root find can bracket it. The two are held consistent in
    `tests/validation/test_events.py`.

    Derivation. With `e` the unit vector along occulter -> source and `p = -to_occulter` the body's
    position relative to the occulter, write `along = p . e` and `perp = |p - along e|`. The umbra is
    the half-infinite cylinder `along < 0`, `perp < r_occ`. Define

        g = hypot(perp, max(along, 0)) - r_occ

    Behind the terminator plane (`along <= 0`) this is `perp - r_occ`, whose zero is exactly the
    cylinder's side surface. Ahead of it (`along > 0`) it is `|p| - r_occ`, the clearance from the
    occulter's own sphere. The two agree at `along = 0`, so `g` is continuous everywhere, and it is
    negative exactly on the umbra for any body **outside the occulter** (`|p| > r_occ`) - which every
    orbiting body is. Inside the occulter's sphere on the sunward side `g` also reads negative while
    `shadow_factor` reads lit; that region is subsurface and unreachable for an orbit.

    The cylinder's *end cap* - the terminator disc `along = 0`, `perp < r_occ` - is not a reachable
    boundary either: `along = 0` and `perp < r_occ` together give `|p| = perp < r_occ`, again inside
    the occulter. So the side surface is the whole of the crossing geometry.

    Parameters
    ----------
    to_source, to_occulter : (k, 3)
        Vectors **from the body** to the light source and to the occulting body, km - the same two
        `shadow_factor` takes.
    occulter_radius_km : (k,)
        Occulter radius, km. `<= 0` means no occulter, and `g` is then `hypot(...) - r_occ >= 0`
        everywhere by construction: no special case is needed and no crossing can ever be reported.

    A degenerate row whose occulter sits exactly on the source (`|axis| = 0`) has no shadow axis; it
    falls back to `|p| - r_occ`, which is the spherical clearance and is the continuous limit.
    """
    axis = to_source - to_occulter                       # occulter -> source
    d_axis: ArrayFloat = np.sqrt(np.einsum("ij,ij->i", axis, axis))
    p2: ArrayFloat = np.einsum("ij,ij->i", to_occulter, to_occulter)     # |p|^2, p = -to_occulter

    inv_axis = np.divide(1.0, d_axis, out=np.zeros_like(d_axis), where=d_axis > 0.0)
    along = -np.einsum("ij,ij->i", to_occulter, axis) * inv_axis
    # A degenerate axis leaves `along` at 0, so the expression below reduces to |p| - r_occ.
    sunward = np.maximum(along, 0.0)
    perp2 = np.maximum(p2 - along * along, 0.0)

    out: ArrayFloat = np.sqrt(perp2 + sunward * sunward) - occulter_radius_km
    return out


def cylindrical_shadow_bodies(sim: "Simulation") -> NDArray[np.int64]:
    """
    Arena slots whose SRP acceleration is **discontinuous**: `"srp"` enabled, `r_occ > 0`, cylinder.

    A conical body is excluded because its `nu` is continuous - there is nothing to stop the step at
    - and an `r_occ <= 0` body has no occulter at all. `events.shadow_event` uses this as its default
    body list; it is separate from that factory so a caller can inspect or subset it.
    """
    params = sim.force_model_params.get(SRP_MODEL)
    if params is None:
        return np.empty(0, dtype=np.int64)
    enabled = (sim.force_model_mask & mask_for((SRP_MODEL,))) != np.uint64(0)
    shadowed = enabled & (params[:, _R_OCC_COL] > 0.0) & (params[:, _SHADOW_MODEL_COL] < 0.5)
    out: NDArray[np.int64] = np.flatnonzero(shadowed & sim.active_mask).astype(np.int64)
    return out


def shadow_clearance(sim: "Simulation", bodies: NDArray[np.int64]) -> ArrayFloat:
    """
    `events.EventFunction` adapter for `umbra_clearance`: `(sim, bodies) -> (k,)`, km.

    A pure read of `sim.global_states`, `sim.parent_indices` and `sim.force_model_params["srp"]` -
    `events.py` evaluates it at trial times inside a step and relies on it leaving no trace. The
    occulter is the Keplerian parent and the source is the `source` coefficient, exactly as in
    `srp_kernel`, so the sign of this function and the value of `nu` inside that kernel are the same
    geometry read twice.
    """
    params = sim.force_model_params[SRP_MODEL]
    sources = params[bodies, _SOURCE_COL].astype(np.int64)
    occulters = sim.parent_indices[bodies]
    state = sim.global_states
    return umbra_clearance(
        state[sources, :3] - state[bodies, :3],
        state[occulters, :3] - state[bodies, :3],
        params[bodies, _R_OCC_COL],
    )


def latch_shadow_branch(
    sim: "Simulation", bodies: NDArray[np.int64], clearance_sign: Optional[ArrayFloat],
) -> None:
    """
    `events.LatchFunction` for the shadow event: pin `bodies` to one branch of the shadow, or release.

    `clearance_sign` is the sign of `shadow_clearance` at the start of a sub-interval that is known
    to contain no crossing: positive is outside the umbra (lit), negative inside (dark). That is
    `umbra_clearance`'s own convention, which is why this translation lives here and not in
    `events.py` - the event layer knows only that the function changed sign, never what the sign
    *means*. `None` releases every row back to `LATCH_AUTO`, which is the geometry.

    Writing `force_model_params` outside `enable_force_model` is deliberate and has precedent:
    `thrust.deplete_mass` does the same to its mass column once per step. Unlike that one, this is
    not physical state - it is a statement about which branch of a piecewise right-hand side the
    current sub-step belongs to, and it is always released before `Simulation.step` returns.
    """
    params = sim.force_model_params.get(SRP_MODEL)
    if params is None:
        return
    if clearance_sign is None:
        params[bodies, _SHADOW_LATCH_COL] = LATCH_AUTO
        return
    params[bodies, _SHADOW_LATCH_COL] = np.where(
        clearance_sign > 0.0, LATCH_LIT, np.where(clearance_sign < 0.0, LATCH_DARK, LATCH_AUTO))


# ==================================================================================================
# Configuration-time validation
# ==================================================================================================

def _validate_srp(
    sim: "Simulation", bodies: NDArray[np.int64], coefficients: Mapping[str, float],
) -> None:
    """
    `validate_coefficients` hook for `"srp"`. Every rule is in the module docstring; the two that
    matter most are that `source` and `p_srp` are **mandatory** (an unwritten row would name slot 0
    at zero pressure and produce exactly no force, silently), and that a shadow needs a real occulter,
    which is the body's Keplerian parent.
    """
    if "shadow_latch" in coefficients:
        raise ValueError(
            f"force model '{SRP_MODEL}': 'shadow_latch' is engine-owned state, not a coefficient. "
            f"It is written and released by Simulation._advance_with_events for the duration of a "
            f"sub-step cut at a terminator crossing, and a value pinned here would freeze the "
            f"shadow for the whole run. Enable the event instead: "
            f"sim.add_event(events.shadow_event(sim))."
        )

    for required in ("source", "p_srp"):
        if required not in coefficients:
            raise ValueError(
                f"force model '{SRP_MODEL}' needs a '{required}' coefficient; an unwritten row would "
                f"silently read slot 0 at zero pressure and produce no force at all. Pass "
                f"source=sim.name_to_index['Sun'] and p_srp=srp.SOLAR_PRESSURE_1AU, or use "
                f"sweep.ForceModelSpec(coefficients={{'p_srp': ...}}, "
                f"body_coefficients={{'source': 'Sun'}})."
            )

    if float(coefficients["p_srp"]) <= 0.0:
        raise ValueError(
            f"force model '{SRP_MODEL}': p_srp={coefficients['p_srp']!r} must be positive "
            f"(N/m^2 at 1 AU; srp.SOLAR_PRESSURE_1AU is {SOLAR_PRESSURE_1AU:.4e})."
        )

    cr = float(coefficients.get("cr", 0.0))
    if not 0.0 <= cr <= _CR_MAX:
        raise ValueError(
            f"force model '{SRP_MODEL}': cr={cr!r} is outside the cannonball model's physical range "
            f"[0, {_CR_MAX}] (0 transparent, 1 perfect absorber, 2 flat specular reflector)."
        )
    for name in ("area_mass", "r_occ", "r_source"):
        value = float(coefficients.get(name, 0.0))
        if value < 0.0:
            raise ValueError(f"force model '{SRP_MODEL}': {name}={value!r} must not be negative.")

    shadow_model = float(coefficients.get("shadow_model", SHADOW_MODEL_CYLINDRICAL))
    if shadow_model not in _SHADOW_MODELS:
        raise ValueError(
            f"force model '{SRP_MODEL}': shadow_model={shadow_model!r} is not a known shadow "
            f"geometry; use srp.SHADOW_MODEL_CYLINDRICAL ({SHADOW_MODEL_CYLINDRICAL}) or "
            f"srp.SHADOW_MODEL_CONICAL ({SHADOW_MODEL_CONICAL})."
        )

    r_occ = float(coefficients.get("r_occ", 0.0))
    if r_occ > 0.0:
        if shadow_model == SHADOW_MODEL_CONICAL and float(coefficients.get("r_source", 0.0)) <= 0.0:
            raise ValueError(
                f"force model '{SRP_MODEL}': the conical shadow needs a positive 'r_source' (the "
                f"light source's radius, km; srp.SUN_RADIUS is {SUN_RADIUS}). Without it the source "
                f"is a point and its disc has no area to occult, so every row would read full sun."
            )
        offending = bodies[sim.is_system[sim.parent_indices[bodies]]]
        if offending.size > 0:
            raise ValueError(
                f"force model '{SRP_MODEL}': the occulting body is each body's Keplerian parent, and "
                f"slot(s) {offending.tolist()} are parented by a barycentre, which has no surface to "
                f"cast a shadow. Pass r_occ=0.0 to disable the shadow, or re-parent them."
            )

    value = float(coefficients["source"])
    if not value.is_integer() or not 0 <= value < sim.max_capacity:
        raise ValueError(
            f"force model '{SRP_MODEL}': source={value!r} is not an arena slot "
            f"(expected a whole number in [0, {sim.max_capacity}))."
        )
    slot = int(value)

    is_self = bodies[bodies == slot]
    if is_self.size > 0:
        raise ValueError(
            f"force model '{SRP_MODEL}': slot(s) {is_self.tolist()} would be lit by themselves "
            f"(zero separation from the source)."
        )
    if r_occ > 0.0:
        is_parent = bodies[sim.parent_indices[bodies] == slot]
        if is_parent.size > 0:
            raise ValueError(
                f"force model '{SRP_MODEL}': source slot {slot} is the Keplerian parent of slot(s) "
                f"{is_parent.tolist()}, so it is also their occulter and would eclipse them "
                f"permanently. Pass r_occ=0.0 for a body orbiting the light source."
            )

    problem = None
    if not sim.active_mask[slot]:
        problem = "is inactive"
    elif sim.is_system[slot]:
        problem = "is a barycentre, which is a mass-weighted point and not where the light comes from"
    if problem is not None:
        raise ValueError(f"force model '{SRP_MODEL}': source slot {slot} {problem}.")


# ==================================================================================================
# The kernel
# ==================================================================================================

@register_force_model(
    SRP_MODEL,
    param_names=SRP_PARAM_NAMES,
    validate_coefficients=_validate_srp,
    citation=(
        "Montenbruck & Gill, Satellite Orbits (2000), Sec. 3.4, Eq. 3.75 (cannonball SRP) and "
        "Sec. 3.4.2, Eq. 3.87 (conical shadow via apparent-disc overlap); Vallado 4e Sec. 8.6.4, "
        "Eq. 8-44. Section and equation numbers from memory, unverified. P_srp = S/c with "
        "S = 1367 W/m^2 (Vallado); IAU 2015 nominal TSI 1361 W/m^2 gives 0.44 % less."
    ),
)
def srp_kernel(
    indices: NDArray[np.int64],
    t: ScalarSeconds,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    params: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """
    Add the anti-sunward cannonball acceleration to `out[indices]`, km/s^2.

    For body `i` with source `S = int(params[i, source])`, occulter `P = parent_indices[i]`:

        u  = state[S,:3] - state[i,:3]        (body -> source)
        nu = shadow_factor(u, state[P,:3] - state[i,:3], r_source, r_occ, shadow_model)
        a  = -nu * cr * p_srp * area_mass * (AU/|u|)^2 * u/|u| * 1e-3

    The geometry is **absolute**, not parent-relative, unlike every other kernel in the engine - see
    the module docstring on why that is still correct under `RK4Integrator`'s staging, and on the
    `h/2` lag the frozen source row costs. `t` and `mu_array` are unused: SRP is time-invariant given
    the geometry, and the source's mass is irrelevant to its light.

    A body coincident with its source contributes exactly `0.0`, with no division evaluated. That is
    unreachable through `enable_force_model`, which rejects a self-source.
    """
    sources = params[indices, _SOURCE_COL].astype(np.int64)
    occulters = parent_indices[indices]
    coeff = params[indices]

    to_source = state[sources, :3] - state[indices, :3]
    to_occulter = state[occulters, :3] - state[indices, :3]

    d2 = np.einsum("ij,ij->i", to_source, to_source)
    # A zero separation contributes exactly zero: `inv_d2` and `inv_d` both stay 0, so does `a`.
    inv_d2 = np.divide(1.0, d2, out=np.zeros_like(d2), where=d2 > 0.0)
    inv_d = np.sqrt(inv_d2)

    nu = shadow_factor(
        to_source, to_occulter,
        coeff[:, _R_SOURCE_COL], coeff[:, _R_OCC_COL],
        coeff[:, _SHADOW_MODEL_COL] >= 0.5,
        coeff[:, _SHADOW_LATCH_COL],
    )

    # cr * p_srp [N/m^2] * (A/m) [m^2/kg] is m/s^2; (AU/d)^2 is dimensionless; 1e-3 gives km/s^2.
    # The trailing `inv_d` turns `to_source` into its own unit vector, and the sign makes it
    # anti-sunward - the one place the whole model's direction is decided.
    magnitude = (
        coeff[:, _CR_COL] * coeff[:, _P_SRP_COL] * coeff[:, _AREA_MASS_COL]
        * (AU2KM * AU2KM * inv_d2) * _M_TO_KM * nu * inv_d
    )
    out[indices] -= magnitude[:, None] * to_source
