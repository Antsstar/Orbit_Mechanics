"""
Third-body point-mass perturbation, registered as the force model `"third_body"`.

Every other force model acts only relative to a body's own `parent_indices` parent. This one adds the
gravitational pull of one *other* named body, the perturber, on a body integrated relative to its
parent: the Sun on the Moon, or the Moon on an Earth satellite. It contributes **only the
perturbation**. The central `-mu r / r^3` term stays with `point_mass_gravity`, so the two compose
additively without double counting.

Physics
-------
Let `R_b`, `R_P`, `R_s` be the inertial positions of the body, its parent and the perturber, with the
body massless. Newton gives the inertial accelerations

    R_b'' = mu_P (R_P - R_b)/|R_P - R_b|^3 + mu_s (R_s - R_b)/|R_s - R_b|^3
    R_P'' =                                   mu_s (R_s - R_P)/|R_s - R_P|^3     (+ other bodies' pulls)

Cowell integrates `r = R_b - R_P` (see `docs/architecture.md`'s Cowell section). Subtracting, and
writing `r_s = R_s - R_P` for the perturber relative to the parent,

    r'' = -mu_P r/|r|^3  +  mu_s [ (r_s - r)/|r_s - r|^3  -  r_s/|r_s|^3 ]
          '-- point_mass_gravity --'  '---- direct ----'   '- indirect -'

The first bracketed term is the perturber's pull on the body (the *direct* term). The second is its
pull on the parent (the *indirect* term). It enters with a minus sign because the frame
rides on the parent and inherits that acceleration. Omitting it is not a small error. For the Moon
both terms are about mu_Sun/AU^2 = 5.9e-6 km/s^2 and their difference, the solar tide, is about
2 mu_Sun r / AU^3 = 3.0e-8 km/s^2. So dropping the indirect term adds a spurious 5.9e-6 km/s^2,
two hundred times the real perturbation. `tests/validation/test_third_body.py`
catches it by seven orders of magnitude.

Only the parent's acceleration *from this perturber* is subtracted. Anything else accelerating the
parent, such as the parent's own parent when that is a different body, is a separate perturber and
needs its own term.

References
----------
Montenbruck, O. & Gill, E., *Satellite Orbits: Models, Methods and Applications* (2000), Section 3.3
  ("Perturbations by the Sun and Moon"), the point-mass form
  `r'' = GM (s - r)/|s - r|^3 - GM s/|s|^3`. **Equation number 3.37 is from memory and unverified.**
Vallado, D. A., *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 8.6.3 ("Third-Body
  Perturbations"), the same direct-minus-indirect form. **Section and equation numbers from memory,
  unverified.**
Neither citation is load-bearing. The five-line derivation above is self-contained and is what should
be checked. Battin, *An Introduction to the Mathematics and Methods of Astrodynamics* (1999),
Section 8.4 gives a form, `f(q)`, that is better conditioned when |r| << |r_s|. It is not needed here: the two
terms cancel to about 5e-3 relative for the Moon, which costs 2.3 decimal digits of a 16-digit
result.

Design decisions
----------------
**The perturber is a float slot index in `force_model_params["third_body"]`, column `"perturber"`.**
`forces.ForceKernel` gives a kernel no other channel for per-body data, and float64 represents every
arena slot exactly (below 2^53). A slot number is not portable across builds, though, so it is never the
*sweep* representation. `sweep.ForceModelSpec.body_coefficients` maps a coefficient name to a body
*name* (`{"perturber": "Sun"}`), and `sweep.apply_config` resolves it through `sim.name_to_index` when
the configuration is applied. Configs therefore stay plain, name-keyed data, like model names. There is
one perturber per body per model. A body wanting two perturbers (Sun *and* Moon on a LEO satellite)
cannot have them yet. That is a limitation of this model, not of the arena.

**Configuration-time validation** (`_validate_perturber`, registered as `validate_coefficients`, which
`Simulation.enable_force_model` runs before setting any bit). The perturber coefficient is mandatory,
unlike `"j2"`'s silent no-op when coefficients are omitted. A zero-initialised row would otherwise
mean "slot 0", which is a plausible-looking wrong answer. It must be a whole number naming an
in-range slot, and that slot must not be:

- *inactive*: its row is stale or empty;
- *a barycentre*: its `mu_array` row holds summed system mass at a point where no mass sits;
- *massless*: it would contribute exactly zero, silently;
- *the body itself*: the direct term is singular;
- *the body's parent*: the indirect term is singular, and the parent is already `point_mass_gravity`'s job.

Because a perturber must be massive, it can never be a Cowell or secular-J2 body, since both require `mu == 0`.
Every perturber therefore moves on the analytic Keplerian path. That is what the frozen-perturber
analysis below assumes.

**Degenerate geometry in the kernel.** A zero separation in either term contributes exactly `0.0` to
that term, with no division evaluated. This happens only if the mask was written directly, bypassing validation.

The approximation: perturbers are frozen within a Cowell step
--------------------------------------------------------------
`integrators.RK4Integrator` advances only the Cowell rows. The perturber's row, and the parent's,
stay at their *start-of-step* values across all four stages, so every stage sees `r_s(t_k)` instead
of `r_s(t_k + c_i h)`. The body's own position `r` *is* staged correctly. For parent-relative forces
this freeze costs nothing (see `integrators.py`). For this model it does, and the loss is derived here.

RK4's weights satisfy `sum b_i c_i = 1/2`. The staged evaluation therefore reproduces the step-mean of
any forcing that varies linearly in time, so the effective evaluation time is `t_k + h/2`. Frozen at
`t_k`, the perturbation is applied with a lag of `h/2`. That is a forcing error
`delta_a(t) = -(h/2) d a_p/dt + (zero-mean ripple)`. The velocity error per step is `O(h^2)`, and
accumulating over `T/h` steps gives a **global error of first order in h**. RK4's own
truncation error is fourth order. So the Cowell + `third_body` combination converges at first order
wherever the freeze dominates, and fourth order only where it does not.

*Magnitude for the Moon under the Sun, 30 days.* A first bound: `d a_p/dt` comes from the Sun's
direction turning at `n_E = 1.99e-7 rad/s`, so `|d a_p/dt| <= 3 n_E mu_s r/d^3 = 9.1e-15 km/s^3`.
Acting coherently, that gives `(1/2)(h/2)(9.1e-15) T^2 = 1.5e-2 h` km (h in seconds). Scaling by the 0.33
that the full solar perturbation reaches of its own coherent bound gave a first estimate of `5e-3 h`.
**Measured: `7.3e-4 h`, seven times smaller.** The estimate was wrong, not the code. Most of the Moon's
30-day solar displacement is the secular along-track drift from the *orbit-averaged* tide, and for a
near-ecliptic orbit that average does not depend on the Sun's direction. A lag cannot change it.

The prediction that holds is a direct consequence of the lag. With the Moon massless, Sun-Earth is an
isolated two-body pair, so the engine integrates exactly the truth with the Sun delayed by `h/2`:

    err(h) = (h/2) dr/dtau + O(h^2),

where `dr/dtau` is the truth's sensitivity to delaying the Sun, computed from `reference.py` alone. It is
1.504e-3 km/s. Rotation alone gets 81 % of it; the rest is Earth's heliocentric distance changing
(e = 0.0167) under a d^-3 tide. The O(h^2) remainder comes mainly from frozen RK4's local position
error `-h^3 a'/6` against the pure lag's `-h^3 a'/4`. Summed over the run that is
`(h^2/12)|Delta a_p| ~ 0.065 km` at h = 3600 s, 2.4 % of the leading term. Measured at 3600 s:
|err| = 2.638 km, mismatch against the vector prediction 2.6 %, cos 0.99997. At 1800 s: 1.336 km,
1.3 %.

RK4's own error on the same run is 1.04 km at h = 21600 s, falling at fourth order to 2.1e-4 km at
2700 s. Against the freeze's 7.5e-4 h, the crossover is near h ~ 1 day. **At every practical step size
the freeze dominates, and convergence is first order.** Reaching even 1e-3 km would take h ~ 1.3 s.
The measurements, and a test-only oracle that restores fourth order by supplying the perturber at each
stage, are in `tests/validation/test_third_body.py`. Fixing the freeze in the engine means advancing
perturbers per stage. It is out of scope here and is the remaining limitation.

The freeze scales with the perturber's angular rate *as seen from the parent*. A LEO satellite under
the Moon sees `n_Moon = 2.66e-6 rad/s`, 13 times the Sun's rate for the Moon, so expect a
correspondingly larger first-order coefficient there. It is unmeasured.

Reference tier only, with no compiled twin. `Simulation._refresh_cowell_plan` treats this model's bit as
foreign, so any Cowell set that includes a `"third_body"` body runs on the NumPy `RK4Integrator` path.
"""
from __future__ import annotations

from typing import Final, Mapping, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .custom_types import ScalarSeconds
from .registry import register_force_model

if TYPE_CHECKING:
    from .simulator import Simulation

__all__ = ["THIRD_BODY_MODEL", "THIRD_BODY_PARAM_NAMES", "third_body_kernel"]

THIRD_BODY_MODEL: Final[str] = "third_body"
THIRD_BODY_PARAM_NAMES: Final = ("perturber",)
_PERTURBER_COL: Final[int] = 0


def _validate_perturber(
    sim: "Simulation", bodies: NDArray[np.int64], coefficients: Mapping[str, float],
) -> None:
    """`validate_coefficients` hook for `"third_body"`. See the module docstring for each rule."""
    if "perturber" not in coefficients:
        raise ValueError(
            f"force model '{THIRD_BODY_MODEL}' needs a 'perturber' coefficient (the perturbing body's "
            f"arena slot); a missing one would silently read slot 0. Pass perturber=sim.name_to_index[...], "
            f"or use sweep.ForceModelSpec(body_coefficients={{'perturber': <name>}})."
        )
    value = float(coefficients["perturber"])
    if not value.is_integer() or not 0 <= value < sim.max_capacity:
        raise ValueError(
            f"force model '{THIRD_BODY_MODEL}': perturber={value!r} is not an arena slot "
            f"(expected a whole number in [0, {sim.max_capacity}))."
        )
    slot = int(value)

    is_self = bodies[bodies == slot]
    is_parent = bodies[sim.parent_indices[bodies] == slot]
    if is_self.size > 0:
        raise ValueError(
            f"force model '{THIRD_BODY_MODEL}': slot(s) {is_self.tolist()} would be perturbed by "
            f"themselves (singular direct term)."
        )
    if is_parent.size > 0:
        raise ValueError(
            f"force model '{THIRD_BODY_MODEL}': perturber slot {slot} is the Keplerian parent of "
            f"slot(s) {is_parent.tolist()} (singular indirect term; the parent's pull is "
            f"point_mass_gravity's job)."
        )

    problem = None
    if not sim.active_mask[slot]:
        problem = "is inactive"
    elif sim.is_system[slot]:
        problem = "is a barycentre (its mu row is summed system mass)"
    elif sim.mu_array[slot] <= 0.0:
        problem = "is massless, so it would contribute exactly zero"
    if problem is not None:
        raise ValueError(f"force model '{THIRD_BODY_MODEL}': perturber slot {slot} {problem}.")


@register_force_model(
    THIRD_BODY_MODEL,
    param_names=THIRD_BODY_PARAM_NAMES,
    validate_coefficients=_validate_perturber,
    citation=(
        "Montenbruck & Gill, Satellite Orbits (2000), Sec. 3.3, Eq. 3.37; Vallado 4e Sec. 8.6.3 "
        "(equation and section numbers from memory, unverified). Direct minus indirect point-mass "
        "form, derived in thirdbody.py."
    ),
)
def third_body_kernel(
    indices: NDArray[np.int64],
    t: ScalarSeconds,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    params: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """
    Add the third-body perturbation on each body, relative to its Keplerian parent, to
    `out[indices]` in km/s^2.

    For body `i` with parent `P = parent_indices[i]` and perturber `s = int(params[i, 0])`:

        r   = state[i,:3] - state[P,:3]
        r_s = state[s,:3] - state[P,:3]
        a   = mu_array[s] * [ (r_s - r)/|r_s - r|^3 - r_s/|r_s|^3 ]

    See the module docstring for the derivation, citation, validation rules, and the frozen-perturber
    approximation under `RK4Integrator`. `t` is unused. The perturber's position comes from `state`,
    like every other row.
    """
    perturbers = params[indices, _PERTURBER_COL].astype(np.int64)
    parents = parent_indices[indices]
    mu_s = mu_array[perturbers]

    r_s = state[perturbers, :3] - state[parents, :3]    # perturber relative to the parent
    d = r_s - (state[indices, :3] - state[parents, :3])  # perturber relative to the body: r_s - r

    d2 = np.einsum("ij,ij->i", d, d)
    s2 = np.einsum("ij,ij->i", r_s, r_s)
    # Exactly-zero separations contribute exactly zero (never reachable through enable_force_model).
    inv_d3 = np.divide(1.0, d2 * np.sqrt(d2), out=np.zeros_like(d2), where=d2 > 0.0)
    inv_s3 = np.divide(1.0, s2 * np.sqrt(s2), out=np.zeros_like(s2), where=s2 > 0.0)

    out[indices] += mu_s[:, None] * (d * inv_d3[:, None] - r_s * inv_s3[:, None])
