from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast
from .custom_types import Seconds, ScalarSeconds, PropagatorType
from .registry import register_propagator

import numpy as np
from numpy.typing import NDArray
from .utilities import Anomalies, Kepler, Barker
from . import frames as fr
from .integrators import Integrator
from .forces import AccelerationProvider


if TYPE_CHECKING:
    # from .body import BaseBody
    pass

tol = 1e-12

class Propagator:
    """Abstract base class for all orbital propagation strategies."""
    @staticmethod
    def propagate(dt: Seconds, **kwargs: Any) -> None:
        raise NotImplementedError # Need to update to make use of **kwargs!


class KeplerianPropagator(Propagator):
    @staticmethod
    def propagate(dt: Seconds, **kwargs: Any) -> None:
        local_states: NDArray[np.float64]   = kwargs['secondary_states']
        coe_states: NDArray[np.float64]     = kwargs['primary_states']
        mu_array: NDArray[np.float64]       = kwargs['mu_array']
        parent_indices: NDArray[np.int32]   = kwargs['parent_indices']
        active_mask: NDArray[np.bool_]      = kwargs['active_mask']
        is_head: NDArray[np.bool_]          = kwargs['is_head']      # Should the input be forced to be arrays?
        is_system: NDArray[np.bool_]        = kwargs['is_system']    # Should these be refactored to .get()?
        body_sys_map: NDArray[np.int32]     = kwargs['body_sys_map'] # Check the numpy typing
        sys_head_map: NDArray[np.int32]     = kwargs['sys_head_map'] # New! will actually likely be useful
        max_capacity: int                   = len(local_states)


        # 1. Find all active bodies that are not heads. An explicit `sib_mask` override lets a caller
        # exclude Cowell-designated bodies from analytic propagation without changing this method's
        # default behaviour for any existing caller - see Simulation.step and CowellPropagator below.
        sib_mask_override: NDArray[np.bool_] | None = kwargs.get('sib_mask')
        sib_mask = sib_mask_override if sib_mask_override is not None else (active_mask & ~is_head)
        sibs = np.where(sib_mask)[0]

        # heads = parent_indices[sibs]
        # Pre-allocate zeroed kicks to ensure they remain in scope.
        kick_r = np.zeros((max_capacity, 3), dtype=np.float64)
        kick_v = np.zeros((max_capacity, 3), dtype=np.float64)

        if len(sibs) > 0:
            # 2. Strict Parent Focus
            parents = parent_indices[sibs]
            mu_calc = mu_array[sibs] + mu_array[parents]

            # In-place Anomaly update, need to pass entire array, with mask to avoid copying.
            KeplerianPropagator._step_anomalies(dt, coe_states, mu_array + mu_array[parent_indices], sib_mask)

            # Generate Pure relative vectors anchored strictly to parent_indices.
            r_rel, v_rel, success = fr.ReferenceFrames.coe_to_rv(coe_states[sibs], mu_calc)

            valid_sibs = sibs[success]
            valid_parents = parents[success]

            # --- True Topological Filter ---

            # Condition 1: Are we in a Barycentric System?
            systems = body_sys_map[valid_sibs]
            is_bary_sys = (systems != -1) & (is_system[systems]) # Guard to avoid -1 indexing

            # Condition 2: Explicitly orbiting the System head?
            heads = sys_head_map[valid_sibs]
            orbits_head = valid_parents == heads

            # Filter Arrays
            is_bary_sib = is_bary_sys & orbits_head

            bary_sibs = valid_sibs[is_bary_sib]
            bary_heads = valid_parents[is_bary_sib]

            if len(bary_sibs) > 0:
                # 1. Accumulate mass-weighted relative vectors for each head
                r_sums = np.zeros_like(kick_r)
                v_sums = np.zeros_like(kick_v)
                sib_mass = mu_array[bary_sibs]
                # m_sums = np.zeros(max_capacity, dtype=np.float64)

                np.add.at(r_sums, bary_heads, r_rel[success][is_bary_sib] * sib_mass[:, None])
                np.add.at(v_sums, bary_heads, v_rel[success][is_bary_sib] * sib_mass[:, None])
                # np.add.at(m_sums, valid_heads, sib_mass)

                # total_mass = m_sums + mu_array
                # 2. O(1) Total Mass Lookup directly from Barycenter's mu_array.
                barycenters = body_sys_map[bary_heads]
                total_mass = mu_array[barycenters]

                # valid_sys = total_mass > 0
                valid_sys = total_mass > 0 #& is_head
                # print(valid_sys.shape, kick_r[valid_sys].shape)

                kick_r[bary_heads[valid_sys]] = -r_sums[bary_heads[valid_sys]] / total_mass[valid_sys, None] # Just to avoid div by zero.
                kick_v[bary_heads[valid_sys]] = -v_sums[bary_heads[valid_sys]] / total_mass[valid_sys, None]

            # 3. Finalize Sibling Local States.
            # Base state: Every sibling gets its pure relative vector from its parent.
            local_states[valid_sibs, :3] = r_rel[success]
            local_states[valid_sibs, 3:] = v_rel[success]

            # Shift ONLY the barycentric Siblings that orbit the Head
            local_states[bary_sibs, :3] += kick_r[bary_heads]
            local_states[bary_sibs, 3:] += kick_v[bary_heads]

        # 4. Finalize Head Local States.
        active_heads = np.where(active_mask & is_head)[0]
        if len(active_heads) > 0:
            local_states[active_heads, :3] = kick_r[active_heads]
            local_states[active_heads, 3:] = kick_v[active_heads]
        

        return


    @staticmethod # Right now just applies flatly to all bodies. But should it run for heads?
    def _step_anomalies(dt: Seconds, coe_states: NDArray[np.float64], mu_array: NDArray[np.float64], mask: NDArray[np.bool_]) -> None:
        """Advance anomaly for all active bodies by delta t"""
        if not np.any(mask):
            return
        
        active_coes = coe_states[mask]
        active_mu = mu_array[mask]

        p_col = active_coes[..., 0]
        e_col = active_coes[..., 1]
        theta_col = active_coes[..., 5]

        is_parabolic = np.isclose(e_col, 1.0, atol=1e-9)
        not_parabolic = ~is_parabolic

        # new_true_anomalies = np.zeros(len(active_coes), dtype=np.float64)

        if np.any(not_parabolic):
            # idx = not_parabolic
            p_np = p_col[not_parabolic]
            e_np = e_col[not_parabolic]
            theta_np = theta_col[not_parabolic]
            mu_np = active_mu[not_parabolic]

            a_np = p_np / (1.0 - e_np**2)

            is_elliptic = e_np < 1.0
            not_elliptic = ~is_elliptic

            a_np[not_elliptic] = np.abs(a_np[not_elliptic])
            delta_M, _ = Kepler.t_to_M(mu_np, a_np, dt)
            old_M = Anomalies.true_to_mean(theta_np, e_np)
            new_M = old_M + delta_M
            
            if np.ndim(new_M) == 1: # Check these flags
                assert isinstance(new_M, np.ndarray)
                new_M[is_elliptic] = new_M[is_elliptic] % (2.0 * np.pi)
            elif is_elliptic:
                new_M = new_M % (2.0*np.pi)

            theta_col[not_parabolic] = Anomalies.mean_to_true(new_M, e_np)

        if np.any(is_parabolic):
            p_p = p_col[is_parabolic]
            theta_p = theta_col[is_parabolic]
            mu_p = active_mu[is_parabolic]

            delta_M = Barker.t_to_M(mu_p, p_p, dt)
            old_M = Anomalies.true_to_mean_parabolic(theta_p)
            new_M = old_M + delta_M
            theta_col[is_parabolic] = Anomalies.mean_to_true_parabolic(new_M)

        coe_states[mask, 5] = theta_col


class CowellPropagator(Propagator):
    """
    Adapter that lets Cowell propagation register through the same `registry._PROPAGATOR_REGISTRY`
    mechanism `KeplerianPropagator` uses, so `Simulation.step` selects between them by reading the
    registry rather than branching on a hardcoded class. Unlike `KeplerianPropagator`, Cowell needs a
    *stateful* integrator - `integrators.RK4Integrator` owns scratch sized to `max_capacity`, allocated
    once by `Simulation.__init__` (see that module's docstring for why) - so this adapter takes that
    integrator as a keyword argument rather than constructing one itself. `Propagator.propagate` stays
    a stateless staticmethod for both propagators; only the object it delegates to differs.
    """
    @staticmethod
    def propagate(dt: Seconds, **kwargs: Any) -> None:
        integrator: Integrator = kwargs['integrator']
        provider: AccelerationProvider = kwargs['provider']
        t: ScalarSeconds = kwargs['t']
        state: NDArray[np.float64] = kwargs['state']
        indices: NDArray[np.int64] = kwargs['indices']
        primaries: NDArray[np.int32] = kwargs['primaries']
        integrator.step(provider, t, state, float(dt), indices, primaries)


class SecularJ2Propagator(Propagator):
    """
    Analytic Keplerian propagation plus first-order secular drift of RAAN, argument of periapsis and
    mean anomaly under the J2 zonal harmonic - the "Kepler + secular J2" tier between the plain
    analytic propagator and Cowell + `point_mass_gravity` + `j2` (`geopotential.py`). Like
    `KeplerianPropagator`, this is closed-form: no accumulated numerical state, only three additional
    per-body rates advanced linearly in time alongside the ordinary two-body anomaly advance.

    **What is held constant.** Under first-order secular J2 theory `p`, `e` and `i` do not drift - only
    RAAN, argument of periapsis and mean anomaly do - so `secular_j2_rates` below computes the three
    rates once, at `Simulation.set_propagator` time, from whatever `p`/`e`/`i` the body held then; they
    are cached (`Simulation._secular_j2_rates`) and reused every step rather than recomputed, mirroring
    the "scratch is arena-owned, recompute only what changes" discipline `kernels.py` already documents
    for `_kick`/`_accum`.

    **Citation.** Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Eq. 9-41 is the
    likely reference for these three rates - **not checked against the text, as `geopotential.py`
    discloses for its own citations**. Curtis, *Orbital Mechanics for Engineering Students*, 3rd ed.,
    covers the RAAN and argument-of-periapsis rates independently and is not checked either.

    **Coefficient storage.** `(j2, r_eq)` are written into the same `Simulation.force_model_params["j2"]`
    array `geopotential.j2_kernel` reads (`geopotential.J2_PARAM_NAMES = ("j2", "r_eq")`), populated by
    `Simulation.set_propagator`'s `SECULAR_J2` branch rather than `enable_force_model`. This is a
    storage choice only: it never sets the `"j2"` force-model mask bit, so it does not add this body to
    `Simulation.accelerations`' dispatch set (irrelevant here - a body on this propagator never calls
    `accelerations`) and does not collide with a body that separately has the `"j2"` force model enabled
    for Cowell. The payoff is that a sweep comparing this tier against the Cowell + `point_mass_gravity`
    + `j2` tier can share one coefficient-population call for both, so a coefficient sensitivity sweep
    cannot accidentally compare the two tiers at different J2 values.

    **Mean-vs-osculating approximation.** The arena stores osculating elements; this propagator advances
    them as if they were mean elements, the quantity the secular theory is actually derived for -
    `CLAUDE.md` forbids converting between the two, so there is no correction applied. Two distinct
    effects follow, and the second dominates:

    - A bounded, orbit-period *oscillation* about the true trajectory - the short-period J2 terms the
      averaging discards - amplitude `O(J2 (R/p)^2)` in the angles (dimensionless) and
      `O(J2 (R/p)^2 * p)` in position; for a 550 km-altitude LEO orbit (`p ~ 6928` km,
      `J2 (R/p)^2 ~ 9.2e-4`) that is on the order of 6 km, neither shrinking nor growing with time.
    - A *linearly growing* along-track drift: the mean motion `n = sqrt(mu/a^3)` this propagator caches
      is computed once from the seeded *osculating* `p`, not the true mean `p` (which differs from the
      osculating value by the same `O(J2 (R/p)^2)` fraction), so the propagated mean anomaly carries a
      small, fixed fractional rate error that accumulates rather than averaging out - `O(J2 (R/p)^2)`
      radians of phase, i.e. roughly `J2 (R/p)^2 * 2*pi*p` km of position, **per orbit elapsed**, not a
      one-time offset. Measured this session at 550 km / 53 deg against Cowell +
      `point_mass_gravity` + `j2`: 57.4 km after 1 orbit, 172.1 km after 3, 573.6 km after 10 - linear
      to better than 1% (ratios 3.00 and 9.99 against the exactly-linear predictions), and the
      ~40 km/orbit order-of-magnitude estimate above matches the measured ~57 km/orbit rate to within a
      factor of ~1.4. This is the effect that actually dominates the comparison beyond a handful of
      orbits, not the bounded oscillation. **The rate is phase-dependent:** it scales as |cos 2u0| in
      the initial argument of latitude, because the bias is the short-period term in the osculating
      semi-major axis at epoch. The 57.4 km/orbit figure is the worst case (u0 = 0). At u0 = 45 deg the
      error after 10 orbits was 0.2 km. See `docs/architecture.md`, secular-J2 section.

    `tests/validation/test_secular_j2_propagator.py` measures both against Cowell + `point_mass_gravity`
    + `j2`. This is the *comparison* case in `CLAUDE.md`'s verification/comparison split, not an engine
    bug: the divergence is what this tier of the frontier plot exists to show, and the second effect in
    particular is why "Kepler + secular J2" is the *interesting middle point* rather than simply
    "correct until it isn't" - its error grows with elapsed time, unlike Cowell's, which is controlled
    by step size instead.

    **Configuration-time restrictions**, enforced by `Simulation.set_propagator` rather than left to
    produce a plausible-looking wrong trajectory - see that method's docstring for the full reasoning:
    active, non-head, non-barycenter, not its own kinematic bubble (same restrictions Cowell carries,
    same reasons); Keplerian parent not a barycentre (`geopotential.barycentre_parented`, since a
    barycentre's J2 is meaningless and the kernel cannot detect it - see `geopotential.py`); `mu == 0`
    (same reflex-kick-corruption reason Cowell requires it - this propagator, like Cowell, is excluded
    from `Simulation._kepler_sib_idx` and so never contributes its mass to any head's reflex kick);
    `0 <= e < 1` (a closed orbit - the rates derive from mean motion `n = sqrt(mu/a^3)`, undefined for
    an open orbit).

    **How position is recovered.** `propagate` only advances `coe_states` and writes the resulting
    parent-relative state vector into caller-owned `rel_out` - it does not touch `local_states` or
    `global_states`, because the result is relative to `parent_indices`, which generally differs from
    the `body_sys_map` bubble `local_states` is measured against (the Moon is the canonical case; see
    `CLAUDE.md`). `Simulation.step` re-bases `rel_out` onto the parent's end-of-step global position
    once `calc_global()` has produced it - exactly the pattern `docs/architecture.md`'s Cowell section
    documents, just without an integrator: this propagator is analytic, so there is no `parent_state_
    at_start` snapshot to subtract, only a fresh add.
    """
    @staticmethod
    def propagate(dt: Seconds, **kwargs: Any) -> None:
        coe_states: NDArray[np.float64] = kwargs['coe_states']
        mu_array: NDArray[np.float64] = kwargs['mu_array']
        parent_indices: NDArray[np.int32] = kwargs['parent_indices']
        rates: NDArray[np.float64] = kwargs['rates']
        indices: NDArray[np.int64] = kwargs['indices']
        rel_out: NDArray[np.float64] = kwargs['rel_out']

        if len(indices) == 0:
            return

        e = coe_states[indices, 1]
        theta = coe_states[indices, 5]

        old_M = Anomalies.true_to_mean(theta, e)
        new_M = (old_M + rates[indices, 2] * dt) % (2.0 * np.pi)
        new_theta = Anomalies.mean_to_true(new_M, e)

        # Elements are updated unconditionally, mirroring kernels.kepler_propagate's own convention -
        # only the derived state vector below is ever withheld on an invalid orbit.
        coe_states[indices, 3] += rates[indices, 0] * dt
        coe_states[indices, 4] += rates[indices, 1] * dt
        coe_states[indices, 5] = new_theta

        parents = parent_indices[indices]
        mu_calc = mu_array[indices] + mu_array[parents]
        r_rel, v_rel, success = fr.ReferenceFrames.coe_to_rv(coe_states[indices], mu_calc)

        # Written only at the valid subset, leaving an invalid orbit's rel_out row untouched - the
        # same convention KeplerianPropagator.propagate uses for local_states via valid_sibs.
        valid = indices[success]
        rel_out[valid, :3] = r_rel[success]
        rel_out[valid, 3:] = v_rel[success]


def secular_j2_rates(
    coe_states: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    j2_params: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    """
    First-order secular J2 rates for `indices`, shape `(len(indices), 3)`:
    `[dRAAN/dt, dARG_PE/dt, dM/dt]`, radians/second.

        a  = p / (1 - e^2)
        n  = sqrt(mu / a^3)
        dRAAN/dt  = -(3/2) n J2 (R/p)^2 cos(i)
        dARGPE/dt =  (3/4) n J2 (R/p)^2 (5 cos^2(i) - 1)
        dM/dt     =  n [1 + (3/4) J2 (R/p)^2 sqrt(1 - e^2) (3 cos^2(i) - 1)]

    `mu` is the two-body mass sum `KeplerianPropagator` and `Simulation._rehydrate_coes` already use:
    `mu_array[body] + mu_array[parent_indices[body]]`. `p`, `e`, `i` are read once, at the moment this
    is called (`Simulation.set_propagator`), and the returned rates are cached rather than recomputed
    every step - see `SecularJ2Propagator`'s docstring for why that is valid under this theory.

    Citation: Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Eq. 9-41 - the likely
    reference, **not checked against the text**; see `SecularJ2Propagator`'s docstring.
    """
    parents = parent_indices[indices]
    mu = mu_array[indices] + mu_array[parents]

    p = coe_states[indices, 0]
    e = coe_states[indices, 1]
    inc = coe_states[indices, 2]

    j2 = j2_params[indices, 0]
    r_eq = j2_params[indices, 1]

    a = p / (1.0 - e * e)
    n = np.sqrt(mu / a**3)
    factor = j2 * (r_eq / p) ** 2
    cos_i = np.cos(inc)
    cos2_i = cos_i * cos_i

    raan_dot = -1.5 * n * factor * cos_i
    argpe_dot = 0.75 * n * factor * (5.0 * cos2_i - 1.0)
    m_dot = n * (1.0 + 0.75 * factor * np.sqrt(1.0 - e * e) * (3.0 * cos2_i - 1.0))

    return np.stack([raan_dot, argpe_dot, m_dot], axis=1)


def mean_seeded_p(
    coe_states: NDArray[np.float64],
    j2_params: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    """
    First-order short-period correction converting the *osculating* semi-latus rectum at epoch into an
    estimate of the constant *mean* `p` this propagator's own secular theory is derived for. An
    optional seeding for `SecularJ2Propagator` (`Simulation.set_propagator(..., mean_seed=True)`); the
    default (`mean_seed=False`) leaves `p` at its osculating, as-given value and is bit-identical to
    the propagator's behaviour before this function existed.

    **Why this is the correction that matters.** `SecularJ2Propagator`'s docstring identifies the
    dominant error of osculating seeding as a *linearly growing* along-track drift, caused by caching
    mean motion `n = sqrt(mu/a^3)` from the osculating `a`, which differs from the true mean `a` by an
    `O(J2 (R/p)^2)` fraction that does not average out. This function removes exactly that bias at the
    one point it is introduced - `set_propagator` time - rather than correcting it per step; `e` and
    `i` are left untouched, since this task narrows `CLAUDE.md`'s osculating<->mean rule specifically
    to "seed the semi-major axis (and so its mean motion)", not a full osculating-to-mean element set.

    **Derivation (near-circular limit, first order in J2; self-contained, checkable without a text -
    the same standard this project's other citations set for themselves).** The J2 potential energy
    per unit mass is `Phi = (mu J2 Re^2 / (2 r^3)) (3 sin^2(i) sin^2(u) - 1)`, `u = arg_pe + theta` the
    argument of latitude (`geopotential.py`'s and `reference.py`'s own convention, `a = -grad Phi`).
    Lagrange's planetary equations are conventionally stated in terms of the *disturbing function*
    `R = -Phi` (so that `d^2r/dt^2 = -mu r/r^3 + grad R`), giving
    `R = (mu J2 Re^2 / (2 r^3)) (1 - 3 sin^2(i) sin^2(u))`. Taking `r ~ a` (the `e -> 0` limit) and
    `du/dt ~ n` (the short-period integration treats the much slower secular RAAN/arg_pe rates as
    constant over one orbit), `da/dt = (2 / (n a)) dR/dM` gives

        da/dt = -3 n J2 Re^2 sin^2(i) sin(2u) / a.

    Integrating with `d(2u) = 2n dt` gives the short-period oscillation of the *osculating* a about
    the (locally constant) mean:

        a_osc(u) - a_mean = (3/2) J2 (Re/a)^2 a sin^2(i) cos(2u),

    so, inverting to leading order at the epoch argument of latitude `u0`,

        a_mean = a_osc * (1 - (3/2) J2 (Re/p)^2 sin^2(i) cos(2*u0)),      p_mean = a_mean (1 - e^2).

    This reproduces the `|cos 2u0|` phase signature `docs/architecture.md`'s secular-J2 section already
    measured for the *osculating*-seeded propagator's error against Cowell + `point_mass_gravity` +
    `j2` - the same mechanism, read the other way round.

    **A sign slip was caught by measurement, not by re-reading the derivation.** The first version of
    this docstring used `R = +Phi` (the potential itself, not the disturbing function), which inverts
    every sign below it and produces `a_mean = a_osc * (1 + frac)` - applying that against the J2 truth
    *doubled* the along-track error instead of removing it (measured: 573.5 km at 10 orbits / u0=0
    became 1146.2 km, almost exactly 2x, rather than collapsing toward the ~0.16 km floor u0=45 deg
    shows). A factor of ~-1 between a correction and what it should have cancelled is the signature of
    an inverted sign, not a wrong magnitude - see `docs/engineering-log.md`'s "asserting an expectation
    instead of deriving it" and "the secular-J2 error was assumed bounded, then measured unbounded"
    entries for the same lesson elsewhere in this project. Corrected here to `R = -Phi`, the standard
    disturbing-function convention; `tests/validation/test_secular_j2_propagator.py`'s mean-seeding
    test is what would have caught the original sign error, had it existed when this was first written.

    Citation: Kozai, Y. (1959), "The Motion of a Close Earth Satellite", *Astronomical Journal* 64,
    p.367, and Brouwer, D. (1959), "Solution of the Problem of Artificial Satellite Theory Without
    Drag", *Astronomical Journal* 64, p.378, are the likely sources for the general (all-order-in-e)
    short-period `a` term this reduces to at `e -> 0` - **not checked against either text**, matching
    this codebase's convention for citations recalled from memory (see `geopotential.py`,
    `SecularJ2Propagator`). The derivation above is the checkable part; the *magnitude* was not fitted
    to any measured figure, only the sign was corrected after measurement exposed it as backwards - see
    `tests/validation/test_secular_j2_propagator.py` for the measured numbers this now predicts.
    """
    p = coe_states[indices, 0]
    e = coe_states[indices, 1]
    inc = coe_states[indices, 2]
    arg_pe = coe_states[indices, 4]
    theta = coe_states[indices, 5]

    j2 = j2_params[indices, 0]
    r_eq = j2_params[indices, 1]

    a_osc = p / (1.0 - e * e)
    u0 = arg_pe + theta
    frac = 1.5 * j2 * (r_eq / p) ** 2 * np.sin(inc) ** 2 * np.cos(2.0 * u0)
    a_mean = a_osc * (1.0 - frac)
    return cast(NDArray[np.float64], a_mean * (1.0 - e * e))


register_propagator(PropagatorType.KEPLERIAN, KeplerianPropagator)
register_propagator(PropagatorType.COWELL, CowellPropagator)
register_propagator(PropagatorType.SECULAR_J2, SecularJ2Propagator)

        