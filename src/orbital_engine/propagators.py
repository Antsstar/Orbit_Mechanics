from __future__ import annotations
from typing import TYPE_CHECKING, Any
from .custom_types import Seconds, PropagatorType
from .registry import register_propagator

import numpy as np
from numpy.typing import NDArray
from .utilities import Anomalies, Kepler, Barker
from . import frames as fr


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


        # 1. Find all active bodies that are not heads
        sib_mask = active_mask & ~is_head
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


register_propagator(PropagatorType.KEPLERIAN, KeplerianPropagator)

        