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


        # copy = coe_states.copy()
        parent_mu = mu_array[parent_indices]

        KeplerianPropagator._step_anomalies(dt=dt,
                                            coe_states=coe_states,
                                            # mu_array=mu_array,
                                            mu_array=parent_mu,
                                            mask=active_mask)


        r, v, success = fr.ReferenceFrames.coe_to_rv(coe_states[active_mask], parent_mu[active_mask]) # Eventually just implement in place value returns.
        valid_idx = np.where(active_mask)[0][success]
        parents = parent_indices[active_mask]
        parent_is_sys = is_system[parents]

        if np.any(parent_is_sys):
            m_c = mu_array[active_mask][parent_is_sys]
            m_s = mu_array[parents][parent_is_sys]
            m_h = m_s - m_c

            scale = m_s / (m_h  + 1e-20) # avoid div by 0
            r[parent_is_sys] /= scale[:, None]
            v[parent_is_sys] /= scale[:, None]


        local_states[valid_idx, :3] = r[success]
        local_states[valid_idx, 3:] = v[success] 

        KeplerianPropagator._reflex_kick(local_states=local_states,
                                         mu_array=mu_array,
                                         mask=active_mask,
                                         is_head=is_head,
                                         is_system=is_system,
                                         body_sys_map=body_sys_map,
                                         sys_head_map=sys_head_map)

        


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

    @staticmethod
    def _reflex_kick(local_states: NDArray[np.float64], mu_array: NDArray[np.float64], mask: NDArray[np.bool_], 
                     is_head: NDArray[np.bool_], is_system: NDArray[np.bool_], body_sys_map: NDArray[np.int32], sys_head_map: NDArray[np.int32]) -> None:
        """Calculate position adjustment for system heads due to other bodies motion, to ensure a fixed barycenter."""
        if not np.any(is_head):
            return
        
        # child_mask = ~is_head & ~is_system & mask & (body_sys_map != -1)
        child_mask = ~is_head & mask & (body_sys_map != -1)
        if not np.any(child_mask):
            return
        
        sys_moments = np.zeros_like(local_states)
        mass_weighted_states = local_states[child_mask] * mu_array[child_mask, None]
        np.add.at(sys_moments, body_sys_map[child_mask], mass_weighted_states)

        head_mask = is_head & mask
        head_sys_id = body_sys_map[head_mask]

        local_states[head_mask] = -sys_moments[head_sys_id] / mu_array[head_sys_id, None]

        # heads_of_children = sys_head_map[body_sys_map[child_mask]]
        heads_of_children = sys_head_map[child_mask]
        valid_head_mask = heads_of_children != -1

        valid_children = np.where(child_mask)[0][valid_head_mask]
        valid_heads = heads_of_children[valid_head_mask]

        local_states[valid_children] += local_states[valid_heads]



register_propagator(PropagatorType.KEPLERIAN, KeplerianPropagator)

        