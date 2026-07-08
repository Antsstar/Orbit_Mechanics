from __future__ import annotations
from typing import TYPE_CHECKING
from .custom_types import Seconds

import numpy as np
from numpy.typing import NDArray
from .utilities import Anomalies, Kepler, Barker
from . import frames as fr


if TYPE_CHECKING:
    # from .body import BaseBody
    pass

tol = 1e-12

class Propagator:
    @staticmethod
    # def propagate(body: BaseBody, dt: Seconds) -> None:
    def propagate(primary_states: NDArray[np.float64], mu_array: NDArray[np.float64], parent_indices: NDArray[np.int32], dt: Seconds, secondary_states: NDArray[np.float64] | None) -> None:
        raise NotImplementedError


# class KeplerianPropagator(Propagator):
#     @staticmethod
#     def propagate(body: BaseBody, dt: Seconds) -> None:

#         if body.parent is None or body.elements is None:
#             return
            
#         mu = body.mu_orbit
#         if mu is None:
#             return
        
#         coe = body.elements

#         if coe.theta is not None:
#             old_anom = coe.theta
#             anom_name = "theta"
#         elif coe.u is not None:
#             old_anom = coe.u
#             anom_name = "u"
#         else:
#             assert coe.lambda_true is not None, "No valid true anomaly found in orbital elements."
#             old_anom = coe.lambda_true
#             anom_name = "lambda_true"


#         if abs(coe.e - 1.0) < tol:
#             delta_M = Barker.t_to_M(mu, coe.p, dt)
#             old_M = Anomalies.true_to_mean_parabolic(old_anom)
#             new_M = old_M + delta_M
#             new_anom = Anomalies.mean_to_true_parabolic(new_M)
#         elif coe.e < 1.0:
#             delta_M = Kepler.t_to_M(mu, coe.a, dt)
#             old_M = Anomalies.true_to_mean(old_anom, coe.e)
#             new_M = (old_M + delta_M) % (2*np.pi)
#             new_anom = Anomalies.mean_to_true(new_M, coe.e)
#         else:
#             delta_M = Kepler.t_to_M(mu, abs(coe.a), dt)
#             old_M = Anomalies.true_to_mean(old_anom, coe.e)
#             new_M = (old_M + delta_M)
#             new_anom = Anomalies.mean_to_true(new_M, coe.e)

#         update_dict = {anom_name: new_anom}
#         body.elements = body.elements._replace(**update_dict) # type: ignore
#         body.sync_state()

class KeplerianPropagator(Propagator):
    @staticmethod
    def propagate(coe_states: NDArray[np.float64], mu_array: NDArray[np.float64], parent_indices: NDArray[np.int32], dt: Seconds, _local_states: NDArray[np.float64] | None = None) -> None:
        p_col = coe_states[..., 0]
        e_col = coe_states[..., 1] # Fetch all eccentricities
        theta_col = coe_states[..., 5]

        active_nodes = (parent_indices != -1)

        parabolic_mask = (np.abs(e_col - 1.0) < tol) & active_nodes

        hyperbolic_mask = (e_col > 1.0) & active_nodes & ~parabolic_mask
        elliptical_mask = active_nodes & ~parabolic_mask & ~hyperbolic_mask

        if np.any(elliptical_mask):
            mu_ell = mu_array[elliptical_mask]
            p_ell = p_col[elliptical_mask]
            e_ell = e_col[elliptical_mask]
            old_theta = theta_col[elliptical_mask]

            a_ell = p_ell / (1.0 - e_ell**2)

            delta_M = Kepler.t_to_M(mu_ell, a_ell, dt)
            old_M = Anomalies.true_to_mean(old_theta, e_ell)

            new_M = (old_M + delta_M) % (2.0 * np.pi)

            new_theta = Anomalies.mean_to_true(new_M, e_ell)
            print(new_theta-old_theta)

            coe_states[elliptical_mask, 5] = new_theta

        if np.any(hyperbolic_mask):
            mu_hyp = mu_array[hyperbolic_mask]
            p_hyp = p_col[hyperbolic_mask]
            e_hyp = e_col[hyperbolic_mask]
            old_theta = theta_col[hyperbolic_mask]

            a_hyp = np.abs(p_hyp / (1.0 - e_hyp**2))

            delta_M = Kepler.t_to_M(mu_hyp, a_hyp, dt)
            old_M = Anomalies.true_to_mean(old_theta, e_hyp)

            new_M = old_M + delta_M

            new_theta = Anomalies.mean_to_true(new_M, e_hyp)

            coe_states[hyperbolic_mask, 5] = new_theta

        if np.any(parabolic_mask):
            mu_par = mu_array[parabolic_mask]
            p_par = p_col[parabolic_mask]
            # e_par = e_col[parabolic_mask]
            old_theta = theta_col[parabolic_mask]


            delta_M = Barker.t_to_M(mu_par, p_par, dt)
            old_M = Anomalies.true_to_mean_parabolic(old_theta)

            new_M = old_M + delta_M

            new_theta = Anomalies.mean_to_true_parabolic(new_M)

            coe_states[parabolic_mask, 5] = new_theta


        return
        # coe_states[parabolic_mask, 5] = Barker.t_to_M_vectorized(mu_array[parabolic_mask], coe_states[parabolic_mask, 0], dt)
        # coe_states[hyperbolic_mask, 5] = Kepler.t_to_M(mu_array[hyperbolic_mask], abs(coe_states[hyperbolic_mask, 0]), dt) % (2*np.pi)
        # coe_states[eliptical_mask, 5] = Kepler.t_to_M(mu_array[eliptical_mask], coe_states[eliptical_mask, 0], dt)
        # pass

        