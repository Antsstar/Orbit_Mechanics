from __future__ import annotations
from typing import TYPE_CHECKING
from .types import Seconds

import numpy as np
from .utilities import Anomalies, Kepler, Barker
from . import frames as fr


if TYPE_CHECKING:
    from .body import BaseBody

tol = 1e-12

class Propagator:
    @staticmethod
    def propagate(body: BaseBody, dt: Seconds) -> None:
        raise NotImplementedError


class KeplerianPropagator(Propagator):
    @staticmethod
    def propagate(body: BaseBody, dt: Seconds) -> None:

        if body.parent is None or body.elements is None:
            return
            
        mu = body.mu_orbit
        if mu is None:
            return
        
        coe = body.elements

        if coe.theta is not None:
            old_anom = coe.theta
            anom_name = "theta"
        elif coe.u is not None:
            old_anom = coe.u
            anom_name = "u"
        else:
            assert coe.lambda_true is not None, "No valid true anomaly found in orbital elements."
            old_anom = coe.lambda_true
            anom_name = "lambda_true"


        if abs(coe.e - 1.0) < tol:
            delta_M = Barker.t_to_M(mu, coe.p, dt)
            old_M = Anomalies.true_to_mean_parabolic(old_anom)
            new_M = old_M + delta_M
            new_anom = Anomalies.mean_to_true_parabolic(new_M)
        elif coe.e < 1.0:
            delta_M = Kepler.t_to_M(mu, coe.a, dt)
            old_M = Anomalies.true_to_mean(old_anom, coe.e)
            new_M = (old_M + delta_M) % (2*np.pi)
            new_anom = Anomalies.mean_to_true(new_M, coe.e)
        else:
            delta_M = Kepler.t_to_M(mu, abs(coe.a), dt)
            old_M = Anomalies.true_to_mean(old_anom, coe.e)
            new_M = (old_M + delta_M)
            new_anom = Anomalies.mean_to_true(new_M, coe.e)

        update_dict = {anom_name: new_anom}
        body.elements = body.elements._replace(**update_dict) # type: ignore
        body.sync_state()

        