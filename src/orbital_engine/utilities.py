# Defining functions and tools for a orb sandbox.

import math
import numpy as np
from numpy.typing import NDArray
from .custom_types import Radians, Kilometers, Seconds
from typing import Union, cast
from .exceptions import ConvergenceError



class Transformations:
    """ 
    Rotation Matrix Toolbox - Takes radians as inputs for euler angles about classic X,Y or Z definitions.
        Also includes cartesian to spherical and vice versa conversions.
    """



    @staticmethod
    # def Rx(angle: Radians) -> NDArray[np.float64]:
    #     return np.array([
    #         [1, 0, 0],
    #         [0, np.cos(angle), -np.sin(angle)],
    #         [0, np.sin(angle), np.cos(angle)]
    #         ])
    def Rx(angle: Radians | NDArray[np.float64]) -> NDArray[np.float64]:
        _angle = np.atleast_1d(angle)
        c, s = np.cos(_angle), np.sin(_angle)

        mat = np.zeros(_angle.shape + (3, 3), dtype=np.float64)

        mat[..., 0, 0] = 1.0
        mat[..., 1, 1] = c
        mat[..., 1, 2] = -s
        mat[..., 2, 1] = s
        mat[..., 2, 2] = c

        if np.asarray(angle).ndim == 0:
            return cast(NDArray[np.float64], mat[0])
        return mat
    
    @staticmethod
    def Ry(angle: Radians | NDArray[np.float64]) -> NDArray[np.float64]:
        _angle = np.atleast_1d(angle)
        c, s = np.cos(_angle), np.sin(_angle)

        mat = np.zeros(_angle.shape + (3, 3), dtype=np.float64)

        mat[..., 0, 0] = c
        mat[..., 0, 2] = s
        mat[..., 1, 1] = 1.0
        mat[..., 2, 0] = -s
        mat[..., 2, 2] = c

        if np.asarray(angle).ndim == 0:
            return cast(NDArray[np.float64], mat[0])
        return mat
    
    @staticmethod
    def Rz(angle: Radians | NDArray[np.float64]) -> NDArray[np.float64]:
        _angle = np.atleast_1d(angle)
        c, s = np.cos(_angle), np.sin(_angle)

        mat = np.zeros(_angle.shape + (3, 3), dtype=np.float64)

        mat[..., 0, 0] = c
        mat[..., 0, 1] = -s
        mat[..., 1, 0] = s
        mat[..., 1, 1] = c
        mat[..., 2, 2] = 1.0

        if np.asarray(angle).ndim == 0:
            return cast(NDArray[np.float64], mat[0])
        return mat
    
    @staticmethod
    def Rxyz(alpha: Radians | NDArray[np.float64], beta: Radians | NDArray[np.float64], gamma: Radians | NDArray[np.float64]) -> NDArray[np.float64]:
        return Transformations.Rz(gamma) @ Transformations.Ry(beta) @ Transformations.Rx(alpha)
    
    @staticmethod
    def Rzyx(alpha: Radians | NDArray[np.float64], beta: Radians | NDArray[np.float64], gamma: Radians | NDArray[np.float64]) -> NDArray[np.float64]:
        return Transformations.Rx(gamma) @ Transformations.Ry(beta) @ Transformations.Rz(alpha)
    
    @staticmethod
    def Rzxz(alpha: Radians | NDArray[np.float64], beta: Radians | NDArray[np.float64], gamma: Radians | NDArray[np.float64]) -> NDArray[np.float64]:
        return Transformations.Rz(gamma) @ Transformations.Rx(beta) @ Transformations.Rz(alpha)
    
    @staticmethod
    def cart_to_spherical(vec: NDArray[np.float64]) -> tuple[float, Radians, Radians] | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        # r = np.linalg.norm(vec)
        _r = np.atleast_2d(vec)
        r = np.linalg.norm(_r, axis=-1)
        azimuth = np.arctan2(_r[..., 1], _r[..., 0])
        elevation = np.arcsin(np.clip(_r[..., 2] / r, -1.0, 1.0))
        if vec.ndim == 1:
            return r.item(), Radians(azimuth.item()), Radians(elevation.item())
        return r, azimuth, elevation
    
    @staticmethod
    def spherical_to_cart(r: float | NDArray[np.float64], azimuth: Radians | NDArray[np.float64], elevation: Radians | NDArray[np.float64]) -> NDArray[np.float64]:
        _r = np.asarray(r)
        _azimuth = np.asarray(azimuth)
        _elevation = np.asarray(elevation)

        x = r * np.cos(_elevation) * np.cos(_azimuth)
        y = r * np.cos(_elevation) * np.sin(_azimuth)
        z = r * np.sin(_elevation)

        # return np.array([x, y, z])
        # vec = np.stack((x, y, z), axis=-1)

        # if _r.ndim == 0:
        #     return vec
        # return vec
        return np.stack((x, y, z), axis=-1)


class Anomalies:
    """
    Anomaly Toolbox - Conversions between true, eccentric and mean anomalies for Elliptic, Parabolic and Hyperbolic orbits.
        All angles are in radians, except for parbolic mean anomaly which is dimensionless. Cardino solution used to solve the cubic from parabolic mean to true anomaly.
    """



    @staticmethod
    # def true_to_eccentric(theta: Union[Radians, NDArray[np.float64]], e: Union[float, NDArray[np.float64]]) -> Union[Radians, NDArray[np.float64]]:
    def true_to_eccentric(theta: Radians | NDArray[np.float64], e: float | NDArray[np.float64]) -> Radians | NDArray[np.float64]:

        # 1. Cast to numpy arrays
        _theta = np.asarray(theta, dtype=np.float64)
        _e = np.asarray(e, dtype=np.float64)

        # 2. Strict Guarding
        if np.any(_e < 0.0):
            raise ValueError(f"Eccentricity cannot be negative. Recieved e = {_e}")
        if np.any(np.abs(_e - 1.0) < 1e-8):
            raise ValueError(f"Parabolic orbits (e=1) do not have eccentric anomalies.")
        
        # 3. Vectorize Math (Run both Elliptical and Hyperbolic)
        ell_mask = _e < 1.0
        hyp_mask = _e > 1.0

        # 4. Initialize empty result array
        E = np.empty_like(_theta)

        # 5. Ellipitcal (+ Circular) cases
        if np.any(ell_mask):
            e_ell = _e[ell_mask]
            theta_ell = _theta[ell_mask]

            tanE_2 = np.sqrt( (1.0 - e_ell) / (1.0 + e_ell) ) * np.tan(theta_ell / 2.0)
            E[ell_mask] = 2.0 * np.arctan(tanE_2)

        # 6. Hyperbolic cases
        if np.any(hyp_mask):
            e_hyp = _e[hyp_mask]
            theta_hyp = _theta[hyp_mask]

            tanhE_2 = np.sqrt( (e_hyp - 1.0) / (e_hyp + 1.0) ) * np.tan(theta_hyp / 2.0)
            E[hyp_mask] = 2.0 * np.arctanh(tanhE_2)

        # 7. Scalar Guard
        if _theta.ndim == 0:
            return Radians(E.item())
        
        return E

    

    @staticmethod
    # def eccentric_to_true(E: Radians, e: float) -> Radians:
    def eccentric_to_true(E: Radians | NDArray[np.float64], e: float | NDArray[np.float64]) -> Radians | NDArray[np.float64]:

        # if E == np.pi:
        #     return Radians(np.pi)
        # if e < 0:
        #     raise ValueError(f"Eccentricity cannot be negative. Received e = {e}")
        # if e <= 1:
        #     tanTheta_2 = np.sqrt( (1 + e) / (1 - e)) * np.tan(E / 2)
        # else:
        #     tanTheta_2 = np.sqrt( (e + 1) / (e - 1)) * np.tanh(E / 2)
        # return Radians(2*np.atan(tanTheta_2))

        # 1. Cast to numpy arrays
        _E = np.asarray(E, dtype=np.float64)
        _e = np.asarray(e, dtype=np.float64)

        # 2. Strict Guarding
        if np.any(_e < 0.0):
            raise ValueError(f"Eccentricity cannot be negative. Recieved e = {_e}")
        if np.any(np.abs(_e - 1.0) < 1e-8):
            raise ValueError(f"Parabolic orbits (e=1) do not have eccentric anomalies.")
        
        # 3. Vectorize Math (Run both Elliptical and Hyperbolic)
        ell_mask = _e < 1.0
        hyp_mask = _e > 1.0

        # 4. Initialize empty result array
        theta = np.empty_like(_E)

        # 5. Ellipitcal (+ Circular) cases
        if np.any(ell_mask):
            e_ell = _e[ell_mask]
            E_ell = _E[ell_mask]

            # tanE_2 = np.sqrt( (1.0 + e_ell) / (1.0 - e_ell) ) * np.tan(E_ell / 2.0)
            y = np.sqrt(1.0 + e_ell) * np.sin(E_ell / 2.0)
            x = np.sqrt(1.0 - e_ell) * np.cos(E_ell / 2.0)
            # theta[ell_mask] = 2.0 * np.arctan(tanE_2)
            theta[ell_mask] = 2.0 * np.arctan2(y, x)

        # 6. Hyperbolic cases
        if np.any(hyp_mask):
            e_hyp = _e[hyp_mask]
            E_hyp = _E[hyp_mask]

            tanhE_2 = np.sqrt( (e_hyp + 1.0) / (e_hyp - 1.0) ) * np.tanh(E_hyp / 2.0)
            theta[hyp_mask] = 2.0 * np.arctan(tanhE_2)

        # 7. Scalar Guard
        if _E.ndim == 0:
            return Radians(theta.item())
        
        return theta



    @staticmethod
    # def eccentric_to_mean(E: Radians, e: float) -> Radians:
    def eccentric_to_mean(E: Radians | NDArray[np.float64], e: float | NDArray[np.float64]) -> Radians | NDArray[np.float64]:

        # if e < 0:
        #     raise ValueError(f"Eccentricity cannot be negative. Received e = {e}")
        # if e <= 1:
        #     return Radians(E - e*np.sin(E))
        # else:
        #     return Radians(e*np.sinh(E) - E)

        # 1. Cast to numpy arrays
        _E = np.asarray(E, dtype=np.float64)
        _e = np.asarray(e, dtype=np.float64)

        # 2. Strict Guarding
        if np.any(_e < 0.0):
            raise ValueError(f"Eccentricity cannot be negative. Recieved e = {_e}")
        if np.any(np.abs(_e - 1.0) < 1e-8):
            raise ValueError(f"Parabolic orbits (e=1) do not have eccentric anomalies.")
        
        # 3. Vectorize Math (Run both Elliptical and Hyperbolic)
        ell_mask = _e < 1.0
        hyp_mask = _e > 1.0

        # 4. Initialize empty result array
        M = np.empty_like(_E)

        # 5. Ellipitcal (+ Circular) cases
        if np.any(ell_mask):
            e_ell = _e[ell_mask]
            E_ell = _E[ell_mask]

            # tanE_2 = np.sqrt( (1.0 + e_ell) / (1.0 - e_ell) ) * np.tan(E_ell / 2.0)
            M[ell_mask] = E_ell - e_ell * np.sin(E_ell)

        # 6. Hyperbolic cases
        if np.any(hyp_mask):
            e_hyp = _e[hyp_mask]
            E_hyp = _E[hyp_mask]

            # tanhE_2 = np.sqrt( (e_hyp + 1.0) / (e_hyp - 1.0) ) * np.tanh(E_hyp / 2.0)
            M[hyp_mask] = e_hyp * np.sinh(E_hyp) - E_hyp

        # 7. Scalar Guard
        if _E.ndim == 0:
            return Radians(M.item())
        
        return M

    @staticmethod
    # def mean_to_eccentric(M: Radians, e: float, *, tol: float = 1e-5, solver: str = "N-R", max_ite: int = 1000) -> Radians:
    def mean_to_eccentric(M: Radians | NDArray[np.float64], e: float | NDArray[np.float64], *, tol: float = 1e-5, solver: str = "N-R", max_ite: int = 1000) -> Radians | NDArray[np.float64]:

        _M = np.asarray(M, dtype=np.float64)
        _e = np.asarray(e, dtype=np.float64)

        # a) Strict Guarding
        if np.any(_e < 0.0):
            raise ValueError(f"Eccentricity cannot be negative. Recieved e = {_e}")
        if np.any(np.abs(_e - 1.0) < 1e-8):
            raise ValueError(f"Parabolic orbits (e=1) do not have eccentric anomalies.")
        
        # 1. 
        E = np.empty_like(_M)

        mask1 = (_e <= 0.55)
        mask2 = (0.55 < _e) & (_e <= 0.95)
        mask3 = (0.95 < _e) & (_e <= 1.0)
        mask4 = (1.0 < _e)

        if np.any(mask1): E[mask1] = _M[mask1]
        if np.any(mask2): E[mask2] = np.cbrt(6.0*_M[mask2])
        if np.any(mask3): E[mask3] = np.pi
        # if np.any(mask4): E[mask4] = np.log(2.0*_M[mask4] / _e[mask4])
        if np.any(mask4): E[mask4] = np.arcsinh(_M[mask4] / _e[mask4])


        active = np.ones_like(_M, dtype=bool)
        ite = 0

        mask_ell = ~mask4
        while np.any(active) and ite < max_ite:
            E_act = E[active]
            e_act = _e[active]
            M_act = _M[active]
            # print(E_act.shape) # Found the error, if they converge out of sync, then active array decreases in size, but the masks (i.e, mask_ell) still refers to the global size

            delta = np.zeros_like(E_act)

            if solver == "N-R":
                if np.any(mask_ell):
                    E_ell = E_act[mask_ell[active]] # Filters out entries that aren't active in the mask!
                    e_e = e_act[mask_ell[active]]
                    f = E_ell - e_e * np.sin(E_ell) - M_act[mask_ell[active]]
                    f_prime = e_e * np.cos(E_ell) - 1.0
                    delta[mask_ell[active]] = f / f_prime
                
                if np.any(mask4):
                    H_hyp = E_act[mask4[active]]
                    e_h = e_act[mask4[active]]
                    f = e_h * np.sinh(H_hyp) - H_hyp - M_act[mask4[active]]
                    f_prime = 1.0 - e_h * np.cosh(H_hyp)
                    delta[mask4[active]] = f / f_prime

            elif solver == "S.S":
                if np.any(mask_ell):
                    delta[mask_ell] = M_act[mask_ell] + e_act[mask_ell] * np.sin(E_act[mask_ell]) - E_act[mask_ell]
                if np.any(mask4):
                    delta[mask4] = e_act[mask4] * np.sinh(E_act[mask4]) - M_act[mask4] - E_act[mask4]

            else:
                raise ValueError(f"Solver '{solver}' not recognised. Valid options are 'N-R' for Newton-Raphson and S.S for Successive Substitution.")
            
            E[active] += delta

            still_active = np.abs(delta) > tol
            active[active] = still_active
            ite += 1


        if np.any(active):
            raise ConvergenceError(f"Solver '{solver}' failed to converge after {max_ite} iterations for all entries"
                                    f"Failed entries values: M={_M[active]}, e={_e[active]}, element mask={active}")

        if _M.ndim == 0:
            return Radians(E.item())
        return E





    @staticmethod
    # def true_to_mean(theta: Radians, e: float) -> Radians:
    def true_to_mean(theta: Radians | NDArray[np.float64], e: float | NDArray[np.float64]) -> Radians | NDArray[np.float64]:
        E = Anomalies.true_to_eccentric(theta, e)
        M = Anomalies.eccentric_to_mean(E, e)
        return M

    @staticmethod
    # def mean_to_true(M: Radians, e: float, *, tol: float = 1e-5, solver: str = "N-R", max_ite: int = 1000) -> Radians:
    def mean_to_true(M: Radians | NDArray[np.float64], e: float | NDArray[np.float64], *, tol: float = 1e-5, solver: str = "N-R", max_ite: int = 1000) -> Radians | NDArray[np.float64]:
        E = Anomalies.mean_to_eccentric(M, e, tol=tol, solver=solver, max_ite=max_ite)
        theta = Anomalies.eccentric_to_true(E, e)
        return theta

    @staticmethod
    # def true_to_mean_parabolic(theta: Radians) -> float:
    def true_to_mean_parabolic(theta: Radians | NDArray[np.float64]) -> float | NDArray[np.float64]:
        _theta = np.asarray(theta, dtype=np.float64)
        M_p = np.tan(_theta/2) + (1.0/3.0)*np.tan(_theta/2)**3
        
        if _theta.ndim == 0:
            return float(M_p.item())
        return M_p
    
    @staticmethod
    # def mean_to_true_parabolic(M_p: float) -> Radians:
    def mean_to_true_parabolic(M_p: float | NDArray[np.float64]) -> Radians | NDArray[np.float64]:
        # Using the Cardino solution for a cubic equation s**3 + 3s - 3Mp = 0
        _M_p = np.asarray(M_p, dtype=np.float64)
        A = 1.5*_M_p
        B = np.cbrt(A + np.sqrt(A**2 + 1.0), dtype=np.float64)
        s = B - (1.0 / B)
        theta = 2.0*np.arctan(s, dtype=np.float64)

        if _M_p.ndim == 0:
            return Radians(theta.item())
        return theta


    
class Kepler:
    """
    Kepler's Equation relating orbital position and orbital period. Inputs and outputs are in terms of seconds
    and/or radians.
    """



    @staticmethod
    def t_to_M(mu: float | NDArray[np.float64], a: Kilometers | NDArray[np.float64], delta_t: Seconds) -> Radians | NDArray[np.float64]:
        _mu = np.asarray(mu, dtype=np.float64)
        _a = np.asarray(a, dtype=np.float64)

        M = np.sqrt(_mu / (_a ** 3), dtype=np.float64) * delta_t
        if _mu.ndim == 0:
            return Radians(M.item())
        # return Radians(np.sqrt(mu / (a**3)) * delta_t)
        return M
        
    @staticmethod
    def M_to_t(mu: float | NDArray[np.float64], a: Kilometers | NDArray[np.float64], delta_M: Radians | NDArray[np.float64]) -> Seconds:
        _mu = np.asarray(mu, dtype=np.float64)
        _a = np.asarray(a, dtype=np.float64)
        _M = np.asarray(delta_M, dtype=np.float64)

        t = np.sqrt((_a ** 3) / _mu) * _M

        if t.ndim > 0 and not np.allclose(t, t[0], atol=1e-8):
            raise NotImplementedError(f"This function expects an homogenous time constant between all entries")
        
        if t.ndim == 0:
            return Seconds(t.item())
        return Seconds(t[0])
        # return Seconds(np.sqrt((a**3) / mu) * delta_M)
    
class Barker:
    """
    Barker's Equation covers the special case of a parabolic trajectory, hence a is undefined and Kepler's
    Equation is no longer sufficient
    """



    @staticmethod
    def t_to_M(mu: float | NDArray[np.float64], p: Kilometers | NDArray[np.float64], delta_t: Seconds) -> float | NDArray[np.float64]:
        _mu = np.asarray(mu, dtype=np.float64)
        _p = np.asarray(p, dtype=np.float64)

        M = 2.0 * np.sqrt(_mu / (_p ** 3), dtype=np.float64) * delta_t
        if _mu.ndim == 0:
            return float(M.item())
        return M
        # return float(2.0*np.sqrt(mu / (p**3)) * delta_t)
    
    @staticmethod
    def M_to_t(mu: float | NDArray[np.float64], p: Kilometers | NDArray[np.float64], delta_M: float | NDArray[np.float64]) -> Seconds:
        _mu = np.asarray(mu, dtype=np.float64)
        _p = np.asarray(p, dtype=np.float64)
        _M = np.asarray(delta_M, dtype=np.float64)

        t = 0.5*np.sqrt((_p ** 3) /_mu) * _M
        # return Seconds(0.5*np.sqrt(p**3 / mu) * delta_M)
        if t.ndim > 0 and not np.allclose(t, t[0], atol=1e-8):
            raise NotImplementedError(f"This function expects an homogenous time constant between all entries")
        
        if t.ndim == 0:
            return Seconds(t.item())
        return Seconds(t[0])