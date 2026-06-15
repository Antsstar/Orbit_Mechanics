# Defining functions and tools for a orb sandbox.

import math
import numpy as np
from numpy.typing import NDArray
from .types import Radians, Kilometers, Seconds
from .exceptions import ConvergenceError



class Transformations:
    """ 
    Rotation Matrix Toolbox - Takes radians as inputs for euler angles about classic X,Y or Z definitions.
        Also includes cartesian to spherical and vice versa conversions.
    """



    @staticmethod
    def Rx(angle: Radians) -> NDArray[np.float64]:
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
            ])
    
    @staticmethod
    def Ry(angle: Radians) -> NDArray[np.float64]:
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
            ])
    
    @staticmethod
    def Rz(angle: Radians) -> NDArray[np.float64]:
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
            ])
    
    @staticmethod
    def Rxyz(alpha: Radians, beta: Radians, gamma: Radians) -> NDArray[np.float64]:
        return Transformations.Rz(gamma) @ Transformations.Ry(beta) @ Transformations.Rx(alpha)
    
    @staticmethod
    def Rzyx(alpha: Radians, beta: Radians, gamma: Radians) -> NDArray[np.float64]:
        return Transformations.Rx(gamma) @ Transformations.Ry(beta) @ Transformations.Rz(alpha)
    
    @staticmethod
    def Rzxz(alpha: Radians, beta: Radians, gamma: Radians) -> NDArray[np.float64]:
        return Transformations.Rz(gamma) @ Transformations.Rx(beta) @ Transformations.Rz(alpha)
    
    @staticmethod
    def cart_to_spherical(vec: NDArray[np.float64]) -> tuple[float, Radians, Radians]:
        r = np.linalg.norm(vec)
        azimuth = np.arctan2(vec[1], vec[0])
        elevation = np.arcsin(vec[2] / r)
        return float(r), Radians(azimuth), Radians(elevation)
    
    @staticmethod
    def spherical_to_cart(r: float, azimuth: Radians, elevation: Radians) -> NDArray[np.float64]:
        x = r * np.cos(elevation) * np.cos(azimuth)
        y = r * np.cos(elevation) * np.sin(azimuth)
        z = r * np.sin(elevation)
        return np.array([x, y, z])

class Anomalies:
    """
    Anomaly Toolbox - Conversions between true, eccentric and mean anomalies for Elliptic, Parabolic and Hyperbolic orbits.
        All angles are in radians, except for parbolic mean anomaly which is dimensionless. Cardino solution used to solve the cubic from parabolic mean to true anomaly.
    """



    @staticmethod
    def true_to_eccentric(theta: Radians, e: float) -> Radians:
        if theta == np.pi:
            return Radians(np.pi)
        if e < 0:
            raise ValueError(f"Eccentricity cannot be negative. Received e = {e}")
        if e <= 1:
            tanE_2 = np.sqrt( (1 - e) / (1 + e) ) * np.tan(theta / 2)
            return Radians(2*np.atan(tanE_2))
        else:
            tanhE_2 = np.sqrt( (e - 1) / (e + 1) ) * np.tan(theta / 2)
            return Radians(2*np.atanh(tanhE_2))
    

    @staticmethod
    def eccentric_to_true(E: Radians, e: float) -> Radians:
        if E == np.pi:
            return Radians(np.pi)
        if e < 0:
            raise ValueError(f"Eccentricity cannot be negative. Received e = {e}")
        if e <= 1:
            tanTheta_2 = np.sqrt( (1 + e) / (1 - e)) * np.tan(E / 2)
        else:
            tanTheta_2 = np.sqrt( (e + 1) / (e - 1)) * np.tanh(E / 2)
        return Radians(2*np.atan(tanTheta_2))

    @staticmethod
    def eccentric_to_mean(E: Radians, e: float) -> Radians:
        if e < 0:
            raise ValueError(f"Eccentricity cannot be negative. Received e = {e}")
        if e <= 1:
            return Radians(E - e*np.sin(E))
        else:
            return Radians(e*np.sinh(E) - E)

    @staticmethod
    def mean_to_eccentric(M: Radians, e: float, *, tol: float = 1e-5, solver: str = "N-R", max_ite: int = 1000) -> Radians:
        E_0 = 0.0

        if e <= 0.55:
            E_0 = float(M)
        elif 0.55 < e <= 0.95:
            E_0 = float(np.cbrt(6*M))
        elif 0.95 < e <= 1:
            E_0 = float(np.pi)
        else:
            E_0 = float(np.log(2*M / (e+1)))
        
        ite = 0
        if solver == "N-R":
            def func(E: float) -> float:
                if e <= 1:
                    return E - ((M - E + e*math.sin(E)) / (e*math.cos(E) - 1))
                else:
                    return E - ((M + E - e*math.sinh(E)) / (1 - e*math.cosh(E)))
            
        elif solver == "S.S":
            def func(E: float) -> float:
                if e <= 1:
                    return M + e*math.sin(E)
                else:
                    return e*math.sinh(E) - M
            
        else:
            raise ValueError(f"Solver '{solver}' not recognised. Valid options are 'N-R' for Newton-Raphson and 'S.S' for Simple Successive.")
        

        while True:
            E_1 = func(E_0)
            error = abs(E_1 - E_0)
            ite += 1
            
            if error < tol:
                break

            if ite >= max_ite:
                raise ConvergenceError(f"Solver '{solver}' failed to converge after {max_ite} iterations "
                                       f"M={M}, e={e}, last error={error}.")

            E_0 = E_1

        # print(f"Converged in {ite}/{max_ite} iterations!")
        return E_1

    @staticmethod
    def true_to_mean(theta: Radians, e: float) -> Radians:
        E = Anomalies.true_to_eccentric(theta, e)
        M = Anomalies.eccentric_to_mean(E, e)
        return M

    @staticmethod
    def mean_to_true(M: Radians, e: float, *, tol: float = 1e-5, solver: str = "N-R", max_ite: int = 1000) -> Radians:
        E = Anomalies.mean_to_eccentric(M, e, tol=tol, solver=solver, max_ite=max_ite)
        theta = Anomalies.eccentric_to_true(E, e)
        return theta

    @staticmethod
    def true_to_mean_parabolic(theta: Radians) -> float:
        M_p = np.tan(theta/2) + (1.0/3.0)*np.tan(theta/2)**3
        return float(M_p)
    
    @staticmethod
    def mean_to_true_parabolic(M_p: float) -> Radians:
        # Using the Cardino solution for a cubic equation s**3 + 3s - 3Mp = 0
        A = 1.5*M_p
        B = np.cbrt(A + np.sqrt(A**2 + 1.0))
        s = B - (1.0 / B)
        theta = 2.0*np.atan(s)
        return Radians(theta)


    
class Kepler:
    """
    Kepler's Equation relating orbital position and orbital period. Inputs and outputs are in terms of seconds
    and/or radians.
    """



    @staticmethod
    def t_to_M(mu: float, a: Kilometers, delta_t: Seconds) -> Radians:
        return Radians(np.sqrt(mu / (a**3)) * delta_t)
        
    @staticmethod
    def M_to_t(mu: float, a: Kilometers, delta_M: Radians) -> Seconds:
        return Seconds(np.sqrt((a**3) / mu) * delta_M)
    
class Barker:
    """
    Barker's Equation covers the special case of a parabolic trajectory, hence a is undefined and Kepler's
    Equation is no longer sufficient
    """



    @staticmethod
    def t_to_M(mu: float, p: Kilometers, delta_t: Seconds) -> float:
        return float(2.0*np.sqrt(mu / (p**3)) * delta_t)
    
    @staticmethod
    def M_to_t(mu: float, p: Kilometers, delta_M: float) -> Seconds:
        return Seconds(0.5*np.sqrt(p**3 / mu) * delta_M)