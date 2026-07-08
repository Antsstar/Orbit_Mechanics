import numpy as np
from numpy.typing import NDArray
from .custom_types import Radians, Kilometers, Seconds
from typing import NamedTuple, Optional, cast
from .utilities import Transformations, Anomalies, Kepler, Barker

# def angle(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
#     """ Returns the angle between two vectors in radians."""
#     return float(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)))
def angle(a: NDArray[np.float64], b: NDArray[np.float64], n: NDArray[np.float64]) -> float | NDArray[np.float64]:
    """ Returns the angle between two vectors in radians."""
    # c = np.cross(a, b)
    # theta = np.arctan2(np.dot(n, np.cross(a, b)), np.dot(a, b))
    theta: NDArray[np.float64] = np.arctan2(np.sum(n * np.cross(a, b), axis=-1), np.sum(a * b, axis=-1)) % (2.0 * np.pi)

    if a.ndim == 1 and b.ndim == 1:
        return Radians(theta.item())
    return theta
    # return np.arctan2(np.dot(c / np.linalg.norm(c), c), np.dot(a, b))



class OrbitalElements(NamedTuple):
    family      : str
    a           : Kilometers
    p           : Kilometers
    e           : float
    i           : Radians
    Omega       : Optional[Radians]
    omega       : Optional[Radians]
    theta       : Optional[Radians]
    tor         : Optional[Seconds]
    omega_true  : Optional[Radians]
    u           : Optional[Radians]
    lambda_true : Optional[Radians]


# class ReferenceFrames:
#     """
#     Transformation Toolbox for converting between different reference frames and orbital elements.
#         But specifically for orbital mechanics. Regular caretsian to spehrical transformations are available in the Transformations class.
#     """

#     @staticmethod
#     def rv_to_coe(r: NDArray[np.float64], v: NDArray[np.float64], mu: float, *, ref_x: NDArray[np.float64] = np.array([1, 0, 0]), ref_z: NDArray[np.float64] = np.array([0, 0, 1])) -> 'OrbitalElements':
#         if np.dot(ref_x, ref_z) != 0:
#             # print("Reference directions are not orthogonal!")
#             raise ValueError("Reference directions are not orthogonal!")
#             # return
        
#         ref_y = np.cross(ref_z, ref_x)
#         tol = 1e-12

#         orb_case    = ""   # Classification of orbit
#         a           = None # Semi-major axis > Semi-latus rectum "p" = h^2 / mu
#         p           = None # Semi-latus Rectum h^2 / mu
#         e_mag       = None # Eccentricity
#         i           = None # Inclination
#         Omega       = None # Right Ascension of the Ascending Node (RAAN)
#         omega       = None # Argument of Perigee
#         theta       = None # True anomaly
#         tor         = None # Time of periapsis passage
#         omega_true  = None # Omega + omega (x_ref.e) Non-Circular Equatorial, True longitude of periapsis
#         u           = None # omega + theta (N.r) Circular Inclined, True argument of latitude
#         lambda_true = None # Omega + omega + theta (x_ref . r) Circular Equatorial, True longitude

#         # h           = None # Angular Momemntum Vector
#         # e           = None # Eccentricity Vector
#         # N           = None # Nodal Vector

        
#         h = np.cross(r, v)
#         h_mag = float(np.linalg.norm(h))

        
#         # if h_mag == 0:
#         if abs(h_mag) < tol:
#             # print("Velocity and displacement are parallel. Entity is not in orbit")
#             raise ValueError("Velocity and displacement are parallel. Entity is not in orbit")
#             # return

#         e = (np.cross(v, h) / mu) - (r / np.linalg.norm(r))
#         e_mag = float(np.linalg.norm(e))

#         p = h_mag**2 / mu

#         if abs(e_mag - 1) < tol:
#             a = np.inf
#         else:
#             a =  p * (1 / (1-e_mag**2))


#         i = np.arccos(np.dot(h, ref_z) / (np.linalg.norm(h) * np.linalg.norm(ref_z)))


#         is_circular     = bool(abs(e_mag) < tol)
#         is_equatorial   = bool(abs(i) < tol or abs(i - np.pi) < tol)
#         i_type          = ""
#         e_type          = ""

#         assert a is not None, "Semi-major axis 'a' should not be None at this point."
#         assert p is not None, "Semi-latus rectum 'p' should not be None at this point."

#         match [is_circular, is_equatorial]:
#             case [True, True]:
#                 # orb_case = "Circular, Equatorial"
#                 e_type = "Circular"
#                 i_type = "Equatorial"
#                 lambda_true = angle(ref_x, r)
#                 if np.dot(ref_y, r) < 0:
#                     lambda_true = 2*np.pi - lambda_true
                
#                 delta_M = Anomalies.true_to_mean(lambda_true, e_mag)
#                 tor = -Kepler.M_to_t(mu, abs(a), delta_M)

#             case [True, _]:
#                 # orb_case = "Circular"
#                 e_type = "Circular"
#                 N = np.cross(ref_z, h)

#                 Omega = angle(N, ref_x)
#                 if np.dot(ref_y, N) < 0:
#                     Omega = 2*np.pi - Omega

#                 u = angle(r, N)
#                 if np.dot(ref_z, r) < 0:
#                     u = 2*np.pi - u

#                 delta_M = Anomalies.true_to_mean(u, e_mag)
#                 tor = -Kepler.M_to_t(mu, abs(a), delta_M)

#             case [_, True]:
#                 # orb_case = "Equatorial"
#                 i_type = "Equatorial"
#                 omega_true = angle(ref_x, e)
#                 if np.dot(ref_y, e) < 0:
#                     omega_true = 2*np.pi - omega_true

#             case _:
#                 N = np.cross(ref_z, h)

#                 Omega = angle(N, ref_x)
#                 if np.dot(ref_y, N) < 0:
#                     Omega = 2*np.pi - Omega

#                 omega = angle(e, N)
#                 if np.dot(ref_z, e) < 0:
#                     omega = 2*np.pi - omega

#         if not is_circular:
#             theta = angle(r, e)
#             if np.dot(r, v) < 0:
#                 theta = 2*np.pi - theta

#             if abs(e_mag - 1) < tol:
#                     delta_Mp = Anomalies.true_to_mean_parabolic(theta)
#                     tor = -Barker.M_to_t(mu, p, delta_Mp)
#             else:
#                 delta_M = Anomalies.true_to_mean(theta, e_mag)
#                 tor = -Kepler.M_to_t(mu, abs(a), delta_M)

#             if abs(e_mag - 1) < tol:
#                 e_type = "Parabolic"
#             elif e_mag < 1:
#                 e_type = "Elliptic"
#             else:
#                 e_type = "Hyperbolic"

                
#         if np.degrees(i) < 90:
#             i_type = " ".join([i_type, "Prograde"]) if i_type != "" else "Prograde"
#         elif np.degrees(i) == 90:
#             # i_type = " ".join([i_type, "Polar"]) if i_type != "" else "Polar"
#             i_type = "Polar"
#         else:
#             i_type = " ".join([i_type, "Retrograde"]) if i_type != "" else "Retrograde"

#             # orb_case = ", ".join([orb_case, i_type])

#         orb_case = ", ".join([e_type, i_type])

#         elements = OrbitalElements(
#             family=orb_case, 
#             a=a, 
#             p=p, 
#             e=e_mag, 
#             i=i, 
#             Omega=Omega if Omega is not None else None, 
#             omega=omega if omega is not None else None, 
#             theta=theta if theta is not None else None,
#             tor=tor if tor is not None else None, 
#             omega_true=omega_true if omega_true is not None else None,
#             u=u if u is not None else None,
#             lambda_true=lambda_true if lambda_true is not None else None
#             )

#         return elements
    
#     @staticmethod
#     def coe_to_rv(coe: 'OrbitalElements', mu: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

#         if coe.theta is not None:
#             anomaly = coe.theta
#         elif coe.u is not None:
#             anomaly = coe.u
#         elif coe.lambda_true is not None:
#             anomaly = coe.lambda_true
#         else:
#             raise ValueError("No valid argument for anomaly")
#             # print("No valid arguemnt for anomaly")
#             # return


#         r_mag = coe.p / (1 + coe.e*np.cos(anomaly))
#         x = np.array([np.cos(anomaly), 0, 0])*r_mag
#         y = np.array([0, np.sin(anomaly), 0])*r_mag



#         mu_h = np.sqrt(mu / coe.p)
#         x_dot = np.array([-np.sin(anomaly), 0, 0])*mu_h
#         y_dot = np.array([0, coe.e + np.cos(anomaly), 0])*mu_h

#         r_p = x + y
#         v_p = x_dot + y_dot

        
#         matrix = Transformations.Rzxz(coe.omega if coe.omega is not None else (coe.omega_true if coe.omega_true is not None else Radians(0.0)), coe.i, coe.Omega if coe.Omega is not None else Radians(0.0))
#         r = matrix @ r_p
#         v = matrix @ v_p


#         return r, v
    
#     @staticmethod
#     def inertia_to_fixed(r_i: NDArray[np.float64], v_i: NDArray[np.float64], theta: Radians) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
#         matrix = Transformations.Rz(theta)
#         r_f = matrix @ r_i
#         v_f = matrix @ v_i
#         return r_f, v_f
    
#     @staticmethod
#     def fixed_to_inertia(r_f: NDArray[np.float64], v_f: NDArray[np.float64], theta: Radians) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
#         r_i, v_i = ReferenceFrames.inertia_to_fixed(r_f, v_f, -theta)
#         return r_i, v_i

#     @staticmethod
#     def inertia_to_RaDec(r_i: NDArray[np.float64]) -> tuple[Radians, Radians, float]:
#         alt, Ra, Dec = Transformations.cart_to_spherical(r_i)
#         return Radians(Ra), Radians(Dec), alt

#     @staticmethod
#     def RaDec_to_inertia(Ra: Radians, Dec: Radians, alt: float = 1.0) -> NDArray[np.float64]:
#         r_i = Transformations.spherical_to_cart(alt, Ra, Dec)
#         return r_i
    
#     @staticmethod
#     def fixed_to_longlat(r_f: NDArray[np.float64]) -> tuple[Radians, Radians, float]:
#         alt, long, lat = Transformations.cart_to_spherical(r_f)
#         return Radians(long), Radians(lat), alt

#     @staticmethod
#     def longlat_to_fixed(long: Radians, lat: Radians, alt: float = 1.0) -> NDArray[np.float64]:
#         r_f = Transformations.spherical_to_cart(alt, long, lat)
#         return r_f

class ReferenceFrames:
    """
    Transformation Toolbox for converting between different reference frames and orbital elements.
        But specifically for orbital mechanics. Regular caretsian to spehrical transformations are available in the Transformations class.
    """

    @staticmethod
    # def rv_to_coe(r: NDArray[np.float64], v: NDArray[np.float64], mu: float, *, ref_x: NDArray[np.float64] = np.array([1, 0, 0]), ref_z: NDArray[np.float64] = np.array([0, 0, 1])) -> 'OrbitalElements':
    def rv_to_coe(r: NDArray[np.float64], v: NDArray[np.float64], mu: float | NDArray[np.float64], *, ref_x: NDArray[np.float64] = np.array([1, 0, 0]), ref_z: NDArray[np.float64] = np.array([0, 0, 1])) -> NDArray[np.float64]:
        if np.any(np.dot(ref_x, ref_z)) != 0:
            raise ValueError("Reference directions are not orthogonal!")
        
        ref_y = np.cross(ref_z, ref_x)
        tol = 1e-12
        _mu = np.asarray(mu)
        _r = np.atleast_2d(r)
        _v = np.atleast_2d(v)

        orb_case    = ""   # Classification of orbit
        a           = None # Semi-major axis > Semi-latus rectum "p" = h^2 / mu
        p           = None # Semi-latus Rectum h^2 / mu
        e_mag       = None # Eccentricity
        i           = None # Inclination
        Omega       = None # Right Ascension of the Ascending Node (RAAN)
        omega       = None # Argument of Perigee
        theta       = None # True anomaly
        tor         = None # Time of periapsis passage
        omega_true  = None # Omega + omega (x_ref.e) Non-Circular Equatorial, True longitude of periapsis
        u           = None # omega + theta (N.r) Circular Inclined, True argument of latitude
        lambda_true = None # Omega + omega + theta (x_ref . r) Circular Equatorial, True longitude

        
        h = np.cross(_r, _v)
        h_mag = np.linalg.norm(h, axis=-1)

        if np.any(np.abs(h_mag)) < tol:
            raise ValueError("Velocity and displacement are parallel. Entity is not in orbit")

        e = (np.cross(_v, h) / _mu[..., np.newaxis]) - (_r / np.linalg.norm(_r, axis=-1)[..., np.newaxis])
        e_mag = np.linalg.norm(e, axis=-1)

        p = h_mag**2 / _mu


        a_mask = (np.abs(e_mag - 1.0) < tol)
        a = np.empty_like(p)
        a[a_mask] = np.inf
        a[~a_mask] = p[~a_mask] / (1.0 - e_mag[~a_mask]**2) # Not sure if I need a...


        # i = np.arccos(np.dot(h, ref_z) / (np.linalg.norm(h) * np.linalg.norm(ref_z)))
        i = np.arccos(np.clip(np.sum(h * ref_z, axis=-1) / (h_mag), -1.0, 1.0))


        # is_circular = (np.abs(e_mag) < tol)
        is_circular = (e_mag < tol)
        is_equatorial = (np.abs(i) < tol) | (np.abs(i - np.pi) < tol)

        # assert a is not None, "Semi-major axis 'a' should not be None at this point."
        # assert p is not None, "Semi-latus rectum 'p' should not be None at this point."

        # assert any(item is not None for item in a)
        # assert any(item is not None for item in p)

        mask_NC_NEqu = ~is_circular & ~is_equatorial    # Standard Orbit
        mask_NC_Equ = ~is_circular & is_equatorial      # Equatorial (N is undefined) 
        mask_C_NEqu = is_circular & ~is_equatorial      # Circular (e is undefined)
        mask_C_Equ = is_circular & is_equatorial        # Circular (N and e are undefined)

        coe_states = np.zeros((_r.shape[0], 6), dtype=np.float64)
        coe_states[:, 0] = p
        coe_states[:, 1] = e_mag
        coe_states[:, 2] = i

        # a . b = |a||b|cos(<ab)
        # a x b = /n\ |a||b|sin(<ab)
        # a x b / a.b = /n\ tan(<ab)
        # /n\ . a x b / a.b = tan(<ab)

        h_hat = h / h_mag[..., np.newaxis]
        N = np.cross(ref_z, h)

        mask_NE = ~is_equatorial
        mask_NC = ~is_circular

        # if np.any(~is_equatorial):
        if np.any(mask_NE):
            # coe_states[~is_equatorial, 3] = angle(ref_x, N[~is_equatorial], ref_z) # Regular Omega
            coe_states[mask_NE, 3] = angle(ref_x, N[mask_NE], ref_z) # Regular Omega
        
        # if np.any(~is_circular):
        #     coe_states[~is_circular, 5] = angle(e[~is_circular], _r[~is_circular], h_hat[~is_circular]) # Regular theta
        if np.any(mask_NC):
            coe_states[mask_NC, 5] = angle(e[mask_NC], _r[mask_NC], h_hat[mask_NC]) # Regular theta

        if np.any(mask_NC_Equ):
            coe_states[mask_NC_Equ, 4] = angle(ref_x, e[mask_NC_Equ], h_hat[mask_NC_Equ]) # omega absorb Omega
        
        if np.any(mask_C_NEqu):
            coe_states[mask_C_NEqu, 5] = angle(N[mask_C_NEqu], _r[mask_C_NEqu], h_hat[mask_C_NEqu]) # theta absorb omega
        
        if np.any(mask_NC_NEqu):
            coe_states[mask_NC_NEqu, 4] = angle(N[mask_NC_NEqu], e[mask_NC_NEqu], h_hat[mask_NC_NEqu]) # Regular omega

        if np.any(mask_C_Equ):
            coe_states[mask_C_Equ, 5] = angle(ref_x, _r[mask_C_Equ], h_hat[mask_C_Equ]) # theta absorbs Omega and omega

        
        if r.ndim == 1:
            return cast(NDArray[np.float64], coe_states[0])
        return coe_states
    
    @staticmethod
    # def coe_to_rv(coe: 'OrbitalElements', mu: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    def coe_to_rv(coe: NDArray[np.float64], mu: float | NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

        # if coe.theta is not None:
        #     anomaly = coe.theta
        # elif coe.u is not None:
        #     anomaly = coe.u
        # elif coe.lambda_true is not None:
        #     anomaly = coe.lambda_true
        # else:
        #     raise ValueError("No valid argument for anomaly")
        #     # print("No valid arguemnt for anomaly")
        #     # return
        _coe = np.atleast_2d(coe)
        _mu = np.asarray(mu)

        p = _coe[..., 0]
        e = _coe[..., 1]
        i = _coe[..., 2]
        Omega = _coe[..., 3]
        omega = _coe[..., 4]
        anomaly = _coe[..., 5]


        # r_mag = coe.p / (1 + coe.e*np.cos(anomaly))
        # x = np.array([np.cos(anomaly), 0, 0])*r_mag
        # y = np.array([0, np.sin(anomaly), 0])*r_mag

        r_mag = p / (1.0 + e * np.cos(anomaly))
        r_x = r_mag * np.cos(anomaly)
        r_y = r_mag * np.sin(anomaly)
        rv_z = np.zeros_like(r_mag)



        # mu_h = np.sqrt(mu / coe.p)
        # x_dot = np.array([-np.sin(anomaly), 0, 0])*mu_h
        # y_dot = np.array([0, coe.e + np.cos(anomaly), 0])*mu_h

        mu_h = np.sqrt(_mu / p)
        v_x = -mu_h * np.sin(anomaly)
        v_y = mu_h * (e + np.cos(anomaly))

        # r_p = x + y
        # v_p = x_dot + y_dot
        r_p = np.stack((r_x, r_y, rv_z), axis=-1)
        v_p = np.stack((v_x, v_y, rv_z), axis=-1)

        
        # matrix = Transformations.Rzxz(coe.omega if coe.omega is not None else (coe.omega_true if coe.omega_true is not None else Radians(0.0)), coe.i, coe.Omega if coe.Omega is not None else Radians(0.0))
        # r = matrix @ r_p
        # v = matrix @ v_p
        matrix = Transformations.Rzxz(omega, i, Omega)
        r = (matrix @ r_p[..., np.newaxis])[..., 0]
        v = (matrix @ v_p[..., np.newaxis])[..., 0]

        if coe.ndim == 1:
            return r[0], v[0]

        return r, v
    
    @staticmethod
    def inertia_to_fixed(r_i: NDArray[np.float64], v_i: NDArray[np.float64], theta: Radians | NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        matrix = Transformations.Rz(theta)
        r_f = matrix @ r_i
        v_f = matrix @ v_i
        return r_f, v_f
    
    @staticmethod
    def fixed_to_inertia(r_f: NDArray[np.float64], v_f: NDArray[np.float64], theta: Radians | NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r_i, v_i = ReferenceFrames.inertia_to_fixed(r_f, v_f, -theta)
        return r_i, v_i

    @staticmethod
    def inertia_to_RaDec(r_i: NDArray[np.float64]) -> tuple[Radians | NDArray[np.float64], Radians | NDArray[np.float64], float | NDArray[np.float64]]:
        alt, Ra, Dec = Transformations.cart_to_spherical(r_i)
        # return Radians(Ra), Radians(Dec), alt
        return Ra, Dec, alt

    @staticmethod
    def RaDec_to_inertia(Ra: Radians | NDArray[np.float64], Dec: Radians | NDArray[np.float64], alt: float | NDArray[np.float64] = 1.0) -> NDArray[np.float64]:
        r_i = Transformations.spherical_to_cart(alt, Ra, Dec)
        return r_i
    
    @staticmethod
    def fixed_to_longlat(r_f: NDArray[np.float64]) -> tuple[Radians | NDArray[np.float64], Radians | NDArray[np.float64], float | NDArray[np.float64]]:
        alt, long, lat = Transformations.cart_to_spherical(r_f)
        # return Radians(long), Radians(lat), alt
        return long, lat, alt

    @staticmethod
    def longlat_to_fixed(long: Radians | NDArray[np.float64], lat: Radians | NDArray[np.float64], alt: float | NDArray[np.float64] = 1.0) -> NDArray[np.float64]:
        r_f = Transformations.spherical_to_cart(alt, long, lat)
        return r_f

if __name__ == "__main__":
    MU_Sun = 1.32712440042 * 10**11
    r = np.array([-145510750, 39268690, 10500])
    v = np.array([-6.995, -29.215, -0.00025])
    elements = ReferenceFrames.rv_to_coe(r, v, mu=MU_Sun)
    print(elements)
    r_new, v_new = ReferenceFrames.coe_to_rv(elements, MU_Sun)

    # 4. Check results (using np.allclose to handle tiny floating point errors)
    print("Position Match:", np.allclose(r, r_new))
    print("Velocity Match:", np.allclose(v, v_new))
    print(r, v, np.linalg.norm(r), np.linalg.norm(v))
    print(r_new, v_new, np.linalg.norm(r_new), np.linalg.norm(v_new))