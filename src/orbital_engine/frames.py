import numpy as np
from numpy.typing import NDArray
from .types import Radians, Kilometers, Seconds
from typing import NamedTuple, Optional
from .utilities import Transformations, Anomalies, Kepler, Barker

def angle(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """ Returns the angle between two vectors in radians."""
    return float(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)))


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


class ReferenceFrames:
    """
    Transformation Toolbox for converting between different reference frames and orbital elements.
        But specifically for orbital mechanics. Regular caretsian to spehrical transformations are available in the Transformations class.
    """

    @staticmethod
    def rv_to_coe(r: NDArray[np.float64], v: NDArray[np.float64], mu: float, *, ref_x: NDArray[np.float64] = np.array([1, 0, 0]), ref_z: NDArray[np.float64] = np.array([0, 0, 1])) -> 'OrbitalElements':
        if np.dot(ref_x, ref_z) != 0:
            # print("Reference directions are not orthogonal!")
            raise ValueError("Reference directions are not orthogonal!")
            # return
        
        ref_y = np.cross(ref_z, ref_x)
        tol = 1e-12

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

        # h           = None # Angular Momemntum Vector
        # e           = None # Eccentricity Vector
        # N           = None # Nodal Vector

        
        h = np.cross(r, v)
        h_mag = float(np.linalg.norm(h))

        
        # if h_mag == 0:
        if abs(h_mag) < tol:
            # print("Velocity and displacement are parallel. Entity is not in orbit")
            raise ValueError("Velocity and displacement are parallel. Entity is not in orbit")
            # return

        e = (np.cross(v, h) / mu) - (r / np.linalg.norm(r))
        e_mag = float(np.linalg.norm(e))

        p = h_mag**2 / mu

        if abs(e_mag - 1) < tol:
            a = np.inf
        else:
            a =  p * (1 / (1-e_mag**2))


        i = np.arccos(np.dot(h, ref_z) / (np.linalg.norm(h) * np.linalg.norm(ref_z)))


        is_circular     = bool(abs(e_mag) < tol)
        is_equatorial   = bool(abs(i) < tol or abs(i - np.pi) < tol)
        i_type          = ""
        e_type          = ""

        assert a is not None, "Semi-major axis 'a' should not be None at this point."
        assert p is not None, "Semi-latus rectum 'p' should not be None at this point."

        match [is_circular, is_equatorial]:
            case [True, True]:
                # orb_case = "Circular, Equatorial"
                e_type = "Circular"
                i_type = "Equatorial"
                lambda_true = angle(ref_x, r)
                if np.dot(ref_y, r) < 0:
                    lambda_true = 2*np.pi - lambda_true
                
                delta_M = Anomalies.true_to_mean(lambda_true, e_mag)
                tor = -Kepler.M_to_t(mu, abs(a), delta_M)

            case [True, _]:
                # orb_case = "Circular"
                e_type = "Circular"
                N = np.cross(ref_z, h)

                Omega = angle(N, ref_x)
                if np.dot(ref_y, N) < 0:
                    Omega = 2*np.pi - Omega

                u = angle(r, N)
                if np.dot(ref_z, r) < 0:
                    u = 2*np.pi - u

                delta_M = Anomalies.true_to_mean(u, e_mag)
                tor = -Kepler.M_to_t(mu, abs(a), delta_M)

            case [_, True]:
                # orb_case = "Equatorial"
                i_type = "Equatorial"
                omega_true = angle(ref_x, e)
                if np.dot(ref_y, e) < 0:
                    omega_true = 2*np.pi - omega_true

            case _:
                N = np.cross(ref_z, h)

                Omega = angle(N, ref_x)
                if np.dot(ref_y, N) < 0:
                    Omega = 2*np.pi - Omega

                omega = angle(e, N)
                if np.dot(ref_z, e) < 0:
                    omega = 2*np.pi - omega

        if not is_circular:
            theta = angle(r, e)
            if np.dot(r, v) < 0:
                theta = 2*np.pi - theta

            if abs(e_mag - 1) < tol:
                    delta_Mp = Anomalies.true_to_mean_parabolic(theta)
                    tor = -Barker.M_to_t(mu, p, delta_Mp)
            else:
                delta_M = Anomalies.true_to_mean(theta, e_mag)
                tor = -Kepler.M_to_t(mu, abs(a), delta_M)

            if abs(e_mag - 1) < tol:
                e_type = "Parabolic"
            elif e_mag < 1:
                e_type = "Elliptic"
            else:
                e_type = "Hyperbolic"

                
        if np.degrees(i) < 90:
            i_type = " ".join([i_type, "Prograde"]) if i_type != "" else "Prograde"
        elif np.degrees(i) == 90:
            # i_type = " ".join([i_type, "Polar"]) if i_type != "" else "Polar"
            i_type = "Polar"
        else:
            i_type = " ".join([i_type, "Retrograde"]) if i_type != "" else "Retrograde"

            # orb_case = ", ".join([orb_case, i_type])

        orb_case = ", ".join([e_type, i_type])

        elements = OrbitalElements(
            family=orb_case, 
            a=a, 
            p=p, 
            e=e_mag, 
            i=i, 
            Omega=Omega if Omega is not None else None, 
            omega=omega if omega is not None else None, 
            theta=theta if theta is not None else None,
            tor=tor if tor is not None else None, 
            omega_true=omega_true if omega_true is not None else None,
            u=u if u is not None else None,
            lambda_true=lambda_true if lambda_true is not None else None
            )

        return elements
    
    @staticmethod
    def coe_to_rv(coe: 'OrbitalElements', mu: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

        if coe.theta is not None:
            anomaly = coe.theta
        elif coe.u is not None:
            anomaly = coe.u
        elif coe.lambda_true is not None:
            anomaly = coe.lambda_true
        else:
            raise ValueError("No valid argument for anomaly")
            # print("No valid arguemnt for anomaly")
            # return


        r_mag = coe.p / (1 + coe.e*np.cos(anomaly))
        x = np.array([np.cos(anomaly), 0, 0])*r_mag
        y = np.array([0, np.sin(anomaly), 0])*r_mag



        mu_h = np.sqrt(mu / coe.p)
        x_dot = np.array([-np.sin(anomaly), 0, 0])*mu_h
        y_dot = np.array([0, coe.e + np.cos(anomaly), 0])*mu_h

        r_p = x + y
        v_p = x_dot + y_dot

        
        matrix = Transformations.Rzxz(coe.omega if coe.omega is not None else (coe.omega_true if coe.omega_true is not None else Radians(0.0)), coe.i, coe.Omega if coe.Omega is not None else Radians(0.0))
        r = matrix @ r_p
        v = matrix @ v_p


        return r, v
    
    @staticmethod
    def inertia_to_fixed(r_i: NDArray[np.float64], v_i: NDArray[np.float64], theta: Radians) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        matrix = Transformations.Rz(theta)
        r_f = matrix @ r_i
        v_f = matrix @ v_i
        return r_f, v_f
    
    @staticmethod
    def fixed_to_inertia(r_f: NDArray[np.float64], v_f: NDArray[np.float64], theta: Radians) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r_i, v_i = ReferenceFrames.inertia_to_fixed(r_f, v_f, -theta)
        return r_i, v_i

    @staticmethod
    def inertia_to_RaDec(r_i: NDArray[np.float64]) -> tuple[Radians, Radians, float]:
        alt, Ra, Dec = Transformations.cart_to_spherical(r_i)
        return Radians(Ra), Radians(Dec), alt

    @staticmethod
    def RaDec_to_inertia(Ra: Radians, Dec: Radians, alt: float = 1.0) -> NDArray[np.float64]:
        r_i = Transformations.spherical_to_cart(alt, Ra, Dec)
        return r_i
    
    @staticmethod
    def fixed_to_longlat(r_f: NDArray[np.float64]) -> tuple[Radians, Radians, float]:
        alt, long, lat = Transformations.cart_to_spherical(r_f)
        return Radians(long), Radians(lat), alt

    @staticmethod
    def longlat_to_fixed(long: Radians, lat: Radians, alt: float = 1.0) -> NDArray[np.float64]:
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