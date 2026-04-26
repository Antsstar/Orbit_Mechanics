import numpy as np
# from collections import namedtuple
from typing import NamedTuple
from utilities import Transformations

def angle(a, b):
    return np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))

# OrbitalElements = namedtuple("OrbitalElements", 
#                              ["family", "a", "e", "i", "Omega", "omega", "theta", "tor", "omega_true", "u", "lambda_true"])
class OrbitalElements(NamedTuple):
    family      : str
    a           : float
    p           : float
    e           : float
    i           : float
    Omega       : float
    omega       : float
    theta       : float
    tor         : float
    omega_true  : float
    u           : float
    lambda_true : float


class ReferenceFrames:
    """
    """

    @staticmethod
    def coe_from_rv(r, v, mu, *, ref_x = np.array([1, 0, 0]), ref_z = np.array([0, 0, 1])):
        if np.dot(ref_x, ref_z) != 0:
            print("Reference directions are not orthogonal!")
            return
        
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

        h           = None # Angular Momemntum Vector
        e           = None # Eccentricity Vector
        N           = None # Nodal Vector

        
        h = np.cross(r, v)
        h_mag = np.linalg.norm(h)

        
        # if h_mag == 0:
        if abs(h_mag) < tol:
            print("Velocity and displacement are parallel. Entity is not in orbit")
            return
        

        e = (np.cross(v, h) / mu) - (r / np.linalg.norm(r))
        e_mag = np.linalg.norm(e)

        p = h_mag**2 / mu
        # a = (h_mag**2 / mu) * (1 / (1 - e_mag**2))

        # match e_mag:
        #     case 0:
        #         orb_case = "Circular"
        #     case 1:

        i = np.arccos(np.dot(h, ref_z) / (np.linalg.norm(h) * np.linalg.norm(ref_z)))

        # if [i, e_mag] == [0, 0]:
        #     orb_case = "Circular, Equatorial"
        #     lambda_true = angle(ref_x, r)
        is_circular     = e_mag < tol
        is_equatorial   = abs(i) < tol or abs(i - np.pi) < tol
        i_type          = ""
        e_type          = ""

        match [is_circular, is_equatorial]:
            case [True, True]:
                # orb_case = "Circular, Equatorial"
                e_type = "Circular"
                i_type = "Equatorial"
                lambda_true = angle(ref_x, r)
                if np.dot(ref_y, r) < 0:
                    lambda_true = 2*np.pi - lambda_true

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

            if e_mag < 1:
                e_type = "Elliptic"
            elif e_mag == 1:
                e_type = "Parabolic"
            else:
                e_type = "Hyperbolic"
        
        if abs(e_mag - 1) < tol:
            a = np.inf
        else:
            a =  p * (1 / (1-e_mag**2))

            # orb_case = ", ".join([e_type, orb_case])
        
        # if i != 0:
        #     if np.degrees(i) < 90:
        #         i_type = "Pro-grade"
        #     elif np.degrees(i) == 90:
        #         i_type = "Polar"
        #     else:
        #         i_type = "Retro-grade"
                
        if np.degrees(i) < 90:
            i_type = " ".join([i_type, "Pro-grade"]) if i_type != "" else "Pro-grade"
        elif np.degrees(i) == 90:
            # i_type = " ".join([i_type, "Polar"]) if i_type != "" else "Polar"
            i_type = "Polar"
        else:
            i_type = " ".join([i_type, "Retro-grade"]) if i_type != "" else "Retro-grade"

            # orb_case = ", ".join([orb_case, i_type])

        orb_case = ", ".join([e_type, i_type])

        # elements = {
        #     "Class": orb_case,
        #     "Semi-major axis": a,
        #     "Eccentricity": e_mag,
        #     "Inclination": i,
        #     "RAAN": Omega,
        #     "Arg of Perigee": omega,
        #     "True anomaly": theta,
        #     "True long of periapsis": omega_true,
        #     "True arg of Lat": u,
        #     "True long": lambda_true
        # }
        elements = OrbitalElements(family=orb_case, a=a, p=p, e=e_mag, i=i, Omega=Omega, omega=omega, theta=theta, 
                                   tor=tor, omega_true=omega_true, u=u, lambda_true=lambda_true)

        return elements
    
    @staticmethod
    def rv_from_coe(coe, mu):

        if coe.theta != None:
            anomaly = coe.theta
        elif coe.u != None:
            anomaly = coe.u
        elif coe.lambda_true != None:
            anomaly = coe.lambda_true
        else:
            print("No valid arguemnt for anomaly")
            return


        r_mag = coe.p / (1 + coe.e*np.cos(anomaly))
        x = np.array([np.cos(anomaly), 0, 0])*r_mag
        y = np.array([0, np.sin(anomaly), 0])*r_mag

        # v = dr/dt, v_theta = h/r, r = h^2/mu * 1/1+ecos_theta
        # dr/dt =   -h^2/mu * (1+ecos_theta)^-2 * -esin_theta theta_dot
        # v_theta = mu / h * (1+ecos_theta)
        # v_r = dr/dt = h^2/mu * (1+ecos_theta)^-2 * esin_theta * v_theta / r
        # v_r = h^2/mu * ((mu/h^2)(r))^2 * esin_theta * v_theta / r
        # v_r = mu / h^2 * r * esin_theta * v_theta ## v_theta*r = h!
        # v_r = mu / h * esin_theta

        # r = xi + yj => v = x_doti + y_dotj
        # x_dot = dx/dt = r_dot*cos_theta - r*sin_theta*theta_dot = v_r*cos_theta - v_theta*sin_theta
        # y_dot = dy/dt = r_dot*sin_theta + r*cos_theta*theta_dot = v_r*sin_theta + v_theta*cos_theta
        # x_dot = mu/h(esin_theta*cos_theta - sin_theta - ecos_theta*sin_theta) = mu/h(-sin_theta)
        # y_dot = mu/h(esin_theta^2 + cos_theta + ecos_theta^2) = mu/h(e+cos_theta)

        mu_h = np.sqrt(mu / coe.p)
        x_dot = np.array([-np.sin(anomaly), 0, 0])*mu_h
        y_dot = np.array([0, coe.e + np.cos(anomaly), 0])*mu_h

        r_p = x + y
        v_p = x_dot + y_dot

        # r = Transformations.Rzxz(-coe.Omega if coe.Omega != None else 0, -coe.i, -coe.omega if coe.omega != None else 0) @ r_p
        # v = Transformations.Rzxz(-coe.Omega if coe.Omega != None else 0, -coe.i, -coe.omega if coe.omega != None else 0) @ v_p
        matrix = Transformations.Rzxz(coe.omega if coe.omega != None else 0, coe.i, coe.Omega if coe.Omega != None else 0)
        r = matrix @ r_p
        v = matrix @ v_p


        return r, v
        


if __name__ == "__main__":
    MU_Sun = 1.32712440042 * 10**11
    r = np.array([-145510750, 39268690, 10500])
    v = np.array([-6.995, -29.215, -0.00025])
    elements = ReferenceFrames.coe_from_rv(r, v, mu=MU_Sun)
    print(elements)
    r_new, v_new = ReferenceFrames.rv_from_coe(elements, MU_Sun)

    # 4. Check results (using np.allclose to handle tiny floating point errors)
    print("Position Match:", np.allclose(r, r_new))
    print("Velocity Match:", np.allclose(v, v_new))
    print(r, v, np.linalg.norm(r), np.linalg.norm(v))
    print(r_new, v_new, np.linalg.norm(r_new), np.linalg.norm(v_new))