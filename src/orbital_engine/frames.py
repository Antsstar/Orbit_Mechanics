# Classical Orbital Elements definitions and Body-fixed <-> Inertial Reference frame transformations.

from __future__ import annotations
import enum
import numpy as np
from numpy.typing import NDArray
from .custom_types import Radians, Kilometers, Seconds, ArrayFloat, Numeric
from typing import Optional, cast
from .utilities import Transformations, Anomalies, Kepler, Barker
from .exceptions import SingularityError

# ==========================================================================================================================================================
# Helper functions. Vector math and indexing definitions
# ==========================================================================================================================================================

def angle(a: ArrayFloat, b: ArrayFloat, n: Optional[ArrayFloat] = None) -> Radians:
    """Returns the rotation angle from Vectors a to b in radians about a defined +-ve normal direction."""
    # a . b = |a||b|cos(<ab)
    # a x b = /n\ |a||b|sin(<ab)
    # a x b / a.b = /n\ tan(<ab)
    # /n\ . a x b / a.b = tan(<ab)

    if n is None:
        ab_mag: Numeric = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        theta: ArrayFloat = np.where( ab_mag > 1e-9, np.arccos( np.sum(a * b, axis=-1) / (ab_mag) ), 0.0 )
    else:
        theta = np.arctan2(np.sum(n * np.cross(a, b), axis=-1), np.sum(a * b, axis=-1)) % (2.0 * np.pi)

    if a.ndim == 1 and b.ndim == 1:
        return theta.item()
    return theta


class COEIndex(enum.IntEnum):
    """
    Helper class, column indices for Classical Orbital Elements (COE) arrays.
    *Special Cases such as **True longitude of perapsis**, **True argument of latitude**, and **True longitude** reduce to fit by zeroing respective entries*
    """
    P           = 0 # Semi-Latus rectum (km) p = h^2 / mu
    E           = 1 # Eccentricity
    I           = 2 # Inclination (rad)
    RAAN        = 3 # Right Ascension of the Ascending Node (Omega, rad)
    ARG_PE      = 4 # Argument of Periapsis (omega, rad)
    THETA       = 5 # True Anomaly (rad)

# ==========================================================================================================================================================
# ReferenceFrames Static Class: Orbital focused, Cartesian Vectors <-> Classical Orbital Elements, Reference frame transition body-fixed or inertial frame.
# ==========================================================================================================================================================

class ReferenceFrames:
    """
    State-Space Transformation Toolbox for orbital mechanics.
    - Bidirectional conversions between Inertial Cartesian (r, v), Classical Orbital Elements (COE), and Body-Fixed Lat/Long coordinates.
    """

    @staticmethod
    def rv_to_coe(r: ArrayFloat, v: ArrayFloat, mu: Numeric, *, ref_x: ArrayFloat = np.array([1, 0, 0]),
                  ref_z: ArrayFloat = np.array([0, 0, 1]), out_coe: Optional[ArrayFloat] = None) -> tuple[ArrayFloat, NDArray[np.bool_]]:
        """
        Inertial Coordinates (r, v) to Classical Orbital Elements (COE).
        *- Custom reference directions can be provided for ref_x and ref_z.*
        """

        if np.any(np.abs(np.dot(ref_x, ref_z)) > 1e-9):
            raise ValueError("Reference directions are not orthogonal!")
        
        # ref_y = np.cross(ref_z, ref_x)
        tol = 1e-12
        _mu = np.asarray(mu, dtype=np.float64)
        _r = np.atleast_2d(r)
        _v = np.atleast_2d(v)

        # orb_case    = ""   # Classification of orbit
        a           = None # Semi-major axis > Semi-latus rectum "p" = h^2 / mu
        p           = None # Semi-latus Rectum h^2 / mu
        e_mag       = None # Eccentricity
        i           = None # Inclination
        Omega       = None # Right Ascension of the Ascending Node (RAAN)
        omega       = None # Argument of Perigee
        theta       = None # True anomaly
        # tor         = None # Time of periapsis passage
        # omega_true  = None # Omega + omega (x_ref.e) Non-Circular Equatorial, True longitude of periapsis
        # u           = None # omega + theta (N.r) Circular Inclined, True argument of latitude
        # lambda_true = None # Omega + omega + theta (x_ref . r) Circular Equatorial, True longitude

        
        h = np.cross(_r, _v)
        h_mag = np.linalg.norm(h, axis=-1)

        if np.any(h_mag < tol):
            # raise SingularityError("Velocity and displacement are parallel. Entity is not in orbit")
            pass
        
        valid = np.abs(h_mag) > 1e-9

        if not np.any(valid):
            return np.zeros(r.shape, dtype=np.float64), valid

        if out_coe is not None:
            coe_states = np.atleast_2d(out_coe)
        else:
            coe_states = np.zeros((_r.shape[0], 6), dtype=np.float64)

        
        # Filter valid orbit slices (these are the entries we will update)
        r_v = _r[valid, ...]
        v_v = _v[valid, ...]
        h_v = h[valid, ...]
        h_mag_v = h_mag[valid]
        mu_v = _mu[valid] if _mu.ndim > 0 else _mu

        # Eccenticity Vector and magnitude
        e = (np.cross(v_v, h_v) / mu_v[..., np.newaxis]) - (r_v / np.linalg.norm(r_v, axis=-1)[..., np.newaxis])
        e_mag = np.linalg.norm(e, axis=-1)

        # Semi-Latus Rectum and Inclination
        p = h_mag_v**2 / mu_v
        i = np.arccos(np.clip(np.sum(h_v * ref_z, axis=-1) / (h_mag_v), -1.0, 1.0))


        # a_mask = (np.abs(e_mag - 1.0) < tol)
        # a = np.empty_like(p)
        # a[a_mask] = np.inf
        # a[~a_mask] = p[~a_mask] / (1.0 - e_mag[~a_mask]**2) # Not sure if I need a...

        # Orbit Classifications
        is_circular = (e_mag < tol)
        is_equatorial = (np.abs(i) < tol) | (np.abs(i - np.pi) < tol)


        mask_NC_NEqu = ~is_circular & ~is_equatorial    # Standard Orbit
        mask_NC_Equ = ~is_circular & is_equatorial      # Equatorial (N is undefined) 
        mask_C_NEqu = is_circular & ~is_equatorial      # Circular (e is undefined)
        mask_C_Equ = is_circular & is_equatorial        # Circular (N and e are undefined)

        # coe_states = np.zeros((_r.shape[0], 6), dtype=np.float64)
        coe_states[valid, COEIndex.P] = p
        coe_states[valid, COEIndex.E] = e_mag
        coe_states[valid, COEIndex.I] = i

        Omega = np.zeros_like(i)
        omega = np.zeros_like(i)
        theta = np.zeros_like(i)

        h_hat = h_v / h_mag_v[..., np.newaxis]
        N = np.cross(ref_z, h_v)

        mask_NE = ~is_equatorial
        mask_NC = ~is_circular

        # print("valid: ", valid)
        # print("mask_NC_NEqu: ", mask_NC_NEqu.shape)
        # print("mask_NC: ", mask_NC.shape)

        # 1. RAAN (Omega)
        if np.any(mask_NE):
            Omega[mask_NE] = angle(ref_x, N[mask_NE], ref_z) # Regular Omega

        # 2. Argument of Periapsis (omega)
        if np.any(mask_NC_Equ):
            omega[mask_NC_Equ] = angle(ref_x, e[mask_NC_Equ], h_hat[mask_NC_Equ]) # omega absorbs Omega
        if np.any(mask_NC_NEqu):
            omega[mask_NC_NEqu] = angle(N[mask_NC_NEqu], e[mask_NC_NEqu], h_hat[mask_NC_NEqu]) # Regular omega

        # 3. True Anomaly (theta)
        if np.any(mask_NC):
            theta[mask_NC] = angle(e[mask_NC], r_v[mask_NC], h_hat[mask_NC]) # Regular theta
        if np.any(mask_C_NEqu):
            theta[mask_C_NEqu] = angle(N[mask_C_NEqu], r_v[mask_C_NEqu], h_hat[mask_C_NEqu]) # theta absorbs omega
        if np.any(mask_C_Equ):
            theta[mask_C_Equ] = angle(ref_x, r_v[mask_C_Equ], h_hat[mask_C_Equ]) # theta absorbs Omega and omega

        
        coe_states[valid, COEIndex.RAAN] = Omega
        coe_states[valid, COEIndex.ARG_PE] = omega
        coe_states[valid, COEIndex.THETA] = theta

        if r.ndim == 1:
            return (cast(ArrayFloat, coe_states[0]), valid)
        return coe_states, valid
    
    @staticmethod
    def coe_to_rv(coe: ArrayFloat, mu: Numeric, *, out_rv: Optional[ArrayFloat] = None) -> tuple[ArrayFloat, ArrayFloat, NDArray[np.bool_]]: #Now return a success array
        """
        Classical Orbital Elements (COE) to Inertial Coordinates (r, v).
        - In-place adjustments possible by passing *out_rv*
        """

        _coe = np.atleast_2d(coe)
        _mu = np.asarray(mu, dtype=np.float64)

        p = _coe[..., 0]
        e = _coe[..., 1]
        i = _coe[..., 2]
        Omega = _coe[..., 3]
        omega = _coe[..., 4]
        anomaly = _coe[..., 5]

        valid = (p > 1e-12) & ~np.isnan(anomaly)
        Vect = np.zeros_like(_coe)

        if not np.any(valid):
            return Vect[..., :3], Vect[..., 3:], valid

        # Allocate or reuse memory.
        if out_rv is not None:
            Vect = np.atleast_2d(out_rv)
        else:
            Vect = np.zeros_like(_coe)
        
        _mu_v = _mu[valid] if _mu.ndim > 0 else _mu
        p_v = p[valid]
        e_v = e[valid]
        i_v = i[valid]
        Omega_v = Omega[valid]
        omega_v = omega[valid]
        anomaly_v = anomaly[valid]


        cos_t, sin_t = np.cos(anomaly_v), np.sin(anomaly_v)
        r_mag = p_v / (1.0 + e_v * cos_t)

        r_x = r_mag * cos_t
        r_y = r_mag * sin_t
        rv_z = np.zeros_like(r_mag)

        mu_h = np.sqrt(_mu_v / p_v)
        v_x = -mu_h * sin_t
        v_y = mu_h * (e_v + cos_t)

        # r_p = x + y
        # v_p = x_dot + y_dot
        r_p = np.stack((r_x, r_y, rv_z), axis=-1)
        v_p = np.stack((v_x, v_y, rv_z), axis=-1)

        
        # matrix = Transformations.Rzxz(coe.omega if coe.omega is not None else (coe.omega_true if coe.omega_true is not None else Radians(0.0)), coe.i, coe.Omega if coe.Omega is not None else Radians(0.0))
        # r = matrix @ r_p
        # v = matrix @ v_p
        matrix = Transformations.Rzxz(omega_v, i_v, Omega_v)
        r = (matrix @ r_p[..., np.newaxis])[..., 0]
        v = (matrix @ v_p[..., np.newaxis])[..., 0]

        if coe.ndim == 1:
            return r[0], v[0], valid

        Vect[valid, :3] = r
        Vect[valid, 3:] = v
        return Vect[..., :3], Vect[..., 3:], valid
    
    @staticmethod
    def inertia_to_fixed(r_i: ArrayFloat, v_i: ArrayFloat, theta: Radians) -> tuple[ArrayFloat, ArrayFloat]:
        """Rotates Cartesian Vectors from an Inertial Frame to a Body-Fixed rotating frame."""
        matrix = Transformations.Rz(theta)
        r_f = matrix @ r_i
        v_f = matrix @ v_i
        return r_f, v_f
    
    @staticmethod
    def fixed_to_inertia(r_f: ArrayFloat, v_f: ArrayFloat, theta: Radians) -> tuple[ArrayFloat, ArrayFloat]:
        """Rotates from a Body-Fixed rotating frame to an Inertial Frame."""
        r_i, v_i = ReferenceFrames.inertia_to_fixed(r_f, v_f, -theta)
        return r_i, v_i

    @staticmethod
    def inertia_to_RaDec(r_i: ArrayFloat) -> ArrayFloat:
        """
        Converts Inertial Cartesian position vectors [x, y, z] to Celestial Spherical [r, Ra, Dec].
        - *r: Radial distance from origin (match units of x,y,z)*
        - *Ra: Right Ascension (Azimuth in celestial equator) [0, 2pi] (rad)*
        - *Dec: Declination (Elevation from celestial equator) [-pi/2. pi/2] (rad)*
        """
        return Transformations.cart_to_sphe(r_i)

    @staticmethod
    def RaDec_to_inertia(V_radec: ArrayFloat) -> ArrayFloat:
        """
        Converts Celestial Spherical [r, Ra, Dec] to Inertial Cartesian position vectors [x, y, z].
        - *V_radec: [r, Ra, Dec], (Distance, rad, rad)*
        """
        return Transformations.sphe_to_cart(V_radec)
    
    @staticmethod
    def fixed_to_longlat(r_f: ArrayFloat) -> ArrayFloat:
        """
        Converts Body-Fixed Cartesian position vectors [x, y, z] to Geographical Spherical [r, long, lat].
        - *r: Radial distance from origin (match units of x,y,z)*
        - *long: Geocentric Longitude wrapped strictly to [-pi, +pi] (rad)*
        - *lat: Geocentric Latitude (Elevation from equator) [-pi/2. pi/2] (rad)*
        """
        sph = Transformations.cart_to_sphe(r_f)
        sph[..., 1] += np.pi
        sph[..., 1] %= (2.0 * np.pi)
        sph[..., 1] -= np.pi
        return sph

    @staticmethod
    def longlat_to_fixed(V_longlat: ArrayFloat) -> ArrayFloat:
        """
        Converts Geographical Spherical [r, long, lat] to Inertial Cartesian position vectors [x, y, z].
        - *V_longlat: [r, long, lat], (Distance, rad, rad)*
        """
        return Transformations.sphe_to_cart(V_longlat)

if __name__ == "__main__":
    MU_Sun = 1.32712440042 * 10**11
    r = np.array([-145510750, 39268690, 10500])
    v = np.array([-6.995, -29.215, -0.00025])
    elements, _ = ReferenceFrames.rv_to_coe(r, v, mu=MU_Sun)
    print(elements)
    r_new, v_new, _ = ReferenceFrames.coe_to_rv(elements, MU_Sun)

    # 4. Check results (using np.allclose to handle tiny floating point errors)
    print("Position Match:", np.allclose(r, r_new))
    print("Velocity Match:", np.allclose(v, v_new))
    print(r, v, np.linalg.norm(r), np.linalg.norm(v))
    print(r_new, v_new, np.linalg.norm(r_new), np.linalg.norm(v_new))