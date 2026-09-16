# Classical Orbital Elements definitions and Body-fixed <-> Inertial Reference frame transformations.

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .custom_types import COEIndex, Radians, Kilometers, Seconds, ArrayFloat, ArrayKilometers, ArrayKmPerSec, Numeric
from typing import Optional, cast
from .utilities import Transformations, Anomalies, Kepler, Barker
from .exceptions import SingularityError

# Degeneracy threshold for the RSW frame, applied to sin(alpha) = |r x v| / (|r| |v|), alpha the angle between r and v.
#
# Rounding leaves each component of a computed r x v in error by up to ~eps |r||v|, so the direction of W carries an
# angular error bounded by roughly eps / sin(alpha): 2e-6 rad at this threshold (measured 7e-8), degrading to pure
# noise by sin(alpha) ~ 1e-15. No bound orbit comes near it - an ellipse has sin(alpha) = cos(flight-path angle)
# >= sqrt(1 - e^2), still 1.4e-3 at e = 0.999999 - so in practice it flags only genuinely radial trajectories.
#
# The test is *relative*, so a state is classified identically in km or AU. rv_to_coe's absolute |h| > 1e-9 km^2/s is
# not: at 1 AU and 30 km/s, rounding noise in |h| is bounded by eps |r||v| ~ 1e-6 km^2/s, so a heliocentric state that
# is rectilinear up to rounding can pass it.
RSW_RECTILINEAR_TOL: float = 1e-10

# ==========================================================================================================================================================
# Helper functions. Vector math
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

    # ==================================================================================================================================================
    # Satellite-based RSW frame: Radial, along-track (S), cross-track (W)
    # ==================================================================================================================================================

    @staticmethod
    def RSW_basis(r: ArrayKilometers, v: ArrayKmPerSec, *, tol: float = RSW_RECTILINEAR_TOL,
                  out_basis: Optional[ArrayFloat] = None) -> tuple[ArrayFloat, NDArray[np.bool_]]:
        """
        Orthonormal RSW basis of a reference state, returned as the Cartesian -> RSW rotation matrix.

        Definition - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Sec. 3.3 (satellite-based
        coordinate systems, "RSW"):

            R = r / |r|                  radial, outward from the central body
            W = (r x v) / |r x v|        cross-track, along the orbital angular momentum
            S = W x R                    along-track, in the orbital plane, toward the direction of motion

        Right-handed by construction: R x S = R x (W x R) = W (R.R) - R (R.W) = W, so det[R; S; W] = +1.

        **S is not the velocity direction.** It is perpendicular to R, so it coincides with v only where the
        flight-path angle is zero - circular orbits, periapsis, apoapsis. A "prograde" S-axis burn is a horizontal
        burn. In this frame v = [v_R, v_S, 0] with v_R = (mu/h) e sin(theta) and v_S = (mu/h)(1 + e cos(theta)).

        Aliases: RTN, Gaussian frame. **Not** interchangeable with LVLH, whose common convention is z = -R (nadir)
        and y = -W.

        Parameters
        ----------
        r, v : (3,) or (N, 3)
            Reference state(s) in any Cartesian frame; the basis is expressed in that same frame. Broadcast
            against each other.
        tol : float
            Rectilinearity threshold on |r x v| / (|r||v|). See `RSW_RECTILINEAR_TOL` for the derivation.
        out_basis : (N, 3, 3) or (3, 3), optional
            Written in place and returned when given.

        Returns
        -------
        basis : (N, 3, 3), or (3, 3) for a single (3,) reference state
            Rows are R, S, W in the input frame's components, so `x_rsw = basis @ x` and, the matrix being
            orthogonal, `x = basis.T @ x_rsw`.
        valid : (N,) bool
            False where the frame is undefined: r parallel to v, zero r or v, or non-finite input. Those rows of
            `basis` are exactly zero, never NaN, so a vector transformed through them comes back as zero instead
            of poisoning an accumulator.

        Numerical notes
        ---------------
        W is re-orthogonalised against R by one Gram-Schmidt step. The exact angular momentum is perpendicular to
        r, but the *computed* r x v is not: rounding leaves R.W growing like eps / sin(alpha). That is ~1e-16 for
        any real orbit, where the step changes nothing measurable, but was measured at 7e-12 for sin(alpha) = 1e-6
        and 1.4e-8 just inside `tol` - enough that the matrix stops being a rotation to working precision. With the
        step, orthonormality holds to ~3 eps across that whole range. Projecting out the R component moves W toward
        the exact answer and leaves R exactly r / |r|.
        """
        _r = np.atleast_2d(np.asarray(r, dtype=np.float64))
        _v = np.atleast_2d(np.asarray(v, dtype=np.float64))
        if _r.ndim != 2 or _v.ndim != 2 or _r.shape[-1] != 3 or _v.shape[-1] != 3:
            raise ValueError(f"RSW frame needs (3,) or (N, 3) reference states; got r {np.shape(r)}, v {np.shape(v)}")

        n = np.broadcast_shapes(_r.shape, _v.shape)[0]
        single = np.ndim(r) == 1 and np.ndim(v) == 1

        if out_basis is None:
            basis: ArrayFloat = np.empty((n, 3, 3), dtype=np.float64)
        else:
            basis = out_basis[np.newaxis] if out_basis.ndim == 2 else out_basis
            if basis.shape != (n, 3, 3):
                raise ValueError(f"out_basis must have shape ({n}, 3, 3); got {out_basis.shape}")

        h = np.cross(_r, _v)
        r_mag = np.linalg.norm(_r, axis=-1)
        v_mag = np.linalg.norm(_v, axis=-1)
        h_mag = np.linalg.norm(h, axis=-1)

        # Phrased as "keep what is demonstrably defined": every comparison against NaN is False, so a non-finite
        # state lands in the invalid set instead of slipping through. See docs/engineering-log.md, the NaN solver.
        valid: NDArray[np.bool_] = h_mag > tol * r_mag * v_mag

        # Basic slices: these are views, so every write below lands directly in `basis`.
        R = basis[:, 0, :]
        S = basis[:, 1, :]
        W = basis[:, 2, :]

        # Invalid rows divide by 1 rather than by a possibly-zero norm and are zeroed at the end, so no 0/0 is ever
        # evaluated. On a valid row |h| > 0, which forces |r| > 0 as well.
        np.divide(_r, np.where(valid, r_mag, 1.0)[:, np.newaxis], out=R)
        np.divide(h, np.where(valid, h_mag, 1.0)[:, np.newaxis], out=W)

        W -= np.sum(W * R, axis=-1, keepdims=True) * R
        np.divide(W, np.where(valid, np.linalg.norm(W, axis=-1), 1.0)[:, np.newaxis], out=W)

        S[...] = np.cross(W, R)

        basis[~valid] = 0.0

        if single:
            return basis[0], valid
        return basis, valid

    @staticmethod
    def cart_to_RSW(r: ArrayKilometers, v: ArrayKmPerSec, vec: ArrayFloat, *, tol: float = RSW_RECTILINEAR_TOL,
                    out: Optional[ArrayFloat] = None) -> tuple[ArrayFloat, NDArray[np.bool_]]:
        """
        Cartesian vector(s) -> RSW components [x_R, x_S, x_W] in the frame of reference state (r, v).

            x_R = x . R,    x_S = x . S,    x_W = x . W

        Projection onto the unit vectors defined in `RSW_basis` (Vallado 4th ed., Sec. 3.3). The use case is
        perturbation analysis: Gauss's variational equations take the perturbing acceleration as (a_R, a_S, a_W).

        Parameters
        ----------
        r, v : (3,) or (N, 3)
            Reference state(s) defining the frame.
        vec : (3,) or (N, 3)
            Vector(s) to express in RSW, in the same Cartesian frame as r and v. Any units - position, velocity,
            acceleration - and returned in those units. A single (3,) applies to every reference state; an (M, 3)
            against a single (3,) reference state expresses M vectors in one frame.
        out : optional
            Written in place and returned when given. Must already have the broadcast output shape. May be `vec`
            itself, rotating in place.

        Returns
        -------
        x_rsw : broadcast of the leading shapes, (..., 3)
            Zero on rows whose reference state is invalid (for finite `vec`).
        valid : (N,) bool
            Frame-validity mask of the reference states; see `RSW_basis`.
        """
        basis, valid = ReferenceFrames.RSW_basis(r, v, tol=tol)
        _vec = np.asarray(vec, dtype=np.float64)

        # Row i of the basis is unit vector i, so component i is (row i) . x  ->  x_rsw = basis @ x.
        if out is None:
            x_rsw: ArrayFloat = np.einsum("...ij,...j->...i", basis, _vec)
            return x_rsw, valid
        np.einsum("...ij,...j->...i", basis, _vec, out=out)
        return out, valid

    @staticmethod
    def RSW_to_cart(r: ArrayKilometers, v: ArrayKmPerSec, vec_rsw: ArrayFloat, *, tol: float = RSW_RECTILINEAR_TOL,
                    out: Optional[ArrayFloat] = None) -> tuple[ArrayFloat, NDArray[np.bool_]]:
        """
        RSW components [x_R, x_S, x_W] -> Cartesian vector(s), in the frame of reference state (r, v).

            x = x_R R + x_S S + x_W W

        The inverse of `cart_to_RSW`, and since the basis is orthogonal, its transpose (Vallado 4th ed., Sec. 3.3).
        The use case is thrust: a direction law stated in RSW ("burn prograde" is +S) must be rotated to Cartesian
        before it can be accumulated with other accelerations.

        Parameters
        ----------
        r, v : (3,) or (N, 3)
            Reference state(s) defining the frame.
        vec_rsw : (3,) or (N, 3)
            RSW components. Any units, returned in those units. A single (3,) - e.g. [0, 1, 0] for prograde -
            applies to every reference state.
        out : optional
            Written in place and returned when given. Must already have the broadcast output shape. May be
            `vec_rsw` itself, rotating in place.

        Returns
        -------
        x : broadcast of the leading shapes, (..., 3)
            Zero on rows whose reference state is invalid (for finite `vec_rsw`) - no thrust is applied through an
            undefined frame, and the mask says so.
        valid : (N,) bool
            Frame-validity mask of the reference states; see `RSW_basis`.
        """
        basis, valid = ReferenceFrames.RSW_basis(r, v, tol=tol)
        _vec = np.asarray(vec_rsw, dtype=np.float64)

        # Cartesian component i sums column i of the basis: x_i = sum_j basis[j, i] x_rsw[j]  ->  x = basis.T @ x_rsw.
        if out is None:
            x: ArrayFloat = np.einsum("...ji,...j->...i", basis, _vec)
            return x, valid
        np.einsum("...ji,...j->...i", basis, _vec, out=out)
        return out, valid

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