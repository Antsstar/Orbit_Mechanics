# Defining functions and tools for a orb sandbox.

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .custom_types import Radians, Kilometers, Seconds, ArrayFloat, ScalarFloat, Numeric, GravitationalParameter
from typing import Union, cast, overload
from .exceptions import ConvergenceError


# ==========================================================================================================================================================
# Transformations Static Class: Rotations, Coordinate Transformations
# ==========================================================================================================================================================

class Transformations:
    """ 
    Rotation Matrix & Spherical Coordinates Toolbox. 
    - Takes radians as inputs for euler angles about classic X, Y or Z definitions.
    - Also includes cartesian to spherical and vice versa conversions.
    - All angles must be provided in Radians.
    """

    @staticmethod
    def Rx(angle: Radians) -> ArrayFloat:
        """Returns 3x3 Euler Matrix for a rotation about the X-axis."""
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
    def Ry(angle: Radians) -> ArrayFloat:
        """Returns 3x3 Euler Matrix for a rotation about the Y-axis."""
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
    def Rz(angle: Radians) -> ArrayFloat:
        """Returns 3x3 Euler Matrix for a rotation about the Z-axis."""
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
    def Rxyz(alpha: Radians, beta: Radians, gamma: Radians) -> ArrayFloat:
        """Returns resultant 3x3 Euler Matrix after a rotation about X, then Y, and finally Z in that specific order."""
        return Transformations.Rz(gamma) @ Transformations.Ry(beta) @ Transformations.Rx(alpha)
    
    @staticmethod
    def Rzyx(alpha: Radians, beta: Radians, gamma: Radians) -> ArrayFloat:
        """Returns resultant 3x3 Euler Matrix after a rotation about Z, then Y, and finally X in that specific order."""
        return Transformations.Rx(gamma) @ Transformations.Ry(beta) @ Transformations.Rz(alpha)
    
    @staticmethod
    def Rzxz(alpha: Radians, beta: Radians, gamma: Radians) -> ArrayFloat:
        """Returns resultant 3x3 Euler Matrix after a rotation about Z, then Y, and finally Z again."""
        return Transformations.Rz(gamma) @ Transformations.Rx(beta) @ Transformations.Rz(alpha)
    
    @staticmethod #Up to here!
    def cart_to_sphe(V_cart: ArrayFloat) -> ArrayFloat:
        """
        Converts a Cartesian Coordinate defined Vector: V_cart = [x, y, z], 
        to a Spherical Coordinate defined Vector: V_sphe = [r, azimuth, elevation].
        *Angles in Radians
        *azimuth: Azimuthal angle in xy-plane [0, 2pi]
        *elevation: Elevation angle from xy-plane [-pi/2, pi/2]
        **Note elevation, not inclination*.
        """

        r = np.linalg.norm(V_cart, axis=-1) # ... Unpacks to fit ndims - any actualy indexed values. i.e. [..., 1] for a (3,) unpacks to [1]
        azimuth = np.arctan2(V_cart[..., 1], V_cart[..., 0]) # for (2, 3) it unpacks to [:, 1], (1, 2, 3) -> [:, :, 1].
        elevation = np.where(r > 1e-15, np.arcsin(np.clip(V_cart[..., 2] / r, -1.0, 1.0)), 0.0) # Avoid div by zero errors
        return np.stack([r, azimuth, elevation], axis=-1)
    
    @staticmethod
    def sphe_to_cart(V_sphe: ArrayFloat) -> ArrayFloat:
        """
        Converts a Spherical Coordinate defined Vector: V_sphe = [r, azimuth, elevation],
        to a Cartesian Coordinate defined Vector: V_cart = [x, y, z].
        *Angles in Radians
        *azimuth: Azimuthal angle in xy-plane [0, 2pi]
        *elevation: Elevation angle from xy-plane [-pi/2, pi/2]
        **Note elevation, not inclination*.
        """

        _r = V_sphe[..., 0]
        _azimuth = V_sphe[..., 1]
        _elevation = V_sphe[..., 2]

        c_ele = np.cos(_elevation)
        x = _r * c_ele * np.cos(_azimuth)
        y = _r * c_ele * np.sin(_azimuth)
        z = _r * np.sin(_elevation)

        return np.stack((x, y, z), axis=-1)


# ==========================================================================================================================================================
# Anomalies Static Class: Orbital Anomaly transformations between True, Eccentric/Hyperbolic, and Mean Anomalies. Includes conversions for Parabolic cases
# ==========================================================================================================================================================

class Anomalies:
    """
    Anomaly Toolbox - Conversions between true, eccentric and mean anomalies for Elliptic, Parabolic and Hyperbolic orbits.
        All angles are in radians, except for parbolic mean anomaly which is dimensionless. 
        Cardino solution used to solve the cubic from parabolic mean to true anomaly.
    """



    @staticmethod
    def true_to_eccentric(theta: Radians, e: Numeric) -> Radians:
        """True Anomaly (theta) to Eccentric/Hyperbolic Anomaly (E or H)."""

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
            return E.item()
        
        return E

    

    @staticmethod
    def eccentric_to_true(E: Radians, e: Numeric) -> Radians:
        """Eccentric/Hyperbolic Anomaly (E or H) to True Anomaly (theta)."""

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

            y = np.sqrt(1.0 + e_ell) * np.sin(E_ell / 2.0)
            x = np.sqrt(1.0 - e_ell) * np.cos(E_ell / 2.0)

            theta[ell_mask] = 2.0 * np.arctan2(y, x)

        # 6. Hyperbolic cases
        if np.any(hyp_mask):
            e_hyp = _e[hyp_mask]
            E_hyp = _E[hyp_mask]

            tanhE_2 = np.sqrt( (e_hyp + 1.0) / (e_hyp - 1.0) ) * np.tanh(E_hyp / 2.0)
            theta[hyp_mask] = 2.0 * np.arctan(tanhE_2)

        # 7. Scalar Guard
        if _E.ndim == 0:
            return theta.item()
        
        return theta



    @staticmethod
    def eccentric_to_mean(E: Radians, e: Numeric) -> Radians:
        """Eccentric/Hyperbolic Anomaly (E or H) to Mean Anomaly (M)"""

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
            # e_ell = _e[ell_mask]
            E_ell = _E[ell_mask]

            M[ell_mask] = E_ell - _e[ell_mask] * np.sin(E_ell)

        # 6. Hyperbolic cases
        if np.any(hyp_mask):
            # e_hyp = _e[hyp_mask]
            E_hyp = _E[hyp_mask]

            M[hyp_mask] = _e[hyp_mask] * np.sinh(E_hyp) - E_hyp

        # 7. Scalar Guard
        if _E.ndim == 0:
            return M.item()
        
        return M

    @staticmethod
    def _iterate_kepler(
        M: NDArray[np.float64],
        e: NDArray[np.float64],
        E: NDArray[np.float64],
        *,
        hyperbolic: bool,
        tol: float,
        solver: str,
        max_ite: int,
    ) -> NDArray[np.float64]:
        """
        Drive a seeded guess `E` to convergence against Kepler's equation. Mutates and returns `E`.

        All three arrays are 1-D, contiguous and the same length, and every element is on the same
        branch - the caller has already split elliptic from hyperbolic. That split is what keeps
        this loop free of per-iteration masking.

        `active` is an array of *integer indices* rather than a boolean mask, so the loop's
        termination test is `active.size`, a plain Python attribute lookup. The previous formulation
        called `np.any()` three times per iteration, two of which were loop-invariant.
        """
        active = np.arange(E.size, dtype=np.intp)

        for _ in range(max_ite):
            if active.size == 0:
                return E

            E_a = E[active]
            e_a = e[active]
            M_a = M[active]

            if hyperbolic:
                if solver == "N-R":
                    # f = e sinh H - H - M ;  f' = e cosh H - 1. Sign folded into the denominator.
                    delta = (e_a * np.sinh(E_a) - E_a - M_a) / (1.0 - e_a * np.cosh(E_a))
                else:
                    delta = e_a * np.sinh(E_a) - M_a - E_a
            else:
                if solver == "N-R":
                    # f = E - e sin E - M ;  f' = 1 - e cos E. Sign folded into the denominator.
                    delta = (E_a - e_a * np.sin(E_a) - M_a) / (e_a * np.cos(E_a) - 1.0)
                else:
                    delta = M_a + e_a * np.sin(E_a) - E_a

            E[active] = E_a + delta

            # Retain anything not *demonstrably* converged, rather than dropping anything that
            # tests as diverged. The distinction is NaN: a diverging hyperbolic iterate overflows
            # sinh to inf and then produces inf - inf = nan, and `abs(nan) > tol` is False. Written
            # that way round, the element would be dropped from the active set and the solver would
            # return NaN reporting success. `~(abs(delta) <= tol)` keeps NaN active instead, so it
            # survives to the max_ite check below and raises. Same cost, opposite failure mode.
            active = active[~(np.abs(delta) <= tol)]

        if active.size > 0:
            branch = "hyperbolic" if hyperbolic else "elliptic"
            raise ConvergenceError(
                f"Solver '{solver}' failed to converge for {active.size} {branch} entries after "
                f"{max_ite} iterations. Worst offenders: M={M[active][:5]}, e={e[active][:5]}"
            )
        return E

    @staticmethod
    def mean_to_eccentric(M: Radians, e: Numeric, *, tol: float = 1e-5, solver: str = "N-R", max_ite: int = 1000) -> Radians:
        """
        Mean Anomaly (M) to Eccentric/Hyperbolic Anomaly (E or H).

        Elliptic and hyperbolic elements are separated once, up front, and each branch is then
        solved over a contiguous subarray. The previous implementation carried both branches
        through a single loop and re-derived the split on every iteration by indexing a
        global-length mask with the shrinking active set.

        That double indexing was also a latent correctness bug in the `"S.S"` branch, which indexed
        `delta` (active-length) with `mask_ell` (global-length). It went unnoticed because nothing
        called that solver; it is fixed here and now covered by tests.

        **Caveat on `"S.S"` for hyperbolic orbits.** Successive substitution on the rearrangement
        H <- e sinh H - M has derivative e cosh H > 1 everywhere, so it is formally divergent and
        will raise `ConvergenceError` for all but trivial inputs. The rearrangement is preserved
        rather than silently replaced; use `"N-R"` for hyperbolic work.

        Seed bands follow Vallado, *Fundamentals of Astrodynamics and Applications* (4th ed.),
        Algorithm 2: E = M below e = 0.55, the Barker-like cube-root guess to e = 0.95, and pi
        above it, where seeding at M would otherwise step Newton-Raphson out of the basin.
        """
        if solver not in ("N-R", "S.S"):
            raise ValueError(
                f"Solver '{solver}' not recognised. Valid options are 'N-R' for Newton-Raphson "
                f"and 'S.S' for Successive Substitution."
            )

        _M = np.asarray(M, dtype=np.float64)
        _e = np.asarray(e, dtype=np.float64)

        # Strict guarding. Parabolic orbits have no eccentric anomaly at all, so this is a domain
        # error rather than something to fall back from.
        if np.any(_e < 0.0):
            raise ValueError(f"Eccentricity cannot be negative. Recieved e = {_e}")
        if np.any(np.abs(_e - 1.0) < 1e-8):
            raise ValueError(f"Parabolic orbits (e=1) do not have eccentric anomalies.")

        out_shape = _M.shape
        M_flat = _M.reshape(-1)
        e_flat = np.broadcast_to(_e, out_shape).reshape(-1)

        hyp = e_flat > 1.0
        n_hyp = int(np.count_nonzero(hyp))

        if n_hyp == 0:
            E_flat = Anomalies._solve_elliptic(M_flat, e_flat, tol=tol, solver=solver, max_ite=max_ite)
        elif n_hyp == M_flat.size:
            E_flat = Anomalies._solve_hyperbolic(M_flat, e_flat, tol=tol, solver=solver, max_ite=max_ite)
        else:
            # Mixed populations are rare enough that the two gathers are not worth avoiding.
            E_flat = np.empty_like(M_flat)
            ell = ~hyp
            E_flat[ell] = Anomalies._solve_elliptic(
                M_flat[ell], e_flat[ell], tol=tol, solver=solver, max_ite=max_ite)
            E_flat[hyp] = Anomalies._solve_hyperbolic(
                M_flat[hyp], e_flat[hyp], tol=tol, solver=solver, max_ite=max_ite)

        if _M.ndim == 0:
            return float(E_flat[0])
        return E_flat.reshape(out_shape)

    @staticmethod
    def _solve_elliptic(
        M: NDArray[np.float64], e: NDArray[np.float64], *,
        tol: float, solver: str, max_ite: int,
    ) -> NDArray[np.float64]:
        """Seed and solve the elliptic/circular branch. `np.where` beats masked assignment here."""
        E = np.where(e <= 0.55, M, np.where(e <= 0.95, np.cbrt(6.0 * M), np.pi))
        return Anomalies._iterate_kepler(
            M, e, E, hyperbolic=False, tol=tol, solver=solver, max_ite=max_ite)

    @staticmethod
    def _solve_hyperbolic(
        M: NDArray[np.float64], e: NDArray[np.float64], *,
        tol: float, solver: str, max_ite: int,
    ) -> NDArray[np.float64]:
        """Seed and solve the hyperbolic branch. e > 1 throughout, so the division is safe."""
        E = np.arcsinh(M / e)
        return Anomalies._iterate_kepler(
            M, e, E, hyperbolic=True, tol=tol, solver=solver, max_ite=max_ite)


    @staticmethod
    def true_to_mean(theta: Radians, e: Numeric) -> Radians | Numeric: # Merge this with the true_to_mean_parabolic! #Got to here
        """
        Compressed function, Converts Mean Anomaly to True Anomaly by:
        Circular, Elliptic, Hyperbolic: theta -> E/H -> M
        Parabolic:                      theta -> M_p
        """
        # E = Anomalies.true_to_eccentric(theta, e)
        # M = Anomalies.eccentric_to_mean(E, e)
        # return M
        _theta = np.asarray(theta, dtype=np.float64)
        _e = np.asarray(e, dtype=np.float64)
        par = np.abs(_e - 1.0) < 1e-8
        n_par = ~par

        M = np.empty_like(_theta)
        if np.any(n_par):
            E = Anomalies.true_to_eccentric(_theta[n_par], _e[n_par])
            M[n_par] = Anomalies.eccentric_to_mean(E, _e[n_par])
        if np.any(par):
            M[par] = Anomalies.true_to_mean_parabolic(_theta[par])

        if _theta.ndim == 0:
            return M.item()
        return M

    @staticmethod
    def mean_to_true(M: Radians | Numeric, e: Numeric, *, tol: float = 1e-5, solver: str = "N-R", max_ite: int = 1000) -> Radians:
        """
        Compressed function, Converts True Anomaly to Mean Anomaly by:
        Circular, Elliptic, Hyperbolic: M -> E/H -> theta
        Parabolic:                      M_p -> theta
        """
        # E = Anomalies.mean_to_eccentric(M, e, tol=tol, solver=solver, max_ite=max_ite)
        # theta = Anomalies.eccentric_to_true(E, e)
        # return theta
        _M = np.asarray(M, dtype=np.float64)
        _e = np.asarray(e, dtype=np.float64)

        par = np.abs(_e - 1.0) < 1e-8
        n_par = ~par

        theta = np.zeros_like(_M)

        if np.any(n_par):
            E = Anomalies.mean_to_eccentric(_M[n_par], _e[n_par], tol=tol, solver=solver, max_ite=max_ite)
            theta[n_par] = Anomalies.eccentric_to_true(E, _e[n_par])

        if np.any(par):
            theta[par] = Anomalies.mean_to_true_parabolic(_M[par])

        if _M.ndim == 0:
            return theta.item()
        return theta
        

    @staticmethod
    def true_to_mean_parabolic(theta: Radians) -> Numeric:
        """Parabolic True Anomaly to Mean Anomaly, Using Cardino Solutions."""
        _theta = np.asarray(theta, dtype=np.float64)
        M_p = np.tan(_theta/2) + (1.0/3.0)*np.tan(_theta/2)**3
        
        if _theta.ndim == 0:
            return float(M_p.item())
        return M_p
    
    @staticmethod
    def mean_to_true_parabolic(M_p: Numeric) -> Radians:
        """Parabolic Mean Anomaly to True Anomaly, Solution to Cardion cubic equation s^3 + 3s - 3M_p = 0."""
        # Using the Cardino solution for a cubic equation s**3 + 3s - 3Mp = 0
        _M_p = np.asarray(M_p)
        A = 1.5 * _M_p
        B = np.cbrt(A + np.sqrt(A**2 + 1.0))
        s = B - (1.0 / B)
        theta = 2.0 * np.arctan(s)

        if _M_p.ndim == 0:
            return float(theta.item())
        return cast(Radians, theta)

# ==========================================================================================================================================================
# Kepler and Barker Static Classes: Time <-> Mean Anomaly conversions for basic propagation logic on all orbital types.
# ==========================================================================================================================================================
    
class Kepler:
    """
    Kepler's Equation relating orbital position and orbital period. Inputs and outputs are in terms of seconds
    and/or radians.
    """


    @staticmethod
    def t_to_M(mu: Numeric, a: Kilometers, delta_t: Seconds) -> tuple[Radians, NDArray[np.bool_]]:
        """Change in Time to change in Mean Anomaly. **Also returns an info array which highlights which entries the function worked on."""
        _mu = np.asarray(mu, dtype=np.float64)
        _a = np.asarray(a, dtype=np.float64)

        valid = np.abs(_a) > 1e-9
        M = np.zeros_like(_a)

        M[valid] = np.sqrt(_mu[valid] / (_a[valid] ** 3)) * delta_t
        if _mu.ndim == 0:
            return float(M.item()), valid
        # return Radians(np.sqrt(mu / (a**3)) * delta_t)
        return M, valid
        
    @staticmethod
    def M_to_t(mu: Numeric, a: Kilometers, delta_M: Radians) -> Seconds:
        """Change in Mean Anomaly to change in Time. *For now only returns a homogenous time constant! 
        **Also returns an info array which highlights which entries the function worked on."""
        _mu = np.asarray(mu, dtype=np.float64)
        _a = np.asarray(a, dtype=np.float64)
        _M = np.asarray(delta_M, dtype=np.float64)

        t = np.sqrt((_a ** 3) / _mu) * _M

        if t.ndim > 0 and not np.allclose(t, t[0], atol=1e-8):
            raise NotImplementedError(f"This function expects an homogenous time constant between all entries")
        
        if t.ndim == 0:
            return float(t.item())
        return cast(Seconds, t[0]) # We'll prob return an array when we start doing orbital transfers i.e. Laplace Equation
    
        # return Seconds(np.sqrt((a**3) / mu) * delta_M)
    
class Barker:
    """
    Barker's Equation covers the special case of a parabolic trajectory, hence a is undefined and Kepler's
    Equation is no longer sufficient
    """


    @staticmethod
    def t_to_M(mu: Numeric, p: Kilometers, delta_t: Seconds) -> Numeric:
        """Change in Time to change in Parabolic Mean Anomaly."""
        _mu = np.asarray(mu, dtype=np.float64)
        _p = np.asarray(p, dtype=np.float64)

        M = 2.0 * np.sqrt(_mu / (_p ** 3)) * delta_t
        if _mu.ndim == 0:
            return M.item()
        return M
        # return float(2.0*np.sqrt(mu / (p**3)) * delta_t)
    
    @staticmethod
    def M_to_t(mu: Numeric, p: Kilometers, delta_M: Numeric) -> Seconds:
        """Change in Parabolic Mean Anomaly to change in Time. *For now only returns a homogenous time constant!"""
        _mu = np.asarray(mu, dtype=np.float64)
        _p = np.asarray(p, dtype=np.float64)
        _M = np.asarray(delta_M, dtype=np.float64)

        t = 0.5*np.sqrt((_p ** 3) /_mu) * _M
        # return Seconds(0.5*np.sqrt(p**3 / mu) * delta_M)
        if t.ndim > 0 and not np.allclose(t, t[0], atol=1e-8):
            raise NotImplementedError(f"This function expects an homogenous time constant between all entries")
        
        if t.ndim == 0:
            return t.item()
        return cast(Seconds, t[0])