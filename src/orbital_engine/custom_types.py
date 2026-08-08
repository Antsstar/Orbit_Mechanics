### Custom types for the orbital engine
from __future__ import annotations
import enum
from typing import TypeAlias, Union
import numpy as np
from numpy.typing import NDArray

class PropagatorType(enum.IntEnum):
    NONE        = 0
    KEPLERIAN   = 1 # Standard 2 body-fixed keplerian mathematics, also allows for barycentric definition incorporating a reflexive body movement.
    CR3BP       = 2 # Circular Restricted 3-Body (Translunar / Lagrange Corridor)
    COWELL      = 3 # Direct Numerical N-Body Integration
    ENCKE       = 4 # Oscilating Reference Derivation Integration
    SGP4        = 5 # LEO Analytical Drag & J2-J4 Harmonics
    LUNAR_MASCON= 6 # Lunar-centric high-fidelity gravity field

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
# Core Data-oriented types
# ==========================================================================================================================================================

# Standard floats, NumPy float(64)s and N-dimensional contiguous arrays of type NumPy float(64)
ScalarFloat: TypeAlias = Union[float, np.float64]
ArrayFloat: TypeAlias = NDArray[np.float64]
Numeric: TypeAlias = Union[ScalarFloat, ArrayFloat]


# ==========================================================================================================================================================
# Physical units type aliases (SI / Astronomical Units)
# ==========================================================================================================================================================

# Distances and positions
ScalarKilometers: TypeAlias = ScalarFloat
ArrayKilometers: TypeAlias = ArrayFloat
Kilometers: TypeAlias = Union[ScalarKilometers, ArrayKilometers]

ScalarAstronomicalUnits: TypeAlias = ScalarFloat
ArrayAstronomicalUnits: TypeAlias = ArrayFloat
AstronomicalUnits: TypeAlias = Union[ScalarAstronomicalUnits, ArrayAstronomicalUnits]

# Areas (Cross-sections, Radiation Pressure profiles)
ScalarSquareMeters: TypeAlias = ScalarFloat
ArraySquareMeters: TypeAlias = ArrayFloat
SquareMeters: TypeAlias = Union[ScalarSquareMeters, ArraySquareMeters]

# Masses & Gravitational Constants
ScalarKilograms: TypeAlias = ScalarFloat
ArrayKilograms: TypeAlias = ArrayFloat
Kilograms: TypeAlias = Union[ScalarKilograms, ArrayKilograms]

ScalarGravitationalParameter: TypeAlias = ScalarFloat # Units: km^3 / s^2
ArrayGravitationalParameter: TypeAlias = ArrayFloat
GravitationalParameter: TypeAlias = Union[ScalarGravitationalParameter, ArrayGravitationalParameter]

# Times & Epochs
ScalarSeconds: TypeAlias = ScalarFloat
ArraySeconds: TypeAlias = ArrayFloat
Seconds: TypeAlias = Union[ScalarSeconds, ArraySeconds]

ScalarEarthDays: TypeAlias = ScalarFloat
ArrayEarthDays: TypeAlias = ArrayFloat
EarthDays: TypeAlias = Union[ScalarEarthDays, ArrayEarthDays]

ScalarEarthYears: TypeAlias = ScalarFloat
ArrayEarthYears: TypeAlias = ArrayFloat
EarthYears: TypeAlias = Union[ScalarEarthYears, ArrayEarthYears]

# Angular Geometry & Anomalies
ScalarDegrees: TypeAlias = ScalarFloat
ArrayDegrees: TypeAlias = ArrayFloat
Degrees: TypeAlias = Union[ScalarDegrees, ArrayDegrees]

ScalarRadians: TypeAlias = ScalarFloat
ArrayRadians: TypeAlias = ArrayFloat
Radians: TypeAlias = Union[ScalarRadians, ArrayRadians]

# Kinematic Velocities & Delta-V
ScalarKmPerSec: TypeAlias = ScalarFloat
ArrayKmPerSec: TypeAlias = ArrayFloat
KmPerSec: TypeAlias = Union[ScalarKmPerSec, ArrayKmPerSec]