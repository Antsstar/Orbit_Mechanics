from __future__ import annotations
from typing import Final
import numpy as np
from .custom_types import (
    ScalarFloat, ScalarRadians, ScalarDegrees, ScalarKilometers, ScalarAstronomicalUnits,
    ScalarSeconds, ScalarEarthDays, ScalarGravitationalParameter, ScalarKmPerSec
)


# ==========================================================================================================================================================
# Angle & Trig transformations
# ==========================================================================================================================================================

DEG2RAD: Final[ScalarRadians] = np.pi / 180
RAD2DEG: Final[ScalarDegrees] = 180 / np.pi

# ==========================================================================================================================================================
# Astronomical distances and time scales (IAU 2012 definition)
# ==========================================================================================================================================================

# Astronomical unit in kilometers
AU2KM: Final[ScalarKilometers]          = 149_597_870.7 # Kilometers
KM2AU: Final[ScalarAstronomicalUnits]   = 1.0 / AU2KM

# Time constants
JD2SEC: Final[ScalarSeconds]            = 86400.0 # Seconds
JY2DAY: Final[ScalarEarthDays]          = 365.25 # EarthDays

# Speed of Light in Vacuum
C_SI: Final[ScalarFloat]                = 299_792_458.0 # m/s
C_KM: Final[ScalarKmPerSec]             = 299_792.458   # km/s
C                                       = C_KM # KmPerSec

# Light Year distances
LY2KM: Final[ScalarKilometers]          = C * JD2SEC * JY2DAY # Kilometers
LY2AU: Final[ScalarAstronomicalUnits]   = LY2KM * KM2AU

# ==========================================================================================================================================================
# Gravitational constants (G)
# ==========================================================================================================================================================

G_SI: Final[ScalarGravitationalParameter] = 6.6743e-11 # m^3 / kgs^2
G_KM: Final[ScalarGravitationalParameter] = 6.6743e-20 # km^3/ kgs^2
G: Final[ScalarGravitationalParameter]    = G_KM
