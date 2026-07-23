# Custom types for the orbital engine
import enum

class PropagatorType(enum.IntEnum):
    NONE = 0
    KEPLERIAN = 1

# Distances
Kilometers = float
AstronomicalUnits = float

# Areas
SquareMeters = float

# Masses
Kilograms = float

# Times
Seconds = float

EarthDays = float
EarthYears = float

# Angles
Degrees = float
Radians = float

# Velocities
KmPerSec = float