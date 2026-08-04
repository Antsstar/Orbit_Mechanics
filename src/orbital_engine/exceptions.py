from __future__ import annotations

class OrbitalEngineError(Exception):
    """
    Base class for all domain-specific errors within orbital_engine.
    Allows API consumers to catch all library-generated errors cleanly.
    """
    pass

# ==========================================================================================================================================================
# Numerical Errors
# ==========================================================================================================================================================

class ConvergenceError(OrbitalEngineError, RuntimeError):
    """Raised when a numerical method fails to converge within the specified number of iterations."""
    pass

class SingularityError(OrbitalEngineError, ValueError):
    """Raised when a bodies motion wouldn't result in a valid orbit. i.e. negative eccentricity, zeroed semi-latus rectum..."""

# ==========================================================================================================================================================
# Graph and Memory Errors
# ==========================================================================================================================================================
class TopologyError(OrbitalEngineError, ValueError):
    """
    Raised when Entity-Component-System (ECS) adjacency list contains an invalid topology, i.e. cyclic parent-child dependency loops, or
    unresolvable orphaned reference ID's during the Breath-First-Search (BFS) stratification.
    """
    pass

class RegistryError(OrbitalEngineError, KeyError):
    """Raised when querying an unregistered model or propagator frome engine registries."""
    pass

class ArenaAllocationError(OrbitalEngineError, MemoryError):
    """Raised when entity instantiation exceeds pre-allocated maximum contingous capacity of the Data-Oriented-Design (DOD) memory arena."""
    pass