"""
Catalogues of things a `Simulation` can be configured to use: propagators, and force models.

**Why this file was empty of readers until now.** It existed as two dictionaries and a handful of
functions with no caller. That was the right amount of code for a catalogue nobody queried yet, and
the wrong shape for one that must support a sweep: the original `register_model`/`get_model` pair
registered bare *classes* under a string name, which invites exactly the pattern the force-model
layer is designed to avoid - a string-keyed lookup, or a subclass, resolved somewhere inside a step.
That pair has been replaced by the force-model section below. `register_propagator`/`get_propagators`
are unchanged; propagator *selection* (Keplerian vs. Cowell vs. ...) is a separate axis from force
*composition* and is being wired by the integrator work, not here.

See `forces.py` for the composition layer this registry feeds, and its module docstring for the full
contract a force model implements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple, Type, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .custom_types import PropagatorType
from .exceptions import RegistryError

if TYPE_CHECKING:
    from .propagators import Propagator
    from .forces import ForceKernel
    from .simulator import Simulation


# ==================================================================================================
# Propagator registry. `Simulation.step()` dispatches non-Keplerian bodies through it; per-body
# selection is `Simulation.set_propagator`.
# ==================================================================================================

_PROPAGATOR_REGISTRY: Dict[int, Type["Propagator"]] = {}


def register_propagator(prop_type: PropagatorType, prop_class: Type["Propagator"]) -> None:
    """
    Register an instantiated propagator to specific IntEnum ID.
    """
    _PROPAGATOR_REGISTRY[prop_type.value] = prop_class


def get_propagators() -> Dict[int, Type["Propagator"]]:
    """
    Returns the dictionary of all registered propagators.
    """
    return _PROPAGATOR_REGISTRY


# ==================================================================================================
# Force-model registry
# ==================================================================================================
#
# A force model is a stateless kernel (see `forces.ForceKernel`) plus a name and the names of
# whatever per-body coefficients it needs. Registering one assigns it the next free bit in the
# per-body `force_model_mask` array (`custom_types.ForceModelMask`, a `uint64` column in the arena).
# That bit is what makes "which physics does this body have" a *value* - composing two models on one
# body is `mask |= bit_a; mask |= bit_b`, no branch and no per-body class.
#
# **Bit assignment is by registration order, within one process.** It is stable for the lifetime of a
# run and is NOT a stable identifier across code changes - a model registered earlier in the import
# graph today could register later tomorrow. A sweep configuration must therefore be serialized as
# model *names* (`["j2", "drag"]`), never as a raw integer mask, and re-resolved through this registry
# every time it is loaded. `mask_for` and `Simulation.enable_force_model` are the name-to-bit boundary;
# nothing above them should ever see a bit position.
#
# **Registration happens at import time**, exactly like `register_propagator` above: a module that
# defines a kernel calls `@register_force_model(...)` on it at module level, and the registration
# takes effect the first time that module is imported anywhere in the process - see `forces.py` for
# the two demonstration models registered this way.

MAX_FORCE_MODELS = 64  # Bit width of ForceModelMask (np.uint64). See the headroom note in custom_types.py.

# A configuration-time check a model can register: called by `Simulation.enable_force_model` with the
# slots about to be enabled, *before* any mask bit is set, and expected to raise `ValueError` for
# bodies the model is meaningless on. It exists for conditions a kernel cannot see from its own
# arguments, such as `j2` on a body whose parent is a barycentre (`is_system` is not in the kernel
# signature). Never called inside a step.
BodyValidator = Callable[["Simulation", NDArray[np.int64]], None]


@dataclass(frozen=True)
class ForceModel:
    """
    One registered force model: its name, its bit in the per-body mask, its stateless kernel, and the
    names of the per-body coefficients it reads from `Simulation.force_model_params[name]`.

    `param_names` is positional-to-column: `param_names[k]` is column `k` of that model's `(C, k)`
    parameter array. An empty tuple means the model takes no per-body coefficients at all (a valid,
    common case - see `forces.py`'s zero-parameter demonstration kernel).
    """
    name: str
    bit: int
    kernel: "ForceKernel"
    param_names: Tuple[str, ...] = ()
    citation: str = ""
    validate_bodies: Optional[BodyValidator] = None

    @property
    def n_params(self) -> int:
        return len(self.param_names)


_FORCE_MODEL_REGISTRY: Dict[str, ForceModel] = {}
_next_force_model_bit = 0


def register_force_model(
    name: str,
    *,
    param_names: Tuple[str, ...] = (),
    citation: str = "",
    validate_bodies: Optional[BodyValidator] = None,
) -> Callable[["ForceKernel"], "ForceKernel"]:
    """
    Decorator. Registers `name` at the next free mask bit and returns the kernel unchanged, so the
    decorated function is still directly callable and directly testable - registration is a side
    effect, not a wrapper.

        @register_force_model("j2", param_names=("j2", "r_eq"), citation="Vallado 4e, Eq. 9-41")
        def j2_kernel(indices, t, state, mu_array, parent_indices, params, out) -> None:
            ...

    `validate_bodies`, if given, is the model's configuration-time check (see `BodyValidator`).

    Raises `RegistryError` (not a bare `ValueError`) on a duplicate name or a full registry, so both
    are catchable alongside every other registry lookup failure in this module.
    """
    def wrapper(kernel: "ForceKernel") -> "ForceKernel":
        global _next_force_model_bit
        if name in _FORCE_MODEL_REGISTRY:
            raise RegistryError(f"force model '{name}' is already registered")
        if _next_force_model_bit >= MAX_FORCE_MODELS:
            raise RegistryError(
                f"force model registry is full at {MAX_FORCE_MODELS} entries (ForceModelMask is a "
                f"{MAX_FORCE_MODELS}-bit field); '{name}' has no bit left to take")

        model = ForceModel(
            name=name, bit=_next_force_model_bit, kernel=kernel,
            param_names=param_names, citation=citation, validate_bodies=validate_bodies,
        )
        _FORCE_MODEL_REGISTRY[name] = model
        _next_force_model_bit += 1
        return kernel
    return wrapper


def get_force_model(name: str) -> ForceModel:
    """
    Look up a registered force model by name.

    This is a **setup-time** call - resolve names once, when a scenario's model configuration is
    built, never per body and never inside a step. `forces.resolve_force_models` is what the
    composition layer actually dispatches from at runtime, and it never calls this function.
    """
    try:
        return _FORCE_MODEL_REGISTRY[name]
    except KeyError:
        raise RegistryError(
            f"force model '{name}' is not registered! "
            f"available: {list(_FORCE_MODEL_REGISTRY.keys())}"
        ) from None


def all_force_models() -> Tuple[ForceModel, ...]:
    """Every registered force model, in bit order. Setup-time use only - see `get_force_model`."""
    return tuple(_FORCE_MODEL_REGISTRY.values())


def mask_for(names: Iterable[str]) -> np.uint64:
    """
    OR together the bits for a set of model names.

    A convenience for building a mask value once, at scenario-configuration time - e.g.
    `sim.force_model_mask[idx] = registry.mask_for(["j2", "drag"])` - rather than one
    `enable_force_model` call per name. Still a setup-time, name-keyed operation; see the module note
    on why bit positions never escape this boundary.
    """
    mask = np.uint64(0)
    for name in names:
        mask |= np.uint64(1) << np.uint64(get_force_model(name).bit)
    return mask
