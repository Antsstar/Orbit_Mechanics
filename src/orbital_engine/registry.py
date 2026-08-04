from __future__ import annotations
from typing import Callable, Dict, Type, Any, TYPE_CHECKING
from .custom_types import PropagatorType


if TYPE_CHECKING:
    from .propagators import Propagator


_MODEL_REGISTRY: Dict[str, Type[Any]] = {}
_PROPAGATOR_REGISTRY: Dict[int, Type[Propagator]] = {}

def register_model(name: str) -> Callable[..., Type[Any]]:
    """
    Decorator to automatically register a physics model class to engine
    """

    def wrapper(cls: Type[Any]) -> Type[Any]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered!")
        _MODEL_REGISTRY[name] = cls
        return cls
    return wrapper

def get_model(name: str) -> Type[Any]:
    """
    Retrieve a registered physics model class by name.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' is not registered!"
            f" Available models: {list(_MODEL_REGISTRY.keys())}"
            )
    return _MODEL_REGISTRY[name]

def register_propagator(prop_type: PropagatorType, prop_class: Type[Propagator]) -> None:
    """
    Register an instantiated propagator to specific IntEnum ID.
    """
    _PROPAGATOR_REGISTRY[prop_type.value] = prop_class

def get_propagators() -> Dict[int, Any]:
    """
    Returns the dictionary of all registered propagators.
    """
    return _PROPAGATOR_REGISTRY