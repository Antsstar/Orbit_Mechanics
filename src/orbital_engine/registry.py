from typing import Callable, Dict, Type, Any

_MODEL_REGISTRY: Dict[str, Type[Any]] = {}

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