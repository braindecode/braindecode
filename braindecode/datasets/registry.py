"""
Dataset registry for Hub integration.

Datasets register themselves here so Hub code can look them up by name
without direct imports (avoiding circular dependencies).
"""

# Authors: Kuntal Kokate
#
# License: BSD (3-clause)

from typing import Any, Dict, Type

# Global registry mapping dataset class names to classes
_DATASET_REGISTRY: Dict[str, Type] = {}


def register_dataset(cls: Type) -> Type:
    """
    Decorator to register a dataset class in the global registry.

    Parameters
    ----------
    cls : Type
        The dataset class to register.

    Returns
    -------
    Type
        The same class (unchanged), so this can be used as a decorator.
    """
    _DATASET_REGISTRY[cls.__name__] = cls
    return cls


def _available_datasets_str() -> str:
    """Return a human-readable list of registered dataset class names."""
    if not _DATASET_REGISTRY:
        return "<no registered datasets>"
    return ", ".join(_DATASET_REGISTRY.keys())


def get_dataset_class(name: str) -> Type:
    """
    Retrieve a registered dataset class by name.

    Parameters
    ----------
    name : str
        Name of the dataset class (e.g., 'WindowsDataset').

    Returns
    -------
    Type
        The dataset class.

    Raises
    ------
    KeyError
        If the class name is not registered.
    """
    try:
        return _DATASET_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Dataset class '{name}' not found in registry. "
            f"Available classes: {_available_datasets_str()}"
        ) from exc


def get_dataset_type(obj: Any) -> str:
    """
    Get the registered type name for a dataset instance.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    str
        The name of the dataset class (e.g., 'WindowsDataset').

    Raises
    ------
    TypeError
        If the object is not an instance of any registered dataset class.
    """
    for cls in _DATASET_REGISTRY.values():
        if isinstance(obj, cls):
            return cls.__name__

    raise TypeError(
        f"Object of type {type(obj).__name__} is not a registered dataset class. "
        f"Available classes: {_available_datasets_str()}"
    )


def is_registered_dataset(obj: Any, class_name: str) -> bool:
    """
    Check if an object is an instance of a registered dataset class.

    Parameters
    ----------
    obj : Any
        The object to check.
    class_name : str
        Name of the dataset class to check against.

    Returns
    -------
    bool
        True if obj is an instance of the named class, False otherwise.
    """
    try:
        cls = get_dataset_class(class_name)
    except KeyError:
        return False
    return isinstance(obj, cls)
