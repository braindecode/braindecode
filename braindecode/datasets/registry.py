"""
Dataset registry for Hub integration.

This module provides a registry pattern to avoid circular imports between
base.py (dataset definitions) and hub.py (Hub mixin). Dataset classes
register themselves when defined, and Hub methods look up classes dynamically.
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

    This allows Hub integration code to look up dataset classes by name
    without importing them directly, avoiding circular imports.

    Parameters
    ----------
    cls : Type
        The dataset class to register.

    Returns
    -------
    Type
        The same class (unchanged), so this can be used as a decorator.

    Examples
    --------
    >>> @register_dataset
    >>> class WindowsDataset(BaseDataset):
    >>>     pass
    """
    _DATASET_REGISTRY[cls.__name__] = cls
    return cls


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

    Examples
    --------
    >>> WindowsDataset = get_dataset_class('WindowsDataset')
    >>> dataset = WindowsDataset(...)
    """
    if name not in _DATASET_REGISTRY:
        raise KeyError(
            f"Dataset class '{name}' not found in registry. "
            f"Available classes: {list(_DATASET_REGISTRY.keys())}"
        )
    return _DATASET_REGISTRY[name]


def get_dataset_type(obj: Any) -> str:
    """
    Get the registered type name for a dataset instance.

    This function checks if the object is an instance of any registered
    dataset class and returns the class name.

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

    Examples
    --------
    >>> dataset = WindowsDataset(...)
    >>> get_dataset_type(dataset)
    'WindowsDataset'
    """
    for name, cls in _DATASET_REGISTRY.items():
        if isinstance(obj, cls):
            return name

    raise TypeError(
        f"Object of type {type(obj).__name__} is not a registered dataset class. "
        f"Available classes: {list(_DATASET_REGISTRY.keys())}"
    )


def is_registered_dataset(obj: Any, class_name: str) -> bool:
    """
    Check if an object is an instance of a registered dataset class.

    This is a safer alternative to isinstance() that works without
    importing the concrete class.

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

    Examples
    --------
    >>> dataset = WindowsDataset(...)
    >>> is_registered_dataset(dataset, 'WindowsDataset')
    True
    >>> is_registered_dataset(dataset, 'EEGWindowsDataset')
    False
    """
    try:
        cls = get_dataset_class(class_name)
        return isinstance(obj, cls)
    except KeyError:
        return False
