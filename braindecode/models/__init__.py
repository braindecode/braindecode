"""Predefined network architectures for EEG decoding."""

import ast
import importlib
from importlib.resources import files
from typing import TYPE_CHECKING

_LAZY: dict[str, str]
__all__: list[str]

if TYPE_CHECKING:
    from ._exports import *  # noqa: F403
    from ._exports import _init_models_dict as _init_models_dict
else:
    # One import in the stub supplies the model name for typing and lazy lookup.
    _LAZY = {
        alias.name: node.module
        for node in ast.parse(
            files(__name__).joinpath("_exports.pyi").read_bytes()
        ).body
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module
        for alias in node.names
    }
    __all__ = list(_LAZY)

    def __getattr__(name):
        if name in _LAZY:
            value = getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
        elif name in _LAZY.values():
            value = importlib.import_module(f".{name}", __name__)
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        globals()[name] = value
        return value

    def __dir__():
        return sorted(set(globals()) | set(__all__))
