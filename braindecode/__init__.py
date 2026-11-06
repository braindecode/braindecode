import importlib
from typing import TYPE_CHECKING

from .version import __version__

if TYPE_CHECKING:  # type checkers and IDEs see the classes; runtime uses __getattr__
    from .classifier import EEGClassifier
    from .regressor import EEGRegressor

__all__ = [
    "__version__",
    "EEGClassifier",
    "EEGRegressor",
]

# Subpackages reachable as ``braindecode.<name>`` after a bare ``import braindecode``.
_SUBMODULES = frozenset(
    {
        "augmentation",
        "classifier",
        "datasets",
        "datautil",
        "eegneuralnet",
        "functional",
        "models",
        "modules",
        "preprocessing",
        "regressor",
        "samplers",
        "training",
        "util",
        "visualization",
    }
)

if not TYPE_CHECKING:  # hidden from checkers so misspelled names stay errors

    def __getattr__(name):
        # PEP 562: skorch, the datasets and the models load on first use.
        if name == "EEGClassifier":
            from .classifier import EEGClassifier

            return EEGClassifier
        if name == "EEGRegressor":
            from .regressor import EEGRegressor

            return EEGRegressor
        if name in _SUBMODULES:
            return importlib.import_module(f".{name}", __name__)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __dir__():
        return sorted(set(globals()) | set(__all__) | _SUBMODULES)
