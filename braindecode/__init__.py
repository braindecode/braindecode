from .version import __version__

from .classifier import EEGClassifier
from .regressor import EEGRegressor

__all__ = [
    "__version__",
    "EEGClassifier",
    "EEGRegressor",
]
