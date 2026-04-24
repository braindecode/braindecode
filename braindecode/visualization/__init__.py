"""
Functions for visualisations, especially of the ConvNets.
"""

from .attribution import attribute_image_features, get_attributions
from .confusion_matrices import plot_confusion_matrix
from .gradients import compute_amplitude_gradients
from .metrics import METRIC_NAMES, compute_metrics
from .topology import project_to_topomap

__all__ = [
    "compute_amplitude_gradients",
    "plot_confusion_matrix",
    "attribute_image_features",
    "get_attributions",
    "compute_metrics",
    "METRIC_NAMES",
    "project_to_topomap",
]
