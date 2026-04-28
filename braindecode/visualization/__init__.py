"""
Functions for visualisations, especially of the ConvNets.
"""

from .attribution import (
    input_x_gradient,
    integrated_gradients,
    layer_grad_cam,
    saliency,
    select_correctly_classified,
)
from .confusion_matrices import plot_confusion_matrix
from .gradients import compute_amplitude_gradients
from .metrics import METRIC_NAMES, compute_metrics
from .topology import project_to_topomap

__all__ = [
    "METRIC_NAMES",
    "compute_amplitude_gradients",
    "compute_metrics",
    "input_x_gradient",
    "integrated_gradients",
    "layer_grad_cam",
    "plot_confusion_matrix",
    "project_to_topomap",
    "saliency",
    "select_correctly_classified",
]
