"""
Functions for visualisations, especially of the ConvNets.
"""

from .attribution import (
    cascading_layer_reset,
    deconvolution,
    deep_lift,
    guided_backprop,
    input_x_gradient,
    integrated_gradients,
    layer_grad_cam,
    lrp,
    random_target,
    saliency,
    select_correctly_classified,
)
from .confusion_matrices import plot_confusion_matrix
from .gradients import compute_amplitude_gradients
from .metrics import (
    METRIC_NAMES,
    SSIM_METRIC_NAMES,
    compute_metrics,
    compute_ssim_metrics,
)
from .topology import project_to_topomap

__all__ = [
    "METRIC_NAMES",
    "SSIM_METRIC_NAMES",
    "cascading_layer_reset",
    "compute_amplitude_gradients",
    "compute_metrics",
    "compute_ssim_metrics",
    "deconvolution",
    "deep_lift",
    "guided_backprop",
    "input_x_gradient",
    "integrated_gradients",
    "layer_grad_cam",
    "lrp",
    "plot_confusion_matrix",
    "project_to_topomap",
    "random_target",
    "saliency",
    "select_correctly_classified",
]
