"""
Functions for visualisations, especially of the ConvNets.
"""

from .attribution import (
    deconvolution,
    deep_lift,
    guided_backprop,
    input_x_gradient,
    integrated_gradients,
    layer_grad_cam,
    lrp,
    saliency,
)
from .confusion_matrices import plot_confusion_matrix
from .frequency import amplitude_gradients, amplitude_gradients_per_trial
from .metrics import (
    METRIC_NAMES,
    SSIM_METRIC_NAMES,
    compute_metrics,
    compute_ssim_metrics,
)
from .sanity import cascading_layer_reset, random_target
from .topology import project_to_topomap

__all__ = [
    "METRIC_NAMES",
    "SSIM_METRIC_NAMES",
    "amplitude_gradients",
    "amplitude_gradients_per_trial",
    "cascading_layer_reset",
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
]
