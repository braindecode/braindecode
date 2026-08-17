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
from .sae import (
    SparseAutoencoder,
    capture_activations,
    fit_sparse_autoencoder,
    grouped_train_valid_test_split,
    run_with_activation_substitution,
    sae_diagnostics,
)
from .sanity import cascading_layer_reset, random_target
from .topology import project_to_topomap

__all__ = [
    "METRIC_NAMES",
    "SSIM_METRIC_NAMES",
    "SparseAutoencoder",
    "amplitude_gradients",
    "amplitude_gradients_per_trial",
    "capture_activations",
    "cascading_layer_reset",
    "compute_metrics",
    "compute_ssim_metrics",
    "deconvolution",
    "deep_lift",
    "fit_sparse_autoencoder",
    "grouped_train_valid_test_split",
    "guided_backprop",
    "input_x_gradient",
    "integrated_gradients",
    "layer_grad_cam",
    "lrp",
    "plot_confusion_matrix",
    "project_to_topomap",
    "random_target",
    "run_with_activation_substitution",
    "sae_diagnostics",
    "saliency",
]
