# Authors: Vandit Shah <shahvandit@gmail.com>
#
# License: BSD (3-clause)

"""Time-domain attribution methods.

All methods are thin wrappers around :mod:`captum.attr` with the
braindecode-friendly ``(model, x, target)`` signature. Captum is a soft
dependency; install with ``pip install braindecode[viz]``. Calling any
method without captum raises :class:`ImportError`.

Frequency-domain attribution lives in
:mod:`braindecode.visualization.frequency`.
"""

import torch
from mne.utils import _soft_import

captum = _soft_import("captum", purpose="attribution methods", strict=False)


def saliency(model, x, target):
    """Vanilla saliency: ``|d y[target] / d x|``.

    Parameters
    ----------
    model : torch.nn.Module
    x : torch.Tensor of shape ``(batch, n_chans, n_times)``
    target : torch.Tensor of shape ``(batch,)``
        Class indices.

    Returns
    -------
    torch.Tensor of the same shape as ``x``.
    """
    _require_captum()
    from captum.attr import Saliency

    return _captum_attribute(Saliency, model, x, target)


def input_x_gradient(model, x, target):
    """Element-wise input × input-gradient."""
    _require_captum()
    from captum.attr import InputXGradient

    return _captum_attribute(InputXGradient, model, x, target)


def integrated_gradients(model, x, target, baseline=None, steps=50):
    """Integrated Gradients (Sundararajan et al., 2017) via captum.

    Path integral of input gradients from ``baseline`` to ``x`` evaluated
    by Gauss-Legendre quadrature with ``steps`` evaluations (captum's
    default). Default baseline is zeros. Even moderate ``steps`` values
    satisfy the completeness axiom (sum of attributions ≈
    ``f(x) - f(baseline)``).
    """
    _require_captum()
    from captum.attr import IntegratedGradients

    return _captum_attribute(
        IntegratedGradients,
        model,
        x,
        target,
        baselines=baseline,
        n_steps=steps,
    )


def layer_grad_cam(model, x, target, layer):
    """Compute a class activation map at a convolutional layer with Captum.

    A layer with a three-dimensional output ``(batch, features, length)``,
    such as a temporal convolution, gives one map per trial of shape
    ``(batch, n_times)``: its last axis, which must be time, is linearly
    interpolated to ``n_times``. A layer with a four-dimensional output
    ``(batch, features, channels, time)`` gives a map of shape
    ``(batch, n_chans, n_times)``: its two last axes are bilinearly
    interpolated to ``(n_chans, n_times)``. Its channel axis must hold
    ``n_chans`` entries, or one once the layer has mixed the channels, and its
    time axis more than one step; otherwise a :class:`ValueError` names the
    layer. This rejects the layers of :class:`~braindecode.models.ShallowFBCSPNet`
    and :class:`~braindecode.models.Deep4Net` after their ``dimshuffle``, which
    hold time on the height axis. Token sequences of transformers hold tokens
    on the features axis, so their maps are not maps over time.
    Localization is limited by the resolution of the selected layer.
    """
    _require_captum()
    from captum.attr import LayerAttribution, LayerGradCam

    x_leaf = x.detach().clone().requires_grad_(True)
    cam = LayerGradCam(model, layer).attribute(x_leaf, target=target)
    if cam.ndim == 3:
        size, mode = x.shape[-1:], "linear"
    elif cam.ndim == 4:
        # cam: (batch, 1, height, width) of the layer. It is a channels x time
        # map only if the height axis is the channels and the width axis time.
        if cam.shape[-2] > x.shape[-2] or cam.shape[-1] == 1:
            name = next(
                (n for n, m in model.named_modules() if m is layer),
                type(layer).__name__,
            )
            raise ValueError(
                f"Grad-CAM at layer {name!r} needs an output (batch, features, "
                f"channels, time) with {x.shape[-2]} or 1 channels and more than "
                f"one time step, got {tuple(cam.shape[-2:])} on its last two axes."
            )
        size, mode = x.shape[-2:], "bilinear"
    else:
        raise ValueError(
            f"Grad-CAM needs a layer with one or two spatial dimensions, "
            f"got an activation map of shape {tuple(cam.shape)}."
        )
    cam = LayerAttribution.interpolate(cam, size, interpolate_mode=mode)
    return cam.squeeze(1).detach()


def guided_backprop(model, x, target):
    """GuidedBackprop attribution (Springenberg et al., 2014) via captum.

    Backpropagates through ReLUs keeping only positive gradients with
    positive activations, which often produces sharper input-space maps
    than vanilla saliency.
    """
    _require_captum()
    from captum.attr import GuidedBackprop

    return _captum_attribute(GuidedBackprop, model, x, target)


def deconvolution(model, x, target):
    """DeconvNet-style attribution (Zeiler & Fergus, 2014) via captum."""
    _require_captum()
    from captum.attr import Deconvolution

    return _captum_attribute(Deconvolution, model, x, target)


def deep_lift(model, x, target, baseline=None):
    """DeepLIFT attribution (Shrikumar et al., 2017) via captum.

    ``baseline`` defaults to zeros. Pass a tensor with the same shape as
    ``x`` to use a different reference (e.g. a per-channel mean).
    """
    _require_captum()
    from captum.attr import DeepLift

    if baseline is None:
        baseline = torch.zeros_like(x)
    return _captum_attribute(DeepLift, model, x, target, baselines=baseline)


def lrp(model, x, target):
    """Layer-wise Relevance Propagation (Bach et al., 2015) via captum.

    Captum picks default propagation rules; pass a configured ``LRP``
    instance directly if you need custom rules.
    """
    _require_captum()
    from captum.attr import LRP

    return _captum_attribute(LRP, model, x, target)


def _require_captum():
    if captum is False:
        raise ImportError(
            "This attribution method requires captum. "
            "Install with `pip install braindecode[viz]`."
        )


def _captum_attribute(algorithm_cls, model, x, target, **attribute_kwargs):
    """Run a captum algorithm constructed from ``model`` on a fresh leaf input.

    Captum's :meth:`attribute` may return a tensor that still requires
    grad (e.g. IG keeps a graph); we detach so the caller can ``.numpy()``
    or pass the result to downstream pure-numpy metrics without surprises.
    """
    x_leaf = x.detach().clone().requires_grad_(True)
    attr = algorithm_cls(model).attribute(x_leaf, target=target, **attribute_kwargs)
    return attr.detach()
