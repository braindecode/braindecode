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
    """Layer GradCAM via captum: class-discriminative localization at ``layer``.

    Captum's :class:`captum.attr.LayerGradCam` returns the cam at the
    layer's spatial dims; we then bilinear-resample to the input's last
    two dimensions via :meth:`captum.attr.LayerAttribution.interpolate`.
    """
    _require_captum()
    from captum.attr import LayerAttribution, LayerGradCam

    x_leaf = x.detach().clone().requires_grad_(True)
    cam = LayerGradCam(model, layer).attribute(x_leaf, target=target)
    cam = LayerAttribution.interpolate(cam, x.shape[-2:], interpolate_mode="bilinear")
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
