# Authors: Vandit Shah <vanditshah@outlook.com>
#          Akshay Sujatha Ravindran <asujatharavindran@uh.edu>  (random-target
#          and cascading-layer-randomization helpers adapted from the author's
#          research code for Sujatha Ravindran & Contreras-Vidal, "An empirical
#          comparison of deep learning explainability approaches for EEG using
#          simulated ground truth," Scientific Reports 13, 2023.
#          DOI: 10.1038/s41598-023-43871-8)
#
# License: BSD (3-clause)

"""Attribution utilities for braindecode models.

Two layers:

1. Pure-PyTorch primitives: :func:`saliency`, :func:`input_x_gradient`,
   :func:`integrated_gradients`, :func:`layer_grad_cam`. All implemented
   on top of :func:`torch.autograd.grad`; no optional dependencies.
2. Captum-backed wrappers: :func:`guided_backprop`,
   :func:`deconvolution`, :func:`deep_lift`, :func:`lrp`. These delegate
   to `captum <https://captum.ai>`_ for methods that aren't trivial to
   reimplement. Captum is a soft dependency; install with
   ``pip install braindecode[viz]``. Calling them without captum raises
   :class:`ImportError`.

Every function takes ``(model, x, target)`` and returns a tensor of
attributions with the same spatial shape as ``x``.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F

try:
    from captum import attr as _captum_attr
except ImportError:  # pragma: no cover, exercised only without captum
    _captum_attr = None

_CAPTUM_HINT = (
    "This attribution method requires captum. "
    "Install with `pip install braindecode[viz]`."
)


def _import_captum():
    """Return :mod:`captum.attr`, raising ImportError if captum is missing."""
    if _captum_attr is None:  # pragma: no cover, exercised only without captum
        raise ImportError(_CAPTUM_HINT)
    return _captum_attr


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
    grad = _input_grad(model, x, target)
    return grad.abs()


def input_x_gradient(model, x, target):
    """Element-wise input × input-gradient."""
    x_leaf = x.detach().clone().requires_grad_(True)
    grad = _input_grad(model, x_leaf, target)
    return x_leaf.detach() * grad


def integrated_gradients(model, x, target, baseline=None, steps=50):
    """Integrated Gradients (Sundararajan et al., 2017).

    Path integral of input gradients from ``baseline`` to ``x`` evaluated
    by the midpoint rule with ``steps`` evaluations. Default baseline is
    zeros. The midpoint rule is second-order accurate, so a moderate
    ``steps`` value already satisfies the completeness axiom
    (sum of attributions ≈ ``f(x) - f(baseline)``).
    """
    if baseline is None:
        baseline = torch.zeros_like(x)
    delta = x - baseline

    accumulated = torch.zeros_like(x)
    for k in range(steps):
        alpha = (k + 0.5) / steps
        xi = baseline + alpha * delta
        accumulated = accumulated + _input_grad(model, xi, target)
    return delta * (accumulated / steps)


def layer_grad_cam(model, x, target, layer):
    """Layer GradCAM: class-discriminative localization at ``layer``.

    Computes the gradient of the target output w.r.t. the layer's
    output activations, averages over spatial dims to obtain
    channel-wise weights, weights the activations, then bilinear-resamples
    the resulting map back to the input's last two dimensions.
    """
    captured = {}

    def _capture(_, __, output):
        captured["activation"] = output

    handle = layer.register_forward_hook(_capture)
    try:
        x_leaf = x.detach().clone().requires_grad_(True)
        output = model(x_leaf)
        target_score = output.gather(1, target.view(-1, 1)).sum()
        (grad,) = torch.autograd.grad(target_score, captured["activation"])
    finally:
        handle.remove()

    activation = captured["activation"]
    spatial_dims = tuple(range(2, activation.ndim))
    weights = grad.mean(dim=spatial_dims, keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return cam.squeeze(1)


def select_correctly_classified(model, X, y, device="cpu"):
    """Filter ``(X, y)`` down to samples the model classifies correctly.

    Accepts numpy arrays or tensors and returns tensors on ``device``.
    """
    model.eval().to(device)
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    correct = preds == y
    return X[correct], y[correct]


def _input_grad(model, x, target):
    """Gradient of the per-sample target output w.r.t. ``x``."""
    x_leaf = (
        x if x.requires_grad and x.is_leaf else x.detach().clone().requires_grad_(True)
    )
    output = model(x_leaf)
    target_score = output.gather(1, target.view(-1, 1)).sum()
    (grad,) = torch.autograd.grad(target_score, x_leaf)
    return grad


# Captum-backed wrappers.
#
# The functions below are thin wrappers around captum.attr classes for
# methods that aren't easily expressed in a few lines of autograd. They
# share the ``(model, x, target)`` signature of the primitives above and
# return :class:`torch.Tensor` attributions on the same device as ``x``.


def _captum_attribute(method_cls, model, x, target, **attribute_kwargs):
    captum_attr = _import_captum()
    if not hasattr(captum_attr, method_cls):
        raise AttributeError(
            f"captum.attr has no '{method_cls}'; the installed captum is too old."
        )
    algorithm = getattr(captum_attr, method_cls)(model)
    x_leaf = x.detach().clone().requires_grad_(True)
    return algorithm.attribute(x_leaf, target=target, **attribute_kwargs)


def guided_backprop(model, x, target):
    """GuidedBackprop attribution (Springenberg et al., 2014) via captum.

    Backpropagates through ReLUs keeping only positive gradients with
    positive activations, which often produces sharper input-space maps
    than vanilla saliency.
    """
    return _captum_attribute("GuidedBackprop", model, x, target)


def deconvolution(model, x, target):
    """DeconvNet-style attribution (Zeiler & Fergus, 2014) via captum."""
    return _captum_attribute("Deconvolution", model, x, target)


def deep_lift(model, x, target, baseline=None):
    """DeepLIFT attribution (Shrikumar et al., 2017) via captum.

    ``baseline`` defaults to zeros. Pass a tensor with the same shape as
    ``x`` to use a different reference (e.g. a per-channel mean).
    """
    if baseline is None:
        baseline = torch.zeros_like(x)
    return _captum_attribute("DeepLift", model, x, target, baselines=baseline)


def lrp(model, x, target):
    """Layer-wise Relevance Propagation (Bach et al., 2015) via captum.

    Captum picks default propagation rules; pass a configured ``LRP``
    instance directly if you need custom rules.
    """
    return _captum_attribute("LRP", model, x, target)


# Sanity-check helpers for attribution methods.
#
# Both helpers implement protocols from Adebayo et al. (2018) "Sanity
# Checks for Saliency Maps" (NeurIPS) as applied to EEG decoders by
# Sujatha Ravindran & Contreras-Vidal (Sci Rep 2023).
# ``cascading_layer_reset`` does the model-parameter randomization
# (reset layers progressively from output toward input);
# ``random_target`` is a small utility for label-randomization checks
# where the trained model is queried with wrong-class targets.


def random_target(target, n_classes, generator=None):
    """Return labels uniformly sampled from ``{0, ..., n_classes-1} \\ target``.

    For each entry of ``target`` pick a different class at random.
    Used in the label-randomization sanity check of
    Sujatha Ravindran & Contreras-Vidal (Sci Rep 2023): query the trained
    model's attribution method with the wrong target and check whether
    the resulting map differs from the correct-target map. Accepts a
    Python int, NumPy array, or torch tensor and returns the same kind
    of object on the same device.

    Parameters
    ----------
    target : int, numpy.ndarray, or torch.Tensor
        True class index/indices.
    n_classes : int
        Total number of classes; must be at least 2.
    generator : numpy.random.Generator, optional
        Source of randomness. Defaults to ``numpy.random.default_rng()``.

    Returns
    -------
    Same type as ``target`` (or ``int`` when ``target`` is a scalar).
    """
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2 to pick a different class.")
    rng = generator if generator is not None else np.random.default_rng()

    if isinstance(target, torch.Tensor):
        original_device = target.device
        flat = target.detach().cpu().numpy().ravel()
        out = _draw_random_targets(flat, n_classes, rng)
        return torch.as_tensor(out, dtype=target.dtype, device=original_device).reshape(
            target.shape
        )

    arr = np.asarray(target)
    if arr.ndim == 0:
        return int(_draw_random_targets(arr.reshape(1), n_classes, rng)[0])
    out = _draw_random_targets(arr.ravel(), n_classes, rng)
    return out.reshape(arr.shape).astype(arr.dtype, copy=False)


def _draw_random_targets(flat, n_classes, rng):
    out = rng.integers(low=0, high=n_classes - 1, size=flat.shape)
    out = np.where(out >= flat, out + 1, out)
    return out


def cascading_layer_reset(model, deepcopy_first=True):
    """Yield model copies with progressively-randomized parameters.

    Walks the modules in *reverse* depth-first order (output → input)
    and yields ``(layer_name, model_copy)`` after each layer's
    ``reset_parameters`` call. Implements the cascading-randomization
    sanity check from Adebayo et al. (NeurIPS 2018), as applied to EEG
    decoders in Sujatha Ravindran & Contreras-Vidal (Sci Rep 2023).
    An attribution method whose maps survive every cascade level is
    suspicious — its output likely depends on architecture rather than
    on learned weights.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model. The original model is not modified when
        ``deepcopy_first=True`` (default).
    deepcopy_first : bool
        If True, deep-copy the model before mutating, so the caller's
        ``model`` stays trained. Set False to skip the copy when memory
        is tight and the caller doesn't need the trained weights again.

    Yields
    ------
    layer_name : str
        Dotted-path name of the layer just reset (e.g. ``"final_layer.conv_classifier"``).
    randomized_model : torch.nn.Module
        A model whose layers from this one back to the output are
        randomly re-initialized.
    """
    target = copy.deepcopy(model) if deepcopy_first else model

    resettable = [
        (name, module)
        for name, module in target.named_modules()
        if hasattr(module, "reset_parameters")
        and callable(module.reset_parameters)
        and module is not target  # skip the root if it has reset_parameters
    ]

    for name, module in reversed(resettable):
        module.reset_parameters()
        yield name, target
