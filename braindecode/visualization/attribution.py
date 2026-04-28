# Authors: Vandit Shah <vanditshah@outlook.com>
#
# License: BSD (3-clause)

"""Attribution utilities built on plain PyTorch autograd.

Functions take ``(model, x, target)`` and return a tensor of attributions
the same spatial shape as ``x`` (after interpolation for layer-CAM). All
heavy lifting goes through :func:`torch.autograd.grad` so braindecode
takes no extra optional dependencies.
"""

import torch
import torch.nn.functional as F


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
    cam = F.interpolate(
        cam, size=x.shape[-2:], mode="bilinear", align_corners=False
    )
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
    x_leaf = x if x.requires_grad and x.is_leaf else x.detach().clone().requires_grad_(True)
    output = model(x_leaf)
    target_score = output.gather(1, target.view(-1, 1)).sum()
    (grad,) = torch.autograd.grad(target_score, x_leaf)
    return grad
