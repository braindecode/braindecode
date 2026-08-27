# Authors: Vandit Shah <shahvanditt@gmail.com>
#
# License: BSD (3-clause)

"""Read and replace the intermediate activations of a trained model.

Two primitives built on :meth:`torch.nn.Module.register_forward_hook`:
:func:`capture_activations` records what a submodule emitted during a
forward pass, and :func:`run_with_activation_substitution` replaces that
output so the rest of the model runs on the substitute. Hooks are removed
in a ``finally`` block, so a raising forward pass cannot leave them
attached.

Notes
-----
The hook lifecycle follows the ``register_forward_hook`` contract in the
PyTorch documentation: a capture hook returns ``None`` and keeps the
module's own output, while a hook that returns a value replaces it. These
functions are written against that public API and are not derived from any
third-party implementation.
"""

from collections.abc import Mapping
from copy import deepcopy

import torch
from torch import nn


def _detach_output(output):
    """Detach every tensor in a possibly nested module output.

    Blocks commonly return a tuple or dict rather than a bare tensor, so
    the structure is walked rather than assumed.
    """
    if torch.is_tensor(output):
        return output.detach()
    if isinstance(output, tuple):
        return tuple(_detach_output(item) for item in output)
    if isinstance(output, list):
        return [_detach_output(item) for item in output]
    if isinstance(output, dict):
        return {key: _detach_output(value) for key, value in output.items()}
    return deepcopy(output)


def _as_module_items(layers):
    """Return ``(key, module)`` pairs from either a mapping or a sequence."""
    if isinstance(layers, Mapping):
        return list(layers.items())
    return list(enumerate(layers))


def capture_activations(model, x, layers, forward_fn=None, detach=True):
    """Capture the outputs of one or more modules during a forward pass.

    Parameters
    ----------
    model : torch.nn.Module
        The model to run.
    x : torch.Tensor
        Input passed to ``forward_fn``.
    layers : torch.nn.Module or list of torch.nn.Module or dict
        Modules whose outputs to capture. A single module is keyed ``0``, a
        sequence is keyed by position, and a mapping keeps its own keys.
    forward_fn : callable or None, default=None
        Callable invoked as ``forward_fn(x)``. Defaults to ``model``. Use
        this to reach a submodule's forward, for instance a backbone's
        feature extractor rather than the full classifier.
    detach : bool, default=True
        Detach captured tensors from the autograd graph. Set to ``False``
        only when gradients through the captured activations are needed.

    Returns
    -------
    dict
        One entry per requested module, keyed as described above. A module
        returning a tuple or dict is stored with that structure intact.

    Raises
    ------
    RuntimeError
        If a requested module was never called during the forward pass.
        Returning a short dictionary instead would push the failure to
        whichever line first indexes the missing key, long after the cause.

    See Also
    --------
    run_with_activation_substitution : Replace a layer's output instead.

    Notes
    -----
    A module called more than once in a single forward pass, as with weight
    sharing or a loop over a shared block, keeps only its last output.

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> from braindecode.visualization import capture_activations
    >>> model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
    >>> captured = capture_activations(model, torch.randn(3, 4), [model[0]])
    >>> captured[0].shape
    torch.Size([3, 6])
    """
    forward_fn = model if forward_fn is None else forward_fn
    if isinstance(layers, nn.Module):
        items = [(0, layers)]
    else:
        items = _as_module_items(layers)
    captured = {}
    handles = []

    def make_hook(key):
        """Build a forward hook that stores its module's output under ``key``."""

        def hook(_module, _inputs, output):
            """Store this module's output, detaching it when asked."""
            captured[key] = _detach_output(output) if detach else output

        return hook

    for key, module in items:
        handles.append(module.register_forward_hook(make_hook(key)))

    try:
        forward_fn(x)
    finally:
        for handle in handles:
            handle.remove()

    missing = [key for key, _ in items if key not in captured]
    if missing:
        raise RuntimeError(
            f"these modules were never called during the forward pass: {missing}. "
            "Check that they belong to the model reached by forward_fn and that "
            "the input takes the branch containing them."
        )

    return captured


def run_with_activation_substitution(model, x, layer, substitute_fn, forward_fn=None):
    """Run ``model`` with the output of ``layer`` replaced.

    Comparing a metric before and after the substitution measures how much
    the rest of the model depends on what was replaced.

    Parameters
    ----------
    model : torch.nn.Module
        The model to run.
    x : torch.Tensor
        Input passed to ``forward_fn``.
    layer : torch.nn.Module
        Module whose output is replaced.
    substitute_fn : callable
        Called with the layer's output; its return value is used instead.
        It receives whatever the module emitted, so a module returning a
        tuple must be handled as a tuple and returned as one.
    forward_fn : callable or None, default=None
        Callable invoked as ``forward_fn(x)``. Defaults to ``model``.

    Returns
    -------
    Any
        Whatever ``forward_fn`` returns, computed with the substitution in
        place.

    Raises
    ------
    TypeError
        If ``substitute_fn`` returns ``None``. A forward hook returning
        ``None`` leaves the module's own output in place, so a
        ``substitute_fn`` missing its ``return`` would otherwise run to
        completion and report an unchanged metric as though the
        substitution had happened.

    See Also
    --------
    capture_activations : Record a layer's output instead of replacing it.

    Examples
    --------
    Zeroing a block measures how much the head relies on it:

    >>> out = run_with_activation_substitution(
    ...     model, torch.randn(3, 4), model[0], torch.zeros_like
    ... )  # doctest: +SKIP
    """
    forward_fn = model if forward_fn is None else forward_fn

    def hook(_module, _inputs, output):
        """Return the substituted output in place of the module's own."""
        replacement = substitute_fn(output)
        if replacement is None:
            raise TypeError(
                "substitute_fn returned None, which torch reads as "
                "'keep the original output'. Return the replacement instead."
            )
        return replacement

    handle = layer.register_forward_hook(hook)
    try:
        return forward_fn(x)
    finally:
        handle.remove()
