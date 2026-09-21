# Authors: Vandit Shah <shahvanditt@gmail.com>
#
# License: BSD (3-clause)

"""Read and replace intermediate module outputs with temporary hooks."""

from torch import nn


def capture_activations(model: nn.Module, x, layer: nn.Module):
    """Return the output emitted by ``layer`` during ``model(x)``.

    Nested outputs are returned unchanged. If a layer runs more than once,
    only its last output is kept.

    Parameters
    ----------
    model : torch.nn.Module
        Model to run.
    x : torch.Tensor
        Model input.
    layer : torch.nn.Module
        Submodule whose output is captured.

    Raises
    ------
    RuntimeError
        If ``layer`` is not called by the forward pass.
    """
    captured: list[object] = []

    def hook(_module, _inputs, output):
        captured[:] = [output]

    with layer.register_forward_hook(hook):
        model(x)
    if not captured:
        raise RuntimeError("layer was not called during the forward pass")
    return captured[0]


def run_with_activation_substitution(
    model: nn.Module, x, layer: nn.Module, substitute_fn
):
    """Run ``model(x)`` after replacing ``layer``'s output.

    Parameters
    ----------
    model : torch.nn.Module
        Model to run.
    x : torch.Tensor
        Model input.
    layer : torch.nn.Module
        Submodule whose output is replaced.
    substitute_fn : callable
        Called with the layer output and must return its replacement.

    Returns
    -------
    Any
        The model output after substitution.

    Raises
    ------
    TypeError
        If ``substitute_fn`` returns ``None``, which PyTorch interprets as
        keeping the original output.
    """

    def hook(_module, _inputs, output):
        replacement = substitute_fn(output)
        if replacement is None:
            raise TypeError("substitute_fn must return a replacement")
        return replacement

    with layer.register_forward_hook(hook):
        return model(x)
