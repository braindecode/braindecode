import math

import torch
from torch import nn
from torch.nn.utils import parametrize


def glorot_weight_zero_bias(model):
    """Set the weights of BatchNorm layers to one and all biases to zero.

    Despite its name, this function does not apply the Glorot (Xavier)
    initialization: other weights keep the initialization of their layer
    (PyTorch's default, or Xavier uniform for the max-norm constrained layers
    such as :class:`braindecode.modules.Conv2dWithConstraint`). The name is
    kept for backward compatibility.

    Parameters
    ----------
    model : Module
    """
    for module in model.modules():
        # ``weight``/``bias`` can be None (e.g. BatchNorm with affine=False)
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and "BatchNorm" in type(module).__name__:
            nn.init.constant_(weight, 1)
        bias = getattr(module, "bias", None)
        if isinstance(bias, torch.Tensor):
            nn.init.constant_(bias, 0)


def rescale_parameter(param, layer_id):
    r"""Recaling the l-th transformer layer.

    Rescales the parameter tensor by the inverse square root of the layer id.
    Made inplace. :math:`\frac{1}{\sqrt{2 \cdot \text{layer\_id}}}` [Beit2022]

    In the labram, this is used to rescale the output matrices
    (i.e., the last linear projection within each sub-layer) of the
    self-attention module.

    Parameters
    ----------
    param: :class:`torch.Tensor`
        tensor to be rescaled
    layer_id: int
        layer id in the neural network

    References
    ----------
    [Beit2022] Hangbo Bao, Li Dong, Songhao Piao, Furu We (2022). BEIT: BERT
    Pre-Training of Image Transformers.
    """
    param.div_(math.sqrt(2.0 * layer_id))


def init_weights_(
    module,
    weights=None,
    norms=(nn.LayerNorm,),
    init_parametrized_original=False,
):
    """Initialize one module by layer type; apply to a model with ``model.apply``.

    The first entry of ``weights`` whose type matches ``module`` initializes its
    ``weight`` with the given function, and its ``bias`` (if any) is set to zero.
    Modules of a type in ``norms`` get a unit weight and a zero bias. Nothing
    else is touched. The default is the timm/ViT scheme (truncated-normal
    ``nn.Linear`` weights with std 0.02, unit/zero ``nn.LayerNorm``); models pass
    their own mapping to reproduce their reference initialization exactly.

    Parameters
    ----------
    module : torch.nn.Module
    weights : dict or None
        ``{layer type or tuple of types: init function}``, e.g.
        ``{nn.Linear: nn.init.xavier_uniform_, nn.Embedding:
        partial(nn.init.normal_, std=0.02)}``. Order matters as in an
        ``if``/``elif`` chain. ``None`` uses
        ``{nn.Linear: partial(nn.init.trunc_normal_, std=0.02)}``.
    norms : tuple of type
        Normalization layer types set to weight 1 and bias 0 (missing affine
        parameters are skipped). ``()`` leaves normalization layers alone.
    init_parametrized_original : bool
        For a layer whose ``weight`` is parametrized (e.g. max-norm), initialize
        the unconstrained ``parametrizations.weight.original`` instead of the
        computed ``weight``.

    Examples
    --------
    >>> from functools import partial
    >>> from torch import nn
    >>> model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    >>> _ = model.apply(init_weights_)  # timm/ViT default
    >>> _ = model.apply(partial(init_weights_, weights={nn.Linear: nn.init.xavier_uniform_}))
    """
    if weights is None:
        weights = {nn.Linear: _trunc_normal_002}
    for types, init in weights.items():
        if isinstance(module, types):
            weight = module.weight
            if init_parametrized_original and parametrize.is_parametrized(
                module, "weight"
            ):
                weight = module.parametrizations.weight.original
            init(weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
            return
    if norms and isinstance(module, norms):
        if getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


def _trunc_normal_002(tensor):
    return nn.init.trunc_normal_(tensor, std=0.02)
