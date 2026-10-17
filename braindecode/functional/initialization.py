import math

import torch
from torch import nn


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


def trunc_normal_init_(module, std=0.02, layers=(nn.Linear,)):
    """Initialize ``module`` as in timm/ViT: weights of ``layers`` from
    :func:`torch.nn.init.trunc_normal_` (default bounds), zero biases, and unit weight / zero bias for LayerNorm and
    BatchNorm. Apply to a whole model with ``model.apply(trunc_normal_init_)``.

    Parameters
    ----------
    module : torch.nn.Module
    std : float
        Standard deviation of the truncated normal.
    layers : tuple of type
        Layer types whose ``weight`` is drawn from the truncated normal.
    """
    if isinstance(module, layers):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.modules.batchnorm._BatchNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
