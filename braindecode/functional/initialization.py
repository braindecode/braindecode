import math

import torch
from torch import nn
from torch.nn.utils import parametrize


def glorot_weight_zero_bias(model):
    """Initialize parameters of all modules by initializing weights with.

    glorot uniform/xavier initialization, and setting biases to zero. Weights from
    batch norm layers are set to 1.

    Parameters
    ----------
    model : Module
    """
    for module in model.modules():
        # ``weight``/``bias`` can be None (e.g. BatchNorm with affine=False)
        # or, on a parametrized module's ``parametrizations``, not a tensor
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            if "BatchNorm" in module.__class__.__name__:
                nn.init.constant_(weight, 1)
            elif weight.dim() >= 2:
                if parametrize.is_parametrized(module, "weight"):
                    # e.g. max-norm constrained layers: initialize the
                    # unconstrained tensor the parametrization reads
                    weight = module.parametrizations.weight.original
                nn.init.xavier_uniform_(weight, gain=1)
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
