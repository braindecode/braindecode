import torch
from torch import nn


class InceptionBlock(nn.Module):
    """
    Inception block module.

    This module applies multiple convolutional branches to the input and concatenates
    their outputs along the channel dimension. Each branch can have a different
    configuration, allowing the model to capture multi-scale features.

    Parameters
    ----------
    branches : list of nn.Module
        List of convolutional branches to apply to the input.
    """

    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], 1)


class MLP(nn.Sequential):
    r"""Multilayer Perceptron (MLP) with GELU activation and optional dropout.

    Also known as fully connected feedforward network, an MLP is a sequence of
    non-linear parametric functions

    .. math:: h_{i + 1} = a_{i + 1}(h_i W_{i + 1}^T + b_{i + 1}),

    over feature vectors :math:`h_i`, with the input and output feature vectors
    :math:`x = h_0` and :math:`y = h_L`, respectively. The non-linear functions
    :math:`a_i` are called activation functions. The trainable parameters of an
    MLP are its weights and biases :math:`\\phi = \{W_i, b_i | i = 1, \dots, L\}`.

    Parameters:
    -----------
    in_features: int
        Number of input features.
    hidden_features: Sequential[int] (default=None)
        Number of hidden features, if None, set to in_features.
        You can increase the size of MLP just passing more int in the
        hidden features vector. The model size increase follow the
        rule 2n (hidden layers)+2 (in and out layers)
    out_features: int (default=None)
        Number of output features, if None, set to in_features.
    act_layer: nn.GELU (default)
        The activation function constructor. If :py:`None`, use
        :class:`torch.nn.GELU` instead.
    drop: float (default=0.0)
        Dropout rate.
    normalize: bool (default=False)
        Whether to apply layer normalization.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features=None,
        out_features=None,
        activation=nn.GELU,
        drop=0.0,
        normalize=False,
    ):
        self.normalization = nn.LayerNorm if normalize else lambda: None
        self.in_features = in_features
        self.out_features = out_features or self.in_features
        if hidden_features:
            self.hidden_features = hidden_features
        else:
            self.hidden_features = (self.in_features, self.in_features)
        self.activation = activation

        layers = []

        for before, after in zip(
            (self.in_features, *self.hidden_features),
            (*self.hidden_features, self.out_features),
        ):
            layers.extend(
                [
                    nn.Linear(in_features=before, out_features=after),
                    self.activation(),
                    self.normalization(),
                ]
            )

        layers = layers[:-2]
        layers.append(nn.Dropout(p=drop))

        # Cleaning if we are not using the normalization layer
        layers = list(filter(lambda layer: layer is not None, layers))

        super().__init__(*layers)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p, activation: nn.Module = nn.GELU):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            activation(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
