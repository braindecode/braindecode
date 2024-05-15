from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import EEGModuleMixin


class DenseLayer(nn.Sequential):
    """
    A densely connected layer with batch normalization and dropout.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    growth_rate : int
        Rate of growth of channels in this layer.
    bn_size : int
        Multiplicative factor for the bottleneck layer (does not affect the output size).
    drop_rate : float, optional
        Dropout rate. Default is 0.5.
    conv_bias : bool, optional
        Whether to use bias in convolutional layers. Default is True.
    batch_norm : bool, optional
        Whether to use batch normalization. Default is True.

    Examples
    --------
    >>> x = torch.randn(128, 5, 1000)
    >>> batch, channels, length = x.shape
    >>> model = DenseLayer(channels, 5, 2)
    >>> y = model(x)
    >>> y.shape
    torch.Size([128, 10, 1000])
    """

    def __init__(
        self,
        input_channels: int,
        growth_rate: int,
        bn_size: int,
        drop_rate=0.5,
        conv_bias=True,
        batch_norm=True,
    ):
        super(DenseLayer, self).__init__()
        if batch_norm:
            (self.add_module("norm1", nn.BatchNorm1d(input_channels)),)
        (self.add_module("elu1", nn.ELU()),)
        (
            self.add_module(
                "conv1",
                nn.Conv1d(
                    input_channels,
                    bn_size * growth_rate,
                    kernel_size=1,
                    stride=1,
                    bias=conv_bias,
                ),
            ),
        )
        if batch_norm:
            (self.add_module("norm2", nn.BatchNorm1d(bn_size * growth_rate)),)
        (self.add_module("elu2", nn.ELU()),)
        (
            self.add_module(
                "conv2",
                nn.Conv1d(
                    bn_size * growth_rate,
                    growth_rate,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=conv_bias,
                ),
            ),
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Sequential):
    """
    A densely connected block that uses DenseLayers.

    Parameters
    ----------
    num_layers : int
        Number of layers in this block.
    input_channels : int
        Number of input channels.
    growth_rate : int
        Rate of growth of channels in this layer.
    bn_size : int
        Multiplicative factor for the bottleneck layer (does not affect the output size).
    drop_rate : float, optional
        Dropout rate. Default is 0.5.
    conv_bias : bool, optional
        Whether to use bias in convolutional layers. Default is True.
    batch_norm : bool, optional
        Whether to use batch normalization. Default is True.

    Examples
    --------
    >>> x = torch.randn(128, 5, 1000)
    >>> batch, channels, length = x.shape
    >>> model = DenseBlock(3, channels, 5, 2)
    >>> y = model(x)
    >>> y.shape
    torch.Size([128, 20, 1000])
    """

    def __init__(
        self,
        num_layers,
        input_channels,
        growth_rate,
        bn_size,
        drop_rate=0.5,
        conv_bias=True,
        batch_norm=True,
    ):
        super(DenseBlock, self).__init__()
        for idx_layer in range(num_layers):
            layer = DenseLayer(
                input_channels + idx_layer * growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
                conv_bias,
                batch_norm,
            )
            self.add_module(f"denselayer{idx_layer + 1}", layer)


class TransitionLayer(nn.Sequential):
    """
    A pooling transition layer.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.
    conv_bias : bool, optional
        Whether to use bias in convolutional layers. Default is True.
    batch_norm : bool, optional
        Whether to use batch normalization. Default is True.

    Examples
    --------
    >>> x = torch.randn(128, 5, 1000)
    >>> model = TransitionLayer(5, 18)
    >>> y = model(x)
    >>> y.shape
    torch.Size([128, 18, 500])
    """

    def __init__(
        self, input_channels, output_channels, conv_bias=True, batch_norm=True
    ):
        super(TransitionLayer, self).__init__()
        if batch_norm:
            self.add_module("norm", nn.BatchNorm1d(input_channels))
        self.add_module("elu", nn.ELU())
        self.add_module(
            "conv",
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=1,
                bias=conv_bias,
            ),
        )
        self.add_module("pool", nn.AvgPool1d(kernel_size=2, stride=2))


class SPARCNet(EEGModuleMixin, nn.Module):
    """Seizures, Periodic and Rhythmic pattern Continuum Neural Network (SPaRCNet) [jing2023]_.

    This is a temporal CNN model for biosignal classification based on the DenseNet
    architecture.

    The model is based on the unofficial implementation [Code2023]_.

    .. versionadded:: 0.9

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors.

    Parameters
    ----------
    block_layers : int, optional
        Number of layers per dense block. Default is 4.
    growth_rate : int, optional
        Growth rate of the DenseNet. Default is 16.
    bn_size : int, optional
        Bottleneck size. Default is 16.
    drop_rate : float, optional
        Dropout rate. Default is 0.5.
    conv_bias : bool, optional
        Whether to use bias in convolutional layers. Default is True.
    batch_norm : bool, optional
        Whether to use batch normalization. Default is True.

    References
    ----------
    .. [jing2023] Jing, J., Ge, W., Hong, S., Fernandes, M. B., Lin, Z.,
       Yang, C., ... & Westover, M. B. (2023). Development of expert-level
       classification of seizures and rhythmic and periodic
       patterns during eeg interpretation. Neurology, 100(17), e1750-e1762.
    .. [Code2023] Yang, C., Westover, M.B. and Sun, J., 2023. BIOT
       Biosignal Transformer for Cross-data Learning in the Wild.
       GitHub https://github.com/ycq091044/BIOT (accessed 2024-02-13)

    """

    def __init__(
        self,
        n_chans: int = None,
        n_times: int = None,
        n_outputs: int = None,
        # Neural network parameters
        block_layers: int = 4,
        growth_rate: int = 16,
        bn_size: int = 16,
        drop_rate: float = 0.5,
        conv_bias: bool = True,
        batch_norm: bool = True,
        # EEGModuleMixin parameters
        # (another way to present the same parameters)
        chs_info: dict = None,
        input_window_seconds: [float | int] = None,
        sfreq: int = None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, sfreq, input_window_seconds

        # add initial convolutional layer
        # the number of output channels is the smallest power of 2
        # that is greater than the number of input channels
        out_channels = 2 ** (math.floor(np.log2(self.n_chans)) + 1)
        first_conv = OrderedDict(
            [
                (
                    "conv0",
                    nn.Conv1d(
                        in_channels=self.n_chans,
                        out_channels=out_channels,
                        kernel_size=7,
                        stride=2,
                        padding=3,
                        bias=conv_bias,
                    ),
                )
            ]
        )
        first_conv["norm0"] = nn.BatchNorm1d(out_channels)
        first_conv["act_layer"] = nn.ELU()
        first_conv["pool0"] = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.encoder = nn.Sequential(first_conv)

        n_channels = out_channels

        # Adding dense blocks
        for n_layer in np.arange(math.floor(np.log2(self.n_times // 4))):
            block = DenseBlock(
                num_layers=block_layers,
                input_channels=n_channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
            )
            self.encoder.add_module("denseblock%d" % (n_layer + 1), block)
            # update the number of channels after each dense block
            n_channels = n_channels + block_layers * growth_rate

            trans = TransitionLayer(
                input_channels=n_channels,
                output_channels=n_channels // 2,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
            )
            self.encoder.add_module("transition%d" % (n_layer + 1), trans)
            # update the number of channels after each transition layer
            n_channels = n_channels // 2

        # add final convolutional layer
        self.final_layer = nn.Sequential(
            nn.ELU(),
            nn.Linear(n_channels, self.n_outputs),
        )

        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the model.

        Official init from torch repo, using kaiming_normal for conv layers
        and normal for linear layers.

        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, X: torch.Tensor):
        """
        Forward pass of the model.

        Parameters
        ----------
        X: torch.Tensor
            The input tensor of the model with shape (batch_size, n_channels, n_times)

        Returns
        -------
        torch.Tensor
            The output tensor of the model with shape (batch_size, n_outputs)
        """
        emb = self.encoder(X).squeeze(-1)
        out = self.final_layer(emb)
        return out
