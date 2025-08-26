from __future__ import annotations

from collections import OrderedDict
from math import floor, log2

import torch
import torch.nn as nn
import torch.nn.functional as F

from braindecode.models.base import EEGModuleMixin


class SPARCNet(EEGModuleMixin, nn.Module):
    """Seizures, Periodic and Rhythmic pattern Continuum Neural Network (SPaRCNet) from Jing et al. (2023) [jing2023]_.

    :bdg-success:`Convolution`

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
    drop_prob : float, optional
        Dropout rate. Default is 0.5.
    conv_bias : bool, optional
        Whether to use bias in convolutional layers. Default is True.
    batch_norm : bool, optional
        Whether to use batch normalization. Default is True.
    activation: nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.

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
        n_chans=None,
        n_times=None,
        n_outputs=None,
        # Neural network parameters
        block_layers: int = 4,
        growth_rate: int = 16,
        bottleneck_size: int = 16,
        drop_prob: float = 0.5,
        conv_bias: bool = True,
        batch_norm: bool = True,
        activation: nn.Module = nn.ELU,
        kernel_size_conv0: int = 7,
        kernel_size_conv1: int = 1,
        kernel_size_conv2: int = 3,
        kernel_size_pool: int = 3,
        stride_pool: int = 2,
        stride_conv0: int = 2,
        stride_conv1: int = 1,
        stride_conv2: int = 1,
        padding_pool: int = 1,
        padding_conv0: int = 3,
        padding_conv2: int = 1,
        kernel_size_trans: int = 2,
        stride_trans: int = 2,
        # EEGModuleMixin parameters
        # (another way to present the same parameters)
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
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
        out_channels = 2 ** (floor(log2(self.n_chans)) + 1)
        first_conv = OrderedDict(
            [
                (
                    "conv0",
                    nn.Conv1d(
                        in_channels=self.n_chans,
                        out_channels=out_channels,
                        kernel_size=kernel_size_conv0,
                        stride=stride_conv0,
                        padding=padding_conv0,
                        bias=conv_bias,
                    ),
                )
            ]
        )
        first_conv["norm0"] = nn.BatchNorm1d(out_channels)
        first_conv["act_layer"] = activation()
        first_conv["pool0"] = nn.MaxPool1d(
            kernel_size=kernel_size_pool,
            stride=stride_pool,
            padding=padding_pool,
        )

        self.encoder = nn.Sequential(first_conv)

        n_channels = out_channels

        # Adding dense blocks
        for n_layer in range(floor(log2(self.n_times // 4))):
            block = _DenseBlock(
                num_layers=block_layers,
                in_channels=n_channels,
                growth_rate=growth_rate,
                bottleneck_size=bottleneck_size,
                drop_prob=drop_prob,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
                activation=activation,
                kernel_size_conv1=kernel_size_conv1,
                kernel_size_conv2=kernel_size_conv2,
                stride_conv1=stride_conv1,
                stride_conv2=stride_conv2,
                padding_conv2=padding_conv2,
            )
            self.encoder.add_module("denseblock%d" % (n_layer + 1), block)
            # update the number of channels after each dense block
            n_channels = n_channels + block_layers * growth_rate

            trans = _TransitionLayer(
                in_channels=n_channels,
                out_channels=n_channels // 2,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
                activation=activation,
                kernel_size_trans=kernel_size_trans,
                stride_trans=stride_trans,
            )
            self.encoder.add_module("transition%d" % (n_layer + 1), trans)
            # update the number of channels after each transition layer
            n_channels = n_channels // 2

        self.adaptative_pool = nn.AdaptiveAvgPool1d(1)
        self.activation_layer = activation()
        self.flatten_layer = nn.Flatten()

        # add final convolutional layer
        self.final_layer = nn.Linear(n_channels, self.n_outputs)

        self._init_weights()

    def _init_weights(self):
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
        emb = self.encoder(X)
        emb = self.adaptative_pool(emb)
        emb = self.activation_layer(emb)
        emb = self.flatten_layer(emb)
        out = self.final_layer(emb)
        return out


class _DenseLayer(nn.Sequential):
    """
    A densely connected layer with batch normalization and dropout.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    growth_rate : int
        Rate of growth of channels in this layer.
    bottleneck_size : int
        Multiplicative factor for the bottleneck layer (does not affect the output size).
    drop_prob : float, optional
        Dropout rate. Default is 0.5.
    conv_bias : bool, optional
        Whether to use bias in convolutional layers. Default is True.
    batch_norm : bool, optional
        Whether to use batch normalization. Default is True.
    activation: nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.

    Examples
    --------
    >>> x = torch.randn(128, 5, 1000)
    >>> batch, channels, length = x.shape
    >>> model = _DenseLayer(channels, 5, 2)
    >>> y = model(x)
    >>> y.shape
    torch.Size([128, 10, 1000])
    """

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bottleneck_size: int,
        drop_prob: float = 0.5,
        conv_bias: bool = True,
        batch_norm: bool = True,
        activation: nn.Module = nn.ELU,
        kernel_size_conv1: int = 1,
        kernel_size_conv2: int = 3,
        stride_conv1: int = 1,
        stride_conv2: int = 1,
        padding_conv2: int = 1,
    ):
        super().__init__()
        if batch_norm:
            self.add_module("norm1", nn.BatchNorm1d(in_channels))

        self.add_module("elu1", activation())
        self.add_module(
            "conv1",
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_size * growth_rate,
                kernel_size=kernel_size_conv1,
                stride=stride_conv1,
                bias=conv_bias,
            ),
        )
        if batch_norm:
            self.add_module("norm2", nn.BatchNorm1d(bottleneck_size * growth_rate))
        self.add_module("elu2", activation())
        self.add_module(
            "conv2",
            nn.Conv1d(
                in_channels=bottleneck_size * growth_rate,
                out_channels=growth_rate,
                kernel_size=kernel_size_conv2,
                stride=stride_conv2,
                padding=padding_conv2,
                bias=conv_bias,
            ),
        )
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Manually pass through each submodule
        out = x
        for layer in self:
            out = layer(out)
        # apply dropout using the functional API
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        # concatenate input and new features
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Sequential):
    """
    A densely connected block that uses DenseLayers.

    Parameters
    ----------
    num_layers : int
        Number of layers in this block.
    in_channels : int
        Number of input channels.
    growth_rate : int
        Rate of growth of channels in this layer.
    bottleneck_size : int
        Multiplicative factor for the bottleneck layer (does not affect the output size).
    drop_prob : float, optional
        Dropout rate. Default is 0.5.
    conv_bias : bool, optional
        Whether to use bias in convolutional layers. Default is True.
    batch_norm : bool, optional
        Whether to use batch normalization. Default is True.
    activation: nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.

    Examples
    --------
    >>> x = torch.randn(128, 5, 1000)
    >>> batch, channels, length = x.shape
    >>> model = _DenseBlock(3, channels, 5, 2)
    >>> y = model(x)
    >>> y.shape
    torch.Size([128, 20, 1000])
    """

    def __init__(
        self,
        num_layers,
        in_channels,
        growth_rate,
        bottleneck_size,
        drop_prob=0.5,
        conv_bias=True,
        batch_norm=True,
        activation: nn.Module = nn.ELU,
        kernel_size_conv1: int = 1,
        kernel_size_conv2: int = 3,
        stride_conv1: int = 1,
        stride_conv2: int = 1,
        padding_conv2: int = 1,
    ):
        super(_DenseBlock, self).__init__()
        for idx_layer in range(num_layers):
            layer = _DenseLayer(
                in_channels=in_channels + idx_layer * growth_rate,
                growth_rate=growth_rate,
                bottleneck_size=bottleneck_size,
                drop_prob=drop_prob,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
                activation=activation,
                kernel_size_conv1=kernel_size_conv1,
                kernel_size_conv2=kernel_size_conv2,
                stride_conv1=stride_conv1,
                stride_conv2=stride_conv2,
                padding_conv2=padding_conv2,
            )
            self.add_module(f"denselayer{idx_layer + 1}", layer)


class _TransitionLayer(nn.Sequential):
    """
    A pooling transition layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    conv_bias : bool, optional
        Whether to use bias in convolutional layers. Default is True.
    batch_norm : bool, optional
        Whether to use batch normalization. Default is True.
    activation: nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.

    Examples
    --------
    >>> x = torch.randn(128, 5, 1000)
    >>> model = _TransitionLayer(5, 18)
    >>> y = model(x)
    >>> y.shape
    torch.Size([128, 18, 500])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_bias=True,
        batch_norm=True,
        activation: nn.Module = nn.ELU,
        kernel_size_trans: int = 2,
        stride_trans: int = 2,
    ):
        super(_TransitionLayer, self).__init__()
        if batch_norm:
            self.add_module("norm", nn.BatchNorm1d(in_channels))
        self.add_module("elu", activation())
        self.add_module(
            "conv",
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=conv_bias,
            ),
        )
        self.add_module(
            "pool", nn.AvgPool1d(kernel_size=kernel_size_trans, stride=stride_trans)
        )
