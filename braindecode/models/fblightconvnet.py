from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from mne.utils import warn
from torch import nn, Tensor

from braindecode.models.base import EEGModuleMixin

from braindecode.models.modules import (
    FilterBankLayer,
    LogVarLayer,
)


class FBLightConvNet(EEGModuleMixin, nn.Module):
    """LightConvNet from Ma, X et al (2023) [lightconvnet]_.

        .. figure:: https://raw.githubusercontent.com/Ma-Xinzhi/LightConvNet/refs/heads/main/network_architecture.png
            :align: center
            :alt: LightConvNet



    Notes
    -----
    This implementation is not guaranteed to be correct and has not been checked
    by the original authors; it is a braindecode adaptation from the Pytorch
    source-code [lightconvnetcode]_.


    Parameters
    ----------
    n_bands : int or None or List[Tuple[int, int]]], default=9
        Number of frequency bands. Could
    n_filters_spat : int, default=32
        The depth of the depthwise convolutional layer.
    n_dim: int, default=3
        Number of dimensions for the temporal reductor

    stride_factor : int, default=4
        Stride factor for reshaping.
    activation : nn.Module, default=nn.ELU
        Activation function class to apply.
    verbose: bool, default False
        Verbose parameter to create the filter using mne
    filter_parameters: dict, default {}
        Parameters for the FilterBankLayer

    References
    ----------
    .. [lightconvnet] Ma, X., Chen, W., Pei, Z., Liu, J., Huang, B., & Chen, J.
        (2023). A temporal dependency learning CNN with attention mechanism
        for MI-EEG decoding. IEEE Transactions on Neural Systems and
        Rehabilitation Engineering.
    .. [lightconvnetcode] Link to source-code:
        https://github.com/Ma-Xinzhi/LightConvNet
    """

    def __init__(
        self,
        # Braindecode parameters
        n_chans=None,
        n_outputs=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        # models parameters
        n_bands=8,
        n_filters_spat: int = 32,  # It will be to the embedding space
        n_dim: int = 3,
        stride_factor: int = 4,
        # In the original code they perform the number of points (250),
        # I think a factor is a little better, but we can discuss.
        heads: int = 8,
        activation: nn.Module = nn.ELU,
        verbose: bool = False,
        filter_parameters: dict = {},
        # Model parameters
        weight_softmax=True,
        bias=False,
    ):
        super().__init__(
            n_chans=n_chans,
            n_outputs=n_outputs,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        # Parameters
        self.n_bands = n_bands
        self.n_filters_spat = n_filters_spat
        self.n_dim = n_dim
        self.stride_factor = stride_factor
        self.activation = activation
        self.filter_parameters = filter_parameters
        self.heads = heads
        self.weight_softmax = weight_softmax
        self.bias = bias

        # Checkers
        need_padding = False
        if self.n_times % self.stride_factor != 0:
            warn(
                f"Time dimension ({self.n_times}) is not divisible by"
                f" stride_factor ({self.stride_factor}). Input will be "
                f"truncated.",
                UserWarning,
            )
            need_padding = True

        # Layers
        # Following paper nomeclature
        self.spectral_filtering = FilterBankLayer(
            n_chans=self.n_chans,
            sfreq=self.sfreq,
            band_filters=self.n_bands,
            verbose=verbose,
            **filter_parameters,
        )
        # As we have an internal process to create the bands,
        # we get the values from the filterbank
        self.n_bands = self.spectral_filtering.n_bands

        # The convolution here is different.
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.n_bands,
                out_channels=self.n_filters_spat,
                kernel_size=(self.n_chans, 1),
            ),
            nn.BatchNorm2d(self.n_filters_spat),
            self.activation(),
        )

        if need_padding:
            self.padding_size = stride_factor - (self.n_times % stride_factor)
            self.n_times_padded = self.n_times + self.padding_size
            self.padding_layer = partial(
                self._apply_padding,
                padding_size=self.padding_size,
            )
        else:
            self.padding_layer = nn.Identity()
            self.n_times_padded = self.n_times

        # Temporal aggregator
        self.temporal_layer = LogVarLayer(dim=self.n_dim, keepdim=False)

        self.flatten_layer = Rearrange("batch ... -> batch (...)")

        # LightWeightConv1D
        self.conv = _LightweightConv1d(
            self.n_filters_spat,
            self.stride_factor,
            heads=self.heads,
            weight_softmax=weight_softmax,
            bias=bias,
        )

        self.final_layer = nn.Linear(
            in_features=self.n_filters_spat,
            out_features=self.n_outputs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FBCNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, n_outputs).
        """
        batch_size, _, _ = x.shape

        # batch, n_chans, n_times
        x = self.spectral_filtering(x)
        # batch, nbands, n_chans, n_times
        x = self.spatial_conv(x)

        x = x.view(
            batch_size,
            self.n_chans,
            self.stride_factor,
            self.n_times_padded // self.stride_factor,
        )

        x = self.temporal_layer(x)
        x = self.conv(x)
        x = self.flatten_layer(x)
        x = self.final_layer(x)

        return x

    @staticmethod
    def _apply_padding(x: Tensor, padding_size: int):
        x = torch.nn.functional.pad(x, (0, padding_size))
        return x


class _LightweightConv1d(nn.Module):
    """
    Lightweight 1D Convolution Module.

    Applies a convolution operation with multiple heads, allowing for
    parallel filter applications. Optionally applies a softmax normalization
    to the convolution weights.

    Parameters
    ----------
    input_size : int
        Number of channels of the input and output.
    kernel_size : int, optional
        Size of the convolution kernel. Default is `1`.
    padding : int, optional
        Amount of zero-padding added to both sides of the input. Default is `0`.
    heads : int, optional
        Number of attention heads used. The weight has shape `(heads, 1, kernel_size)`.
        Default is `1`.
    weight_softmax : bool, optional
        If `True`, normalizes the convolution weights with softmax before applying the convolution.
        Default is `False`.
    bias : bool, optional
        If `True`, adds a learnable bias to the output. Default is `False`.

    """

    def __init__(
        self,
        input_size,
        kernel_size=1,
        padding=0,
        heads=1,
        weight_softmax=False,
        bias=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.heads = heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(heads, 1, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input):
        B, C, T = input.size()
        H = self.heads

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        input = input.view(-1, H, T)
        output = F.conv1d(input, weight, padding=self.padding, groups=self.heads)
        output = output.view(B, C, -1)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)

        return output
