from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from mne.utils import warn
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import (
    FilterBankLayer,
    LogVarLayer,
)


class FBLightConvNet(EEGModuleMixin, nn.Module):
    """LightConvNet from Ma, X et al (2023) [lightconvnet]_.

    .. figure:: https://raw.githubusercontent.com/Ma-Xinzhi/LightConvNet/refs/heads/main/network_architecture.png
        :align: center
        :alt: LightConvNet Neural Network

    A lightweight convolutional neural network incorporating temporal
    dependency learning and attention mechanisms. The architecture is
    designed to efficiently capture spatial and temporal features through
    specialized convolutional layers and **multi-head attention**.

    The network architecture consists of four main modules:

    1. **Spatial and Spectral Information Learning**:
        Applies filterbank and spatial convolutions.
        This module is followed by batch normalization and
        an activation function to enhance feature representation.

    2. **Temporal Segmentation and Feature Extraction**:
        Divides the processed data into non-overlapping temporal windows.
        Within each window, a variance-based layer extracts discriminative features,
        which are then log-transformed to stabilize variance before being
        passed to the attention module.

    3. **Temporal Attention Module**: Utilizes a multi-head attention
        mechanism with depthwise separable convolutions to capture dependencies
        across different temporal segments. The attention weights are normalized
        using softmax and aggregated to form a comprehensive temporal
        representation.

    4. **Final Layer**: Flattens the aggregated features and passes them
        through a linear layer to with kernel sizes matching the input
        dimensions to integrate features across different channels generate the
        final output predictions.

    Notes
    -----
    This implementation is not guaranteed to be correct and has not been checked
    by the original authors; it is a braindecode adaptation from the Pytorch
    source-code [lightconvnetcode]_.

    Parameters
    ----------
    n_bands : int or None or list of tuple of int, default=8
        Number of frequency bands or a list of frequency band tuples. If a list of tuples is provided,
        each tuple defines the lower and upper bounds of a frequency band.
    n_filters_spat : int, default=32
        Number of spatial filters in the depthwise convolutional layer.
    n_dim : int, default=3
        Number of dimensions for the temporal reduction layer.
    stride_factor : int, default=4
        Stride factor used for reshaping the temporal dimension.
    activation : nn.Module, default=nn.ELU
        Activation function class to apply after convolutional layers.
    verbose : bool, default=False
        If True, enables verbose output during filter creation using mne.
    filter_parameters : dict, default={}
        Additional parameters for the FilterBankLayer.
    heads : int, default=8
        Number of attention heads in the multi-head attention mechanism.
    weight_softmax : bool, default=True
        If True, applies softmax to the attention weights.
    bias : bool, default=False
        If True, includes a bias term in the convolutional layers.

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
        n_bands=9,
        n_filters_spat: int = 32,
        n_dim: int = 3,
        stride_factor: int = 4,
        win_len: int = 250,
        heads: int = 8,
        weight_softmax: bool = True,
        bias: bool = False,
        activation: nn.Module = nn.ELU,
        verbose: bool = False,
        filter_parameters: Optional[dict] = None,
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
        self.win_len = win_len
        self.activation = activation
        self.heads = heads
        self.weight_softmax = weight_softmax
        self.bias = bias
        self.filter_parameters = filter_parameters or {}

        # Checkers
        self.n_times_truncated = self.n_times
        if self.n_times % self.win_len != 0:
            warn(
                f"Time dimension ({self.n_times}) is not divisible by"
                f" win_len ({self.win_len}). Input will be "
                f"truncated in {self.n_times % self.win_len} temporal points ",
                UserWarning,
            )
            self.n_times_truncated = self.n_times - (self.n_times % self.win_len)

        # Layers
        # Following paper nomeclature
        self.spectral_filtering = FilterBankLayer(
            n_chans=self.n_chans,
            sfreq=self.sfreq,
            band_filters=self.n_bands,
            verbose=verbose,
            **self.filter_parameters,
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

        # Temporal aggregator
        self.temporal_layer = LogVarLayer(self.n_dim, False)

        self.flatten_layer = Rearrange("batch ... -> batch (...)")

        # LightWeightConv1D
        self.attn_conv = _LightweightConv1d(
            self.n_filters_spat,
            (self.n_times // self.win_len),
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
        Forward pass of the FBLightConvNet model.
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
        # x.shape: batch, n_chans, n_times

        x = self.spectral_filtering(x)
        # x.shape: batch, nbands, n_chans, n_times

        x = self.spatial_conv(x)
        # x.shape: batch, n_filters_spat, n_times

        x = x[:, :, :, : self.n_times_truncated]
        # batch, n_filters_spat, n_times_trucated

        x = x.reshape([batch_size, self.n_filters_spat, -1, self.win_len])
        # batch, n_filters_spat, n_windows, win_len
        # where the n_windows = n_times_truncated / win_len
        # and win_len = 250 by default

        x = self.temporal_layer(x)
        # x.shape : batch, n_filters_spat, n_windows

        x = self.attn_conv(x)
        # x.shape : batch, n_filters_spat, 1

        x = self.flatten_layer(x)
        # x.shape : batch, n_filters_spat

        x = self.final_layer(x)
        # x.shape : batch, n_outputs
        return x


class _LightweightConv1d(nn.Module):
    """Lightweight 1D Convolution Module.

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
        input_size: int,
        kernel_size: int = 1,
        padding: int = 0,
        heads: int = 1,
        weight_softmax: bool = False,
        bias: bool = False,
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

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input):
        # batch, n_filters_spat, n_windows
        B, C, T = input.size()

        H = self.heads

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)
            # shape: (heads, 1, kernel_size)

        # reshape input so each head is its own “batch”
        # original C = H * (C/H), so view to (B * (C/H), H, T) then transpose
        # but since C/H == 1 here per head-channel grouping, .view(-1, H, T) works
        # new shape: (B * channels_per_head, H, T)
        input = input.view(-1, H, T)
        output = F.conv1d(input, weight, padding=self.padding, groups=self.heads)
        # 4, 8, 1
        output = output.view(B, C, -1)
        # 1, 32, 1
        if self.bias is not None:
            # Add bias if it exists
            output = output + self.bias.view(1, -1, 1)
        # final shape: batch, n_filters_spat
        return output
