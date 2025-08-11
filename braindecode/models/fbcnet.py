from __future__ import annotations

from typing import Any

import torch
from einops.layers.torch import Rearrange
from mne.utils import warn
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import (
    Conv2dWithConstraint,
    FilterBankLayer,
    LinearWithConstraint,
    LogVarLayer,
    MaxLayer,
    MeanLayer,
    StdLayer,
    VarLayer,
)

_valid_layers = {
    "VarLayer": VarLayer,
    "StdLayer": StdLayer,
    "LogVarLayer": LogVarLayer,
    "MeanLayer": MeanLayer,
    "MaxLayer": MaxLayer,
}


class FBCNet(EEGModuleMixin, nn.Module):
    """FBCNet from Mane, R et al (2021) [fbcnet2021]_.

    .. figure:: https://raw.githubusercontent.com/ravikiran-mane/FBCNet/refs/heads/master/FBCNet-V2.png
        :align: center
        :alt: FBCNet Architecture

    The FBCNet model applies spatial convolution and variance calculation along
    the time axis, inspired by the Filter Bank Common Spatial Pattern (FBCSP)
    algorithm.

    Notes
    -----
    This implementation is not guaranteed to be correct and has not been checked
    by the original authors; it has only been reimplemented from the paper
    description and source code [fbcnetcode2021]_. There is a difference in the
    activation function; in the paper, the ELU is used as the activation function,
    but in the original code, SiLU is used. We followed the code.

    Parameters
    ----------
    n_bands : int or None or list[tuple[int, int]]], default=9
        Number of frequency bands. Could
    n_filters_spat : int, default=32
        Number of spatial filters for the first convolution.
    n_dim: int, default=3
        Number of dimensions for the temporal reductor
    temporal_layer : str, default='LogVarLayer'
        Type of temporal aggregator layer. Options: 'VarLayer', 'StdLayer',
        'LogVarLayer', 'MeanLayer', 'MaxLayer'.
    stride_factor : int, default=4
        Stride factor for reshaping.
    activation : nn.Module, default=nn.SiLU
        Activation function class to apply in Spatial Convolution Block.
    cnn_max_norm : float, default=2.0
        Maximum norm for the spatial convolution layer.
    linear_max_norm : float, default=0.5
        Maximum norm for the final linear layer.
    filter_parameters: dict, default None
        Parameters for the FilterBankLayer

    References
    ----------
    .. [fbcnet2021] Mane, R., Chew, E., Chua, K., Ang, K. K., Robinson, N.,
        Vinod, A. P., ... & Guan, C. (2021). FBCNet: A multi-view convolutional
        neural network for brain-computer interface. preprint arXiv:2104.01233.
    .. [fbcnetcode2021] Link to source-code:
        https://github.com/ravikiran-mane/FBCNet
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
        temporal_layer: str = "LogVarLayer",
        n_dim: int = 3,
        stride_factor: int = 4,
        activation: nn.Module = nn.SiLU,
        linear_max_norm: float = 0.5,
        cnn_max_norm: float = 2.0,
        filter_parameters: dict[Any, Any] | None = None,
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
        self.filter_parameters = filter_parameters or {}

        # Checkers
        if temporal_layer not in _valid_layers:
            raise NotImplementedError(
                f"Temporal layer '{temporal_layer}' is not implemented."
            )

        if self.n_times % self.stride_factor != 0:
            warn(
                f"Time dimension ({self.n_times}) is not divisible by"
                f" stride_factor ({self.stride_factor}). Input will be padded.",
                UserWarning,
            )

        # Layers
        # Following paper nomenclature
        self.spectral_filtering = FilterBankLayer(
            n_chans=self.n_chans,
            sfreq=self.sfreq,
            band_filters=self.n_bands,
            verbose=False,
            **self.filter_parameters,
        )
        # As we have an internal process to create the bands,
        # we get the values from the filterbank
        self.n_bands = self.spectral_filtering.n_bands

        # Spatial Convolution Block (SCB)
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=self.n_bands,
                out_channels=self.n_filters_spat * self.n_bands,
                kernel_size=(self.n_chans, 1),
                groups=self.n_bands,
                max_norm=cnn_max_norm,
                padding=0,
            ),
            nn.BatchNorm2d(self.n_filters_spat * self.n_bands),
            self.activation(),
        )

        # Padding layer
        if self.n_times % self.stride_factor != 0:
            self.padding_size = stride_factor - (self.n_times % stride_factor)
            self.n_times_padded = self.n_times + self.padding_size
            self.padding_layer = nn.ConstantPad1d((0, self.padding_size), 0.0)
        else:
            self.padding_layer = nn.Identity()
            self.n_times_padded = self.n_times

        # Temporal aggregator
        self.temporal_layer = _valid_layers[temporal_layer](dim=self.n_dim)  # type: ignore

        # Flatten layer
        self.flatten_layer = Rearrange("batch ... -> batch (...)")

        # Final fully connected layer
        self.final_layer = LinearWithConstraint(
            in_features=self.n_filters_spat * self.n_bands * self.stride_factor,
            out_features=self.n_outputs,
            max_norm=linear_max_norm,
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
        # output: (batch_size, n_chans, n_times)
        x = self.spectral_filtering(x)

        # output: (batch_size, n_bands, n_chans, n_times)
        x = self.spatial_conv(x)
        batch_size, channels, _, _ = x.shape

        # shape: (batch_size, n_filters_spat * n_bands, 1, n_times)
        x = self.padding_layer(x)

        # shape: (batch_size, n_filters_spat * n_bands, 1, n_times_padded)
        x = x.view(
            batch_size,
            channels,
            self.stride_factor,
            self.n_times_padded // self.stride_factor,
        )
        # shape: batch_size, n_filters_spat * n_bands, stride, n_times_padded/stride
        x = self.temporal_layer(x)  # type: ignore[operator]

        # shape: batch_size, n_filters_spat * n_bands, stride, 1
        x = self.flatten_layer(x)

        # shape: batch_size, n_filters_spat * n_bands * stride
        x = self.final_layer(x)
        # shape: batch_size, n_outputs
        return x
