from __future__ import annotations

from functools import partial

import torch
from einops.layers.torch import Rearrange
from mne.utils import warn
from torch import nn, Tensor

from braindecode.models.base import EEGModuleMixin
from braindecode.models.eegnet import Conv2dWithConstraint
from braindecode.models.modules import (
    FilterBankLayer,
    LinearWithConstraint,
    LogVarLayer,
)


class FBLightConvNet(EEGModuleMixin, nn.Module):
    """LightConvNet from Ma, X et al (2023) [lightconvnet]_.

        .. figure:: https://raw.githubusercontent.com/Ma-Xinzhi/LightConvNet/refs/heads/main/network_architecture.png
            :align: center
            :alt: LightConvNet

    Fill here:

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
    activation : nn.Module, default=Swish
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
        n_bands=9,
        n_filters_spat: int = 32,
        n_dim: int = 3,
        stride_factor: int = 4,
        activation: nn.Module = nn.SiLU,
        verbose: bool = False,
        filter_parameters: dict = {},
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

        # Checkers
        if self.n_times % self.stride_factor != 0:
            warn(
                f"Time dimension ({self.n_times}) is not divisible by"
                f" stride_factor ({self.stride_factor}). Input will be "
                f"truncated.",
                UserWarning,
            )

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

        # Spatial Convolution Block (SCB)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.n_bands,
                out_channels=self.embed,
                kernel_size=(self.n_chans, 1),
                groups=self.n_bands,
            ),
            nn.BatchNorm2d(self.embed),
            self.activation(),
        )

        # Temporal aggregator
        self.temporal_layer = LogVarLayer(dim=self.n_dim)

        self.flatten_layer = Rearrange("batch ... -> batch (...)")

        if self.n_times % self.stride_factor != 0:
            self.padding_size = stride_factor - (self.n_times % stride_factor)
            self.n_times_padded = self.n_times + self.padding_size
            self.padding_layer = partial(
                self._apply_padding,
                padding_size=self.padding_size,
            )
        else:
            self.padding_layer = nn.Identity()
            self.n_times_padded = self.n_times

        # Final fully connected layer
        self.final_layer = LinearWithConstraint(
            in_features=self.embed,
            out_features=self.n_outputs,
            max_norm=0.5,
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

        x = self.spectral_filtering(x)

        x = self.spatial_conv(x)
        batch_size, channels, _, _ = x.shape

        # Check if time is divisible by stride_factor
        x = self.padding_layer(x)

        x = x.view(
            batch_size,
            channels,
            self.stride_factor,
            self.n_times_padded // self.stride_factor,
        )

        x = self.temporal_layer(x)  # type: ignore[operator]
        x = self.flatten_layer(x)
        x = self.final_layer(x)
        return x

    @staticmethod
    def _apply_padding(x: Tensor, padding_size: int):
        x = torch.nn.functional.pad(x, (0, padding_size))
        return x
