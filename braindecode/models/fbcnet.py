from __future__ import annotations

from typing import Optional
from mne.utils import warn

import torch

from torch import nn

from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin
from braindecode.models.eegnet import Conv2dWithConstraint
from braindecode.models.modules import (
    FilterBankLayer,
    VarLayer,
    StdLayer,
    LogVarLayer,
    MaxLayer,
    MeanLayer,
    LinearWithConstraint,
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

    The FBCNet model applies spatial convolution and variance calculation along
    the time axis, inspired by the Filter Bank Common Spatial Pattern (FBCSP)
    algorithm.

    Notes
    -----
    This implementation is not guaranteed to be correct and has not been checked
    by the original authors; it has only been reimplemented from the paper
    description and source code [fbcnetcode2021]_. There is a difference in the
    activation function; in the paper, the ELU is used as the activation function,
    but in the original code, SiLU is used. We follow the paper.

    Parameters
    ----------
    n_bands : int or None or List[Tuple[int, int]]], default=9
        Number of frequency bands. Could
    n_filters_spat : int, default=32
        The depth of the depthwise convolutional layer.
    n_dim: int, default=3
        Number of dimensions for the temporal reductor
    temporal_layer : str, default='LogVarLayer'
        Type of temporal aggregator layer. Options: 'VarLayer', 'StdLayer',
        'LogVarLayer', 'MeanLayer', 'MaxLayer'.
    stride_factor : int, default=4
        Stride factor for reshaping.
    activation : nn.Module, default=Swish
        Activation function class to apply.
    verbose: bool, default False
        Verbose parameter to create the filter using mne

    References
    ----------
    .. [fbcnet2021] Mane, R., Chew, E., Chua, K., Ang, K. K., Robinson, N.,
        Vinod, A. P., ... & Guan, C. (2021). FBCNet: A multi-view convolutional
        neural network for brain-computer interface. preprint arXiv:2104.01233.
    .. [fbcnetcode2021] Mane, R., Chew, E., Chua, K., Ang, K. K., Robinson, N.,
        Vinod, A. P., ... & Guan, C. (2021). FBCNet: A multi-view convolutional
        neural network for brain-computer interface.
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
        verbose: bool = False,
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

        # Checkers
        if temporal_layer not in _valid_layers:
            raise NotImplementedError(
                f"Temporal layer '{temporal_layer}' is not implemented."
            )

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
                max_norm=2,
                padding=0,
            ),
            nn.BatchNorm2d(self.n_filters_spat * self.n_bands),
            self.activation(),
        )

        # Temporal aggregator
        self.temporal_layer = _valid_layers[temporal_layer](dim=self.n_dim)

        self.flatten_layer = Rearrange("batch ... -> batch (...)")

        # Final fully connected layer
        self.final_layer = LinearWithConstraint(
            in_features=self.n_filters_spat * self.n_bands * self.stride_factor,
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
        batch_size, channels, _, time = x.shape
        # Check if time is divisible by stride_factor
        if time % self.stride_factor != 0:
            # Pad x to make time divisible by stride_factor
            padding = self.stride_factor - (time % self.stride_factor)
            x = torch.nn.functional.pad(x, (0, padding))
            time += padding  # Update the time dimension after padding

        x = x.view(batch_size, channels, self.stride_factor, time // self.stride_factor)

        x = self.temporal_layer(x)  # type: ignore[operator]
        x = self.flatten_layer(x)
        x = self.final_layer(x)
        return x


if __name__ == "__main__":
    x = torch.randn(1, 22, 1001)

    model = FBCNet(n_chans=22, n_outputs=2, n_times=1001, sfreq=250)

    with torch.no_grad():
        out = model(x)
