from __future__ import annotations

from typing import Optional
from mne.utils import warn

import torch

from torch import nn

from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin
from braindecode.models.eegnet import Conv2dWithConstraint
from braindecode.models.fbcnet import _valid_layers

from braindecode.models.modules import LinearWithConstraint, FilterBank


class FBMSNet(EEGModuleMixin, nn.Module):
    """FBMSNet model adapted for braindecode.

    Parameters
    ----------
    n_chans : int
        Number of EEG channels.
    n_outputs : int
        Number of output classes.
    n_times : int
        Number of time samples in the input data.
    n_bands : int, default=9
        Number of input channels (e.g., number of frequency bands).
    stride_factor : int, default=4
        Stride factor for temporal segmentation.
    temporal_layer : str, default='LogVarLayer'
        Temporal aggregation layer to use.
    n_filters_spat : int, default=36
        Number of output channels from the MixedConv2d layer.
    dilatability : int, default=8
        Expansion factor for the spatial convolution block.
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
        n_filters_spat: int = 36,
        temporal_layer: str = "LogVarLayer",
        n_dim: int = 3,
        stride_factor: int = 4,
        dilatability: int = 8,
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
        self.dilatability = dilatability

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
        self.spectral_filtering = FilterBank(
            n_chans=self.n_chans,
            sfreq=self.sfreq,
            band_filters=self.n_bands,
            verbose=verbose,
        )
        # As we have an internal process to create the bands,
        # we get the values from the filterbank
        self.n_bands = self.spectral_filtering.n_bands

        # MixedConv2d Layer
        self.mix_conv = nn.Sequential(
            _MixedConv2d(
                in_channels=self.n_bands,
                out_channels=self.n_filters_spat,
                kernel_size=[(1, 15), (1, 31), (1, 63), (1, 125)],
                stride=1,
                dilation=1,
                depthwise=False,
            ),
            nn.BatchNorm2d(self.n_filters_spat),
        )

        # Spatial Convolution Block (SCB)
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=self.n_filters_spat,
                out_channels=self.n_filters_spat * self.dilatability,
                kernel_size=(self.n_chans, 1),
                groups=self.num_features,
                max_norm=2,
                weight_norm=True,
                padding=0,
            ),
            nn.BatchNorm2d(self.n_filters_spat * self.dilatability),
            nn.SiLU(),
        )

        # Temporal Aggregation Layer
        self.temporal_layer = _valid_layers[temporal_layer](dim=3)

        # Flatten Layer
        self.flatten_layer = Rearrange("batch ... -> batch (...)")

        # Final Linear Layer
        self.final_layer = LinearWithConstraint(
            in_features=self._get_feature_dim(),
            out_features=self.n_outputs,
            max_norm=0.5,
        )

    def _get_feature_dim(self):
        # Create a dummy input to calculate the output feature dimension
        dummy_input = torch.ones(1, self.n_chans, self.n_times)
        x = self.spectral_filtering(dummy_input)
        x = self.mix_conv(x)
        x = self.spatial_conv(x)
        x = x.view(x.size(0), x.size(1), self.stride_factor, -1)
        x = self.temporal_layer(x)
        x = x.flatten(start_dim=1)
        return x.size(1)

    def forward(self, x):
        """
        Forward pass of the FBMSNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, n_outputs).
        """
        # Reshape input to (batch_size, in_channels, n_chans, n_times)
        x = self.spectral_filtering(x)
        # Mixed convolution
        x = self.mix_conv(x)
        # Spatial convolution block
        x = self.spatial_conv(x)
        # Reshape for temporal layer
        x = x.view(x.size(0), x.size(1), self.stride_factor, -1)
        # Temporal aggregation
        x = self.temporal_layer(x)
        # Flatten and classify
        x = self.flatten_layer(x)
        x = self.final_layer(x)
        return x


class _MixedConv2d(nn.ModuleDict):
    """Mixed Grouped Convolution for multiscale feature extraction."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=[(1, 15), (1, 31), (1, 63), (1, 125)],
        stride=1,
        dilation=1,
        depthwise=False,
    ):
        super().__init__()

        num_groups = len(kernel_size)
        in_splits = self._split_channels(in_channels, num_groups)
        out_splits = self._split_channels(out_channels, num_groups)
        self.splits = in_splits

        for idx, (k, in_ch, out_ch) in enumerate(
            zip(kernel_size, in_splits, out_splits)
        ):
            conv_groups = out_ch if depthwise else 1
            padding_value = self._get_padding_value(k, stride, dilation)
            self.add_module(
                str(idx),
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=k,
                    stride=stride,
                    padding=padding_value,
                    dilation=dilation,
                    groups=conv_groups,
                    bias=False,
                ),
            )

    @staticmethod
    def _split_channels(num_chan, num_groups):
        split = [num_chan // num_groups for _ in range(num_groups)]
        split[0] += num_chan - sum(split)
        return split

    @staticmethod
    def _get_padding_value(kernel_size, stride, dilation):
        padding = []
        for k, s, d in zip(
            kernel_size,
            stride if isinstance(stride, tuple) else (stride, stride),
            dilation if isinstance(dilation, tuple) else (dilation, dilation),
        ):
            pad = ((s - 1) + d * (k - 1)) // 2
            padding.append(pad)
        return tuple(padding)

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [conv(x_split[i]) for i, conv in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x
