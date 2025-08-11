from __future__ import annotations

from typing import Optional, Sequence

import torch
from einops.layers.torch import Rearrange
from mne.utils import warn
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.fbcnet import _valid_layers
from braindecode.modules import (
    Conv2dWithConstraint,
    FilterBankLayer,
    LinearWithConstraint,
)


class FBMSNet(EEGModuleMixin, nn.Module):
    """FBMSNet from Liu et al (2022) [fbmsnet]_.

    .. figure:: https://raw.githubusercontent.com/Want2Vanish/FBMSNet/refs/heads/main/FBMSNet.png
        :align: center
        :alt: FBMSNet Architecture

    0. **FilterBank Layer**: Applying filterbank to transform the input.

    1. **Temporal Convolution Block**: Utilizes mixed depthwise convolution
       (MixConv) to extract multiscale temporal features from multiview EEG
       representations. The input is split into groups corresponding to different
       views each convolved with kernels of varying sizes.
       Kernel sizes are set relative to the EEG
       sampling rate, with ratio coefficients [0.5, 0.25, 0.125, 0.0625],
       dividing the input into four groups.

    2. **Spatial Convolution Block**: Applies depthwise convolution with a kernel
       size of (n_chans, 1) to span all EEG channels, effectively learning spatial
       filters. This is followed by batch normalization and the Swish activation
       function. A maximum norm constraint of 2 is imposed on the convolution
       weights to regularize the model.

    3. **Temporal Log-Variance Block**: Computes the log-variance.

    4. **Classification Layer**: A fully connected with weight constraint.

    Notes
    -----
    This implementation is not guaranteed to be correct and has not been checked
    by the original authors; it has only been reimplemented from the paper
    description and source code [fbmsnetcode]_. There is an extra layer here to
    compute the filterbank during bash time and not on data time. This avoids
    data-leak, and allows the model to follow the braindecode convention.

    Parameters
    ----------
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
    activation : nn.Module, default=nn.SiLU
        Activation function class to apply.
    verbose: bool, default False
        Verbose parameter to create the filter using mne.

    References
    ----------
    .. [fbmsnet] Liu, K., Yang, M., Yu, Z., Wang, G., & Wu, W. (2022).
        FBMSNet: A filter-bank multi-scale convolutional neural network for
        EEG-based motor imagery decoding. IEEE Transactions on Biomedical
        Engineering, 70(2), 436-445.
    .. [fbmsnetcode] Liu, K., Yang, M., Yu, Z., Wang, G., & Wu, W. (2022).
        FBMSNet: A filter-bank multi-scale convolutional neural network for
        EEG-based motor imagery decoding.
        https://github.com/Want2Vanish/FBMSNet
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
        n_bands: int = 9,
        n_filters_spat: int = 36,
        temporal_layer: str = "LogVarLayer",
        n_dim: int = 3,
        stride_factor: int = 4,
        dilatability: int = 8,
        activation: nn.Module = nn.SiLU,
        kernels_weights: Sequence[int] = (15, 31, 63, 125),
        cnn_max_norm: float = 2,
        linear_max_norm: float = 0.5,
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
        self.activation = activation
        self.dilatability = dilatability
        self.kernels_weights = kernels_weights
        self.filter_parameters = filter_parameters or {}
        self.out_channels_spatial = self.n_filters_spat * self.dilatability

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

        # MixedConv2d Layer
        self.mix_conv = nn.Sequential(
            _MixedConv2d(
                in_channels=self.n_bands,
                out_channels=self.n_filters_spat,
                stride=1,
                dilation=1,
                depthwise=False,
                kernels_weights=kernels_weights,
            ),
            nn.BatchNorm2d(self.n_filters_spat),
        )

        # Spatial Convolution Block (SCB)
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=self.n_filters_spat,
                out_channels=self.out_channels_spatial,
                kernel_size=(self.n_chans, 1),
                groups=self.n_filters_spat,
                max_norm=cnn_max_norm,
                padding=0,
            ),
            nn.BatchNorm2d(self.out_channels_spatial),
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

        # Temporal Aggregation Layer
        self.temporal_layer = _valid_layers[temporal_layer](dim=self.n_dim)  # type: ignore

        self.flatten_layer = Rearrange("batch ... -> batch (...)")

        # Final fully connected layer
        self.final_layer = LinearWithConstraint(
            in_features=self.out_channels_spatial * self.stride_factor,
            out_features=self.n_outputs,
            max_norm=linear_max_norm,
        )

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
        batch, _, _ = x.shape

        # shape: (batch, n_chans, n_times)
        x = self.spectral_filtering(x)
        # shape: (batch, n_bands, n_chans, n_times)

        # Mixed convolution
        x = self.mix_conv(x)
        # shape: (batch, self.n_filters_spat, n_chans, n_times)

        # Spatial convolution block
        x = self.spatial_conv(x)
        # shape: (batch, self.out_channels_spatial, 1, n_times)

        # Apply some padding to the input to make it divisible by the stride factor
        x = self.padding_layer(x)
        # shape: (batch, self.out_channels_spatial, 1, n_times_padded)

        # Reshape for temporal layer
        x = x.view(batch, self.out_channels_spatial, self.stride_factor, -1)
        # shape: (batch, self.out_channels_spatial, self.stride_factor, n_times/self.stride_factor)

        # Temporal aggregation
        x = self.temporal_layer(x)
        # shape: (batch, self.out_channels_spatial, self.stride_factor, 1)

        # Flatten and classify
        x = self.flatten_layer(x)
        # shape: (batch, self.out_channels_spatial*self.stride_factor)

        x = self.final_layer(x)
        # shape: (batch, n_outputs)
        return x


class _MixedConv2d(nn.Module):
    """Mixed Grouped Convolution for multiscale feature extraction."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernels_weights=(15, 31, 63, 125),
        stride=1,
        dilation=1,
        depthwise=False,
    ):
        super().__init__()

        num_groups = len(kernels_weights)
        in_splits = self._split_channels(in_channels, num_groups)
        out_splits = self._split_channels(out_channels, num_groups)
        self.splits = in_splits

        self.convs = nn.ModuleList()
        # Create a convolutional layer for each kernel size
        for k, in_ch, out_ch in zip(kernels_weights, in_splits, out_splits):
            conv_groups = out_ch if depthwise else 1
            conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=(1, k),
                stride=stride,
                padding="same",
                dilation=dilation,
                groups=conv_groups,
                bias=False,
            )
            self.convs.append(conv)

    @staticmethod
    def _split_channels(num_chan, num_groups):
        """
        Splits the total number of channels into a specified
        number of groups as evenly as possible.

        Parameters
        ----------
        num_chan : int
            The total number of channels to split.
        num_groups : int
            The number of groups to split the channels into.

        Returns
        -------
        list of int
            A list containing the number of channels in each group.
            The first group may have more channels if the division is not even.
        """
        split = [num_chan // num_groups for _ in range(num_groups)]
        split[0] += num_chan - sum(split)
        return split

    def forward(self, x):
        # Split the input tensor `x` along the channel dimension (dim=1) into groups.
        # The size of each group is defined by `self.splits`, which is calculated
        # based on the number of input channels and the number of kernel sizes.
        x_split = torch.split(x, self.splits, 1)

        # For each split group, apply the corresponding convolutional layer.
        # `self.values()` returns the convolutional layers in the order they were added.
        # The result is a list of output tensors, one for each group.
        x_out = [conv(x_split[i]) for i, conv in enumerate(self.convs)]

        # Concatenate the outputs from all groups along the channel dimension (dim=1)
        # to form a single output tensor.
        x = torch.cat(x_out, 1)

        # Return the concatenated tensor as the output of the mixed convolution.
        return x
