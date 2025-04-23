"""IFNet Neural Network.

Authors: Jiaheng Wang
         Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
License: MIT (https://github.com/Jiaheng-Wang/IFNet/blob/main/LICENSE)

J. Wang, L. Yao and Y. Wang, "IFNet: An Interactive Frequency Convolutional
Neural Network for Enhancing Motor Imagery Decoding from EEG," in IEEE
Transactions on Neural Systems and Rehabilitation Engineering,
doi: 10.1109/TNSRE.2023.3257319.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from einops.layers.torch import Rearrange
from mne.utils import warn
from torch import nn
from torch.nn.init import trunc_normal_

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import (
    FilterBankLayer,
    LinearWithConstraint,
    LogPowerLayer,
)


class IFNet(EEGModuleMixin, nn.Module):
    """IFNetV2 from Wang J et al (2023) [ifnet]_.

    .. figure:: https://raw.githubusercontent.com/Jiaheng-Wang/IFNet/main/IFNet.png
        :align: center
        :alt: IFNetV2 Architecture

    Overview of the Interactive Frequency Convolutional Neural Network architecture.

    IFNetV2 is designed to effectively capture spectro-spatial-temporal
    features for motor imagery decoding from EEG data. The model consists of
    three stages: Spectro-Spatial Feature Representation, Cross-Frequency
    Interactions, and Classification.

    - **Spectro-Spatial Feature Representation**: The raw EEG signals are
      filtered into two characteristic frequency bands: low (4-16 Hz) and
      high (16-40 Hz), covering the most relevant motor imagery bands.
      Spectro-spatial features are then extracted through 1D point-wise
      spatial convolution followed by temporal convolution.

    - **Cross-Frequency Interactions**: The extracted spectro-spatial
      features from each frequency band are combined through an element-wise
      summation operation, which enhances feature representation while
      preserving distinct characteristics.

    - **Classification**: The aggregated spectro-spatial features are further
      reduced through temporal average pooling and passed through a fully
      connected layer followed by a softmax operation to generate output
      probabilities for each class.

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper description and
    Torch source code [ifnetv2code]_. Version 2 is present only in the repository,
    and the main difference is one pooling layer, describe at the TABLE VII
    from the paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10070810


    Parameters
    ----------
    bands : list[tuple[int, int]] or int or None, default=[[4, 16], (16, 40)]
        Frequency bands for filtering.
    out_planes : int, default=64
        Number of output feature dimensions.
    kernel_sizes : tuple of int, default=(63, 31)
        List of kernel sizes for temporal convolutions.
    patch_size : int, default=125
        Size of the patches for temporal segmentation.
    drop_prob : float, default=0.5
        Dropout probability.
    activation : nn.Module, default=nn.GELU
        Activation function after the InterFrequency Layer.
    verbose : bool, default=False
        Verbose to control the filtering layer
    filter_parameters : dict, default={}
        Additional parameters for the filter bank layer.

    References
    ----------
    .. [ifnet] Wang, J., Yao, L., & Wang, Y. (2023). IFNet: An interactive
        frequency convolutional neural network for enhancing motor imagery
        decoding from EEG. IEEE Transactions on Neural Systems and
        Rehabilitation Engineering, 31, 1900-1911.
    .. [ifnetv2code] Wang, J., Yao, L., & Wang, Y. (2023). IFNet: An interactive
        frequency convolutional neural network for enhancing motor imagery
        decoding from EEG.
        https://github.com/Jiaheng-Wang/IFNet
    """

    def __init__(
        self,
        # Braindecode parameters
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # Model-specific parameters
        bands: list[tuple[float, float]] | int | None = [(4.0, 16.0), (16, 40)],
        n_filters_spat: int = 64,
        kernel_sizes: tuple[int, int] = (63, 31),
        stride_factor: int = 8,
        drop_prob: float = 0.5,
        linear_max_norm: float = 0.5,
        activation: type[nn.Module] = nn.GELU,
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

        self.bands = bands
        self.n_filters_spat = n_filters_spat
        self.stride_factor = stride_factor
        self.kernel_sizes = kernel_sizes
        self.verbose = verbose
        self.filter_parameters = filter_parameters
        self.drop_prob = drop_prob
        self.activation = activation
        self.linear_max_norm = linear_max_norm
        self.filter_parameters = filter_parameters or {}

        # Layers
        # Following paper nomenclature
        self.spectral_filtering = FilterBankLayer(
            n_chans=self.n_chans,
            sfreq=self.sfreq,
            band_filters=self.bands,
            verbose=verbose,
            **self.filter_parameters,
        )
        # As we have an internal process to create the bands,
        # we get the values from the filterbank
        self.n_bands = self.spectral_filtering.n_bands

        # My interpretation from the TABLE VII IFNet Architecture from the
        # paper.
        self.ensuredim = Rearrange(
            "batch nbands chans time -> batch (nbands chans) time"
        )

        # SpatioTemporal Feature Block
        self.feature_block = _SpatioTemporalFeatureBlock(
            in_channels=self.n_chans * self.n_bands,
            out_channels=self.n_filters_spat,
            kernel_sizes=self.kernel_sizes,
            stride_factor=self.stride_factor,
            n_bands=self.n_bands,
            drop_prob=self.drop_prob,
            activation=self.activation,
            n_times=self.n_times,
        )

        # Final classification layer
        self.final_layer = LinearWithConstraint(
            in_features=self.n_filters_spat * stride_factor,
            out_features=self.n_outputs,
            max_norm=self.linear_max_norm,
        )

        self.flatten = Rearrange("batch ... -> batch (...)")

        # Initialize parameters
        self._initialize_weights(self)

    @staticmethod
    def _initialize_weights(m):
        """Initializes weights of the network.

        Parameters
        ----------
        m : nn.Module
            Module to initialize.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of IFNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, n_outputs).
        """
        # Pass through the spectral filtering layer
        x = self.spectral_filtering(x)
        # x is now of shape (batch_size, n_bands, n_chans, n_times)
        x = self.ensuredim(x)
        # x is now of shape (batch_size, n_bands * n_chans, n_times)

        # Pass through the feature block
        x = self.feature_block(x)

        # Flatten and pass through the final layer
        x = self.flatten(x)

        x = self.final_layer(x)

        return x


class _InterFrequencyModule(nn.Module):
    """Module that combines outputs from different frequency bands."""

    def __init__(self, activation: nn.Module = nn.GELU):
        """

        Parameters
        ----------
        activation: nn.Module
            Activation function for the InterFrequency Module

        """
        super().__init__()

        self.activation = activation()

    def forward(self, x_list: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x_list : list of torch.Tensor
            List of tensors to be combined.

        Returns
        -------
        torch.Tensor
            Combined tensor after applying GELU activation.
        """
        x = torch.stack(x_list, dim=0).sum(dim=0)
        x = self.activation(x)
        return x


class _SpatioTemporalFeatureBlock(nn.Module):
    """SpatioTemporal Feature Block consisting of spatial and temporal convolutions."""

    def __init__(
        self,
        n_times: int,
        in_channels: int,
        out_channels: int = 64,
        kernel_sizes: Sequence[int] = [63, 31],
        stride_factor: int = 8,
        n_bands: int = 2,
        drop_prob: float = 0.5,
        activation: nn.Module = nn.GELU,
        dim: int = 3,
    ):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int, default=64
            Number of output channels.
        kernel_sizes : list of int, default=[63, 31]
            List of kernel sizes for temporal convolutions.
        stride_factor : int, default=4
            Stride factor for temporal segmentation.
        n_bands : int, default=2
            Number of frequency bands or groups.
        drop_prob : float, default=0.5
            Dropout probability.
        activation: nn.Module, default=nn.GELU
            Activation function after the InterFrequency Layer
        dim: int, default=3
            Internal dimensional to apply the LogPowerLayer
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_bands = n_bands
        self.stride_factor = stride_factor
        self.drop_prob = drop_prob
        self.activation = activation
        self.dim = dim
        self.n_times = n_times
        self.kernel_sizes = kernel_sizes

        if self.n_bands != len(self.kernel_sizes):
            warn(
                f"Got {self.n_bands} bands, different from {len(self.kernel_sizes)} amount of "
                "kernels to build the temporal convolution, we will apply "
                "min(n_bands, len(self.kernel_size) to apply the convolution.",
                UserWarning,
            )
            if self.n_bands > len(self.kernel_sizes):
                self.n_bands = len(self.kernel_sizes)
                warn(
                    f"Reducing number of bands to {len(self.kernel_sizes)} to match the number of kernels.",
                    UserWarning,
                )
            elif self.n_bands < len(self.kernel_sizes):
                self.kernel_sizes = self.kernel_sizes[: self.n_bands]
                warn(
                    f"Reducing number of kernels to {self.n_bands} to match the number of bands.",
                    UserWarning,
                )

        if self.n_times % self.stride_factor != 0:
            warn(
                f"Time dimension ({self.n_times}) is not divisible by"
                f" stride_factor ({self.stride_factor}). Input will be padded.",
                UserWarning,
            )

        out_channels_spatial = self.out_channels * self.n_bands

        # Spatial convolution
        self.spatial_conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=out_channels_spatial,
            kernel_size=1,
            groups=self.n_bands,
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm1d(out_channels_spatial)

        self.unpack_bands = nn.Unflatten(
            dim=1, unflattened_size=(self.n_bands, self.out_channels)
        )

        # Temporal convolutions for each radix
        self.temporal_convs = nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.temporal_convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=self.out_channels,
                        out_channels=self.out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        groups=self.out_channels,
                        bias=False,
                    ),
                    nn.BatchNorm1d(self.out_channels),
                )
            )
        # Inter-frequency module
        self.inter_frequency = _InterFrequencyModule(activation=self.activation)

        if self.n_times % self.stride_factor != 0:
            self.padding_size = stride_factor - (self.n_times % stride_factor)
            self.n_times_padded = self.n_times + self.padding_size
            self.padding_layer = nn.ConstantPad1d((0, self.padding_size), 0.0)
        else:
            self.padding_layer = nn.Identity()
            self.n_times_padded = self.n_times

        # Log-Power layer
        self.log_power = LogPowerLayer(dim=self.dim)  # type: ignore
        # Dropout
        self.dropout_layer = nn.Dropout(self.drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor after processing.
        """
        batch_size, _, _ = x.shape

        # Spatial convolution
        x = self.spatial_conv(x)

        x = self.spatial_bn(x)

        # Split the output by bands for each frequency
        x_split = self.unpack_bands(x)

        x_t = []
        for idx, conv in enumerate(self.temporal_convs):
            x_t.append(conv(x_split[::, idx]))

        # Inter-frequency interaction
        x = self.inter_frequency(x_t)

        # Reshape for temporal segmentation
        x = self.padding_layer(x)
        # x is now of shape (batch_size, ..., n_times_padded)

        # Reshape for log-power computation
        x = x.view(
            batch_size,
            self.out_channels,
            self.stride_factor,
            self.n_times_padded // self.stride_factor,
        )

        # Log-Power layer
        x = self.log_power(x)

        # Dropout
        x = self.dropout_layer(x)

        return x
