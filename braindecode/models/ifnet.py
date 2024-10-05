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

from typing import List, Tuple

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from braindecode.models.base import EEGModuleMixin
from braindecode.models.modules import (
    FilterBankLayer,
    LinearWithConstraint,
    LogPowerLayer,
)


class _InterFrequencyModule(nn.Module):
    """Module that combines outputs from different frequency bands."""

    def __init__(self, activation: nn.Module = nn.GELU):
        """

        Parameters
        ----------
        activation: nn.Module
            Activation function for the InterFrequency Module

        """
        self.activation = activation()
        super().__init__()

    def forward(self, x_list: list) -> torch.Tensor:
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
        x = sum(x_list)
        x = self.activation(x)
        return x


class _SpatioTemporalFeatureBlock(nn.Module):
    """SpatioTemporal Feature Block consisting of spatial and temporal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        kernel_sizes=[63, 31],
        patch_size: int = 125,
        radix: int = 2,
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
        patch_size : int, default=125
            Size of the patches for temporal segmentation.
        radix : int, default=2
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
        self.radix = radix
        self.patch_size = patch_size
        self.drop_prob = drop_prob
        self.activation = activation
        self.dim = dim

        # Spatial convolution
        self.spatial_conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels * self.radix,
            kernel_size=1,
            groups=self.radix,
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm1d(self.out_channels * self.radix)

        # Temporal convolutions for each radix
        self.temporal_convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
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

        # Log-Power layer
        self.log_power = LogPowerLayer(dim=self.dim)

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
        batch_size, _, times = x.shape

        # Spatial convolution
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)

        # Split the output into radix groups
        x_split = torch.split(x, self.out_channels, dim=1)

        # Apply temporal convolutions
        x_t = [conv(x_i) for x_i, conv in zip(x_split, self.temporal_convs)]

        # Inter-frequency interaction
        x = self.inter_frequency(x_t)

        # Reshape for log-power computation
        x = x.view(
            batch_size, self.out_channels, times // self.patch_size, self.patch_size
        )

        # Log-Power layer
        x = self.log_power(x)

        # Dropout
        x = self.dropout_layer(x)

        return x


class IFNetV2(EEGModuleMixin, nn.Module):
    """Interactive Frequency Convolutional Neural Network (IFNet).

        .. figure:: https://raw.githubusercontent.com/Jiaheng-Wang/IFNet/main/IFNet.png
           :align: center
           :alt: IFNetV2 Architecture

    Overview of the IFNetV2 architecture.

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

    Parameters
    ----------
    bands : List[Tuple[int, int]] or int or None, default=[[4, 16], (16, 40)]
        Frequency bands for filtering.
    out_planes : int, default=64
        Number of output feature dimensions.
    kernel_sizes : list of int, default=[63, 31]
        List of kernel sizes for temporal convolutions.
    radix : int, default=2
        Number of cross-frequency domains.
    patch_size : int, default=125
        Size of the patches for temporal segmentation.
    drop_prob : float, default=0.5
        Dropout probability.
    activation : nn.Module, default=nn.GELU
        Activation function after the InterFrequency Layer.
    verbose : bool, default=False
        Verbose output.
    filter_parameters : dict, default={}
        Additional parameters for the filter bank layer.
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
        out_planes: int = 64,
        kernel_sizes: tuple[int, int] = (63, 31),
        radix: int = 2,
        patch_size: int = 125,
        drop_prob: float = 0.5,
        activation: nn.Module = nn.GELU,
        verbose: bool = False,
        filter_parameters: dict = {},
    ):
        super().__init__(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            chs_info=chs_info,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.bands = bands
        self.in_planes = self.n_chans * radix
        self.out_planes = out_planes
        self.patch_size = patch_size
        self.verbose = verbose
        self.filter_parameters = filter_parameters

        self.spectral_filtering = FilterBankLayer(
            n_chans=self.n_chans,
            sfreq=self.sfreq,
            band_filters=self.bands,
            verbose=self.verbose,
            **self.filter_parameters,
        )
        # SpatioTemporal Feature Block
        self.feature_block = _SpatioTemporalFeatureBlock(
            in_channels=self.in_planes,
            out_channels=self.out_planes,
            kernel_sizes=kernel_sizes,
            patch_size=self.patch_size,
            radix=radix,
            drop_prob=drop_prob,
            activation=activation,
        )

        # Final classification layer
        self.final_layer = LinearWithConstraint(
            in_features=self.out_planes * (self.n_times // self.patch_size),
            out_features=self.n_outputs,
            max_norm=0.5,
        )

        # Initialize parameters
        self.apply(self._initialize_weights)

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

        # Pass through the feature block
        x = self.feature_block(x)

        # Flatten and pass through the final layer
        x = x.flatten(1)

        x = self.final_layer(x)

        return x
