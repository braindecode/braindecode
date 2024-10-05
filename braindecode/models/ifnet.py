"""
IFNet Neural Network.
Authors: Jiaheng Wang
         Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
License: MIT (https://github.com/Jiaheng-Wang/IFNet/blob/main/LICENSE)
"""

from __future__ import annotations

import torch
from torch import nn


from braindecode.models.base import EEGModuleMixin
from braindecode.models.eegnet import Conv2dWithConstraint
from braindecode.models.modules import LinearWithConstraint

from torch.nn.init import trunc_normal_


class LogPowerLayer(nn.Module):
    """
    Layer that computes the logarithm of the power of the input signal.
    """

    def __init__(self, dim: int, log_min: float = 1e-4, log_max: float = 1e4):
        """
        Parameters
        ----------
        dim : int
            Dimension over which to compute the power.
        """
        super().__init__()
        self.dim = dim
        self.log_min = log_min
        self.log_max = log_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Log-power of the input tensor.
        """
        power = torch.mean(x**2, dim=self.dim)
        log_power = torch.log(torch.clamp(power, min=self.log_min, max=self.log_max))
        return log_power


class InterFrequencyModule(nn.Module):
    """
    Module that combines outputs from different frequency bands.
    """

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
        """
        Forward pass.

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


class Stem(nn.Module):
    """
    Stem module consisting of spatial and temporal convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        kernel_sizes: list = [63, 31],
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
        for idx, kernel_size in enumerate(kernel_sizes):
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
        self.inter_frequency = InterFrequencyModule(activation=self.activation)

        # Log-Power layer
        self.log_power = LogPowerLayer(dim=self.dim)

        # Dropout
        self.dropout_layer = nn.Dropout(self.drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor after processing.
        """
        batch_size, channels, times = x.shape

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
    """
    Interactive Frequency Convolutional Neural Network (IFNet).

    Parameters
    ----------
    out_planes : int, default=64
        Number of output feature dimensions.
    kernel_sizes : list of int, default=[63, 31]
        List of kernel sizes for temporal convolutions.
    radix : int, default=2
        Number of frequency bands or groups.
    patch_size : int, default=125
        Size of the patches for temporal segmentation.
    drop_prob : float, default=0.5
        Dropout probability.
    """

    def __init__(
        self,
        # Braindecode parameters
        n_chans: int,
        n_outputs: int,
        n_times: int,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # Model-specific parameters
        out_planes: int = 64,
        kernel_sizes: list = [63, 31],
        radix: int = 2,
        patch_size: int = 125,
        drop_prob: float = 0.5,
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

        self.in_planes = self.n_chans * radix
        self.out_planes = out_planes
        self.patch_size = patch_size

        # Stem module
        self.stem = Stem(
            in_channels=self.in_planes,
            out_channels=self.out_planes,
            kernel_sizes=kernel_sizes,
            patch_size=self.patch_size,
            radix=radix,
            drop_prob=drop_prob,
        )

        # Final classification layer
        self.final_layer = LinearWithConstraint(
            in_features=self.out_planes * (self.n_times // self.patch_size),
            out_features=self.n_outputs,
            max_norm=0.5,
            do_weight_norm=True,
        )

        # Initialize parameters
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(m):
        """
        Initializes weights of the network.

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
        """
        Forward pass of IFNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, n_outputs).
        """
        # Reshape input to match expected dimensions
        batch_size = x.size(0)
        x = x.view(batch_size, self.in_planes, self.time_points)

        # Pass through the stem module
        x = self.stem(x)

        # Flatten and pass through the final layer
        x = x.flatten(1)

        x = self.final_layer(x)

        return x
