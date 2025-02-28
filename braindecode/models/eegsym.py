from __future__ import annotations
import re
from math import ceil

import torch

import numpy as np
import mne
from mne.channels import make_standard_montage
from re import search

from torch import nn
from typing import List, Tuple
from braindecode.models.base import EEGModuleMixin


class EEGSym(EEGModuleMixin, nn.Module):
    """EEGSym from Pérez-Velasco et al (2022) [eegsym2022]_.

    .. figure:: https://raw.githubusercontent.com/Serpeve/EEGSym/refs/heads/main/EEGSym_scheme_online.png
        :align: center
        :alt: EEGSym Architecture

    TO-DO: Use more EEGInceptionERP components.


    Parameters
    ----------
    filters_per_branch : int, optional
        Number of filters in each inception branch. Should be a multiple of 8.
        Default is 8.
    scales_time : tuple of int, optional
        Temporal scales (in milliseconds) for the temporal convolutions in the first
        inception module. Default is (500, 250, 125).
    drop_prob : float, optional
        Dropout probability. Default is 0.25.
    activation : nn.Module, optional
        Activation function to use. Default is nn.ELU().
    spatial_resnet_repetitions : int, optional
        Number of repetitions of the spatial analysis operations at each step.
        Default is 1.
    left_right_chs : list of tuple of str, optional
        Optional list of tuples with the names of the left and right hemisphere channels.
        If not provided, the channels will be split using function division_channels_idx,
        and left/right channels will be matched by the function match_hemisphere_chans.
    middle_chs : list of str, optional
        Optional list of the names of the middle channels. If not provided, the channels
        will be split using function division_channels_idx.

    References
    ----------
    .. [eegsym2022] Pérez-Velasco, S., Santamaría-Vázquez, E., Martínez-Cagigal, V.,
       Marcos-Martínez, D., & Hornero, R. (2022). EEGSym: Overcoming inter-subject
       variability in motor imagery based BCIs with deep learning. IEEE Transactions
       on Neural Systems and Rehabilitation Engineering, 30, 1766-1775.
    .. [eegsym2022code] Pérez-Velasco, S., EEGSym source code.
        https://github.com/Serpeve/EEGSym
    """

    def __init__(
        self,
        # braidecode parameters
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # Model parameters
        filters_per_branch: int = 8,
        scales_time: Tuple[int, int, int] = (500, 250, 125),
        drop_prob: float = 0.25,
        activation: nn.Module = nn.ELU(),
        spatial_resnet_repetitions: int = 1,
        left_right_chs: list[tuple[str, str]] | None = None,
        middle_chs: list[str] | None = None,
    ):
        if (left_right_chs is None) != (middle_chs is None):
            raise ValueError(
                "Either both or none of left_right_chs and middle_chs must be provided."
            )
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.filters_per_branch = filters_per_branch
        self.scales_time = scales_time
        self.drop_prob = drop_prob
        self.activation = activation
        self.spatial_resnet_repetitions = spatial_resnet_repetitions

        # Calculate scales in samples
        self.scales_samples = [int(s * self.sfreq / 2000) * 2 + 1 for s in scales_time]

        ch_names = [ch["ch_name"] for ch in self.chs_info]
        if left_right_chs is None:
            left_chs, right_chs, middle_chs = division_channels_idx(ch_names)
            left_chs, right_chs = zip(*match_hemisphere_chans(left_chs, right_chs))
        else:
            left_chs, right_chs = zip(*left_right_chs)
        self.left_idx, self.right_idx, self.middle_idx = [
            [ch_names.index(ch) for ch in ch_subset]
            for ch_subset in (left_chs, right_chs, middle_chs)
        ]

        self.n_channels_per_hemi = len(self.left_idx) + len(self.middle_idx)

        # Build the model
        self._build_model()

    def _build_model(self):
        # Initial inception modules
        self.inception_block1 = self._create_inception_block(
            in_channels=1,
            scales_samples=self.scales_samples,
            filters_per_branch=self.filters_per_branch,
            ncha=self.n_channels_per_hemi,
            average_pool=2,
            init=True,
        )
        self.inception_block2 = self._create_inception_block(
            in_channels=self.filters_per_branch * len(self.scales_samples),
            scales_samples=[max(1, s // 4) for s in self.scales_samples],
            filters_per_branch=self.filters_per_branch,
            ncha=1,
            average_pool=2,
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            self._create_residual_block(
                in_channels=self.filters_per_branch * len(self.scales_samples),
                filters=int(self.filters_per_branch * len(self.scales_samples) / 2),
                kernel_size=16,
                average_pool=2,
            ),
            self._create_residual_block(
                in_channels=int(self.filters_per_branch * len(self.scales_samples) / 2),
                filters=int(self.filters_per_branch * len(self.scales_samples) / 2),
                kernel_size=8,
                average_pool=2,
            ),
            self._create_residual_block(
                in_channels=int(self.filters_per_branch * len(self.scales_samples) / 2),
                filters=int(self.filters_per_branch * len(self.scales_samples) / 4),
                kernel_size=4,
                average_pool=2,
            ),
        )

        # Temporal reduction
        self.temporal_reduction = nn.Sequential(
            TemporalBlock(
                in_channels=int(self.filters_per_branch * len(self.scales_samples) / 4),
                filters=int(self.filters_per_branch * len(self.scales_samples) / 4),
                kernel_size=4,
                activation=self.activation,
                drop_prob=self.drop_prob,
            ),
            nn.AvgPool3d(kernel_size=(1, 2, 1)),
        )

        # Channel merging
        self.channel_merging = ChannelMergingBlock(
            in_channels=int(self.filters_per_branch * len(self.scales_samples) / 4),
            filters=int(self.filters_per_branch * len(self.scales_samples) / 4),
            groups=int(self.filters_per_branch * len(self.scales_samples) / 8),
            activation=self.activation,
            drop_prob=self.drop_prob,
        )

        # Temporal merging
        self.temporal_merging = TemporalMergingBlock(
            in_channels=int(self.filters_per_branch * len(self.scales_samples) / 4),
            filters=int(self.filters_per_branch * len(self.scales_samples) / 2),
            groups=int(self.filters_per_branch * len(self.scales_samples) / 4),
            activation=self.activation,
            drop_prob=self.drop_prob,
        )

        # Output layers
        self.output_blocks = nn.Sequential(
            OutputBlock(
                in_channels=int(self.filters_per_branch * len(self.scales_samples) / 2),
                activation=self.activation,
                drop_prob=self.drop_prob,
            ),
            nn.Flatten(),
        )

        # Final fully connected layer
        self.final_layer = nn.Linear(
            in_features=int(
                int(self.filters_per_branch * len(self.scales_samples) / 2) * 2
            ),
            out_features=self.n_outputs,
        )

    def _create_inception_block(
        self,
        in_channels: int,
        scales_samples: List[int],
        filters_per_branch: int,
        ncha: int,
        average_pool: int,
        init: bool = False,
    ):
        return InceptionBlock(
            in_channels=in_channels,
            scales_samples=scales_samples,
            filters_per_branch=filters_per_branch,
            ncha=ncha,
            activation=self.activation,
            drop_prob=self.drop_prob,
            average_pool=average_pool,
            spatial_resnet_repetitions=self.spatial_resnet_repetitions,
            init=init,
        )

    def _create_residual_block(
        self, in_channels: int, filters: int, kernel_size: int, average_pool: int
    ):
        return ResidualBlock(
            in_channels=in_channels,
            filters=filters,
            kernel_size=kernel_size,
            activation=self.activation,
            drop_prob=self.drop_prob,
            average_pool=average_pool,
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_channels, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_classes).
        """
        # Reshape and split into left and right hemispheres
        x = x[  # (batch_size, 1, 2, n_channels_per_hemi, n_times)
            :,
            (self.left_idx + self.middle_idx, self.right_idx + self.middle_idx),
            :,
        ].unsqueeze(1)

        # Initial inception modules
        x = self.inception_block1([x])
        x = self.inception_block2(x)

        # Residual blocks
        x = [self.residual_blocks(xi) for xi in x]

        # Temporal reduction
        x = [self.temporal_reduction(xi) for xi in x]

        # Channel merging
        x = [self.channel_merging(xi) for xi in x]

        # Temporal merging
        x = [self.temporal_merging(xi) for xi in x]

        # Output blocks
        x = [self.output_blocks(xi) for xi in x]

        # Concatenate outputs
        x = torch.cat(x, dim=1)

        # Final fully connected layer
        x = self.final_layer(x)

        return x


class InceptionBlock(nn.Module):
    """
    Inception module used in EEGSym architecture.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    scales_samples : list of int
        List of sample sizes for the temporal convolution kernels.
    filters_per_branch : int
        Number of filters in each inception branch.
    ncha : int
        Number of input channels.
    activation : nn.Module
        Activation function to use.
    drop_prob : float
        Dropout probability.
    average_pool : int
        Kernel size for average pooling.
    spatial_resnet_repetitions : int
        Number of repetitions of the spatial analysis operations.
    residual : bool
        If True, includes residual connections.
    init : bool
        If True, applies channel merging operation if residual is False.
    """

    def __init__(
        self,
        in_channels: int,
        scales_samples: List[int],
        filters_per_branch: int,
        ncha: int,
        activation: nn.Module,
        drop_prob: float,
        average_pool: int,
        spatial_resnet_repetitions: int,
        init: bool = False,
    ):
        super().__init__()
        self.activation = activation
        self.drop_prob = drop_prob
        self.average_pool = average_pool
        self.init = init

        # Temporal convolutions
        self.temporal_convs = nn.ModuleList()
        for scale in scales_samples:
            self.temporal_convs.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=filters_per_branch,
                        kernel_size=(1, 1, scale),
                        padding=(0, 0, scale // 2),
                    ),
                    nn.BatchNorm3d(filters_per_branch),
                    activation,
                    nn.Dropout(drop_prob),
                )
            )

        # Spatial convolutions
        if ncha != 1:
            self.spatial_convs = nn.ModuleList()
            for _ in range(spatial_resnet_repetitions):
                self.spatial_convs.append(
                    nn.Sequential(
                        nn.Conv3d(
                            in_channels=filters_per_branch * len(scales_samples),
                            out_channels=filters_per_branch * len(scales_samples),
                            kernel_size=(1, ncha, 1),
                            groups=filters_per_branch * len(scales_samples),
                            padding=(0, 0, 0),
                        ),
                        nn.BatchNorm3d(filters_per_branch * len(scales_samples)),
                        activation,
                        nn.Dropout(drop_prob),
                    )
                )

        self.pool = (
            nn.AvgPool3d(kernel_size=(1, 1, average_pool))
            if average_pool != 1
            else nn.Identity()
        )

    def forward(self, x_list):
        outputs = []
        for x in x_list:
            # Apply temporal convolutions
            temp_outputs = [conv(x) for conv in self.temporal_convs]
            x_out = torch.cat(temp_outputs, dim=1)

            # Residual connection
            x_out = x_out + x

            # Average pooling
            x_out = self.pool(x_out)

            # Apply spatial convolutions
            if hasattr(self, "spatial_convs"):
                for spatial_conv in self.spatial_convs:
                    if self.init:
                        x_out = spatial_conv(x_out)
                    else:
                        x_spatial = spatial_conv(x_out)
                        x_out = x_out + x_spatial

        outputs.append(x_out)
        return outputs


class ResidualBlock(nn.Module):
    """
    Residual block used in EEGSym architecture.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    filters : int
        Number of filters for the convolutional layers.
    kernel_size : int
        Kernel size for the temporal convolution.
    activation : nn.Module
        Activation function to use.
    drop_prob : float
        Dropout probability.
    average_pool : int
        Kernel size for average pooling.
    spatial_resnet_repetitions : int
        Number of repetitions of the spatial analysis operations.
    residual : bool
        If True, includes residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        filters: int,
        kernel_size: int,
        activation: nn.Module,
        drop_prob: float,
        average_pool: int,
    ):
        super().__init__()
        self.activation = activation
        self.drop_prob = drop_prob

        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=(1, 1, kernel_size),
                padding=(0, 0, kernel_size // 2),
            ),
            nn.BatchNorm3d(filters),
            activation,
            nn.Dropout(drop_prob),
        )

        # Average pooling
        self.avg_pool = nn.AvgPool3d(kernel_size=(1, 1, average_pool))

    def forward(self, x):
        x_res = self.temporal_conv(x)
        x_res = x_res[..., : x.shape[-1]] + x
        x_out = self.avg_pool(x_res)
        return x_out


class TemporalBlock(nn.Module):
    """
    Temporal reduction block used in EEGSym architecture.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    filters : int
        Number of filters for the convolutional layers.
    kernel_size : int
        Kernel size for the temporal convolution.
    activation : nn.Module
        Activation function to use.
    drop_prob : float
        Dropout probability.
    residual : bool
        If True, includes residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        filters: int,
        kernel_size: int,
        activation: nn.Module,
        drop_prob: float,
    ):
        super().__init__()
        self.activation = activation
        self.drop_prob = drop_prob

        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=(1, 1, kernel_size),
                padding=(0, 0, kernel_size // 2),
            ),
            nn.BatchNorm3d(filters),
            activation,
            nn.Dropout(drop_prob),
        )

    def forward(self, x):
        x_res = self.conv(x)
        x_res = x_res + x
        return x_res


class ChannelMergingBlock(nn.Module):
    """
    Channel merging block used in EEGSym architecture.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    filters : int
        Number of filters for the convolutional layers.
    groups : int
        Number of groups for the convolutional layers.
    activation : nn.Module
        Activation function to use.
    drop_prob : float
        Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        filters: int,
        groups: int,
        activation: nn.Module,
        drop_prob: float,
    ):
        super().__init__()
        self.activation = activation
        self.drop_prob = drop_prob

        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=(2, 1, 1),
                groups=groups,
                padding=(0, 0, 0),
            ),
            nn.BatchNorm3d(filters),
            activation,
            nn.Dropout(drop_prob),
        )

    def forward(self, x):
        x_res = self.conv(x)
        x_res = x_res + x
        return x_res


class TemporalMergingBlock(nn.Module):
    """
    Temporal merging block used in EEGSym architecture.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    filters : int
        Number of filters for the convolutional layers.
    groups : int
        Number of groups for the convolutional layers.
    activation : nn.Module
        Activation function to use.
    drop_prob : float
        Dropout probability.
    residual : bool
        If True, includes residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        filters: int,
        groups: int,
        activation: nn.Module,
        drop_prob: float,
    ):
        super().__init__()
        self.activation = activation
        self.drop_prob = drop_prob

        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=(1, 1, 1),
                groups=groups,
                padding=(0, 0, 0),
            ),
            nn.BatchNorm3d(filters),
            activation,
            nn.Dropout(drop_prob),
        )

    def forward(self, x):
        x_res = self.conv(x)
        x_res = x_res + x
        return x_res


class OutputBlock(nn.Module):
    """
    Output block used in EEGSym architecture.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    activation : nn.Module
        Activation function to use.
    drop_prob : float
        Dropout probability.
    residual : bool
        If True, includes residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        activation: nn.Module,
        drop_prob: float,
    ):
        super().__init__()
        self.activation = activation
        self.drop_prob = drop_prob

        self.conv_blocks = nn.ModuleList()
        for _ in range(4):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    nn.BatchNorm3d(in_channels),
                    activation,
                    nn.Dropout(drop_prob),
                )
            )

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x_res = conv_block(x)
            x = x + x_res
        return x


def match_hemisphere_chans(left_chs, right_chs):
    """
    This function matches the channels of the left and right hemispheres based on their names.
    It returns a list of tuples with matched channel names.

    Parameters
    ----------
    left_chs : list of str
        A list of channel names from the left hemisphere.
    right_chs : list of str
        A list of channel names from the right hemisphere.

    Returns
    -------
    list of tuples
        Returns a list of tuples with matched channel names from the left and right hemispheres.
    Raises
    ------
    ValueError
        If the left anr right channels do not match.
    """
    if len(left_chs) != len(right_chs):
        raise ValueError("Left and right channels do not match.")
    right_chs = list(right_chs)
    regexp = r"\d+"
    out = []
    for left in left_chs:
        match = re.search(regexp, left)
        if match is None:
            raise ValueError(f"Channel '{left}' does not contain a number.")
        chan_idx = 1 + int(match.group())
        target_r = re.sub(regexp, str(chan_idx), left)
        for right in right_chs:
            if right == target_r:
                out.append((left, right))
                right_chs.remove(right)
                break
        else:
            raise ValueError(
                f"Found no right hemisphere matching channel for '{left}'."
            )
    return out


def division_channels_idx(ch_names):
    """
    This function divides a list of EEG channel names into three lists: left,
    right, and middle, based on their names.  It categorizes each channel
    by its number: odd-numbered channels go into the left list, even-numbered
    channels into the right list, and channels without numbers into the
    middle list.

    Parameters
    ----------
    ch_names : list of str
        A list of EEG channel names to be divided based on their numbering and arranged.

    Returns
    -------
    tuple of lists
        Returns three lists containing the left, right, and middle channel names in the original list:
        - left: Odd-numbered channels.
        - right: Even-numbered channels.
        - middle: Channels that do not contain numbers.

    Notes
    -----
    The function identifies channel numbers by searching for numeric characters in the channel names.
    Odd-numbered channels are classified as left, even-numbered as right, and channels without numbers go into the middle list.
    Each list is sorted by the channel's y-coordinate if 'front_to_back' is set to True.

    Examples
    --------
    >>> channels = ['FP1', 'FP2', 'O1', 'O2', 'FZ']
    >>> division_channels_idx(channels)
    (['FP1, 'O1'], ['FP2', 'O2'], ['Fz'])
    """
    left, right, middle = [], [], []
    for ch in ch_names:
        number = search(r"\d+", ch)
        if number is not None:
            (left if int(number[0]) % 2 else right).append(ch)
        else:
            middle.append(ch)

    return left, right, middle


if __name__ == "__main__":
    ch_names = ["FP1", "FP2", "O1", "O2", "FZ"]
    chs_info = [{"ch_name": ch} for ch in ch_names]
    x = torch.zeros(1, 5, 1000)

    model = EEGSym(chs_info=chs_info, n_times=1000, n_outputs=2, sfreq=250)

    with torch.no_grad():
        out = model(x)
