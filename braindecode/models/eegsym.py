import torch

import numpy as np
import mne
from mne.channels import make_standard_montage
from re import search

from torch import nn
from typing import List, Tuple
from braindecode.models.base import EEGModuleMixin


class EEGSym(EEGModuleMixin, nn.Module):
    """EEGSym from Sergio et al (2022) [eegsym2022]_.

    XXXXX.

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
    ch_lateral : int, optional
        Number of channels attributed to one hemisphere of the head. Default is 3.
    spatial_resnet_repetitions : int, optional
        Number of repetitions of the spatial analysis operations at each step.
        Default is 1.


    References
    ----------
    .. [eegsym2022] Pérez-Velasco, S., Santamaría-Vázquez, E., Martínez-Cagigal, V.,
       Marcos-Martínez, D., & Hornero, R. (2022). EEGSym: Overcoming inter-subject
       variability in motor imagery based BCIs with deep learning. IEEE Transactions
       on Neural Systems and Rehabilitation Engineering, 30, 1766-1775.
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
        ch_lateral: int = 3,
        spatial_resnet_repetitions: int = 1,
    ):
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
        self.ch_lateral = ch_lateral
        self.spatial_resnet_repetitions = spatial_resnet_repetitions

        # Calculate scales in samples
        self.scales_samples = [int(s * self.sfreq / 1000) for s in scales_time]

        if self.ch_info is None:
            raise ValueError("ch_info must be provided when symmetric is True")
        if ch_lateral < self.n_chans // 2:
            self.superposition = True
        else:
            self.superposition = False
        self.n_channels_per_hemi = self.n_chans - ch_lateral
        self.division = 2

        # Build the model
        self._build_model()

    def _build_model(self):
        # Initial inception modules
        self.inception_block1 = self._create_inception_block(
            in_channels=1,
            scales_samples=self.scales_samples,
            filters_per_branch=self.filters_per_branch,
            ncha=self.n_channels_per_hemi if self.symmetric else self.n_channels,
            average_pool=2,
            init=True,
        )
        self.inception_block2 = self._create_inception_block(
            in_channels=self.filters_per_branch * len(self.scales_samples),
            scales_samples=[max(1, s // 4) for s in self.scales_samples],
            filters_per_branch=self.filters_per_branch,
            ncha=self.n_channels_per_hemi if self.symmetric else self.n_channels,
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
                residual=self.residual,
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
            residual=self.residual,
        )

        # Temporal merging
        self.temporal_merging = TemporalMergingBlock(
            in_channels=int(self.filters_per_branch * len(self.scales_samples) / 4),
            filters=int(self.filters_per_branch * len(self.scales_samples) / 2),
            groups=int(self.filters_per_branch * len(self.scales_samples) / 4),
            activation=self.activation,
            drop_prob=self.drop_prob,
            residual=self.residual,
        )

        # Output layers
        self.output_blocks = nn.Sequential(
            OutputBlock(
                in_channels=int(self.filters_per_branch * len(self.scales_samples) / 2),
                activation=self.activation,
                drop_prob=self.drop_prob,
                residual=self.residual,
            ),
            nn.Flatten(),
        )

        # Final fully connected layer
        self.final_layers = nn.Linear(
            in_features=int(
                int(self.filters_per_branch * len(self.scales_samples) / 2)
                * self.division
            ),
            out_features=self.n_classes,
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
            residual=self.residual,
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
            residual=self.residual,
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
        x = x.unsqueeze(1)  # (batch_size, 1, n_channels, n_times)
        left = x[:, :, : self.ch_lateral, :]
        right = x[:, :, -self.ch_lateral :, :]

        if self.superposition:
            central = x[:, :, self.ch_lateral : -self.ch_lateral, :]
            left = torch.cat((left, central), dim=2)
            right = torch.cat((right, central), dim=2)

        x = torch.cat((left.unsqueeze(1), right.unsqueeze(1)), dim=1)

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
        x = self.final_layers(x)

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
        residual: bool,
        init: bool = False,
    ):
        super().__init__()
        self.residual = residual
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
                        kernel_size=(1, scale, 1),
                        padding=(0, scale // 2, 0),
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
                            kernel_size=(1, 1, ncha),
                            groups=filters_per_branch * len(scales_samples),
                            padding=(0, 0, 0),
                        ),
                        nn.BatchNorm3d(filters_per_branch * len(scales_samples)),
                        activation,
                        nn.Dropout(drop_prob),
                    )
                )

    def forward(self, x_list):
        outputs = []
        for x in x_list:
            # Apply temporal convolutions
            temp_outputs = [conv(x) for conv in self.temporal_convs]
            x_out = torch.cat(temp_outputs, dim=1)

            # Residual connection
            if self.residual:
                x_out = x_out + x

            # Average pooling
            if self.average_pool != 1:
                x_out = nn.AvgPool3d(kernel_size=(1, self.average_pool, 1))(x_out)

            # Apply spatial convolutions
            if hasattr(self, "spatial_convs"):
                for spatial_conv in self.spatial_convs:
                    if self.residual:
                        x_spatial = spatial_conv(x_out)
                        x_out = x_out + x_spatial
                    elif self.init:
                        x_out = spatial_conv(x_out)
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
        residual: bool,
    ):
        super().__init__()
        self.residual = residual
        self.activation = activation
        self.drop_prob = drop_prob

        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=(1, kernel_size, 1),
                padding=(0, kernel_size // 2, 0),
            ),
            nn.BatchNorm3d(filters),
            activation,
            nn.Dropout(drop_prob),
        )

        # Average pooling
        self.avg_pool = nn.AvgPool3d(kernel_size=(1, average_pool, 1))

    def forward(self, x):
        x_res = self.temporal_conv(x)
        if self.residual:
            x_res = x_res + x
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
        residual: bool,
    ):
        super().__init__()
        self.residual = residual
        self.activation = activation
        self.drop_prob = drop_prob

        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=(1, kernel_size, 1),
                padding=(0, kernel_size // 2, 0),
            ),
            nn.BatchNorm3d(filters),
            activation,
            nn.Dropout(drop_prob),
        )

    def forward(self, x):
        x_res = self.conv(x)
        if self.residual:
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
        residual: bool,
    ):
        super().__init__()
        self.residual = residual
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
        if self.residual:
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
        residual: bool,
    ):
        super().__init__()
        self.residual = residual
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
        if self.residual:
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
        residual: bool,
    ):
        super().__init__()
        self.residual = residual
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
            if self.residual:
                x = x + x_res
            else:
                x = x_res
        return x


#### Sérgio codigo


def sort_channels_by_y(channels, front_to_back=True):
    """Sort a list of EEG channels based on their y-coordinate position in the
    10-05 system.

    Parameters
    ------------
    channels: list
        List of EEG channel names.
    front_to_back: bool, optional
        Determines the sorting order, from front to back or from back to front.
        Defaults to True.

    Returns
    ------------
    sorted_channels: list
        Sorted list of EEG channel names.

    """
    montage = make_standard_montage("standard_1005", head_size=1)
    channels_temp = montage.ch_names.copy()
    for item in ["T3", "T4", "T5", "T6"]:
        channels_temp.pop(channels_temp.index(item))
    info = mne.create_info(ch_names=channels_temp, sfreq=120, ch_types="eeg")
    info.set_montage("standard_1005")
    eeg_layout = mne.channels.make_eeg_layout(info)
    layout_names = [ch.upper() for ch in eeg_layout.names]
    positions = eeg_layout.pos[:, :2]
    positions[:, 0] = positions[:, 0] - 0.5

    # montage_2 = mne.channels.montage.transform_to_head(montage)
    # eeg_dictionary_temp = montage_2._get_ch_pos()
    # eeg_dictionary = {key.upper(): value for key, value in eeg_dictionary_temp.items()}

    region_channels = {}
    for channel in channels:
        match = search(r"^[a-zA-Z]+(?=[\d|z|Z])", channel)
        if match:
            prefix = match.group()
            # Replace 'C' with 'T' in the prefix to treat them as the same
            normalized_prefix = prefix.replace("C", "T")
            if normalized_prefix not in region_channels:
                region_channels[normalized_prefix] = []
            region_channels[normalized_prefix].append(channel)

    # sorted_channels = sorted(channels, key=lambda x: (np.mean([
    #     eeg_dictionary[ch.upper()][1] for ch in region_channels[search(r'^['
    #                                                                r'a-zA-Z]+(?=[\d|z|Z])', x).group().replace('C',
    #                                                                             'T')]]),
    #                                                   (int(search(r'\d+', x).group()) if search(r'\d+', x) else 0,
    #              'h' not in x),
    #                          reverse=front_to_back)
    sorted_channels = sorted(
        channels,
        key=lambda x: (
            np.mean(
                [
                    positions[layout_names.index(ch.upper())][1]
                    for ch in region_channels[
                        search(r"^[a-zA-Z]+(?=[\d|z|Z])", x).group().replace("C", "T")
                    ]
                ]
            ),
            (
                int(search(r"\d+", x).group()) if search(r"\d+", x) else 0,
                "h" not in x,
            ),  # Sorting by number in descending order and prioritizing channels without 'h'
        ),
        reverse=front_to_back,
    )

    return sorted_channels


def division_channels_idx(channels, front_to_back=True):
    """
    This function divides a list of EEG channel names into three lists: left,
    right, and middle, based on their indices.  It categorizes each channel
    by its number: odd-numbered channels go into the left list, even-numbered
    channels into the right list, and channels without numbers into the
    middle list. Each list is then sorted by their y-coordinates if specified by
    'front_to_back'. Finally, it returns the indices of the channels in the
    original list for each category.

     Parameters
    ----------
    channels : list of str
        A list of EEG channel names to be divided based on their numbering and arranged.
    front_to_back : bool, optional
        A boolean flag that indicates whether the channels should be sorted from front to back based on their y-coordinates.
        Default is True.

    Returns
    -------
    tuple of lists
        Returns three lists containing the indices of left, right, and middle channel names in the original list:
        - left_idx: Indices of odd-numbered channels.
        - right_idx: Indices of even-numbered channels.
        - middle_idx: Indices of channels that do not contain numbers.

    Notes
    -----
    The function identifies channel numbers by searching for numeric characters in the channel names.
    Odd-numbered channels are classified as left, even-numbered as right, and channels without numbers go into the middle list.
    Each list is sorted by the channel's y-coordinate if 'front_to_back' is set to True.

    Examples
    --------
    >>> channels = ['FP1', 'FP2', 'O1', 'O2', 'FZ']
    >>> division_channels_idx(channels)
    ([0, 2], [1, 3], [4])
    """
    left, right, middle = [], [], []
    for channel in channels:
        number = search(r"\d+", channel)
        if number is not None:
            (left if int(number[0]) % 2 else right).append(channel)
        else:
            middle.append(channel)

    left = sort_channels_by_y(left, front_to_back=front_to_back)
    right = sort_channels_by_y(right, front_to_back=front_to_back)
    middle = sort_channels_by_y(middle, front_to_back=front_to_back)

    left_idx = [list(channels).index(channel) for channel in left]
    right_idx = [list(channels).index(channel) for channel in right]
    middle_idx = [list(channels).index(channel) for channel in middle]

    return left_idx, right_idx, middle_idx
