from __future__ import annotations

from typing import Any, List, Tuple, cast

import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.datautil.channel_utils import (
    division_channels_idx,
    match_hemisphere_chans,
)
from braindecode.models.base import EEGModuleMixin


class EEGSym(EEGModuleMixin, nn.Module):
    """EEGSym from Pérez-Velasco et al (2022) [eegsym2022]_.

    :bdg-success:`Convolution` :bdg-dark-line:`Channel`

    .. figure:: ../../docs/_static/model/eegsym.png
        :align: center
        :alt: EEGSym Architecture


    The **EEGSym** is a novel Convolutional Neural Network (CNN) architecture designed for
    Motor Imagery (MI) based Brain-Computer Interfaces (BCIs), primarily aimed at
    **overcoming inter-subject variability** and significantly **reducing BCI inefficiency**
    [eegsym2022]_.

    The architecture integrates advances from Deep Learning (DL), complemented by
    Transfer Learning (TL) techniques and Data Augmentation (DA), to achieve strong
    performance in inter-subject MI classification [eegsym2022]_.

    .. rubric:: Architectural Overview

    EEGSym systematically incorporates three core features:

    #. **Inception Modules** for multi-scale temporal analysis [eegsym2022]_.
    #. **Residual Connections** maintain spatio-temporal signal structure and
       enable deeper feature extraction [eegsym2022]_.
    #. A **Siamese-network design** exploits the inherent symmetry of the brain
       across the mid-sagittal plane [eegsym2022]_.

    .. rubric:: Macro Components

    - `EEGSym.symmetric_division` **(Input Processing)**
        - *Operations.* The input is virtually split into left, right, and middle channels.
           Middle (central) channels are duplicated and concatenated to both left
           and right lateralized electrodes to form the two hemisphere inputs [eegsym2022]_.
        - *Role.* Prepares the data for the siamese-network approach,
           reducing the number of parameters in the spatial filters
           for the tempospatial analysis stage [eegsym2022]_.

    - `EEGSym.inception_block` **(Tempospatial Analysis - Temporal Feature Extraction)**
        - *Operations.* Uses :class:`_InceptionBlock` modules, which apply parallel
          temporal convolutions with different kernel sizes (scales) [eegsym2022]_.
          This is followed by concatenation, residual connections, and average
          pooling for temporal dimensionality reduction [eegsym2022]_.
        - *Role.* Captures detailed temporal relationships in the architecture,
          similarly to :class:`~braindecode.models.eeginception_mi.EEGInceptionMI`
          [eeginception2020]_. The first block uses large temporal kernels
          (e.g., 500 ms, 250 ms, 125 ms) [eegsym2022]_.

    - `EEGSym.residual_blocks` **(Tempospatial Analysis - Spatial Feature Extraction)**
        - *Operations.* Composed of multiple :class:`_ResidualBlock` modules (typically three instances)
          [eegsym2022]_. Each block applies temporal convolution, pooling, and a spatial analysis layer
          (convolution or grouped convolution) [eegsym2022]_.
        - *Role.* Enhances spatial feature extraction by incorporating residual
           connections across all CNN stages, which helps maintain the spatio-temporal
           structure of the signal through deeper layers [eegsym2022]_.

    - `EEGSym.channel_merging` **(Hemisphere Merging)**
        - *Operations.* The :class:`_ChannelMergingBlock` reduces the spatial dimensionality
          (Z and C) to 1, performing two residual convolutions followed by a final grouped
          convolution that merges the feature information from the two hemispheres [eegsym2022]_.
        - *Role.* Extracts complex relationships between channels of both hemispheres as part of the
          symmetry exploitation [eegsym2022]_.

    - `EEGSym.temporal_merging` **(Temporal Collapse)**
        - *Operations.* The :class:`_TemporalMergingBlock` uses residual convolution
          followed by grouped convolution to reduce the temporal dimension (S) to 1 [eegsym2022]_.
        - *Role.* Final step of temporal aggregation before the output module [eegsym2022]_.

    - `EEGSym.output_blocks` **(Output Processing)**
        - *Operations.* The :class:`_OutputBlock` applies four residual convolution iterations
          (1x1x1 convolutions) followed by flattening [eegsym2022]_.
        - *Role.* Final feature refinement through residual connections before the
          fully connected classification layer [eegsym2022]_.

    .. rubric:: How the information is encoded temporally, spatially, and spectrally

    * **Temporal.**
        Temporal features are extracted across multiple scales in the inception modules
        using different temporal convolution kernel sizes (e.g., corresponding to
        500 ms, 250 ms, and 125 ms windows for a 128 Hz sampling rate), very similar to [eeginception2020]_.
        Subsequent pooling operations and residual blocks continue to reduce the temporal dimension
        [eegsym2022]_.

    * **Spatial.**

        Spatial features are extracted via two main mechanisms:

        - (1) The **siamese-network design** implicitly introduces brain symmetry by treating the two hemispheres
          equally during feature extraction [eegsym2022]_.
        - (2) **Residual connections** are utilized in the Tempospatial Analysis stage to enhance the extraction of
          spatial correlations between electrodes [eegsym2022]_.

    * **Spectral.**
        Spectral information is implicitly captured by the varying kernel sizes of the temporal convolutions
        in the inception modules [eegsym2022]_. These kernels filter the signal across different temporal windows,
        corresponding to different frequency characteristics.

    Notes
    ----------
    * EEGSym achieved competitive accuracies across five large MI datasets [eegsym2022]_.
    * The model maintained high accuracy using a reduced set of electrodes (8 or 16 channels)
      [eegsym2022]_.
    * This is PyTorch implementation of the EEGSym model of the TensorFlow original [eegsym2022code]_.

    Parameters
    ----------
    filters_per_branch : int, optional
        Number of filters in each inception branch. Should be a multiple of 8.
        Default is 12 [eegsym2022]_.
    scales_time : tuple of int, optional
        Temporal scales (in milliseconds) for the temporal convolutions in the first
        inception module. Default is (500, 250, 125) [eegsym2022]_.
    drop_prob : float, optional
        Dropout probability. Default is 0.25 [eegsym2022]_.
    activation : type[nn.Module], optional
        Activation function class to use. Default is :class:`nn.ELU` [eegsym2022]_.
    spatial_resnet_repetitions : int, optional
        Number of repetitions of the spatial analysis operations at each step.
        Default is 5 [eegsym2022]_.
    left_right_chs : list of tuple of str, optional
        List of tuples pairing left and right hemisphere channel names,
        e.g., ``[('C3', 'C4'), ('FC5', 'FC6')]``. If not provided, channels
        are automatically split into left/right hemispheres using
        :func:`~braindecode.datautil.channel_utils.division_channels_idx` and
        :func:`~braindecode.datautil.channel_utils.match_hemisphere_chans`.
        Must be provided together with ``middle_chs`` [eegsym2022]_.
    middle_chs : list of str, optional
        List of midline (central) channel names that lie on the mid-sagittal plane,
        e.g., ``['FZ', 'CZ', 'PZ']``. These channels are duplicated and concatenated
        to both hemispheres. If not provided, channels are automatically identified
        using :func:`~braindecode.datautil.channel_utils.division_channels_idx`.
        Must be provided together with ``left_right_chs`` [eegsym2022]_.

    References
    ----------
    .. [eegsym2022] Pérez-Velasco, S., Santamaría-Vázquez, E., Martínez-Cagigal, V.,
       Marcos-Martínez, D., & Hornero, R. (2022). EEGSym: Overcoming inter-subject
       variability in motor imagery based BCIs with deep learning. IEEE Transactions
       on Neural Systems and Rehabilitation Engineering, 30, 1766-1775.
    .. [eegsym2022code] Pérez-Velasco, S., EEGSym source code.
        https://github.com/Serpeve/EEGSym
    .. [eeginception2020] Santamaría-Vázquez, E., Martínez-Cagigal, V.,
       Vaquerizo-Villar, F., & Hornero, R. (2020). EEG-Inception: A novel deep
       convolutional neural network for assistive ERP-based brain-computer interfaces.
       IEEE Transactions on Neural Systems and Rehabilitation Engineering, 28, 2773-2782.
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
        filters_per_branch: int = 12,
        scales_time: Tuple[int, int, int] = (500, 250, 125),
        drop_prob: float = 0.25,
        activation: type[nn.Module] = nn.ELU,
        spatial_resnet_repetitions: int = 5,
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
        self.activation = activation()
        self.spatial_resnet_repetitions = spatial_resnet_repetitions

        # Calculate scales in samples
        self.scales_samples = [int(s * self.sfreq / 2000) * 2 + 1 for s in scales_time]

        # Note: chs_info is actually list[dict] despite base class type hint
        # saying list[str]
        ch_names = [cast(dict[str, Any], ch)["ch_name"] for ch in self.chs_info]
        if left_right_chs is None:
            left_chs, right_chs, middle_chs = division_channels_idx(ch_names)
            try:
                # Try to match hemispheres based on channel naming
                left_chs, right_chs = zip(*match_hemisphere_chans(left_chs, right_chs))
            except (ValueError, IndexError):
                # Fallback: if matching fails, treat all channels as one hemisphere
                # This allows the model to work with arbitrary channel configurations
                left_chs = ch_names
                right_chs = ch_names
                middle_chs = []
        else:
            left_chs, right_chs = zip(*left_right_chs)
            # middle_chs is guaranteed to be not None when left_right_chs is not None
            # (checked in __init__ validation)
            assert middle_chs is not None, (
                "middle_chs must be provided with left_right_chs"
            )

        # Convert to indices and store as tensors for TorchScript compatibility
        left_idx = [ch_names.index(ch) for ch in left_chs]
        right_idx = [ch_names.index(ch) for ch in right_chs]
        middle_idx = [ch_names.index(ch) for ch in middle_chs]

        # Register as buffers (non-trainable tensors) for TorchScript compatibility
        self.register_buffer("left_idx", torch.tensor(left_idx, dtype=torch.long))
        self.register_buffer("right_idx", torch.tensor(right_idx, dtype=torch.long))
        self.register_buffer("middle_idx", torch.tensor(middle_idx, dtype=torch.long))

        self.n_channels_per_hemi = len(left_idx) + len(middle_idx)
        ##################
        # Build the model
        ##################
        self.include_extra_dim = Rearrange("batch channel time -> batch 1 channel time")

        self.permute_layer = Rearrange(
            "batch features z time space -> batch features z space time"
        )

        # Build the model
        self.inception_block1 = _InceptionBlock(
            in_channels=1,
            scales_samples=self.scales_samples,
            filters_per_branch=self.filters_per_branch,
            ncha=self.n_channels_per_hemi,
            activation=self.activation,
            drop_prob=self.drop_prob,
            average_pool=2,
            spatial_resnet_repetitions=self.spatial_resnet_repetitions,
            init=True,
        )
        self.inception_block2 = _InceptionBlock(
            in_channels=self.filters_per_branch * len(self.scales_samples),
            scales_samples=[max(1, s // 4) for s in self.scales_samples],
            filters_per_branch=self.filters_per_branch,
            ncha=self.n_channels_per_hemi,
            activation=self.activation,
            drop_prob=self.drop_prob,
            average_pool=2,
            spatial_resnet_repetitions=self.spatial_resnet_repetitions,
            init=False,
        )

        # Residual blocks (spatial dim is still n_channels_per_hemi through the network)
        self.residual_blocks = nn.Sequential(
            _ResidualBlock(
                in_channels=self.filters_per_branch * len(self.scales_samples),
                filters=self.filters_per_branch
                * len(self.scales_samples),  # No reduction
                kernel_size=16,
                ncha=self.n_channels_per_hemi,
                activation=self.activation,
                drop_prob=self.drop_prob,
                average_pool=2,
                spatial_resnet_repetitions=self.spatial_resnet_repetitions,
            ),
            _ResidualBlock(
                in_channels=self.filters_per_branch * len(self.scales_samples),
                filters=int(
                    self.filters_per_branch * len(self.scales_samples) / 2
                ),  # Reduce by /2
                kernel_size=8,
                ncha=self.n_channels_per_hemi,
                activation=self.activation,
                drop_prob=self.drop_prob,
                average_pool=2,
                spatial_resnet_repetitions=self.spatial_resnet_repetitions,
            ),
            _ResidualBlock(
                in_channels=int(self.filters_per_branch * len(self.scales_samples) / 2),
                filters=int(
                    self.filters_per_branch * len(self.scales_samples) / 4
                ),  # Reduce by /2
                kernel_size=4,
                ncha=self.n_channels_per_hemi,
                activation=self.activation,
                drop_prob=self.drop_prob,
                average_pool=2,
                spatial_resnet_repetitions=self.spatial_resnet_repetitions,
            ),
        )

        # Temporal reduction
        self.temporal_reduction = nn.Sequential(
            _TemporalBlock(
                in_channels=int(self.filters_per_branch * len(self.scales_samples) / 4),
                filters=int(self.filters_per_branch * len(self.scales_samples) / 4),
                kernel_size=4,
                activation=self.activation,
                drop_prob=self.drop_prob,
            ),
            nn.AvgPool3d(kernel_size=(1, 2, 1)),
        )

        # Channel merging
        self.channel_merging = _ChannelMergingBlock(
            in_channels=int(self.filters_per_branch * len(self.scales_samples) / 4),
            filters=int(self.filters_per_branch * len(self.scales_samples) / 4),
            groups=int(
                self.filters_per_branch * len(self.scales_samples) / 12
            ),  # 36/12=3 groups
            ncha=self.n_channels_per_hemi,
            division=2,
            activation=self.activation,
            drop_prob=self.drop_prob,
        )

        # Temporal merging
        # Calculate temporal dimension at this point
        # After: Inc1 (pool/2), Inc2 (pool/2), Res1-3 (pool/2 each), TempRed (pool/2)
        # Total reduction: 2^6 = 64
        temporal_dim_at_merging = self.n_times // 64

        self.temporal_merging = _TemporalMergingBlock(
            in_channels=int(self.filters_per_branch * len(self.scales_samples) / 4),
            filters=int(self.filters_per_branch * len(self.scales_samples) / 2),
            groups=int(self.filters_per_branch * len(self.scales_samples) / 4),
            n_times=temporal_dim_at_merging,
            activation=self.activation,
            drop_prob=self.drop_prob,
        )

        # Output layers
        self.output_blocks = nn.Sequential(
            _OutputBlock(
                in_channels=int(self.filters_per_branch * len(self.scales_samples) / 2),
                activation=self.activation,
                drop_prob=self.drop_prob,
            ),
            nn.Flatten(),
        )

        # Final fully connected layer
        self.final_layer = nn.Linear(
            in_features=int(self.filters_per_branch * len(self.scales_samples) / 2),
            out_features=self.n_outputs,
        )

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_channels, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_classes).
        """
        # Input: (B, C, T) = (batch, channels, time)
        # Step 1: Add feature dimension
        x = self.include_extra_dim(x)  # (B, 1, C, T)

        # Step 2: Split into left, right, and middle channels
        # Use index_select for TorchScript compatibility
        left_data = torch.index_select(x, 2, self.left_idx)  # (B, 1, n_left, T)
        right_data = torch.index_select(x, 2, self.right_idx)  # (B, 1, n_right, T)
        middle_data = torch.index_select(x, 2, self.middle_idx)  # (B, 1, n_middle, T)

        # Step 3: Concatenate middle channels to both hemispheres
        left_hemi = torch.cat(
            [left_data, middle_data], dim=2
        )  # (B, 1, n_left+n_middle, T)
        right_hemi = torch.cat(
            [right_data, middle_data], dim=2
        )  # (B, 1, n_right+n_middle, T)

        # Step 4: Stack along Z dimension
        x = torch.stack([left_hemi, right_hemi], dim=2)  # (B, 1, 2, n_ch_per_hemi, T)

        # Step 5:
        # From: (B, F, Z, Space, Time)
        # To:   (B, F, Z, Time, Space)
        x = self.permute_layer(x)

        # Now x is in correct format: (Batch, Features, Z, Time, Space)

        # Initial inception modules
        x = self.inception_block1([x])[0]  # Returns list, take first element
        x = self.inception_block2([x])[0]  # Returns list, take first element

        # Residual blocks
        x = self.residual_blocks(x)

        # Temporal reduction
        x = self.temporal_reduction(x)

        # Channel merging
        x = self.channel_merging(x)

        # Temporal merging
        x = self.temporal_merging(x)

        # Output blocks
        x = self.output_blocks(x)

        # Final fully connected layer
        x = self.final_layer(x)

        return x


class _InceptionBlock(nn.Module):
    """Inception module used in EEGSym architecture.

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
                            padding=(0, 0, 0),
                        ),
                        nn.BatchNorm3d(filters_per_branch * len(scales_samples)),
                        activation,
                        nn.Dropout(drop_prob),
                    )
                )

        self.pool = (
            nn.AvgPool3d(kernel_size=(1, average_pool, 1))
            if average_pool != 1
            else nn.Identity()
        )

    def forward(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for x in x_list:
            # Apply temporal convolutions
            temp_outputs = [conv(x) for conv in self.temporal_convs]
            x_out = torch.cat(temp_outputs, dim=1)

            # Trim temporal dimension if needed (due to even kernel sizes with padding)
            if x_out.shape[3] > x.shape[3]:
                x_out = x_out[:, :, :, : x.shape[3], :]

            # Residual connection
            x_out = x_out + x

            # Average pooling
            x_out = self.pool(x_out)

            # Apply spatial convolutions
            if hasattr(self, "spatial_convs"):
                for spatial_conv in self.spatial_convs:
                    x_spatial = spatial_conv(x_out)
                    x_out = x_out + x_spatial  # Always use residual connection

            outputs.append(x_out)
        return outputs


class _ResidualBlock(nn.Module):
    """Residual block used in EEGSym architecture.

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
        ncha: int,
        activation: nn.Module,
        drop_prob: float,
        average_pool: int,
        spatial_resnet_repetitions: int = 5,
    ):
        super().__init__()
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

        # Projection layer for dimension matching if needed
        if in_channels != filters:
            self.projection = nn.Conv3d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=(1, 1, 1),
            )
        else:
            self.projection = None

        # Average pooling
        self.avg_pool = nn.AvgPool3d(
            kernel_size=(1, average_pool, 1)
        )  # FIXED: pool Time

        # Spatial convolutions (multiple repetitions like in InceptionBlock)
        if ncha != 1:
            self.spatial_convs = nn.ModuleList()
            for _ in range(spatial_resnet_repetitions):
                self.spatial_convs.append(
                    nn.Sequential(
                        nn.Conv3d(
                            in_channels=filters,
                            out_channels=filters,
                            kernel_size=(1, 1, ncha),  # Spatial convolution
                            padding=(0, 0, 0),
                        ),
                        nn.BatchNorm3d(filters),
                        activation,
                        nn.Dropout(drop_prob),
                    )
                )
        else:
            self.spatial_convs = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = self.temporal_conv(x)

        # Trim temporal dimension if needed (due to even kernel sizes with padding)
        if x_res.shape[3] > x.shape[3]:
            x_res = x_res[:, :, :, : x.shape[3], :]

        # Handle channel dimension mismatch if needed
        if self.projection is not None:
            x = self.projection(x)

        x_out = x_res + x  # Residual connection
        x_out = self.avg_pool(x_out)

        # Apply spatial convolutions if present (multiple repetitions)
        if self.spatial_convs is not None:
            for spatial_conv in self.spatial_convs:
                x_spatial = spatial_conv(x_out)
                x_out = x_out + x_spatial  # Residual connection with broadcasting

        return x_out


class _TemporalBlock(nn.Module):
    """Temporal reduction block used in EEGSym architecture.

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
                kernel_size=(1, kernel_size, 1),
                padding=(0, kernel_size // 2, 0),
            ),
            nn.BatchNorm3d(filters),
            activation,
            nn.Dropout(drop_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = self.conv(x)

        # Trim temporal dimension if needed (due to even kernel sizes with padding)
        if x_res.shape[3] > x.shape[3]:
            x_res = x_res[:, :, :, : x.shape[3], :]

        x_res = x_res + x
        return x_res


class _ChannelMergingBlock(nn.Module):
    """Channel merging block used in EEGSym architecture.

    This block performs hemisphere merging through:
    1. Two residual convolution iterations (with full spatial kernel)
    2. One grouped convolution (merges Z dimension from 2 to 1)

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    filters : int
        Number of filters for the convolutional layers.
    groups : int
        Number of groups for the final grouped convolution.
    ncha : int
        Number of spatial channels to merge.
    division : int
        Z dimension size to merge (typically 2 for two hemispheres).
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
        ncha: int,
        division: int,
        activation: nn.Module,
        drop_prob: float,
    ):
        super().__init__()
        self.activation = activation
        self.drop_prob = drop_prob

        # TWO residual convolution iterations
        # Each reduces spatial dimension: ncha → 1
        self.residual_convs = nn.ModuleList()
        for _ in range(2):
            self.residual_convs.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=filters,
                        kernel_size=(division, 1, ncha),  # (Z, Time, Space)
                        padding=(0, 0, 0),  # Valid padding
                    ),
                    nn.BatchNorm3d(filters),
                    activation,
                    nn.Dropout(drop_prob),
                )
            )

        # Final grouped convolution
        # Merges Z dimension: 2 → 1
        self.grouped_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=(division, 1, ncha),  # (Z, Time, Space)
                groups=groups,
                padding=(0, 0, 0),
            ),
            nn.BatchNorm3d(filters),
            activation,
            nn.Dropout(drop_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply 2 residual iterations
        # Each iteration: conv reduces dims, then Add broadcasts back
        for residual_conv in self.residual_convs:
            x_res = residual_conv(x)
            x = x + x_res  # Broadcasts x_res (1,T,1) to match x (2,T,5)

        # Apply final grouped conv (permanently reduces dimensions)
        x = self.grouped_conv(x)

        return x


class _TemporalMergingBlock(nn.Module):
    """Temporal merging block used in EEGSym architecture.

    This block performs temporal dimension collapse through:
    1. One residual convolution (temporal collapse with residual connection)
    2. One grouped convolution (temporal collapse + double filters)

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    filters : int
        Number of output filters (should be 2x input channels).
    groups : int
        Number of groups for the grouped convolution.
    n_times : int
        Current temporal dimension size.
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
        n_times: int,
        activation: nn.Module,
        drop_prob: float,
    ):
        super().__init__()
        self.activation = activation
        self.drop_prob = drop_prob

        # Calculate temporal kernel size
        # At this point in network, temporal dim has been reduced by pooling
        self.temporal_kernel = n_times  # Should be 6 for 384 input samples

        # Residual convolution (collapses time dimension)
        self.residual_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,  # Same channels for residual
                kernel_size=(1, self.temporal_kernel, 1),  # (Z, Time, Space)
                padding=(0, 0, 0),  # Valid padding - reduces time to 1
            ),
            nn.BatchNorm3d(in_channels),
            activation,
            nn.Dropout(drop_prob),
        )

        # Grouped convolution (collapses time dimension, doubles filters)
        self.grouped_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=filters,  # Double the channels
                kernel_size=(1, self.temporal_kernel, 1),  # (Z, Time, Space)
                groups=groups,
                padding=(0, 0, 0),
            ),
            nn.BatchNorm3d(filters),
            activation,
            nn.Dropout(drop_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual convolution with broadcasting
        x_res = self.residual_conv(x)
        x = x + x_res  # Broadcasts x_res (1,1,1) back to x shape (1,6,1)

        # Grouped convolution (reduces time to 1, doubles channels)
        x = self.grouped_conv(x)

        return x


class _OutputBlock(nn.Module):
    """Output block used in EEGSym architecture.

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
        n_residual: int = 4,
    ):
        super().__init__()
        self.activation = activation
        self.drop_prob = drop_prob

        self.conv_blocks = nn.ModuleList()
        for _ in range(n_residual):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_block in self.conv_blocks:
            x_res = conv_block(x)
            x = x + x_res
        return x
