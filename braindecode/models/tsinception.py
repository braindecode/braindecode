# Authors: Bruno Aristimunha <b.aristimunha>
#
# License: BSD (3-clause)

from __future__ import annotations

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from mne.utils import deprecated, warn

from braindecode.models.base import EEGModuleMixin


class TSception(EEGModuleMixin, nn.Module):
    """TSception model from Ding et al. (2020) from [ding2020]_.

    :bdg-success:`Convolution`

    TSception: A deep learning framework for emotion detection using EEG.

    .. figure:: https://user-images.githubusercontent.com/58539144/74716976-80415e00-526a-11ea-9433-02ab2b753f6b.PNG
        :align: center
        :alt: TSception Architecture

    The model consists of temporal and spatial convolutional layers
    (Tception and Sception) designed to learn temporal and spatial features
    from EEG data.

    Parameters
    ----------
    number_filter_temp : int
        Number of temporal convolutional filters.
    number_filter_spat : int
        Number of spatial convolutional filters.
    hidden_size : int
        Number of units in the hidden fully connected layer.
    drop_prob : float
        Dropout rate applied after the hidden layer.
    activation : nn.Module, optional
        Activation function class to apply. Should be a PyTorch activation
        module like ``nn.ReLU`` or ``nn.LeakyReLU``. Default is ``nn.LeakyReLU``.
    pool_size : int, optional
        Pooling size for the average pooling layers. Default is 8.
    inception_windows : list[float], optional
        List of window sizes (in seconds) for the inception modules.
        Default is [0.5, 0.25, 0.125].

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors. The modifications are minimal and the model is expected
    to work as intended. the original code from [code2020]_.

    References
    ----------
    .. [ding2020] Ding, Y., Robinson, N., Zeng, Q., Chen, D., Wai, A. A. P.,
        Lee, T. S., & Guan, C. (2020, July). Tsception: a deep learning framework
        for emotion detection using EEG. In 2020 international joint conference
        on neural networks (IJCNN) (pp. 1-7). IEEE.
    .. [code2020] Ding, Y., Robinson, N., Zeng, Q., Chen, D., Wai, A. A. P.,
        Lee, T. S., & Guan, C. (2020, July). Tsception: a deep learning framework
        for emotion detection using EEG.
        https://github.com/deepBrains/TSception/blob/master/Models.py
    """

    def __init__(
        self,
        # Braindecode parameters
        n_chans=None,
        n_outputs=None,
        input_window_seconds=None,
        chs_info=None,
        n_times=None,
        sfreq=None,
        # Model parameters
        number_filter_temp: int = 9,
        number_filter_spat: int = 6,
        hidden_size: int = 128,
        drop_prob: float = 0.5,
        activation: nn.Module = nn.LeakyReLU,
        pool_size: int = 8,
        inception_windows: tuple[float, float, float] = (0.5, 0.25, 0.125),
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

        self.activation = activation
        self.pool_size = pool_size
        self.inception_windows = inception_windows
        self.number_filter_spat = number_filter_spat
        self.number_filter_temp = number_filter_temp
        self.drop_prob = drop_prob

        ### Layers
        self.ensuredim = Rearrange("batch nchans time -> batch 1 nchans time")
        if self.input_window_seconds < max(self.inception_windows):
            inception_windows = (
                self.input_window_seconds,
                self.input_window_seconds / 2,
                self.input_window_seconds / 4,
            )
            warning_msg = (
                "Input window size is smaller than the maximum inception window size. "
                "We are adjusting the input window size to match the maximum inception window size.\n"
                f"Original input window size: {self.inception_windows}, \n"
                f"Adjusted inception windows: {inception_windows}"
            )
            warn(warning_msg, UserWarning)
            self.inception_windows = inception_windows
        # Define temporal convolutional layers (Tception)
        self.temporal_blocks = nn.ModuleList()
        for window in self.inception_windows:
            # 1. Calculate the temporal kernel size for this block
            kernel_size_t = int(window * self.sfreq)

            # 2. Calculate the output length of the convolution
            conv_out_len = self.n_times - kernel_size_t + 1

            # 3. Ensure the pooling size is not larger than the conv output
            #    and is at least 1.
            dynamic_pool_size = max(1, min(self.pool_size, conv_out_len))

            # 4. Create the block with the dynamic pooling size
            block = self._conv_block(
                in_channels=1,
                out_channels=self.number_filter_temp,
                kernel_size=(1, kernel_size_t),
                stride=1,
                pool_size=dynamic_pool_size,  # Use the dynamic size
                activation=self.activation,
            )
            self.temporal_blocks.append(block)

        self.batch_temporal_lay = nn.BatchNorm2d(self.number_filter_temp)

        # Define spatial convolutional layers (Sception)

        pool_size_spat = self.pool_size // 4

        self.spatial_block_1 = self._conv_block(
            in_channels=self.number_filter_temp,
            out_channels=self.number_filter_spat,
            kernel_size=(self.n_chans, 1),
            stride=1,
            pool_size=pool_size_spat,
            activation=self.activation,
        )

        kernel_size_spat_2 = (max(1, self.n_chans // 2), 1)

        self.spatial_block_2 = self._conv_block(
            in_channels=self.number_filter_temp,
            out_channels=self.number_filter_spat,
            kernel_size=kernel_size_spat_2,
            stride=kernel_size_spat_2,
            pool_size=pool_size_spat,
            activation=self.activation,
        )
        self.batch_spatial_lay = nn.BatchNorm2d(self.number_filter_spat)

        # Calculate the size of the features after convolution and pooling layers
        self.feature_size = self._calculate_feature_size()
        # self.feature_size = self.number_filter_spat *
        # Define the final classification layers

        self.dense_layer = nn.Sequential(
            nn.Linear(self.feature_size, hidden_size),
            self.activation(),
            nn.Dropout(self.drop_prob),
        )

        self.final_layer = nn.Linear(hidden_size, self.n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TSception model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_channels, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_classes).
        """
        # Temporal Convolution
        # shape: (batch_size, n_channels, n_times)
        x = self.ensuredim(x)
        # shape: (batch_size, 1, n_channels, n_times)

        t_features = [layer(x) for layer in self.temporal_blocks]
        # shape: (batch_size, number_filter_temp, n_channels,
        #
        t_out = torch.cat(t_features, dim=-1)

        t_out = self.batch_temporal_lay(t_out)

        # Spatial Convolution
        s_out1 = self.spatial_block_1(t_out)
        s_out2 = self.spatial_block_2(t_out)
        s_out = torch.cat((s_out1, s_out2), dim=2)
        s_out = self.batch_spatial_lay(s_out)

        # Flatten and apply final layers
        s_out = s_out.view(s_out.size(0), -1)
        output = self.dense_layer(s_out)
        output = self.final_layer(output)
        return output

    def _calculate_feature_size(self) -> int:
        """
        Calculates the size of the features after convolution and pooling layers.

        Returns
        -------
        int
            Flattened size of the features after convolution and pooling layers.
        """
        with torch.no_grad():
            dummy_input = torch.ones(1, 1, self.n_chans, self.n_times)
            t_features = [layer(dummy_input) for layer in self.temporal_blocks]
            t_out = torch.cat(t_features, dim=-1)
            t_out = self.batch_temporal_lay(t_out)

            s_out1 = self.spatial_block_1(t_out)
            s_out2 = self.spatial_block_2(t_out)
            s_out = torch.cat((s_out1, s_out2), dim=2)
            s_out = self.batch_spatial_lay(s_out)

            feature_size = s_out.view(1, -1).size(1)
        return feature_size

    @staticmethod
    def _conv_block(
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple[int, int] | int,
        pool_size: int,
        activation: nn.Module,
    ) -> nn.Sequential:
        """
        Creates a convolutional block with Conv2d, activation, and AvgPool2d layers.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : tuple
            Size of the convolutional kernel.
        stride : int
            Stride of the convolution.
        pool_size : int
            Size of the pooling kernel.
        activation : nn.Module
            Activation function class.

        Returns
        -------
        nn.Sequential
            A sequential container of the convolutional block.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
            ),
            activation(),
            nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_size)),
        )


@deprecated(
    "`TSceptionV1` was renamed to `TSception` in v1.12; "
    "this alias will be removed in v1.14."
)
class TSceptionV1(TSception):
    """Deprecated alias for TSception."""

    pass
