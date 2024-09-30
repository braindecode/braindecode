from __future__ import annotations

import torch
import torch.nn as nn
from typing import List

from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin


class TSceptionV1(EEGModuleMixin, nn.Module):
    """TSception model from Ding et al. (2020) from [ding2020]_.

    TSception: A deep learning framework for emotion detection using EEG.

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
    inception_windows : List[float], optional
        List of window sizes (in seconds) for the inception modules.
        Default is [0.5, 0.25, 0.125].

    References
    ----------
    [ding2020] Ding, Y., Robinson, N., Zeng, Q., Chen, D., Wai, A. A. P., Lee,
        T. S., & Guan, C. (2020, July). Tsception: a deep learning framework
        for emotion detection using EEG. In 2020 international joint conference
         on neural networks (IJCNN) (pp. 1-7). IEEE.
    [code2020] Ding, Y., Robinson, N., Zeng, Q., Chen, D., Wai, A. A. P., Lee,
        T. S., & Guan, C. (2020, July). Tsception: a deep learning framework
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
        inception_windows: List[float] = [0.5, 0.25, 0.125],
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
        # Define temporal convolutional layers (Tception)
        self.Tception_layers = nn.ModuleList(
            [
                self._conv_block(
                    in_channels=1,
                    out_channels=number_filter_temp,
                    kernel_size=(1, int(window * self.sfreq)),
                    stride=1,
                    pool_size=self.pool_size,
                    activation=self.activation,
                )
                for window in self.inception_windows
            ]
        )

        # Define spatial convolutional layers (Sception)
        self.Sception1 = self._conv_block(
            in_channels=self.number_filter_temp,
            out_channels=self.number_filter_spat,
            kernel_size=(self.n_chans, 1),
            stride=1,
            pool_size=int(self.pool_size * 0.25),
            activation=self.activation,
        )
        self.Sception2 = self._conv_block(
            in_channels=self.number_filter_temp,
            out_channels=self.number_filter_spat,
            kernel_size=(max(1, int(self.n_chans * 0.5)), 1),
            stride=(max(1, int(self.n_chans * 0.5)), 1),
            pool_size=int(self.pool_size * 0.25),
            activation=self.activation,
        )

        self.BN_t = nn.BatchNorm2d(self.number_filter_temp)
        self.BN_s = nn.BatchNorm2d(self.number_filter_spat)

        # Calculate the size of the features after convolution and pooling layers
        # self.feature_size = self._calculate_feature_size(
        #     n_channels, input_window_samples, sampling_rate
        # )

        # Define the final classification layers
        self.final_layers = nn.Sequential(
            nn.Linear(self.feature_size, hidden_size),
            self.activation(),
            nn.Dropout(self.drop_prob),
            nn.Linear(hidden_size, self.n_outputs),
        )

    def _conv_block(
        self,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TSception model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, n_channels, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_classes).
        """
        # Temporal Convolution
        x = self.ensuredim(x)
        t_features = [layer(x) for layer in self.Tception_layers]
        t_out = torch.cat(t_features, dim=-1)
        t_out = self.BN_t(t_out)

        # Spatial Convolution
        s_out1 = self.Sception1(t_out)
        s_out2 = self.Sception2(t_out)
        s_out = torch.cat((s_out1, s_out2), dim=2)
        s_out = self.BN_s(s_out)

        # Flatten and apply final layers
        s_out = s_out.view(s_out.size(0), -1)
        # output = self.final_layers(s_out)
        return s_out

    def _calculate_feature_size(
        self, n_channels: int, input_window_samples: int, sampling_rate: float
    ) -> int:
        """
        Calculates the size of the features after convolution and pooling layers.

        Parameters
        ----------
        n_channels : int
            Number of EEG channels.
        input_window_samples : int
            Number of time samples in the input window.
        sampling_rate : float
            Sampling rate of the EEG data.

        Returns
        -------
        int
            Flattened size of the features after convolution and pooling layers.
        """
        with torch.no_grad():
            dummy_input = torch.ones(1, 1, n_channels, input_window_samples)
            t_features = [layer(dummy_input) for layer in self.Tception_layers]
            t_out = torch.cat(t_features, dim=-1)
            t_out = self.BN_t(t_out)

            s_out1 = self.Sception1(t_out)
            s_out2 = self.Sception2(t_out)
            s_out = torch.cat((s_out1, s_out2), dim=2)
            s_out = self.BN_s(s_out)

            feature_size = s_out.view(1, -1).size(1)
        return feature_size
