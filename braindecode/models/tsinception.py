# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

from __future__ import annotations

import torch
import torch.nn as nn

from braindecode.models.base import EEGModuleMixin


class TSception(EEGModuleMixin, nn.Module):
    """TSception model for EEG signal classification.

    Temporal-Spatial Convolutional Neural Network (TSception) as described in
    [1]_. This model is designed to capture both temporal and spatial
    features of EEG signals for tasks like emotion recognition.

    Code from: https://github.com/deepBrains/TSception

    TO-DO: put warning and note

    Parameters
    ----------
    num_temporal_filters : int, optional
        Number of temporal convolutional filters. Default is 9.
    num_spatial_filters : int, optional
        Number of spatial convolutional filters. Default is 6.
    hidden_size : int, optional
        Number of units in the hidden fully connected layer. Default is 128.
    dropout : float, optional
        Dropout rate. Default is 0.5.
    pooling_size : int, optional
        Pooling size. Default is 8.
    activation : nn.Module, optional
        Activation function. Default is `nn.LeakyReLU()`.

    References
    ----------
    .. [1] Ding, Y., Robinson, N., Zhang, S., Zeng, Q., & Guan, C. (2022).
           Tsception: Capturing temporal dynamics and spatial asymmetry from
           EEG for emotion recognition. IEEE Transactions on Affective Computing,
           14(3), 2238-2250.
           https://ieeexplore.ieee.org/document/9762054
    """

    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        input_window_seconds=None,
        sfreq=None,
        chs_info=None,
        n_times=None,
        num_temporal_filters: int = 9,
        num_spatial_filters: int = 6,
        hidden_size: int = 128,
        dropout: float = 0.5,
        pooling_size: int = 8,
        activation: nn.Module = nn.LeakyReLU(),
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

        # Inception windows in seconds
        self.inception_windows = [0.5, 0.25, 0.125]
        self.pooling_size = pooling_size
        self.num_temporal_filters = num_temporal_filters
        self.num_spatial_filters = num_spatial_filters
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation

        # Temporal Convolutional Layers (Tception)
        self.tception_layers = nn.ModuleList()
        for window in self.inception_windows:
            kernel_size = (1, int(window * self.sfreq))
            self.tception_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=num_temporal_filters,
                        kernel_size=kernel_size,
                        stride=1,
                    ),
                    activation,
                    nn.AvgPool2d(
                        kernel_size=(1, self.pooling_size),
                        stride=(1, self.pooling_size),
                    ),
                )
            )

        # Batch Normalization after Tception
        self.bn_t = nn.BatchNorm2d(num_temporal_filters)

        # Spatial Convolutional Layers (Sception)
        self.sception1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_temporal_filters,
                out_channels=num_spatial_filters,
                kernel_size=(self.n_chans, 1),
                stride=1,
            ),
            activation,
            nn.AvgPool2d(
                kernel_size=(1, max(1, int(self.pooling_size * 0.25))),
                stride=(1, max(1, int(self.pooling_size * 0.25))),
            ),
        )

        self.sception2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_temporal_filters,
                out_channels=num_spatial_filters,
                kernel_size=(max(1, self.n_chans // 2), 1),
                stride=(max(1, self.n_chans // 2), 1),
            ),
            activation,
            nn.AvgPool2d(
                kernel_size=(1, max(1, int(self.pooling_size * 0.25))),
                stride=(1, max(1, int(self.pooling_size * 0.25))),
            ),
        )

        # Batch Normalization after Sception
        self.bn_s = nn.BatchNorm2d(num_spatial_filters)

        # Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=num_spatial_filters,
                out_channels=num_spatial_filters,
                kernel_size=(3, 1),
                stride=1,
                padding=(1, 0),  # To maintain the spatial dimension
            ),
            activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
        )
        self.bn_fusion = nn.BatchNorm2d(num_spatial_filters)

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(num_spatial_filters, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.n_outputs),
        )
        print("to-do: put a warning about the channels order")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TSception model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_outputs).
        """
        # x shape: (batch_size, n_chans, n_times)
        x = x.unsqueeze(1)  # (batch_size, 1, n_chans, n_times)

        # Tception layers
        t_features = []
        for layer in self.tception_layers:
            t_out = layer(x)
            t_features.append(t_out)

        out = torch.cat(t_features, dim=-1)  # Concatenate along time dimension
        out = self.bn_t(out)

        # Sception layers
        s_out1 = self.sception1(out)
        s_out2 = self.sception2(out)
        out_combined = torch.cat(
            (s_out1, s_out2), dim=2
        )  # Concatenate along channel dimension
        out = self.bn_s(out_combined)

        # Fusion layer
        out = self.fusion_layer(out)
        out = self.bn_fusion(out)

        # Global average pooling
        out = torch.mean(out, dim=-1)  # Mean over the time dimension
        out = torch.squeeze(out, dim=-1)  # Remove redundant dimension

        # Fully connected layers
        out = self.fc(out)

        return out
