# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

import torch
from torch import nn
import numpy as np


class SleepStagerChambon2018(nn.Module):
    """
    Sleep staging architecture from [1]_.

    Convolutional neural network for sleep staging described in [1]_.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    sfreq : float
        EEG sampling frequency.
    n_conv_chs : int
        Number of convolutional channels. Set to 8 in [1]_.
    time_conv_size_s : float
        Size of filters in temporal convolution layers, in seconds. Set to 0.5
        in [1]_ (64 samples at sfreq=128).
    max_pool_size_s : float
        Max pooling size, in seconds. Set to 0.125 in [1]_ (16 samples at
        sfreq=128).
    pad_size_s : float
        Paddind size, in seconds. Set to 0.25 in [1]_ (half the temporal
        convolution kernel size).
    input_size_s : float
        Size of the input, in seconds.
    n_classes : int
        Number of classes.
    dropout : float
        Dropout rate before the output dense layer.
    apply_batch_norm : bool
        If True, apply batch normalization after both temporal convolutional
        layers.

    References
    ----------
    .. [1] Chambon, S., Galtier, M. N., Arnal, P. J., Wainrib, G., &
           Gramfort, A. (2018). A deep learning architecture for temporal sleep
           stage classification using multivariate and multimodal time series.
           IEEE Transactions on Neural Systems and Rehabilitation Engineering,
           26(4), 758-769.
    """
    def __init__(self, n_channels, sfreq, n_conv_chs=8, time_conv_size_s=0.5,
                 max_pool_size_s=0.125, pad_size_s=0.25, input_size_s=30,
                 n_classes=5, dropout=0.25, apply_batch_norm=False):
        super().__init__()

        time_conv_size = np.ceil(time_conv_size_s * sfreq).astype(int)
        max_pool_size = np.ceil(max_pool_size_s * sfreq).astype(int)
        input_size = np.ceil(input_size_s * sfreq).astype(int)
        pad_size = np.ceil(pad_size_s * sfreq).astype(int)

        self.n_channels = n_channels

        if n_channels > 1:
            self.spatial_conv = nn.Conv2d(1, n_channels, (n_channels, 1))

        batch_norm = nn.BatchNorm2d if apply_batch_norm else nn.Identity

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, n_conv_chs, (1, time_conv_size), padding=(0, pad_size)),
            batch_norm(n_conv_chs),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chs, n_conv_chs, (1, time_conv_size),
                padding=(0, pad_size)),
            batch_norm(n_conv_chs),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size))
        )
        self.len_last_layer = self._len_last_layer(n_channels, input_size)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.len_last_layer, n_classes)
        )

    def _len_last_layer(self, n_channels, input_size):
        self.feature_extractor.eval()
        with torch.no_grad():
            out = self.feature_extractor(
                torch.Tensor(1, 1, n_channels, input_size))
        self.feature_extractor.train()
        return len(out.flatten())

    def embed(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        if self.n_channels > 1:
            x = self.spatial_conv(x)
            x = x.transpose(1, 2)

        return self.feature_extractor(x).flatten(start_dim=1)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ---------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        return self.fc(self.embed(x))
