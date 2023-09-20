# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

import torch
from torch import nn
import numpy as np
from .base import EEGModuleMixin, deprecated_args


class SleepStagerChambon2018(EEGModuleMixin, nn.Module):
    """Sleep staging architecture from Chambon et al 2018.

    Convolutional neural network for sleep staging described in [Chambon2018]_.

    Parameters
    ----------
    n_conv_chs : int
        Number of convolutional channels. Set to 8 in [Chambon2018]_.
    time_conv_size_s : float
        Size of filters in temporal convolution layers, in seconds. Set to 0.5
        in [Chambon2018]_ (64 samples at sfreq=128).
    max_pool_size_s : float
        Max pooling size, in seconds. Set to 0.125 in [Chambon2018]_ (16
        samples at sfreq=128).
    pad_size_s : float
        Padding size, in seconds. Set to 0.25 in [Chambon2018]_ (half the
        temporal convolution kernel size).
    dropout : float
        Dropout rate before the output dense layer.
    apply_batch_norm : bool
        If True, apply batch normalization after both temporal convolutional
        layers.
    return_feats : bool
        If True, return the features, i.e. the output of the feature extractor
        (before the final linear layer). If False, pass the features through
        the final linear layer.
    n_channels : int
        Alias for `n_chans`.
    input_size_s:
        Alias for `input_window_seconds`.
    n_classes:
        Alias for `n_outputs`.

    References
    ----------
    .. [Chambon2018] Chambon, S., Galtier, M. N., Arnal, P. J., Wainrib, G., &
           Gramfort, A. (2018). A deep learning architecture for temporal sleep
           stage classification using multivariate and multimodal time series.
           IEEE Transactions on Neural Systems and Rehabilitation Engineering,
           26(4), 758-769.
    """

    def __init__(
            self,
            n_chans=None,
            sfreq=None,
            n_conv_chs=8,
            time_conv_size_s=0.5,
            max_pool_size_s=0.125,
            pad_size_s=0.25,
            input_window_seconds=30,
            n_outputs=5,
            dropout=0.25,
            apply_batch_norm=False,
            return_feats=False,
            chs_info=None,
            n_times=None,
            n_channels=None,
            input_size_s=None,
            n_classes=None,
    ):
        n_chans, n_outputs, input_window_seconds, = deprecated_args(
            self,
            ("n_channels", "n_chans", n_channels, n_chans),
            ("n_classes", "n_outputs", n_classes, n_outputs),
            ("input_size_s", "input_window_seconds", input_size_s, input_window_seconds),
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
        del n_channels, n_classes, input_size_s

        self.mapping = {
            "fc.1.weight": "final_layer.1.weight",
            "fc.1.bias": "final_layer.1.bias"
        }

        time_conv_size = np.ceil(time_conv_size_s * self.sfreq).astype(int)
        max_pool_size = np.ceil(max_pool_size_s * self.sfreq).astype(int)
        pad_size = np.ceil(pad_size_s * self.sfreq).astype(int)

        if self.n_chans > 1:
            self.spatial_conv = nn.Conv2d(1, self.n_chans, (self.n_chans, 1))

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
        self.len_last_layer = self._len_last_layer(self.n_chans, self.n_times)
        self.return_feats = return_feats

        # TODO: Add new way to handle return_features == True
        if not return_feats:
            self.final_layer = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.len_last_layer, self.n_outputs),
            )

    def _len_last_layer(self, n_channels, input_size):
        self.feature_extractor.eval()
        with torch.no_grad():
            out = self.feature_extractor(
                torch.Tensor(1, 1, n_channels, input_size))
        self.feature_extractor.train()
        return len(out.flatten())

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)

        if self.n_chans > 1:
            x = self.spatial_conv(x)
            x = x.transpose(1, 2)

        feats = self.feature_extractor(x).flatten(start_dim=1)

        if self.return_feats:
            return feats
        else:
            return self.final_layer(feats)
