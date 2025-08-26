# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

import math

import torch
from torch import nn

from braindecode.models.base import EEGModuleMixin


class SleepStagerChambon2018(EEGModuleMixin, nn.Module):
    """Sleep staging architecture from Chambon et al. (2018) [Chambon2018]_.

    :bdg-success:`Convolution`

    .. figure:: https://braindecode.org/dev/_static/model/SleepStagerChambon2018.jpg
        :align: center
        :alt: SleepStagerChambon2018 Architecture

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
    drop_prob : float
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
    activation: nn.Module, default=nn.ReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.

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
        activation: nn.Module = nn.ReLU,
        input_window_seconds=None,
        n_outputs=5,
        drop_prob=0.25,
        apply_batch_norm=False,
        return_feats=False,
        chs_info=None,
        n_times=None,
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

        self.mapping = {
            "fc.1.weight": "final_layer.1.weight",
            "fc.1.bias": "final_layer.1.bias",
        }

        time_conv_size = math.ceil(time_conv_size_s * self.sfreq)
        max_pool_size = math.ceil(max_pool_size_s * self.sfreq)
        pad_size = math.ceil(pad_size_s * self.sfreq)

        if self.n_chans > 1:
            self.spatial_conv = nn.Conv2d(1, self.n_chans, (self.n_chans, 1))
        else:
            self.spatial_conv = nn.Identity()

        batch_norm = nn.BatchNorm2d if apply_batch_norm else nn.Identity

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, n_conv_chs, (1, time_conv_size), padding=(0, pad_size)),
            batch_norm(n_conv_chs),
            activation(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chs, n_conv_chs, (1, time_conv_size), padding=(0, pad_size)
            ),
            batch_norm(n_conv_chs),
            activation(),
            nn.MaxPool2d((1, max_pool_size)),
        )
        self.return_feats = return_feats

        dim_conv_1 = (
            self.n_times + 2 * pad_size - (time_conv_size - 1)
        ) // max_pool_size
        dim_after_conv = (
            dim_conv_1 + 2 * pad_size - (time_conv_size - 1)
        ) // max_pool_size

        self.len_last_layer = n_conv_chs * self.n_chans * dim_after_conv

        # TODO: Add new way to handle return_features == True
        if not return_feats:
            self.final_layer = nn.Sequential(
                nn.Dropout(p=drop_prob),
                nn.Linear(in_features=self.len_last_layer, out_features=self.n_outputs),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        return self.final_layer(feats)
