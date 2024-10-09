"""Spatial component-wise convolutional network (SCCNet) for motor-imagery EEG
    classification.
Authors: Chun-Shu Wei
         Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)

Wei, C. S., Koike-Akino, T., & Wang, Y. (2019, March). Spatial component-wise
convolutional network (SCCNet) for motor-imagery EEG classification. In 2019
9th International IEEE/EMBS Conference on Neural Engineering (NER) (pp. 328-331).
 IEEE.
"""

import math
import numpy as np
import torch
from torch import nn

from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.models.modules import LogActivation


class SCCNet(EEGModuleMixin, nn.Module):
    """SCCNet from Wei, C S (2019) [sccnet]_.

    Spatial component-wise convolutional network (SCCNet) for motor-imagery EEG
    classification.

    .. figure:: https://dt5vp8kor0orz.cloudfront.net/6e3ec5d729cd51fe8acc5a978db27d02a5df9e05/2-Figure1-1.png
       :align: center
       :alt:  Spatial component-wise convolutional network

    Fill here.

    Parameters
    ----------
    n_filters_spat : int, optional
        Number of spatial filters in the first convolutional layer. Default is 22.
    n_filters_spat_filt : int, optional
        Number of spatial filters used as filter in the second convolutional
        layer. Default is 20.
    drop_prob : float, optional
        Dropout probability. Default is 0.5.
    activation : nn.Module, optional
        Activation function after the second convolutional layer. Default is
        logarithm activation.

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper description and
    source that we are not sure if it is correct [sccnetcode]_.


    References
    ----------
    .. [sccnet] Wei, C. S., Koike-Akino, T., & Wang, Y. (2019, March). Spatial
        component-wise convolutional network (SCCNet) for motor-imagery EEG
        classification. In 2019 9th International IEEE/EMBS Conference on
        Neural Engineering (NER) (pp. 328-331). IEEE.
    .. [sccnetcode] Hsieh, C. Y., Chou, J. L., Chang, Y. H., & Wei, C. S.
        XBrainLab: An Open-Source Software for Explainable Artificial
        Intelligence-Based EEG Analysis. In NeurIPS 2023 AI for
        Science Workshop.

    """

    def __init__(
        self,
        # Signal related parameters
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # Model related parameters
        n_filters_spat: int = 22,
        n_filters_spat_filt: int = 20,
        drop_prob: float = 0.5,
        activation: nn.Module = LogActivation,
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
        # Parameters
        self.n_filters_spat = n_filters_spat
        self.n_filters_spat_filt = n_filters_spat_filt
        self.drop_prob = drop_prob
        self.samples_100ms = int(math.floor(self.sfreq * 0.1))
        self.padding_time = int(np.ceil((self.samples_100ms - 1) / 2))

        # Layer calculation
        # Compute the number of features for the final linear layer
        n_times_avgpool = self.n_times + 2 * self.padding_time - self.samples_100ms + 1
        kernel_size_pool = int(self.sfreq / 2)

        w_out_pool = int((n_times_avgpool - kernel_size_pool) / self.samples_100ms) + 1
        num_features = self.n_filters_spat_filt * w_out_pool

        # Layers
        self.ensure_dim = Rearrange("batch nchan times -> batch 1 nchan times")

        if activation is None:
            self.activation = LogActivation()
        else:
            self.activation = activation()

        self.spatial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.n_filters_spat,
            kernel_size=(self.n_chans, 1),
        )

        self.batch_norm1 = nn.BatchNorm2d(self.n_filters_spat)

        self.spatial_filt_conv = nn.Conv2d(
            in_channels=self.n_filters_spat,
            out_channels=self.n_filters_spat_filt,
            kernel_size=(1, self.samples_100ms),
            padding=(0, self.padding_time),
        )
        self.batch_norm2 = nn.BatchNorm2d(self.n_filters_spat_filt)

        self.dropout = nn.Dropout(self.drop_prob)
        self.temporal_smoothing = nn.AvgPool2d(
            kernel_size=(1, int(self.sfreq / 2)),
            stride=(1, self.samples_100ms),
        )

        self.final_layer = nn.Linear(num_features, self.n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: (batch_size, n_chans, n_times)
        x = self.ensure_dim(x)
        # Shape: (batch_size, 1, n_chans, n_times)
        x = self.spatial_conv(x)
        # Shape: (batch_size, n_filters, 1, n_times)
        x = self.batch_norm1(x)
        # Shape: (batch_size, n_filters, 1, n_times)
        x = self.spatial_filt_conv(x)
        # Shape: (batch_size, n_filters, 1, n_times)
        x = self.batch_norm2(x)
        # Shape: (batch_size, n_filters, 1, n_times)
        x = torch.pow(x, 2)
        # Shape: (batch_size, n_filters, 1, n_times)
        x = self.dropout(x)
        # Shape: (batch_size, n_filters, 1, n_times)
        x = self.temporal_smoothing(x)
        # Shape: (batch_size, n_filters, 1, pool_size)
        x = self.activation(x)
        # Shape: (batch_size, n_filters, 1, pool_size)
        x = x.view(x.size(0), -1)
        # Shape: (batch_size, n_outputs)
        x = self.final_layer(x)
        return x
