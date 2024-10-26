# Authors: Chun-Shu Wei
#         Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: BSD (3-clause)

import math
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


    1. **Spatial Component Analysis**: Performs convolution spatial filtering
        across all EEG channels to extract spatial components, effectively
        reducing the channel dimension.
    2. **Spatio-Temporal Filtering**: Applies convolution across the spatial
        components and temporal domain to capture spatio-temporal patterns.
    3. **Temporal Smoothing (Pooling)**: Uses average pooling over time to smooth the
       features and reduce the temporal dimension, focusing on longer-term patterns.
    4. **Classification**: Flattens the features and applies a fully connected
       layer.


    Parameters
    ----------
    n_spatial_filters : int, optional
        Number of spatial filters in the first convolutional layer. Default is 22.
    n_spatial_filters_smooth : int, optional
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
    the source that have not been tested [sccnetcode]_.


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
        n_spatial_filters: int = 22,
        n_spatial_filters_smooth: int = 20,
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
        self.n_filters_spat = n_spatial_filters
        self.n_spatial_filters_smooth = n_spatial_filters_smooth
        self.drop_prob = drop_prob

        self.samples_100ms = int(math.floor(self.sfreq * 0.1))
        self.kernel_size_pool = int(self.sfreq * 0.5)
        # Equivalent to 0.5 seconds

        # Compute the number of features for the final linear layer
        w_out_conv2 = (
            self.n_times - self.samples_100ms + 1  # After second conv layer
        )
        w_out_pool = (
            (w_out_conv2 - self.kernel_size_pool) // self.samples_100ms + 1
            # After pooling layer
        )
        num_features = self.n_spatial_filters_smooth * w_out_pool

        # Layers
        self.ensure_dim = Rearrange("batch nchan times -> batch 1 nchan times")

        self.activation = LogActivation() if activation is None else activation()

        self.spatial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.n_filters_spat,
            kernel_size=(self.n_chans, 1),
        )

        self.permute = Rearrange(
            "batch filspat nchans time -> batch nchans filspat time"
        )

        self.spatial_filt_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.n_spatial_filters_smooth,
            kernel_size=(self.n_filters_spat, self.samples_100ms),
            padding=0,
            bias=False,
        )
        # Momentum following keras
        self.batch_norm = nn.BatchNorm2d(self.n_spatial_filters_smooth, momentum=0.9)

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
        x = self.permute(x)
        # Shape: (batch_size, 1, n_filters, n_times)
        x = self.spatial_filt_conv(x)
        # Shape: (batch_size, n_filters_filt, 1, n_times_reduced)
        x = self.batch_norm(x)
        # Shape: (batch_size, n_filters_filt, 1, n_times_reduced)
        x = torch.pow(x, 2)
        # Shape: (batch_size, n_filters_filt, 1, n_times_reduced)
        x = self.dropout(x)
        # Shape: (batch_size, n_filters_filt, 1, n_times_reduced)
        x = self.temporal_smoothing(x)
        # Shape: (batch_size, n_filters_filt, 1, n_times_reduced_avg_pool)
        x = self.activation(x)
        # Shape: (batch_size, n_filters_filt, 1, n_times_reduced_avg_pool)
        x = x.view(x.size(0), -1)
        # Shape: (batch_size, n_filters_filt*n_times_reduced_avg_pool)
        x = self.final_layer(x)
        # Shape: (batch_size, n_outputs)
        return x
