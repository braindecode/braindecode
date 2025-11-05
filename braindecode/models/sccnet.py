# Authors: Chun-Shu Wei
#         Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: BSD (3-clause)

import math
from warnings import warn

import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import LogActivation


class SCCNet(EEGModuleMixin, nn.Module):
    """SCCNet from Wei, C S (2019) [sccnet]_.

    :bdg-success:`Convolution`

    Spatial component-wise convolutional network (SCCNet) for motor-imagery EEG
    classification.

    .. figure:: https://dt5vp8kor0orz.cloudfront.net/6e3ec5d729cd51fe8acc5a978db27d02a5df9e05/2-Figure1-1.png
       :align: center
       :alt:  Spatial component-wise convolutional network
       :width: 680px

    .. rubric:: Architectural Overview

    SCCNet is a spatial-first convolutional layer that fixes temporal kernels in seconds
    to make its filters correspond to neurophysiologically aligned windows. The model
    comprises four stages:

    1. **Spatial Component Analysis**: Performs convolution spatial filtering
        across all EEG channels to extract spatial components, effectively
        reducing the channel dimension.
    2. **Spatio-Temporal Filtering**: Applies convolution across the spatial
        components and temporal domain to capture spatio-temporal patterns.
    3. **Temporal Smoothing (Pooling)**: Uses average pooling over time to smooth the
       features and reduce the temporal dimension, focusing on longer-term patterns.
    4. **Classification**: Flattens the features and applies a fully connected
       layer.

    .. rubric:: Macro Components

    - `SCCNet.spatial_conv` **(spatial component analysis)**

        - *Operations.*
        - :class:`~torch.nn.Conv2d` with kernel `(n_chans, N_t)` and stride `(1, 1)` on an input reshaped to `(B, 1, n_chans, T)`; typical choice `N_t=1` yields a pure across-channel projection (montage-wide linear spatial filter).
        - Zero padding to preserve time, :class:`~torch.nn.BatchNorm2d`; output has `N_u` component signals shaped `(B, 1, N_u, T)` after a permute step.

    *Interpretability/robustness.* Mimics CSP-like spatial filtering: each learned filter is a channel-weighted component, easing inspection and reducing channel noise.

    - `SCCNet.spatial_filt_conv` **(spatio-temporal filtering)**

        - *Operations.*
        - :class:`~torch.nn.Conv2d` with kernel `(N_u, 12)` over components and time (12 samples ~ 0.1 s at 125 Hz),
        - :class:`~torch.nn.BatchNorm2d`;
        - Nonlinearity is **power-like**: the original paper uses **square** like :class:`~braindecode.models.ShallowFBCSPNet` with the class :class:`~braindecode.modules.LogActivation` as default.
        - :class:`~torch.nn.Dropout` with rate `p=0.5`.

    - *Role.* Learns frequency-selective energy features and inter-component interactions within a 0.1 s context (beta/alpha cycle scale).

    - `SCCNet.temporal_smoothing` **(aggregation + readout)**

        - *Operations.*
        - :class:`~torch.nn.AvgPool2d` with size `(1, 62)` (~ 0.5 s) for temporal smoothing and downsampling
        - :class:`~torch.nn.Flatten`
        - :class:`~torch.nn.Linear` to `n_outputs`.


    .. rubric:: Convolutional Details

    * **Temporal (where time-domain patterns are learned).**
        The second block's kernel length is fixed to 12 samples (≈ 100 ms) and slides with
        stride 1; average pooling `(1, 62)` (≈ 500 ms) integrates power over longer spans.
        These choices bake in short-cycle detection followed by half-second trend smoothing.

    * **Spatial (how electrodes are processed).**
        The first block's kernel spans **all electrodes** `(n_chans, N_t)`. With `N_t=1`,
        it reduces to a montage-wide linear projection, mapping channels → `N_u` components.
        The second block mixes **across components** via kernel height `N_u`.

    * **Spectral (how frequency information is captured).**
        No explicit transform is used; learned **temporal kernels** serve as bandpass-like
        filters, and the **square/log power** nonlinearity plus 0.5 s averaging approximate
        band-power estimation (ERD/ERS-style features).

    .. rubric:: Attention / Sequential Modules

    This model contains **no attention** and **no recurrent units**.

    .. rubric:: Additional Mechanisms

    - :class:`~torch.nn.BatchNorm2d` and zero-padding are applied to both convolutions;
      L2 weight decay was used in the original paper; dropout `p=0.5` combats overfitting.
    - Contrasting with other compact neural network, in EEGNet performs a temporal depthwise conv
      followed by a **depthwise spatial** conv (separable), learning temporal filters first.
      SCCNet inverts this order: it performs a **full spatial projection first** (CSP-like),
      then a short **spatio-temporal** conv with an explicit 0.1 s kernel, followed by
      **power-like** nonlinearity and longer temporal averaging. EEGNet's ELU and
      separable design favor parameter efficiency; SCCNet's second-scale kernels and
      square/log emphasize interpretable **band-power** features.

    - Reference implementation: see [sccnetcode]_.

    .. rubric:: Usage and Configuration

    * **Training from the original authors.**

    * Match window length so that `T` is comfortably larger than pooling length
        (e.g., > 1.5-2 s for MI).
    * Start with standard MI augmentations (channel dropout/shuffle, time reverse)
        and tune `n_spatial_filters` before deeper changes.

    Parameters
    ----------
    n_spatial_filters : int, optional
        Number of spatial filters in the first convolutional layer, variable `N_u` from the
        original paper. Default is 22.
    n_spatial_filters_smooth : int, optional
        Number of spatial filters used as filter in the second convolutional
        layer. Default is 20.
    drop_prob : float, optional
        Dropout probability. Default is 0.5.
    activation : nn.Module, optional
        Activation function after the second convolutional layer. Default is
        logarithm activation.

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
        batch_norm_momentum: float = 0.1,
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
        self.n_spatial_filters = n_spatial_filters
        self.n_spatial_filters_smooth = n_spatial_filters_smooth
        self.drop_prob = drop_prob

        # Original logical for SCCNet
        conv_kernel_time = 0.1  # 100ms
        pool_kernel_time = 0.5  # 500ms

        # Calculate sample-based sizes from time durations
        conv_kernel_samples = int(math.floor(self.sfreq * conv_kernel_time))
        pool_kernel_samples = int(math.floor(self.sfreq * pool_kernel_time))

        # If the input window is too short for the default kernel sizes,
        # scale them down proportionally.
        total_kernel_samples = conv_kernel_samples + pool_kernel_samples

        if self.n_times < total_kernel_samples:
            warning_msg = (
                f"Input window seconds ({self.input_window_seconds:.2f}s) is smaller than the "
                f"model's combined kernel sizes ({(total_kernel_samples / self.sfreq):.2f}s). "
                "Scaling temporal parameters down proportionally."
            )
            warn(warning_msg, UserWarning, stacklevel=2)

            scaling_factor = self.n_times / total_kernel_samples
            conv_kernel_samples = int(math.floor(conv_kernel_samples * scaling_factor))
            pool_kernel_samples = int(math.floor(pool_kernel_samples * scaling_factor))

        # Ensure kernels are at least 1 sample wide
        self.samples_100ms = max(1, conv_kernel_samples)
        self.kernel_size_pool = max(1, pool_kernel_samples)

        num_features = self._calc_num_features()

        # Layers
        self.ensure_dim = Rearrange("batch nchan times -> batch 1 nchan times")

        self.activation = LogActivation() if activation is None else activation()

        self.spatial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.n_spatial_filters,
            kernel_size=(self.n_chans, 1),
        )

        self.spatial_batch_norm = nn.BatchNorm2d(
            self.n_spatial_filters, momentum=batch_norm_momentum
        )

        self.permute = Rearrange(
            "batch filspat nchans time -> batch nchans filspat time"
        )

        self.spatial_filt_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.n_spatial_filters_smooth,
            kernel_size=(self.n_spatial_filters, self.samples_100ms),
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(
            self.n_spatial_filters_smooth, momentum=batch_norm_momentum
        )

        self.dropout = nn.Dropout(self.drop_prob)
        self.temporal_smoothing = nn.AvgPool2d(
            kernel_size=(1, self.kernel_size_pool),
            stride=(1, self.samples_100ms),
        )

        self.final_layer = nn.Linear(num_features, self.n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: (batch_size, n_chans, n_times)
        x = self.ensure_dim(x)
        # Shape: (batch_size, 1, n_chans, n_times)
        x = self.spatial_conv(x)
        # Shape: (batch_size, n_filters, 1, n_times)
        x = self.spatial_batch_norm(x)
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

    def _calc_num_features(self) -> int:
        # Compute the number of features for the final linear layer
        w_out_conv2 = (
            self.n_times - self.samples_100ms + 1  # After second conv layer
        )
        w_out_pool = (
            (w_out_conv2 - self.kernel_size_pool) // self.samples_100ms + 1
            # After pooling layer
        )
        num_features = self.n_spatial_filters_smooth * w_out_pool
        return num_features
