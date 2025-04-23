import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin


class SincShallowNet(EEGModuleMixin, nn.Module):
    """Sinc-ShallowNet from Borra, D et al (2020) [borra2020]_.

    .. figure:: https://ars.els-cdn.com/content/image/1-s2.0-S0893608020302021-gr2_lrg.jpg
        :align: center
        :alt: SincShallowNet Architecture

    The Sinc-ShallowNet architecture has these fundamental blocks:

    1. **Block 1: Spectral and Spatial Feature Extraction**
        - *Temporal Sinc-Convolutional Layer*:
            Uses parametrized sinc functions to learn band-pass filters,
            significantly reducing the number of trainable parameters by only
            learning the lower and upper cutoff frequencies for each filter.
       - *Spatial Depthwise Convolutional Layer*:
            Applies depthwise convolutions to learn spatial filters for
            each temporal feature map independently, further reducing
            parameters and enhancing interpretability.
       - *Batch Normalization*

    2. **Block 2: Temporal Aggregation**
        - *Activation Function*: ELU
        - *Average Pooling Layer*: Aggregation by averaging spatial dim
        - *Dropout Layer*
        - *Flatten Layer*

    3. **Block 3: Classification**
        - *Fully Connected Layer*: Maps the feature vector to n_outputs.

    **Implementation Notes:**

    - The sinc-convolutional layer initializes cutoff frequencies uniformly
        within the desired frequency range and updates them during training while
        ensuring the lower cutoff is less than the upper cutoff.

    Parameters
    ----------
    num_time_filters : int
        Number of temporal filters in the SincFilter layer.
    time_filter_len : int
        Size of the temporal filters.
    depth_multiplier : int
        Depth multiplier for spatial filtering.
    activation : nn.Module, optional
        Activation function to use. Default is nn.ELU().
    drop_prob : float, optional
        Dropout probability. Default is 0.5.
    first_freq : float, optional
        The starting frequency for the first Sinc filter. Default is 5.0.
    min_freq : float, optional
        Minimum frequency allowed for the low frequencies of the filters. Default is 1.0.
    freq_stride : float, optional
        Frequency stride for the Sinc filters. Controls the spacing between the filter frequencies.
        Default is 1.0.
    padding : str, optional
        Padding mode for convolution, either 'same' or 'valid'. Default is 'same'.
    bandwidth : float, optional
        Initial bandwidth for each Sinc filter. Default is 4.0.
    pool_size : int, optional
        Size of the pooling window for the average pooling layer. Default is 55.
    pool_stride : int, optional
        Stride of the pooling operation. Default is 12.

    Notes
    -----
    This implementation is based on the implementation from [sincshallowcode]_.

    References
    ----------
    .. [borra2020] Borra, D., Fantozzi, S., & Magosso, E. (2020). Interpretable
       and lightweight convolutional neural network for EEG decoding: Application
       to movement execution and imagination. Neural Networks, 129, 55-74.
    .. [sincshallowcode] Sinc-ShallowNet re-implementation source code:
       https://github.com/marcellosicbaldi/SincNet-Tensorflow
    """

    def __init__(
        self,
        num_time_filters: int = 32,
        time_filter_len: int = 33,
        depth_multiplier: int = 2,
        activation: Optional[nn.Module] = nn.ELU,
        drop_prob: float = 0.5,
        first_freq: float = 5.0,
        min_freq: float = 1.0,
        freq_stride: float = 1.0,
        padding: str = "same",
        bandwidth: float = 4.0,
        pool_size: int = 55,
        pool_stride: int = 12,
        # braindecode parameters
        n_chans=None,
        n_outputs=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        chs_info=None,
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

        # Checkers and creating variables
        if activation is None:
            activation = nn.ELU()

        # Define low frequencies for the SincFilter
        low_freqs = torch.arange(
            first_freq,
            first_freq + num_time_filters * freq_stride,
            freq_stride,
            dtype=torch.float32,
        )
        self.n_filters = len(low_freqs)

        if padding.lower() == "valid":
            n_times_after_sinc_filter = self.n_times - time_filter_len + 1
        elif padding.lower() == "same":
            n_times_after_sinc_filter = self.n_times
        else:
            raise ValueError("Padding must be 'valid' or 'same'.")

        size_after_pooling = (
            (n_times_after_sinc_filter - pool_size) // pool_stride
        ) + 1
        flattened_size = num_time_filters * depth_multiplier * size_after_pooling

        # Layers
        self.ensuredims = Rearrange("batch chans times -> batch chans times 1")

        # Block 1: Sinc filter
        self.sinc_filter_layer = _SincFilter(
            low_freqs=low_freqs,
            kernel_size=time_filter_len,
            sfreq=self.sfreq,
            padding=padding,
            bandwidth=bandwidth,
            min_freq=min_freq,
        )

        self.depthwiseconv = nn.Sequential(
            # Matching dim to depth wise conv!
            Rearrange("batch timefil time nfilter -> batch nfilter timefil time"),
            nn.BatchNorm2d(
                self.n_filters, momentum=0.99
            ),  # To match keras implementation
            nn.Conv2d(
                in_channels=self.n_filters,
                out_channels=depth_multiplier * self.n_filters,
                kernel_size=(self.n_chans, 1),
                groups=self.n_filters,
                bias=False,
            ),
        )

        # Block 2: Batch norm, activation, pooling, dropout
        self.temporal_aggregation = nn.Sequential(
            nn.BatchNorm2d(depth_multiplier * self.n_filters, momentum=0.99),
            activation(),
            nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride)),
            nn.Dropout(p=drop_prob),
            nn.Flatten(),
        )

        # Final classification layer
        self.final_layer = nn.Linear(
            flattened_size,
            self.n_outputs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, num_channels, num_samples].

        Returns
        -------
        torch.Tensor
            Output logits of shape [batch_size, num_classes].
        """
        x = self.ensuredims(x)
        x = self.sinc_filter_layer(x)
        x = self.depthwiseconv(x)
        x = self.temporal_aggregation(x)

        return self.final_layer(x)


class _SincFilter(nn.Module):
    """Sinc-Based Convolutional Layer for Band-Pass Filtering from Ravanelli and Bengio (2018) [ravanelli]_.

    The `SincFilter` layer implements a convolutional layer where each kernel is
    defined using a parametrized sinc function.
    This design enforces each kernel to represent a band-pass filter,
    reducing the number of trainable parameters.

    Parameters
    ----------
    low_freqs : torch.Tensor
        Initial low cutoff frequencies for each filter.
    kernel_size : int
        Size of the convolutional kernels (filters). Must be odd.
    sfreq : float
        Sampling rate of the input signal.
    bandwidth : float, optional
        Initial bandwidth for each filter. Default is 4.0.
    min_freq : float, optional
        Minimum frequency allowed for low frequencies. Default is 1.0.
    padding : str, optional
        Padding mode, either 'same' or 'valid'. Default is 'same'.

    References
    ----------
    .. [ravanelli] Ravanelli, M., & Bengio, Y. (2018, December). Speaker
       recognition from raw waveform with sincnet. In 2018 IEEE spoken language
       technology workshop (SLT) (pp. 1021-1028). IEEE.
    """

    def __init__(
        self,
        low_freqs: torch.Tensor,
        kernel_size: int,
        sfreq: float,
        bandwidth: float = 4.0,
        min_freq: float = 1.0,
        padding: str = "same",
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        self.num_filters = low_freqs.numel()
        self.kernel_size = kernel_size
        self.sfreq = sfreq
        self.min_freq = min_freq
        self.padding = padding.lower()

        # Precompute constants
        window = torch.hamming_window(kernel_size, periodic=False)

        self.register_buffer("window", window[: kernel_size // 2].unsqueeze(-1))

        n_pi = (
            torch.arange(-(kernel_size // 2), 0, dtype=torch.float32)
            / sfreq
            * 2
            * math.pi
        )
        self.register_buffer("n_pi", n_pi.unsqueeze(-1))

        # Initialize learnable parameters
        bandwidths = torch.full((1, self.num_filters), bandwidth)
        self.bandwidths = nn.Parameter(bandwidths)
        self.low_freqs = nn.Parameter(low_freqs.unsqueeze(0))

        # Constant tensor of ones for filter construction
        self.register_buffer("ones", torch.ones(1, 1, 1, self.num_filters))

    def build_sinc_filters(self) -> torch.Tensor:
        """Builds the sinc filters based on current parameters."""
        # Computing the low frequencies of the filters
        low_freqs = self.min_freq + torch.abs(self.low_freqs)
        # Setting a minimum band and minimum freq
        high_freqs = torch.clamp(
            low_freqs + torch.abs(self.bandwidths),
            min=self.min_freq,
            max=self.sfreq / 2.0,
        )
        bandwidths = high_freqs - low_freqs

        # Passing from n_ to the corresponding f_times_t domain
        low = self.n_pi * low_freqs  # [kernel_size // 2, num_filters]
        high = self.n_pi * high_freqs  # [kernel_size // 2, num_filters]

        filters_left = (torch.sin(high) - torch.sin(low)) / (self.n_pi / 2.0)
        filters_left *= self.window
        filters_left /= 2.0 * bandwidths

        # [1, kernel_size // 2, 1, num_filters]
        filters_left = filters_left.unsqueeze(0).unsqueeze(2)
        filters_right = torch.flip(filters_left, dims=[1])

        filters = torch.cat(
            [filters_left, self.ones, filters_right], dim=1
        )  # [1, kernel_size, 1, num_filters]
        filters = filters / torch.std(filters)
        return filters

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply sinc filters to the input signal.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape [batch_size, num_channels, num_samples, 1].

        Returns
        -------
        torch.Tensor
            Filtered output tensor of shape [batch_size, num_channels, num_samples, num_filters].
        """
        filters = self.build_sinc_filters().to(
            inputs.device
        )  # [1, kernel_size, 1, num_filters]

        # Convert from channels_last to channels_first format
        inputs = inputs.permute(0, 3, 1, 2)
        # Permuting to match conv:
        filters = filters.permute(3, 2, 0, 1)
        # Apply convolution
        outputs = F.conv2d(inputs, filters, padding=self.padding)
        # Changing the dimensional
        outputs = outputs.permute(0, 2, 3, 1)

        return outputs
