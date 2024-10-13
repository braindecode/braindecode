import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin


class SincFilter(nn.Module):
    """
    Applies a set of learnable sinc filters to the input signal.

    Parameters
    ----------
    low_freqs : torch.Tensor
        Initial low cutoff frequencies for each filter.
    kernel_size : int
        Size of the convolutional kernels (filters). Must be odd.
    sample_rate : float
        Sampling rate of the input signal.
    bandwidth : float, optional
        Initial bandwidth for each filter. Default is 4.0.
    min_freq : float, optional
        Minimum frequency allowed for low frequencies. Default is 1.0.
    padding : str, optional
        Padding mode, either 'same' or 'valid'. Default is 'same'.
    """

    def __init__(
        self,
        low_freqs: torch.Tensor,
        kernel_size: int,
        sample_rate: float,
        bandwidth: float = 4.0,
        min_freq: float = 1.0,
        padding: str = "same",
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        self.num_filters = low_freqs.numel()
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.padding = padding.lower()

        # Precompute constants
        window = torch.hamming_window(kernel_size, periodic=False)
        self.register_buffer("window", window[: kernel_size // 2].unsqueeze(-1))

        n_pi = (
            torch.arange(-(kernel_size // 2), 0, dtype=torch.float32)
            / sample_rate
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
        low_freqs = self.min_freq + torch.abs(self.low_freqs)
        high_freqs = torch.clamp(
            low_freqs + torch.abs(self.bandwidths),
            min=self.min_freq,
            max=self.sample_rate / 2.0,
        )
        bandwidths = high_freqs - low_freqs

        low = self.n_pi * low_freqs  # [kernel_size // 2, num_filters]
        high = self.n_pi * high_freqs  # [kernel_size // 2, num_filters]

        filters_left = (torch.sin(high) - torch.sin(low)) / (self.n_pi / 2.0)
        filters_left *= self.window
        filters_left /= 2.0 * bandwidths

        filters_left = filters_left.unsqueeze(0).unsqueeze(
            2
        )  # [1, kernel_size // 2, 1, num_filters]
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

        # Reshape filters to [num_filters, 1, kernel_size]
        filters = filters.squeeze(0).squeeze(1).permute(3, 2, 1)

        # Reshape inputs to [batch_size * num_channels, 1, num_samples]
        inputs = inputs.squeeze(-1)  # [batch_size, num_channels, num_samples]
        batch_size, num_channels, num_samples = inputs.shape
        inputs = inputs.view(batch_size * num_channels, 1, num_samples)

        # Apply convolution
        if self.padding == "same":
            padding = self.kernel_size // 2
            outputs = F.conv1d(inputs, filters, padding=padding)
        elif self.padding == "valid":
            outputs = F.conv1d(inputs, filters)
        else:
            raise ValueError(f"Unsupported padding mode: {self.padding}")

        # Reshape outputs to [batch_size, num_channels, num_samples, num_filters]
        output_length = outputs.shape[-1]
        outputs = outputs.view(
            batch_size, num_channels, self.num_filters, output_length
        )
        outputs = outputs.permute(
            0, 1, 3, 2
        )  # [batch_size, num_channels, num_samples, num_filters]

        return outputs


class SincShallowNet(EEGModuleMixin, nn.Module):
    """
    Sinc-ShallowNet model adapted for EEG data sampled at 128 Hz.

    Parameters
    ----------
    num_temp_filters : int
        Number of temporal filters in the SincFilter layer.
    temp_filter_size : int
        Size of the temporal filters.
    sample_rate : float
        Sampling rate of the input signal.
    num_spatial_filters_x_temp : int
        Depth multiplier for spatial filtering.
    activation : nn.Module, optional
        Activation function to use. Default is nn.ELU().
    drop_prob : float, optional
        Dropout probability. Default is 0.5.
    """

    def __init__(
        self,
        num_temp_filters: int = 32,
        temp_filter_size: int = 33,
        num_spatial_filters_x_temp: int = 2,
        activation: Optional[nn.Module] = nn.ELU,
        drop_prob: float = 0.5,
        first_freq: float = 5.0,
        freq_stride: float = 1.0,
        n_times=None,
        n_outputs=None,
        chs_info=None,
        n_chans=None,
        sfreq=None,
        input_window_seconds=None,
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

        if activation is None:
            activation = nn.ELU()

        # Define low frequencies for the SincFilter

        low_freqs = torch.arange(
            first_freq,
            first_freq + num_temp_filters * freq_stride,
            freq_stride,
            dtype=torch.float32,
        )

        # Block 1: Sinc filter, batch norm, depthwise spatial convolution
        self.block_1 = nn.Sequential(
            SincFilter(
                low_freqs=low_freqs,
                kernel_size=temp_filter_size,
                sample_rate=self.sfreq,
                padding="valid",
            ),
            nn.BatchNorm2d(self.n_chans),
            nn.Conv2d(
                in_channels=self.n_chans,
                out_channels=self.n_chans * num_spatial_filters_x_temp,
                kernel_size=(1, 1),
                groups=self.n_chans,
                bias=False,
            ),
        )

        # Block 2: Batch norm, activation, pooling, dropout
        self.block_2 = nn.Sequential(
            nn.BatchNorm2d(self.n_chans * num_spatial_filters_x_temp),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 55), stride=(1, 12)),
            nn.Dropout(p=drop_prob),
        )

        # Final classification layer
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self._calculate_flattened_size(
                    self.n_chans, num_spatial_filters_x_temp
                ),
                self.n_outputs,
            ),
        )

    def _calculate_flattened_size(
        self, num_channels: int, depth_multiplier: int
    ) -> int:
        """Calculates the flattened size after the convolutional and pooling layers."""
        # Create a dummy input tensor
        dummy_input = torch.zeros(
            1, num_channels, 128, 1
        )  # Assuming input length of 128 samples
        x = self.block_1(dummy_input)
        x = self.block_2(x)
        return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, num_channels, num_samples, 1].

        Returns
        -------
        torch.Tensor
            Output logits of shape [batch_size, num_classes].
        """
        x = self.block_1(x)
        x = self.block_2(x)
        logits = self.final_layer(x)
        return logits


if __name__ == "__main__":
    x = torch.zeros(1, 22, 1001)

    model = SincShallowNet(n_outputs=2, n_chans=22, n_times=1000, sfreq=128)

    with torch.no_grad():
        out = model(x)
