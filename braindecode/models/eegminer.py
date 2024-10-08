"""
* Copyright (C) Cogitat, Ltd.
* Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
* Patent GB2609265 - Learnable filters for eeg classification
* https://www.ipo.gov.uk/p-ipsum/Case/ApplicationNumber/GB2113420.0
"""

import torch
import torch.nn.functional as F
from scipy.fftpack import fftfreq
from torch import nn

from braindecode.models.base import EEGModuleMixin


def torch_hilbert_freq(x, forward_fourier=True):
    """Computes the Hilbert transform using PyTorch,
    with the real and imaginary parts as separate dimensions.

    Input shape (forward_fourier=True): (..., seq_len)
    Input shape (forward_fourier=False): (..., seq_len / 2 + 1, 2)
    Output shape: (..., seq_len, 2)
    """
    if forward_fourier:
        x = torch.fft.rfft(x, norm=None, dim=-1)
        x = torch.view_as_real(x)
    x = x * 2.0
    x[..., 0, :] = x[..., 0, :] / 2.0  # Don't multiply the DC-term by 2
    x = F.pad(
        x, [0, 0, 0, x.shape[-2] - 2]
    )  # Fill Fourier coefficients to retain shape
    x = torch.view_as_complex(x)
    x = torch.fft.ifft(x, norm=None, dim=-1)  # returns complex signal
    x = torch.view_as_real(x)

    return x


def plv_time(x, forward_fourier=True):
    """PLV metric in time domain.
    x (..., channels, time/(freqs, 2)) -> (..., channels, channels)"""
    x_a = torch_hilbert_freq(x, forward_fourier)
    amp = torch.sqrt(x_a[..., 0] ** 2 + x_a[..., 1] ** 2 + 1e-6)
    x_u = x_a / amp.unsqueeze(-1)
    x_u_rr = torch.matmul(x_u[..., 0], x_u[..., 0].transpose(-2, -1))
    x_u_ii = torch.matmul(x_u[..., 1], x_u[..., 1].transpose(-2, -1))
    x_u_ri = torch.matmul(x_u[..., 0], x_u[..., 1].transpose(-2, -1))
    x_u_ir = torch.matmul(x_u[..., 1], x_u[..., 0].transpose(-2, -1))
    r = x_u_rr + x_u_ii
    i = x_u_ri - x_u_ir
    time = amp.shape[-1]
    plv = 1 / time * torch.sqrt(r**2 + i**2 + 1e-6)

    return plv


class GeneralizedGaussianFilter(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        sequence_length,
        sample_rate,
        inverse_fourier=True,
        affine_group_delay=False,
        group_delay=(20.0,),
        f_mean=(23.0,),
        bandwidth=(44.0,),
        shape=(2.0,),
        clamp_f_mean=(1.0, 45.0),
    ):
        super(GeneralizedGaussianFilter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.inverse_fourier = inverse_fourier
        self.affine_group_delay = affine_group_delay
        self.clamp_f_mean = clamp_f_mean
        assert (
            out_channels % in_channels == 0
        ), "out_channels has to be multiple of in_channels"
        assert len(f_mean) * in_channels == out_channels
        assert len(bandwidth) * in_channels == out_channels
        assert len(shape) * in_channels == out_channels

        # Range from 0 to half sample rate, normalized
        self.n_range = nn.Parameter(
            torch.tensor(
                list(
                    fftfreq(n=sequence_length, d=1 / sample_rate)[
                        : sequence_length // 2
                    ]
                )
                + [sample_rate / 2]
            )
            / (sample_rate / 2),
            requires_grad=False,
        )

        # Trainable filter parameters
        self.f_mean = nn.Parameter(
            torch.tensor(f_mean * in_channels) / (sample_rate / 2), requires_grad=True
        )
        self.bandwidth = nn.Parameter(
            torch.tensor(bandwidth * in_channels) / (sample_rate / 2),
            requires_grad=True,
        )  # full width half maximum
        self.shape = nn.Parameter(torch.tensor(shape * in_channels), requires_grad=True)

        # Normalize group delay so that group_delay=1 corresponds to 1000ms
        self.group_delay = nn.Parameter(
            torch.tensor(group_delay * in_channels) / 1000,
            requires_grad=affine_group_delay,
        )

        # Construct filters from parameters
        self.filters = self.construct_filters()

    def exponential_power(self, x, mean, fwhm, shape):
        mean = mean.unsqueeze(1)
        fwhm = fwhm.unsqueeze(1)
        shape = shape.unsqueeze(1)
        log2 = torch.log(torch.tensor(2.0, device=x.device, dtype=x.dtype))
        scale = fwhm / (2 * log2 ** (1 / shape))
        # Add small constant to difference between x and mean since grad of 0 ** shape is nan
        return torch.exp(-((((x - mean).abs() + 1e-8) / scale) ** shape))

    def construct_filters(self):
        # Clamp parameters
        self.f_mean.data = torch.clamp(
            self.f_mean.data,
            min=self.clamp_f_mean[0] / (self.sample_rate / 2),
            max=self.clamp_f_mean[1] / (self.sample_rate / 2),
        )
        self.bandwidth.data = torch.clamp(
            self.bandwidth.data, min=1.0 / (self.sample_rate / 2), max=1.0
        )
        self.shape.data = torch.clamp(self.shape.data, min=2.0, max=3.0)

        # Create magnitude response with gain=1 -> (channels, freqs)
        mag_response = self.exponential_power(
            self.n_range, self.f_mean, self.bandwidth, self.shape * 8 - 14
        )
        mag_response = mag_response / mag_response.max(dim=-1, keepdim=True)[0]

        # Create phase response, scaled so that normalized group_delay=1
        # corresponds to group delay of 1000ms.
        f = torch.linspace(
            0,
            self.sample_rate,
            self.sequence_length // 2 + 1,
            device=mag_response.device,
            dtype=mag_response.dtype,
        )
        f = f.expand(mag_response.shape[0], -1)  # repeat for filter channels
        pha_response = -self.group_delay.unsqueeze(-1) * f * torch.pi

        # Create real and imaginary parts of the filters
        real = mag_response * torch.cos(pha_response)
        imag = mag_response * torch.sin(pha_response)

        # Stack real and imaginary parts to create filters
        # -> (channels, freqs, 2)
        filters = torch.stack((real, imag), dim=-1)

        return filters

    def forward(self, x):
        """x: (..., channels, time)"""
        # Construct filters from parameters
        self.filters = self.construct_filters()

        # Apply FFT -> (..., channels, freqs, 2)
        x = torch.fft.rfft(x, dim=-1)
        x = torch.view_as_real(x)  # separate real and imag

        # Repeat channels in case of multiple filters per channel
        x = torch.repeat_interleave(x, self.out_channels // self.in_channels, dim=-3)

        # Apply filters in the frequency domain
        x = x * self.filters

        # Apply inverse FFT if requested
        if self.inverse_fourier:
            x = torch.view_as_complex(x)
            x = torch.fft.irfft(x, n=self.sequence_length, dim=-1)

        x = x.to(dtype=torch.float32)

        return x


class MagEEGminer(EEGModuleMixin, nn.Module):
    def __init__(
        self,  # Signal related parameters
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
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

        # Initialize filter parameters
        self.fs = 128
        self.filter_f_mean = [23.0, 23.0]
        self.filter_bandwidth = [44.0, 44.0]
        self.filter_shape = [2.0, 2.0]
        self.n_filters = len(self.filter_f_mean)
        self.n_electrodes = self.n_chans
        self.time = self.n_times

        # Generalized Gaussian Filter
        self.filter = GeneralizedGaussianFilter(
            self.n_electrodes,
            self.n_electrodes * self.n_filters,
            self.time,
            sample_rate=self.fs,
            f_mean=self.filter_f_mean,
            bandwidth=self.filter_bandwidth,
            shape=self.filter_shape,
            affine_group_delay=False,
            inverse_fourier=True,
            group_delay=(20.0, 20.0),
            clamp_f_mean=(1.0, 45.0),
        )

        # Classifier
        self.n_features = self.n_electrodes * self.n_filters
        self.ft_bn = nn.BatchNorm1d(self.n_features, affine=False)
        self.fc_out = nn.Linear(self.n_features, self.n_outputs)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        """x: (batch, electrodes, time)"""
        batch = x.shape[0]

        # Apply Gaussian filters in frequency domain
        # x -> (batch, electrodes * filters, time)
        x = self.filter(x)

        # Signal magnitude
        x = x * x
        x = x.mean(dim=-1)
        x = torch.sqrt(x)

        # Classifier
        # Note that the order of dimensions before flattening the feature vector is important
        # for attributing feature weights during interpretation.
        x = x.reshape(batch, self.n_features)
        x = self.ft_bn(x)
        x = self.fc_out(x)
        x = torch.sigmoid(x)

        return x


class CorrEEGminer(EEGModuleMixin, nn.Module):
    def __init__(
        self,  # Signal related parameters
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
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

        self.n_out = self.n_outputs

        # Initialize filter parameters
        self.fs = 128
        self.filter_f_mean = [23.0, 23.0]
        self.filter_bandwidth = [44.0, 44.0]
        self.filter_shape = [2.0, 2.0]
        self.n_filters = len(self.filter_f_mean)
        self.n_electrodes = self.n_chans
        self.time = self.n_times

        # Generalized Gaussian Filter
        self.filter = GeneralizedGaussianFilter(
            self.n_electrodes,
            self.n_electrodes * self.n_filters,
            self.time,
            sample_rate=self.fs,
            f_mean=self.filter_f_mean,
            bandwidth=self.filter_bandwidth,
            shape=self.filter_shape,
            inverse_fourier=True,
            affine_group_delay=False,
            group_delay=(20.0, 20.0),
            clamp_f_mean=(1.0, 45.0),
        )

        # Classifier
        self.n_features = (
            self.n_filters * self.n_electrodes * (self.n_electrodes - 1) // 2
        )
        self.ft_bn = nn.BatchNorm1d(self.n_features, affine=False)
        self.fc_out = nn.Linear(self.n_features, self.n_outputs)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        """x: (batch, electrodes, time)"""
        batch = x.shape[0]

        # Apply Gaussian filters in frequency domain
        # x -> (batch, electrodes * filters, time)
        x = self.filter(x)

        # Compute signal correlations
        # x -> (batch, electrodes, electrodes, filters)
        x = x.reshape(batch, self.n_electrodes, self.n_filters, self.time).transpose(
            -3, -2
        )
        x = (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(
            x.var(dim=-1, keepdim=True) + 1e-6
        )
        x = torch.matmul(x, x.transpose(-2, -1)) / x.shape[-1]
        x = x.transpose(-3, -2).transpose(-2, -1)  # move filter channels to the end
        x = x.abs()

        # Get upper triu of symmetric connectivity matrix
        triu = torch.triu_indices(self.n_electrodes, self.n_electrodes, 1)
        x = x[:, triu[0], triu[1], :]

        # Classifier
        # Note that the order of dimensions before flattening the feature vector is important
        # for attributing feature weights during interpretation.
        x = x.reshape(batch, self.n_features)
        x = self.ft_bn(x)
        x = self.fc_out(x)
        x = torch.sigmoid(x)

        return x


class PLVEEGminer(EEGModuleMixin, nn.Module):
    def __init__(
        self,  # Signal related parameters
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
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

        # Initialize filter parameters
        self.fs = 128
        self.filter_f_mean = [23.0, 23.0]
        self.filter_bandwidth = [44.0, 44.0]
        self.filter_shape = [2.0, 2.0]
        self.n_filters = len(self.filter_f_mean)
        self.n_electrodes = self.n_chans
        self.time = self.n_times

        # Generalized Gaussian Filter
        self.filter = GeneralizedGaussianFilter(
            1,
            1 * self.n_filters,
            self.time,
            sample_rate=self.fs,
            f_mean=self.filter_f_mean,
            bandwidth=self.filter_bandwidth,
            shape=self.filter_shape,
            inverse_fourier=False,
            affine_group_delay=False,
            group_delay=(20.0, 20.0),
            clamp_f_mean=(1.0, 45.0),
        )

        # Classifier
        self.n_features = (
            self.n_filters * self.n_electrodes * (self.n_electrodes - 1) // 2
        )
        self.ft_bn = nn.BatchNorm1d(self.n_features, affine=False)
        self.fc_out = nn.Linear(self.n_features, self.n_outputs)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        """x: (batch, electrodes, time)"""
        _ = x.shape[0]

        # Apply Gaussian filters in frequency domain
        # x -> (batch, electrodes, filters, n_freq, 2)
        x = self.filter(x.unsqueeze(-2))

        # Compute PLV connectivity
        # x -> (batch, electrodes, electrodes, filters)
        x = x.transpose(-4, -3)  # swap electrodes and filters
        x = plv_time(x, forward_fourier=False)
        x = x.transpose(-3, -2).transpose(-2, -1)  # move filter channels to the end

        # Get upper triu of symmetric connectivity matrix
        triu = torch.triu_indices(self.n_electrodes, self.n_electrodes, 1)
        x = x[:, triu[0], triu[1], :]

        # Classifier
        # Note that the order of dimensions before flattening the feature vector is important
        # for attributing feature weights during interpretation.

        x = x.flatten(1)
        x = self.ft_bn(x)
        x = self.fc_out(x)
        x = torch.sigmoid(x)

        return x
