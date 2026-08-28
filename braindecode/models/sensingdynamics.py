# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Raul C. Sîmpetru, Alessandro Del Vecchio (original SensingDynamics)
#
# License: BSD (3-clause)
"""SensingDynamics adapted to the 16-channel emg2pose wristband."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from scipy.signal import butter
from torch import nn

from braindecode.models.base import EEGModuleMixin


class _SMU(nn.Module):
    """Smooth Maximum Unit with the paper's learnable ``mu`` parameter."""

    def __init__(self, alpha: float = 0.01, mu: float = 2.5) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.mu = nn.Parameter(torch.tensor(float(mu)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positive = (1.0 + self.alpha) * x
        smooth = (1.0 - self.alpha) * x * torch.erf(self.mu * (1.0 - self.alpha) * x)
        return 0.5 * (positive + smooth)


class SensingDynamics(EEGModuleMixin, nn.Module):
    r"""SensingDynamics from Sîmpetru et al. (2022) [simpetru2022]_.

    :bdg-success:`Convolution` :bdg-dark-line:`Channel`

    .. figure:: https://braindecode.org/dev/_static/model/sensingdynamics.png
       :align: center
       :alt: SensingDynamics architecture with circular electrode padding

       Figure 7 of Sîmpetru et al. (2022), CC BY-ND. Raw and low-pass
       filtered sEMG enter stacked, the electrode axis is circularly
       padded, and a 512/512 MLP reads out the joint angles.


    .. rubric:: Architecture Overview

    The emg2pose adaptation replaces the original five-grid 3D convolutions
    with 2D convolutions over the 16 wristband electrodes and time. Dividing
    the original spatial geometry by four preserves its relative receptive
    field. The network directly regresses joint angles.

    .. rubric:: Macro Components

    ``SensingDynamics.lowpass`` (spectral input copy)
        **Operations.** Periodic fourth-order, zero-phase 20 Hz Butterworth
        response.
        **Role.** Exposes the low-frequency neural drive alongside broadband
        monopolar EMG.

    ``SensingDynamics.conv1`` (action-potential detector)
        **Operations.** Conv2d ``(1, 31)``, stride ``(1, 8)``, BatchNorm, SMU,
        and spatial dropout.
        **Role.** Detects short motor-unit waveform patterns independently at
        each electrode.

    ``SensingDynamics.conv2`` / ``SensingDynamics.conv3`` (ring mixer)
        **Operations.** Circular padding by four electrodes; dilated Conv2d
        ``(8, 18)`` with dilation ``(2, 1)``; Conv2d ``(3, 1)``; BatchNorm and
        SMU after both convolutions.
        **Role.** Mixes synchronous activity around the wristband with a
        17-electrode spatial and 167-sample temporal receptive field.

    ``SensingDynamics.mlp`` / ``SensingDynamics.final_layer``
        **Operations.** Flatten the 64 by 8 spatial features with an einops
        layer, then apply a ``512, 512, n_outputs`` MLP.
        **Role.** Regresses joint angles at the feature rate before linear
        interpolation to the input grid.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Time:** valid temporal kernels cover 167 input samples and stride by 8.
    - **Channels/space:** circular padding respects the wristband topology.
    - **Frequency:** broadband EMG is depth-stacked with a 20 Hz low-pass copy.

    .. rubric:: Additional Mechanisms

    SMU keeps its published fixed ``alpha=0.01`` and independently learnable
    ``mu=2.5`` initialization. The source paper's 150 ms output moving average
    is postprocessing and is not part of this neural-network module. Pass an
    externally filtered tensor as ``x_lowpass`` to reproduce a particular
    historical filter's boundary handling exactly. The emg2pose benchmark
    reports a 10,167-sample training window for this baseline; it does not
    publish how the adapted network assigns its 167-sample receptive field
    between target alignment and window context.

    .. note::
       The emg2pose paper specifies that kernels, strides, and dilations were
       changed to preserve the original receptive fields, but its public code
       and checkpoint omit SensingDynamics. The 16-channel spatial dimensions
       in this implementation are consequently derived from that published
       rule; exact checkpoint equivalence cannot be asserted without the
       unreleased implementation.

    .. versionadded:: 1.8

    Parameters
    ----------
    temporal_channels : int, optional
        Output width of the action-potential detector. Default is ``256``.
    mid_channels : int, optional
        Output width of the dilated electrode mixer. Default is ``32``.
    spatial_channels : int, optional
        Output width of the final convolution. Default is ``64``.
    mlp_hidden : int, optional
        Width of both hidden MLP layers. Default is ``512``.
    conv_drop_prob : float, optional
        Spatial dropout after the first convolution. Default is ``0.25``.
    mlp_drop_prob : float, optional
        Dropout before the MLP. Default is ``0.4``.
    lowpass_hz : float, optional
        Cutoff of the Butterworth input copy. Default is ``20``.
    lowpass_order : int, optional
        Order of that Butterworth filter. Default is ``4``.
    temporal_kernel : tuple of int, optional
        ``(electrodes, time)`` kernel of the action-potential detector.
        Default is ``(1, 31)``.
    temporal_stride : tuple of int, optional
        Stride of that convolution. Default is ``(1, 8)``.
    mid_kernel : tuple of int, optional
        ``(electrodes, time)`` kernel of the electrode mixer. Default is
        ``(8, 18)``.
    mid_dilation : tuple of int, optional
        Dilation of the mixer. Default is ``(2, 1)``.
    spatial_kernel : tuple of int, optional
        ``(electrodes, time)`` kernel of the final convolution. Default is
        ``(3, 1)``.
    electrode_wrap : int, optional
        Circular padding applied to each end of the electrode axis, which
        is a ring on the wristband. Default is ``4``.
    activation : type[nn.Module], optional
        Activation class. The default is the paper's learnable SMU.

    References
    ----------
    .. [simpetru2022] Sîmpetru, Arkudas, Braun, Osswald, Oliveira,
       Eskofier, Kinfe, Del Vecchio (2022). Sensing the Full Dynamics
       of the Human Hand with a Neural Interface and Deep Learning.
       bioRxiv 2022.07.29.502064. doi:10.1101/2022.07.29.502064
    """

    #: Temporal receptive field of the derived emg2pose adaptation.
    #: Temporal receptive field of the paper's default configuration. The
    #: value actually enforced is derived per instance in ``__init__``, so a
    #: reconfigured stack validates against its own receptive field.
    receptive_field_samples = 167
    #: Best training-window length reported by the emg2pose benchmark.
    benchmark_window_samples = 10_167

    def __init__(
        self,
        # braindecode parameters
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        # model-specific parameters
        temporal_channels: int = 256,
        mid_channels: int = 32,
        spatial_channels: int = 64,
        mlp_hidden: int = 512,
        conv_drop_prob: float = 0.25,
        mlp_drop_prob: float = 0.4,
        lowpass_hz: float = 20.0,
        lowpass_order: int = 4,
        temporal_kernel: tuple[int, int] = (1, 31),
        temporal_stride: tuple[int, int] = (1, 8),
        mid_kernel: tuple[int, int] = (8, 18),
        mid_dilation: tuple[int, int] = (2, 1),
        spatial_kernel: tuple[int, int] = (3, 1),
        electrode_wrap: int = 4,
        activation: type[nn.Module] = _SMU,
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        if self.n_chans != 16:
            raise ValueError(
                "SensingDynamics implements the 16-channel emg2pose "
                f"adaptation; got n_chans={self.n_chans}."
            )
        # Standard receptive-field accumulation over the temporal axis:
        # each layer contributes (kernel - 1) * dilation, scaled by the
        # strides before it. At the defaults this is 1 + 30 + 136 = 167.
        receptive_field = 1
        stride_so_far = 1
        for kernel, dilation, stride in (
            (temporal_kernel[1], 1, temporal_stride[1]),
            (mid_kernel[1], mid_dilation[1], 1),
            (spatial_kernel[1], 1, 1),
        ):
            receptive_field += (kernel - 1) * dilation * stride_so_far
            stride_so_far *= stride
        self.receptive_field_samples = receptive_field
        if self.n_times < self.receptive_field_samples:
            raise ValueError(
                f"SensingDynamics requires at least "
                f"{self.receptive_field_samples} input samples; got {self.n_times}."
            )
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive; got {self.sfreq}.")
        if not 0 < lowpass_hz < self.sfreq / 2:
            raise ValueError("lowpass_hz must lie between 0 and the Nyquist frequency")

        self.lowpass = _ButterworthLowpass(
            sfreq=self.sfreq, cutoff_hz=lowpass_hz, order=lowpass_order
        )
        self.to_feature_plane = Rearrange(
            "batch nchans ntimes -> batch 1 nchans ntimes"
        )
        # (left, right, top, bottom): wrap the electrode axis only.
        self.circular_pad = nn.CircularPad2d(
            padding=(0, 0, electrode_wrap, electrode_wrap)
        )
        self.conv1 = _ConvBlock(
            in_channels=2,
            out_channels=temporal_channels,
            kernel_size=temporal_kernel,
            stride=temporal_stride,
            activation=activation,
        )
        self.conv_dropout = nn.Dropout2d(p=conv_drop_prob)
        self.conv2 = _ConvBlock(
            in_channels=temporal_channels,
            out_channels=mid_channels,
            kernel_size=mid_kernel,
            dilation=mid_dilation,
            activation=activation,
        )
        self.conv3 = _ConvBlock(
            in_channels=mid_channels,
            out_channels=spatial_channels,
            kernel_size=spatial_kernel,
            activation=activation,
        )

        # Width of the electrode axis where conv3 leaves it. Previously a
        # literal 8, which silently assumed the default kernels *and* the
        # 16-electrode montage; deriving it lets both vary.
        electrode_features = self.n_chans
        electrode_features -= temporal_kernel[0] - 1
        electrode_features += 2 * electrode_wrap
        electrode_features -= mid_dilation[0] * (mid_kernel[0] - 1)
        electrode_features -= spatial_kernel[0] - 1
        if electrode_features < 1:
            raise ValueError(
                f"The configured kernels consume the electrode axis: "
                f"n_chans={self.n_chans} leaves {electrode_features} positions "
                "after conv3. Widen electrode_wrap or shrink the kernels."
            )
        self.to_sequence = Rearrange(
            "batch features nelectrodes ntimes -> batch ntimes (features nelectrodes)"
        )
        self.sequence_to_channels = Rearrange(
            "batch ntimes njoints -> batch njoints ntimes"
        )
        self.channels_to_sequence = Rearrange(
            "batch njoints ntimes -> batch ntimes njoints"
        )
        self.mlp = nn.Sequential(
            nn.Dropout(p=mlp_drop_prob),
            nn.Linear(
                in_features=spatial_channels * electrode_features,
                out_features=mlp_hidden,
            ),
            activation(),
            nn.Linear(in_features=mlp_hidden, out_features=mlp_hidden),
            activation(),
        )
        self.final_layer = nn.Linear(
            in_features=mlp_hidden, out_features=self.n_outputs
        )

    def reset_head(self, n_outputs: int) -> None:
        """Swap the output layer for a new output dimensionality."""
        self._set_n_outputs(n_outputs)
        old = self.final_layer
        self.final_layer = nn.Linear(
            in_features=old.in_features, out_features=n_outputs
        ).to(device=old.weight.device, dtype=old.weight.dtype)

    def forward(
        self, x: torch.Tensor, x_lowpass: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict angles, optionally using an externally filtered input copy."""
        n_times = x.shape[-1]
        if x_lowpass is None:
            x_lowpass = self.lowpass(x)
        elif x_lowpass.shape != x.shape:
            raise ValueError(
                "x_lowpass must have the same shape as x; "
                f"got {x_lowpass.shape} and {x.shape}."
            )

        raw = self.to_feature_plane(x)
        lowpassed = self.to_feature_plane(x_lowpass)
        features = self.conv_dropout(self.conv1(torch.cat((raw, lowpassed), dim=1)))
        features = self.conv2(self.circular_pad(features))
        features = self.conv3(features)
        prediction = self.final_layer(self.mlp(self.to_sequence(features)))
        return self.channels_to_sequence(
            F.interpolate(
                self.sequence_to_channels(prediction),
                size=n_times,
                mode="linear",
                align_corners=False,
            )
        )


class _ButterworthLowpass(nn.Module):
    """Periodic fourth-order zero-phase Butterworth low-pass layer.

    This applies the exact digital filter magnitude response, but its FFT
    implementation treats each input window as periodic. Callers requiring
    the paper's forward-backward boundary convention should filter the
    continuous recording first and pass that window as ``x_lowpass``.
    """

    def __init__(self, sfreq: float, cutoff_hz: float, order: int) -> None:
        super().__init__()
        b_coeffs, a_coeffs = butter(order, cutoff_hz, btype="low", fs=sfreq)
        self.register_buffer("a_coeffs", torch.as_tensor(a_coeffs, dtype=torch.float64))
        self.register_buffer("b_coeffs", torch.as_tensor(b_coeffs, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_freqs = x.shape[-1] // 2 + 1
        omega = (
            2
            * torch.pi
            * torch.arange(n_freqs, device=x.device, dtype=self.a_coeffs.dtype)
            / x.shape[-1]
        )
        unit_delay = torch.exp(-1j * omega)
        numerator = torch.zeros_like(unit_delay)
        denominator = torch.zeros_like(unit_delay)
        for delay, (b_coeff, a_coeff) in enumerate(zip(self.b_coeffs, self.a_coeffs)):
            numerator = numerator + b_coeff * unit_delay**delay
            denominator = denominator + a_coeff * unit_delay**delay
        zero_phase_response = (numerator / denominator).abs().square()
        spectrum = torch.fft.rfft(x, dim=-1)
        return torch.fft.irfft(
            spectrum * zero_phase_response.to(dtype=spectrum.dtype),
            n=x.shape[-1],
            dim=-1,
        )


class _ConvBlock(nn.Module):
    """SensingDynamics Conv-BN-SMU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        *,
        stride: tuple[int, int] = (1, 1),
        dilation: tuple[int, int] = (1, 1),
        activation: type[nn.Module] = _SMU,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.batch_norm(self.conv(x)))
