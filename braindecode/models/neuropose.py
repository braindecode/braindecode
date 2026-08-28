# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Yilin Liu, Shijia Zhang, Mahanth Gowda (original NeuroPose)
#
# License: BSD (3-clause)
# Reimplementation for research/benchmarking; original work is
# Liu et al., WWW 2021 / IEEE IoT-J 2022.
"""``NeuroPoseNet``: CNN encoder-decoder with ResNet bottleneck."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import _disable_batch_norm_training_if_batch_size_one


class NeuroPoseNet(EEGModuleMixin, nn.Module):
    r"""NeuroPoseNet from Liu et al (2021) [liu2021neuropose]_.

    :bdg-success:`Convolution`

    Convolutional encoder-decoder with a residual bottleneck mapping
    wearable sEMG windows to continuous finger joint angles. The
    original consumes 8-channel Myo-band EMG at 200 Hz (5 s windows)
    and emits 16 regressed dimensions plus 5 anatomically derived
    joints; this port generalizes it to arbitrary channel counts and
    sampling rates so it runs natively on emg2pose data.

    .. rubric:: Architecture Overview

    ``(B, n_chans, T)`` → channel adapter → exact resampling to
    ``internal_sfreq`` (200 Hz) → three Conv2d-BN-ReLU-Dropout-MaxPool
    stages downsampling (time × bands) by (5, 2), (4, 2), (2, 2) →
    linear projection to ``encoder_dim`` → ``n_res_blocks`` residual
    blocks over time → mirrored Conv-BN-ReLU-Upsample(nearest) decoder
    → linear head → upsample to ``T`` → ``(B, T, n_outputs)``.

    .. rubric:: Macro Components

    ``NeuroPoseNet.adapter`` (channel/time front-end)
        **Operations.** ``channel_adapter="tile"`` repeats/truncates the
        electrode axis to ``n_bands``; ``"learned"`` uses a linear map.
        Time is then resampled to ``internal_sfreq`` using average pooling
        for integral downsampling ratios and linear interpolation otherwise.
        **Role.** Bridges consumer-band layouts (8 ch @ 200 Hz) and
        research layouts (16 ch @ 2 kHz) into one tensor geometry.

    ``NeuroPoseNet.encoder``
        **Operations.** Three Conv-BN-ReLU-Dropout-MaxPool blocks
        (filters ``base_channels``, ``×2``, ``×4``), kernel 3×2,
        downsampling factors (5, 2), (4, 2), (2, 2).
        **Role.** Compact spatio-temporal representation of the 5 s
        window, mirroring the paper's schedule.

    ``NeuroPoseNet.resnet`` (residual bottleneck)
        **Operations.** ``n_res_blocks`` basic blocks computing
        ``act(x + f(x))`` with ``n_convs_per_block`` 1-D convolutions each.
        **Role.** The paper's key accuracy lever: deeper feature
        extraction without convergence loss ({3, 5, 7} swept upstream).

    ``NeuroPoseNet.decoder`` / ``NeuroPoseNet.final_layer``
        **Operations.** Mirrored Conv-BN-ReLU-Upsample(nearest) stages
        restore 1,000 temporal positions and expand the spatial axis to
        16 bands; ``final_layer`` maps flattened band channels to
        ``n_outputs`` per frame before interpolation to the input rate.
        **Role.** Dense per-frame joint-angle regression.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Time:** The pooling/upsampling pyramid captures temporal structure.
    - **Channels/space:** Small 2-D kernels mix neighboring electrode bands.
    - **Frequency:** Spectral content is learned from the resampled waveform.

    .. rubric:: Additional Mechanisms

    The paper's anatomical constraints (bounded activations over joint
    ranges), group-wise MSE losses and the temporal-smoothness penalty
    are training-side concerns outside this module; its BN-freezing
    transfer-learning recipe applies unchanged (freeze all but BatchNorm
    layers, fine-tune on ~90 s of target-user data).

    .. rubric:: Relationship to emg2pose's NeuroPose baseline

    The defaults reproduce Liu et al.'s original architecture, **not** the
    adaptation emg2pose used to produce its published numbers, and they will
    not reproduce that paper's angular error. The two differ in capacity and
    in front-end strategy:

    ==========================  ====================  ====================
    Setting                     Default here          emg2pose
    ==========================  ====================  ====================
    Encoder widths              32 / 64 / 128         32 / 128 / 256
    Residual blocks             3                     5
    Convs per residual block    2                     3
    Front end                   decimate to 200 Hz    keep 2 kHz, widen
                                                      pooling 8× / 2×
    Parameters                  ~1.44 M               ~6.36 M
    ==========================  ====================  ====================

    ``encoder_channels`` and ``n_convs_per_block`` exist so that capacity
    can be raised toward the reference without editing the class::

        NeuroPoseNet(
            n_chans=16, n_outputs=20, n_times=10_000, sfreq=2_000.0,
            n_bands=16, encoder_channels=(32, 128, 256),
            encoder_dim=320, n_res_blocks=5, n_convs_per_block=3,
        )  # ~5.25 M parameters, 0.83x the released checkpoint

    This narrows the capacity gap but is **not** an equivalence: as the
    ``mapping`` table below records, the decoder geometry, padding and
    resampling still differ, so no argument combination reproduces the
    released checkpoint exactly. The pooling schedule is fixed, so
    emg2pose's widened front end cannot be expressed at all; this class
    decimates to ``internal_sfreq`` instead and therefore discards EMG
    content above 100 Hz.

    .. versionadded:: 1.8

    Parameters
    ----------
    internal_sfreq : float, optional
        Sampling rate the conv stack operates at. The default is
        ``200.0`` (paper value); inputs are resampled to this rate.
    n_bands : int, optional
        Band dimension seen by the 2-D conv stack. The default is
        ``8`` (Myo layout).
    channel_adapter : {'tile', 'learned'}, optional
        How to reach ``n_bands`` from ``n_chans``. The default is
        ``'tile'`` (parameter-free repeat/truncate).
    base_channels : int, optional
        First-stage filters (doubled per stage). The default is ``32``.
        Ignored when ``encoder_channels`` is given.
    encoder_channels : tuple of int, optional
        Explicit widths of the three encoder stages. The default is
        ``None``, deriving ``(base_channels, ×2, ×4)``. Provide this to
        reach width schedules the doubling rule cannot express, such as
        emg2pose's ``(32, 128, 256)``.
    encoder_dim : int, optional
        Bottleneck width after the projection. The default is ``256``.
    n_res_blocks : int, optional
        Residual blocks between encoder and decoder. The default is
        ``3``.
    n_convs_per_block : int, optional
        Convolutions inside each residual block. The default is ``2``;
        emg2pose's NeuroPose configuration uses ``3``.
    activation : type[nn.Module], optional
        Activation class used throughout. The default is ``nn.ReLU``.
    drop_prob : float, optional
        Dropout probability inside encoder blocks. The default is
        ``0.05`` (paper value).

    References
    ----------
    .. [liu2021neuropose] Liu, Zhang, Gowda (2021). NeuroPose: 3D Hand
       Pose Tracking using EMG Wearables. The Web Conference 2021 /
       IEEE Internet of Things Journal 46(1), 2022.
       doi:10.1145/3442381.3449890
    """

    # Operation-equivalent subset of Meta's regression_neuropose.ckpt. Later
    # blocks differ in width, padding, resampling, and decoder geometry.
    mapping = {
        "model.network.network.0.network.0.weight": "encoder.0.conv.weight",
        "model.network.network.0.network.0.bias": "encoder.0.conv.bias",
        "model.network.network.0.network.1.weight": "encoder.0.bn.weight",
        "model.network.network.0.network.1.bias": "encoder.0.bn.bias",
        "model.network.network.0.network.1.running_mean": ("encoder.0.bn.running_mean"),
        "model.network.network.0.network.1.running_var": "encoder.0.bn.running_var",
        "model.network.network.0.network.1.num_batches_tracked": (
            "encoder.0.bn.num_batches_tracked"
        ),
    }

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
        internal_sfreq: float = 200.0,
        n_bands: int = 8,
        channel_adapter: str = "tile",
        base_channels: int = 32,
        encoder_channels: tuple[int, int, int] | None = None,
        encoder_dim: int = 256,
        n_res_blocks: int = 3,
        n_convs_per_block: int = 2,
        activation: type[nn.Module] = nn.ReLU,
        drop_prob: float = 0.05,
    ):
        if channel_adapter not in ("tile", "learned"):
            raise ValueError("channel_adapter must be 'tile' or 'learned'")
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.internal_sfreq = float(internal_sfreq)
        if self.internal_sfreq <= 0:
            raise ValueError(f"internal_sfreq must be positive; got {internal_sfreq}.")
        self.n_bands = int(n_bands)
        if self.n_bands < 7:
            raise ValueError(
                "n_bands must be at least 7 for the encoder's spatial pooling; "
                f"got {n_bands}."
            )
        self.channel_adapter_mode = channel_adapter
        self.input_sfreq = float(self.sfreq)
        sampling_ratio = self.input_sfreq / self.internal_sfreq
        rounded_ratio = round(sampling_ratio)
        self.decim = (
            int(rounded_ratio)
            if rounded_ratio >= 1 and math.isclose(sampling_ratio, rounded_ratio)
            else None
        )

        if channel_adapter == "learned":
            self.adapter = nn.Sequential(
                Rearrange("b c t -> b t c"),
                nn.Linear(self.n_chans, self.n_bands),
                Rearrange("b t c -> b c t"),
            )
        else:
            self.adapter = _TileChannels(self.n_chans, self.n_bands)

        if encoder_channels is None:
            c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        else:
            if len(encoder_channels) != 3:
                raise ValueError(
                    "encoder_channels must give exactly three widths; got "
                    f"{len(encoder_channels)}."
                )
            c1, c2, c3 = (int(width) for width in encoder_channels)
            if min(c1, c2, c3) < 1:
                raise ValueError(
                    f"encoder_channels must be positive; got {encoder_channels}."
                )
        self.encoder = nn.Sequential(
            _ConvBlock(1, c1, 3, 2, 5, 2, activation, drop_prob),
            _ConvBlock(c1, c2, 3, 2, 4, 2, activation, drop_prob),
            _ConvBlock(c2, c3, 3, 1, 2, 1, activation, drop_prob),
        )
        self.input_to_encoder = Rearrange("b c t -> b 1 t c")
        self.encoder_to_sequence = Reduce("b c t s -> b t c", "mean")

        self.proj = nn.Linear(c3, encoder_dim)
        self.resnet = nn.Sequential(
            *[
                _ResBlock(encoder_dim, activation, n_convs_per_block)
                for _ in range(n_res_blocks)
            ]
        )
        self.sequence_to_resnet = Rearrange("b t c -> b c t")
        self.resnet_to_sequence = Rearrange("b c t -> b t c")
        self.sequence_to_decoder = Rearrange("b t c -> b c t 1")
        self.decoder = nn.Sequential(
            _UpBlock(encoder_dim, c2, 2, 4, activation),
            _UpBlock(c2, c1, 4, 2, activation),
            _UpBlock(c1, c1, 5, 2, activation),
        )
        self.decoder_to_sequence = Rearrange("b c t s -> b t (s c)")
        self.sequence_to_channels = Rearrange("b t j -> b j t")
        self.channels_to_sequence = Rearrange("b j t -> b t j")
        self.head_ff = nn.Dropout(drop_prob)
        self.final_layer = nn.Linear(c1 * 16, self.n_outputs)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def reset_head(self, n_outputs: int) -> None:
        """Swap the output layer for a new output dimensionality."""
        if n_outputs <= 0:
            raise ValueError(f"n_outputs must be positive; got {n_outputs}.")
        old = self.final_layer
        self.final_layer = nn.Linear(old.in_features, n_outputs).to(
            device=old.weight.device, dtype=old.weight.dtype
        )
        self.final_layer.apply(self._init_weights)
        self._n_outputs = n_outputs
        init_kwargs = getattr(self, "_braindecode_init_kwargs", None)
        if init_kwargs is not None and "n_outputs" in init_kwargs:
            init_kwargs["n_outputs"] = n_outputs
        hub_config = getattr(self, "_hub_mixin_config", None)
        if hub_config is not None and "n_outputs" in hub_config:
            hub_config["n_outputs"] = n_outputs

    @_disable_batch_norm_training_if_batch_size_one
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[-1]
        h = self.adapter(x)
        # Resample to the paper's internal temporal grid.
        if self.decim is not None:
            t_int = (h.shape[-1] // self.decim) * self.decim
            min_input_times = 40 * self.decim
            if h.shape[-1] < min_input_times:
                raise ValueError(
                    f"Input must contain at least {min_input_times} time samples "
                    "so the resampled signal has the 40 samples required by the "
                    f"encoder; got {h.shape[-1]}."
                )
            h = h[..., :t_int]
            if self.decim > 1:
                h = torch.nn.functional.avg_pool1d(h, self.decim)
        else:
            t_int = round(h.shape[-1] * self.internal_sfreq / self.input_sfreq)
            if t_int < 40:
                raise ValueError(
                    f"Input resamples to {t_int} internal samples, but the encoder "
                    "requires at least 40."
                )
            h = torch.nn.functional.interpolate(
                h, size=t_int, mode="linear", align_corners=False
            )

        # Layout contract of _ConvBlock/_UpBlock: (B, C, TIME, BANDS).
        z = self.encoder(self.input_to_encoder(h))
        z = self.encoder_to_sequence(z)
        z = self.proj(z)
        z = self.resnet_to_sequence(self.resnet(self.sequence_to_resnet(z)))

        z = self.sequence_to_decoder(z)
        z = self.decoder(z)  # (B, C1, T''', B')
        z = self.decoder_to_sequence(z)

        out = self.final_layer(self.head_ff(z))  # (B, T_internal, J)
        return self.channels_to_sequence(
            torch.nn.functional.interpolate(
                self.sequence_to_channels(out),
                size=t,
                mode="linear",
                align_corners=False,
            )
        )


class _TileChannels(nn.Module):
    def __init__(self, n_chans: int, n_bands: int):
        super().__init__()
        self.register_buffer(
            "channel_indices", torch.arange(n_bands) % n_chans, persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(1, self.channel_indices)


class _ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k_t, k_b, ds_t, ds_b, act, drop_prob):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, (k_t, k_b), padding=(k_t // 2, 0))
        self.bn = nn.BatchNorm2d(out_c)
        self.act = act()
        self.drop = nn.Dropout(drop_prob)
        self.pool = nn.MaxPool2d((ds_t, ds_b))

    def forward(self, x):
        return self.pool(self.drop(self.act(self.bn(self.conv(x)))))


class _UpBlock(nn.Module):
    def __init__(self, in_c, out_c, up_t, up_b, act):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = act()
        self.up_t, self.up_b = up_t, up_b

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return F.interpolate(
            x,
            scale_factor=[float(self.up_t), float(self.up_b)],
            mode="nearest",
        )


class _ResBlock(nn.Module):
    def __init__(self, c, act, n_convs: int = 2):
        super().__init__()
        if n_convs < 1:
            raise ValueError(f"n_convs must be >= 1; got {n_convs}.")
        layers: list[nn.Module] = []
        for index in range(n_convs):
            layers += [nn.Conv1d(c, c, 3, padding=1), nn.BatchNorm1d(c)]
            if index < n_convs - 1:
                layers.append(act())
        self.f = nn.Sequential(*layers)
        self.out_act = act()

    def forward(self, x):
        return self.out_act(x + self.f(x))
