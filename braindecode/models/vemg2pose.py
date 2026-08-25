# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Meta Platforms, Inc. and affiliates (original emg2pose architecture)
#
# License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# Architecture follows Salter et al. 2024 (CC BY-NC-SA 4.0) as revised by
# Hadidi et al. 2026 (arXiv 2603.08212, CC BY 4.0).
"""``VEMG2PoseNet``: state-conditioned sEMG-to-pose decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin


class VEMG2PoseNet(EEGModuleMixin, nn.Module):
    r"""VEMG2PoseNet from Salter et al (2024) [salter2024]_, as revised by
    Hadidi et al (2026) [hadidi2026]_.

    :bdg-success:`Convolution` :bdg-secondary:`Recurrent`

    Causal time-depth-separable (TDS) convolutional encoder producing
    64-D features at 25 Hz, decoded autoregressively at 50 Hz by a
    state-conditioned LSTM whose input concatenates the encoder feature
    with the previous pose estimate. Supports both *position* decoding
    (absolute joint angles) and *velocity* decoding (integrated
    increments) under one architecture.

    .. rubric:: Architecture Overview

    Raw sEMG ``x ∈ (B, n_chans, T)`` → strided causal conv stem
    (kernel/stride 11/5 → 5/2, 256 ch) → TDS stage (subsampling conv
    kernel/stride 17/4) → linear squeeze to ``feature_dim`` → TDS stage
    (subsampling conv kernel/stride 9/2) → features ``(B, K, 64)`` at
    ~25 Hz → causal nearest-neighbor upsampling to ``decoder_rate``
    (50 Hz) → LSTM rollout with pose-state feedback → causal
    nearest-neighbor upsampling back to ``T``.

    .. rubric:: Macro Components

    ``VEMG2PoseNet.stem`` (strided temporal front-end)
        **Operations.** Two causal conv blocks (LayerNorm + LeakyReLU),
        mapping ``(B, n_chans, T) → (B, encoder_channels, T/10)``
        (kernel 11 stride 5, then kernel 5 stride 2; valid convolutions
        preserve causality).
        **Role.** Raw-waveform feature extraction at ~200 Hz.

    ``VEMG2PoseNet.tds_stages`` (Time-Depth-Separable bottleneck)
        **Operations.** Subsampling conv (kernel 17, stride 4) followed
        by ``tds_blocks`` depthwise-conv + pointwise + feedforward
        residual pairs; a 1×1 projection to ``feature_dim``; a second
        stage with subsampling kernel 9 stride 2.
        **Role.** Hannun-et-al-style factorized sequence modeling down
        to the 25 Hz decoder grid ([hannun2019]_).

    ``VEMG2PoseNet.lstm`` / ``VEMG2PoseNet.final_layer``
        **Operations.** At rollout step t: input
        ``z_t = [f_t ; ŷ_{t−1}] ∈ R^{feature_dim + n_outputs}``;
        2-layer LSTM (hidden ``hidden_size``); head = Linear →
        LeakyReLU → ``final_layer`` (Linear), scaled by the fixed
        scalar ``output_scalar``.
        **Role.** State-conditioned decoding: the model always sees the
        pose it last emitted, enabling both task formulations.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Time:** The strided stem and TDS stages capture causal temporal context.
    - **Channels/space:** Electrode order is fixed; there is no explicit spatial
      geometry.
    - **Frequency:** Spectral content is learned from the raw waveform without a
      hand-crafted filterbank.

    .. rubric:: Additional Mechanisms

    - **Output parameterization**: ``parameterization="position"``
      emits absolute joint angles; ``"velocity"`` emits increments that
      are integrated during the rollout.
    - **Output scalar**: Hadidi et al. show position decoding collapses
      into low-movement minima at Salter's default 0.01; Tracking needs
      ≈0.1, Regression ≈1.0. This is THE stability knob of the recipe.
    - **Learned regression anchor** ``p_init`` replaces the original
      zero-vector initialization (zero is a valid flat-hand pose).
      Pass ``y0=(B, n_outputs)`` to :meth:`forward` for Tracking.

    .. versionadded:: 1.8

    Parameters
    ----------
    hidden_size : int, optional
        LSTM hidden width. The default is ``512``.
    encoder_channels : int, optional
        Width of the strided stem convolutions. The default is ``256``.
    feature_dim : int, optional
        Encoder output width fed to the decoder. The default is ``64``.
    decoder_rate : float, optional
        Autoregressive rollout rate in Hz. The default is ``50.0``.
    tds_blocks : int, optional
        Residual conv/FF block pairs per TDS stage. The default is ``2``.
    parameterization : {'position', 'velocity'}, optional
        Decoder output interpretation. The default is ``'position'``.
    output_scalar : float, optional
        Fixed multiplier on the decoder head output. The default is
        ``0.1`` (position+tracking regime; use 1.0 for regression).
    activation : type[nn.Module], optional
        Activation class used inside the stem/TDS blocks and head. The
        default is ``nn.LeakyReLU``.
    drop_prob : float, optional
        Dropout inside the decoder head. The default is
        ``0.0`` (the paper uses none).

    References
    ----------
    .. [salter2024] Salter, Warren, Schlager, Spurr, Han, Bhasin, Rohin,
       et al. (2024). emg2pose: A Large and Diverse Benchmark for Surface
       Electromyographic Hand Pose Estimation. NeurIPS Datasets and
       Benchmarks. arXiv:2412.02725.
    .. [hadidi2026] Hadidi, Lee, Feghhi, Yuan, Kao (2026). Re-evaluating
       Position and Velocity Decoding for Hand Pose Estimation with
       Surface Electromyography. arXiv:2603.08212.
    .. [hannun2019] Hannun et al. (2019). Sequence Modeling with
       Time-Depth Separable Convolutions. arXiv:1904.01619.
    """

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
        hidden_size: int = 512,
        encoder_channels: int = 256,
        feature_dim: int = 64,
        decoder_rate: float = 50.0,
        tds_blocks: int = 2,
        parameterization: str = "position",
        output_scalar: float = 0.1,
        activation: type[nn.Module] = nn.LeakyReLU,
        drop_prob: float = 0.0,
    ):
        if parameterization not in ("position", "velocity"):
            raise ValueError("parameterization must be 'position' or 'velocity'")
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.parameterization = parameterization
        self.output_scalar = float(output_scalar)
        self.decoder_rate = float(decoder_rate)
        self.activation_cls = activation

        # braindecode parameters read post-inference:
        n_out = self.n_outputs
        chs = self.n_chans

        self.stem = nn.Sequential(
            nn.Conv1d(chs, encoder_channels, 11, stride=5),
            _TimeStepNorm(encoder_channels),
            activation(),
            nn.Conv1d(encoder_channels, encoder_channels, 5, stride=2),
            _TimeStepNorm(encoder_channels),
            activation(),
        )
        self.tds_stages = nn.Sequential(
            _TDSStage(
                encoder_channels,
                subsample_kernel=17,
                subsample_stride=4,
                n_blocks=tds_blocks,
                activation=activation,
            ),
            nn.Conv1d(encoder_channels, feature_dim, 1),
            _TDSStage(
                feature_dim,
                subsample_kernel=9,
                subsample_stride=2,
                n_blocks=tds_blocks,
                activation=activation,
            ),
        )

        self.enc_hz = float(self.sfreq) / (5 * 2 * 4 * 2)  # paper strides → 25 Hz
        self.lstm = nn.LSTM(
            feature_dim + n_out, hidden_size, num_layers=2, batch_first=True
        )
        self.head_ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            activation(),
            nn.Dropout(drop_prob),
        )
        self.encoder_to_sequence = Rearrange("b f t -> b t f")
        self.sequence_to_channels = Rearrange("b t f -> b f t")
        self.channels_to_sequence = Rearrange("b f t -> b t f")
        self.trajectory_to_channels = Rearrange("b t j -> b j t")
        self.channels_to_trajectory = Rearrange("b j t -> b t j")
        # final_layer LAST so it lands in the last two named_children().
        self.final_layer = nn.Linear(hidden_size // 2, n_out)

        self.p_init = nn.Parameter(torch.zeros(n_out))
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def reset_head(self, n_outputs: int) -> None:
        """Swap the output layer for a new output dimensionality.

        Because the decoded pose feeds back as decoder input
        (``[f_t ; ŷ_{t-1}]``), changing ``n_outputs`` also requires
        rebuilding the recurrent stack — its weights are re-initialized.
        """
        if n_outputs <= 0:
            raise ValueError(f"n_outputs must be positive; got {n_outputs}.")
        old = self.final_layer
        device, dtype = old.weight.device, old.weight.dtype
        self.final_layer = nn.Linear(old.in_features, n_outputs).to(
            device=device, dtype=dtype
        )
        self.final_layer.apply(self._init_weights)
        self.p_init = nn.Parameter(torch.zeros(n_outputs, device=device, dtype=dtype))
        feat_dim = self.tds_stages[-1].sub_conv.out_channels
        self.lstm = nn.LSTM(
            feat_dim + n_outputs,
            self.lstm.hidden_size,
            num_layers=2,
            batch_first=True,
        ).to(device=device, dtype=dtype)
        self._n_outputs = n_outputs
        init_kwargs = getattr(self, "_braindecode_init_kwargs", None)
        if init_kwargs is not None and "n_outputs" in init_kwargs:
            init_kwargs["n_outputs"] = n_outputs
        hub_config = getattr(self, "_hub_mixin_config", None)
        if hub_config is not None and "n_outputs" in hub_config:
            hub_config["n_outputs"] = n_outputs

    def forward(self, x: torch.Tensor, y0: torch.Tensor | None = None) -> torch.Tensor:
        """Decode hand pose from raw EMG.

        Parameters
        ----------
        x : Tensor (B, n_chans, T)
            Raw sEMG window at ``sfreq``.
        y0 : Tensor (B, n_outputs) | None
            Ground-truth initial pose (Tracking mode). Falls back to the
            learned ``p_init`` (Regression mode) when omitted.
        """
        n_t = x.shape[-1]
        feats = self.encoder_to_sequence(self.tds_stages(self.stem(x)))
        y_prev = y0 if y0 is not None else self.p_init.expand(x.shape[0], -1)

        k_dec = max(1, int(round(feats.shape[1] * self.decoder_rate / self.enc_hz)))
        steps = self.channels_to_sequence(
            F.interpolate(
                self.sequence_to_channels(feats),
                size=k_dec,
                mode="nearest",
            )
        )

        outs = []
        h = None
        for t in range(k_dec):
            z = torch.cat([steps[:, t], y_prev], dim=-1).unsqueeze(1)
            h_t, h = self.lstm(z, h)
            o_t = self.output_scalar * self.final_layer(self.head_ff(h_t.squeeze(1)))
            y_prev = y_prev + o_t if self.parameterization == "velocity" else o_t
            outs.append(y_prev)
        traj = torch.stack(outs, dim=1)  # (B, K_dec, J)
        pred = self.channels_to_trajectory(
            F.interpolate(
                self.trajectory_to_channels(traj),
                size=n_t,
                mode="nearest",
            )
        )
        return pred

    @torch.no_grad()
    def tracking_forward(self, x: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        """Explicit Tracking-mode entry point (ground-truth anchor required)."""
        return self.forward(x, y0=y0)


class _TDSStage(nn.Module):
    """Subsampling conv + TDS residual blocks (Hannun et al. layout)."""

    def __init__(
        self,
        channels: int,
        subsample_kernel: int,
        subsample_stride: int,
        n_blocks: int,
        activation: type[nn.Module],
    ):
        super().__init__()
        pad = subsample_kernel - 1  # causal
        self.sub_conv = nn.Conv1d(
            channels, channels, subsample_kernel, stride=subsample_stride
        )
        self.sub_pad = pad
        self.sub_norm = _TimeStepNorm(channels)
        self.act = activation()
        self.channels_to_sequence = Rearrange("b c t -> b t c")
        self.sequence_to_channels = Rearrange("b t c -> b c t")
        self.conv_blocks = nn.ModuleList(
            [_CausalTDSConvBlock(channels, activation) for _ in range(n_blocks)]
        )
        self.ff_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(channels, 4 * channels),
                    activation(),
                    nn.Linear(4 * channels, channels),
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.sub_norm(self.sub_conv(F.pad(x, (self.sub_pad, 0)))))
        for conv_b, ff_b in zip(self.conv_blocks, self.ff_blocks):
            x = x + conv_b(x)
            x = self.channels_to_sequence(x)
            x = x + ff_b(x)
            x = self.sequence_to_channels(x)
        return x


class _TimeStepNorm(nn.Sequential):
    """Layer normalization over channels, independently at each time step."""

    def __init__(self, channels: int):
        super().__init__(
            Rearrange("b c t -> b t c"),
            nn.LayerNorm(channels),
            Rearrange("b t c -> b c t"),
        )


class _CausalTDSConvBlock(nn.Module):
    """Depthwise TDS convolution with left-only temporal padding."""

    def __init__(self, channels: int, activation: type[nn.Module]):
        super().__init__()
        self.left_padding = 6
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, 7, groups=channels),
            _TimeStepNorm(channels),
            activation(),
            nn.Conv1d(channels, channels, 1),
            _TimeStepNorm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(F.pad(x, (self.left_padding, 0)))
