# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Meta Platforms, Inc. and affiliates (original emg2pose architecture)
#
# License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# Architecture follows Salter et al. 2024 (CC BY-NC-SA 4.0) as revised by
# Hadidi et al. 2026 (arXiv 2603.08212, CC BY 4.0).
"""``VEMG2Pose``: state-conditioned sEMG-to-pose decoder."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import MLP, CausalConv1d


class VEMG2Pose(
    EEGModuleMixin,
    nn.Module,
    license="cc-by-nc-sa-4.0",
):
    r"""VEMG2Pose from Salter et al (2024) [salter2024]_, as revised by
    Hadidi et al (2026) [hadidi2026]_.

    :bdg-success:`Convolution` :bdg-secondary:`Recurrent`

    .. figure:: https://arxiv.org/html/2412.02725v1/figures/fig1_clip.png
       :align: center
       :alt: emg2pose wristband and predicted hand pose

       Figure 1 of Salter et al. (2024). The paper describes vemg2pose in
       prose (Section 3.5, Appendix C) and publishes no architecture
       diagram, so this is its overview figure.


    Causal time-depth-separable (TDS) convolutional encoder producing
    64-D features at 25 Hz, decoded autoregressively at 50 Hz by a
    state-conditioned LSTM whose input concatenates the encoder feature
    with the previous pose estimate. Supports both *position* decoding
    (absolute joint angles) and *velocity* decoding (integrated
    increments) under one architecture.

    .. rubric:: Architecture Overview

    Raw sEMG ``x ∈ (B, n_chans, T)`` → left-padded causal conv stem
    (kernel/stride 11/5 → 5/2, 256 ch) → TDS stage (subsampling conv
    kernel/stride 17/4) → linear squeeze to ``feature_dim`` → TDS stage
    (subsampling conv kernel/stride 9/2) → features ``(B, K, 64)`` at
    ~25 Hz → fixed-rate causal indexing at ``decoder_rate`` (50 Hz)
    → LSTM rollout with pose-state feedback → fixed-rate causal
    indexing back to ``T``.

    .. rubric:: Macro Components

    ``VEMG2Pose.stem`` (strided temporal front-end)
        **Operations.** Two causal conv blocks (LayerNorm + LeakyReLU),
        mapping ``(B, n_chans, T) → (B, encoder_channels, T/10)``
        (kernel 11 stride 5, then kernel 5 stride 2; left-only padding
        preserves causality and sample alignment).
        **Role.** Raw-waveform feature extraction at ~200 Hz.

    ``VEMG2Pose.tds_stages`` (Time-Depth-Separable bottleneck)
        **Operations.** Subsampling conv (kernel 17, stride 4) followed
        by ``tds_blocks`` depthwise-conv + pointwise + feedforward
        residual pairs; a 1×1 projection to ``feature_dim``; a second
        stage with subsampling kernel 9 stride 2.
        **Role.** Hannun-et-al-style factorized sequence modeling down
        to the 25 Hz decoder grid ([hannun2019]_).

    ``VEMG2Pose.lstm`` / ``VEMG2Pose.final_layer``
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
    - **Learned regression anchor** ``initial_pose`` replaces the original
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
    stem_kernel_sizes : tuple of int, optional
        Kernel of each strided stem convolution. The default is ``(11, 5)``.
        Padding is ``kernel - 1`` on the left, keeping the stem causal.
    stem_strides : tuple of int, optional
        Stride of each stem convolution. The default is ``(5, 2)``.
    tds_subsample_kernels : tuple of int, optional
        Subsampling kernel of each TDS stage. The default is ``(17, 9)``.
    tds_subsample_strides : tuple of int, optional
        Subsampling stride of each TDS stage. The default is ``(4, 2)``.
        Together with ``stem_strides`` this fixes the encoder rate:
        ``sfreq / prod(strides)``, i.e. 25 Hz for the paper's 2 kHz input.
    lstm_layers : int, optional
        Depth of the decoder LSTM. The default is ``2``.
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

    # Operation-equivalent subset of Meta's tracking_vemg2pose.ckpt. The
    # remaining upstream TDS blocks and decoder head have different layouts.
    mapping = {
        "model.network.layers.0.conv.0.weight": "stem.1.weight",
        "model.network.layers.0.conv.0.bias": "stem.1.bias",
        "model.network.layers.0.norm.weight": "stem.2.1.weight",
        "model.network.layers.0.norm.bias": "stem.2.1.bias",
        "model.network.layers.1.conv.0.weight": "stem.5.weight",
        "model.network.layers.1.conv.0.bias": "stem.5.bias",
        "model.network.layers.1.norm.weight": "stem.6.1.weight",
        "model.network.layers.1.norm.bias": "stem.6.1.bias",
        "model.network.layers.2.layers.conv1dblock.conv.0.weight": (
            "tds_stages.0.sub_conv.weight"
        ),
        "model.network.layers.2.layers.conv1dblock.conv.0.bias": (
            "tds_stages.0.sub_conv.bias"
        ),
        "model.network.layers.2.layers.conv1dblock.norm.weight": (
            "tds_stages.0.sub_norm.1.weight"
        ),
        "model.network.layers.2.layers.conv1dblock.norm.bias": (
            "tds_stages.0.sub_norm.1.bias"
        ),
        "model.decoder.lstm.weight_ih_l0": "lstm.weight_ih_l0",
        "model.decoder.lstm.weight_hh_l0": "lstm.weight_hh_l0",
        "model.decoder.lstm.bias_ih_l0": "lstm.bias_ih_l0",
        "model.decoder.lstm.bias_hh_l0": "lstm.bias_hh_l0",
        "model.decoder.lstm.weight_ih_l1": "lstm.weight_ih_l1",
        "model.decoder.lstm.weight_hh_l1": "lstm.weight_hh_l1",
        "model.decoder.lstm.bias_ih_l1": "lstm.bias_ih_l1",
        "model.decoder.lstm.bias_hh_l1": "lstm.bias_hh_l1",
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
        hidden_size: int = 512,
        encoder_channels: int = 256,
        feature_dim: int = 64,
        decoder_rate: float = 50.0,
        tds_blocks: int = 2,
        stem_kernel_sizes: tuple[int, int] = (11, 5),
        stem_strides: tuple[int, int] = (5, 2),
        tds_subsample_kernels: tuple[int, int] = (17, 9),
        tds_subsample_strides: tuple[int, int] = (4, 2),
        lstm_layers: int = 2,
        parameterization: str = "position",
        output_scalar: float = 0.1,
        activation: type[nn.Module] = nn.LeakyReLU,
        drop_prob: float = 0.0,
    ):
        if parameterization not in ("position", "velocity"):
            raise ValueError("parameterization must be 'position' or 'velocity'")
        if decoder_rate <= 0:
            raise ValueError(f"decoder_rate must be positive; got {decoder_rate}.")
        if sfreq is not None and sfreq <= 0:
            raise ValueError(f"sfreq must be positive; got {sfreq}.")
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
        self.input_sfreq = float(self.sfreq)
        if self.input_sfreq <= 0:
            raise ValueError(f"sfreq must be positive; got {self.input_sfreq}.")

        # braindecode parameters read post-inference:
        n_pose_outputs = self.n_outputs

        stem_layers: list[nn.Module] = []
        for index, (kernel, stride) in enumerate(
            zip(stem_kernel_sizes, stem_strides, strict=True)
        ):
            in_channels = self.n_chans if index == 0 else encoder_channels
            stem_layers += [
                # Left-only padding: keeps the block causal and the output
                # aligned with the input, so it follows the kernel rather
                # than being restated per stage.
                nn.ConstantPad1d(padding=(kernel - 1, 0), value=0.0),
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=encoder_channels,
                    kernel_size=kernel,
                    stride=stride,
                ),
                _TimeStepNorm(channels=encoder_channels),
                activation(),
            ]
        self.stem = nn.Sequential(*stem_layers)
        self.tds_stages = nn.Sequential(
            _TDSStage(
                channels=encoder_channels,
                subsample_kernel=tds_subsample_kernels[0],
                subsample_stride=tds_subsample_strides[0],
                n_blocks=tds_blocks,
                activation=activation,
            ),
            nn.Conv1d(
                in_channels=encoder_channels,
                out_channels=feature_dim,
                kernel_size=1,
            ),
            _TDSStage(
                channels=feature_dim,
                subsample_kernel=tds_subsample_kernels[1],
                subsample_stride=tds_subsample_strides[1],
                n_blocks=tds_blocks,
                activation=activation,
            ),
        )

        # Read off the strides actually configured rather than restated as a
        # literal: at the defaults this is 5 * 2 * 4 * 2 = 80, so a 2 kHz input
        # reaches the decoder at 25 Hz, and it stays right if they change.
        self.encoder_decimation = math.prod((*stem_strides, *tds_subsample_strides))
        self.encoder_sfreq = float(self.sfreq) / self.encoder_decimation
        self.lstm = nn.LSTM(
            input_size=feature_dim + n_pose_outputs,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.head_ff = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size // 2),
            activation(),
            nn.Dropout(p=drop_prob),
        )
        self.encoder_to_sequence = Rearrange(
            "batch features ntimes -> batch ntimes features"
        )
        # final_layer LAST so it lands in the last two named_children().
        self.final_layer = nn.Linear(
            in_features=hidden_size // 2, out_features=n_pose_outputs
        )

        self.initial_pose = nn.Parameter(torch.zeros(n_pose_outputs))
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
        self._set_n_outputs(n_outputs)
        old = self.final_layer
        device, dtype = old.weight.device, old.weight.dtype
        self.final_layer = nn.Linear(
            in_features=old.in_features, out_features=n_outputs
        ).to(device=device, dtype=dtype)
        self.final_layer.apply(self._init_weights)
        self.initial_pose = nn.Parameter(
            torch.zeros(n_outputs, device=device, dtype=dtype)
        )
        self.lstm = nn.LSTM(
            input_size=self.tds_stages[-1].sub_conv.out_channels + n_outputs,
            hidden_size=self.lstm.hidden_size,
            num_layers=self.lstm.num_layers,
            batch_first=True,
        ).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, y0: torch.Tensor | None = None) -> torch.Tensor:
        """Decode hand pose from raw EMG.

        Parameters
        ----------
        x : Tensor (B, n_chans, T)
            Raw sEMG window at ``sfreq``.
        y0 : Tensor (B, n_outputs) | None
            Ground-truth initial pose (Tracking mode). Falls back to the
            learned ``initial_pose`` (Regression mode) when omitted.
        """
        batch_size, _, n_times = x.shape
        encoder_features = self.encoder_to_sequence(self.tds_stages(self.stem(x)))

        # Rollout length and both resampling maps follow the input length, so
        # variable-length windows stay supported. With static input shapes these
        # resolve to constants: torch.export unrolls the rollout (graph size then
        # grows with window length), while TorchScript keeps a real prim::Loop.
        n_decoder_steps = max(
            1, math.ceil(n_times * self.decoder_rate / self.input_sfreq)
        )
        encoder_step_indices = (
            (
                torch.arange(n_decoder_steps, device=x.device, dtype=torch.float32)
                * self.encoder_sfreq
                / self.decoder_rate
            )
            .floor()
            .to(dtype=torch.long)
            .clamp_max(encoder_features.shape[1] - 1)
        )
        # Hold-last-sample upsampling from the encoder rate to the decoder rate.
        decoder_features = encoder_features.index_select(1, encoder_step_indices)

        predicted_pose = (
            y0 if y0 is not None else self.initial_pose.expand(batch_size, -1)
        )
        # Explicit zero state rather than ``None``: same numerics as the LSTM
        # default, but a stable Tuple[Tensor, Tensor] type across the loop.
        # torch.jit.script rejects rebinding a NoneType variable to a tuple.
        hidden_state = torch.zeros(
            self.lstm.num_layers,
            batch_size,
            self.lstm.hidden_size,
            device=x.device,
            dtype=x.dtype,
        )
        cell_state = torch.zeros_like(hidden_state)

        pose_per_step = []
        for step in range(n_decoder_steps):
            decoder_input = torch.cat(
                [decoder_features[:, step], predicted_pose], dim=-1
            ).unsqueeze(1)
            lstm_output, (hidden_state, cell_state) = self.lstm(
                decoder_input, (hidden_state, cell_state)
            )
            pose_step = self.output_scalar * self.final_layer(
                self.head_ff(lstm_output.squeeze(1))
            )
            predicted_pose = (
                predicted_pose + pose_step
                if self.parameterization == "velocity"
                else pose_step
            )
            pose_per_step.append(predicted_pose)

        pose_trajectory = torch.stack(pose_per_step, dim=1)
        # Hold-last-sample downsampling from the decoder rate back to n_times.
        decoder_step_indices = (
            (
                torch.arange(n_times, device=x.device, dtype=torch.float32)
                * self.decoder_rate
                / self.input_sfreq
            )
            .floor()
            .to(dtype=torch.long)
            .clamp_max(n_decoder_steps - 1)
        )
        return pose_trajectory.index_select(1, decoder_step_indices)

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
        self.sub_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=subsample_kernel,
            stride=subsample_stride,
        )
        self.sub_pad = subsample_kernel - 1  # left-only padding keeps causality
        self.sub_norm = _TimeStepNorm(channels=channels)
        self.act = activation()
        self.channels_to_sequence = Rearrange(
            "batch channels ntimes -> batch ntimes channels"
        )
        self.sequence_to_channels = Rearrange(
            "batch ntimes channels -> batch channels ntimes"
        )
        self.conv_blocks = nn.ModuleList(
            [
                _CausalTDSConvBlock(channels=channels, activation=activation)
                for _ in range(n_blocks)
            ]
        )
        self.ff_blocks = nn.ModuleList(
            [
                MLP(
                    in_features=channels,
                    hidden_features=(4 * channels,),
                    out_features=channels,
                    activation=activation,
                    drop=0.0,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.sub_norm(self.sub_conv(F.pad(x, (self.sub_pad, 0)))))
        for conv_block, ff_block in zip(self.conv_blocks, self.ff_blocks):
            x = x + conv_block(x)
            x = self.channels_to_sequence(x)
            x = x + ff_block(x)
            x = self.sequence_to_channels(x)
        return x


class _TimeStepNorm(nn.Sequential):
    """Layer normalization over channels, independently at each time step."""

    def __init__(self, channels: int):
        super().__init__(
            Rearrange("batch channels ntimes -> batch ntimes channels"),
            nn.LayerNorm(normalized_shape=channels),
            Rearrange("batch ntimes channels -> batch channels ntimes"),
        )


class _CausalTDSConvBlock(nn.Module):
    """Depthwise TDS convolution with left-only temporal padding."""

    def __init__(self, channels: int, activation: type[nn.Module]):
        super().__init__()
        self.layers = nn.Sequential(
            CausalConv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=7,
                groups=channels,
            ),
            _TimeStepNorm(channels=channels),
            activation(),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1),
            _TimeStepNorm(channels=channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
