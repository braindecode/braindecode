# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Meta Platforms, Inc. and affiliates (original emg2pose architecture)
#
# License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# Architecture follows Salter et al. 2024 (CC BY-NC-SA 4.0).
"""``VEMG2Pose``: state-conditioned sEMG-to-pose decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from braindecode.models.base import EEGModuleMixin


class VEMG2Pose(
    EEGModuleMixin,
    nn.Module,
    license="cc-by-nc-sa-4.0",
):
    r"""VEMG2Pose from Salter et al (2024) [salter2024]_.

    :bdg-success:`Convolution` :bdg-secondary:`Recurrent`

    .. figure:: https://arxiv.org/html/2412.02725v1/figures/fig1_clip.png
       :align: center
       :alt: emg2pose wristband and predicted hand pose

       Figure 1 of Salter et al. (2024). The paper describes vemg2pose in
       prose (Section 3.5, Appendix C) and publishes no architecture
       diagram, so this is its overview figure.

    Time-depth-separable (TDS) convolutional encoder [hannun2019]_ feeding a
    state-conditioned LSTM that rolls the hand pose out at 50 Hz. The decoder
    emits a position *and* a velocity for every joint at each step: it takes
    the position for the first ``num_position_steps``, then integrates the
    velocity from there on. That hybrid is what the leading ``v`` names.

    .. rubric:: Architecture Overview

    ``(B, n_chans, T)`` → two strided ``Conv1d`` blocks (kernel 11 stride 5,
    then 5 / 2, LayerNorm over channels) → two TDS stages, each a strided
    subsampling convolution followed by ``tds_blocks`` pairs of a
    time-depth-separable 2-D convolution and a residual feed-forward block →
    a linear projection to ``feature_dim`` → linear resampling to
    ``rollout_rate`` → LSTM rollout conditioned on the previous pose →
    ``(B, T, n_outputs)``.

    Every convolution is *valid* (no padding), so the encoder consumes a
    ``left_context`` of ``1790`` samples at the default schedule; the
    prediction spans the remainder and is resampled back to ``T``.

    .. rubric:: Macro Components

    ``VEMG2Pose.encoder``
        **Operations.** ``_Conv1dBlock`` ×2 then ``_TdsStage`` ×2. A stage is
        a strided ``Conv1d`` + LayerNorm, then ``tds_blocks`` × (
        ``_TDSConv2dBlock``, ``_TDSFullyConnectedBlock``), both residual and
        each closing with a LayerNorm over the channel axis.
        **Role.** Compresses 2 kHz sEMG to ``feature_dim`` features at
        ``sfreq / prod(strides)``, i.e. 25 Hz for the paper's input.

    ``VEMG2Pose.lstm`` / ``VEMG2Pose.final_layer``
        **Operations.** Two-layer LSTM over ``[feature ; previous pose]``,
        then LeakyReLU and a linear map to ``2 * n_outputs``, scaled by
        ``output_scalar``. The halves are the position and the velocity.
        **Role.** Autoregressive pose rollout with state feedback.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Time:** Strided convolutions then an explicit 50 Hz rollout.
    - **Channels/space:** The first convolution mixes all electrodes; the TDS
      blocks then convolve within a channel/width factorisation.
    - **Frequency:** Learned from the raw waveform; nothing is filtered out.

    .. rubric:: Additional Mechanisms

    Passing ``y0`` conditions the rollout on a known initial pose (the
    paper's *tracking* regime). Omitting it starts from zeros, which is the
    *regression* regime the released ``regression_vemg2pose.ckpt`` was
    trained under.

    .. rubric:: Pre-trained weights

    Meta's released checkpoints are rehosted with these parameter names::

        # vemg2pose: recurrent decoder
        VEMG2Pose.from_pretrained("braindecode/VEMG2Pose-emg2pose")
        VEMG2Pose.from_pretrained("braindecode/VEMG2Pose-emg2pose-tracking")
        # emg2pose: same encoder, stateless MLP decoder
        VEMG2Pose.from_pretrained("braindecode/EMG2Pose-emg2pose")
        VEMG2Pose.from_pretrained("braindecode/EMG2Pose-emg2pose-tracking")

    Each config records its own ``decoder`` and ``parameterization``, so the
    right rollout is restored automatically.

    The original ``.ckpt`` files also load directly, since ``mapping``
    rewrites the upstream key names. Both routes reproduce the reference
    implementation exactly. The weights remain under emg2pose's
    CC BY-NC-SA 4.0.

    .. versionadded:: 1.8

    Parameters
    ----------
    encoder_channels : int, optional
        Width of the strided stem convolutions and of the TDS stages. The
        default is ``256``.
    feature_dim : int, optional
        Width the encoder projects to before the decoder. The default is
        ``64``.
    hidden_size : int, optional
        LSTM hidden width. The default is ``512``.
    lstm_layers : int, optional
        Depth of the decoder LSTM, when ``decoder='lstm'``. The default is
        ``2``.
    decoder : {'lstm', 'mlp'}, optional
        Rollout decoder. ``'lstm'`` (the default) is the recurrent decoder
        the ``vemg2pose`` checkpoints use. ``'mlp'`` is the stateless
        normalised MLP behind emg2pose's own ``emg2pose`` baseline; both are
        state-conditioned on the previous pose either way.
    decoder_hidden_sizes : tuple of int, optional
        Hidden widths of the MLP decoder, unused when ``decoder='lstm'``.
        The default is ``(512, 512)``.
    stem_kernel_sizes : tuple of int, optional
        Kernel of each strided stem convolution. The default is ``(11, 5)``.
    stem_strides : tuple of int, optional
        Stride of each stem convolution. The default is ``(5, 2)``.
    tds_subsample_kernels : tuple of int, optional
        Subsampling kernel of each TDS stage. The default is ``(17, 9)``.
    tds_subsample_strides : tuple of int, optional
        Subsampling stride of each TDS stage. The default is ``(4, 2)``.
    tds_kernel_widths : tuple of int, optional
        Temporal kernel inside each stage's TDS blocks, which must be odd.
        The default is ``(9, 5)``.
    tds_blocks : int, optional
        Conv/feed-forward block pairs per TDS stage. The default is ``2``.
    tds_channels : int, optional
        Channel count of the TDS factorisation; the width is
        ``encoder_channels // tds_channels``. The default is ``16``.
    rollout_rate : float, optional
        Rate in Hz at which the decoder is unrolled. The default is
        ``50.0``.
    parameterization : {'hybrid', 'velocity', 'position'}, optional
        How the decoder output becomes a pose. ``'hybrid'`` (the default,
        and what ``regression_vemg2pose.ckpt`` was trained as) emits a
        position and a velocity per joint, taking the position for
        ``num_position_steps`` and integrating the velocity thereafter.
        ``'velocity'`` integrates a single velocity from ``y0``, which is
        the ``tracking_vemg2pose.ckpt`` setting. ``'position'`` reads the
        pose out directly.
    num_position_steps : int, optional
        Input samples for which the decoder's position output is used
        directly, before it switches to integrating velocity. The default is
        ``500`` (250 ms at 2 kHz).
    output_scalar : float, optional
        Fixed multiplier on the decoder output. The default is ``0.01``.
    activation : type[nn.Module], optional
        Activation before the output projection. The default is
        ``nn.LeakyReLU``.
    drop_prob : float, optional
        Dropout inside the convolution blocks. The default is ``0.0``.

    References
    ----------
    .. [salter2024] Salter, Warren, Schlager, Spurr, Han, Bhasin, Rohin,
       et al. (2024). emg2pose: A Large and Diverse Benchmark for Surface
       Electromyographic Hand Pose Estimation. NeurIPS Datasets and
       Benchmarks. arXiv:2412.02725.
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
        encoder_channels: int = 256,
        feature_dim: int = 64,
        hidden_size: int = 512,
        lstm_layers: int = 2,
        decoder: str = "lstm",
        decoder_hidden_sizes: tuple[int, ...] = (512, 512),
        stem_kernel_sizes: tuple[int, ...] = (11, 5),
        stem_strides: tuple[int, ...] = (5, 2),
        tds_subsample_kernels: tuple[int, ...] = (17, 9),
        tds_subsample_strides: tuple[int, ...] = (4, 2),
        tds_kernel_widths: tuple[int, ...] = (9, 5),
        tds_blocks: int = 2,
        tds_channels: int = 16,
        rollout_rate: float = 50.0,
        num_position_steps: int = 500,
        parameterization: str = "hybrid",
        output_scalar: float = 0.01,
        activation: type[nn.Module] = nn.LeakyReLU,
        drop_prob: float = 0.0,
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

        if self.sfreq is None or float(self.sfreq) <= 0:
            raise ValueError(f"sfreq must be positive; got {self.sfreq}.")
        if rollout_rate <= 0:
            raise ValueError(f"rollout_rate must be positive; got {rollout_rate}.")
        if len(tds_subsample_kernels) != len(tds_subsample_strides) or len(
            tds_subsample_kernels
        ) != len(tds_kernel_widths):
            raise ValueError(
                "tds_subsample_kernels, tds_subsample_strides and "
                "tds_kernel_widths must have the same length."
            )
        if decoder not in ("lstm", "mlp"):
            raise ValueError(f"decoder must be 'lstm' or 'mlp'; got {decoder!r}.")
        if parameterization not in ("hybrid", "velocity", "position"):
            raise ValueError(
                "parameterization must be 'hybrid', 'velocity' or 'position'; "
                f"got {parameterization!r}."
            )
        if encoder_channels % tds_channels:
            raise ValueError(
                f"tds_channels ({tds_channels}) must divide encoder_channels "
                f"({encoder_channels})."
            )

        self.input_sfreq = float(self.sfreq)
        self.rollout_rate = float(rollout_rate)
        self.num_position_steps = int(num_position_steps)
        self.parameterization = parameterization
        self.output_scalar = float(output_scalar)

        # ``layers`` mirrors upstream's single Sequential so the checkpoint
        # map only has to rewrite the prefix.
        layers: list[nn.Module] = []
        for index, (kernel, stride) in enumerate(
            zip(stem_kernel_sizes, stem_strides, strict=True)
        ):
            layers.append(
                _Conv1dBlock(
                    in_channels=self.n_chans if index == 0 else encoder_channels,
                    out_channels=encoder_channels,
                    kernel_size=kernel,
                    stride=stride,
                    drop_prob=drop_prob,
                )
            )
        n_stages = len(tds_subsample_kernels)
        for index in range(n_stages):
            layers.append(
                _TdsStage(
                    in_channels=encoder_channels,
                    subsample_kernel=tds_subsample_kernels[index],
                    subsample_stride=tds_subsample_strides[index],
                    n_blocks=tds_blocks,
                    channels=tds_channels,
                    feature_width=encoder_channels // tds_channels,
                    kernel_width=tds_kernel_widths[index],
                    # Only the last stage projects down to the decoder width.
                    out_channels=feature_dim if index == n_stages - 1 else None,
                    drop_prob=drop_prob,
                )
            )
        self.encoder = nn.Sequential(*layers)

        # Samples the valid convolutions consume before the first prediction.
        left_context, stride = 0, 1
        for kernel, kernel_stride in zip(stem_kernel_sizes, stem_strides, strict=True):
            left_context += (kernel - 1) * stride
            stride *= kernel_stride
        for index in range(n_stages):
            left_context += (tds_subsample_kernels[index] - 1) * stride
            stride *= tds_subsample_strides[index]
            left_context += tds_blocks * (tds_kernel_widths[index] - 1) * stride
        self.left_context = left_context
        if self.n_times <= self.left_context:
            raise ValueError(
                f"VEMG2Pose consumes a left context of {self.left_context} "
                f"samples before its first prediction; got n_times="
                f"{self.n_times}. Use a longer window."
            )

        # Both decoders take (feature, hidden, cell) and return the same
        # triple, so ``forward`` never branches -- TorchScript compiles every
        # branch it sees, and would reject a conditionally-typed submodule.
        decoder_in = feature_dim + self.n_outputs
        if decoder == "lstm":
            self.decoder: nn.Module = _LstmPoseDecoder(
                in_features=decoder_in,
                hidden_size=hidden_size,
                num_layers=lstm_layers,
                activation=activation,
            )
            head_in = hidden_size
        else:
            self.decoder = _MlpPoseDecoder(
                in_features=decoder_in,
                hidden_sizes=decoder_hidden_sizes,
                activation=activation,
            )
            head_in = decoder_hidden_sizes[-1]
        self.decoder_type = decoder
        # State carried through the rollout; the MLP decoder ignores it, so a
        # single dummy element keeps the signature uniform.
        self._state_layers = lstm_layers if decoder == "lstm" else 1
        self._state_size = hidden_size if decoder == "lstm" else 1
        # ``hybrid`` emits a position *and* a velocity per joint, so the head
        # is twice as wide and the halves are split in ``forward``. The other
        # two modes emit one value per joint. Last child, as braindecode
        # expects of the head.
        self._head_multiplier = 2 if parameterization == "hybrid" else 1
        self.final_layer = nn.Linear(
            in_features=head_in,
            out_features=self._head_multiplier * self.n_outputs,
        )

    @property
    def mapping(self) -> dict[str, str]:
        """Map the released emg2pose checkpoints onto this module's names.

        Upstream wraps the encoder in a LightningModule as ``model.network``
        and the rollout head as ``model.decoder``. The block layout is
        identical, so the map is read off the encoder rather than restated,
        and stays correct if the schedule is reconfigured.
        """
        mapping = {
            f"model.network.layers.{index}.{key}": f"encoder.{index}.{key}"
            for index, block in enumerate(self.encoder)
            for key in block.state_dict()
        }
        for key in self.decoder.state_dict():
            # The LSTM decoder keeps upstream's own attribute name; the MLP
            # one calls its stack ``layers`` where upstream calls it ``mlp``.
            upstream = (
                key
                if self.decoder_type == "lstm"
                else key.replace("layers.", "mlp.", 1)
            )
            mapping[f"model.decoder.{upstream}"] = f"decoder.{key}"
        # Upstream's decoder ends in the head this class calls final_layer:
        # mlp_out is (LeakyReLU, Linear) for the LSTM, and the trailing Linear
        # of the MLP stack otherwise.
        head = (
            "mlp_out.1"
            if self.decoder_type == "lstm"
            else f"mlp.{len(self.decoder.layers)}"
        )
        mapping[f"model.decoder.{head}.weight"] = "final_layer.weight"
        mapping[f"model.decoder.{head}.bias"] = "final_layer.bias"
        return mapping

    def reset_head(self, n_outputs: int) -> None:
        """Swap the output layer for a new output dimensionality.

        The decoded pose feeds back as decoder input, so changing
        ``n_outputs`` also rebuilds the recurrent stack; its weights are
        re-initialised.
        """
        self._set_n_outputs(n_outputs)
        old = self.final_layer
        device, dtype = old.weight.device, old.weight.dtype
        self.final_layer = nn.Linear(
            in_features=old.in_features,
            out_features=self._head_multiplier * n_outputs,
        ).to(device=device, dtype=dtype)
        feature_dim = self.encoder[-1].out_channels
        self.decoder = self.decoder.rebuilt(in_features=feature_dim + n_outputs).to(
            device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor, y0: torch.Tensor | None = None) -> torch.Tensor:
        """Decode hand pose from raw EMG.

        Parameters
        ----------
        x : Tensor (B, n_chans, T)
            Raw sEMG window at ``sfreq``.
        y0 : Tensor (B, n_outputs) | None
            Ground-truth initial pose (Tracking mode). Falls back to zeros
            (Regression mode) when omitted, which is how the released
            ``regression`` checkpoint was trained.
        """
        batch_size, _, n_times = x.shape
        features = self.encoder(x)

        # Predictions span the input past the left context, unrolled at
        # ``rollout_rate``.
        valid_seconds = (n_times - self.left_context) / self.input_sfreq
        # int(): TorchScript's round() returns a float, which interpolate rejects.
        n_steps = max(1, int(round(valid_seconds * self.rollout_rate)))
        features = F.interpolate(
            features, size=n_steps, mode="linear", align_corners=True
        )

        # Steps taken as position before switching to velocity integration.
        position_steps = int(
            round(self.num_position_steps * (self.rollout_rate / self.input_sfreq))
        )

        predicted_pose = (
            y0 if y0 is not None else x.new_zeros((batch_size, self.n_outputs))
        )
        hidden_state = x.new_zeros((self._state_layers, batch_size, self._state_size))
        cell_state = torch.zeros_like(hidden_state)

        pose_per_step = []
        for step in range(n_steps):
            decoder_input = torch.cat([features[:, :, step], predicted_pose], dim=-1)
            decoder_output, hidden_state, cell_state = self.decoder(
                decoder_input, hidden_state, cell_state
            )
            output = self.final_layer(decoder_output) * self.output_scalar
            if self.parameterization == "hybrid":
                position, velocity = torch.split(output, self.n_outputs, dim=1)
                predicted_pose = (
                    position if step < position_steps else predicted_pose + velocity
                )
            elif self.parameterization == "velocity":
                predicted_pose = predicted_pose + output
            else:
                predicted_pose = output
            pose_per_step.append(predicted_pose)

        trajectory = torch.stack(pose_per_step, dim=-1)  # (B, n_outputs, steps)
        # Back to the input grid, so the model keeps braindecode's contract.
        trajectory = F.interpolate(trajectory, size=n_times, mode="linear")
        return trajectory.transpose(1, 2)

    def tracking_forward(self, x: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        """Explicit Tracking-mode entry point (ground-truth anchor required)."""
        return self.forward(x, y0=y0)


class _Conv1dBlock(nn.Module):
    """Valid ``Conv1d`` + activation + dropout, then LayerNorm over channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        drop_prob: float,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        # ``conv`` and ``norm`` mirror the upstream attribute names.
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
        )
        self.norm = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)


class _TDSConv2dBlock(nn.Module):
    """Time-depth-separable convolution over a channel/width factorisation."""

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        if kernel_width % 2 == 0:
            raise ValueError(f"kernel_width must be odd; got {kernel_width}.")
        self.channels = channels
        self.width = width
        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(normalized_shape=channels * width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_features, n_times = x.shape
        out = x.reshape(batch, self.channels, self.width, n_times)
        out = self.relu(self.conv2d(out))
        out = out.reshape(batch, n_features, -1)
        # Valid convolution shortens the sequence, so the residual is taken
        # from the tail of the input.
        out = out + x[..., -out.shape[-1] :]
        return self.layer_norm(out.swapaxes(-1, -2)).swapaxes(-1, -2)


class _TDSFullyConnectedBlock(nn.Module):
    """Residual position-wise feed-forward block with a trailing LayerNorm."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=n_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=n_features, out_features=n_features),
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc_block(x.swapaxes(-1, -2)).swapaxes(-1, -2) + x
        return self.layer_norm(out.swapaxes(-1, -2)).swapaxes(-1, -2)


class _TDSConvEncoder(nn.Module):
    """``n_blocks`` pairs of a TDS convolution and a feed-forward block."""

    def __init__(
        self, n_features: int, n_blocks: int, channels: int, kernel_width: int
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for _ in range(n_blocks):
            blocks += [
                _TDSConv2dBlock(
                    channels=channels,
                    width=n_features // channels,
                    kernel_width=kernel_width,
                ),
                _TDSFullyConnectedBlock(n_features=n_features),
            ]
        self.tds_conv_blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(x)


class _TdsStage(nn.Module):
    """Subsampling convolution, TDS blocks, and an optional projection."""

    def __init__(
        self,
        in_channels: int,
        subsample_kernel: int,
        subsample_stride: int,
        n_blocks: int,
        channels: int,
        feature_width: int,
        kernel_width: int,
        out_channels: int | None,
        drop_prob: float,
    ) -> None:
        super().__init__()
        n_features = channels * feature_width
        self.out_channels = out_channels if out_channels is not None else n_features
        self.layers = nn.Sequential()
        self.layers.add_module(
            "conv1dblock",
            _Conv1dBlock(
                in_channels=in_channels,
                out_channels=n_features,
                kernel_size=subsample_kernel,
                stride=subsample_stride,
                drop_prob=drop_prob,
            ),
        )
        self.layers.add_module(
            "tds_block",
            _TDSConvEncoder(
                n_features=n_features,
                n_blocks=n_blocks,
                channels=channels,
                kernel_width=kernel_width,
            ),
        )
        # Identity when the stage does not project, so the attribute always
        # exists (TorchScript rejects a conditionally-defined submodule) and
        # contributes no state_dict keys.
        self.linear_layer: nn.Module = (
            nn.Linear(in_features=n_features, out_features=out_channels)
            if out_channels is not None
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return self.linear_layer(x.swapaxes(-1, -2)).swapaxes(-1, -2)


class _LstmPoseDecoder(nn.Module):
    """Recurrent rollout decoder: an LSTM step followed by an activation."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        num_layers: int,
        activation: type[nn.Module],
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.activation = activation()
        self._activation_cls = activation

    def rebuilt(self, in_features: int) -> "_LstmPoseDecoder":
        """A fresh decoder of the same shape for a new input width."""
        return _LstmPoseDecoder(
            in_features=in_features,
            hidden_size=self.lstm.hidden_size,
            num_layers=self.lstm.num_layers,
            activation=self._activation_cls,
        )

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, (hidden, cell) = self.lstm(x.unsqueeze(1), (hidden, cell))
        return self.activation(output[:, 0]), hidden, cell


class _MlpPoseDecoder(nn.Module):
    """Stateless rollout decoder: a normalised MLP over the same input.

    The hidden and cell tensors are accepted and returned untouched so the
    two decoders share one signature, which keeps ``forward`` branch-free.
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: tuple[int, ...],
        activation: type[nn.Module],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = in_features
        for hidden_size in hidden_sizes:
            layers += [
                nn.Linear(in_features=width, out_features=hidden_size),
                nn.LayerNorm(normalized_shape=hidden_size),
                activation(),
            ]
            width = hidden_size
        self.layers = nn.Sequential(*layers)
        self._hidden_sizes = tuple(hidden_sizes)
        self._activation_cls = activation

    def rebuilt(self, in_features: int) -> "_MlpPoseDecoder":
        """A fresh decoder of the same shape for a new input width."""
        return _MlpPoseDecoder(
            in_features=in_features,
            hidden_sizes=self._hidden_sizes,
            activation=self._activation_cls,
        )

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.layers(x), hidden, cell
