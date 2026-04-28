# Authors: Meta Platforms, Inc. and affiliates (original emg2qwerty)
#          Bruno Aristimunha <b.aristimunha@gmail.com> (Braindecode adaptation)
#
# License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# Adapted from https://github.com/facebookresearch/emg2qwerty (NeurIPS 2024).
# Inherits CC BY-NC 4.0; not covered by braindecode's BSD-3 license.
"""``EMG2QwertyNet``: TDS-Conv-CTC for sEMG-to-keystroke decoding."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from braindecode.models.base import EEGModuleMixin


class EMG2QwertyNet(EEGModuleMixin, nn.Module):
    r"""Decoder mapping surface electromyography (sEMG) to keystrokes
    (emg2qwerty) [emg2qwerty2024]_.

    :bdg-success:`Convolution`

    Time-Depth-Separable (TDS) [hannun2019tds]_ convolutional encoder
    followed by a Connectionist Temporal Classification (CTC) head
    [graves2006ctc]_. Takes raw 32-channel sEMG (2 wristbands × 16
    electrodes) at 2 kHz and emits per-frame scores over the 99-class
    typing vocabulary (98 keys + 1 CTC blank).

    .. rubric:: Pipeline

    1. **Log-spectrogram front-end**: per-channel Short-Time Fourier
       Transform (:func:`~torch.stft`) with Hann window, ``center=False``,
       then squared magnitude and ``log10(p + log_eps)``. With the
       defaults (``n_fft=64``, ``hop_length=16``, ``sfreq=2000``) the
       output frame rate is 125 Hz. No trainable parameters.
    2. **Spectrogram BatchNorm**: :class:`~torch.nn.BatchNorm2d` over
       the ``(batch, freq, time)`` slice for each of the
       ``num_bands × electrodes_per_band`` channels.
    3. **Per-band rotation-invariant multi-layer perceptron (MLP)**:
       for each band, a shared MLP is applied to circular rolls
       ``(-1, 0, +1)`` of the electrode axis, then mean-pooled.
    4. **TDS convolutional encoder**: stack of ``len(block_channels)``
       TDS conv blocks interleaved with feedforward blocks. No temporal
       padding, so each conv block strips ``kernel_width - 1`` frames.
    5. **Linear classification head**: :class:`~torch.nn.Linear`
       projecting to ``n_outputs``, optionally followed by
       :func:`~torch.nn.functional.log_softmax` (off by default;
       braindecode models return logits).

    .. rubric:: Output

    Returns ``(batch, T_out, n_outputs)``. With ``n_times=8000`` and
    defaults, ``T_out=373``. For :class:`~torch.nn.CTCLoss`, transpose
    to ``(T_out, batch, n_outputs)``; use :meth:`compute_output_lengths`
    for emission lengths.

    .. rubric:: Paper training recipe

    - **Loss**: :class:`~torch.nn.CTCLoss` on log-softmax outputs.
    - **Vocabulary**: 98 keys + 1 blank (``n_outputs = 99``).
    - **Optimizer**: :class:`~torch.optim.Adam`, lr 1e-3, weight decay 0.
    - **Schedule**: 10-epoch linear warmup from lr 1e-8, then cosine
      annealing to 1e-6 over 150 epochs. The slow warmup is required.
      Without it, CTC collapses to all-blank within one epoch (a trivial
      local minimum).
    - **Augmentation**: per-band electrode rotations by -1/0/+1 positions,
      ±60-sample temporal jitter, and SpecAugment [park2019specaug]_ on
      the log-spectrogram.
    - **Decoding**: greedy CTC. Upstream also reports a 6-gram KenLM
      beam decoder, not ported here.

    .. warning::
        The rotation-invariant MLP assumes circular adjacency of the
        electrodes within each band (the wristband geometry of the
        paper, ``electrodes_per_band=16``). For arbitrary EEG montages
        the symmetry does not hold and this model should not be used
        as-is.

    .. warning::
        **License: noncommercial use only.** This module is a derivative
        of Meta's reference implementation released under
        `CC BY-NC 4.0 <https://creativecommons.org/licenses/by-nc/4.0/>`_,
        the same license as the upstream repository. Not covered by
        braindecode's BSD-3 license. Must not be used in commercial
        products or services. Pretrained weights from the upstream release
        carry the same restriction.

    .. versionadded:: 1.5

    Parameters
    ----------
    n_outputs : int
        Vocabulary size for CTC, including the blank class. Defaults to
        ``99`` (98 keys + 1 blank).
    n_chans : int
        Number of EMG channels. Must equal
        ``num_bands * electrodes_per_band`` (default ``32`` = ``2 * 16``).
    sfreq : float
        Sampling frequency in Hz. Defaults to ``2000``. ``n_fft`` and
        ``hop_length`` defaults are calibrated for this rate; pass
        matching values when changing ``sfreq``.
    num_bands : int
        Number of EMG bands (e.g. one per wristband). Defaults to ``2``.
    electrodes_per_band : int
        Number of electrodes per band. Defaults to ``16``. The
        rotation-invariant MLP assumes circular adjacency along this
        axis.
    n_fft : int
        STFT window size in samples.
    hop_length : int
        STFT hop in samples.
    log_eps : float
        Floor added inside ``log10(power + log_eps)`` to keep the log
        finite at silent samples. Defaults to ``1e-6``.
    mlp_features : sequence of int
        Hidden sizes of the rotation-invariant MLP. Output dim per band
        is ``mlp_features[-1]``.
    rotation_offsets : sequence of int
        Circular electrode offsets used to enforce approximate
        rotation invariance. Defaults to ``(-1, 0, 1)``.
    pooling : {"mean", "max"}
        Pool reduction across the rotation rolls. Defaults to ``"mean"``.
    block_channels : sequence of int
        Channel count per TDS convolutional block. The model's internal
        ``num_features = num_bands * mlp_features[-1]`` must be evenly
        divisible by each entry.
    kernel_width : int
        Temporal kernel size of each TDS convolutional block.
    log_softmax : bool
        If ``True``, apply :func:`~torch.nn.functional.log_softmax` to the
        emissions. Disabled by default (braindecode models return logits).
    activation : type of nn.Module
        Activation class used inside the rotation-invariant MLP and the
        TDS blocks. Defaults to :class:`~torch.nn.ReLU` (matches upstream
        emg2qwerty). Pass any non-parametrized activation class
        (:class:`~torch.nn.GELU`, :class:`~torch.nn.SiLU`, …) for
        ablations.
    drop_prob : float
        Dropout probability applied inside each TDS feedforward block,
        once after the activation between the two :class:`~torch.nn.Linear`
        layers and again after the second :class:`~torch.nn.Linear`.
        Default ``0.0`` matches the upstream paper recipe (no dropout).
        Set ``> 0`` for regularized training.

    Examples
    --------
    Build the model and run a forward pass on a 4-second sEMG batch::

        import torch
        from braindecode.models import EMG2QwertyNet

        model = EMG2QwertyNet(
            n_outputs=99, n_chans=32, n_times=8000, sfreq=2000,
        )
        x = torch.randn(2, 32, 8000)
        emissions = model(x)

    Compute a CTC loss on the emissions::

        import torch.nn as nn
        import torch.nn.functional as F

        log_probs = F.log_softmax(emissions, dim=-1).transpose(0, 1)
        input_lengths = model.compute_output_lengths(
            torch.tensor([8000, 8000])
        )
        targets = torch.randint(0, 98, (2, 20), dtype=torch.long)
        target_lengths = torch.tensor([20, 15], dtype=torch.int32)
        loss = nn.CTCLoss(blank=98, zero_infinity=True)(
            log_probs, targets, input_lengths, target_lengths,
        )

    References
    ----------
    .. [emg2qwerty2024] Sivakumar, V., Seely, J., Du, A., Bittner, S. R.,
        Berenzweig, A., Bolarinwa, A., Gramfort, A., Mandel, M. I., 2024.
        emg2qwerty: A Large Dataset with Baselines for Touch Typing using
        Surface Electromyography. Advances in Neural Information Processing
        Systems 37, 91373-91389.
    .. [graves2006ctc] Graves, A., Fernandez, S., Gomez, F.,
        Schmidhuber, J., 2006. Connectionist temporal classification:
        labelling unsegmented sequence data with recurrent neural
        networks. Proc. ICML, 369-376.
    .. [hannun2019tds] Hannun, A., Lee, A., Xu, Q., Collobert, R., 2019.
        Sequence-to-Sequence Speech Recognition with Time-Depth Separable
        Convolutions. arXiv:1904.02619.
    .. [park2019specaug] Park, D. S. et al., 2019. SpecAugment: a simple
        data augmentation method for automatic speech recognition.
        Proc. Interspeech, 2613-2617.
    """

    mapping = {
        "model.4.weight": "final_layer.weight",
        "model.4.bias": "final_layer.bias",
    }

    def __init__(
        self,
        n_outputs: int = 99,
        n_chans: int = 32,
        sfreq: float = 2000.0,
        num_bands: int = 2,
        electrodes_per_band: int = 16,
        n_fft: int = 64,
        hop_length: int = 16,
        log_eps: float = 1e-6,
        mlp_features: Sequence[int] = (384,),
        rotation_offsets: Sequence[int] = (-1, 0, 1),
        pooling: str = "mean",
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
        log_softmax: bool = False,
        activation: type[nn.Module] = nn.ReLU,
        drop_prob: float = 0.0,
        # Standard braindecode args
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        chs_info: list | None = None,
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, sfreq, n_times, input_window_seconds, chs_info

        if num_bands < 1 or electrodes_per_band < 1:
            raise ValueError(
                f"num_bands and electrodes_per_band must be >= 1; got "
                f"num_bands={num_bands}, electrodes_per_band={electrodes_per_band}."
            )
        if self.n_chans != num_bands * electrodes_per_band:
            raise ValueError(
                f"EMG2QwertyNet expects n_chans == num_bands * "
                f"electrodes_per_band ({num_bands} * {electrodes_per_band} "
                f"= {num_bands * electrodes_per_band}); got "
                f"n_chans={self.n_chans}."
            )
        if not mlp_features:
            raise ValueError("mlp_features must contain at least one entry.")
        if not block_channels:
            raise ValueError("block_channels must contain at least one entry.")
        if pooling not in {"max", "mean"}:
            raise ValueError(f"pooling must be 'max' or 'mean'; got {pooling!r}.")

        self.num_bands = num_bands
        self.electrodes_per_band = electrodes_per_band
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.kernel_width = kernel_width
        self.log_softmax = log_softmax

        n_freq_bins = n_fft // 2 + 1
        in_features = electrodes_per_band * n_freq_bins
        num_features = num_bands * mlp_features[-1]

        self.spectrogram = _LogSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            num_bands=num_bands,
            electrodes_per_band=electrodes_per_band,
            log_eps=log_eps,
        )

        # Indices 0/1/3 match upstream's ``TDSConvCTCModule.model``;
        # index 2 is a parameter-free Flatten; upstream's index 4 (head)
        # is broken out as ``self.final_layer`` and remapped via :attr:`mapping`.
        self.model = nn.Sequential(
            _SpectrogramNorm(channels=num_bands * electrodes_per_band),
            _MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=list(mlp_features),
                num_bands=num_bands,
                offsets=rotation_offsets,
                pooling=pooling,
                activation=activation,
            ),
            nn.Flatten(start_dim=2),
            _TDSConvEncoder(
                num_features=num_features,
                block_channels=list(block_channels),
                kernel_width=kernel_width,
                activation=activation,
                drop_prob=drop_prob,
            ),
        )

        self.final_layer = nn.Linear(num_features, self.n_outputs)

        self._n_conv_blocks = sum(
            isinstance(m, _TDSConv2dBlock) for m in self.model[3].tds_conv_blocks
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full pipeline.

        Parameters
        ----------
        x : torch.Tensor
            Raw EMG of shape ``(batch, n_chans=32, n_times)``. ``n_times``
            must be at least the encoder's receptive field, ``n_fft +
            n_conv_blocks * (kernel_width - 1) * hop_length``.

        Returns
        -------
        emissions : torch.Tensor
            Shape ``(batch, T_out, n_outputs)``. Log-probabilities if
            ``log_softmax=True``, otherwise logits.
        """
        if x.ndim != 3 or x.shape[-2] != self.n_chans:
            raise ValueError(
                f"expected (batch, {self.n_chans}, T) input; got shape {x.shape}."
            )
        min_n_times = (
            self.n_fft + self._n_conv_blocks * (self.kernel_width - 1) * self.hop_length
        )
        if x.shape[-1] < min_n_times:
            raise ValueError(
                f"n_times={x.shape[-1]} is shorter than the encoder's "
                f"receptive field ({min_n_times} samples = n_fft + "
                f"n_conv_blocks * (kernel_width - 1) * hop_length, with "
                f"n_fft={self.n_fft}, hop_length={self.hop_length}, "
                f"n_conv_blocks={self._n_conv_blocks}, "
                f"kernel_width={self.kernel_width})."
            )
        spectrogram = self.spectrogram(x)
        encoded = self.model(spectrogram)
        emissions = self.final_layer(encoded)
        if self.log_softmax:
            emissions = F.log_softmax(emissions, dim=-1)
        return emissions.transpose(0, 1).contiguous()

    def reset_head(self, n_outputs: int) -> None:
        """Replace the classification head for a new vocabulary size."""
        if n_outputs <= 0:
            raise ValueError(f"n_outputs must be positive; got {n_outputs}.")
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)
        self._n_outputs = n_outputs

    def compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Map per-sample input lengths to CTC emission lengths.

        ``T_out = (T - n_fft) // hop_length + 1 -
        n_conv_blocks * (kernel_width - 1)``, clamped to zero.
        """
        # ``floor`` (not ``trunc``) keeps ``T < n_fft`` mapping to 0 instead
        # of 1 — matters when ``kernel_width == 1`` (no encoder shrink).
        spec_len = (
            torch.div(
                input_lengths - self.n_fft,
                self.hop_length,
                rounding_mode="floor",
            )
            + 1
        )
        return (spec_len - self._n_conv_blocks * (self.kernel_width - 1)).clamp_min(0)

    def get_output_shape(self) -> tuple[int, ...]:
        """Shape of ``forward`` output for a batch of size 1.

        Overrides the base impl so the dummy ``n_times`` is at least the
        encoder's receptive field (otherwise downstream LayerNorms crash
        on empty inputs). Falls back to that minimum when ``n_times`` was
        not set at construction time.
        """
        min_n_times = (
            self.n_fft + self._n_conv_blocks * (self.kernel_width - 1) * self.hop_length
        )
        try:
            user_n_times = self.n_times
        except ValueError:
            user_n_times = min_n_times
        n_times = max(user_n_times, min_n_times)
        with torch.inference_mode():
            dtype = next(self.parameters()).dtype
            device = next(self.parameters()).device
            dummy_input = torch.zeros(
                1, self.n_chans, n_times, dtype=dtype, device=device
            )
            return tuple(self.forward(dummy_input).shape)


class _LogSpectrogram(nn.Module):
    r"""Per-channel log10-power STFT, reshaped per band.

    ``(batch, num_bands * electrodes_per_band, n_times)`` →
    ``(T_spec, batch, num_bands, electrodes_per_band, freq)`` with
    ``T_spec = (n_times - n_fft) // hop_length + 1`` and
    ``freq = n_fft // 2 + 1``. Hann window is non-persistent (excluded
    from ``state_dict``).
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        num_bands: int,
        electrodes_per_band: int,
        log_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_bands = num_bands
        self.electrodes_per_band = electrodes_per_band
        self.log_eps = log_eps
        # ``periodic=True`` (the torch default) matches upstream emg2qwerty,
        # which builds the window via ``torchaudio.transforms.Spectrogram``
        # with default ``window_fn``. Made explicit so the choice is visible.
        self.register_buffer(
            "window",
            torch.hann_window(n_fft, periodic=True),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_channels, n_samples = x.shape
        if n_channels != self.num_bands * self.electrodes_per_band:
            raise ValueError(
                f"_LogSpectrogram expected "
                f"{self.num_bands * self.electrodes_per_band} channels "
                f"({self.num_bands} bands × {self.electrodes_per_band} "
                f"electrodes); got {n_channels}."
            )
        stft_complex = torch.stft(
            x.reshape(batch_size * n_channels, n_samples),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(device=x.device, dtype=x.dtype),
            normalized=True,
            center=False,
            return_complex=True,
        )
        power_spec = stft_complex.abs().pow(2)
        n_freq_bins, n_frames = power_spec.shape[-2], power_spec.shape[-1]
        log_power = torch.log10(power_spec + self.log_eps).reshape(
            batch_size, n_channels, n_freq_bins, n_frames
        )
        return log_power.reshape(
            batch_size,
            self.num_bands,
            self.electrodes_per_band,
            n_freq_bins,
            n_frames,
        ).movedim(-1, 0)


class _SpectrogramNorm(nn.Module):
    r""":class:`~torch.nn.BatchNorm2d` over (band × electrode) channels.

    Input ``(T, N, bands, electrodes, freq)``; stats computed over
    ``(N, freq, T)`` per ``bands * electrodes`` channel.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_frames, batch_size, n_bands, n_electrodes, n_freq_bins = x.shape
        if self.channels != n_bands * n_electrodes:
            raise ValueError(
                f"_SpectrogramNorm expected {self.channels} flat channels "
                f"({n_bands} bands × {n_electrodes} electrodes); "
                f"got bands*electrodes={n_bands * n_electrodes}."
            )
        flattened = x.movedim(0, -1).reshape(
            batch_size, n_bands * n_electrodes, n_freq_bins, n_frames
        )
        normalized = self.batch_norm(flattened)
        return normalized.reshape(
            batch_size, n_bands, n_electrodes, n_freq_bins, n_frames
        ).movedim(-1, 0)


class _RotationInvariantMLP(nn.Module):
    r"""Single-band rotation-invariant MLP.

    Applies the MLP to each circular roll in ``offsets`` and pools
    (mean/max) across rolls — approximate wristband-rotation invariance.
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if not mlp_features:
            raise ValueError("mlp_features must contain at least one entry.")
        layers: list[nn.Module] = []
        in_dim = in_features
        for out_dim in mlp_features:
            layers.extend([nn.Linear(in_dim, out_dim), activation()])
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        if pooling not in {"max", "mean"}:
            raise ValueError(f"pooling must be 'max' or 'mean'; got {pooling}.")
        self.pooling = pooling
        self.offsets = tuple(offsets) if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rolled = torch.stack(
            [inputs.roll(offset, dims=2) for offset in self.offsets], dim=2
        )
        per_roll_features = self.mlp(rolled.flatten(start_dim=3))
        if self.pooling == "max":
            return per_roll_features.max(dim=2).values
        return per_roll_features.mean(dim=2)


class _MultiBandRotationInvariantMLP(nn.Module):
    r"""Independent rotation-invariant MLP per band.

    ``(T, N, num_bands, electrodes, ...)`` →
    ``(T, N, num_bands, mlp_features[-1])``.
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim
        self.mlps = nn.ModuleList(
            [
                _RotationInvariantMLP(
                    in_features,
                    mlp_features,
                    pooling,
                    offsets,
                    activation=activation,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[self.stack_dim] != self.num_bands:
            raise ValueError(
                f"_MultiBandRotationInvariantMLP expected dim "
                f"{self.stack_dim} == {self.num_bands}; got input shape "
                f"{inputs.shape}."
            )
        per_band_inputs = inputs.unbind(self.stack_dim)
        # Indexed loop (instead of ``zip(self.mlps, per_band_inputs)``) so the
        # forward is TorchScript-scriptable: ``zip`` over a ``ModuleList`` and
        # a dynamic-length ``unbind`` list can't be statically sized.
        band_outputs: list[torch.Tensor] = []
        for band_idx, mlp in enumerate(self.mlps):
            band_outputs.append(mlp(per_band_inputs[band_idx]))
        return torch.stack(band_outputs, dim=self.stack_dim)


class _TDSConv2dBlock(nn.Module):
    r"""Time-Depth-Separable 2-D conv block ([hannun2019tds]_).

    1×``kernel_width`` conv + activation + residual + LayerNorm. No
    temporal padding: strips ``kernel_width - 1`` frames per block.
    """

    def __init__(
        self,
        channels: int,
        width: int,
        kernel_width: int,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        n_in_frames, batch_size, n_features = inputs.shape
        folded = inputs.movedim(0, -1).reshape(
            batch_size, self.channels, self.width, n_in_frames
        )
        conv_out = self.activation(self.conv2d(folded))
        conv_out = conv_out.reshape(batch_size, n_features, -1).movedim(-1, 0)
        n_out_frames = conv_out.shape[0]
        # Output frame i aligns with input frame i+(kernel_width-1), so
        # the residual is the LAST n_out_frames input frames.
        residual_added = conv_out + inputs[-n_out_frames:]
        return self.layer_norm(residual_added)


class _TDSFullyConnectedBlock(nn.Module):
    r"""Two-layer FFN with residual skip, LayerNorm, and optional dropout."""

    def __init__(
        self,
        num_features: int,
        activation: type[nn.Module] = nn.ReLU,
        drop_prob: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(num_features, num_features),
            activation(),
        ]
        if drop_prob > 0:
            layers.append(nn.Dropout(drop_prob))
        layers.append(nn.Linear(num_features, num_features))
        if drop_prob > 0:
            layers.append(nn.Dropout(drop_prob))
        self.fc_block = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.fc_block(inputs) + inputs)


class _TDSConvEncoder(nn.Module):
    r"""``_TDSConv2dBlock`` + ``_TDSFullyConnectedBlock`` stack.

    With ``K = len(block_channels)`` blocks and no temporal padding, the
    encoder strips ``K * (kernel_width - 1)`` frames in total. Each ``ch``
    must evenly divide ``num_features`` so the conv block can fold features
    as ``(channels=ch, width=num_features // ch)``.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
        activation: type[nn.Module] = nn.ReLU,
        drop_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if not block_channels:
            raise ValueError("block_channels must be non-empty.")
        blocks: list[nn.Module] = []
        for n_block_channels in block_channels:
            if num_features % n_block_channels != 0:
                raise ValueError(
                    f"block_channels entry {n_block_channels} does not evenly "
                    f"divide num_features={num_features}."
                )
            blocks.extend(
                [
                    _TDSConv2dBlock(
                        n_block_channels,
                        num_features // n_block_channels,
                        kernel_width,
                        activation=activation,
                    ),
                    _TDSFullyConnectedBlock(
                        num_features,
                        activation=activation,
                        drop_prob=drop_prob,
                    ),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)
