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
import torchaudio.transforms as ta_transforms
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
    for emission lengths. Pass ``return_features=True`` to return the
    pre-classifier encoder representation as a
    ``{"features": (batch, T_out, num_features), "cls_token": None}``
    dict, matching the BIOT / signal-JEPA convention used by downstream
    wrappers (e.g. neuroai's ``DownstreamWrapperModel``).

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
      the log-spectrogram. SpecAugment is built into the model
      (``spec_augment=True``) and only fires in training mode; the
      time/frequency-jitter pieces are dataset-side augmentations.
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
    spec_augment : bool
        If ``True``, apply SpecAugment [park2019specaug]_ time/frequency
        masking on the log-spectrogram during training only. Disabled in
        ``eval`` mode and absent from the parameter / state-dict count.
        Defaults to ``False``; set to ``True`` to match the upstream
        emg2qwerty paper recipe.
    n_time_masks : int
        Maximum number of time masks applied per call. Each forward pass
        samples a uniform integer in ``[0, n_time_masks]``. Defaults to
        ``3`` (Sivakumar et al. Sec 5.2).
    time_mask_param : int
        Maximum time-mask width in spectrogram frames. Defaults to ``25``.
    n_freq_masks : int
        Maximum number of frequency masks applied per call. Each forward
        pass samples a uniform integer in ``[0, n_freq_masks]``. Defaults
        to ``2``.
    freq_mask_param : int
        Maximum frequency-mask width in STFT bins. Defaults to ``4``.
    spec_augment_prob : float
        Probability of running SpecAugment on a given training batch
        (Bernoulli gate before sampling mask counts). Defaults to ``1.0``.

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
        spec_augment: bool = False,
        n_time_masks: int = 3,
        time_mask_param: int = 25,
        n_freq_masks: int = 2,
        freq_mask_param: int = 4,
        spec_augment_prob: float = 1.0,
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

        # Built-in SpecAugment lives between the spectrogram and the BatchNorm
        # so it operates on the log-power tensor (matches upstream emg2qwerty
        # and the previous neuralbench callback). ``nn.Identity`` keeps the
        # forward path symmetrical without contributing parameters or
        # state-dict keys when SpecAugment is disabled.
        self.spec_augment: nn.Module
        if spec_augment:
            self.spec_augment = _SpecAugment(
                n_time_masks=n_time_masks,
                time_mask_param=time_mask_param,
                n_freq_masks=n_freq_masks,
                freq_mask_param=freq_mask_param,
                prob=spec_augment_prob,
            )
        else:
            self.spec_augment = nn.Identity()

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

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Run the full pipeline.

        Parameters
        ----------
        x : torch.Tensor
            Raw EMG of shape ``(batch, n_chans=32, n_times)``. ``n_times``
            must be at least the encoder's receptive field, ``n_fft +
            n_conv_blocks * (kernel_width - 1) * hop_length``.
        return_features : bool
            If ``True``, return a ``dict`` with the encoder representation
            instead of the classification emissions. The encoder is the
            full TDS-Conv stack up to (but not including)
            ``self.final_layer`` — i.e. what downstream wrappers want
            when they apply their own probe/aggregation. Matches the
            BIOT / signal-JEPA convention so the same neuroai
            ``DownstreamWrapperModel(model_output_key="features")``
            can consume it.

        Returns
        -------
        torch.Tensor or dict
            Default (``return_features=False``): ``torch.Tensor`` of
            shape ``(batch, T_out, n_outputs)``. Log-probabilities if
            ``log_softmax=True``, otherwise logits.

            If ``return_features=True``: ``dict`` with
            ``"features"`` (shape ``(batch, T_out, num_features)``,
            where ``num_features = num_bands * mlp_features[-1]``) and
            ``"cls_token"`` (always ``None`` — TDS-Conv has no
            ``[CLS]``).
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
        spectrogram = self.spec_augment(spectrogram)
        encoded = self.model(spectrogram)
        if return_features:
            # ``encoded`` is (T_out, B, num_features). Transpose to
            # (B, T_out, num_features) so feature consumers see the
            # same batch-first layout as the default emissions tensor.
            return {
                "features": encoded.transpose(0, 1).contiguous(),
                "cls_token": None,
            }
        emissions = self.final_layer(encoded)
        if self.log_softmax:
            emissions = F.log_softmax(emissions, dim=-1)
        return emissions.transpose(0, 1).contiguous()

    def reset_head(self, n_outputs: int) -> None:
        """Replace the classification head for a new vocabulary size.

        The replacement :class:`~torch.nn.Linear` inherits the existing
        head's dtype and device so a subsequent ``forward()`` does not
        crash after ``model.double()`` or ``model.to(device)``. The
        captured init config (``get_config()``) is also kept in sync so
        save/load round-trips rebuild the new head.
        """
        if n_outputs <= 0:
            raise ValueError(f"n_outputs must be positive; got {n_outputs}.")
        old_head = self.final_layer
        self.final_layer = nn.Linear(old_head.in_features, n_outputs).to(
            device=old_head.weight.device, dtype=old_head.weight.dtype
        )
        self._n_outputs = n_outputs
        # Keep the init-kwargs snapshot used by ``get_config()`` aligned with
        # the live head, so ``EMG2QwertyNet.from_config(m.get_config())``
        # rebuilds the head with the new vocab size.
        init_kwargs = getattr(self, "_braindecode_init_kwargs", None)
        if init_kwargs is not None and "n_outputs" in init_kwargs:
            init_kwargs["n_outputs"] = n_outputs
        hub_config = getattr(self, "_hub_mixin_config", None)
        if hub_config is not None and "n_outputs" in hub_config:
            hub_config["n_outputs"] = n_outputs

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

        Uses the user-supplied ``n_times`` so this method's reported
        shape is consistent with what ``forward()`` accepts. If the
        configured ``n_times`` is below the encoder's receptive field,
        ``forward()`` would raise; we mirror that here. Falls back to
        the receptive-field minimum only when ``n_times`` was not set
        at construction time.
        """
        min_n_times = (
            self.n_fft + self._n_conv_blocks * (self.kernel_width - 1) * self.hop_length
        )
        try:
            n_times = self.n_times
        except ValueError:
            n_times = min_n_times
        if n_times < min_n_times:
            raise ValueError(
                f"n_times={n_times} is shorter than the encoder's receptive "
                f"field ({min_n_times} samples); ``forward()`` would reject "
                f"this input. Increase n_times or override n_fft / hop_length "
                f"/ kernel_width."
            )
        with torch.inference_mode():
            dtype = next(self.parameters()).dtype
            device = next(self.parameters()).device
            dummy_input = torch.zeros(
                1, self.n_chans, n_times, dtype=dtype, device=device
            )
            # ``return_features=False`` is the default; assert here so mypy
            # narrows the union return type to ``Tensor`` for ``.shape``.
            emissions = self.forward(dummy_input, return_features=False)
            assert isinstance(emissions, torch.Tensor)
            return tuple(emissions.shape)


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
        if log_eps <= 0:
            raise ValueError(f"log_eps must be > 0; got {log_eps}.")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_bands = num_bands
        self.electrodes_per_band = electrodes_per_band
        self.log_eps = log_eps
        # Use ``torchaudio.transforms.Spectrogram`` directly (instead of
        # ``torch.stft`` + ``abs().pow(2)``) so that ``normalized=True``
        # divides by ``sum(window**2)`` as upstream emg2qwerty does, rather
        # than by ``n_fft`` (which is what ``torch.stft(normalized=True)``
        # would do — silently rescales every power bin and shifts the log
        # by a constant, breaking upstream-checkpoint numerics).
        self.spectrogram = ta_transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            normalized=True,
            center=False,
        )
        # Re-register the Hann window as non-persistent so it doesn't pollute
        # ``state_dict`` (upstream emg2qwerty checkpoints don't ship this
        # buffer either: their spectrogram lives in the dataset transform,
        # not in the model).
        self.spectrogram.register_buffer(
            "window", self.spectrogram.window, persistent=False
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
        # ``torchaudio.transforms.Spectrogram`` returns ``(..., freq, time)``
        # for power=2 (the default).
        power_spec = self.spectrogram(x)
        n_freq_bins, n_frames = power_spec.shape[-2], power_spec.shape[-1]
        log_power = torch.log10(power_spec + self.log_eps)
        return log_power.reshape(
            batch_size,
            self.num_bands,
            self.electrodes_per_band,
            n_freq_bins,
            n_frames,
        ).movedim(-1, 0)


class _SpecAugment(nn.Module):
    r"""SpecAugment masking on the log-spectrogram during training.

    Applies up to ``n_time_masks`` × ``time_mask_param``-frame time bands
    and ``n_freq_masks`` × ``freq_mask_param``-bin frequency bands, with
    one mask drawn per ``(sample × band)`` row and broadcast across all
    electrodes within that band — so a wristband's 16 electrodes share
    the same masked window. No-op outside ``training``. Defaults match
    Sivakumar et al. Sec 5.2 (3×25 / 2×4 frames-bins, ``prob=1.0``).

    Implemented as pure tensor ops (per-row vectorised start / width
    sampling, on-device mean fill, ``masked_fill`` broadcast) so the
    forward pass keeps no host round-trips on GPU. ``torchaudio``'s
    ``TimeMasking(iid_masks=True)`` cannot be reused here because, on a
    ``(B*num_bands, electrodes, freq, T)`` tensor, it samples one mask
    per ``(batch, electrode)`` pair instead of per ``(sample × band)``.
    """

    def __init__(
        self,
        n_time_masks: int = 3,
        time_mask_param: int = 25,
        n_freq_masks: int = 2,
        freq_mask_param: int = 4,
        prob: float = 1.0,
    ) -> None:
        super().__init__()
        if n_time_masks < 0 or n_freq_masks < 0:
            raise ValueError(
                f"n_time_masks and n_freq_masks must be >= 0; got "
                f"n_time_masks={n_time_masks}, n_freq_masks={n_freq_masks}."
            )
        if time_mask_param < 0 or freq_mask_param < 0:
            raise ValueError(
                f"time_mask_param and freq_mask_param must be >= 0; got "
                f"time_mask_param={time_mask_param}, "
                f"freq_mask_param={freq_mask_param}."
            )
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1]; got {prob}.")
        self.n_time_masks = n_time_masks
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.freq_mask_param = freq_mask_param
        self.prob = prob

    @staticmethod
    def _band_shared_axis_mask(
        n_rows: int,
        axis_len: int,
        max_mask_param: int,
        device: torch.device,
    ) -> torch.Tensor:
        """One mask per row, shared across the orthogonal axes.

        Returns a ``(n_rows, axis_len)`` boolean tensor where row ``i``
        marks a contiguous span ``[start_i, start_i + width_i)`` to be
        masked. ``width_i`` is sampled in ``[0, min(max_mask_param,
        axis_len)]``; ``start_i`` in ``[0, axis_len - width_i]``. Both
        draws stay on-device.
        """
        effective = min(max_mask_param, axis_len)
        if effective <= 0 or n_rows <= 0:
            return torch.zeros(n_rows, axis_len, dtype=torch.bool, device=device)
        widths = torch.randint(0, effective + 1, (n_rows,), device=device)
        # Float-then-cast keeps integer precision under fp16/bf16 — same trick
        # ``_MaskAug`` uses in ``meta_neuromotor``.
        max_starts = (axis_len - widths).to(torch.float32)
        starts = (
            torch.rand(n_rows, device=device, dtype=torch.float32) * max_starts
        ).to(torch.long)
        positions = torch.arange(axis_len, device=device)
        return (positions[None, :] >= starts[:, None]) & (
            positions[None, :] < (starts + widths)[:, None]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x``: (T_spec, B, num_bands, electrodes, freq).
        if (
            not self.training
            or self.prob <= 0.0
            or (self.n_time_masks == 0 and self.n_freq_masks == 0)
        ):
            return x
        if self.prob < 1.0 and torch.rand((), device="cpu").item() >= self.prob:
            return x
        n_frames, batch_size, n_bands, n_electrodes, n_freq_bins = x.shape
        n_rows = batch_size * n_bands
        # Reshape to ``(B*num_bands, electrodes, freq, T_spec)`` so a mask of
        # shape ``(n_rows, ..., L)`` broadcasts across the electrode axis —
        # the missing dim is 1, so every electrode in a band sees the same
        # masked window.
        flat = x.movedim(0, -1).reshape(n_rows, n_electrodes, n_freq_bins, n_frames)
        # 0-D on-device fill value; in log-spec space the natural zero is
        # ``log(power=1)`` which sits well above the typical distribution and
        # would inject artificial spikes. Mean over all elements matches the
        # original neuralbench callback behaviour but stays on-device (no
        # ``.item()`` host sync per forward).
        mask_value = flat.mean()
        n_t = int(torch.randint(self.n_time_masks + 1, ()).item())
        for _ in range(n_t):
            time_mask = self._band_shared_axis_mask(
                n_rows, n_frames, self.time_mask_param, flat.device
            )
            flat = flat.masked_fill(time_mask[:, None, None, :], mask_value)
        n_f = int(torch.randint(self.n_freq_masks + 1, ()).item())
        for _ in range(n_f):
            freq_mask = self._band_shared_axis_mask(
                n_rows, n_freq_bins, self.freq_mask_param, flat.device
            )
            flat = flat.masked_fill(freq_mask[:, None, :, None], mask_value)
        return flat.reshape(
            batch_size, n_bands, n_electrodes, n_freq_bins, n_frames
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
