# Authors: Meta Platforms, Inc. and affiliates (original emg2qwerty)
#          Bruno Aristimunha <b.aristimunha@gmail.com> (Braindecode adaptation)
#
# License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
#
# Architecture adapted from the official implementation at
# https://github.com/facebookresearch/emg2qwerty (NeurIPS 2024)
# released under CC BY-NC 4.0. This derivative inherits the same
# noncommercial terms and is not covered by braindecode's BSD-3 license!
"""``EMG2QwertyNet``: TDS-Conv-CTC for sEMG-to-keystroke decoding.

Single self-contained module: the model class plus its private layer
helpers. Mirrors the file layout of
:mod:`braindecode.models.meta_neuromotor` (one file under CC BY-NC,
private layer helpers prefixed with ``_``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from braindecode.models.base import EEGModuleMixin

# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class EMG2QwertyNet(EEGModuleMixin, nn.Module):
    r"""TDS-Conv-CTC sEMG-to-keystroke decoder (emg2qwerty) [emg2qwerty2024]_.

    :bdg-success:`Convolution` :bdg-info:`CTC`

    .. figure:: https://user-images.githubusercontent.com/172884/131012947-66cab4c4-963c-4f1a-af12-47fea1681f09.png
        :align: center
        :alt: emg2qwerty dataset and TDS-Conv-CTC pipeline.
        :width: 500px

        Touch-typing on a QWERTY keyboard while wearing two 16-electrode
        sEMG wristbands (left + right). The TDS-Conv-CTC baseline maps
        the dual-armband EMG stream into per-frame character probabilities
        trained with Connectionist Temporal Classification [graves2006ctc]_.

    Time-Depth-Separable convolutional encoder with a CTC head, ported from
    `facebookresearch/emg2qwerty
    <https://github.com/facebookresearch/emg2qwerty>`_ [emg2qwerty2024]_.
    Takes raw 32-channel sEMG (two wristbands × 16 electrodes) at 2 kHz and
    emits a per-token score sequence over the 99-class typing vocabulary
    (98 keys + CTC blank).

    .. rubric:: Macro Components

    The forward pass has five stages:

    1. Log-spectrogram front-end (``self.spectrogram``, no trainable
       parameters). Channel-wise STFT with ``n_fft=64`` (32 ms at 2 kHz),
       ``hop_length=16`` (8 ms), Hann window, ``normalized=True``,
       ``center=False`` (strict causality). Power is log10-transformed
       (``log10(p + 1e-6)``) and the channel axis is split into
       ``(num_bands=2, electrodes=16)``. Output shape ``(T_spec, batch,
       2, 16, freq=33)`` at 125 Hz frame rate.

    2. ``_SpectrogramNorm`` (``self.model[0]``): per-electrode-per-band
       :class:`~torch.nn.BatchNorm2d` over the ``(batch, freq, time)`` slice
       for each of the ``2 × 16 = 32`` channels.

    3. ``_MultiBandRotationInvariantMLP`` (``self.model[1]``): per-band
       rotation-invariant MLP. Each 16-electrode band is circularly rolled
       by the fixed offsets ``(-1, 0, +1)``, passed through a shared MLP,
       and mean-pooled across rolls. Approximates rigid-rotation
       invariance of the wristband.

    4. ``_TDSConvEncoder`` (``self.model[3]``, after ``nn.Flatten`` at
       ``self.model[2]``): stack of ``_TDSConv2dBlock`` interleaved
       with ``_TDSFullyConnectedBlock``. Each conv block uses
       ``1 × kernel_width`` 2-D convolution with no temporal padding,
       so it strips ``kernel_width - 1`` frames from the start of the
       sequence per block. With the default
       ``block_channels=(24, 24, 24, 24)`` and ``kernel_width=32``, the
       encoder removes ``4 × 31 = 124`` frames in total.

    5. :class:`~torch.nn.Linear` classification head exposed as the
       top-level submodule ``self.final_layer`` (so it shows up in
       :func:`~torch.nn.Module.named_modules` and ``print(model)``;
       :meth:`reset_head` swaps it). Optionally followed by
       :func:`~torch.nn.functional.log_softmax` (``log_softmax``
       constructor flag, disabled by default since braindecode models
       conventionally return logits).

    .. rubric:: State-dict layout and upstream-checkpoint compatibility

    ``self.model`` is an ``nn.Sequential`` whose children sit at the same
    integer indices as upstream emg2qwerty's ``TDSConvCTCModule.model``
    for the encoder portion (``0/1/3``); the classification head is
    broken out as ``self.final_layer`` so it shows up in
    :func:`~torch.nn.Module.named_modules` and ``print(model)``. Two
    explicit entries in :attr:`mapping` translate upstream's
    ``model.4.{weight,bias}`` into our ``final_layer.{weight,bias}``.
    Stripping the PyTorch-Lightning ``network.`` prefix from an upstream
    checkpoint is the only manual step:

    .. code-block::

        sd = {k[len("network."):]: v
              for k, v in ckpt["state_dict"].items()
              if k.startswith("network.")}
        model.load_state_dict(sd, strict=True)

    The log-spectrogram front-end (``self.spectrogram``) sits outside
    ``self.model`` and contributes zero parameters; its STFT window is
    a non-persistent buffer that does not appear in ``state_dict``.

    .. rubric:: Output shape and CTC usage

    The forward pass returns ``(batch, T_out, n_outputs)``, the layout
    that :class:`~torch.nn.CTCLoss` expects after a transpose. ``T_out``
    is the downsampled emission length recoverable from the input length
    via :meth:`compute_output_lengths`. For ``CTCLoss``, transpose the
    time dimension first: ``emissions.transpose(0, 1)``. With the default
    config and ``n_times=8000`` (4 s @ 2 kHz), ``T_out=373``.

    .. rubric:: Training recipe (paper values)

    - **Loss**: plain :class:`~torch.nn.CTCLoss` on log-softmax of the
      model's emissions (no FastEmit, no entropy regularization).
    - **Vocabulary**: 98 keys + 1 blank → ``n_outputs=99``. Lowercase
      ``[a-z]``, uppercase ``[A-Z]``, digits ``[0-9]``, ASCII punctuation,
      and four modifiers (``Key.{backspace,enter,space,shift}``).
    - **Optimizer**: :class:`~torch.optim.Adam`, ``lr=1e-3``,
      ``weight_decay=0`` (plain Adam, **not** AdamW).
    - **Schedule**: linear warmup from ``lr=1e-8`` over the first 10 epochs
      (essentially zero), then cosine annealing to ``eta_min=1e-6``,
      150 epochs total. The slow warmup is structural. Without it, CTC
      collapses to all-blank prediction within one epoch on small data,
      because predicting blank everywhere is a trivial local minimum.
    - **Augmentation**: random circular per-band channel rotations
      ``{-1, 0, +1}`` plus per-band temporal jitter up to ±60 samples
      (30 ms), and SpecAugment on the log-spectrogram
      (``n_time_masks=3``, ``time_mask_param=25``, ``n_freq_masks=2``,
      ``freq_mask_param=4``).
    - **Decoding**: greedy CTC by default; the upstream paper also
      reports a beam decoder with a 6-gram character-level KenLM language
      model (not ported here).

    .. warning::
        The rotation-invariant MLP assumes circular adjacency of the 16
        electrodes within each band (the wristband geometry of the paper).
        For arbitrary EEG montages the rotation invariance is not
        meaningful and this model should not be used as-is.

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
        Number of EMG channels. Must equal ``32`` (2 bands × 16
        electrodes).
    sfreq : float
        Sampling frequency in Hz. Defaults to ``2000``. ``n_fft`` and
        ``hop_length`` defaults are calibrated for this rate; passing a
        different ``sfreq`` without overriding the STFT params raises
        :class:`ValueError` to avoid silent time/frequency rescaling.
    n_fft : int
        STFT window size in samples.
    hop_length : int
        STFT hop in samples.
    mlp_features : sequence of int
        Hidden sizes of the rotation-invariant MLP. Output dim per band
        is ``mlp_features[-1]``.
    block_channels : sequence of int
        Channel count per ``_TDSConv2dBlock``. The model's internal
        ``num_features = 2 * mlp_features[-1]`` must be evenly divisible
        by each entry.
    kernel_width : int
        Temporal kernel size of each ``_TDSConv2dBlock``.
    log_softmax : bool
        If ``True``, apply :func:`~torch.nn.functional.log_softmax` to the
        emissions. Disabled by default (braindecode models return logits).
    activation : type of nn.Module
        Activation class used inside the rotation-invariant MLP and the
        TDS conv/FC blocks. Defaults to :class:`~torch.nn.ReLU` (matches
        upstream emg2qwerty). Pass any non-parametrized activation class
        (``nn.GELU``, ``nn.SiLU``, …) for ablations.
    drop_prob : float
        Dropout probability applied inside each ``_TDSFullyConnectedBlock``
        between the two ``Linear`` layers. Default ``0.0`` matches the
        upstream paper recipe (no dropout). Set ``> 0`` for regularized
        training.

    Examples
    --------
    Train-time forward pass and CTC loss::

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from braindecode.models import EMG2QwertyNet

        model = EMG2QwertyNet(
            n_outputs=99, n_chans=32, n_times=8000, sfreq=2000,
        )
        x = torch.randn(2, 32, 8000)              # (batch, n_chans, n_times)
        emissions = model(x)                       # (batch, T_out, 99)
        log_probs = F.log_softmax(emissions, dim=-1).transpose(0, 1)
        emit_lens = model.compute_output_lengths(torch.tensor([8000, 8000]))
        targets = torch.randint(0, 98, (2, 20), dtype=torch.long)
        target_lengths = torch.tensor([20, 15], dtype=torch.int32)
        loss = nn.CTCLoss(blank=98, zero_infinity=True)(
            log_probs, targets, emit_lens, target_lengths,
        )

    Load an upstream emg2qwerty checkpoint::

        ckpt = torch.load("personalized.ckpt", weights_only=False)
        # PyTorch Lightning prefixes ``TDSConvCTCModule.model`` keys with
        # "network.". Strip that prefix so upstream keys (``model.0.*``,
        # ``model.1.*``, ``model.3.*``, ``model.4.{weight,bias}``) land
        # where :attr:`mapping` expects them.
        sd = {
            k[len("network."):]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("network.")
        }
        model.load_state_dict(sd, strict=True)

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
    .. [fastemit2021] Yu, J. et al., 2021. FastEmit: low-latency
        streaming ASR with sequence-level emission regularization.
        Proc. ICASSP.
    """

    #: State-dict key remapping for upstream emg2qwerty checkpoints.
    #: Upstream's ``TDSConvCTCModule.model`` is an ``nn.Sequential`` whose
    #: head sits at index ``4``; here the head is broken out as
    #: ``self.final_layer`` so it shows up in :func:`~torch.nn.Module.named_modules`
    #: and ``print(model)``. The two entries below remap the head; the
    #: encoder keys (``model.0.*``, ``model.1.*``, ``model.3.*``) match
    #: upstream verbatim.
    mapping = {
        "model.4.weight": "final_layer.weight",
        "model.4.bias": "final_layer.bias",
    }

    def __init__(
        self,
        n_outputs: int = 99,
        n_chans: int = 32,
        sfreq: float = 2000.0,
        n_fft: int = 64,
        hop_length: int = 16,
        mlp_features: Sequence[int] = (384,),
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
        # Free the keyword-arg names so subsequent code reads them only via
        # the EEGModuleMixin properties (``self.n_outputs`` etc.). Mirrors
        # the pattern in ``MetaNeuromotorHand``.
        del n_outputs, n_chans, sfreq, n_times, input_window_seconds, chs_info

        # --- Validate inputs ---------------------------------------------
        # Architectural fixed values: 2 wristbands of 16 electrodes each.
        if self.n_chans != 32:
            raise ValueError(
                f"EMG2QwertyNet expects 32 channels (2 bands × 16 "
                f"electrodes); got {self.n_chans}."
            )
        # Default STFT params (n_fft=64 / hop=16) are calibrated for
        # sfreq=2000 (32 ms / 8 ms windows). Refuse silently-different
        # time semantics.
        if self.sfreq != 2000.0 and (n_fft, hop_length) == (64, 16):
            raise ValueError(
                f"EMG2QwertyNet's default n_fft=64, hop_length=16 are "
                f"calibrated for sfreq=2000 Hz. Got sfreq={self.sfreq}. "
                f"Pass sfreq=2000 or override n_fft / hop_length to match."
            )
        if not mlp_features:
            raise ValueError("mlp_features must contain at least one entry.")
        if not block_channels:
            raise ValueError("block_channels must contain at least one entry.")

        # --- Stash hyperparameters used outside ``__init__`` --------------
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.kernel_width = kernel_width
        self.log_softmax = log_softmax

        n_freq_bins = n_fft // 2 + 1
        # _NUM_BANDS=2, _ELECTRODES_PER_BAND=16 are inlined as literals
        # below: those numbers are fixed by emg2qwerty's two-wristband
        # architecture and validated against ``self.n_chans`` above.
        in_features = 16 * n_freq_bins  # electrodes × freq
        num_features = 2 * mlp_features[-1]  # bands × mlp_out

        # --- Build submodules --------------------------------------------
        # Log-spectrogram front-end. Kept SEPARATE from ``self.model`` so
        # the inner Sequential's children sit at the same integer indices
        # as upstream emg2qwerty's encoder (which computes the spectrogram
        # in its dataset transform, not in the model). Hann window is a
        # non-persistent buffer — contributes zero keys to ``state_dict``.
        self.spectrogram = _LogSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            num_bands=2,
            electrodes_per_band=16,
        )

        # Encoder ``nn.Sequential``. The classification head is broken
        # out as ``self.final_layer`` (below) so ``print(model)`` and
        # ``named_modules`` show it. The encoder children index-match
        # upstream emg2qwerty's ``TDSConvCTCModule.model`` for indices
        # 0/1/3 (index 2 is a parameter-free Flatten); upstream's index
        # 4 is the final Linear, remapped to ``final_layer`` via
        # :attr:`mapping`.
        self.model = nn.Sequential(
            _SpectrogramNorm(channels=2 * 16),  # 0
            _MultiBandRotationInvariantMLP(  # 1
                in_features=in_features,
                mlp_features=list(mlp_features),
                num_bands=2,
                activation=activation,
            ),
            nn.Flatten(start_dim=2),  # 2
            _TDSConvEncoder(  # 3
                num_features=num_features,
                block_channels=list(block_channels),
                kernel_width=kernel_width,
                activation=activation,
                drop_prob=drop_prob,
            ),
        )

        # Classification head as a top-level submodule. Direct assignment
        # in :meth:`reset_head` works because there is no @property
        # interfering with ``nn.Module.__setattr__``'s Module-aware path.
        self.final_layer = nn.Linear(num_features, self.n_outputs)

        # Cache the conv-block count so ``compute_output_lengths`` can't
        # silently desync if a future ``_TDSConvEncoder`` variant adds
        # other block types. ``self.model[3]`` is the encoder per the
        # layout above.
        self._n_conv_blocks = sum(
            isinstance(m, _TDSConv2dBlock) for m in self.model[3].tds_conv_blocks
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full pipeline.

        Parameters
        ----------
        x : torch.Tensor
            Raw EMG of shape ``(batch, n_chans=32, n_times)``.
            ``n_times`` must be at least the encoder's full receptive
            field, ``n_fft + n_conv_blocks * (kernel_width - 1) *
            hop_length``. Shorter inputs are rejected upfront with a
            clear ``ValueError`` so we never crash deep inside
            :func:`torch.stft` or :class:`~torch.nn.Conv2d`.

        Returns
        -------
        emissions : torch.Tensor
            Shape ``(batch, T_out, n_outputs)``. Log-probabilities if
            ``log_softmax=True``, otherwise logits.
        """
        if x.ndim != 3 or x.shape[-2] != self.n_chans:
            raise ValueError(
                f"expected (batch, {self.n_chans}, T) input; got shape "
                f"{tuple(x.shape)}."
            )
        # Tightest forward precondition: enforce the full encoder
        # receptive field, not just ``n_fft``. With only the n_fft check
        # ``torch.stft`` succeeds for ``n_times`` in
        # ``[n_fft, min_required)`` and the model crashes deeper inside
        # the no-padding ``Conv2d`` of ``_TDSConv2dBlock`` (kernel size
        # exceeds available frames). Raise upfront with the same formula
        # used by :meth:`compute_output_lengths` and
        # :meth:`get_output_shape`.
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
                f"kernel_width={self.kernel_width}). Use a longer window "
                f"or override n_fft / hop_length / kernel_width."
            )
        h = self.spectrogram(x)  # (T, N, 2, 16, freq)
        h = self.model(h)  # (T_out, N, num_features)
        h = self.final_layer(h)  # (T_out, N, n_outputs)
        if self.log_softmax:
            h = F.log_softmax(h, dim=-1)
        return h.transpose(0, 1).contiguous()  # (N, T_out, n_outputs)

    def reset_head(self, n_outputs: int) -> None:
        """Replace the classification head for a new vocabulary size."""
        if n_outputs <= 0:
            raise ValueError(f"n_outputs must be positive; got {n_outputs}.")
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)
        self._n_outputs = n_outputs

    def compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Map per-sample input lengths to CTC emission lengths.

        ``T_out = (T - n_fft) // hop_length + 1 -
        n_conv_blocks * (kernel_width - 1)``, clamped to zero. Inputs with
        ``T < n_fft`` produce zero-length emissions; :meth:`forward`
        rejects those inputs upfront so the two methods agree on what's
        a usable shape.
        """
        spec_len = (
            torch.div(
                input_lengths - self.n_fft,
                self.hop_length,
                rounding_mode="trunc",
            )
            + 1
        )
        return (spec_len - self._n_conv_blocks * (self.kernel_width - 1)).clamp_min(0)

    def get_output_shape(self) -> tuple[int, ...]:
        """Shape of ``forward`` output for a batch of size 1.

        Overrides the base implementation so the dummy ``n_times`` is
        always at least the encoder's full receptive field; otherwise
        ``forward`` produces zero-length emissions and downstream layer
        norms crash on empty inputs. If ``n_times`` was not specified at
        construction time, the receptive-field minimum is used as a
        fallback (``EEGModuleMixin.n_times`` would otherwise raise).
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
            x = torch.zeros(1, self.n_chans, n_times, dtype=dtype, device=device)
            return tuple(self.forward(x).shape)


# ---------------------------------------------------------------------------
# Private helpers — model layers
# ---------------------------------------------------------------------------


class _LogSpectrogram(nn.Module):
    r"""Per-channel log10 power STFT, reshaped per band.

    Turns raw multi-channel EMG into the log-spectrogram layout expected
    by :class:`_SpectrogramNorm` and the rotation-invariant MLP downstream.

    Input  : ``(batch, num_bands * electrodes_per_band, n_times)``.
    Output : ``(T_spec, batch, num_bands, electrodes_per_band, freq)``
    where ``T_spec = (n_times - n_fft) // hop_length + 1`` and
    ``freq = n_fft // 2 + 1``.

    Computes :func:`torch.stft` per channel with a Hann window
    (``normalized=True``, ``center=False``), takes squared magnitude,
    log10-transforms with a small floor of ``1e-6``, and splits the
    flattened channel axis into ``(num_bands, electrodes_per_band)``
    before moving time to the leading axis.

    The Hann window is registered as a non-persistent buffer
    (``persistent=False``): it moves with ``.to(device)`` but does not
    appear in :meth:`~torch.nn.Module.state_dict`.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        num_bands: int,
        electrodes_per_band: int,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_bands = num_bands
        self.electrodes_per_band = electrodes_per_band
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T = x.shape
        if C != self.num_bands * self.electrodes_per_band:
            raise ValueError(
                f"_LogSpectrogram expected "
                f"{self.num_bands * self.electrodes_per_band} channels "
                f"({self.num_bands} bands × {self.electrodes_per_band} "
                f"electrodes); got {C}."
            )
        # Compute STFT per channel: flatten (N, C) into a single batch.
        spec = torch.stft(
            x.reshape(N * C, T),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(x.dtype),
            normalized=True,
            center=False,
            return_complex=True,
        )  # (N*C, freq, T_spec)
        power = spec.abs().pow(2)  # squared magnitude
        n_freq, T_spec = power.shape[-2], power.shape[-1]
        logspec = torch.log10(power + 1e-6).reshape(N, C, n_freq, T_spec)
        # Split channels into (bands, electrodes); move time to leading axis.
        return logspec.reshape(
            N, self.num_bands, self.electrodes_per_band, n_freq, T_spec
        ).movedim(-1, 0)


class _SpectrogramNorm(nn.Module):
    r""":class:`~torch.nn.BatchNorm2d` over (band × electrode) channels.

    Input shape ``(T, N, num_bands, electrodes, freq)``. Stats are
    computed over the ``(N, freq, T)`` slice for each of the
    ``num_bands * electrodes`` channels independently.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = x.shape
        if self.channels != bands * C:
            raise ValueError(
                f"_SpectrogramNorm expected {self.channels} flat channels "
                f"({bands} bands × {C} electrodes); got bands*C={bands * C}."
            )
        x = x.movedim(0, -1).reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        return x.reshape(N, bands, C, freq, T).movedim(-1, 0)


class _RotationInvariantMLP(nn.Module):
    r"""Single-band rotation-invariant MLP.

    Applies the MLP to each of ``len(offsets)`` circular rolls of the
    electrode axis, then mean- or max-pools across rolls. Enforces
    approximate rigid-rotation invariance of the wristband.
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
        in_f = in_features
        for out_f in mlp_features:
            layers.extend([nn.Linear(in_f, out_f), activation()])
            in_f = out_f
        self.mlp = nn.Sequential(*layers)
        if pooling not in {"max", "mean"}:
            raise ValueError(f"pooling must be 'max' or 'mean'; got {pooling}.")
        self.pooling = pooling
        self.offsets = tuple(offsets) if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = torch.stack([inputs.roll(o, dims=2) for o in self.offsets], dim=2)
        x = self.mlp(x.flatten(start_dim=3))
        return x.max(dim=2).values if self.pooling == "max" else x.mean(dim=2)


class _MultiBandRotationInvariantMLP(nn.Module):
    r"""Per-band rotation-invariant MLP, repeated independently per band.

    Input shape ``(T, N, num_bands, electrodes, ...)``;
    output shape ``(T, N, num_bands, mlp_features[-1])``.
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
                f"{tuple(inputs.shape)}."
            )
        per_band = inputs.unbind(self.stack_dim)
        return torch.stack(
            [mlp(b) for mlp, b in zip(self.mlps, per_band)],
            dim=self.stack_dim,
        )


class _TDSConv2dBlock(nn.Module):
    r"""Time-Depth-Separable 2-D conv block ([hannun2019tds]_).

    1×``kernel_width`` convolution followed by ReLU, residual skip, and
    LayerNorm. No temporal padding: each block strips
    ``kernel_width - 1`` frames from the start of the sequence.
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
        T_in, N, C = inputs.shape
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.activation(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)
        T_out = x.shape[0]
        # Skip-connection alignment: with ``kernel_size=(1, kernel_width)``
        # and no temporal padding, output frame ``i`` looks at input
        # frames ``[i, i + kernel_width - 1]`` and is naturally aligned
        # with input frame ``i + (kernel_width - 1)``. So the residual
        # for the ``T_out`` outputs is the LAST ``T_out`` input frames,
        # i.e. ``inputs[T_in - T_out:] = inputs[-T_out:]``.
        x = x + inputs[-T_out:]
        return self.layer_norm(x)


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
    r"""Stack of :class:`_TDSConv2dBlock` interleaved with ``_TDSFullyConnectedBlock``.

    With ``len(block_channels) = K`` and no temporal padding, the encoder
    strips ``K * (kernel_width - 1)`` frames in total. Each ``ch`` in
    ``block_channels`` must evenly divide ``num_features`` (the input's
    last dim) so the conv block can fold features as
    ``(channels=ch, width=num_features // ch)``.

    Design constraints (used by ``EMG2QwertyNet.compute_output_lengths``
    and ``get_output_shape``):

    - Conv blocks are ``_TDSConv2dBlock`` instances; FC blocks are
      ``_TDSFullyConnectedBlock`` instances. The output-length math
      assumes only the conv blocks shrink ``T``.
    - ``kernel_width`` is identical across all blocks.
    - No temporal padding anywhere; each conv block takes ``T_in`` frames
      to ``T_in - kernel_width + 1`` frames.
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
        for ch in block_channels:
            if num_features % ch != 0:
                raise ValueError(
                    f"block_channels entry {ch} does not evenly divide "
                    f"num_features={num_features}."
                )
            blocks.extend(
                [
                    _TDSConv2dBlock(
                        ch,
                        num_features // ch,
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
