# Authors: Adam Mounir <am91ris@gmail.com>
#
# License: BSD (3-clause)
"""BrainBERT (Wang et al., ICLR 2023): a spectrogram Transformer for intracranial
signals, ported weight-for-weight from https://github.com/czlwang/BrainBERT with
the STFT front-end moved inside the model. See :class:`BrainBERT`.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from braindecode.functional import (
    check_ch_mask,
    masked_mean,
    pack_channels,
    pad_or_crop_to_patches,
    patch_count,
    sinusoidal_positional_encoding,
    spectral_input,
    unpack_channels,
)
from braindecode.functional.patches import _check_on_non_divisible
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import warn_if_sfreq_differs


class BrainBERT(EEGModuleMixin, nn.Module, license="unknown"):
    r"""BrainBERT from Wang et al. (2023) [BrainBERT2023]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. figure:: https://arxiv.org/html/2302.14367v1/figures/model.png
       :align: center
       :alt: BrainBERT architecture

    BrainBERT is a self-supervised foundation model for intracranial signals
    (sEEG/iEEG). It takes as input not the waveform but its **spectrogram**: a
    short-time Fourier transform maps each channel onto a series of time frames,
    and each frame is one token. A linear projection and a fixed sinusoidal
    positional encoding are followed by a stack of standard Transformer encoder
    layers. The model is pre-trained by masked spectrogram modelling, i.e. by
    reconstructing masked time/frequency patches, and the representation of each
    frame thus serves downstream decoding.

    In the present implementation, we compute the STFT inside ``forward`` (the
    ``spectrogram`` submodule), so that the model still takes the
    standard ``(batch, n_chans, n_times)`` input, whereas the upstream reference
    takes a pre-computed spectrogram as input. The input encoding and the encoder
    match the upstream ``MaskedTFModel`` to 1e-5.

    The released checkpoint was trained on signals sampled at **2048 Hz** and
    re-referenced with a Laplacian, with ``nperseg=400``, ``noverlap=350`` and the
    first ``idx_freq_cutoff=40`` frequency bins. The defaults below give a modest,
    ready-to-run model; the **released ("large") model** uses ``hidden_dim=768``,
    ``ffn_dim=3072``, ``n_heads=12`` and ``n_layers=6`` (about 43M parameters),
    and passing these values reproduces it.

    **The pooling follows the published downstream protocol.** Upstream processes
    one electrode at a time, averages the ``pool_n_frames=10`` encoder outputs
    centred on the window (``outputs[:, middle-5:middle+5].mean`` in
    ``preprocessors/spec_pretrained.py``) and applies a linear probe; the average
    over every frame appears in that file only as a commented-out alternative. We
    use the average over the central frames and add a mean over channels, which
    is the identity for ``n_chans=1``: a single-channel BrainBERT thus reproduces
    the upstream feature exactly, and several channels remain supported as the
    braindecode generalisation. Pass ``pool_n_frames=None`` to average every
    frame. To obtain one output per electrode as upstream does, build the model
    with ``n_chans=1`` and stack the electrodes in the batch.

    As channels are encoded independently, the number of channels may differ
    between batches, and within one batch when the recordings are padded to a
    common count and ``ch_mask`` (see :meth:`forward`) flags the real ones.

    .. important::
       **Pre-trained weights are available.** The checkpoint released by the
       authors is available from the Hugging Face Hub::

           model = BrainBERT.from_pretrained(
               "braindecode/brainbert-pretrained", n_outputs=2
           )

       It has the "large" configuration above; ``n_chans``, ``n_times`` and
       ``n_outputs`` may change freely, since the frames are pooled and the head
       is specific to the task (pass these, not ``chs_info`` or
       ``input_window_seconds``, which conflict with the saved config). The
       upstream repository provides no LICENSE file, so we mark the licence of
       the weights as unknown rather than assume a permissive one.

    .. note::
       As in the upstream front-end, a channel whose z-scored spectrogram is
       flat (a dead channel) is set to ones, and a ``NaN`` sample zeroes that
       channel's whole spectrogram for the window, without a warning.

    .. versionadded:: 1.9

    Parameters
    ----------
    hidden_dim : int, optional
        Transformer model width ``D``. Default 192. The released model uses 768.
    ffn_dim : int, optional
        Inner dimension of the Transformer feed-forward blocks. Default 384.
        The released model uses 3072.
    n_layers : int, optional
        Number of Transformer encoder layers. Default 2. Released model: 6.
    n_heads : int, optional
        Number of attention heads. Default 4. Released model: 12.
    nperseg : int, optional
        STFT window length in samples. Default 400.
    noverlap : int, optional
        STFT overlap in samples. Default 350 (hop of 50).
    idx_freq_cutoff : int, optional
        Number of low-frequency STFT bins kept; the Transformer ``input_dim``.
        This is a **bin index**, not a frequency in Hz: the default 40 bins
        reach about 200 Hz at 2048 Hz with ``nperseg=400``. Default 40.
    stft_clip : int, optional
        Boundary frames trimmed from each end of the spectrogram. Default 10,
        as in the upstream ``preprocessors/stft.py`` used with the released
        checkpoint (the demo notebook uses 5; see ``stft_zscore_before_clip``).
    stft_zscore_before_clip : bool, optional
        Whether the spectrogram is z-scored before the boundary frames are
        trimmed. Default ``True``, which together with ``stft_clip=10``
        reproduces the recipe behind the published numbers. Set to ``False``
        with ``stft_clip=5`` to reproduce the upstream demo notebook instead;
        the two recipes are close but not equal, and they yield sequences of
        different length.
    pool_n_frames : int or None, optional
        Number of encoder frames, centred on the window, averaged into the
        pooled representation. Default 10, as upstream. ``None`` averages all
        frames.
    activation : type[nn.Module], optional
        Transformer feed-forward activation class. Default ``nn.GELU`` (as
        pretrained).
    drop_prob : float, optional
        Dropout probability. Default 0.1.
    on_non_divisible : {"pad", "crop", "error"}, optional
        A window too short to give the ``pool_n_frames`` frames the pooling
        needs (``n_times < 1401`` with the defaults). ``"pad"`` (default)
        adds the missing STFT frames as zeros, half before and half after the
        window, with a warning at construction: the ``stft_clip`` frames at
        each end are trimmed and the centre ones pooled, so the signal stays
        in the pooled frames. ``"crop"`` and ``"error"`` raise. A long enough
        window gives the same output under all three.

    References
    ----------
    .. [BrainBERT2023] Wang, C., Subramaniam, V., Yaari, A.U., Kreiman, G.,
       Katz, B., Cases, I. and Barbu, A., 2023. BrainBERT: Self-supervised
       representation learning for intracranial recordings. In International
       Conference on Learning Representations, ICLR.
       Code: https://github.com/czlwang/BrainBERT
    """

    def __init__(
        self,
        # --- BrainBERT hyper-parameters (modest defaults; "large" in docstring) ---
        hidden_dim: int = 192,
        ffn_dim: int = 384,
        n_layers: int = 2,
        n_heads: int = 4,
        nperseg: int = 400,
        noverlap: int = 350,
        idx_freq_cutoff: int = 40,
        stft_clip: int = 10,
        stft_zscore_before_clip: bool = True,
        pool_n_frames: int | None = 10,
        activation: type[nn.Module] = nn.GELU,
        drop_prob: float = 0.1,
        on_non_divisible: str = "pad",
        # --- braindecode mandatory signal parameters ---
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        channel_strategy: str = "native",
        channel_strategy_kwargs: dict | None = None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            channel_strategy=channel_strategy,
            channel_strategy_kwargs=channel_strategy_kwargs,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        # Refuse rather than silently fall back to a different pooling: the
        # number of frames averaged is part of the published protocol.
        if pool_n_frames is not None and pool_n_frames <= 0:
            raise ValueError(
                f"pool_n_frames must be positive; got {pool_n_frames}. "
                "Pass None to average every frame instead."
            )
        _check_on_non_divisible(on_non_divisible)
        self.pool_n_frames = pool_n_frames
        self.min_frames = 1 if pool_n_frames is None else pool_n_frames
        warn_if_sfreq_differs("BrainBERT", self._sfreq, 2048)

        self.spectrogram = _STFTSpectrogram(
            nperseg=nperseg,
            noverlap=noverlap,
            idx_freq_cutoff=idx_freq_cutoff,
            clip=stft_clip,
            zscore_before_clip=stft_zscore_before_clip,
            min_frames=self.min_frames if on_non_divisible == "pad" else 0,
        )
        n_frames = self.spectrogram.n_frames(self.n_times)
        if n_frames < self.min_frames:
            min_n_times = self.spectrogram.min_n_times(self.min_frames)
            message = (
                f"n_times={self.n_times} gives {n_frames} STFT frames; the pooling "
                f"needs {self.min_frames}, i.e. n_times >= {min_n_times} with "
                "these STFT settings"
            )
            if on_non_divisible != "pad":
                raise ValueError(message + ".")
            warnings.warn(
                message + ": the input will be padded.", UserWarning, stacklevel=2
            )
            n_frames = self.min_frames

        # The re-hosted Hub checkpoint already uses this port's names. The
        # authors' original ``.pth`` calls the input block ``input_encoding``;
        # load ``torch.load(path, weights_only=False)["model"]`` with
        # ``strict=False`` (its fixed ``pe`` table and pre-training head are
        # dropped).
        self.mapping = {
            f"input_encoding.{name}": f"input_embedding.{name}"
            for name in (
                "in_proj.weight",
                "in_proj.bias",
                "layer_norm.weight",
                "layer_norm.bias",
            )
        }

        self.input_embedding = _BrainBERTInputEmbedding(
            input_dim=idx_freq_cutoff,
            hidden_dim=hidden_dim,
            drop_prob=drop_prob,
            # upstream table size; inputs longer than n_times stay allowed.
            max_len=max(5000, n_frames),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            activation=activation(),
            dropout=drop_prob,
            batch_first=True,
        )
        # No padding mask is ever passed, so the nested-tensor fast path (and its
        # warning for activations other than ReLU/GELU) is simply off.
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        # Upstream's downstream probe is a bare linear layer
        # (``models/linear_wav_baseline.py``), with no normalisation in front.
        self.final_layer = nn.Linear(hidden_dim, self.n_outputs)

    def reset_head(self, n_outputs: int) -> None:
        """Swap the classification head for a new number of outputs."""
        self._set_n_outputs(n_outputs)
        head = nn.Linear(self.final_layer.in_features, n_outputs)
        self.final_layer = head.to(self.final_layer.weight)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        ch_mask: torch.Tensor | None = None,
    ):
        """Decode a batch of signals.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``; ``n_chans`` may differ
            from the construction one.
        return_features : bool
            If ``True``, return the pooled encoder embedding instead of the
            class logits, as ``{"features": pooled, "cls_token": None}``
            (braindecode foundation-model convention). BrainBERT pools the
            centre frames and then the channels, and has no class token, hence
            ``cls_token`` is ``None``. A scripted model (``torch.jit.script``)
            ignores this flag and always returns the logits.
        ch_mask : torch.Tensor, optional
            Boolean mask of shape ``(batch, n_chans)``, ``True`` for real
            channels and ``False`` for padding, as returned by
            :func:`~braindecode.datasets.pad_channels_collate`, so recordings
            with different numbers of channels can share a batch. Padded
            channels are dropped before the spectrogram, so they cost no
            compute and are left out of the mean over channels.

        Returns
        -------
        torch.Tensor or dict
            Class logits of shape ``(batch, n_outputs)``, or the feature dict
            ``{"features", "cls_token"}`` when ``return_features`` is set.
        """
        mask = check_ch_mask(ch_mask, x)
        n_chans = x.shape[1]
        # Each real channel is its own sequence, as upstream feeds one electrode.
        x = pack_channels(x, mask)  # (n_real, n_times)

        # 1. spectrogram front-end.
        spec = self.spectrogram(x)  # (n_real, n_frames, idx_freq_cutoff)
        n_frames = spec.shape[1]
        # An input shorter than the model was built for must still give enough
        # frames, otherwise the centre slice below silently shrinks.
        if n_frames < self.min_frames:
            raise ValueError(
                f"Input of {x.shape[-1]} samples gives {n_frames} spectrogram "
                f"frames; at least {self.min_frames} are needed."
            )

        # 2. input encoding + Transformer over the sequence of frames.
        h = self.input_embedding(spec)  # (n_real, n_frames, hidden_dim)
        z = self.transformer(h)  # (n_real, n_frames, hidden_dim)

        # 3. pool the centre frames, then frames and real channels.
        if self.pool_n_frames is not None:
            start = n_frames // 2 - self.pool_n_frames // 2
            z = z[:, start : start + self.pool_n_frames]
        z = unpack_channels(z, mask, n_chans)  # (batch, n_chans, frames, hidden_dim)
        pooled = masked_mean(z, mask, [1, 2])  # (batch, hidden_dim)
        logits = self.final_layer(pooled)
        if return_features:
            if torch.jit.is_scripting():
                return logits
            return {"features": pooled, "cls_token": None}  # nosec B105
        return logits


class _STFTSpectrogram(nn.Module):
    """Magnitude STFT spectrogram, computed inside the model.

    Pure-torch replacement for BrainBERT's scipy front-end (``signal.stft`` +
    magnitude + z-score), so the model consumes raw signal rather than a
    pre-extracted spectrogram. No learnable parameters. Framing follows
    :func:`scipy.signal.stft` with ``boundary="zeros"``, ``padded=True``, a
    periodic Hann window and ``scaling="spectrum"``.

    Upstream ships two recipes that differ in the order of the z-score and the
    boundary trimming, and in how many frames are trimmed; see ``stft_clip`` and
    ``stft_zscore_before_clip`` in :class:`BrainBERT`.

    Parameters
    ----------
    nperseg : int
        STFT window length, in samples.
    noverlap : int
        Number of samples of overlap between consecutive windows.
    idx_freq_cutoff : int
        Number of low-frequency one-sided bins kept (a bin index, not Hz).
    clip : int
        Number of boundary frames trimmed from each end.
    zscore_before_clip : bool
        Whether the per-bin z-score over time precedes the trimming.
    min_frames : int
        A signal giving fewer output frames gets the missing STFT frames as
        zeros, half before and half after it. ``0`` (default): scipy's framing
        only.
    """

    def __init__(
        self,
        nperseg: int = 400,
        noverlap: int = 350,
        idx_freq_cutoff: int = 40,
        clip: int = 10,
        zscore_before_clip: bool = True,
        min_frames: int = 0,
    ):
        super().__init__()
        if not 0 <= noverlap < nperseg:
            raise ValueError(
                f"noverlap ({noverlap}) must be in [0, nperseg={nperseg})."
            )
        if not 1 <= idx_freq_cutoff <= nperseg // 2 + 1:
            raise ValueError(
                f"idx_freq_cutoff ({idx_freq_cutoff}) must be in [1, "
                f"{nperseg // 2 + 1}] for nperseg={nperseg}."
            )
        if clip < 0:
            raise ValueError(f"clip must be >= 0; got {clip}.")
        self.nperseg = nperseg
        self.idx_freq_cutoff = idx_freq_cutoff
        self.clip = clip
        self.zscore_before_clip = zscore_before_clip
        self.min_frames = min_frames
        # "zscore" as upstream; pipelines that apply their own statistics (the
        # Neuroprobe global z-score recipes) set it to "none" on the instance.
        self.normalizing = "zscore"
        self.step = nperseg - noverlap
        self.boundary_pad = nperseg // 2  # scipy boundary="zeros"
        # scipy's periodic Hann window; non-persistent, rebuilt in __init__.
        self.register_buffer(
            "window", torch.hann_window(nperseg, periodic=True), persistent=False
        )

    def n_frames(self, n_times: int) -> int:
        """Number of frames scipy's framing gives ``n_times`` samples: segments of
        ``nperseg`` every ``step`` (the last one zero-padded), minus the clip."""
        length = n_times + 2 * self.boundary_pad
        return patch_count(length, self.nperseg, "pad", self.step) - 2 * self.clip

    def min_n_times(self, n_frames: int) -> int:
        """Shortest signal that yields at least ``n_frames`` output frames."""
        n_seg = n_frames + 2 * self.clip
        return self.nperseg + (n_seg - 2) * self.step + 1 - 2 * self.boundary_pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., n_times)`` -> ``(..., n_frames, cutoff)``."""
        if not x.is_floating_point():
            raise TypeError(
                f"BrainBERT expects a floating-point signal, got {x.dtype}."
            )
        # scipy boundary="zeros" (half a window of zeros on each end), then
        # padded=True (right zero-pad so the segments tile the signal). A signal
        # short of min_frames gets the missing segments as zeros, half before
        # it: the clip frames at each end are trimmed, so a right pad alone
        # would leave it outside the frames that are kept.
        xp = F.pad(x, (self.boundary_pad, self.boundary_pad))
        min_segments = self.min_frames + 2 * self.clip if self.min_frames else 1
        xp = pad_or_crop_to_patches(
            xp, self.nperseg, "pad", self.step, min_segments, center=True
        )
        frames = xp.unfold(-1, self.nperseg, self.step)  # (b, c, n_frames, nperseg)
        frames = spectral_input(frames)
        win = self.window.to(frames)
        scale = 1.0 / win.sum()  # scaling="spectrum"
        spec = torch.fft.rfft(frames * win, dim=-1)  # (b, c, n_frames, nperseg//2+1)
        low = spec[..., : self.idx_freq_cutoff] * scale  # (b, c, n_frames, cutoff)
        mag = low.abs()
        # The time axis is -2. The z-score and the trimming happen in the order
        # the recipe dictates (see the class docstring).
        if self.normalizing == "zscore" and self.zscore_before_clip:
            mag = self._zscore(mag)
            # upstream stft.py: a flat (e.g. dead) channel becomes ones. Written
            # branch-free so torch.export sees no data-dependent guard.
            degenerate = mag.flatten(start_dim=-2).std(dim=-1) == 0
            mag = mag.masked_fill(degenerate[..., None, None], 1.0)
        if self.clip:
            mag = mag[..., self.clip : -self.clip, :]
        if self.normalizing == "zscore" and not self.zscore_before_clip:
            mag = self._zscore(mag)
        # upstream stft.py: NaNs surviving the statistics are zeroed, not kept.
        return torch.nan_to_num(mag, nan=0.0).to(x)

    @staticmethod
    def _zscore(mag: torch.Tensor) -> torch.Tensor:
        """Per-bin z-score over time (ddof=0); zero standard deviations become one."""
        mean = mag.mean(dim=-2, keepdim=True)
        std = mag.std(dim=-2, unbiased=False, keepdim=True)
        std = std.masked_fill(std == 0, 1.0)
        return (mag - mean) / std


class _BrainBERTInputEmbedding(nn.Module):
    """Upstream ``TransformerEncoderInput``: linear projection, fixed sinusoidal
    position encoding, LayerNorm and dropout. ``in_proj`` and ``layer_norm``
    keep the upstream names; the sinusoidal table is rebuilt, not persisted."""

    def __init__(self, input_dim: int, hidden_dim: int, drop_prob: float, max_len: int):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=drop_prob)
        self.register_buffer(
            "pe", sinusoidal_positional_encoding(max_len, hidden_dim), persistent=False
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(spec)  # (batch, n_frames, hidden_dim)
        if h.size(1) > self.pe.size(0):
            raise ValueError(
                f"{h.size(1)} frames exceed the positional table ({self.pe.size(0)})."
            )
        h = h + self.pe[: h.size(1)]
        h = self.layer_norm(h)
        return self.dropout(h)
