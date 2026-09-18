"""Building blocks for :class:`braindecode.models.BrainBERT`.

Faithful re-implementation of the upstream BrainBERT reference code
(``BrainBERT/models/masked_tf_model.py`` and friends,
https://github.com/czlwang/BrainBERT) as standalone braindecode modules — no new
runtime dependency (the encoder is a stock :class:`torch.nn.TransformerEncoder`).

The only braindecode-native change lives in :class:`_STFTSpectrogram`: upstream
computes the short-time Fourier transform *outside* the model (scipy, fed in as
the spectrogram argument); here it is computed **inside** the forward pass so the
model keeps the standard ``(batch, n_chans, n_times)`` input signature. Every
other module (input embedding, spectrogram-prediction head) is ported verbatim,
so its parameters map 1:1 to the upstream ``TransformerEncoderInput`` /
``SpecPredictionHead``. That mapping is asserted by the parity gate in
``test/unit_tests/models/test_brainbert.py``, which runs against a real clone of
the upstream repository when ``BRAINBERT_SRC`` is set.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _STFTSpectrogram(nn.Module):
    """Magnitude STFT spectrogram, computed inside the model.

    braindecode-native replacement for BrainBERT's scipy front-end
    (``signal.stft`` + magnitude + z-score), reproducing it in pure torch so the
    model consumes raw signal, not a pre-extracted spectrogram. This module has
    no learnable parameters.

    .. note::
       **The upstream repository ships two different STFT recipes, and they are
       not equivalent.** They differ in the *order* of the trimming and the
       z-score, and in the number of frames trimmed:

       * ``preprocessors/stft.py`` (``STFTPreprocessor``) z-scores **first**,
         then trims **10** frames per side. This is the recipe reached from
         ``conf/preprocessor/stft_pretrained.yaml`` through
         ``preprocessors/spec_pretrained.py``, i.e. the one used to produce the
         published downstream numbers with the released checkpoint. Upstream
         chose this order deliberately, to keep NaNs out of the statistics
         (see the ``TODO`` comment on that line).
       * ``notebooks/demo.ipynb`` (``get_stft``) trims **5** frames per side
         **first**, then z-scores.

       The defaults here follow the *released-checkpoint* recipe
       (``clip=10``, ``zscore_before_clip=True``). Set ``clip=5,
       zscore_before_clip=False`` to reproduce the demo notebook instead. On
       filtered noise the two agree to a correlation of 0.999 but differ by up
       to 0.48 z-unit per bin, and produce sequences whose lengths differ by 10
       frames, so the choice is not cosmetic.

    The remaining defaults match the released checkpoint: ``fs=2048`` Hz,
    ``nperseg=400``, ``noverlap=350`` (hop 50), the first ``idx_freq_cutoff=40``
    one-sided frequency bins, and ``boundary="zeros"`` + ``padded=True`` framing
    (as in :func:`scipy.signal.stft`).

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the input signal, in Hz.
    nperseg : int
        STFT window length, in samples.
    noverlap : int
        Number of samples of overlap between consecutive windows.
    idx_freq_cutoff : int
        Number of low-frequency one-sided bins kept (the model ``input_dim``).
        This is a **bin index**, not a frequency: with ``nperseg=400`` and
        ``sfreq=2048`` Hz, the default 40 bins reach about 200 Hz.
    clip : int
        Number of boundary frames trimmed from each end (handles STFT edge
        effects). 10 upstream in ``preprocessors/stft.py``, 5 in the demo
        notebook.
    normalizing : str
        ``"zscore"`` (per-bin z-score over time, upstream default) or ``"none"``.
    zscore_before_clip : bool
        Whether the z-score is computed before the boundary frames are trimmed
        (``preprocessors/stft.py``) or after (demo notebook). Ignored when
        ``normalizing != "zscore"``.
    """

    def __init__(
        self,
        sfreq: float,
        nperseg: int = 400,
        noverlap: int = 350,
        idx_freq_cutoff: int = 40,
        clip: int = 10,
        normalizing: str = "zscore",
        zscore_before_clip: bool = True,
    ):
        super().__init__()
        if noverlap >= nperseg:
            raise ValueError(f"noverlap ({noverlap}) must be < nperseg ({nperseg}).")
        if idx_freq_cutoff > nperseg // 2 + 1:
            raise ValueError(
                f"idx_freq_cutoff ({idx_freq_cutoff}) exceeds the number of "
                f"one-sided bins ({nperseg // 2 + 1}) for nperseg={nperseg}."
            )
        if normalizing not in ("zscore", "none"):
            raise ValueError(
                f"normalizing must be 'zscore' or 'none', got {normalizing!r}."
            )
        self.sfreq = float(sfreq)
        self.nperseg = int(nperseg)
        self.noverlap = int(noverlap)
        self.idx_freq_cutoff = int(idx_freq_cutoff)
        self.clip = int(clip)
        self.normalizing = normalizing
        self.zscore_before_clip = bool(zscore_before_clip)
        # scipy uses a periodic ("fftbins") Hann window; scaling='spectrum'
        # normalises by the window sum. Registered as a buffer so it follows
        # device / dtype moves without being a learnable parameter.
        # deterministic constant, regenerated in __init__: keep it out of the
        # state_dict (persistent=False) so checkpoints stay lean and safetensors
        # serialization does not trip on a non-owning buffer.
        win = torch.hann_window(nperseg, periodic=True)
        self.register_buffer("window", win, persistent=False)

    def n_frames(self, n_times: int) -> int:
        """Number of output frames for a signal of ``n_times`` samples."""
        step = self.nperseg - self.noverlap
        pad = self.nperseg // 2  # boundary="zeros"
        length = n_times + 2 * pad
        n_add = (-(length - self.nperseg)) % step  # padded=True
        length += n_add
        n_seg = (length - self.nperseg) // step + 1
        return n_seg - 2 * self.clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the magnitude spectrogram of every channel.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_chans, n_times)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_chans, n_frames, idx_freq_cutoff)``.
        """
        step = self.nperseg - self.noverlap
        pad = self.nperseg // 2
        # scipy boundary="zeros": half-window of zeros on each end.
        xp = F.pad(x, (pad, pad))
        # scipy padded=True: extend so an integer number of segments fits.
        n_add = (-(xp.shape[-1] - self.nperseg)) % step
        if n_add:
            xp = F.pad(xp, (0, n_add))
        # frame the signal: (batch, n_chans, n_frames, nperseg)
        frames = xp.unfold(-1, self.nperseg, step)
        win = self.window.to(frames.dtype)
        scale = 1.0 / win.sum()  # scaling="spectrum"
        spec = torch.fft.rfft(frames * win, n=self.nperseg, dim=-1) * scale
        # keep the low-frequency bins, take magnitude: (b, c, n_frames, cutoff)
        mag = spec[..., : self.idx_freq_cutoff].abs()
        # The time axis is -2. Upstream applies the z-score and the boundary
        # trimming in an order that depends on the recipe (see the class note).
        if self.normalizing == "zscore" and self.zscore_before_clip:
            mag = self._zscore(mag)
            # upstream stft.py: a window whose z-scored spectrogram is entirely
            # constant (a flat or dead channel) is replaced by ones rather than
            # left at zero. Upstream tests this on a single electrode, so the
            # faithful generalisation is per (batch, channel).
            # Written branch-free on purpose: an ``if degenerate.any()`` would
            # be a data-dependent guard and break ``torch.export``.
            degenerate = mag.flatten(start_dim=-2).std(dim=-1) == 0
            mag = torch.where(degenerate[..., None, None], torch.ones_like(mag), mag)
        if self.clip:
            mag = mag[..., self.clip : -self.clip, :]
        if self.normalizing == "zscore" and not self.zscore_before_clip:
            mag = self._zscore(mag)
        # upstream stft.py: NaNs surviving the statistics are zeroed rather than
        # propagated. Without this a single bad sample poisons the whole window.
        return torch.nan_to_num(mag, nan=0.0)

    @staticmethod
    def _zscore(mag: torch.Tensor) -> torch.Tensor:
        """Per-bin z-score over time, with the upstream zero-variance guard."""
        mean = mag.mean(dim=-2, keepdim=True)
        # ddof=0, as scipy's default and as upstream's hand-rolled zscore().
        std = mag.std(dim=-2, unbiased=False, keepdim=True)
        # upstream: `std[(std==0)] = 1.0`, described there as "a hack".
        std = torch.where(std == 0, torch.ones_like(std), std)
        return (mag - mean) / std


class _SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (upstream ``PositionalEncoding``).

    ``pe`` is a non-learnable buffer of shape ``(1, max_len, d_model)``; the
    forward adds the leading ``seq_len`` positions to the input, and refuses a
    sequence longer than the table rather than silently truncating it.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # fixed sinusoidal table, regenerated in __init__: not persisted.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        seq_len = seq.size(1)
        max_len = self.pe.size(1)
        if seq_len > max_len:
            raise ValueError(
                f"Sequence of {seq_len} frames exceeds the positional encoding "
                f"table ({max_len}). Build the model with a larger max_len."
            )
        return seq + self.pe[:, :seq_len, :]


class _BrainBERTInputEmbedding(nn.Module):
    """Input encoding of BrainBERT (upstream ``TransformerEncoderInput``).

    Linear projection ``input_dim -> hidden_dim``, additive sinusoidal position
    encoding, LayerNorm and dropout. Attribute names (``in_proj``,
    ``positional_encoding``, ``layer_norm``) match upstream so weights map
    directly.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        drop_prob: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = _SinusoidalPositionalEncoding(hidden_dim, max_len)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(spec)
        h = self.positional_encoding(h)
        h = self.layer_norm(h)
        return self.dropout(h)


class _SpecPredictionHead(nn.Module):
    """Masked-spectrogram reconstruction head (upstream ``SpecPredictionHead``).

    Kept so the pre-training parameters map 1:1 to upstream (weight parity);
    unused by the braindecode classification path.
    """

    def __init__(self, hidden_dim: int, input_dim: int):
        super().__init__()
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.act_fn = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = self.hidden_layer(hidden)
        h = self.act_fn(h)
        h = self.layer_norm(h)
        return self.output(h)


class _BrainBERTHead(nn.Module):
    """Downstream classification head: a bare linear probe.

    Upstream evaluates the frozen encoder with a plain ``nn.Linear`` on the
    pooled representation and nothing else — ``models/linear_wav_baseline.py``
    is literally ``nn.Linear(input_dim, 1)``. An earlier revision of this port
    put a :class:`~torch.nn.LayerNorm` in front of it; it was removed so a
    frozen-encoder probe reproduces the published protocol rather than a
    variant of it.
    """

    def __init__(self, hidden_dim: int, n_outputs: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, n_outputs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)
