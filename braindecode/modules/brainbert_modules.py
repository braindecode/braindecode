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
``SpecPredictionHead`` (see ``scripts/brainbert_parity_check``).
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

    The defaults match the released ("stft") checkpoint: ``fs=2048`` Hz,
    ``nperseg=400``, ``noverlap=350`` (hop 50), the first ``freq_cutoff=40``
    one-sided frequency bins, ``boundary="zeros"`` + ``padded=True`` framing
    (as in :func:`scipy.signal.stft`), ``clip`` boundary frames trimmed from each
    end and per-bin z-score over time.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the input signal, in Hz.
    nperseg : int
        STFT window length, in samples.
    noverlap : int
        Number of samples of overlap between consecutive windows.
    freq_cutoff : int
        Number of low-frequency one-sided bins kept (the model ``input_dim``).
    clip : int
        Number of boundary frames trimmed from each end (handles STFT edge
        effects, as in the upstream demo).
    normalizing : str
        ``"zscore"`` (per-bin z-score over time, upstream default) or ``"none"``.
    """

    def __init__(
        self,
        sfreq: float,
        nperseg: int = 400,
        noverlap: int = 350,
        freq_cutoff: int = 40,
        clip: int = 5,
        normalizing: str = "zscore",
    ):
        super().__init__()
        if noverlap >= nperseg:
            raise ValueError(f"noverlap ({noverlap}) must be < nperseg ({nperseg}).")
        if freq_cutoff > nperseg // 2 + 1:
            raise ValueError(
                f"freq_cutoff ({freq_cutoff}) exceeds the number of one-sided "
                f"bins ({nperseg // 2 + 1}) for nperseg={nperseg}."
            )
        self.sfreq = float(sfreq)
        self.nperseg = int(nperseg)
        self.noverlap = int(noverlap)
        self.freq_cutoff = int(freq_cutoff)
        self.clip = int(clip)
        self.normalizing = normalizing
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
            Shape ``(batch, n_chans, n_frames, freq_cutoff)``.
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
        mag = spec[..., : self.freq_cutoff].abs()
        # trim boundary frames (time axis == -2)
        if self.clip:
            mag = mag[..., self.clip : -self.clip, :]
        if self.normalizing == "zscore":
            mean = mag.mean(dim=-2, keepdim=True)
            std = mag.std(dim=-2, unbiased=False, keepdim=True)
            std = torch.where(std == 0, torch.ones_like(std), std)
            mag = (mag - mean) / std
        return mag


class _SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (upstream ``PositionalEncoding``).

    ``pe`` is a non-learnable buffer of shape ``(1, max_len, d_model)``; the
    forward adds the leading ``seq_len`` positions to the input.
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
        return seq + self.pe[:, : seq.size(1), :]


class _BrainBERTInputEmbedding(nn.Module):
    """Input encoding of BrainBERT (upstream ``TransformerEncoderInput``).

    Linear projection ``input_dim -> hidden_dim``, additive sinusoidal position
    encoding, LayerNorm and dropout. Attribute names (``in_proj``,
    ``positional_encoding``, ``layer_norm``) match upstream so weights map
    directly.
    """

    def __init__(self, input_dim: int, hidden_dim: int, drop_prob: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = _SinusoidalPositionalEncoding(hidden_dim)
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
    """braindecode-native classification head: LayerNorm + linear on the pooled
    representation. Upstream has no fixed downstream head (linear probing /
    task-specific finetuning), so this is a minimal, standard choice."""

    def __init__(self, hidden_dim: int, n_outputs: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_outputs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(z))
