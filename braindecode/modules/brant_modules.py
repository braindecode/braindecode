"""Building blocks for :class:`braindecode.models.Brant`.

Faithful re-implementation of the upstream Brant reference code
(``Brant_src/pretrain/pre_model.py``, Apache-2.0,
https://huggingface.co/Daoze/Brant) as standalone braindecode modules — no new
runtime dependency (the encoders are stock :class:`torch.nn.TransformerEncoder`).

The only braindecode-native change lives in :class:`_BandPowerFeatures`: upstream
computes the per-band spectral power *outside* the model (scipy periodogram, fed
in as an argument); here it is computed **inside** the forward pass so the model
keeps the standard ``(batch, n_chans, n_times)`` input signature. The two
Transformer encoders preserve the upstream parameter layout and are checked
for numerical parity within a documented tolerance (see
``scripts/brant_parity_check``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _BandPowerFeatures(nn.Module):
    """Per-patch log spectral power in a set of frequency bands.

    braindecode-native replacement for the upstream ``compute_power`` (scipy
    ``signal.periodogram`` + per-band log-sum). Computed inside ``forward`` so
    the model consumes raw signal, not pre-extracted features. This module has
    no learnable parameters.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the input signal, in Hz.
    bands : tuple of (float, float)
        Frequency band edges ``(low, high)`` in Hz. A frequency ``f`` belongs to
        a band when ``low < f <= high`` (upstream convention).
    """

    def __init__(self, sfreq: float, bands: tuple[tuple[float, float], ...]):
        super().__init__()
        self.sfreq = float(sfreq)
        self.bands = tuple(bands)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Compute log band-power of every patch.

        Parameters
        ----------
        patches : torch.Tensor
            Shape ``(batch, n_chans, seq_len, patch_size)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_chans, seq_len, n_bands)``.
        """
        n = patches.shape[-1]
        # scipy periodogram default: detrend='constant' (remove the mean).
        x = patches - patches.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(x, dim=-1)
        # one-sided power spectral density, density scaling (boxcar window).
        psd = spectrum.abs().pow(2) / (self.sfreq * n)
        psd[..., 1:] = psd[..., 1:] * 2
        if n % 2 == 0:  # do not double the Nyquist bin
            psd[..., -1] = psd[..., -1] / 2
        freqs = torch.fft.rfftfreq(n, d=1.0 / self.sfreq, device=patches.device)

        out = []
        for low, high in self.bands:
            mask = (freqs > low) & (freqs <= high)
            band = psd[..., mask].sum(dim=-1)
            out.append(torch.log10(band + 1.0))
        return torch.stack(out, dim=-1)


class _BrantInputEmbedding(nn.Module):
    """Input encoding of Brant: linear patch projection + frequency + position.

    Ports the ``linear`` / ``use_power=True`` / ``need_mask=False`` path of the
    upstream ``InputEmbedding``. Attribute names (``proj``, ``band_encoding``,
    ``positional_encoding``) match upstream so weights map directly.
    """

    def __init__(self, patch_size: int, d_model: int, seq_len: int, n_bands: int):
        super().__init__()
        self.band_encoding = nn.Parameter(torch.randn(n_bands, d_model))
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        self.proj = nn.Sequential(nn.Linear(patch_size, d_model))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, data: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            Patches, shape ``(batch, n_chans, seq_len, patch_size)``.
        power : torch.Tensor
            Raw (pre-softmax) log band-power, shape
            ``(batch, n_chans, seq_len, n_bands)``.

        Returns
        -------
        torch.Tensor
            Input encoding, shape ``(batch * n_chans, seq_len, d_model)``.
        """
        bat_size, ch_num, seq_len, seg_len = data.shape
        power = self.softmax(power)
        power_emb = torch.einsum("hijk, kl->hijl", power, self.band_encoding)

        data = data.reshape(bat_size * ch_num, seq_len, seg_len)
        input_emb = self.proj(data)
        input_emb = input_emb + power_emb.reshape(bat_size * ch_num, seq_len, -1)
        input_emb = input_emb + self.positional_encoding
        return input_emb


class _BrantTemporalEncoder(nn.Module):
    """Temporal Transformer encoder (upstream ``TimeEncoder``).

    Applies the input encoding then a stack of ``TransformerEncoderLayer`` over
    the ``seq_len`` consecutive patches of each channel.
    """

    def __init__(
        self,
        patch_size: int,
        d_model: int,
        seq_len: int,
        n_bands: int,
        dim_feedforward: int,
        n_layers: int,
        n_heads: int,
        drop_prob: float,
    ):
        super().__init__()
        self.input_embedding = _BrantInputEmbedding(
            patch_size=patch_size,
            d_model=d_model,
            seq_len=seq_len,
            n_bands=n_bands,
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=drop_prob,
            batch_first=True,
        )
        self.trans_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, data: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
        input_emb = self.input_embedding(data, power)
        return self.trans_enc(input_emb)


class _BrantSpatialEncoder(nn.Module):
    """Spatial Transformer encoder (upstream ``ChannelEncoder``).

    Attends over the ``n_chans`` channels sharing a time index. ``proj_out`` is
    the upstream reconstruction projection (kept for weight parity; unused by the
    classification path).
    """

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        dim_feedforward: int,
        n_layers: int,
        n_heads: int,
        drop_prob: float,
    ):
        super().__init__()
        self.proj_out = nn.Sequential(nn.Linear(d_model, out_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=drop_prob,
            batch_first=True,
        )
        self.trans_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, time_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ch_z = self.trans_enc(time_z)
        rec = self.proj_out(ch_z)
        return ch_z, rec


class _BrantHead(nn.Module):
    """Downstream classification head (upstream ``model.MLP``): a 3-layer MLP."""

    def __init__(
        self, in_dim: int, out_dim: int, activation: type[nn.Module] = nn.ReLU
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            activation(),
            nn.Linear(in_dim // 2, in_dim // 4),
            activation(),
            nn.Linear(in_dim // 4, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)
