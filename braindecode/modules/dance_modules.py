# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Meta Platforms, Inc. and affiliates (original DANCE)
#
# License: MIT
# Re-implemented from facebookresearch/dance (MIT); the upstream modules live
# in the `neuraltrain` package (neuraltrain.models.common / .simpleconv).
from __future__ import annotations

import math

import torch
from torch import nn


class FourierEmb(nn.Module):
    """2D Fourier positional embedding over electrode (x, y) in [0, 1].

    Re-implemented to match ``neuraltrain.models.common.FourierEmbModel`` so the
    ``ChannelMerger`` produces identical attention scores (parity-gated).

    Parameters
    ----------
    dimension : int
        Output embedding dimension. The number of frequencies per axis is
        ``int((dimension / 2) ** (1 / n_dims))`` with ``n_dims=2``.
    """

    # Transcribed verbatim from neuraltrain.models.common.FourierEmbModel
    # (margin, 2*pi*freqs/width scaling, einsum + _outer_sum grid, cat[cos,sin]).
    # dimension == total_dim == (n_freqs ** n_dims) * 2; with n_dims=2,
    # n_freqs = (dimension/2) ** (1/2) (must be an integer; 32 for 2048).
    def __init__(self, dimension: int = 2048, margin: float = 0.2):
        super().__init__()
        n_freqs_f = (dimension / 2) ** (1 / 2)
        n_freqs = round(n_freqs_f)
        if abs(n_freqs_f - n_freqs) > 1e-6:
            raise ValueError("(dimension / 2) ** (1 / 2) must be an integer.")
        self.n_freqs = n_freqs
        self.n_dims = 2
        self.margin = margin
        self.dimension = dimension
        # pos buffer: 2 * pi * arange(n_freqs) / (1 + 2*margin)  (width).
        freqs = torch.arange(n_freqs)
        width = 1 + 2 * self.margin
        pos = 2 * math.pi * freqs / width
        self.register_buffer("pos", pos)

    @staticmethod
    def _outer_sum(x: torch.Tensor) -> torch.Tensor:
        # Outer sum over the last (n_dims) axes of x -> grid of size n_freqs**n_dims.
        # Verbatim from neuraltrain.models.common.FourierEmbModel._outer_sum.
        from collections import deque

        inds = deque([slice(None)] + [None] * (x.shape[-1] - 1))
        out = x[..., 0][(...,) + tuple(inds)]
        for i in range(1, x.shape[-1]):
            inds.rotate()
            out = out + x[..., i][(...,) + tuple(inds)]
        return out

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: (B, n_chans, 2) in [0, 1]
        *o, d = positions.shape
        if d != self.n_dims:
            raise ValueError(f"Expected {self.n_dims} positions, got {d}.")
        positions = positions + self.margin
        locs = torch.einsum("bcd,f->bcfd", positions, self.pos)
        loc_grid = self._outer_sum(locs).view(*o, -1)  # (B, n_chans, n_freqs**2)
        emb = torch.cat([torch.cos(loc_grid), torch.sin(loc_grid)], dim=-1)
        return emb  # (B, n_chans, (n_freqs**2)*2) == (..., dimension)


class ChannelMerger(nn.Module):
    """Spatial Fourier-attention merge: ``n_chans`` -> ``out_channels``.

    Re-implemented from ``neuraltrain.models.common.ChannelMergerModel``
    (parity-gated). braindecode has no subjects, so ``per_subject=False`` and
    the upstream ``subject_ids`` argument is dropped (always ``None``).

    Parameters
    ----------
    out_channels : int
        Number of virtual output channels. The default is ``270``.
    pos_dim : int
        Fourier embedding dimension. The default is ``2048``.
    dropout : float
        Attention-score dropout (non-parametric). The default is ``0.2``.
    """

    def __init__(
        self,
        out_channels: int = 270,
        pos_dim: int = 2048,
        dropout: float = 0.2,
        invalid_value: float = -0.1,
    ):
        super().__init__()
        # heads: (out_channels, pos_dim), scaled by pos_dim**-0.5 (upstream).
        self.heads = nn.Parameter(torch.randn(out_channels, pos_dim))
        self.heads.data /= pos_dim**0.5
        self.embedding = FourierEmb(pos_dim)
        self.n_dims = self.embedding.n_dims
        self.dropout = dropout
        self.invalid_value = invalid_value

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        # x: (B, n_chans, T); positions: (B, n_chans, 2). Transcribed from
        # neuraltrain ChannelMergerModel (per_subject=False, embed_ref=False,
        # unmerge=False; subject_ids unused). braindecode has no subjects.
        B, C, _ = positions.shape
        embedding = self.embedding(positions)  # (B, n_chans, pos_dim)
        score_offset = torch.zeros(B, C, device=x.device)
        invalid_mask = (positions == self.invalid_value).all(dim=-1)
        score_offset = score_offset.masked_fill(invalid_mask, float("-inf"))
        # Spatial dropout: ban channels within a random radius (training only).
        if self.training and self.dropout:
            center = torch.rand(self.n_dims, device=x.device)
            banned = (positions[:, :, : self.n_dims] - center).norm(
                dim=-1
            ) <= self.dropout
            score_offset = score_offset.masked_fill(banned, float("-inf"))
        heads = self.heads[None].expand(B, -1, -1)  # (B, out, pos_dim)
        scores = torch.einsum("bcd,bod->boc", embedding, heads)  # (B, out, n_chans)
        scores = scores + score_offset[:, None]
        weights = torch.softmax(scores, dim=2).nan_to_num()  # over n_chans
        out = weights @ x  # (B, out, n_chans) @ (B, n_chans, T) -> (B, out, T)
        return out  # (B, out_channels, T)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, activation, last):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, stride=1, padding=pad, dilation=dilation
        )
        self.act = None if last else activation()

    def forward(self, x):
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SimpleConv(nn.Module):
    """Dilated conv front-end (Defossez lineage), parity-gated.

    No BatchNorm, no input/conv dropout, no skip in the DANCE config
    (verified from upstream ``_default_encoder_config``). ``activation``
    defaults to ``nn.ReLU`` to match upstream. Owns an OPTIONAL nested
    ``ChannelMerger`` (``self.merger``) applied first, mirroring
    ``neuraltrain.models.simpleconv.SimpleConvModel`` so the call path and
    state_dict keys (``merger.*``, ``initial_linear.*``, ``sequence.{k}.*``)
    align with upstream for parity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
        hidden: int = 512,
        depth: int = 10,
        kernel_size: int = 9,
        dilation_growth: float = 2.5,
        initial_linear: int = 256,
        initial_depth: int = 1,
        drop_prob: float = 0.2,
        activation: type[nn.Module] = nn.ReLU,
        merger: "ChannelMerger | None" = None,
    ):
        super().__init__()
        if dilation_growth > 1 and kernel_size % 2 == 0:
            raise ValueError("Odd kernel required with dilation.")
        # Optional nested merger; when present the conv blocks consume its
        # out_channels (upstream sets in_channels = merger.n_virtual_channels).
        self.merger = merger
        conv_in = merger.heads.shape[0] if merger is not None else in_channels
        layers: list[nn.Module] = []
        for _ in range(initial_depth):
            layers.append(nn.Conv1d(conv_in, initial_linear, 1))
            conv_in = initial_linear
        self.initial_linear = nn.Sequential(*layers)
        sizes = [initial_linear] + [hidden] * (depth - 1) + [out_channels]
        blocks = []
        # Match ConvSequence: dilation starts at 1.0, int() per block, then
        # multiply by dilation_growth AFTER each block (no dilation_period reset).
        dilation = 1.0
        for i in range(depth):
            d = int(dilation)
            last = i == depth - 1
            blocks.append(
                _ConvBlock(sizes[i], sizes[i + 1], kernel_size, d,
                           activation, last)
            )
            dilation = dilation * dilation_growth
        self.sequence = nn.ModuleList(blocks)
        # Dilation of the LAST block, for the model's min-length guard.
        self.max_dilation = int(dilation_growth ** (depth - 1))

    def forward(
        self,
        x: torch.Tensor,
        positions: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        # x: (B, in_channels, T); positions: (B, in_channels, 2) when merger set.
        length = x.shape[-1]
        if self.merger is not None:
            if positions is None:
                raise ValueError("SimpleConv with a merger requires positions.")
            x = self.merger(x, positions)  # (B, merger.out_channels, T)
        x = self.initial_linear(x)
        for block in self.sequence:
            x = block(x)
        if x.shape[-1] < length:
            raise ValueError(
                f"Expected output time dim >= {length}, got {x.shape[-1]}"
            )
        return x[..., :length]
