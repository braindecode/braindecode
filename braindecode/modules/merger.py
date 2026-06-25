# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Meta Platforms, Inc. and affiliates (original brainmagick/DANCE)
#
# License: MIT
# Spatial Fourier channel-attention merger shared by ``BrainModule`` (SimpleConv)
# and DANCE. Re-implemented to match upstream ``neuraltrain.models.common``
# (FourierEmbModel / ChannelMergerModel), so scores are numerically identical.
from __future__ import annotations

import math
from collections import deque

import torch
from torch import nn

__all__ = ["FourierEmb", "ChannelMerger"]


class FourierEmb(nn.Module):
    """2D Fourier positional embedding over electrode ``(x, y)`` in ``[0, 1]``.

    Matches ``neuraltrain.models.common.FourierEmbModel`` (parity-gated) so
    :class:`ChannelMerger` produces identical attention scores.

    Parameters
    ----------
    dimension : int
        Output embedding dimension. ``dimension == (n_freqs ** 2) * 2`` with
        ``n_freqs = round((dimension / 2) ** 0.5)`` (must be an integer; ``32``
        for the default ``2048``).
    margin : float
        Position offset/width term from the upstream construction.
    """

    def __init__(self, dimension: int = 2048, margin: float = 0.2):
        super().__init__()
        n_freqs_f = (dimension / 2) ** (1 / 2)
        n_freqs = round(n_freqs_f)
        if abs(n_freqs_f - n_freqs) > 1e-6:
            raise ValueError("(dimension / 2) ** (1 / 2) must be an integer.")
        self.n_dims = 2
        self.margin = margin
        freqs = torch.arange(n_freqs)
        width = 1 + 2 * self.margin
        pos = 2 * math.pi * freqs / width
        self.register_buffer("pos", pos)

    @staticmethod
    def _outer_sum(x: torch.Tensor) -> torch.Tensor:
        # Outer sum over the last (n_dims) axes -> grid of size n_freqs**n_dims.
        inds = deque([slice(None)] + [None] * (x.shape[-1] - 1))
        out = x[..., 0][(...,) + tuple(inds)]
        for i in range(1, x.shape[-1]):
            inds.rotate()
            out = out + x[..., i][(...,) + tuple(inds)]
        return out

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: (B, n_chans, 2) in [0, 1].
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

    Montage-agnostic spatial attention: each virtual channel softmax-attends
    over the input channels using a learned head weighted against a Fourier
    embedding of the electrode ``(x, y)`` positions. Matches
    ``neuraltrain.models.common.ChannelMergerModel`` (parity-gated); braindecode
    has no subjects, so ``per_subject=False`` and ``subject_ids`` is dropped.

    Parameters
    ----------
    out_channels : int
        Number of virtual output channels. The default is ``270``.
    pos_dim : int
        Fourier embedding dimension. The default is ``2048``.
    dropout : float
        Spatial-attention dropout *radius* (non-parametric, training only) --
        NOT a Bernoulli probability. Each training step draws a random center in
        normalized ``[0, 1]^2`` position space and bans every channel within
        this radius from the softmax. Values ``>= 1`` ban all channels and zero
        the output. The default is ``0.2``.
    invalid_value : float
        Position value marking padded/invalid channels (masked out of the
        softmax). The default is ``-0.1``.
    """

    def __init__(
        self,
        out_channels: int = 270,
        pos_dim: int = 2048,
        dropout: float = 0.2,
        invalid_value: float = -0.1,
    ):
        super().__init__()
        self.heads = nn.Parameter(torch.randn(out_channels, pos_dim))
        self.heads.data /= pos_dim**0.5
        self.embedding = FourierEmb(pos_dim)
        self.n_dims = self.embedding.n_dims
        self.dropout = dropout
        self.invalid_value = invalid_value

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: (B, n_chans, T); positions: (B, n_chans, 2). Transcribed from
        # neuraltrain ChannelMergerModel (per_subject=False, embed_ref=False,
        # unmerge=False; subject_ids unused -- braindecode has no subjects).
        B, C, _ = positions.shape
        embedding = self.embedding(positions)  # (B, n_chans, pos_dim)
        # score_offset lives on ``positions``' device (the invalid_mask is derived
        # from positions) and in ``x``'s dtype (so the final ``weights @ x`` works
        # under half precision); numerics-neutral in the usual fp32 same-device case.
        score_offset = torch.zeros(B, C, device=positions.device, dtype=x.dtype)
        invalid_mask = (positions == self.invalid_value).all(dim=-1)
        score_offset = score_offset.masked_fill(invalid_mask, float("-inf"))
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
