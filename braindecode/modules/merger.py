# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Meta Platforms, Inc. and affiliates (original brainmagick/DANCE)
#
# License: MIT
# Spatial Fourier channel-attention merger, shared by the brainmagick
# ``BrainModule`` (a.k.a. SimpleConv) and the DANCE model. Re-implemented to
# match the upstream ``neuraltrain.models.common`` (FourierEmbModel /
# ChannelMergerModel), so the attention scores are numerically identical to
# upstream (verified weight-for-weight with ``torch.allclose``).
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

__all__ = ["FourierEmb", "ChannelMerger", "positions_from_chs_info"]


def positions_from_chs_info(chs_info) -> np.ndarray:
    """``(n_chans, 2)`` electrode xy normalized to ``[0, 1]`` per axis.

    Takes the raw 3D sensor coordinates ``ch["loc"][:3]``, keeps ``xy`` and
    min-max normalizes each axis to ``[0, 1]``. This matches the upstream
    brainmagick/DANCE convention (``dance/example/data.py``): it is **not** MNE's
    ``_find_topomap_coords`` / the ``visualization/topology.py`` projection
    (which returns an interpolated scalp grid rather than per-channel coords and
    would change the merger softmax). The ``1e-9`` floor avoids a divide-by-zero
    on a degenerate (constant) axis.

    Parameters
    ----------
    chs_info : list of dict
        MNE-style channel info dicts, each carrying a ``"loc"`` array whose first
        three entries are the head-frame ``x, y, z`` coordinates.

    Returns
    -------
    numpy.ndarray
        ``(n_chans, 2)`` float positions in ``[0, 1]``.
    """
    xyz = np.array([ch["loc"][:3] for ch in chs_info], dtype=float)
    xy = xyz[:, :2]
    mn, mx = xy.min(axis=0), xy.max(axis=0)
    return (xy - mn) / np.maximum(mx - mn, 1e-9)


def has_valid_locations(chs_info) -> bool:
    """``True`` if ``chs_info`` carries finite, non-all-zero electrode locations."""
    if chs_info is None:
        return False
    try:
        xyz = np.array([ch["loc"][:3] for ch in chs_info], dtype=float)
    except (KeyError, TypeError, ValueError):
        return False
    if xyz.size == 0 or not np.isfinite(xyz).all():
        return False
    if np.allclose(xyz, 0.0):
        return False
    return True


class FourierEmb(nn.Module):
    """2D Fourier positional embedding over electrode ``(x, y)`` in ``[0, 1]``.

    Re-implemented to match ``neuraltrain.models.common.FourierEmbModel`` so the
    :class:`ChannelMerger` produces identical attention scores (parity-gated).

    Parameters
    ----------
    dimension : int
        Output embedding dimension. The number of frequencies per axis is
        ``round((dimension / 2) ** (1 / 2))`` (must be an integer; ``32`` for the
        default ``2048``).
    margin : float
        Position offset/width term from the upstream construction.
    """

    # Transcribed from neuraltrain.models.common.FourierEmbModel (margin,
    # 2*pi*freqs/width scaling, einsum + _outer_sum grid, cat[cos, sin]).
    # dimension == (n_freqs ** 2) * 2; n_freqs = round((dimension/2) ** 0.5).
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
        freqs = torch.arange(n_freqs)
        width = 1 + 2 * self.margin
        pos = 2 * math.pi * freqs / width
        self.register_buffer("pos", pos)

    @staticmethod
    def _outer_sum(x: torch.Tensor) -> torch.Tensor:
        # Outer sum over the last (n_dims) axes -> grid of size n_freqs**n_dims.
        # Verbatim from neuraltrain.models.common.FourierEmbModel._outer_sum.
        from collections import deque

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

    A montage-agnostic spatial attention: each of ``out_channels`` virtual
    channels attends (softmax over the input channels) using a learned head
    weighted against a Fourier embedding of the electrode ``(x, y)`` positions.
    Re-implemented from ``neuraltrain.models.common.ChannelMergerModel``
    (parity-gated). braindecode has no subjects, so ``per_subject=False`` and the
    upstream ``subject_ids`` argument is dropped (always ``None``).

    Parameters
    ----------
    out_channels : int
        Number of virtual output channels. The default is ``270``.
    pos_dim : int
        Fourier embedding dimension. The default is ``2048``.
    dropout : float
        Spatial-attention dropout *radius* (non-parametric, training only) -- NOT
        a Bernoulli probability. On each training step a random center is drawn
        in the normalized ``[0, 1]^2`` position space and every channel within
        this radius is banned from the softmax. Values ``>= 1`` ban all channels
        and zero the output. The default is ``0.2``.
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
        # Build the score offset on ``positions``' device (so the masked_fill with
        # ``invalid_mask`` -- derived from ``positions`` -- never hits a device
        # mismatch) and in ``x``'s dtype (so the final ``weights @ x`` matmul works
        # under half precision). Both are numerics-neutral in the usual fp32,
        # same-device case (the upstream copy omits the dtype).
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
