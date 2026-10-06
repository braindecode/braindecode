# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Sensor-space strategies without a head model: exact, zero, nearest, idw."""

from __future__ import annotations

import numpy as np

from .base import ChannelStrategy, register_channel_strategy


def _scatter(src, use, rows):
    """Place rows over the used input channels into full ``(M, C)`` rows."""
    out = np.zeros((len(rows), len(src.names)))
    out[:, use] = rows
    return out


def _distances(src, use, tgt_pos):
    return np.linalg.norm(tgt_pos[:, None] - src.positions[use][None], axis=-1)


@register_channel_strategy("exact")
class ExactStrategy(ChannelStrategy):
    """Permutation / subset by resolved name; a missing target is an error."""

    reconstructs = False


@register_channel_strategy("zero")
class ZeroStrategy(ChannelStrategy):
    """Missing targets are zero rows with ``observed=False`` (baseline)."""

    min_positions = 0

    def _fill(self, src, use, tgt_pos):
        return np.zeros((len(tgt_pos), len(src.names)))


@register_channel_strategy("nearest")
class NearestStrategy(ChannelStrategy):
    """Missing target = copy of the nearest positioned input electrode."""

    def _fill(self, src, use, tgt_pos):
        d = _distances(src, use, tgt_pos)
        hit = (d == d.min(1, keepdims=True)).astype(float)
        return _scatter(src, use, hit / hit.sum(1, keepdims=True))  # ties share


@register_channel_strategy("idw")
class IDWStrategy(ChannelStrategy):
    """Missing target = inverse-distance weighted mean (``1 / d**p``)."""

    def __init__(self, p: float = 2.0):
        super().__init__()
        self.p = p

    def _fill(self, src, use, tgt_pos):
        w = 1.0 / np.maximum(_distances(src, use, tgt_pos), 1e-6) ** self.p
        return _scatter(src, use, w / w.sum(1, keepdims=True))
