# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Data-driven and learned strategies: wiener (fitted), region, latent (trained)."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

from ..resolve import nearest_vocabulary, resolve_montage
from .base import (
    SUPPORT_SCALE_MM,
    ChannelStrategy,
    SpatialMap,
    _f32,
    register_channel_strategy,
)
from .source import _mlp


def _numpy(t: Tensor) -> np.ndarray:
    """float64 numpy copy of a buffer on any device (MPS has no float64)."""
    return t.detach().cpu().double().numpy()


def _channel_stats(x: Tensor, used: Tensor) -> Tensor:
    """``(B, C, 2)`` log-variance and log line-length, centred over used channels."""
    w = used.to(x.dtype)
    stats = torch.stack(
        [
            torch.log(x.var(-1) + 1e-12),
            torch.log(x.diff(dim=-1).abs().mean(-1) + 1e-12),
        ],
        -1,
    )
    return stats - (stats * w[:, None]).sum(1, keepdim=True) / w.sum()


@register_channel_strategy("wiener")
class WienerStrategy(ChannelStrategy):
    """Linear MMSE from a spatial covariance fitted on dense recordings.

    Call :meth:`fit` (or :meth:`ChannelTokenizer.fit`) with dense training
    data first. Input and target electrodes are matched to the dense montage
    by position (within 15 mm). The fitted covariance is saved in the
    ``state_dict``.

    Parameters
    ----------
    noise : float
        Diagonal loading, relative to the mean observed variance.
    """

    def __init__(self, noise: float = 0.01):
        super().__init__()
        self.noise = noise
        self.register_buffer("cov", torch.zeros(0, 0))
        self.register_buffer("dense_positions", torch.zeros(0, 3))

    @property
    def fitted(self) -> bool:
        return self.cov.numel() > 0

    def fit(self, X: np.ndarray, chs_info_dense: list[dict]) -> "WienerStrategy":
        """Fit the covariance on ``X`` of shape ``(n_samples, n_dense_channels)``."""
        dense = resolve_montage(chs_info_dense)
        if not dense.positioned.all() or X.shape[1] != len(dense.names):
            raise ValueError(
                "fit() needs X of shape (n_samples, n_channels) and a position "
                f"for every dense channel; got X {X.shape} for {len(dense.names)} "
                "channels."
            )
        device = self.cov.device
        self.cov = _f32(np.cov(np.asarray(X, float).T)).to(device)
        self.dense_positions = _f32(dense.positions).to(device)
        return self

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in ("cov", "dense_positions"):
            if prefix + name in state_dict:
                setattr(self, name, torch.empty_like(state_dict[prefix + name]))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def build(self, src, target):
        if not self.fitted:
            raise ValueError(
                "The 'wiener' strategy needs a covariance: call fit() first."
            )
        return super().build(src, target)

    def _dense_index(self, positions, names):
        j = nearest_vocabulary(positions, _numpy(self.dense_positions))
        if (j < 0).any():
            raise ValueError(
                f"'wiener': channels {[n for n, i in zip(names, j) if i < 0]} have no "
                f"electrode of the fitted dense montage within 15 mm."
            )
        return j

    def _fill(self, src, use, tgt_pos):
        cov = _numpy(self.cov)
        js = self._dense_index(src.positions[use], np.asarray(src.names)[use])
        jt = self._dense_index(tgt_pos, [f"target at {p.round(3)}" for p in tgt_pos])
        C_obs = cov[np.ix_(js, js)]
        k = len(js)
        rows = np.zeros((len(tgt_pos), len(src.names)))
        rows[:, use] = cov[np.ix_(jt, js)] @ np.linalg.inv(
            C_obs + self.noise * np.trace(C_obs) / k * np.eye(k)
        )
        return rows


@register_channel_strategy("region")
class RegionStrategy(ChannelStrategy):
    """Missing target = mean of the input electrodes within ``radius_mm``.

    An empty neighbourhood gives a zero row (``observed=False``, support 0).
    """

    def __init__(self, radius_mm: float = 40.0):
        super().__init__()
        self.radius_mm = radius_mm

    def _fill(self, src, use, tgt_pos):
        d = np.linalg.norm(tgt_pos[:, None] - src.positions[use][None], axis=-1)
        near = (d <= self.radius_mm * 1e-3).astype(float)
        rows = np.zeros((len(tgt_pos), len(src.names)))
        rows[:, use] = near / np.maximum(near.sum(1, keepdims=True), 1.0)
        return rows


@register_channel_strategy("latent")
class LatentStrategy(ChannelStrategy):
    """POYO-style learned cross-attention from any electrode set.

    Keys: MLP of Fourier features of each input position + MLP of its signal
    statistics. Queries: learned latents (``free`` targets, ``n_latents``
    outputs) or an MLP of each target position's Fourier features (target
    electrodes the input does not carry; carried ones are copied).

    Parameters
    ----------
    n_latents : int
        Number of outputs for ``free`` targets.
    d_model : int
        Width of queries and keys.
    n_freqs : int
        Fourier frequencies per coordinate.
    """

    trainable = True

    def __init__(self, n_latents: int = 64, d_model: int = 64, n_freqs: int = 8):
        super().__init__()
        self.n_freqs = n_freqs
        self.latents = nn.Parameter(
            torch.randn(n_latents, d_model) / math.sqrt(d_model)
        )
        self.key_position = _mlp(6 * n_freqs, d_model)
        self.key_signal = _mlp(2, d_model)
        self.query_position = _mlp(6 * n_freqs, d_model)

    def _fourier(self, pos: np.ndarray) -> Tensor:
        # Positions in decimetres: head-sized coordinates in [-1, 1].
        arg = (
            np.nan_to_num(pos)[:, :, None] * 10 * np.pi * 2.0 ** np.arange(self.n_freqs)
        )
        return _f32(
            np.concatenate([np.sin(arg), np.cos(arg)], -1).reshape(len(pos), -1)
        )

    def build(self, src, target):
        use = self._usable(src)
        tgt = target.sensors()
        if tgt is None and target.interface != "free":
            return self._pass_through(src, target)
        extra = {"used": torch.as_tensor(use), "src_feat": self._fourier(src.positions)}
        if tgt is None:
            K = self.latents.shape[0]
            m = SpatialMap(
                None,
                None,
                None,
                torch.ones(K, dtype=torch.bool),
                torch.ones(K),
                {**extra, "recon": torch.ones(K, dtype=torch.bool)},
            )
            return m
        copy, dist = self._match(src, tgt)
        hit = copy >= 0
        W = np.zeros((len(hit), len(src.names)))
        W[np.flatnonzero(hit), copy[hit]] = 1.0
        recon = ~hit & ~tgt.non_electrode & np.isfinite(tgt.positions).all(1)
        d = np.linalg.norm(
            tgt.positions[:, None] - src.positions[use][None], axis=-1
        ).min(1)
        scale = SUPPORT_SCALE_MM * 1e-3
        support = np.where(
            hit, np.exp(-dist / scale), np.where(recon, np.exp(-d / scale), 0)
        )
        extra.update(
            recon=torch.as_tensor(recon), tgt_feat=self._fourier(tgt.positions)
        )
        return SpatialMap(
            _f32(W),
            None if tgt.channel_ids is None else torch.as_tensor(tgt.channel_ids),
            _f32(tgt.positions),
            torch.as_tensor(hit),
            _f32(support),
            extra,
        )

    def apply(self, x: Tensor, m: SpatialMap) -> Tensor:
        e = m.extra
        used = e["used"]
        keys = self.key_position(e["src_feat"]) + self.key_signal(
            _channel_stats(x, used)
        )
        q = self.query_position(e["tgt_feat"]) if "tgt_feat" in e else self.latents
        logits = q @ keys.transpose(1, 2) / math.sqrt(keys.shape[-1])
        attn = logits.masked_fill(~used, float("-inf")).softmax(-1)  # (B, K, C)
        learned = (attn @ x) * e["recon"].to(x.dtype)[:, None]
        return learned if m.weights is None else m.weights @ x + learned
