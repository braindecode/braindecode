# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""``source``: electrodes -> cortical parcels of a template sphere head -> the target.

Non-trainable: minimum-norm inverse, then forward projection onto the target
electrodes (or the parcels themselves for ``free`` backbones). Trainable: plus
a correction from cross-attention between learned parcel queries and keys
built from each electrode's lead-field row and signal statistics, behind a
gate initialised at zero (at init the output is the physics solution).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
from scipy.linalg import solve
from torch import Tensor, nn

from .resolve import _standard_1005
from .strategies import (
    ChannelStrategy,
    SpatialMap,
    _channel_stats,
    _f32,
    _mlp,
    register_channel_strategy,
)


def _free_leadfield(sphere, src, positions: np.ndarray) -> np.ndarray:
    """Free-orientation lead field ``(C, 3 * n_grid)`` (MNE forward on the sphere)."""
    import mne

    # MNE's sphere EEG formula is singular for an electrode exactly on the z axis
    # through the centre (biosemi64 Cz): shift it by 1 um.
    positions = np.array(positions, float)
    rel = positions - np.asarray(sphere["r0"], float)
    positions[np.hypot(rel[:, 0], rel[:, 1]) < 1e-9, 0] += 1e-6
    names = [f"E{i}" for i in range(len(positions))]
    msg = (
        f"MNE could not compute a finite lead field of the template sphere head (3-shell, "
        f"r0=(0, 0, 0.04), radius 90 mm) at {len(positions)} electrode positions "
        f"{np.round(positions, 4).tolist()[:4]}..."
    )
    try:
        info = mne.create_info(names, 100.0, "eeg")
        info.set_montage(
            mne.channels.make_dig_montage(
                dict(zip(names, positions)), coord_frame="head"
            )
        )
        with mne.utils.use_log_level("ERROR"):
            fwd = mne.make_forward_solution(
                info, trans=None, src=src, bem=sphere, eeg=True, meg=False
            )
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"{msg}: {exc}") from exc
    sol = fwd["sol"]["data"]
    if not np.isfinite(sol).all():
        raise ValueError(f"{msg}: non-finite values; check the positions.")
    return sol


@lru_cache(maxsize=4)
def _sphere_head(n_parcels: int, grid_mm: float):
    """Sphere, source space, parcel centroids and per-parcel orientation patterns.

    MNE 3-shell sphere (``r0=(0, 0, 0.04)``, radius 90 mm), volume grid of
    ``grid_mm``, grouped into ``n_parcels`` by k-means (seed 0); each parcel's
    orientation is the first right singular vector of its lead field on the
    dense ``standard_1005`` montage. Built once per configuration (~1 s).
    """
    import mne
    from scipy.cluster.vq import kmeans2

    with mne.utils.use_log_level("ERROR"):
        sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.04), head_radius=0.09)
        src = mne.setup_volume_source_space(
            sphere=sphere, pos=grid_mm, mindist=5.0, exclude=20.0
        )
    rr = src[0]["rr"][src[0]["vertno"]]
    if len(rr) < n_parcels:
        raise ValueError(
            f"n_parcels={n_parcels} exceeds the {len(rr)} grid points of a {grid_mm} mm grid; "
            f"lower n_parcels or grid_mm."
        )
    _, labels = kmeans2(rr, n_parcels, seed=0, minit="++")
    if np.bincount(labels, minlength=n_parcels).min() == 0:
        raise ValueError(
            f"k-means left an empty parcel for n_parcels={n_parcels}; change it."
        )
    centroids = np.stack([rr[labels == p].mean(0) for p in range(n_parcels)])
    dense = np.unique(np.stack(list(_standard_1005().values())).round(6), axis=0)
    L = _free_leadfield(sphere, src, dense)
    L = L - L.mean(0, keepdims=True)
    orient = np.zeros((L.shape[1], n_parcels))
    for p in range(n_parcels):
        cols = (3 * np.flatnonzero(labels == p)[:, None] + np.arange(3)).ravel()
        orient[cols, p] = np.linalg.svd(L[:, cols], full_matrices=False)[2][0]
    return sphere, src, centroids, orient


@lru_cache(maxsize=32)
def _leadfield(n_parcels: int, grid_mm: float, positions: bytes) -> np.ndarray:
    """``(C, n_parcels)`` lead field, average-referenced over the positions.

    Cached on the exact float64 bytes of the positions: ``source`` asks for
    the same lead field again for its trainable part and after map eviction.
    """
    sphere, src, _, orient = _sphere_head(n_parcels, grid_mm)
    L = _free_leadfield(sphere, src, np.frombuffer(positions).reshape(-1, 3)) @ orient
    return L - L.mean(0, keepdims=True)


@register_channel_strategy("source")
class SourceStrategy(ChannelStrategy):
    """Physics-anchored source-space channel strategy.

    The inverse works on the input average-referenced over the used channels;
    reconstructed rows add that mean back, so they sum to 1 and keep the
    input's own reference (a constant input gives a constant output). Parcel
    outputs (``free`` targets) are reference-free. Scalp EEG only.

    Parameters
    ----------
    n_parcels : int
        Number of cortical parcels of the template head.
    lam : float
        Tikhonov regularisation, relative to ``trace(L L^T) / k``.
    trainable : bool
        Add the gated cross-attention correction.
    d_model : int
        Width of the attention queries and keys (trainable only).
    grid_mm : float
        Source grid spacing of the template head.
    """

    min_positions = 4

    def __init__(
        self,
        n_parcels: int = 64,
        lam: float = 0.1,
        trainable: bool = False,
        d_model: int = 64,
        grid_mm: float = 15.0,
    ):
        super().__init__()
        self.n_parcels, self.lam, self.grid_mm = n_parcels, lam, grid_mm
        self.trainable = trainable
        if trainable:
            self.queries = nn.Parameter(
                torch.randn(n_parcels, d_model) / math.sqrt(d_model)
            )
            self.key_leadfield = _mlp(n_parcels, d_model)
            self.key_signal = _mlp(2, d_model)
            self.gate = nn.Parameter(torch.zeros(()))

    def _inverse(self, src, use, extra_pos):
        """Parcels-from-input ``S (P, C)``, lead field at ``extra_pos`` minus the
        used-input mean, and the referenced input lead field ``(C, P)``."""
        k = int(use.sum())
        pos = np.ascontiguousarray(
            np.vstack([src.positions[use], extra_pos]), dtype=float
        )
        LU = _leadfield(self.n_parcels, self.grid_mm, pos.tobytes())
        A = np.eye(k) - 1.0 / k
        Lo = A @ LU[:k]
        G = Lo @ Lo.T
        S = np.zeros((self.n_parcels, len(src.names)))
        S[:, use] = (
            solve(G + self.lam * np.trace(G) / k * np.eye(k), Lo, assume_a="sym").T @ A
        )
        Lo_full = np.zeros((len(src.names), self.n_parcels))
        Lo_full[use] = Lo
        return S, LU[k:] - LU[:k].mean(0), Lo_full

    def _fill(self, src, use, tgt_pos):
        S, Lt, _ = self._inverse(src, use, tgt_pos)
        return Lt @ S + use / use.sum()  # rows sum to 1: the input's own reference

    def build(self, src, target):
        other = sorted({k for k in src.kinds if k != "eeg"})
        if other:
            raise ValueError(
                f"channel_strategy='source' uses the scalp-EEG sphere head model and cannot serve "
                f"channels of kind {other}. Use a sensor strategy ('exact', 'nearest', 'idw', "
                f"'spline', 'field', ...) instead."
            )
        if target.interface == "free":
            use = self._usable(src)
            S, _, Lo = self._inverse(src, use, np.zeros((0, 3)))
            m = SpatialMap(
                _f32(S),
                None,
                _f32(_sphere_head(self.n_parcels, self.grid_mm)[2]),
                torch.ones(self.n_parcels, dtype=torch.bool),
                _f32(np.clip(np.diag(S @ Lo), 0.0, 1.0)),  # resolution
            )
            R = np.eye(self.n_parcels)
        else:
            m = super().build(src, target)
            if not self.trainable or m.weights is None:
                return m
            recon = (
                ~m.observed & (m.weights.abs().sum(1) > 0)
            ).numpy()  # physics-filled rows
            use = self._usable(src)
            _, Lt, Lo = self._inverse(src, use, m.positions.numpy()[recon])
            R = np.zeros((len(recon), self.n_parcels))
            R[recon] = Lt
        if self.trainable:
            norm = np.linalg.norm(Lo, axis=1, keepdims=True)
            m.extra = {
                "R": _f32(R),
                "leadfield": _f32(Lo / np.where(norm > 0, norm, 1.0)),
                "used": torch.as_tensor(use),
            }
        return m

    def _free_size(self, n_input: int) -> int:
        return self.n_parcels

    def project(self, x: Tensor, m: SpatialMap) -> Tensor:
        out = super().project(x, m)
        if not self.trainable or not m.extra:
            return out
        used = m.extra["used"]
        w = used.to(x.dtype)
        xc = (x - (x * w[:, None]).sum(1, keepdim=True) / w.sum()) * w[:, None]
        keys = self.key_leadfield(m.extra["leadfield"]) + self.key_signal(
            _channel_stats(x, used)
        )
        logits = self.queries @ keys.transpose(1, 2) / math.sqrt(keys.shape[-1])
        attn = logits.masked_fill(~used, float("-inf")).softmax(-1)  # (B, P, C)
        return out + self.gate * (m.extra["R"] @ (attn @ xc))
