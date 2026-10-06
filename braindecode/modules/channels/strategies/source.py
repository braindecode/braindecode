# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""``source``: electrodes -> cortical parcels -> what the backbone consumes.

Non-trainable: minimum-norm inverse on a template sphere head, then forward
projection to the target electrodes (or the parcels themselves for ``free``
backbones). Trainable (direction C): the same, plus a gated correction from
cross-attention between learned parcel queries and electrode keys built
from each electrode's lead-field row and its signal statistics; the gate
starts at zero, so at initialisation the output is the physics solution.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

from ..head import get_sphere_head
from ._nn import _channel_stats, _mlp
from .base import ChannelStrategy, SpatialMap, _f32, register_channel_strategy


@register_channel_strategy("source")
class SourceStrategy(ChannelStrategy):
    """Physics-anchored source-space channel strategy.

    Reference: the inverse works on the input average-referenced over the
    used channels; reconstructed target rows add that mean back, so they sum
    to 1 and share the input's own reference with the copied rows (a
    constant input gives a constant output). Parcel outputs (``free``
    targets) are reference-free.

    Parameters
    ----------
    n_parcels : int
        Number of cortical parcels of the template head.
    lam : float
        Tikhonov regularisation, relative to ``trace(L L^T) / k``.
    trainable : bool
        Add the learned cross-attention correction (direction C).
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

    @property
    def head(self):
        return get_sphere_head(self.n_parcels, self.grid_mm)

    def _inverse(self, src, use, extra_pos):
        """Parcels-from-input ``S (P, C)``, lead field at ``extra_pos``, input lead field.

        The input and its lead field are average-referenced over the used
        channels (``S`` kills constants). The lead field at ``extra_pos`` is
        returned relative to the same mean, so ``Lt @ S`` is a target minus
        the mean of the used inputs.
        """
        k = int(use.sum())
        LU = self.head.leadfield(np.vstack([src.positions[use], extra_pos]))
        A = np.eye(k) - 1.0 / k
        Lo = A @ LU[:k]
        G = Lo @ Lo.T
        M = Lo.T @ np.linalg.inv(G + self.lam * np.trace(G) / k * np.eye(k))
        S = np.zeros((self.n_parcels, len(src.names)))
        S[:, use] = M @ A
        Lo_full = np.zeros((len(src.names), self.n_parcels))
        Lo_full[use] = Lo
        return S, LU[k:] - LU[:k].mean(0), Lo_full

    def _fill(self, src, use, tgt_pos):
        # Target minus the mean of the used inputs, plus that mean: each row
        # sums to 1 and the output stays in the input's own reference, like
        # the copied rows.
        S, Lt, _ = self._inverse(src, use, tgt_pos)
        return Lt @ S + use / use.sum()

    def build(self, src, target):
        if target.interface == "free":
            use = self._usable(src)
            S, _, Lo = self._inverse(src, use, np.zeros((0, 3)))
            resolution = np.clip(np.diag(S @ Lo), 0.0, 1.0)
            m = SpatialMap(
                weights=_f32(S),
                channel_ids=None,
                positions=_f32(self.head.centroids),
                observed=torch.ones(self.n_parcels, dtype=torch.bool),
                support=_f32(resolution),
            )
            R = np.eye(self.n_parcels)
        else:
            m = super().build(src, target)
            if not self.trainable or m.weights is None:
                return m
            # Rows the physics filled; copies and zero rows stay as they are.
            recon = (~m.observed & (m.weights.abs().sum(1) > 0)).numpy()
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

    def apply(self, x: Tensor, m: SpatialMap) -> Tensor:
        out = super().apply(x, m)
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
