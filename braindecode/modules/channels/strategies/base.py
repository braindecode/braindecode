# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Strategy interface and registry of the channel layer."""

from __future__ import annotations

import difflib
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn

from ..resolve import (
    ResolvedMontage,
    match_names,
    nearest_vocabulary,
    standard_position,
)
from ..target import ChannelTarget, TargetSensors

SUPPORT_SCALE_MM = 30.0
_REGISTRY: dict[str, type["ChannelStrategy"]] = {}


@dataclass
class SpatialMap:
    """What a strategy built for one (montage, target) pair.

    Attributes
    ----------
    weights : Tensor (K, C) or None
        Linear map applied to the input; ``None`` = pass-through.
    channel_ids : LongTensor (K,) or None
        Vocabulary ids of the outputs (``ids`` targets).
    positions : Tensor (K, 3) or None
        Output positions in metres (NaN = unknown).
    observed : BoolTensor (K,)
        Output copies a measured channel (``False`` = reconstructed / zero).
    support : Tensor (K,)
        Confidence in ``[0, 1]``: 1 for copies by name, ``exp(-d / 30 mm)``
        with ``d`` the distance to the nearest used input electrode
        otherwise, 0 for rows left at zero.
    extra : dict of str -> Tensor
        Strategy-specific per-montage tensors (moved with the map).
    """

    weights: Optional[Tensor]
    channel_ids: Optional[Tensor]
    positions: Optional[Tensor]
    observed: Tensor
    support: Tensor
    extra: dict = field(default_factory=dict)

    def to(self, ref: Tensor) -> "SpatialMap":
        """Copy with float tensors on ``ref``'s device/dtype, ints on its device."""

        def mv(t):
            if t is None:
                return None
            if t.is_floating_point():
                return t.to(device=ref.device, dtype=ref.dtype)
            return t.to(device=ref.device)

        return SpatialMap(
            mv(self.weights),
            mv(self.channel_ids),
            mv(self.positions),
            mv(self.observed),
            mv(self.support),
            {k: mv(v) for k, v in self.extra.items()},
        )


def register_channel_strategy(name: str):
    """Class decorator adding a :class:`ChannelStrategy` to the registry."""

    def deco(cls):
        if name in _REGISTRY or name == "native":
            raise ValueError(f"Channel strategy {name!r} is already registered.")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_channel_strategy(name: str, **kwargs) -> "ChannelStrategy":
    """Instantiate a registered strategy by name."""
    if name not in _REGISTRY:
        close = difflib.get_close_matches(name, list(_REGISTRY), n=3)
        hint = f" Did you mean {close}?" if close else ""
        raise ValueError(
            f"Unknown channel strategy {name!r}.{hint} Available: "
            f"{['native', *sorted(_REGISTRY)]}."
        )
    return _REGISTRY[name](**kwargs)


def _f32(a) -> Tensor:
    return torch.as_tensor(np.asarray(a), dtype=torch.float32)


def _unknown_name(name: str) -> bool:
    return bool(np.isnan(standard_position(name)).any())


class ChannelStrategy(nn.Module):
    """Map a resolved montage onto a :class:`ChannelTarget`.

    Subclasses implement :meth:`_fill` (rows for target electrodes the input
    does not carry). Name matching, copies, non-electrode targets, support
    and the pass-through targets are shared here.

    Attributes
    ----------
    name : str
        Registry name (set by :func:`register_channel_strategy`).
    trainable : bool
        Whether the strategy holds parameters (saved under
        ``channel_tokenizer.*``).
    min_positions : int
        Positioned input channels :meth:`_fill` needs.
    """

    name: str = ""
    trainable: bool = False
    min_positions: int = 1
    #: ``False``: missing targets are an error (``exact``).
    reconstructs: bool = True

    def build(self, src: ResolvedMontage, target: ChannelTarget) -> SpatialMap:
        """Spatial map from ``src`` to what ``target`` consumes."""
        tgt = target.sensors()
        if tgt is None:
            return self._pass_through(src, target)
        if target.interface == "ids" and not self.reconstructs:
            return self._ids_subset(src, target, tgt)
        W, hit, support = self._copies(src, tgt)
        todo = ~hit & ~tgt.non_electrode & np.isfinite(tgt.positions).all(1)
        if not self.reconstructs:
            missing = [
                n
                for n, h, ne in zip(tgt.names, hit, tgt.non_electrode)
                if not (h or ne)
            ]
            if missing:
                raise ValueError(
                    f"Strategy {self.name!r}: target channels {missing} are not in "
                    f"the input montage {list(src.names)}. Use a reconstructing "
                    f"strategy (e.g. 'spline', 'source') or supply them."
                )
        elif todo.any():
            use = self._usable(src)
            W[todo] = self._fill(src, use, tgt.positions[todo])
            filled = todo & (np.abs(W).sum(1) > 0)
            support[filled] = self._fill_support(src, use, tgt.positions[filled])
            self._warn_quality(W[todo], support[todo], len(tgt.names))
        self._warn_unused(src, W)
        return SpatialMap(
            weights=_f32(W),
            channel_ids=None
            if tgt.channel_ids is None
            else torch.as_tensor(tgt.channel_ids),
            positions=_f32(tgt.positions),
            observed=torch.as_tensor(hit),
            support=_f32(support),
        )

    def apply(self, x: Tensor, m: SpatialMap) -> Tensor:
        """Apply a built map to ``x`` of shape ``(B, C, T)``."""
        return x if m.weights is None else m.weights @ x

    def _fill(
        self, src: ResolvedMontage, use: np.ndarray, tgt_pos: np.ndarray
    ) -> np.ndarray:
        """Rows ``(M, C)`` for ``M`` positioned targets from ``src[use]``."""
        raise NotImplementedError

    # -- shared helpers ------------------------------------------------------

    def _copies(self, src: ResolvedMontage, tgt: TargetSensors):
        """Copy rows ``W (K, C)``, which targets are copies, and their support."""
        copy, dist = self._match(src, tgt)
        W = np.zeros((len(tgt.names), len(src.names)))
        hit = copy >= 0
        W[np.flatnonzero(hit), copy[hit]] = 1.0
        support = np.where(hit, np.exp(-dist / (SUPPORT_SCALE_MM * 1e-3)), 0.0)
        return W, hit, support

    @staticmethod
    def _fill_support(
        src: ResolvedMontage, use: np.ndarray, tgt_pos: np.ndarray
    ) -> np.ndarray:
        """``exp(-d / 30 mm)``, ``d`` the distance to the nearest used input."""
        d = np.linalg.norm(tgt_pos[:, None] - src.positions[use][None], axis=-1).min(
            1, initial=np.inf
        )
        return np.exp(-d / (SUPPORT_SCALE_MM * 1e-3))

    def _warn_quality(self, rows: np.ndarray, support: np.ndarray, K: int) -> None:
        """Warn (once per built montage) about amplifying rows and far targets."""
        gain = np.abs(rows).sum(1).max(initial=0.0)
        if gain > 2:
            warnings.warn(
                f"Strategy {self.name!r}: reconstructed row gain |w|_1 = "
                f"{gain:.3g} > 2; the map amplifies noise. Supply more input "
                f"channels or a smoother strategy.",
                UserWarning,
                stacklevel=4,
            )
        low = int((support < 0.5).sum())
        if low:
            warnings.warn(
                f"Strategy {self.name!r}: {low} of {K} target channels have "
                f"support < 0.5 (no used input within "
                f"{SUPPORT_SCALE_MM * np.log(2):.0f} mm); they are guesses.",
                UserWarning,
                stacklevel=4,
            )

    def _usable(self, src: ResolvedMontage) -> np.ndarray:
        use = src.positioned
        if use.sum() < self.min_positions:
            raise ValueError(
                f"Strategy {self.name!r} needs at least {self.min_positions} "
                f"channels with a position; got {int(use.sum())} of "
                f"{len(src.names)} ({list(src.names)}). Supply 'loc' or standard "
                f"10-05 names."
            )
        return use

    @staticmethod
    def _match(src: ResolvedMontage, tgt: TargetSensors):
        """Input channel copied to each target (-1 = none) and its distance (m).

        By name (exact, then alias); then an input whose name is unknown to
        ``standard_1005`` takes the target within 15 mm of its position.
        """
        copy = match_names(tgt.names, src.names)
        dist = np.zeros(len(copy))
        free = np.array(
            [
                i not in set(copy) and src.has_position[i] and _unknown_name(n)
                for i, n in enumerate(src.names)
            ],
            dtype=bool,
        )
        open_ = (copy < 0) & ~tgt.non_electrode
        if free.any() and open_.any():
            cand = np.flatnonzero(free)
            j = nearest_vocabulary(tgt.positions[open_], src.positions[cand])
            rows = np.flatnonzero(open_)[j >= 0]
            copy[rows] = cand[j[j >= 0]]
            dist[rows] = np.linalg.norm(
                tgt.positions[rows] - src.positions[copy[rows]], axis=1
            )
        return copy, dist

    def _ids_subset(self, src, target, tgt) -> SpatialMap:
        """``ids`` without reconstruction: each input channel -> its vocabulary id."""
        ids = match_names(src.names, tgt.names)
        lost = ids < 0
        if lost.any():
            j = nearest_vocabulary(src.positions[lost], tgt.positions)
            ids[np.flatnonzero(lost)] = j
        bad = [n for n, i in zip(src.names, ids) if i < 0]
        if bad:
            close = difflib.get_close_matches(bad[0], tgt.names, n=3, cutoff=0.0)
            raise ValueError(
                f"Channel {bad[0]!r} is not in the model vocabulary "
                f"({len(tgt.names)} names; closest: {close}) and has no position "
                f"within 15 mm of one. Rename it, supply its position, or use a "
                f"reconstructing strategy. Unknown channels: {bad}."
            )
        return SpatialMap(
            weights=None,
            channel_ids=torch.as_tensor(ids),
            positions=_f32(tgt.positions[ids]),
            observed=torch.ones(len(ids), dtype=torch.bool),
            support=torch.ones(len(ids)),
        )

    def _pass_through(self, src, target) -> SpatialMap:
        if target.interface == "positions" and not src.positioned.all():
            missing = [n for n, ok in zip(src.names, src.positioned) if not ok]
            raise ValueError(
                f"Channels {missing} have no position and no standard 10-05 "
                f"name; this model needs coordinates for every channel."
            )
        C = len(src.names)
        return SpatialMap(
            weights=None,
            channel_ids=None,
            positions=_f32(src.positions),
            observed=torch.ones(C, dtype=torch.bool),
            support=torch.ones(C),
        )

    def _warn_unused(self, src: ResolvedMontage, W: np.ndarray) -> None:
        unused = [n for n, col in zip(src.names, np.abs(W).sum(0)) if col == 0]
        if unused:
            warnings.warn(
                f"{len(unused)} input channel(s) contribute nothing to the output of "
                f"strategy {self.name!r}: {unused[:10]}.",
                UserWarning,
                stacklevel=3,
            )
