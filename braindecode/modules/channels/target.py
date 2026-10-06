# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""What a backbone consumes: the channel contract a model declares once."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional, get_args

import numpy as np

from .resolve import match_names, resolve_montage, standard_position

Interface = Literal["montage", "ids", "positions", "slots", "free"]


class TargetSensors(NamedTuple):
    """The ``K`` electrodes a strategy must produce for a target."""

    names: tuple[str, ...]
    positions: np.ndarray  # (K, 3) metres, NaN = unknown
    channel_ids: Optional[np.ndarray]  # (K,) vocabulary ids for ``ids``
    non_electrode: np.ndarray  # (K,) bool, never interpolated


@dataclass(frozen=True, eq=False)
class ChannelTarget:
    """Channel contract of a backbone.

    Parameters
    ----------
    interface : {"montage", "ids", "positions", "slots", "free"}
        ``montage``: ``(B, K, T)`` in the fixed order of ``chs_info``.
        ``ids``: ``x`` plus per-channel ids into ``vocabulary``.
        ``positions``: ``x`` plus ``(K, 3)`` coordinates (training montage
        ``chs_info`` optional). ``slots``: the first ``n_slots`` channels of
        ``chs_info``. ``free``: any ``C``, no channel identity.
    chs_info : list of dict, optional
        Training montage (``montage``, ``slots``; optional for ``ids`` and
        ``positions``). For ``ids`` it gives the vocabulary positions.
    vocabulary : tuple of str, optional
        Channel names of the embedding table (``ids``).
    n_slots : int, optional
        Number of positional slots (``slots``).
    non_electrode : tuple of str
        Target channels that are not electrodes (e.g. BENDR's ``SCALE``):
        never interpolated, zero unless the input carries them by name.
    """

    interface: Interface
    chs_info: Optional[list[dict]] = None
    vocabulary: Optional[tuple[str, ...]] = None
    n_slots: Optional[int] = None
    non_electrode: tuple[str, ...] = ()

    def __post_init__(self):
        if self.interface not in get_args(Interface):
            raise ValueError(
                f"Unknown channel interface {self.interface!r}; "
                f"expected one of {get_args(Interface)}."
            )
        need = {"montage": "chs_info", "slots": "chs_info", "ids": "vocabulary"}
        field = need.get(self.interface)
        if field is not None and getattr(self, field) is None:
            raise ValueError(f"ChannelTarget({self.interface!r}) requires {field}.")
        if self.interface == "slots" and self.n_slots is None:
            raise ValueError("ChannelTarget('slots') requires n_slots.")

    def sensors(self) -> Optional[TargetSensors]:
        """Electrodes to produce, or ``None`` for a pass-through target."""
        if self.interface == "free" or (
            self.interface == "positions" and self.chs_info is None
        ):
            return None
        if self.interface == "ids":
            names = tuple(self.vocabulary or ())
            pos = np.stack([standard_position(n) for n in names])
            if self.chs_info is not None:
                given = resolve_montage(self.chs_info)
                j = match_names(names, given.names)
                pos[j >= 0] = given.positions[j[j >= 0]]
            ids = np.arange(len(names))
        else:
            chs = self.chs_info or []
            if self.interface == "slots":
                chs = chs[: self.n_slots]
            given = resolve_montage(chs)
            names, pos, ids = given.names, given.positions, None
        non_el = {n.lower() for n in self.non_electrode}
        mask = np.array([n.lower() in non_el for n in names], dtype=bool)
        return TargetSensors(names, pos, ids, mask)
