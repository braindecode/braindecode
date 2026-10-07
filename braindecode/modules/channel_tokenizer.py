# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Unified channel adaptation for pretrained EEG models.

A pretrained checkpoint identifies its input channels in exactly one of a few
ways, and every model that loads Hub weights has so far re-implemented that
logic on its own -- which is where the geometry bugs of #1227 come from
(EEG-DINO mapping input channel *i* to embedding slot *i* regardless of the
montage, SignalJEPA returning ``NaN`` for names without coordinates, BENDR
refusing a plain permutation). :class:`ChannelTokenizer` collects those five
ways into one module with a single, explicit policy for the awkward cases:

``names``
    The checkpoint knows a fixed vocabulary of channel *names*. The input is
    matched to that vocabulary through a shared alias table (``T5`` == ``P7``
    ...). ``on_unknown`` chooses what happens to a name that is not in the
    vocabulary: ``"error"`` raises, ``"zero"`` maps it to a zero embedding.
``coords``
    The checkpoint places channels by their 3D position. ``on_missing_loc``
    chooses what happens when the montage carries no usable coordinates:
    ``"error"`` raises (ZUNA / BaRISTA), ``"zero"`` projects onto a zero row
    (LUNA / REVE / DIVER-1, which accept coordinate-less montages today).
``fixed_order``
    Today's :class:`~braindecode.modules.ChannelInterpolationLayer`: the input
    is projected onto a canonical electrode set by name match (a permutation
    when every name is present) with an MNE spline for the rest, falling back
    to a pure name match + zero rows when the source has no coordinates.
``index_slots``
    The checkpoint has a fixed number of positional slots (EEG-DINO). Input
    channel ``i`` fills slot ``i``; more channels than slots is a declared
    error rather than a silent truncation.
``agnostic``
    The checkpoint does not identify channels at all; the tokenizer is a
    pass-through.

For the ``names`` / ``index_slots`` strategies the resolved mapping is exposed
as :attr:`channel_indices` (a ``LongTensor``) and :meth:`forward` returns the
input unchanged -- the consuming model feeds ``channel_indices`` to its
embedding table. For the ``coords`` / ``fixed_order`` strategies the resolution
is a projection matrix and :meth:`forward` applies it to ``(B, C, T)`` input.
``agnostic`` is always a pass-through.
"""

from __future__ import annotations

from typing import Literal, Optional, get_args

import torch
from torch import nn

from braindecode.modules.interpolation import ChannelInterpolationLayer

Strategy = Literal["names", "coords", "fixed_order", "index_slots", "agnostic"]

# Shared alias table: legacy / equivalent electrode names mapped to the
# 10-20 spelling used by the model vocabularies. Extend here, once, rather
# than per model.
CHANNEL_NAME_ALIASES: dict[str, str] = {
    "t3": "t7",
    "t4": "t8",
    "t5": "p7",
    "t6": "p8",
    "a1": "m1",
    "a2": "m2",
}


def _canon_name(name: str) -> str:
    """Lower-case a channel name and resolve it through the alias table."""
    key = name.lower()
    return CHANNEL_NAME_ALIASES.get(key, key)


class ChannelTokenizer(nn.Module):
    """Resolve user channels to a pretrained model's channel representation.

    .. warning:: Experimental. Public API may change without a deprecation cycle.

    Parameters
    ----------
    strategy : {"names", "coords", "fixed_order", "index_slots", "agnostic"}
        How the checkpoint identifies its channels (see module docstring).
    src_chs_info : list of dict, optional
        User channel info (``"ch_name"`` / ``"loc"`` keys). Required for every
        strategy except ``"agnostic"``.
    vocabulary : list of str, optional
        Canonical channel names of the checkpoint. Required for ``"names"``.
    target_chs_info : list of dict, optional
        Canonical channel set the backbone expects. Required for ``"coords"``
        and ``"fixed_order"``.
    n_slots : int, optional
        Number of positional slots. Required for ``"index_slots"``.
    on_unknown : {"error", "zero"}
        Policy for ``"names"`` when a source name is outside ``vocabulary``.
    on_missing_loc : {"error", "zero"}
        Policy for ``"coords"`` when the montage has no usable coordinates.
    method : str
        MNE interpolation method for the coordinate-based matrices.

    Attributes
    ----------
    channel_indices : torch.LongTensor or None
        For ``"names"`` / ``"index_slots"``, the per-channel row into the
        model's embedding table (non-persistent buffer). ``None`` otherwise.
    projection : torch.Tensor or None
        For ``"coords"`` / ``"fixed_order"``, the ``(n_tgt, n_src)`` matrix
        applied in :meth:`forward` (non-persistent buffer). ``None`` otherwise.
    """

    def __init__(
        self,
        strategy: Strategy,
        src_chs_info: Optional[list[dict]] = None,
        vocabulary: Optional[list[str]] = None,
        target_chs_info: Optional[list[dict]] = None,
        n_slots: Optional[int] = None,
        on_unknown: Literal["error", "zero"] = "error",
        on_missing_loc: Literal["error", "zero"] = "error",
        method: str = "spline",
    ) -> None:
        super().__init__()
        if strategy not in get_args(Strategy):
            raise ValueError(f"Unknown channel strategy {strategy!r}.")
        self.strategy = strategy
        # Registered (non-persistent) so they follow ``.to(device)`` yet stay
        # out of ``state_dict`` -- Hub checkpoints still load strict.
        self.register_buffer("channel_indices", None, persistent=False)
        self.register_buffer("projection", None, persistent=False)

        if strategy == "agnostic":
            return
        if strategy == "names":
            if src_chs_info is None or vocabulary is None:
                raise ValueError(
                    "strategy='names' requires src_chs_info and vocabulary."
                )
            self.channel_indices = _name_indices(
                [ch["ch_name"] for ch in src_chs_info], vocabulary, on_unknown
            )
        elif strategy == "index_slots":
            if src_chs_info is None or n_slots is None:
                raise ValueError(
                    "strategy='index_slots' requires src_chs_info and n_slots."
                )
            n = len(src_chs_info)
            if n > n_slots:
                raise ValueError(
                    f"index_slots: got {n} channels but only {n_slots} positional "
                    f"slots; the released weights use {n_slots}. Reduce the "
                    f"channel count or use a model with a names/coords channel "
                    f"strategy."
                )
            self.channel_indices = torch.arange(n, dtype=torch.long)
        else:  # coords / fixed_order
            if src_chs_info is None or target_chs_info is None:
                raise ValueError(
                    f"strategy={strategy!r} requires src_chs_info and target_chs_info."
                )
            # Lazy: a top-level import would cycle modules -> models -> modules.
            from braindecode.models.util import has_valid_locations

            if has_valid_locations(src_chs_info):
                # MNE spline, with a name-match short-circuit so a permutation
                # is an exact one-hot.
                self.projection = ChannelInterpolationLayer(
                    src_chs_info, target_chs_info, mode="name_match", method=method
                ).matrix
            elif strategy == "coords" and on_missing_loc == "error":
                raise ValueError(
                    "strategy='coords' with on_missing_loc='error': the montage "
                    "has no usable channel coordinates (all 'loc' are zero or "
                    "missing). Supply electrode positions or set "
                    "on_missing_loc='zero'."
                )
            else:
                # No coordinates: one-hot for matched names, zero rows else.
                idx = _name_indices(
                    [ch["ch_name"] for ch in target_chs_info],
                    [ch["ch_name"] for ch in src_chs_info],
                    "zero",
                )
                hit = idx >= 0
                W = torch.zeros(len(target_chs_info), len(src_chs_info))
                W[hit.nonzero().squeeze(1), idx[hit]] = 1.0
                self.projection = W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``(B, C, T)`` input for coordinate/fixed-order strategies.

        For ``names`` / ``index_slots`` / ``agnostic`` the input is returned
        unchanged -- the consuming model reads :attr:`channel_indices`.
        """
        if self.projection is not None:
            return self.projection.to(x.dtype) @ x
        return x


def _name_indices(names, vocabulary, on_unknown):
    """Index of each (alias-resolved) name in ``vocabulary``; ``-1`` if absent."""
    lookup = {_canon_name(n): i for i, n in enumerate(vocabulary)}
    out = []
    for name in names:
        j = lookup.get(_canon_name(name))
        if j is None:
            if on_unknown == "error":
                raise ValueError(
                    f"Channel {name!r} is not in the model vocabulary "
                    f"({len(vocabulary)} names) and no alias matches. Pass "
                    f"on_unknown='zero' to map it to a zero embedding, or "
                    f"supply a recognised channel name."
                )
            j = -1  # caller must treat -1 as a zero embedding
        out.append(j)
    return torch.tensor(out, dtype=torch.long)
