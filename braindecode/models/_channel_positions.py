# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Private glue between the ``positions``-interface models and the channel layer.

LUNA, REVE, ZUNA, BaRISTA, DIVER-1 and PopT read electrode coordinates. Under a
channel strategy other than ``"native"`` they take ``x`` and the coordinates
from :class:`~braindecode.modules.channels.ChannelEncoding` instead of their own
``chs_info`` parsing. Two targets exist:

- the instance montage (the ``chs_info`` the model was built with), when the
  read-out is tied to that channel set (a flattened head, a learned token
  pooling, fixed rotary buffers): any per-call montage is mapped onto it;
- otherwise a pass-through on the user's resolved positions (the channel set
  of each call, positions from ``loc`` or ``standard_1005``).

Intracranial channels (``seeg``, ``ecog``, ``dbs``) pass the resolve stage as
electrodes with positions; the ``"source"`` strategy, whose sphere head model
is scalp-EEG only, raises a declared ``ValueError`` for them.
"""

from __future__ import annotations

from typing import Optional

import torch

from braindecode.models.util import INTRACRANIAL_CH_TYPES, channel_types_from_chs_info
from braindecode.modules.channels import ChannelEncoding, ChannelTarget
from braindecode.modules.channels.tokenizer import ChannelTokenizer

#: Appended to a model's ``__jit_ignored_attributes__``: the (non-scriptable)
#: channel layer is only reached from eager code.
JIT_IGNORED = "channel_tokenizer"

#: Strategies that rely on the scalp-EEG sphere head model.
EEG_ONLY_STRATEGIES = frozenset({"source"})


def as_electrodes(chs_info: Optional[list[dict]]) -> Optional[list[dict]]:
    """Copy of ``chs_info`` with intracranial kinds relabelled ``"eeg"``.

    The resolve stage accepts EEG only; sEEG / ECoG / DBS contacts are
    electrodes with positions too, so the sensor strategies serve them.
    """
    if chs_info is None:
        return None
    kinds = channel_types_from_chs_info(chs_info)
    return [
        dict(ch, kind="eeg") if kind in INTRACRANIAL_CH_TYPES else ch
        for ch, kind in zip(chs_info, kinds)
    ]


def check_strategy_kind(
    model_name: str,
    strategy: str,
    chs_info: Optional[list[dict]],
    model_kind: str = "eeg",
) -> None:
    """Declared ``ValueError`` for an EEG-only strategy on non-EEG channels."""
    if strategy not in EEG_ONLY_STRATEGIES:
        return
    kinds = set(channel_types_from_chs_info(chs_info)) if chs_info else set()
    bad = sorted(kinds & INTRACRANIAL_CH_TYPES)
    if model_kind != "eeg" or bad:
        what = f"channels of kind {bad}" if bad else f"a {model_kind} model"
        raise ValueError(
            f"{model_name}: channel_strategy={strategy!r} uses the scalp-EEG "
            f"sphere head model and cannot serve {what}. Use a sensor strategy "
            f"('exact', 'nearest', 'idw', 'spline', 'field', ...) instead."
        )


def init_positions_layer(
    model,
    channel_strategy: str,
    channel_strategy_kwargs: Optional[dict],
    *,
    fixed_montage: bool,
    model_kind: str = "eeg",
    require_positions: bool = True,
) -> None:
    """Build ``model.channel_tokenizer`` for a ``positions`` model.

    Sets ``model._channel_layer`` (``False`` under ``"native"``, where
    ``channel_tokenizer`` is ``None`` so the native model is unchanged and
    stays scriptable).
    """
    model._channel_layer = channel_strategy != "native"
    if not model._channel_layer:
        if channel_strategy_kwargs:
            raise ValueError(
                f"strategy='native' takes no options; got "
                f"{sorted(channel_strategy_kwargs)}."
            )
        model.channel_tokenizer = None
        return
    name = type(model).__name__
    chs = model._chs_info
    check_strategy_kind(name, channel_strategy, chs, model_kind)
    chs = as_electrodes(chs)
    if fixed_montage and chs:
        target = ChannelTarget("positions", chs_info=chs)
    else:
        target = type(model)._channel_target or ChannelTarget("positions")
    model.channel_tokenizer = ChannelTokenizer(
        target,
        channel_strategy,
        src_chs_info=chs,
        **(channel_strategy_kwargs or {}),
    )
    sensors = target.sensors()
    if require_positions and sensors is not None:
        unknown = [
            n
            for n, p in zip(sensors.names, sensors.positions)
            if not torch.isfinite(torch.as_tensor(p)).all()
        ]
        if unknown:
            raise ValueError(
                f"{name} maps every montage onto the {len(sensors.names)} "
                f"channels it was built with, but {unknown} have no position: "
                f"give their 'loc' in chs_info or use standard_1005 names."
            )


def encode_positions(
    model,
    x: torch.Tensor,
    chs_info: Optional[list[dict]],
    *,
    model_kind: str = "eeg",
    require_positions: bool = True,
) -> ChannelEncoding:
    """Run the channel layer; positions are guaranteed finite if required."""
    tok = model.channel_tokenizer
    check_strategy_kind(type(model).__name__, tok.strategy_name, chs_info, model_kind)
    enc = tok(x, as_electrodes(chs_info))
    if require_positions and enc.positions is not None:
        bad = ~torch.isfinite(enc.positions).all(dim=1)
        if bool(bad.any()):
            sensors = tok.target.sensors()
            names = sensors.names if sensors is not None else ()
            missing = [
                names[i] for i in torch.flatnonzero(bad).tolist() if i < len(names)
            ]
            raise ValueError(
                f"{type(model).__name__} needs a position for every channel, but "
                f"{missing} have none: give their 'loc' in chs_info or use "
                f"standard_1005 names."
            )
    return enc


def key_padding_mask(observed: torch.Tensor) -> Optional[torch.Tensor]:
    """``True`` for unobserved channels, or ``None`` when nothing is masked.

    A mask hiding every channel would make attention undefined (NaN), so it
    is dropped too: the backbone then attends to the reconstructed channels.
    """
    unobserved = ~observed.bool()
    if not bool(unobserved.any()) or bool(unobserved.all()):
        return None
    return unobserved


def batch_positions(enc: ChannelEncoding, batch_size: int) -> torch.Tensor:
    """The layer's ``(K, 3)`` positions broadcast to ``(batch_size, K, 3)``."""
    if enc.positions is None:  # not reached for a ``positions`` target
        raise ValueError("The channel layer returned no positions.")
    return enc.positions.unsqueeze(0).expand(batch_size, -1, -1)
