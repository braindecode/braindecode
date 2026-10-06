# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Private glue between pretrained models and the channel layer.

Shared by every model on the layer: :data:`JIT_IGNORED`,
:func:`backbone_n_chans`, and for the ``montage``/``slots`` models
:func:`names_chs_info` and :func:`warn_if_not_canonical`.

``positions`` models (LUNA, REVE, ZUNA, BaRISTA, DIVER-1, PopT) read electrode
coordinates. Under a channel strategy other than ``"native"`` they take ``x``
and the coordinates from :class:`~braindecode.modules.channels.ChannelEncoding`
instead of their own ``chs_info`` parsing. Two targets exist:

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

import warnings
from typing import Optional, Sequence

import torch

from braindecode.models.util import INTRACRANIAL_CH_TYPES, channel_types_from_chs_info
from braindecode.modules.channels import ChannelEncoding, ChannelTarget
from braindecode.modules.channels.resolve import canon_name
from braindecode.modules.channels.tokenizer import ChannelTokenizer

#: Appended to a model's ``__jit_ignored_attributes__`` so that
#: :func:`torch.jit.script` skips the (non-scriptable) channel layer, which is
#: only reached from eager code (``torch.jit.is_scripting()`` guards).
JIT_IGNORED = "channel_tokenizer"


def names_chs_info(names: Sequence[str]) -> list[dict]:
    """Minimal ``chs_info`` for a named montage (positions from standard_1005)."""
    return [{"ch_name": n, "kind": "eeg"} for n in names]


def warn_if_not_canonical(
    model, canonical: Sequence[str], also_accept: Sequence[Sequence[str]] = ()
) -> None:
    """``FutureWarning`` when ``native`` would feed an unchecked montage.

    The montage is non-canonical when the given ``chs_info`` names differ from
    ``canonical`` and every ``also_accept`` order (case and aliases ignored)
    or, without ``chs_info``, when ``n_chans`` matches none of their sizes. An
    unknown montage (no ``chs_info`` and no ``n_chans``) is not flagged.
    """
    orders = [list(canonical), *(list(o) for o in also_accept)]
    chs = getattr(model, "_chs_info", None)
    if chs is not None:
        given = [canon_name(ch["ch_name"]) for ch in chs]
        ok = any(given == [canon_name(n) for n in o] for o in orders)
    else:
        try:
            ok = model.n_chans in {len(o) for o in orders}
        except ValueError:
            return
    if ok:
        return
    name = type(model).__name__
    warnings.warn(
        f"{name} with channel_strategy='native' feeds this montage unchecked to a "
        f"backbone trained on its {len(canonical)}-channel order "
        f"({', '.join(canonical[:4])}, ...). This will raise in the next release: "
        f"pass channel_strategy=... ('exact' to reorder or select, 'spline', "
        f"'field' or 'source' to reconstruct missing channels).",
        FutureWarning,
        stacklevel=3,
    )


def backbone_n_chans(model) -> int:
    """Channels the backbone receives after the channel layer.

    ``native`` (or no layer): the model's ``n_chans``; otherwise
    :meth:`~braindecode.modules.channels.ChannelTokenizer.n_outputs` of the
    construction montage, with ``n_chans`` as the fallback input size.
    """
    tok = getattr(model, "channel_tokenizer", None)
    if tok is None or tok.strategy is None:
        return model.n_chans
    try:
        n_chans = model.n_chans
    except ValueError:
        n_chans = None
    return tok.n_outputs(n_chans=n_chans)


# -- positions models ---------------------------------------------------------

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
