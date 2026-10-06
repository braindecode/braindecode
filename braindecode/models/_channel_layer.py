# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Private glue between the pretrained models and the channel layer."""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import torch

from braindecode.modules.channels import ELECTRODE_KINDS, ChannelEncoding, ChannelTarget
from braindecode.modules.channels.resolve import canon_name

#: Appended to ``__jit_ignored_attributes__``: the channel layer is not
#: scriptable and only reached from eager code.
JIT_IGNORED = "channel_tokenizer"


def names_chs_info(names: Sequence[str]) -> list[dict]:
    """Minimal ``chs_info`` for a named montage (positions from standard_1005)."""
    return [{"ch_name": n, "kind": "eeg"} for n in names]


def warn_if_not_canonical(model, canonical, also_accept=()) -> None:
    """``FutureWarning`` when ``native`` feeds a montage the checkpoint was not trained on.

    Compares the ``chs_info`` names (case and aliases ignored), or without
    ``chs_info`` the channel count, with ``canonical`` and ``also_accept``.
    """
    orders = [list(canonical), *(list(o) for o in also_accept)]
    chs = getattr(model, "_chs_info", None)
    if chs is not None:
        given = [canon_name(ch["ch_name"]) for ch in chs]
        ok = any(given == [canon_name(n) for n in o] for o in orders)
    else:
        try:
            ok = model.n_chans in {len(o) for o in orders}
        except ValueError:  # unknown montage: nothing to check
            return
    if not ok:
        warnings.warn(
            f"{type(model).__name__} with channel_strategy='native' feeds this montage "
            f"unchecked to a backbone trained on its {len(canonical)}-channel order "
            f"({', '.join(canonical[:4])}, ...). This will raise in the next release: pass "
            f"channel_strategy=... ('exact' to reorder or select, 'spline', 'field' or "
            f"'source' to reconstruct missing channels).",
            FutureWarning,
            stacklevel=3,
        )


def backbone_n_chans(model) -> int:
    """Channels the backbone receives after the channel layer."""
    tok = getattr(model, "channel_tokenizer", None)
    if tok is None or tok.strategy is None:
        return model.n_chans
    try:
        n_chans = model.n_chans
    except ValueError:
        n_chans = None
    return tok.n_outputs(n_chans=n_chans)


def init_positions_layer(
    model,
    channel_strategy: str,
    channel_strategy_kwargs: Optional[dict],
    *,
    fixed_montage: bool,
    model_kind: str = "eeg",
    require_positions: bool = True,
) -> None:
    """Channel layer of a ``positions`` model (LUNA, REVE, ZUNA, BaRISTA, DIVER-1, PopT).

    ``fixed_montage``: the read-out is tied to the construction montage, so any
    montage is mapped onto it; otherwise the layer passes the call's channels
    through with their positions. Every electrode kind is accepted.
    """
    name = type(model).__name__
    if channel_strategy == "source" and model_kind != "eeg":
        raise ValueError(
            f"{name}: channel_strategy='source' uses the scalp-EEG sphere head model and "
            f"cannot serve a {model_kind} model. Use a sensor strategy ('exact', 'nearest', "
            f"'idw', 'spline', 'field', ...) instead."
        )
    chs = model._chs_info
    target = ChannelTarget("positions", chs_info=chs) if fixed_montage and chs else None
    model._init_channel_tokenizer(
        channel_strategy, channel_strategy_kwargs, target=target, kinds=ELECTRODE_KINDS
    )
    tok = model.channel_tokenizer
    sensors = (
        tok.target.sensors() if tok is not None and tok.target is not None else None
    )
    if require_positions and sensors is not None:
        unknown = [
            n
            for n, p in zip(sensors.names, sensors.positions)
            if not np.isfinite(p).all()
        ]
        if unknown:
            raise ValueError(
                f"{name} maps every montage onto the {len(sensors.names)} channels it was built "
                f"with, but {unknown} have no position: give their 'loc' in chs_info or use "
                f"standard_1005 names."
            )


def key_padding_mask(observed: torch.Tensor) -> Optional[torch.Tensor]:
    """``True`` for unobserved channels; ``None`` when none or all are (all-masked attention is NaN)."""
    unobserved = ~observed.bool()
    if not bool(unobserved.any()) or bool(unobserved.all()):
        return None
    return unobserved


def batch_positions(enc: ChannelEncoding, batch_size: int) -> torch.Tensor:
    """The layer's ``(K, 3)`` positions broadcast to ``(batch_size, K, 3)``."""
    assert enc.positions is not None  # ``positions`` targets always carry them
    return enc.positions.unsqueeze(0).expand(batch_size, -1, -1)
