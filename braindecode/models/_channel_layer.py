# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Private glue between pretrained models and the channel layer."""

from __future__ import annotations

import warnings
from typing import Sequence

from braindecode.modules.channels.resolve import canon_name

#: Appended to a model's ``__jit_ignored_attributes__`` so that
#: :func:`torch.jit.script` skips the (non-scriptable) channel layer; the
#: scripted forward only reaches it through a ``@torch.jit.unused`` method.
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

    ``native``: the model's ``n_chans``. Sensor targets (``montage``,
    ``slots``): the target size. ``free`` targets: set by the strategy
    (``source`` parcels, ``latent`` latents, else the input channels), read
    from the map of the construction montage.
    """
    tok = model.channel_tokenizer
    if tok is None or tok.strategy is None:
        return model.n_chans
    sensors = tok.target.sensors()
    if sensors is not None:
        return len(sensors.names)
    src = tok._src
    if src is None:  # no montage yet: assume a pass-through strategy
        return model.n_chans
    if not getattr(tok.strategy, "fitted", True):  # e.g. unfitted wiener
        return len(src.picks)
    return int(tok._map(src).observed.shape[0])
